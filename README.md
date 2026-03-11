# rustgpt

RustGPT is a local Rust LLM project focused on one simple workflow:

1. train a tokenizer
2. pretrain a small base model on open text data
3. continue training with chat SFT on open chat data
4. chat with the resulting checkpoint in the terminal

The default path in this repo uses the open datasets already included under [data/open/README.md](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/data/open/README.md).

## Included Open Datasets

- [data/open/tinystories/train-00000-of-00004.parquet](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/data/open/tinystories/train-00000-of-00004.parquet)
  TinyStories text shard for pretraining
- [data/open/tinystories/validation.parquet](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/data/open/tinystories/validation.parquet)
  TinyStories validation split
- [data/open/smoltalk-everyday/train.parquet](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/data/open/smoltalk-everyday/train.parquet)
  SmolTalk chat data for supervised fine-tuning
- [data/open/smoltalk-everyday/test.parquet](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/data/open/smoltalk-everyday/test.parquet)
  SmolTalk held-out chat split

## Before You Start

Use release builds for anything that actually trains:

```bash
cargo test
mkdir -p checkpoints
```

All commands below use `cargo run --release -- ...`.

## Fastest Path To Your Own Chat Model

### 1. Train a tokenizer on TinyStories

```bash
cargo run --release -- train-tokenizer \
  --data data/open/tinystories/train-00000-of-00004.parquet \
  --format parquet-text \
  --out checkpoints/tinystories_bpe.json \
  --vocab-size 4096 \
  --min-frequency 2
```

### 2. Pretrain a base model on TinyStories

This creates a base checkpoint you can later resume for chat SFT.

```bash
cargo run --release -- train \
  --data data/open/tinystories/train-00000-of-00004.parquet \
  --format parquet-text \
  --valid-data data/open/tinystories/validation.parquet \
  --valid-format parquet-text \
  --tokenizer checkpoints/tinystories_bpe.json \
  --mode pretrain \
  --steps 2000 \
  --batch-size 8 \
  --sample-every 100 \
  --block-size 128 \
  --n-layer 4 \
  --n-embd 128 \
  --n-head 4 \
  --activation swiglu \
  --tied-embeddings \
  --warmup-steps 100 \
  --lr-schedule cosine \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --device auto \
  --checkpoint-out checkpoints/tinystories_base.ckpt
```

### 3. Resume from that checkpoint for chat SFT

This switches the training data to `parquet-chat` and uses assistant-only loss internally.

```bash
cargo run --release -- train \
  --data data/open/smoltalk-everyday/train.parquet \
  --format parquet-chat \
  --valid-data data/open/smoltalk-everyday/test.parquet \
  --valid-format parquet-chat \
  --chat-template simple \
  --valid-chat-template simple \
  --mode sft \
  --steps 600 \
  --batch-size 4 \
  --sample-every 50 \
  --device auto \
  --resume checkpoints/tinystories_base.best.ckpt \
  --checkpoint-out checkpoints/rustgpt_chat.ckpt
```

### 4. Chat with the SFT checkpoint

```bash
cargo run --release -- chat \
  --checkpoint checkpoints/rustgpt_chat.best.ckpt \
  --device auto \
  --system "Be concise and helpful." \
  --max-new-tokens 64 \
  --stream \
  --seed 7
```

REPL commands:

- `/exit`
- `/reset`
- `/history`

Submit a message with an empty line. Multi-line input is supported.

## What To Expect

- `train` prints loss, validation loss, learning rate, and checkpoint progress.
- If validation is enabled and you pass `--checkpoint-out foo.ckpt`, RustGPT also writes `foo.best.ckpt`.
- `chat` loads the checkpoint metadata and uses the same chat template that was saved during training.

## Minimal Evaluation

Held-out chat loss:

```bash
cargo run --release -- eval \
  --checkpoint checkpoints/rustgpt_chat.best.ckpt \
  --data data/open/smoltalk-everyday/test.parquet \
  --format parquet-chat \
  --chat-template simple \
  --max-examples 64 \
  --device auto
```

Sample from a checkpoint without entering the REPL:

```bash
cargo run --release -- sample \
  --checkpoint checkpoints/rustgpt_chat.best.ckpt \
  --prompt "User: What is Rust?\nAssistant:" \
  --top-k 20 \
  --temperature 0.8 \
  --max-new-tokens 64 \
  --device auto
```

## Notes For New Users

- Use `--device auto` on Apple Silicon unless you specifically want a CPU-only run.
- When resuming from a checkpoint, RustGPT reuses the saved tokenizer and model config from that checkpoint.
- Keep checkpoint files together. A checkpoint such as `model.ckpt` also writes sidecar files next to it.
- The current Burn runtime supports `--position learned` and requires `--n-kv-head` to match `--n-head`.
- Start with the included datasets before changing model size or data format.

## Small Smoke Test

If you want a very short end-to-end run before the full recipe, use the TinyStories validation split and only a couple of steps.

Train a tiny tokenizer:

```bash
cargo run --release -- train-tokenizer \
  --data data/open/tinystories/validation.parquet \
  --format parquet-text \
  --out /tmp/rustgpt_smoke_bpe.json \
  --vocab-size 512 \
  --min-frequency 2
```

Train a tiny base checkpoint:

```bash
cargo run --release -- train \
  --data data/open/tinystories/validation.parquet \
  --format parquet-text \
  --valid-data data/open/tinystories/validation.parquet \
  --valid-format parquet-text \
  --tokenizer /tmp/rustgpt_smoke_bpe.json \
  --mode pretrain \
  --steps 2 \
  --batch-size 2 \
  --sample-every 1 \
  --block-size 32 \
  --n-layer 1 \
  --n-embd 32 \
  --n-head 4 \
  --device cpu \
  --checkpoint-out /tmp/rustgpt_smoke_base.ckpt
```

Run a tiny chat SFT resume:

```bash
cargo run --release -- train \
  --data data/open/smoltalk-everyday/train.parquet \
  --format parquet-chat \
  --valid-data data/open/smoltalk-everyday/test.parquet \
  --valid-format parquet-chat \
  --chat-template simple \
  --valid-chat-template simple \
  --mode sft \
  --steps 2 \
  --batch-size 1 \
  --sample-every 1 \
  --device cpu \
  --resume /tmp/rustgpt_smoke_base.best.ckpt \
  --checkpoint-out /tmp/rustgpt_smoke_chat.ckpt
```

Chat with it:

```bash
cargo run --release -- chat \
  --checkpoint /tmp/rustgpt_smoke_chat.best.ckpt \
  --device cpu \
  --max-new-tokens 32 \
  --stream
```
