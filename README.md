# RustGPT

RustGPT is an educational Rust codebase for a modern small decoder-only language model on Apple Silicon.

This repo is designed to teach one clean path:

1. train or load a tokenizer
2. pretrain a dense causal LM on text
3. continue with small chat SFT
4. evaluate and chat with the checkpoint

It intentionally does not mix in scratch tensor code, MoE, sparse attention, RLHF, or other frontier features that would distract from the main implementation.

Start with [architecture.md](architecture.md). It explains the supported design and the module boundaries used throughout the project.

## What Students Should Learn

By the end of this project, students should be able to explain and trace:

- how raw text and chat data become tokenized training examples
- how a decoder-only Transformer maps token IDs to logits
- where `RMSNorm`, `RoPE`, `SwiGLU`, KV cache, and grouped-query attention live in code
- how pretraining differs from SFT in data handling and loss masking
- how checkpoints, resume, evaluation, and terminal chat fit into one consistent pipeline

## Supported Mainline

The supported architecture is:

- dense decoder-only causal Transformer
- BPE tokenizer or external `tokenizer.json`
- pre-norm `RMSNorm`
- causal self-attention with standard multi-head attention or grouped-query attention
- `RoPE` or learned absolute positions
- `SwiGLU`, `GELU`, or `ReLU` MLP
- tied or untied output projection
- `AdamW` with warmup, cosine or linear decay, and gradient clipping
- gradient accumulation and optional activation checkpointing
- checkpointed training, evaluation, sampling, and terminal chat

## How To Read The Repo

If you are new to the codebase, read it in this order:

1. [architecture.md](architecture.md)
2. [src/model/lm.rs](src/model/lm.rs)
3. [src/train/training.rs](src/train/training.rs)
4. [src/data/training_data.rs](src/data/training_data.rs)
5. [src/infer/sample.rs](src/infer/sample.rs)
6. [src/runtime/checkpoint.rs](src/runtime/checkpoint.rs)
7. [src/app/cli/](src/app/cli/)

That order follows the actual learning path: model first, then training loop, then data shaping, then inference/runtime wiring.

## Blessed Train Presets

Use `--preset` early in the command line, then override individual flags after it when needed.

- `debug-tiny`: shortest correctness and smoke-test profile
- `class-small`: default classroom pretraining profile
- `class-serious`: longer instructor-oriented M5/16 GB run with activation checkpointing
- `class-chat`: chat SFT profile for resuming a pretrained checkpoint

## Included Open Datasets

The default path in this repo uses the open datasets documented in [data/open/README.md](data/open/README.md).

- [data/open/tinystories/train-00000-of-00004.parquet](data/open/tinystories/train-00000-of-00004.parquet): TinyStories text shard for pretraining
- [data/open/tinystories/validation.parquet](data/open/tinystories/validation.parquet): TinyStories validation split
- [data/open/smoltalk-everyday/train.parquet](data/open/smoltalk-everyday/train.parquet): SmolTalk chat data for supervised fine-tuning
- [data/open/smoltalk-everyday/test.parquet](data/open/smoltalk-everyday/test.parquet): SmolTalk held-out chat split

## Before You Start

Use release builds for anything that actually trains.

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
mkdir -p checkpoints
```

All command examples below use `cargo run --release -- ...`.

## Recommended Student Workflow

### 1. Run the tiny smoke test

Do this before longer training runs. It confirms that tokenization, training, resume, and chat all work on your machine.

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
  --preset debug-tiny \
  --tokenizer /tmp/rustgpt_smoke_bpe.json \
  --mode pretrain \
  --steps 2 \
  --sample-every 1 \
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
  --preset class-chat \
  --chat-template simple \
  --valid-chat-template simple \
  --steps 2 \
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

### 2. Train a tokenizer on TinyStories

```bash
cargo run --release -- train-tokenizer \
  --data data/open/tinystories/train-00000-of-00004.parquet \
  --format parquet-text \
  --out checkpoints/tinystories_bpe.json \
  --vocab-size 4096 \
  --min-frequency 2
```

### 3. Pretrain a base model on TinyStories

This creates a base checkpoint that you can later resume for chat SFT.

```bash
cargo run --release -- train \
  --data data/open/tinystories/train-00000-of-00004.parquet \
  --format parquet-text \
  --valid-data data/open/tinystories/validation.parquet \
  --valid-format parquet-text \
  --preset class-small \
  --tokenizer checkpoints/tinystories_bpe.json \
  --mode pretrain \
  --device auto \
  --checkpoint-out checkpoints/tinystories_base.ckpt
```

### 4. Resume that checkpoint for chat SFT

This switches the data format to `parquet-chat` and uses assistant-only loss masking internally.

```bash
cargo run --release -- train \
  --data data/open/smoltalk-everyday/train.parquet \
  --format parquet-chat \
  --valid-data data/open/smoltalk-everyday/test.parquet \
  --valid-format parquet-chat \
  --preset class-chat \
  --chat-template simple \
  --valid-chat-template simple \
  --device auto \
  --resume checkpoints/tinystories_base.best.ckpt \
  --checkpoint-out checkpoints/rustgpt_chat.ckpt
```

### 5. Evaluate and chat

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

Prompt-regression suites:

```bash
cargo run --release -- eval \
  --checkpoint checkpoints/rustgpt_chat.best.ckpt \
  --prompt-file evals/instruction_following_simple.jsonl \
  --prompt-file evals/retention_simple.jsonl \
  --prompt-file evals/repetition_simple.jsonl \
  --prompt-file evals/assistant_termination_simple.jsonl \
  --temperature 0.0 \
  --top-k 1 \
  --max-new-tokens 24 \
  --device auto
```

Interactive terminal chat:

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

## What To Expect During Training

- `train` prints loss, validation loss, learning rate, token throughput, and checkpoint progress
- if validation is enabled and you pass `--checkpoint-out foo.ckpt`, RustGPT also writes `foo.best.ckpt`
- checkpoints include the model config, tokenizer asset, train config, data paths, and recent run metrics
- `chat` reloads the saved tokenizer, model config, and chat template from the checkpoint

## Useful Features To Explore After The Baseline

- `prepare-data` for exact dedup plus simple length and chat-quality filtering
- `--position rope` for rotary position embeddings
- `--n-kv-head` for grouped-query attention when it divides `--n-head`
- `--activation-checkpointing` for memory-constrained training
- `--grad-accum-steps` for stable effective batch size on small hardware

## Notes

- Use `--device auto` on Apple Silicon unless you specifically want CPU-only execution.
- Keep checkpoint files together. A checkpoint such as `model.ckpt` also writes sidecar files next to it.
- When resuming from a checkpoint, RustGPT reuses the saved tokenizer and model config from that checkpoint.
- Start with the included datasets and presets before changing model size or data format.
