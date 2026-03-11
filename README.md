# rustgpt

RustGPT is a local decoder-only LLM stack in pure Rust.

It is built for students who want to:

- train a tokenizer
- pretrain a small base model
- fine-tune it for chat with assistant-only SFT
- evaluate it with checked prompt suites
- use it from a terminal chat interface

This repo is not a wrapper around `llama.cpp`, `mlx`, or PyTorch. The model, training loop, sampling, checkpointing, and chat runtime all live in this codebase.

## Recommended Local Path

For a MacBook Pro M5 with 16 GB RAM, the most practical student workflow in this repo is:

1. train a BPE tokenizer on `TinyStories`
2. pretrain a compact base model on `TinyStories`
3. resume training with `SmolTalk` chat data in `--mode sft`
4. run `eval` with checked prompt suites
5. launch the `chat` REPL against the SFT checkpoint

The repo already includes downloaded open datasets for that exact path in `data/open/`.

## Open Datasets Included

See [data/open/README.md](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/data/open/README.md) for the source URLs and file notes.

Included now:

- [data/open/tinystories/train-00000-of-00004.parquet](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/data/open/tinystories/train-00000-of-00004.parquet)
  - one official TinyStories training shard
  - `237 MB`
  - use for pretraining
- [data/open/tinystories/validation.parquet](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/data/open/tinystories/validation.parquet)
  - official TinyStories validation split
  - `9.5 MB`
  - use for validation and smoke tests
- [data/open/smoltalk-everyday/train.parquet](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/data/open/smoltalk-everyday/train.parquet)
  - `HuggingFaceTB/smoltalk` everyday-conversations subset
  - `2260` chat records
  - use for SFT
- [data/open/smoltalk-everyday/test.parquet](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/data/open/smoltalk-everyday/test.parquet)
  - held-out chat split
  - `119` chat records
  - use for validation during SFT

Why this exact setup:

- TinyStories is the first recommended pretraining dataset in [improvement-plan.md](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/improvement-plan.md#L190)
- SmolTalk is the current public dataset path that matches the small chat-SFT dataset direction in the same plan
- one TinyStories shard is large enough to be real, but still reasonable for local iteration
- the everyday-conversations SmolTalk subset is small enough that students can run end-to-end SFT locally

## Quick Start

### 1. Run tests

```bash
cargo test
```

### 2. Train a tokenizer

This is the first real step in the full workflow.

```bash
mkdir -p checkpoints

cargo run -- train-tokenizer \
  --data data/open/tinystories/train-00000-of-00004.parquet \
  --format parquet-text \
  --out checkpoints/tinystories_bpe.json \
  --vocab-size 4096 \
  --min-frequency 2
```

### 3. Pretrain a base model

This is the recommended first local base-model recipe.

```bash
cargo run -- train \
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
  --n-kv-head 2 \
  --activation swiglu \
  --position rope \
  --tied-embeddings \
  --warmup-steps 100 \
  --lr-schedule cosine \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --device auto \
  --checkpoint-out checkpoints/tinystories_base.ckpt \
  --best-checkpoint-out checkpoints/tinystories_base.best.ckpt
```

Notes:

- `--device auto` is the default recommendation on Apple Silicon.
- use `--device cpu` in a classroom if you want every student on the same path.
- keep `--block-size` the same for pretraining and SFT when you plan to resume from a checkpoint.

### 4. Fine-tune the base model for chat

Use the pretrained checkpoint and switch to `--mode sft` on the SmolTalk chat data.

The simplest end-to-end path is to keep the default `simple` chat template:

- `System: ...`
- `User: ...`
- `Assistant: ...`

```bash
cargo run -- train \
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
  --checkpoint-out checkpoints/rustgpt_chat.ckpt \
  --best-checkpoint-out checkpoints/rustgpt_chat.best.ckpt
```

Important:

- when you resume from a checkpoint, RustGPT reuses that checkpoint's tokenizer and model config
- for the first local workflow, keep the template simple and consistent across SFT, eval, and chat
- SFT data uses assistant-only loss masking internally

### 5. Evaluate the chat model

Held-out loss on the chat split:

```bash
cargo run -- eval \
  --checkpoint checkpoints/rustgpt_chat.best.ckpt \
  --data data/open/smoltalk-everyday/test.parquet \
  --format parquet-chat \
  --chat-template simple \
  --max-examples 64 \
  --device auto
```

Checked prompt-suite evaluation for assistant turn termination and over-generation:

```bash
cargo run -- eval \
  --checkpoint checkpoints/rustgpt_chat.best.ckpt \
  --prompt-file evals/assistant_termination_simple.jsonl \
  --temperature 1.0 \
  --top-k 1 \
  --device auto
```

The `eval` command will now:

- load checked prompt cases from `--prompt-file`
- print each prompt and output
- mark each case as `pass` or `fail`
- return a non-zero exit code if a checked case fails

### 6. Launch the chat interface

```bash
cargo run -- chat \
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

The REPL supports:

- multi-line UTF-8 input
- token-budgeted history dropping
- token-aware stop conditions
- checkpointed chat-template metadata

## Smoke-Test Workflow

If you want a verified fast path before the larger student run, these exact commands were exercised locally in this repo.

### 1. Tiny tokenizer on TinyStories validation

```bash
cargo run -- train-tokenizer \
  --data data/open/tinystories/validation.parquet \
  --format parquet-text \
  --out /tmp/rustgpt_open_bpe.json \
  --vocab-size 512 \
  --min-frequency 2
```

### 2. Tiny pretrain checkpoint

```bash
cargo run -- train \
  --data data/open/tinystories/validation.parquet \
  --format parquet-text \
  --valid-data data/open/tinystories/validation.parquet \
  --valid-format parquet-text \
  --tokenizer /tmp/rustgpt_open_bpe.json \
  --steps 2 \
  --batch-size 2 \
  --block-size 32 \
  --n-layer 1 \
  --n-embd 32 \
  --n-head 4 \
  --n-kv-head 2 \
  --activation swiglu \
  --position rope \
  --tied-embeddings \
  --device cpu \
  --checkpoint-out /tmp/rustgpt_open_pretrain.ckpt
```

### 3. Tiny chat SFT resume

```bash
cargo run -- train \
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
  --resume /tmp/rustgpt_open_pretrain.ckpt \
  --checkpoint-out /tmp/rustgpt_open_chat.ckpt
```

### 4. Tiny checked eval

```bash
cargo run -- eval \
  --checkpoint /tmp/rustgpt_open_chat.ckpt \
  --prompt-file evals/assistant_termination_simple.jsonl \
  --temperature 1.0 \
  --top-k 1 \
  --device cpu
```

### 5. Tiny chat REPL

```bash
cargo run -- chat \
  --checkpoint /tmp/rustgpt_open_chat.ckpt \
  --device cpu \
  --system "be brief" \
  --max-new-tokens 8 \
  --stream \
  --seed 7
```

## Data Formats

RustGPT trains directly from:

- `text`
- `lines`
- `jsonl-text`
- `jsonl-chat`
- `parquet-text`
- `parquet-chat`

Structured chat records use:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Supported roles are:

- `system`
- `user`
- `assistant`
- `tool`

If students want to inspect or normalize datasets before training, use `prepare-data`.

Example:

```bash
cargo run -- prepare-data \
  --data data/open/smoltalk-everyday/train.parquet \
  --format parquet-chat \
  --out /tmp/smoltalk_everyday_train.jsonl \
  --out-format jsonl-chat
```

## Recommended Model Sizes

Good starting points on a 16 GB Apple laptop:

- fast smoke runs: `1 layer`, `32 embd`, `block size 32`
- first real pretrain/SFT run: `4 layers`, `128 embd`, `block size 128`
- more ambitious local experiments: increase `steps` before increasing model width

Keep expectations realistic:

- this repo can demonstrate real local pretraining and chat SFT
- it will not match current production small-model quality from scratch on one laptop
- progress should be measured with held-out loss and checked prompt suites, not just by reading one lucky sample

## Useful Commands

- `cargo test`
- `cargo run -- inspect-vocab --data <PATH> --format <MODE>`
- `cargo run -- gpu-info`
- `cargo run -- prepare-data ...`
- `cargo run -- train-tokenizer ...`
- `cargo run -- train ...`
- `cargo run -- eval ...`
- `cargo run -- chat ...`

## Repo Layout

- `src/data/` for corpora, schema, tokenizer, checkpointing, and data prep
- `src/model/` for transformer parameters and architecture helpers
- `src/runtime/` for forward, backward, training, sampling, eval, and backend code
- `evals/` for checked prompt suites
- `data/open/` for the downloaded open datasets used in the recommended workflow
- [improvement-plan.md](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/improvement-plan.md) for the longer roadmap
