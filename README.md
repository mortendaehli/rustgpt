# rustgpt

Compact GPT-from-scratch implementation in pure Rust.

This repository contains:

- a small decoder-only byte-level GPT in `src/`
- manual forward and backward passes
- Adam training
- checkpoint save/load
- standalone sampling
- a simple terminal chat REPL
- small offline datasets in `data/`

The current code is intentionally compact and CPU-friendly. It is suitable for learning, testing, and iterating on architecture ideas before adding larger-model features.

The educational baseline is the CPU reference path. The GPU backend is included as an advanced systems layer, not as the first thing students should read.

## Status

Working today:

- `cargo test`
- `cargo run -- inspect-vocab ...`
- `cargo run -- train ...`
- `cargo run -- bench-train ...`
- `cargo run -- sample ...`
- `cargo run -- bench-sample ...`
- `cargo run -- chat ...`
- `cargo run -- gpu-info`

Current limitations:

- byte-level tokenizer only
- grouped GPU batches currently require equal effective sequence length; mixed-length batches fall back to the reference per-sequence accumulation path
- checkpoint version 2 uses the new byte tokenizer; legacy version 1 char-token checkpoints are not loaded
- the chat REPL only makes sense with a checkpoint trained on chat-style text
- the GPU path is intentionally educational rather than fully optimized
- GPU training now uses full-sequence kernels for the main projection path, grouped equal-length GPU batches, and sequence-level GPU backward for attention/query accumulation
- Metal through `wgpu` is measurably faster than CPU on the M5 for realistic training configs, but tiny configs still favor CPU and attention backward remains the dominant GPU cost

## Read This First

If you are teaching or learning from this repo, the recommended code-reading order is:

1. `src/data/tokenizer.rs`
2. `src/model/mod.rs`
3. `src/runtime/forward.rs`
4. `src/runtime/backward.rs`
5. `src/runtime/training.rs`
6. `src/runtime/backend/` only after the CPU path makes sense

## What This Project Teaches

- how a tokenizer turns text into token IDs
- how embeddings, attention, residual connections, and an MLP fit together in a decoder-only GPT
- how to write explicit forward and backward passes in Rust without an autograd framework
- how training data differs between line-based corpora and plain-text token streams
- how checkpointing, sampling, and a simple chat REPL sit on top of the core model
- how a portable GPU backend can accelerate the same model while staying structurally separate from the CPU reference path

## Repo Layout

```text
.
├── Cargo.toml
├── Cargo.lock
├── src/
├── data/
└── architecture.md
```

## Included Test Datasets

The `data/` folder is intentionally small so the project is runnable offline and on modest hardware.

- `data/names_demo.txt`
  - line-delimited names
  - best for the original name-generation workflow
- `data/chat_tiny.txt`
  - tiny transcript-style corpus using `System:`, `User:`, `Assistant:`
  - best for testing the `chat` command end-to-end
- `data/stories_tiny.txt`
  - tiny plain-text story corpus
  - useful for testing general text modeling with punctuation and whitespace

For a larger next-step dataset that still fits a MacBook Pro M5 with 16 GB RAM comfortably, use `tinyshakespeare.txt` as a single plain-text file in `data/`. Plain-text corpora are now trained as sliding token windows across the whole byte stream instead of repeating the same prefix every step.

## Quick Start

### 1. Run the test suite

```bash
cargo test
```

The repo keeps `checkpoints/` empty by default. All `.ckpt` files are local outputs that you generate yourself while following the examples below.

### 2. Inspect a tokenizer

```bash
cargo run -- inspect-vocab --data data/names_demo.txt
```

### 3. Inspect GPU availability

```bash
cargo run -- gpu-info
```

The default device is `cpu` for classroom reproducibility. Use `--device cpu` in the main teaching flow so every student sees the same execution path. On macOS, `train`, `sample`, and `chat` use Metal through `wgpu` when you select `--device auto` or `--device gpu`. On systems without a compatible GPU adapter, `--device auto` falls back to CPU.

### 3b. Profile a short training run

```bash
cargo run -- train \
  --data data/names_demo.txt \
  --steps 20 \
  --device cpu \
  --profile
```

This prints cumulative stage timings such as `sync.weights`, `forward.matvec`, `forward.attention`, `backward.matvec_t`, `backward.grad_accum`, `backward.row_grad`, and `optimizer.adam`. On a real GPU path you will also see device-resident stages such as `forward.add`, `forward.attention_scores`, `forward.attention_values`, `forward.readback`, `backward.attention`, and `sample.readback_logits`.

### 4. Train a name model

```bash
mkdir -p checkpoints

cargo run -- train \
  --data data/names_demo.txt \
  --steps 300 \
  --sample-every 50 \
  --device cpu \
  --checkpoint-out checkpoints/names_demo.ckpt
```

### 5. Sample from a saved checkpoint

```bash
cargo run -- sample \
  --checkpoint checkpoints/names_demo.ckpt \
  --device cpu \
  --samples 5 \
  --max-new-tokens 12 \
  --seed 7
```

### 6. Resume training

```bash
cargo run -- train \
  --data data/names_demo.txt \
  --steps 200 \
  --device cpu \
  --resume checkpoints/names_demo.ckpt \
  --checkpoint-out checkpoints/names_demo_v2.ckpt
```

## Chat Demo

Train on the included transcript-style corpus:

```bash
mkdir -p checkpoints

cargo run -- train \
  --data data/chat_tiny.txt \
  --format text \
  --steps 500 \
  --block-size 128 \
  --n-embd 32 \
  --n-head 4 \
  --n-layer 1 \
  --device cpu \
  --sample-every 100 \
  --checkpoint-out checkpoints/chat_tiny.ckpt
```

Launch the REPL:

```bash
cargo run -- chat \
  --checkpoint checkpoints/chat_tiny.ckpt \
  --device cpu \
  --system "be brief" \
  --max-new-tokens 32 \
  --seed 7
```

REPL commands:

- `/history`
- `/reset`
- `/exit`

Important:

- The REPL uses the checkpoint tokenizer.
- A checkpoint trained only on names will not be useful for chat.
- A chat checkpoint needs the transcript bytes and role prefixes in its training corpus.
- The included chat corpus is only a plumbing demo. Expect rough outputs unless you train longer and on a better transcript dataset.
- The REPL accepts multi-line UTF-8 input. Submit a message with an empty line.

## Plain-Text Training Demo

Train on a tiny story corpus:

```bash
cargo run -- train \
  --data data/stories_tiny.txt \
  --format text \
  --steps 800 \
  --block-size 64 \
  --n-embd 32 \
  --n-head 4 \
  --n-layer 1 \
  --device cpu \
  --checkpoint-out checkpoints/stories_tiny.ckpt
```

Sample with a prompt:

```bash
cargo run -- sample \
  --checkpoint checkpoints/stories_tiny.ckpt \
  --device cpu \
  --prompt "The lighthouse" \
  --samples 3 \
  --max-new-tokens 48 \
  --temperature 0.7
```

## Benchmarking

Use the built-in benchmark commands when comparing CPU vs GPU changes or before starting a new runtime refactor.

For hardware comparisons, use release mode:

```bash
cargo run --release -- bench-compare-train \
  --data data/tinyshakespeare.txt \
  --format text \
  --steps 1 \
  --block-size 128 \
  --n-embd 256 \
  --n-head 8 \
  --n-layer 2 \
  --iters 3 \
  --warmup 1
```

`bench-compare-train` forces one CPU run and one GPU run with the same model and dataset settings, then prints the average wall-clock time and `speedup_vs_cpu`.

You should re-run the benchmark after any backend or tokenizer change. Earlier benchmark snapshots from this repo are no longer authoritative because the training path now uses sliding windows for plain-text corpora and a byte-level tokenizer.

`--batch-size` now has two behaviors:

- on CPU, it is still gradient accumulation
- on GPU, it uses one grouped equal-length batch path when every sequence in the batch has the same effective training length; otherwise it falls back to the reference per-sequence accumulation path

For teaching, keep the main exercises on CPU first and only switch to `--device auto` or `--device gpu` after students understand the reference path.

For the current codebase, `--format text` is the easiest way to exercise the grouped GPU path because every optimizer step sees the same truncated sequence length.

Benchmark repeated training runs:

```bash
cargo run -- bench-train \
  --data data/tinyshakespeare.txt \
  --format text \
  --steps 50 \
  --block-size 64 \
  --n-embd 32 \
  --n-head 4 \
  --n-layer 1 \
  --device auto \
  --iters 5 \
  --warmup 1
```

Benchmark repeated sampling runs:

```bash
cargo run -- bench-sample \
  --checkpoint checkpoints/shakespeare.ckpt \
  --device auto \
  --prompt "ROMEO:" \
  --max-new-tokens 64 \
  --iters 10 \
  --warmup 2
```

Both benchmark commands print an aggregate wall-clock summary plus cumulative runtime stage timings.

Important:

- Benchmark training in `--release`, not debug mode.
- `--device auto` is useful for normal workflows, but `bench-compare-train` is the honest way to compare CPU and GPU on the same machine.
- The current GPU training backend is educational and improving quickly, but it still pays a measurable cost in `backward.attention`, and grouped-batch scaling is not linear yet.

## Recommended MacBook Dataset

For a more meaningful text run on a MacBook Pro M5 with 16 GB RAM, use `tinyshakespeare.txt`.

Suggested command:

```bash
cargo run -- train \
  --data data/tinyshakespeare.txt \
  --format text \
  --steps 2000 \
  --block-size 64 \
  --n-embd 32 \
  --n-head 4 \
  --n-layer 1 \
  --device cpu \
  --sample-every 200 \
  --checkpoint-out checkpoints/shakespeare.ckpt
```

Why this is a good fit:

- small enough for fast CPU iteration
- large enough to show real language structure
- better signal than a tiny demo corpus
- still aligned with the current byte-level implementation

I would avoid jumping to large modern corpora until batching, longer-context handling, and better tokenization are implemented.

## Command Summary

### Train

```bash
cargo run -- train --data <PATH> [options]
```

Useful options:

- `--format lines|text`
- `--steps <N>`
- `--batch-size <N>`
- `--block-size <N>`
- `--n-layer <N>`
- `--n-embd <N>`
- `--n-head <N>`
- `--lr <F32>`
- `--beta1 <F32>`
- `--beta2 <F32>`
- `--eps <F32>`
- `--seed <N>`
- `--sample-every <N>`
- `--device cpu|auto|gpu`
- `--profile`
- `--separate-eos`
- `--checkpoint-out <PATH>`
- `--resume <PATH>`

### Sample

```bash
cargo run -- sample --checkpoint <PATH> [options]
```

Useful options:

- `--prompt <TEXT>`
- `--temperature <F32>`
- `--max-new-tokens <N>`
- `--samples <N>`
- `--seed <N>`
- `--device cpu|auto|gpu`
- `--profile`

### Chat

```bash
cargo run -- chat --checkpoint <PATH> [options]
```

Useful options:

- `--system <TEXT>`
- `--temperature <F32>`
- `--max-new-tokens <N>`
- `--seed <N>`
- `--device cpu|auto|gpu`

### GPU Info

```bash
cargo run -- gpu-info [--device cpu|auto|gpu]
```

This reports the adapter `wgpu` would use for compute. In sandboxed or headless environments it may report that no compatible GPU adapter is available, in which case `train`, `sample`, and `chat` will use CPU when run with `--device auto`.

### Bench-Train

```bash
cargo run -- bench-train --data <PATH> [train options] [--iters <N>] [--warmup <N>]
```

### Bench-Sample

```bash
cargo run -- bench-sample --checkpoint <PATH> [sample options] [--iters <N>] [--warmup <N>]
```

## Engineering Notes

- The Rust baseline follows the verified behavior of the local `gpt.py` before optional architectural changes.
- See `architecture.md` for the expansion path toward larger architectures and research tooling.
- GPU compute is intentionally educational: `src/gpu.rs` uses explicit WGSL kernels instead of hiding the math behind a large framework.
- Training uses the same backend abstraction as inference.
- The current GPU scope keeps parameter buffers, parameter gradients, and optimizer state on device for GPU training.
- Dense matrix-vector work, linear gradient accumulation, embedding row-gradient accumulation, and GPU-training Adam run on device, while attention score loops, norm/relu backward helpers, softmax backward, and training attention flow still remain on CPU.
