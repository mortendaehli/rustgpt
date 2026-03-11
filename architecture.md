# RustGPT Architecture

RustGPT is a single educational codebase for a modern small LLM on Apple Silicon.

This document describes the supported mainline architecture that students should learn from. If something is not described here, it should be treated as out of scope for the main teaching path.

## Design Goals

The architecture is optimized for teaching, not for frontier benchmarking.

The goals are:

- one clean modern dense LLM path
- clear module boundaries
- truthful implementation with no “planned but not implemented” surfaces
- code that can be read top-down by third-year systems students
- runs that fit an M5 MacBook Pro with `16 GB` unified memory

## Supported Architecture

The mainline implementation supports one architecture family:

- dense decoder-only causal Transformer
- BPE tokenizer or external `tokenizer.json`
- pre-norm `RMSNorm`
- causal self-attention with standard multi-head attention or grouped-query attention
- `RoPE` or learned absolute positions
- `SwiGLU`, `GELU`, or `ReLU` MLP
- tied or untied LM head
- `AdamW` with warmup, cosine or linear decay, and gradient clipping
- gradient accumulation and optional activation checkpointing
- checkpointed training, evaluation, sampling, and terminal chat

The mainline intentionally does not include:

- scratch tensor or matrix training paths
- MoE
- sparse or local attention
- long-context tricks beyond classroom-sized context windows
- RLHF or RLVR
- CLI flags for features that are not implemented

## End-To-End Flow

The main learning flow is:

1. raw documents or chat records are loaded in `data`
2. `data` normalizes them and turns them into training sequences
3. `model` maps token IDs to logits
4. `train` computes weighted next-token loss and updates parameters
5. `runtime` saves checkpoints and resolves devices
6. `infer` reuses the same model for sampling and chat
7. `app` only wires the whole system to the CLI

This is the key architectural rule in the repo:

- `model` should not know about files, CLI, or datasets
- `data` should not know about optimizer logic
- `train` should not implement model internals
- `app` should not contain reusable ML logic

## Module Boundaries

```text
src/
  app/        CLI and command wiring
  data/       datasets, schemas, tokenization, training examples
  model/      decoder-only transformer implementation
  train/      optimizer config, training loop, validation, metrics
  infer/      sampling, stop conditions, chat session logic
  runtime/    checkpointing, device selection, profiling
  core/       shared config, error, and RNG utilities
```

### `app`

Responsibilities:

- parse CLI arguments
- dispatch commands
- print progress and summaries

Rules:

- no reusable ML logic
- no duplicated model behavior

### `data`

Responsibilities:

- load `lines`, `text`, `jsonl-*`, and `parquet-*`
- normalize text and chat records
- train or load tokenizers
- convert normalized records into token IDs and loss masks

Rules:

- the output of this layer is token-oriented, not string-oriented
- chat formatting belongs here or in `infer`, never in `model`

### `model`

Responsibilities:

- token embeddings
- optional learned position embeddings
- RoPE application
- causal self-attention
- grouped-query attention when `n_kv_head < n_head`
- feed-forward blocks
- decoder stack
- KV-cache aware inference

Rules:

- no file I/O
- no CLI concerns
- no dataset loading

### `train`

Responsibilities:

- prepare training runs from configs and datasets
- build batches
- compute weighted cross-entropy loss
- step the optimizer
- run validation
- produce logs and summaries

Rules:

- consume `model` and `data`
- use `runtime` for checkpoints and device handling

### `infer`

Responsibilities:

- next-token sampling
- stop conditions
- prompt truncation
- chat session state

Rules:

- reuse the same `model` path as training
- keep sampling policy separate from CLI concerns

### `runtime`

Responsibilities:

- CPU and GPU device resolution
- checkpoint save and load
- runtime profiling

Rules:

- runtime concerns should not leak model semantics into other layers

## Training Modes

There are two supported training modes in the teaching path:

- pretraining: next-token prediction over plain text
- SFT: next-token prediction over chat transcripts, with assistant-only loss masking

Students should notice that the optimizer and the model stay the same across both modes. What changes is the data representation and the loss mask.

## Key Invariants

These are important invariants students should rely on when reading or modifying the code:

- the same `LanguageModel` implementation is used for training and inference
- checkpoints are the boundary between runs and include model config plus tokenizer state
- `n_embd` must be divisible by `n_head`
- grouped-query attention is valid only when `n_kv_head` divides `n_head`
- RoPE requires an even head dimension
- presets are meant to be safe starting points, not hidden magic

## Suggested Reading Order

Read the code in this order:

1. `src/model/lm.rs`
2. `src/train/training.rs`
3. `src/data/training_data.rs`
4. `src/infer/sample.rs`
5. `src/runtime/checkpoint.rs`
6. `src/app/cli/`

Why this order:

- `model` contains the mathematical core
- `train` shows how the core is optimized
- `data` explains what the model is actually trained on
- `infer` shows how generation reuses the same core
- `runtime` explains persistence and resume
- `app` is only the outer wiring

## Recommended Classroom Targets

All targets assume Apple Silicon with `16 GB` unified memory.

- `debug-tiny`: smoke test, 4 layers, `d_model=256`, `seq_len=128`, CPU by default
- `class-small`: default classroom pretraining baseline, 8 layers, `d_model=512`, `seq_len=256`, micro-batch `4`, grad accumulation `2`
- `class-serious`: longer instructor-sized run, 12 layers, `d_model=768`, `seq_len=512`, micro-batch `2`, grad accumulation `4`, activation checkpointing enabled
- `class-chat`: short chat SFT continuation over a pretrained checkpoint, 12 layers, `d_model=768`, `seq_len=512`, micro-batch `2`, grad accumulation `4`, activation checkpointing enabled

The recommended student progression is:

1. `debug-tiny`
2. `class-small`
3. `class-chat`
4. inspect `class-serious` as the scaled-up version of the same codepath

## What Is Deferred

These are explicitly deferred until they are fully implemented and tested:

- 1024-context training as the default classroom path
- quantized inference
- LoRA

Those are useful extensions, but they are not part of the core teaching architecture.
