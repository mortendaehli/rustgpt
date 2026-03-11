# Improvement Plan

This file is the current roadmap for RustGPT after the baseline modernization work has been completed.

The project now has:

- CPU and GPU training/inference paths
- direct `text`, `lines`, `jsonl-*`, and `parquet-*` dataset ingestion
- dataset normalization/export via `prepare-data`
- BPE tokenizer training plus external `tokenizer.json` loading
- masked SFT training for chat data
- chat templates, token-aware stop conditions, streaming, and token-budgeted REPL prompts
- RoPE, GELU, SwiGLU, tied embeddings, final RMSNorm, and GQA
- evaluation, validation splits, AdamW-style weight decay, warmup/cosine schedules, gradient clipping, and best-checkpoint saving

That means the original “make this a serious small-model educational stack” roadmap is no longer about fixing missing basics. The next roadmap is about choosing the right *next scale of ambition* without increasing cognitive load too early.

## Design Goal

Keep the project in this shape:

1. readable enough for second-year students,
2. strong enough to demonstrate modern small-model practice,
3. modular enough to support future model experiments,
4. efficient enough to train and demo on a MacBook Pro M5 16 GB.

The project should remain “framework-shaped”, not become a full training framework.

## Completed Foundation

These phases are now complete enough to treat as the stable base:

### 1. Data Layer V2

- normalized schemas for text and chat
- direct ingestion for:
  - `lines`
  - `text`
  - `jsonl-text`
  - `jsonl-chat`
  - `parquet-text`
  - `parquet-chat`
- `prepare-data` for export/normalization

### 2. Tokenizer V2

- byte tokenizer still available as the simplest teaching path
- BPE tokenizer training inside the repo
- external Hugging Face `tokenizer.json` loading
- BOS/EOS handling and tokenizer metadata in checkpoints

### 3. Proper SFT Data

- `SequenceExample { input_ids, target_ids, loss_mask }`
- assistant-only loss masking for chat SFT
- sliding-window plain-text training instead of repeated-prefix training

### 4. Chat Runtime V2

- explicit chat templates
- checkpointed template metadata
- token-aware stop conditions
- multi-line UTF-8 REPL input
- token-budgeted prompt construction and history dropping
- streaming generation

### 5. Model Modernization

- learned absolute positions or RoPE
- `relu`, `gelu`, or `swiglu`
- tied embeddings
- final RMSNorm before logits
- grouped-query attention via `n_kv_head`

### 6. Training/Eval Stability

- `eval`
- held-out validation
- validation split by ratio or explicit validation dataset
- AdamW-style weight decay
- warmup + cosine or linear schedules
- optional gradient clipping
- best-checkpoint saving

## Recommended Next Steps

The next steps should be chosen in this order.

## Phase A: Stronger Evaluation

Goal:

- make future quality improvements measurable before adding more architecture

Deliverables:

- checked-in `evals/` prompts for:
  - instruction following
  - short factual continuation
  - multi-turn retention
  - repetition failure
  - structured output sanity
- deterministic eval mode for generation checks
- a single summary command that prints:
  - held-out loss/perplexity
  - prompt outputs
  - tokens/sec

Why this is next:

- the model stack is now rich enough that “it feels better” is no longer a good iteration loop

## Phase B: Data Prep V3

Goal:

- make larger experiments repeatable and fast

Deliverables:

- pretokenized binary shards for repeated training runs
- tokenizer manifest written next to `tokenizer.json`
- dataset manifest written next to prepared JSONL/Parquet exports
- documented recommended open-dataset recipes:
  - plain-text pretrain
  - chat SFT

Why this matters:

- data prep time and schema drift become the main friction once students move beyond toy datasets

## Phase C: Decoding V3

Goal:

- improve the classroom “wow factor” of chat demos

Deliverables:

- `min-p`
- explicit greedy mode
- better stop-token handling in all prompt/eval paths
- token and character throughput reporting during chat

Why this matters:

- for small models, decoding quality changes often matter as much as architecture changes

## Phase D: Quality-Focused Model Upgrades

Goal:

- extend the model in ways that are still teachable

Recommended order:

1. MQA as a simpler special case of GQA
2. optional larger MLP ratio
3. longer context with better KV-cache ergonomics
4. optional local/sliding-window attention

What should wait:

- MoE
- 128K context
- speculative decoding
- tool use / agent runtime

These are interesting later, but they are not the next best educational step.

## Phase E: Post-Training Research

Goal:

- move from “small SFT chat model” toward “small aligned chat model”

Recommended order:

1. SFT quality sweeps on better open chat datasets
2. distillation from stronger open models
3. preference data support
4. DPO / preference optimization

Why this comes after the current stack:

- preference tuning without a solid tokenizer, data layer, eval loop, and stable SFT base is wasted effort

## Dataset Strategy

Recommended open-dataset progression:

### Pretraining

1. `TinyStories`
2. `WikiText-103 Raw`
3. sampled `FineWeb-Edu` or `Dolma`

### Chat / SFT

1. `Smol-SmolTalk`
2. `UltraChat 200k`
3. `OpenAssistant / OASST1`
4. `OpenOrca`

Keep the project focused on *high-quality subsets* and repeatable experiments, not on trying to match large production model scale from scratch.

## Hardware Target

For the M5 16 GB machine, the best practical research band is still:

- roughly 20M to 100M parameters for active iteration

Stretch band:

- 100M to 250M parameters if training becomes more of a systems exercise

Bad tradeoff:

- trying to reproduce current open small-model pretraining scale from scratch on one laptop

## Architectural Guardrails

Do not do these unless a later phase clearly needs them:

- a Burn-like internal framework
- a TensorFlow/PyTorch-style graph abstraction
- a large plugin system
- a second unrelated runtime path for “research”

The current module split is the right shape:

- `app/` for CLI entrypoints
- `core/` for reusable primitives
- `data/` for corpus/tokenizer/checkpoint/prep
- `model/` for transformer parameters and architecture helpers
- `runtime/` for forward/backward/training/sampling/backend

## Immediate Next Milestone

If only one more milestone is taken next, it should be:

1. add checked-in eval suites and deterministic generation checks,
2. add pretokenized binary shard support,
3. tune the chat demo on a real open SFT dataset using the new tokenizer workflow.

That combination gives the best next jump in:

- teaching value
- repeatability
- student-facing demo quality
- future research readiness
