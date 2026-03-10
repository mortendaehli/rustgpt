# RustGPT Architecture

## Purpose

This document describes the target architecture for the Rust GPT codebase and how it should evolve from a compact educational implementation into a broader research and experimentation platform.

The architecture has three horizons:

1. A compact local GPT training and inference engine.
2. A flexible runtime that can support more modern LLM patterns.
3. A research system that can propose, run, evaluate, and safely promote improvements.

The third horizon must be designed carefully. "Self-improving" should not mean unbounded self-modification. It should mean controlled, auditable, test-gated iteration.

## Architectural Goals

- Keep the core model engine understandable and debuggable.
- Separate pure model logic from data, CLI, checkpointing, and future research automation.
- Support incremental adoption of modern transformer features.
- Support reproducible experiments and safe rollback.
- Allow a future terminal chat interface without forking the model runtime.
- Allow a future research loop without granting the model direct, unchecked authority over the codebase.

## Design Principles

### 1. Separate the model kernel from orchestration

The tensor math, model parameters, forward pass, backward pass, optimizer, and sampling logic should live in a stable core. Training workflows, evaluation pipelines, chat interfaces, and automated experimentation should depend on that core instead of reimplementing it.

### 2. Build a narrow waist

The most important interface in the system is the model runtime boundary:

- input token IDs
- model config
- checkpoint state
- output logits, losses, gradients, and samples

If this interface is clean, almost everything else can evolve independently.

### 3. Prefer compatibility layers over branching implementations

The system should avoid maintaining:

- one codepath for training
- another for sampling
- another for chat

Instead, it should expose one model engine with small wrapper layers.

### 4. Treat research automation as a separate control plane

The future research system should not be embedded directly inside the trusted model kernel. It should operate outside the core as an orchestrator that proposes experiments, runs them in isolation, and promotes results only when they pass explicit gates.

## High-Level Architecture

```text
                +---------------------------+
                |       Terminal CLI        |
                | train | sample | chat     |
                +------------+--------------+
                             |
                             v
                +---------------------------+
                |    Application Layer      |
                | config | commands | I/O   |
                +------------+--------------+
                             |
            +----------------+----------------+
            |                                 |
            v                                 v
 +-------------------------+       +-------------------------+
 |    Data/Tokenizer       |       |    Checkpoint Layer     |
 | corpus | vocab | encode |       | save | load | version   |
 +------------+------------+       +------------+------------+
              |                                 |
              +----------------+----------------+
                               |
                               v
                   +-------------------------+
                   |      Model Runtime      |
                   | forward | backward      |
                   | optimize | sample       |
                   +------------+------------+
                                |
                                v
                   +-------------------------+
                   |   Math / Tensor Kernel  |
                   | vecs | matrices | RNG   |
                   +-------------------------+
```

## Layer-by-Layer Description

## 1. Math / Tensor Kernel

This is the lowest layer and should remain small and explicit.

Responsibilities:

- matrix and vector storage
- low-level ops such as `linear`, `softmax`, `rmsnorm`
- activation functions
- RNG and Gaussian initialization
- numerical helpers

Rules:

- no business logic
- no tokenizer logic
- no CLI concerns
- keep data layout explicit

Why it matters:

If this layer stays simple, optimization and correctness work remain tractable.

## 2. Model Runtime

This is the heart of the system.

Responsibilities:

- parameter initialization
- token embedding and positional embedding
- transformer layers
- causal attention with KV cache
- forward cache creation
- analytical backward pass
- optimizer updates
- autoregressive sampling

Key interfaces:

- `forward_token(...) -> TokenForwardCache`
- `backward_token(...) -> gradients`
- `train_step(...) -> loss`
- `sample(...) -> token stream`

Why it matters:

Every higher-level feature depends on this layer being correct and stable.

## 3. Data and Tokenization

Responsibilities:

- load corpora
- normalize data where configured
- build vocabularies
- encode and decode tokens
- support multiple dataset formats over time

Near-term formats:

- line-delimited documents
- plain text chunking

Later formats:

- prompt-response pairs
- chat transcripts
- tool traces
- code corpora
- research logs and experiment summaries

Key architectural rule:

Tokenization must not be hardcoded inside the model runtime. The model only consumes integer token IDs.

## 4. Checkpoint Layer

Responsibilities:

- persist model parameters
- persist optimizer state
- persist tokenizer metadata
- persist configuration and training step
- support checkpoint versioning

Future responsibilities:

- support model family upgrades
- support migration tools
- support storing evaluation summaries with each checkpoint

Architectural rule:

Checkpoints are the handoff format between training, sampling, chat, and research orchestration.

## 5. Application Layer

Responsibilities:

- command routing
- argument parsing
- logging
- progress output
- wiring config and runtime together

Commands expected over time:

- `train`
- `sample`
- `resume`
- `inspect-vocab`
- `chat`
- `eval`
- `research`

The application layer should remain thin. If logic becomes reusable, move it down into the library.

## Core Runtime Flows

## Training flow

```text
dataset -> tokenizer -> token sequence
       -> forward pass over sequence with KV cache
       -> loss aggregation
       -> reverse-time backward pass with KV gradient accumulation
       -> optimizer update
       -> metrics + optional checkpoint
```

## Sampling flow

```text
checkpoint -> tokenizer -> BOS/prompt tokens
          -> iterative forward pass with growing KV cache
          -> logits -> temperature/sample strategy
          -> next token
          -> stop on EOS or max length
```

## Chat flow

```text
chat history -> prompt formatter -> tokens
             -> context truncation
             -> sampling loop
             -> streamed or buffered response
             -> updated chat history
```

## Recommended Near-Term Module Boundaries

While the codebase remains small, keep a single crate with modular files. When growth justifies it, split into a workspace.

### Single-crate stage

```text
rustgpt/
  src/
    lib.rs
    cli.rs
    config.rs
    tokenizer.rs
    data.rs
    model.rs
    forward.rs
    backward.rs
    optim.rs
    checkpoint.rs
    sample.rs
    train.rs
```

### Future workspace stage

```text
crates/
  core/         # model runtime and math kernel
  cli/          # terminal application
  evals/        # evaluation harnesses and benchmark tasks
  research/     # experiment planner/orchestrator
  formats/      # checkpoint and dataset schema tools
```

Do not split into a workspace immediately. Split only when module boundaries harden and multiple binaries or services appear.

## Expansion Path to More Modern LLM Architecture

The current baseline is intentionally simple. The architecture should allow the following upgrades without rewriting the system.

## Tokenization upgrades

Current:

- byte-level tokenizer with BOS/EOS support

Next:

- BPE tokenizer
- reserved control tokens
- role tags for chat

Architectural impact:

- tokenizer interface must return token IDs and metadata
- checkpoint must store tokenizer version and vocabulary

## Embedding and positional upgrades

Current:

- learned token embeddings
- learned positional embeddings

Next:

- separate `BOS` and `EOS`
- rotary embeddings
- ALiBi
- longer context support

Architectural impact:

- positional logic should live in a dedicated component, not be fused into token embedding code

## Transformer block upgrades

Current:

- RMSNorm
- causal self-attention
- plain ReLU MLP
- residual connections

Next:

- GeLU
- SwiGLU
- gated MLP blocks
- pre-norm or post-norm variants
- residual scaling
- dropout in train mode
- bias toggles

Architectural impact:

- block config should be explicit
- activation functions should be swappable by enum/config, not by copy-paste

## Attention upgrades

Current:

- full causal attention over all prior positions
- multi-head attention

Next:

- grouped-query attention
- multi-query attention
- sliding-window attention
- paged KV cache
- more efficient attention kernels

Architectural impact:

- keep KV cache as an explicit structure
- do not hardcode one cache layout forever

## Optimization and training upgrades

Current:

- Adam
- batch size = 1
- full precision `f32`

Next:

- minibatching
- gradient accumulation
- mixed precision
- gradient clipping
- learning-rate schedulers
- weight decay
- sequence packing
- curriculum learning

Architectural impact:

- optimizer state should be modular
- training loop should be separate from model math

## Adaptation and finetuning upgrades

Future:

- LoRA
- adapters
- prompt tuning
- instruction fine-tuning
- preference optimization

Architectural impact:

- checkpointing should distinguish base weights from adaptation layers
- training code should allow freezing subsets of parameters

## Inference/runtime upgrades

Future:

- top-k and nucleus sampling
- repetition penalties
- streaming token generation
- conversation templates
- tool calling interfaces

Architectural impact:

- decoding logic should be separated from raw forward pass

## Terminal Chat Interface Architecture

The terminal chat interface should be a thin layer on top of the runtime, not a separate model implementation.

### Components

- checkpoint loader
- tokenizer
- prompt formatter
- context manager
- sampler
- terminal renderer

### Prompt formatting layer

The chat layer should own prompt formatting, for example:

```text
<system>
You are a helpful assistant.
</system>
<user>
Hello
</user>
<assistant>
```

This formatting must be explicit and versioned if saved in checkpoints or evaluation fixtures.

### Context manager

Responsibilities:

- append new turns
- truncate old turns to fit `block_size`
- preserve special tokens and separators

### Decoding layer

Responsibilities:

- temperature
- top-k
- max new tokens
- stop sequences

## Architecture for a Self-Improving Research System

This section addresses the long-term goal of building a system that can improve itself through sustained AI research. The architecture must distinguish between:

- the model being improved
- the research agent proposing improvements
- the trusted evaluation and promotion system

Those must not collapse into one opaque loop.

## Control-plane architecture

```text
                +----------------------------+
                |     Research Orchestrator  |
                | hypotheses | planning      |
                +-------------+--------------+
                              |
                              v
                +----------------------------+
                |    Experiment Generator    |
                | config edits | code patches|
                +-------------+--------------+
                              |
                              v
                +----------------------------+
                |   Isolated Experiment Run  |
                | sandbox | train | eval     |
                +-------------+--------------+
                              |
                              v
                +----------------------------+
                |     Evaluation Harness     |
                | metrics | regressions      |
                +-------------+--------------+
                              |
                 +------------+-------------+
                 |                          |
                 v                          v
      +----------------------+   +----------------------+
      | Promotion Gate       |   | Experiment Archive   |
      | policy | approval    |   | data | traces | code |
      +-----------+----------+   +----------------------+
                  |
                  v
      +----------------------+
      | Trusted Main Branch  |
      | model | configs      |
      +----------------------+
```

## Research loop components

### 1. Hypothesis generator

Purpose:

- propose candidate improvements such as:
  - optimizer changes
  - architecture changes
  - tokenizer changes
  - curriculum changes
  - prompt formatting changes
  - evaluation additions

Inputs:

- prior experiment results
- failure cases
- benchmark regressions
- codebase state

Outputs:

- structured experiment proposals

### 2. Experiment generator

Purpose:

- turn a proposal into:
  - config diffs
  - code changes
  - dataset variants
  - evaluation plans

Key rule:

Generated changes must be explicit artifacts, not hidden state.

### 3. Sandboxed experiment runner

Purpose:

- execute candidate changes in isolation
- run training and evaluation
- collect logs, metrics, checkpoints, and diffs

Key rule:

Experiments never run directly against the trusted production branch or production checkpoints without isolation.

### 4. Evaluation harness

Purpose:

- determine whether an experiment is actually better

Evaluation dimensions should include:

- training loss
- validation loss
- sample quality
- benchmark tasks
- performance cost
- memory cost
- stability
- reproducibility
- safety regressions

### 5. Promotion gate

Purpose:

- decide whether an experiment can modify the trusted codebase or become the new default model

Promotion should require:

- passing tests
- passing benchmark thresholds
- no safety regressions
- human approval for high-impact changes

## Requirements for safe self-improvement

### Reproducibility

Every experiment must record:

- code revision
- config
- seed
- dataset version
- checkpoint lineage
- evaluation results

### Isolation

Research agents should operate on branches, worktrees, or sandboxed copies, not on the trusted runtime directly.

### Auditability

The system must persist:

- proposed change
- rationale
- generated diff
- experiment logs
- metrics
- promotion decision

### Rollback

Every promoted change must be reversible. Checkpoints and code versions should be addressable by stable IDs.

### Human oversight

For the foreseeable future, architecture changes, evaluation changes, and self-modifying code changes should remain human-approved.

## What "self-improving" should mean in practice

Good interpretation:

- the system proposes experiments
- runs them in sandboxes
- measures them against stable evals
- presents promotion candidates
- learns from prior results

Bad interpretation:

- the model rewrites its own core loop in place
- changes evaluation criteria to flatter itself
- promotes its own checkpoints without audit
- gains unrestricted access to code, data, and deployment

The architecture should support the first and structurally prevent the second.

## Evaluation Architecture for Research Automation

The research system needs its own durable evaluation layer.

### Evaluation tiers

Tier 1:

- unit tests
- checkpoint format tests
- deterministic smoke tests

Tier 2:

- training quality tests
- held-out validation loss
- decoding sanity tests

Tier 3:

- task benchmarks
- conversation quality rubrics
- code generation tasks
- reasoning tasks

Tier 4:

- safety and policy checks
- resource budget checks
- reproducibility checks

Promotion should require passing all relevant tiers.

## Data Architecture for Research

The model and research system should use versioned datasets.

Recommended layers:

- raw corpora
- cleaned corpora
- tokenized corpora
- train/validation/test splits
- benchmark fixtures
- conversation templates
- experiment outputs

Each dataset artifact should have:

- version ID
- schema description
- provenance
- licensing metadata

## Observability and Telemetry

Over time, the architecture should expose structured metrics such as:

- step loss
- validation loss
- gradient norm
- parameter norm
- tokens per second
- time per step
- memory footprint
- sample outputs
- evaluation pass/fail status

These metrics should feed both human dashboards and future research automation.

## Security Boundaries

If the system grows into automated research, security boundaries become part of the architecture.

Recommended boundaries:

- read-only access to trusted baselines
- write access only inside sandboxed experiment directories
- explicit approval for promotion
- restricted network access by default
- separate credentials for data access, compute access, and deployment access

## Recommended Evolution Path

## Stage 1: Compact educational engine

- single crate
- std-only
- byte tokenizer
- one-machine training
- checkpointing and sampling

## Stage 2: Practical local LLM lab

- configurable tokenizer
- better decoding
- eval harness
- terminal chat
- generalized data formats

## Stage 3: Research platform

- experiment tracking
- benchmark suite
- controlled adaptation methods
- automated proposal generation
- sandboxed experiment execution

## Stage 4: Safe self-improving system

- structured hypothesis generation
- experiment search over model, data, and optimizer choices
- trusted promotion gates
- persistent experiment memory
- human-supervised promotion and rollback

## Architectural Decision Summary

The most important decisions are:

1. Keep the model runtime as a small, trusted kernel.
2. Keep tokenizer, checkpoint, CLI, and research layers outside the kernel.
3. Add modern LLM features by extending interfaces, not by forking the runtime.
4. Treat self-improvement as a gated control-plane problem, not as unrestricted recursive self-modification.

If these constraints hold, the codebase can stay compact in the near term and still grow into a serious research platform later.
