use crate::core::presets::train_presets_help;

pub(super) fn global_help(bin: &str) -> String {
    format!(
        "\
RustGPT

Usage:
  {bin} train [options]
  {bin} bench-train [options]
  {bin} bench-compare-train [options]
  {bin} inspect-vocab [options]
  {bin} prepare-data [options]
  {bin} train-tokenizer [options]
  {bin} sample [options]
  {bin} bench-sample [options]
  {bin} chat [options]
  {bin} eval [options]
  {bin} gpu-info [options]
  {bin} help

Commands:
  train          Train a model from data or resume from a checkpoint
  bench-train    Benchmark repeated training runs with runtime stage timings
  bench-compare-train
                 Benchmark train on cpu and gpu and print a speedup summary
  inspect-vocab  Inspect the tokenizer vocabulary and encoded boundary tokens
  prepare-data   Normalize and export datasets as text, JSONL, or Parquet
  train-tokenizer
                 Train and save a tokenizer.json asset from normalized data
  sample         Load a checkpoint and generate samples
  bench-sample   Benchmark repeated sampling runs with runtime stage timings
  chat           Start a simple terminal chat loop over a checkpoint
  eval           Evaluate checkpoint loss on held-out data and optional prompts
  gpu-info       Inspect the GPU adapter that wgpu would use for inference

Run `{bin} <command> --help` for command-specific options."
    )
}

pub(super) fn train_help(bin: &str) -> String {
    let preset_help = train_presets_help();
    format!(
        "\
Usage:
  {bin} train --data <PATH> [options]

Options:
  --data <PATH>         Path to the training corpus. Default: input.txt
  --preset <NAME>       Apply one of: {preset_help}. Later flags override the preset
  --checkpoint-out <P>  Write a checkpoint after training
  --resume <P>          Resume model weights and optimizer state from a checkpoint
  --best-checkpoint-out <P>
                        Save the best validation-loss checkpoint separately
  --format <MODE>       Dataset format: lines | text | jsonl-text | jsonl-chat | parquet-text | parquet-chat. Default: lines
  --tokenizer <PATH>    Optional external tokenizer.json asset
  --bos-token <TEXT>    Override BOS token lookup for external tokenizers
  --eos-token <TEXT>    Override EOS token lookup for external tokenizers
  --chat-template <K>   Chat template for structured chat data: simple | chatml. Default: simple
  --no-shuffle          Disable deterministic dataset shuffling
  --lowercase           Lowercase the corpus before tokenization
  --mode <MODE>         Training mode: auto | pretrain | sft. Default: auto
  --steps <N>           Training steps. Default: 1000
  --valid-data <PATH>   Optional held-out dataset path for validation loss
  --valid-format <MODE> Held-out format. Default: same as --format
  --valid-chat-template <K>
                        Held-out chat template. Default: same as --chat-template
  --valid-lowercase     Lowercase the held-out dataset before tokenization
  --valid-ratio <F32>   Split off a held-out ratio from the training data. Default: 0.0
  --valid-max-examples <N>
                        Maximum validation examples/windows per eval. Default: 64
  --batch-size <N>      Sequences per optimizer step. Default: 1
  --grad-accum-steps <N>
                        Gradient accumulation micro-steps per optimizer step. Default: 1
  --activation-checkpointing
                        Recompute selected activations during backward to reduce memory use
  --seed <N>            RNG seed. Default: 42
  --sample-every <N>    Sampling/log interval. Default: 100
  --block-size <N>      Context window size. Default: 16
  --n-layer <N>         Number of transformer layers. Default: 1
  --n-embd <N>          Embedding width. Default: 16
  --n-head <N>          Attention heads. Default: 4
  --n-kv-head <N>       Key/value heads. Must divide --n-head for grouped-query attention
  --tied-embeddings     Reuse token embeddings as the output projection
  --activation <K>      MLP activation: relu | gelu | swiglu. Default: relu
  --position <K>        Position encoding: learned | rope. Default: learned
  --lr <F32>            Learning rate. Default: 0.01
  --beta1 <F32>         Adam beta1. Default: 0.85
  --beta2 <F32>         Adam beta2. Default: 0.99
  --eps <F32>           Adam epsilon. Default: 1e-8
  --weight-decay <F32>  AdamW-style decoupled weight decay. Default: 0.0
  --warmup-steps <N>    Learning-rate warmup steps. Default: 0
  --lr-schedule <K>     Learning-rate schedule: linear | cosine. Default: linear
  --grad-clip <F32>     Global gradient clipping threshold. Default: 0.0 (disabled)
  --device <MODE>       Training device: cpu | auto | gpu. Default: cpu
  --profile             Print cumulative runtime stage timings
  --separate-eos        Use distinct BOS and EOS instead of shared BOS boundaries"
    )
}

pub(super) fn inspect_vocab_help(bin: &str) -> String {
    format!(
        "\
Usage:
  {bin} inspect-vocab --data <PATH> [options]

Options:
  --data <PATH>         Path to the corpus. Default: input.txt
  --format <MODE>       Dataset format: lines | text | jsonl-text | jsonl-chat | parquet-text | parquet-chat. Default: lines
  --tokenizer <PATH>    Optional external tokenizer.json asset
  --bos-token <TEXT>    Override BOS token lookup for external tokenizers
  --eos-token <TEXT>    Override EOS token lookup for external tokenizers
  --chat-template <K>   Chat template for structured chat data: simple | chatml. Default: simple
  --lowercase           Lowercase the corpus before tokenization
  --show-tokens <N>     Number of symbols to print. Default: 32
  --separate-eos        Use distinct BOS and EOS instead of shared BOS boundaries"
    )
}

pub(super) fn sample_help(bin: &str) -> String {
    format!(
        "\
Usage:
  {bin} sample --checkpoint <PATH> [options]

Options:
  --checkpoint <PATH>   Path to a saved checkpoint
  --prompt <TEXT>       Prompt prefix to condition on
  --temperature <F32>   Sampling temperature. Default: 0.5
  --top-k <N>           Keep only the top-k tokens. Default: 40
  --top-p <F32>         Nucleus sampling threshold. Default: 1.0
  --repetition-penalty <F32>
                       Penalize already-seen tokens. Default: 1.0
  --presence-penalty <F32>
                       Subtract once for seen tokens. Default: 0.0
  --frequency-penalty <F32>
                       Subtract in proportion to seen count. Default: 0.0
  --max-new-tokens <N>  Maximum tokens to generate. Default: 16
  --samples <N>         Number of samples to emit. Default: 5
  --seed <N>            RNG seed for sampling. Default: 42
  --device <MODE>       Inference device: cpu | auto | gpu. Default: cpu
  --profile             Print cumulative runtime stage timings"
    )
}

pub(super) fn prepare_data_help(bin: &str) -> String {
    format!(
        "\
Usage:
  {bin} prepare-data --data <PATH> [options]

Options:
  --data <PATH>         Path to the source dataset
  --format <MODE>       Source format: lines | text | jsonl-text | jsonl-chat | parquet-text | parquet-chat. Default: lines
  --chat-template <K>   Chat template used when rendering chat records to text. Default: simple
  --lowercase           Lowercase the source dataset before export
  --out <PATH>          Output path. Default: prepared.jsonl
  --out-format <MODE>   Output format: lines | text | jsonl-text | jsonl-chat | parquet-text | parquet-chat. Default: jsonl-text
  --dedup               Drop exact duplicate rendered documents during export
  --min-chars <N>       Drop records shorter than N rendered characters. Default: 0
  --max-chars <N>       Drop records longer than N rendered characters. Default: 0 (disabled)
  --min-messages <N>    Drop chat records with fewer than N messages. Default: 0
  --require-assistant   Drop chat records that do not contain an assistant turn
  --pretty              Pretty-print JSONL rows for inspection"
    )
}

pub(super) fn train_tokenizer_help(bin: &str) -> String {
    format!(
        "\
Usage:
  {bin} train-tokenizer --data <PATH> [options]

Options:
  --data <PATH>         Path to the tokenizer-training dataset
  --format <MODE>       Dataset format: lines | text | jsonl-text | jsonl-chat | parquet-text | parquet-chat. Default: lines
  --chat-template <K>   Chat template used when rendering structured chat data. Default: simple
  --lowercase           Lowercase the dataset before tokenizer training
  --out <PATH>          Output tokenizer.json path. Default: tokenizer.json
  --model <K>           Tokenizer model: bpe. Default: bpe
  --vocab-size <N>      Target vocabulary size. Default: 2048
  --min-frequency <N>   Minimum pair frequency. Default: 2
  --show-progress       Show tokenizer training progress"
    )
}

pub(super) fn bench_train_help(bin: &str) -> String {
    format!(
        "{}\n  --iters <N>          Measured benchmark iterations. Default: 5\n  --warmup <N>         Warmup iterations excluded from summary. Default: 1",
        train_help(bin).replace(&format!("{bin} train"), &format!("{bin} bench-train"))
    )
}

pub(super) fn bench_compare_train_help(bin: &str) -> String {
    format!(
        "{}\n\nThis command ignores any --device flag in the forwarded train options and runs the same benchmark on cpu and gpu back-to-back.\n  --iters <N>          Measured benchmark iterations. Default: 5\n  --warmup <N>         Warmup iterations excluded from summary. Default: 1",
        train_help(bin).replace(
            &format!("{bin} train"),
            &format!("{bin} bench-compare-train")
        )
    )
}

pub(super) fn bench_sample_help(bin: &str) -> String {
    format!(
        "{}\n  --iters <N>          Measured benchmark iterations. Default: 5\n  --warmup <N>         Warmup iterations excluded from summary. Default: 1",
        sample_help(bin).replace(&format!("{bin} sample"), &format!("{bin} bench-sample"))
    )
}

pub(super) fn chat_help(bin: &str) -> String {
    format!(
        "\
Usage:
  {bin} chat --checkpoint <PATH> [options]

Options:
  --checkpoint <PATH>   Path to a saved checkpoint
  --system <TEXT>       Optional system prompt prepended to the chat transcript
  --temperature <F32>   Sampling temperature. Default: 0.5
  --top-k <N>           Keep only the top-k tokens. Default: 40
  --top-p <F32>         Nucleus sampling threshold. Default: 1.0
  --repetition-penalty <F32>
                       Penalize already-seen tokens. Default: 1.0
  --presence-penalty <F32>
                       Subtract once for seen tokens. Default: 0.0
  --frequency-penalty <F32>
                       Subtract in proportion to seen count. Default: 0.0
  --max-new-tokens <N>  Maximum tokens per assistant turn. Default: 32
  --seed <N>            RNG seed for chat sampling. Default: 42
  --device <MODE>       Inference device: cpu | auto | gpu. Default: cpu
  --stream              Stream assistant text as tokens are decoded"
    )
}

pub(super) fn eval_help(bin: &str) -> String {
    format!(
        "\
Usage:
  {bin} eval --checkpoint <PATH> [options]

Options:
  --checkpoint <PATH>   Path to a saved checkpoint
  --data <PATH>         Optional held-out dataset for loss / perplexity evaluation
  --format <MODE>       Dataset format: lines | text | jsonl-text | jsonl-chat | parquet-text | parquet-chat
  --chat-template <K>   Chat template for structured held-out chat data. Default: simple
  --lowercase           Lowercase the dataset before tokenization
  --max-examples <N>    Maximum evaluation examples or windows. Default: 64
  --prompt <TEXT>       Prompt to sample during evaluation. May be repeated
  --prompt-file <PATH>  JSONL prompt-eval suite with checked cases. May be repeated
  --temperature <F32>   Sampling temperature for prompt evals. Default: 0.7
  --top-k <N>           Keep only the top-k tokens. Default: 0 (disabled)
  --top-p <F32>         Nucleus sampling threshold. Default: 1.0
  --repetition-penalty <F32>
                       Penalize already-seen tokens. Default: 1.0
  --presence-penalty <F32>
                       Subtract once for seen tokens. Default: 0.0
  --frequency-penalty <F32>
                       Subtract in proportion to seen count. Default: 0.0
  --max-new-tokens <N>  Maximum tokens for prompt completions. Default: 32
  --device <MODE>       Inference/eval device: cpu | auto | gpu. Default: cpu"
    )
}

pub(super) fn gpu_info_help(bin: &str) -> String {
    format!(
        "\
Usage:
  {bin} gpu-info [options]

Options:
  --device <MODE>       Adapter preference: cpu | auto | gpu. Default: auto"
    )
}
