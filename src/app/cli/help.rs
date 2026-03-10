pub(super) fn global_help(bin: &str) -> String {
    format!(
        "\
RustGPT

Usage:
  {bin} train [options]
  {bin} bench-train [options]
  {bin} bench-compare-train [options]
  {bin} inspect-vocab [options]
  {bin} sample [options]
  {bin} bench-sample [options]
  {bin} chat [options]
  {bin} gpu-info [options]
  {bin} help

Commands:
  train          Train a model from data or resume from a checkpoint
  bench-train    Benchmark repeated training runs with runtime stage timings
  bench-compare-train
                 Benchmark train on cpu and gpu and print a speedup summary
  inspect-vocab  Inspect the tokenizer vocabulary and encoded boundary tokens
  sample         Load a checkpoint and generate samples
  bench-sample   Benchmark repeated sampling runs with runtime stage timings
  chat           Start a simple terminal chat loop over a checkpoint
  gpu-info       Inspect the GPU adapter that wgpu would use for inference

Run `{bin} <command> --help` for command-specific options."
    )
}

pub(super) fn train_help(bin: &str) -> String {
    format!(
        "\
Usage:
  {bin} train --data <PATH> [options]

Options:
  --data <PATH>         Path to the training corpus. Default: input.txt
  --checkpoint-out <P>  Write a checkpoint after training
  --resume <P>          Resume model weights and optimizer state from a checkpoint
  --format <MODE>       Dataset format: lines | text. Default: lines
  --no-shuffle          Disable deterministic dataset shuffling
  --lowercase           Lowercase the corpus before tokenization
  --steps <N>           Training steps. Default: 1000
  --batch-size <N>      Sequences per optimizer step. Default: 1
  --seed <N>            RNG seed. Default: 42
  --sample-every <N>    Sampling/log interval. Default: 100
  --block-size <N>      Context window size. Default: 16
  --n-layer <N>         Number of transformer layers. Default: 1
  --n-embd <N>          Embedding width. Default: 16
  --n-head <N>          Attention heads. Default: 4
  --lr <F32>            Learning rate. Default: 0.01
  --beta1 <F32>         Adam beta1. Default: 0.85
  --beta2 <F32>         Adam beta2. Default: 0.99
  --eps <F32>           Adam epsilon. Default: 1e-8
  --device <MODE>       Training device: cpu | auto | gpu. Default: auto
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
  --format <MODE>       Dataset format: lines | text. Default: lines
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
  --max-new-tokens <N>  Maximum tokens to generate. Default: 16
  --samples <N>         Number of samples to emit. Default: 5
  --seed <N>            RNG seed for sampling. Default: 42
  --device <MODE>       Inference device: cpu | auto | gpu. Default: auto
  --profile             Print cumulative runtime stage timings"
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
  --max-new-tokens <N>  Maximum tokens per assistant turn. Default: 32
  --seed <N>            RNG seed for chat sampling. Default: 42
  --device <MODE>       Inference device: cpu | auto | gpu. Default: auto"
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
