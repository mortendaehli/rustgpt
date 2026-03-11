use crate::core::config::{
    ActivationKind, BoundaryMode, ChatTemplateKind, DataFormat, DeviceKind, LrScheduleKind,
    PositionEncodingKind, TokenizerModelKind, TrainMode,
};

use super::{Command, parse_args};

#[test]
fn train_parser_applies_overrides() {
    let command = parse_args([
        "rustgpt".to_string(),
        "train".to_string(),
        "--data".to_string(),
        "names.txt".to_string(),
        "--format".to_string(),
        "text".to_string(),
        "--n-embd".to_string(),
        "32".to_string(),
        "--device".to_string(),
        "gpu".to_string(),
        "--separate-eos".to_string(),
    ])
    .unwrap();

    let Command::Train(parsed) = command else {
        panic!("expected train command");
    };
    assert_eq!(parsed.data.data_path.to_string_lossy(), "names.txt");
    assert_eq!(parsed.data.format, DataFormat::PlainText);
    assert_eq!(parsed.train.n_embd, 32);
    assert_eq!(parsed.train.device, DeviceKind::Gpu);
    assert!(!parsed.train.profile);
    assert_eq!(parsed.train.mode, TrainMode::Auto);
    assert_eq!(parsed.train.boundary_mode, BoundaryMode::SeparateBosEos);
}

#[test]
fn train_parser_reads_mode_override() {
    let command = parse_args([
        "rustgpt".to_string(),
        "train".to_string(),
        "--data".to_string(),
        "chat.jsonl".to_string(),
        "--format".to_string(),
        "jsonl-chat".to_string(),
        "--mode".to_string(),
        "sft".to_string(),
    ])
    .unwrap();

    let Command::Train(parsed) = command else {
        panic!("expected train command");
    };
    assert_eq!(parsed.data.format, DataFormat::JsonlChat);
    assert_eq!(parsed.train.mode, TrainMode::Sft);
}

#[test]
fn train_parser_reads_external_tokenizer_flags() {
    let command = parse_args([
        "rustgpt".to_string(),
        "train".to_string(),
        "--data".to_string(),
        "chat.jsonl".to_string(),
        "--format".to_string(),
        "jsonl-chat".to_string(),
        "--tokenizer".to_string(),
        "tokenizer.json".to_string(),
        "--bos-token".to_string(),
        "<s>".to_string(),
        "--eos-token".to_string(),
        "</s>".to_string(),
        "--chat-template".to_string(),
        "chatml".to_string(),
    ])
    .unwrap();

    let Command::Train(parsed) = command else {
        panic!("expected train command");
    };
    assert_eq!(
        parsed
            .data
            .tokenizer_path
            .as_ref()
            .unwrap()
            .to_string_lossy(),
        "tokenizer.json"
    );
    assert_eq!(parsed.data.tokenizer_bos.as_deref(), Some("<s>"));
    assert_eq!(parsed.data.tokenizer_eos.as_deref(), Some("</s>"));
    assert_eq!(parsed.data.chat_template, ChatTemplateKind::ChatMl);
}

#[test]
fn train_parser_reads_activation_override() {
    let command = parse_args([
        "rustgpt".to_string(),
        "train".to_string(),
        "--data".to_string(),
        "names.txt".to_string(),
        "--activation".to_string(),
        "gelu".to_string(),
    ])
    .unwrap();

    let Command::Train(parsed) = command else {
        panic!("expected train command");
    };
    assert_eq!(parsed.train.activation, ActivationKind::Gelu);
}

#[test]
fn train_parser_reads_swiglu_activation_override() {
    let command = parse_args([
        "rustgpt".to_string(),
        "train".to_string(),
        "--data".to_string(),
        "names.txt".to_string(),
        "--activation".to_string(),
        "swiglu".to_string(),
    ])
    .unwrap();

    let Command::Train(parsed) = command else {
        panic!("expected train command");
    };
    assert_eq!(parsed.train.activation, ActivationKind::SwiGlu);
}

#[test]
fn train_parser_reads_position_override() {
    let command = parse_args([
        "rustgpt".to_string(),
        "train".to_string(),
        "--data".to_string(),
        "names.txt".to_string(),
        "--position".to_string(),
        "rope".to_string(),
    ])
    .unwrap();

    let Command::Train(parsed) = command else {
        panic!("expected train command");
    };
    assert_eq!(parsed.train.position_encoding, PositionEncodingKind::Rope);
}

#[test]
fn train_parser_reads_kv_head_override() {
    let command = parse_args([
        "rustgpt".to_string(),
        "train".to_string(),
        "--data".to_string(),
        "names.txt".to_string(),
        "--n-head".to_string(),
        "8".to_string(),
        "--n-kv-head".to_string(),
        "2".to_string(),
    ])
    .unwrap();

    let Command::Train(parsed) = command else {
        panic!("expected train command");
    };
    assert_eq!(parsed.train.n_head, 8);
    assert_eq!(parsed.train.n_kv_head, 2);
}

#[test]
fn train_parser_enables_profile_flag() {
    let command = parse_args([
        "rustgpt".to_string(),
        "train".to_string(),
        "--data".to_string(),
        "names.txt".to_string(),
        "--profile".to_string(),
    ])
    .unwrap();

    let Command::Train(parsed) = command else {
        panic!("expected train command");
    };
    assert!(parsed.train.profile);
}

#[test]
fn train_parser_reads_validation_and_schedule_flags() {
    let command = parse_args([
        "rustgpt".to_string(),
        "train".to_string(),
        "--data".to_string(),
        "train.jsonl".to_string(),
        "--valid-data".to_string(),
        "valid.jsonl".to_string(),
        "--valid-format".to_string(),
        "jsonl-chat".to_string(),
        "--valid-chat-template".to_string(),
        "chatml".to_string(),
        "--warmup-steps".to_string(),
        "10".to_string(),
        "--lr-schedule".to_string(),
        "cosine".to_string(),
        "--weight-decay".to_string(),
        "0.1".to_string(),
        "--grad-clip".to_string(),
        "1.0".to_string(),
        "--valid-ratio".to_string(),
        "0.2".to_string(),
    ])
    .unwrap();

    let Command::Train(parsed) = command else {
        panic!("expected train command");
    };
    assert_eq!(
        parsed
            .validation_data
            .as_ref()
            .unwrap()
            .data_path
            .to_string_lossy(),
        "valid.jsonl"
    );
    assert_eq!(
        parsed.validation_data.as_ref().unwrap().format,
        DataFormat::JsonlChat
    );
    assert_eq!(
        parsed.validation_data.as_ref().unwrap().chat_template,
        ChatTemplateKind::ChatMl
    );
    assert_eq!(parsed.train.warmup_steps, 10);
    assert_eq!(parsed.train.lr_schedule, LrScheduleKind::Cosine);
    assert_eq!(parsed.train.weight_decay, 0.1);
    assert_eq!(parsed.train.grad_clip, 1.0);
    assert_eq!(parsed.train.validation_ratio, 0.2);
}

#[test]
fn prepare_data_parser_reads_output_options() {
    let command = parse_args([
        "rustgpt".to_string(),
        "prepare-data".to_string(),
        "--data".to_string(),
        "raw.parquet".to_string(),
        "--format".to_string(),
        "parquet-chat".to_string(),
        "--out".to_string(),
        "prepared.jsonl".to_string(),
        "--out-format".to_string(),
        "jsonl-chat".to_string(),
        "--pretty".to_string(),
    ])
    .unwrap();

    let Command::PrepareData(parsed) = command else {
        panic!("expected prepare-data command");
    };
    assert_eq!(parsed.data.format, DataFormat::ParquetChat);
    assert_eq!(parsed.prepare.output_format, DataFormat::JsonlChat);
    assert!(parsed.prepare.pretty);
}

#[test]
fn train_tokenizer_parser_reads_bpe_options() {
    let command = parse_args([
        "rustgpt".to_string(),
        "train-tokenizer".to_string(),
        "--data".to_string(),
        "tiny.txt".to_string(),
        "--format".to_string(),
        "text".to_string(),
        "--out".to_string(),
        "tokenizer.json".to_string(),
        "--model".to_string(),
        "bpe".to_string(),
        "--vocab-size".to_string(),
        "4096".to_string(),
        "--min-frequency".to_string(),
        "3".to_string(),
        "--show-progress".to_string(),
    ])
    .unwrap();

    let Command::TrainTokenizer(parsed) = command else {
        panic!("expected train-tokenizer command");
    };
    assert_eq!(parsed.data.format, DataFormat::PlainText);
    assert_eq!(parsed.tokenizer.model, TokenizerModelKind::Bpe);
    assert_eq!(parsed.tokenizer.vocab_size, 4096);
    assert_eq!(parsed.tokenizer.min_frequency, 3);
    assert!(parsed.tokenizer.show_progress);
}

#[test]
fn sample_parser_requires_checkpoint() {
    let command = parse_args([
        "rustgpt".to_string(),
        "sample".to_string(),
        "--checkpoint".to_string(),
        "model.ckpt".to_string(),
        "--temperature".to_string(),
        "0.8".to_string(),
        "--device".to_string(),
        "cpu".to_string(),
    ])
    .unwrap();

    let Command::Sample(parsed) = command else {
        panic!("expected sample command");
    };
    assert_eq!(parsed.checkpoint.to_string_lossy(), "model.ckpt");
    assert_eq!(parsed.sample.temperature, 0.8);
    assert_eq!(parsed.sample.device, DeviceKind::Cpu);
    assert!(!parsed.sample.profile);
}

#[test]
fn sample_parser_reads_sampling_controls() {
    let command = parse_args([
        "rustgpt".to_string(),
        "sample".to_string(),
        "--checkpoint".to_string(),
        "model.ckpt".to_string(),
        "--top-k".to_string(),
        "16".to_string(),
        "--top-p".to_string(),
        "0.9".to_string(),
        "--repetition-penalty".to_string(),
        "1.2".to_string(),
    ])
    .unwrap();

    let Command::Sample(parsed) = command else {
        panic!("expected sample command");
    };
    assert_eq!(parsed.sample.top_k, 16);
    assert_eq!(parsed.sample.top_p, 0.9);
    assert_eq!(parsed.sample.repetition_penalty, 1.2);
}

#[test]
fn sample_parser_enables_profile_flag() {
    let command = parse_args([
        "rustgpt".to_string(),
        "sample".to_string(),
        "--checkpoint".to_string(),
        "model.ckpt".to_string(),
        "--profile".to_string(),
    ])
    .unwrap();

    let Command::Sample(parsed) = command else {
        panic!("expected sample command");
    };
    assert!(parsed.sample.profile);
}

#[test]
fn chat_parser_reads_checkpoint_and_system_prompt() {
    let command = parse_args([
        "rustgpt".to_string(),
        "chat".to_string(),
        "--checkpoint".to_string(),
        "model.ckpt".to_string(),
        "--system".to_string(),
        "be terse".to_string(),
        "--device".to_string(),
        "gpu".to_string(),
    ])
    .unwrap();

    let Command::Chat(parsed) = command else {
        panic!("expected chat command");
    };
    assert_eq!(parsed.checkpoint.to_string_lossy(), "model.ckpt");
    assert_eq!(parsed.chat.system_prompt, "be terse");
    assert_eq!(parsed.chat.device, DeviceKind::Gpu);
}

#[test]
fn chat_parser_reads_stream_flag() {
    let command = parse_args([
        "rustgpt".to_string(),
        "chat".to_string(),
        "--checkpoint".to_string(),
        "model.ckpt".to_string(),
        "--stream".to_string(),
    ])
    .unwrap();

    let Command::Chat(parsed) = command else {
        panic!("expected chat command");
    };
    assert!(parsed.chat.stream);
}

#[test]
fn eval_parser_reads_optional_data_and_prompts() {
    let command = parse_args([
        "rustgpt".to_string(),
        "eval".to_string(),
        "--checkpoint".to_string(),
        "model.ckpt".to_string(),
        "--data".to_string(),
        "heldout.jsonl".to_string(),
        "--format".to_string(),
        "jsonl-chat".to_string(),
        "--prompt".to_string(),
        "User: hello\nAssistant:".to_string(),
        "--prompt-file".to_string(),
        "evals/assistant_termination_simple.jsonl".to_string(),
        "--max-examples".to_string(),
        "8".to_string(),
    ])
    .unwrap();

    let Command::Eval(parsed) = command else {
        panic!("expected eval command");
    };
    assert_eq!(parsed.checkpoint.to_string_lossy(), "model.ckpt");
    assert_eq!(
        parsed.data.as_ref().unwrap().data_path.to_string_lossy(),
        "heldout.jsonl"
    );
    assert_eq!(parsed.data.as_ref().unwrap().format, DataFormat::JsonlChat);
    assert_eq!(parsed.eval.max_examples, 8);
    assert_eq!(parsed.eval.prompts.len(), 1);
    assert_eq!(parsed.eval.prompt_files.len(), 1);
}

#[test]
fn gpu_info_parser_reads_device_override() {
    let command = parse_args([
        "rustgpt".to_string(),
        "gpu-info".to_string(),
        "--device".to_string(),
        "gpu".to_string(),
    ])
    .unwrap();

    let Command::GpuInfo(parsed) = command else {
        panic!("expected gpu-info command");
    };
    assert_eq!(parsed.gpu.device, DeviceKind::Gpu);
}

#[test]
fn bench_train_parser_reads_benchmark_flags() {
    let command = parse_args([
        "rustgpt".to_string(),
        "bench-train".to_string(),
        "--data".to_string(),
        "names.txt".to_string(),
        "--steps".to_string(),
        "4".to_string(),
        "--batch-size".to_string(),
        "3".to_string(),
        "--iters".to_string(),
        "3".to_string(),
        "--warmup".to_string(),
        "2".to_string(),
    ])
    .unwrap();

    let Command::BenchTrain(parsed) = command else {
        panic!("expected bench-train command");
    };
    assert_eq!(parsed.train.data.data_path.to_string_lossy(), "names.txt");
    assert_eq!(parsed.train.train.steps, 4);
    assert_eq!(parsed.train.train.batch_size, 3);
    assert_eq!(parsed.bench.iterations, 3);
    assert_eq!(parsed.bench.warmup, 2);
}

#[test]
fn bench_compare_train_parser_reads_benchmark_flags() {
    let command = parse_args([
        "rustgpt".to_string(),
        "bench-compare-train".to_string(),
        "--data".to_string(),
        "names.txt".to_string(),
        "--steps".to_string(),
        "4".to_string(),
        "--batch-size".to_string(),
        "3".to_string(),
        "--iters".to_string(),
        "3".to_string(),
        "--warmup".to_string(),
        "2".to_string(),
    ])
    .unwrap();

    let Command::BenchCompareTrain(parsed) = command else {
        panic!("expected bench-compare-train command");
    };
    assert_eq!(parsed.train.data.data_path.to_string_lossy(), "names.txt");
    assert_eq!(parsed.train.train.steps, 4);
    assert_eq!(parsed.train.train.batch_size, 3);
    assert_eq!(parsed.bench.iterations, 3);
    assert_eq!(parsed.bench.warmup, 2);
}

#[test]
fn bench_sample_parser_reads_benchmark_flags() {
    let command = parse_args([
        "rustgpt".to_string(),
        "bench-sample".to_string(),
        "--checkpoint".to_string(),
        "model.ckpt".to_string(),
        "--iters".to_string(),
        "4".to_string(),
        "--warmup".to_string(),
        "1".to_string(),
    ])
    .unwrap();

    let Command::BenchSample(parsed) = command else {
        panic!("expected bench-sample command");
    };
    assert_eq!(parsed.sample.checkpoint.to_string_lossy(), "model.ckpt");
    assert_eq!(parsed.bench.iterations, 4);
    assert_eq!(parsed.bench.warmup, 1);
}
