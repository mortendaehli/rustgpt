use crate::core::config::{BoundaryMode, DataFormat, DeviceKind};

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
    assert_eq!(parsed.train.boundary_mode, BoundaryMode::SeparateBosEos);
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
