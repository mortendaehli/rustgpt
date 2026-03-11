use burn::tensor::backend::Backend;

use crate::app::cli::SampleCommand;
use crate::core::error::Result;
use crate::core::rng::Rng;
use crate::infer::sample::{SamplingStrategy, generate_sample};
use crate::runtime::checkpoint::load_inference_checkpoint;
use crate::runtime::device::{CpuBackend, GpuBackend, ResolvedDeviceKind, cpu_device, gpu_device};
use crate::runtime::profile::RuntimeProfile;

pub fn run_sample(command: SampleCommand) -> Result<()> {
    let resolved = ResolvedDeviceKind::resolve(command.sample.device)?;
    match resolved {
        ResolvedDeviceKind::Cpu => run_sample_impl::<CpuBackend>(command, cpu_device(), resolved),
        ResolvedDeviceKind::Gpu => run_sample_impl::<GpuBackend>(command, gpu_device(), resolved),
    }
}

fn run_sample_impl<B: Backend>(
    command: SampleCommand,
    device: B::Device,
    resolved: ResolvedDeviceKind,
) -> Result<()> {
    let checkpoint = load_inference_checkpoint::<B>(&command.checkpoint, &device)?;
    let mut rng = Rng::from_seed(command.sample.seed);
    let runtime_profile = command.sample.profile.then(RuntimeProfile::default);

    println!(
        "RustGPT sample  checkpoint={}  vocab={}  params={}  seed={}  temperature={}  top_k={}  top_p={}  backend={}",
        command.checkpoint.display(),
        checkpoint.tokenizer.vocab_size(),
        checkpoint.model.num_parameters(),
        command.sample.seed,
        command.sample.temperature,
        command.sample.top_k,
        command.sample.top_p,
        resolved.description()
    );
    println!(
        "trained_steps={}  max_new_tokens={}  samples={}",
        checkpoint.trained_steps, command.sample.max_new_tokens, command.sample.samples
    );

    let strategy = SamplingStrategy {
        temperature: command.sample.temperature,
        top_k: command.sample.top_k,
        top_p: command.sample.top_p,
        repetition_penalty: command.sample.repetition_penalty,
        presence_penalty: command.sample.presence_penalty,
        frequency_penalty: command.sample.frequency_penalty,
    };

    for sample_idx in 0..command.sample.samples {
        let sample = generate_sample(
            &checkpoint.model,
            &checkpoint.tokenizer,
            &command.sample.prompt,
            command.sample.max_new_tokens,
            &strategy,
            runtime_profile.as_ref(),
            &mut rng,
        )?;
        println!("sample {:>2}: {}", sample_idx + 1, sample);
    }

    if let Some(runtime_profile) = runtime_profile.as_ref()
        && !runtime_profile.is_empty()
    {
        println!("stage timings:");
        println!("{}", runtime_profile.format_table());
    }

    Ok(())
}
