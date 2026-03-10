use crate::app::cli::SampleCommand;
use crate::core::error::Result;
use crate::core::rng::Rng;
use crate::data::checkpoint::load_checkpoint;
use crate::runtime::backend::ComputeBackend;
use crate::runtime::profile::RuntimeProfile;
use crate::runtime::sampling::generate_sample_with_backend;

pub fn run_sample(command: SampleCommand) -> Result<()> {
    let checkpoint = load_checkpoint(&command.checkpoint)?;
    let backend = ComputeBackend::from_model(&checkpoint.model, command.sample.device)?;
    let mut rng = Rng::from_seed(command.sample.seed);
    let runtime_profile = command.sample.profile.then(RuntimeProfile::default);

    println!(
        "RustGPT sample  checkpoint={}  vocab={}  params={}  seed={}  temperature={}  backend={}",
        command.checkpoint.display(),
        checkpoint.tokenizer.vocab_size(),
        checkpoint.model.num_parameters(),
        command.sample.seed,
        command.sample.temperature,
        backend.description()
    );
    println!(
        "trained_steps={}  max_new_tokens={}  samples={}",
        checkpoint.trained_steps, command.sample.max_new_tokens, command.sample.samples
    );

    for sample_idx in 0..command.sample.samples {
        let sample = generate_sample_with_backend(
            &checkpoint.model,
            &backend,
            &checkpoint.tokenizer,
            &command.sample.prompt,
            command.sample.max_new_tokens,
            command.sample.temperature,
            runtime_profile.as_ref(),
            &mut rng,
        )?;
        println!("sample {:>2}: {}", sample_idx + 1, sample);
    }

    if let Some(runtime_profile) = runtime_profile.as_ref() {
        if !runtime_profile.is_empty() {
            println!("stage timings:");
            println!("{}", runtime_profile.format_table());
        }
    }

    Ok(())
}
