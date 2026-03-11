use crate::app::cli::TrainTokenizerCommand;
use crate::core::error::Result;
use crate::data::corpus::Dataset;
use crate::data::tokenizer::train_tokenizer_from_dataset;

pub fn run_train_tokenizer(command: TrainTokenizerCommand) -> Result<()> {
    let dataset = Dataset::from_path(&command.data)?;
    let summary =
        train_tokenizer_from_dataset(&dataset, command.data.chat_template, &command.tokenizer)?;

    println!(
        "RustGPT train-tokenizer  data={}  format={}  docs={}  model={}  vocab={}  out={}",
        command.data.data_path.display(),
        command.data.format,
        dataset.len(),
        summary.model,
        summary.vocab_size,
        summary.output_path.display()
    );
    println!(
        "bos_token={}  eos_token={}  chat_template={}",
        summary.bos_token, summary.eos_token, command.data.chat_template
    );
    Ok(())
}
