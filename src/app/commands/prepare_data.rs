use crate::app::cli::PrepareDataCommand;
use crate::core::error::Result;
use crate::data::corpus::Dataset;
use crate::data::prepare::write_dataset;

pub fn run_prepare_data(command: PrepareDataCommand) -> Result<()> {
    let dataset = Dataset::from_path(&command.data)?;
    let summary = write_dataset(
        &dataset,
        command.data.chat_template,
        &command.prepare.output_path,
        command.prepare.output_format,
        command.prepare.pretty,
    )?;

    println!(
        "RustGPT prepare-data  in={}  in_format={}  out={}  out_format={}  records={}",
        command.data.data_path.display(),
        command.data.format,
        summary.output_path.display(),
        summary.output_format,
        summary.records
    );
    Ok(())
}
