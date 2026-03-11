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
        &command.prepare,
    )?;

    println!(
        "RustGPT prepare-data  in={}  in_format={}  out={}  out_format={}  input_records={}  written_records={}  deduped={}  quality_filtered={}",
        command.data.data_path.display(),
        command.data.format,
        summary.output_path.display(),
        summary.output_format,
        summary.input_records,
        summary.records,
        summary.duplicate_records_removed,
        summary.quality_filtered_records,
    );
    Ok(())
}
