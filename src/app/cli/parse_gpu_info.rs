use crate::app::cli::help::gpu_info_help;
use crate::app::cli::{Command, GpuInfoCommand};
use crate::core::config::GpuInfoConfig;
use crate::core::error::{Result, RustGptError};

use super::parse_shared::parse_device_kind;

pub(super) fn parse_gpu_info(bin: &str, args: Vec<String>) -> Result<Command> {
    let mut gpu = GpuInfoConfig::default();

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => return Ok(Command::Help(gpu_info_help(bin))),
            "--device" => {
                idx += 1;
                gpu.device = parse_device_kind(&args, idx, "--device")?;
            }
            other => {
                return Err(RustGptError::Cli(format!(
                    "unknown gpu-info argument {other:?}\n\n{}",
                    gpu_info_help(bin)
                )));
            }
        }
        idx += 1;
    }

    Ok(Command::GpuInfo(GpuInfoCommand { gpu }))
}
