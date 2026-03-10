//! Device backends.
//! `ComputeBackend` is the small public façade; GPU details stay in the nested module.

mod gpu;

pub use self::gpu::ComputeBackend;

use crate::app::cli::GpuInfoCommand;
use crate::core::error::Result;

pub fn run_gpu_info(command: GpuInfoCommand) -> Result<()> {
    gpu::run_gpu_info(command)
}
