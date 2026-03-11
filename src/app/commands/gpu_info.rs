use crate::app::cli::GpuInfoCommand;
use crate::core::config::DeviceKind;
use crate::core::error::Result;
use crate::runtime::device::gpu_adapter_info;

pub fn run_gpu_info(command: GpuInfoCommand) -> Result<()> {
    println!("RustGPT gpu-info");
    println!("requested_device={}", command.gpu.device);
    println!("burn_gpu_backend=wgpu");

    match command.gpu.device {
        DeviceKind::Cpu => {
            println!("status=cpu-requested");
        }
        DeviceKind::Auto | DeviceKind::Gpu => {
            if let Some(info) = gpu_adapter_info() {
                println!("status=ok");
                println!("adapter_name={}", info.name);
                println!("backend={:?}", info.backend);
                println!("device_type={:?}", info.device_type);
                println!("driver={}", info.driver);
                println!("driver_info={}", info.driver_info);
            } else {
                println!("status=no-compatible-gpu-adapter");
            }
        }
    }

    Ok(())
}
