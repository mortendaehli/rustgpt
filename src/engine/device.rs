use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, NdArray, Wgpu, ndarray::NdArrayDevice};
use wgpu::{
    AdapterInfo, Backends, DeviceType, Instance, InstanceDescriptor, PowerPreference,
    RequestAdapterOptions,
};

use crate::core::config::DeviceKind;
use crate::core::error::{Result, RustGptError};

pub type CpuBackend = NdArray<f32>;
pub type CpuAutodiffBackend = Autodiff<CpuBackend>;
pub type GpuBackend = Wgpu<f32, i32>;
pub type GpuAutodiffBackend = Autodiff<GpuBackend>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolvedDeviceKind {
    Cpu,
    Gpu,
}

impl ResolvedDeviceKind {
    pub fn resolve(requested: DeviceKind) -> Result<Self> {
        match requested {
            DeviceKind::Cpu => Ok(Self::Cpu),
            DeviceKind::Auto => Ok(if gpu_adapter_info().is_some() {
                Self::Gpu
            } else {
                Self::Cpu
            }),
            DeviceKind::Gpu => {
                if gpu_adapter_info().is_some() {
                    Ok(Self::Gpu)
                } else {
                    Err(RustGptError::Gpu(
                        "wgpu could not find a compatible non-CPU adapter".to_string(),
                    ))
                }
            }
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Cpu => "cpu:ndarray",
            Self::Gpu => "gpu:wgpu",
        }
    }
}

pub fn cpu_device() -> NdArrayDevice {
    NdArrayDevice::Cpu
}

pub fn gpu_device() -> WgpuDevice {
    WgpuDevice::default()
}

pub fn gpu_adapter_info() -> Option<AdapterInfo> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    let info = adapter.get_info();
    (!matches!(info.device_type, DeviceType::Cpu)).then_some(info)
}
