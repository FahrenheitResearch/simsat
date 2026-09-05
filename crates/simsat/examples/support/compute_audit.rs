//! Shared offscreen compute/readback for scientific kernel audits.
use eframe::egui_wgpu::wgpu;
use std::future::Future;
use std::task::{Context, Poll, Waker};
use wgpu::util::DeviceExt;
fn block_on<F: Future>(future: F) -> F::Output {
    let mut f = Box::pin(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        match f.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}
pub struct ComputeAudit {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
}
impl ComputeAudit {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        let info = adapter.get_info();
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("simsat-science-audit"),
            ..Default::default()
        }))?;
        Ok(Self {
            device,
            queue,
            info,
        })
    }
    /// Shader contract: readonly storage at binding 0; one output vec4 at binding 1
    /// per case; @compute @workgroup_size(64) main with a case-count bounds check.
    pub fn run(
        &self,
        source: String,
        input_f32: &[f32],
        cases: usize,
    ) -> Result<Vec<[f32; 4]>, Box<dyn std::error::Error>> {
        let bytes: Vec<u8> = input_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        let output_size = (cases as u64)
            .checked_mul(16)
            .ok_or("output size overflow")?;
        let input = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("science-input"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("science-output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("science-readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("science-kernel"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("science-pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("science-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("science-audit"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((cases as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_size);
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        let values = {
            let data = readback.slice(..).get_mapped_range();
            data.chunks_exact(16)
                .map(|r| {
                    std::array::from_fn(|c| {
                        f32::from_le_bytes(r[c * 4..c * 4 + 4].try_into().unwrap())
                    })
                })
                .collect()
        };
        readback.unmap();
        Ok(values)
    }
}
