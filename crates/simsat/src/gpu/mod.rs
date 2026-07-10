//! GPU surface pass (design doc section 9: `gpu/` + `gpu/shaders/*.wgsl`).
//!
//! Follows the vol3d wgpu resources/pipeline pattern
//! (`crates/app_ui/src/vol3d.rs::Vol3dResources` / `init_gpu`), PORTED with these
//! deliberate M1 changes:
//!   - SimSat owns its own resources struct ([`SurfaceResources`]);
//!   - the WGSL lives in a separate `include_str!` file (design section 9), so it
//!     is diffable and `naga`-validatable headlessly, not an inline `const`;
//!   - the frame renders OFFSCREEN to an `Rgba8Unorm` target and is read back to
//!     CPU (not painted in an egui callback), so the exact same pixels feed both
//!     the studio's display texture and the sat-store `rgb_r/g/b` planes (display
//!     == stored). Readback is why we diverge from vol3d's in-callback paint.
//!
//! Nodes are headless (no GPU): this module is `naga`-validated
//! ([`tests::every_shader_validates`]) and its shading math is CPU-referenced in
//! [`crate::render`]. The live GPU render is exercised by the owner's exe.

use eframe::egui_wgpu::wgpu;

use crate::bluemarble::BlueMarbleCrop;
use crate::camera::SurfaceRaster;
use crate::render::FLAT_ALBEDO_SRGB;
use crate::solar::SolarFrame;

/// The offscreen render-target format. The shader outputs sRGB-encoded display
/// values directly, so the stored bytes ARE the display values (no sRGB target).
pub const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

const SURFACE_WGSL: &str = include_str!("shaders/surface.wgsl");

/// One rendered surface frame read back to CPU. Row-major RGBA8, row 0 = north.
/// Alpha is the on-earth mask (0 = space) — the store writer turns `a == 0` into a
/// transparent/NaN plane value; the display forces alpha opaque so space is black.
#[derive(Debug, Clone)]
pub struct RenderedFrame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// GPU resources created once (pipeline + layout + sampler). Per-frame textures
/// and bind groups are built in [`SurfaceResources::render`].
pub struct SurfaceResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

/// The packed per-frame uniform (design section 3/6, M2). Mirrors the WGSL
/// `Uniforms` (9 vec4 = 144 bytes). Built on the CPU from the camera geometry,
/// scan grid, sun, atmosphere params, and the output transform.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceUniforms {
    /// Camera ECEF position (m).
    pub cam: [f32; 3],
    /// Bottom-of-atmosphere radius (m).
    pub r_ground: f32,
    /// Unit ECEF sun direction (sun at infinity; constant across the domain).
    pub sun: [f32; 3],
    /// Top-of-atmosphere radius (m).
    pub r_top: f32,
    /// Look basis toward the earth centre (ECEF) + scan `x_min` (rad).
    pub ex: [f32; 3],
    pub x_min: f32,
    /// East basis (ECEF) + scan `y_max` (rad).
    pub ey: [f32; 3],
    pub y_max: f32,
    /// North basis (ECEF) + scan `pitch_x` (rad/px).
    pub ez: [f32; 3],
    pub pitch_x: f32,
    /// Band solar irradiance (W m^-2) + scan `pitch_y` (rad/px).
    pub solar: [f32; 3],
    pub pitch_y: f32,
    /// Mie ground scattering / extinction (m^-1), asymmetry g, PW ratio.
    pub mie_sca: f32,
    pub mie_ext: f32,
    pub mie_g: f32,
    pub pw_ratio: f32,
    /// Blue Marble present (0/1), water albedo scale, flat albedo, output transform.
    pub bm_present: f32,
    pub water_scale: f32,
    pub flat_albedo: f32,
    pub output_transform: f32,
    /// Ambient table elevation range (deg) and entry count.
    pub ambient_elev_min: f32,
    pub ambient_elev_max: f32,
    pub ambient_n: f32,
}

impl SurfaceUniforms {
    /// Pack into the 144-byte std140 uniform buffer the WGSL `Uniforms` expects
    /// (9 vec4, each `xyz` + a `w` scalar).
    pub fn to_bytes(&self) -> [u8; 144] {
        let vec4s: [[f32; 4]; 9] = [
            [self.cam[0], self.cam[1], self.cam[2], self.r_ground],
            [self.sun[0], self.sun[1], self.sun[2], self.r_top],
            [self.ex[0], self.ex[1], self.ex[2], self.x_min],
            [self.ey[0], self.ey[1], self.ey[2], self.y_max],
            [self.ez[0], self.ez[1], self.ez[2], self.pitch_x],
            [self.solar[0], self.solar[1], self.solar[2], self.pitch_y],
            [self.mie_sca, self.mie_ext, self.mie_g, self.pw_ratio],
            [
                self.bm_present,
                self.water_scale,
                self.flat_albedo,
                self.output_transform,
            ],
            [
                self.ambient_elev_min,
                self.ambient_elev_max,
                self.ambient_n,
                0.0,
            ],
        ];
        let mut out = [0u8; 144];
        for (i, v) in vec4s.iter().flatten().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }
}

/// The CPU-side inputs to one surface render.
pub struct SurfaceFrameInputs<'a> {
    /// Output raster width/height (= the scan-angle raster).
    pub width: u32,
    pub height: u32,
    /// Per-pixel `(bm_u, bm_v, grid_u, grid_v)`, `width*height*4` f32.
    pub lut_geo: &'a [f32],
    /// Per-pixel `(sun_e, sun_n, sun_u, sun_elev_deg)`, `width*height*4` f32.
    pub lut_light: &'a [f32],
    /// WRF domain dims (the normal / landmask textures are `nx * ny`).
    pub nx: u32,
    pub ny: u32,
    /// HGT-derived normal map, `nx*ny*4` RGBA8 (`n*0.5+0.5`), WRF row order.
    pub normals_rgba: &'a [u8],
    /// LANDMASK, `nx*ny` R8 (255 = land, 0 = water), WRF row order.
    pub landmask_r8: &'a [u8],
    /// The Blue Marble crop, or `None` to render with the flat albedo.
    pub bluemarble: Option<&'a BlueMarbleCrop>,
    /// Transmittance LUT, `256*64*4` f32 RGBA (optics config).
    pub transmittance_lut: &'a [f32],
    /// Multiple-scattering LUT, `32*32*4` f32 RGBA (optics config).
    pub multiscatter_lut: &'a [f32],
    /// Ambient irradiance vs sun elevation, `ambient_n*4` f32 RGBA (per frame).
    pub ambient_lut: &'a [f32],
    pub ambient_n: u32,
    /// The packed per-frame uniform.
    pub uniforms: SurfaceUniforms,
}

impl SurfaceResources {
    /// One-time GPU setup (call from the app constructor with the wgpu device).
    pub fn init(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("simsat-surface"),
            source: wgpu::ShaderSource::Wgsl(SURFACE_WGSL.into()),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("simsat-surface-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let non_filterable = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let filterable = wgpu::BindGroupLayoutEntry {
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            ..non_filterable
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("simsat-surface-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    ..non_filterable
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    ..non_filterable
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    ..filterable
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    ..filterable
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    ..filterable
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Atmosphere LUTs (Rgba32Float, manual-bilinear via textureLoad).
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    ..non_filterable
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    ..non_filterable
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    ..non_filterable
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("simsat-surface-pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("simsat-surface-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: OFFSCREEN_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            bind_group_layout,
            sampler,
        }
    }

    /// Render one surface frame offscreen and read it back to CPU.
    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &SurfaceFrameInputs,
    ) -> RenderedFrame {
        let (w, h) = (inputs.width.max(1), inputs.height.max(1));

        // The 144-byte packed uniform (design section 3/6).
        let uniform_bytes = inputs.uniforms.to_bytes();
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simsat-surface-uniforms"),
            size: uniform_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniforms, 0, &uniform_bytes);

        // Per-pixel LUT textures (Rgba32Float, textureLoad).
        let lut_geo = self.upload_rgba32f(device, queue, w, h, inputs.lut_geo, "lut-geo");
        let lut_light = self.upload_rgba32f(device, queue, w, h, inputs.lut_light, "lut-light");

        // Atmosphere LUT textures (Rgba32Float, manual-bilinear via textureLoad).
        let transmittance_tex = self.upload_rgba32f(
            device,
            queue,
            256,
            64,
            inputs.transmittance_lut,
            "transmittance",
        );
        let multiscatter_tex = self.upload_rgba32f(
            device,
            queue,
            32,
            32,
            inputs.multiscatter_lut,
            "multiscatter",
        );
        let ambient_tex = self.upload_rgba32f(
            device,
            queue,
            inputs.ambient_n.max(1),
            1,
            inputs.ambient_lut,
            "ambient",
        );

        // Domain textures (normals + landmask), always present.
        let normal_tex = self.upload_rgba8(
            device,
            queue,
            inputs.nx,
            inputs.ny,
            inputs.normals_rgba,
            "normals",
        );
        let landmask_tex = self.upload_r8(
            device,
            queue,
            inputs.nx,
            inputs.ny,
            inputs.landmask_r8,
            "landmask",
        );

        // Blue Marble crop (or a 1x1 gray dummy when absent).
        let bm_tex = match inputs.bluemarble {
            Some(bm) => {
                self.upload_rgba8(device, queue, bm.width, bm.height, &bm.rgba, "bluemarble")
            }
            None => {
                let gray = (FLAT_ALBEDO_SRGB * 255.0) as u8;
                self.upload_rgba8(device, queue, 1, 1, &[gray, gray, gray, 255], "bm-dummy")
            }
        };

        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("simsat-surface-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view(&lut_geo)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view(&lut_light)),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&view(&bm_tex)),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&view(&normal_tex)),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&view(&landmask_tex)),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&view(&transmittance_tex)),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&view(&multiscatter_tex)),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&view(&ambient_tex)),
                },
            ],
        });

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("simsat-surface-target"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OFFSCREEN_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());

        // 256-byte aligned readback buffer.
        let unpadded_bpr = w * 4;
        let padded_bpr = align_up(unpadded_bpr, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simsat-surface-readback"),
            size: (padded_bpr * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("simsat-surface-encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("simsat-surface-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        // Map + block until the GPU is done (headless-safe; run on a worker).
        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let _ = rx.recv();

        let mut rgba = Vec::with_capacity((unpadded_bpr * h) as usize);
        {
            let data = readback.slice(..).get_mapped_range();
            for row in 0..h {
                let start = (row * padded_bpr) as usize;
                let end = start + unpadded_bpr as usize;
                rgba.extend_from_slice(&data[start..end]);
            }
        }
        readback.unmap();

        RenderedFrame {
            width: w,
            height: h,
            rgba,
        }
    }

    fn upload_rgba32f(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        data: &[f32],
        label: &str,
    ) -> wgpu::Texture {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let bytes: &[u8] = bytemuck_f32(data);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 16),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        tex
    }

    fn upload_rgba8(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        data: &[u8],
        label: &str,
    ) -> wgpu::Texture {
        self.upload_bytes(
            device,
            queue,
            w,
            h,
            data,
            wgpu::TextureFormat::Rgba8Unorm,
            4,
            label,
        )
    }

    fn upload_r8(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        data: &[u8],
        label: &str,
    ) -> wgpu::Texture {
        self.upload_bytes(
            device,
            queue,
            w,
            h,
            data,
            wgpu::TextureFormat::R8Unorm,
            1,
            label,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_bytes(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        data: &[u8],
        format: wgpu::TextureFormat,
        bytes_per_texel: u32,
        label: &str,
    ) -> wgpu::Texture {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * bytes_per_texel),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        tex
    }
}

/// Reinterpret an `&[f32]` as `&[u8]` (little-endian target; wgpu uploads LE).
fn bytemuck_f32(data: &[f32]) -> &[u8] {
    // Safety: f32 has no padding/invalid bit patterns; the slice stays borrowed
    // for the call and lengths are exact. wgpu targets are little-endian.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

fn align_up(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

/// Pack RGBA8 normal-map bytes (`n*0.5+0.5`) from ENU normals in WRF row order.
pub fn normals_to_rgba8(normals: &[[f32; 3]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(normals.len() * 4);
    for n in normals {
        out.push(((n[0] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push(((n[1] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push(((n[2] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push(255);
    }
    out
}

/// Pack LANDMASK f32 (`>= 0.5` = land) to R8 (255 = land, 0 = water).
pub fn landmask_to_r8(landmask: &[f32]) -> Vec<u8> {
    landmask
        .iter()
        .map(|&v| if v >= 0.5 { 255u8 } else { 0u8 })
        .collect()
}

/// Build the two per-pixel lookup textures for a frame from the surface raster,
/// the Blue Marble crop bounds (or `None`), the WRF domain dims, and the frame's
/// solar geometry. Returns `(lut_geo, lut_light)`, each `width*height*4` f32.
pub fn build_luts(
    raster: &SurfaceRaster,
    bm: Option<&BlueMarbleCrop>,
    nx: usize,
    ny: usize,
    solar: &SolarFrame,
) -> (Vec<f32>, Vec<f32>) {
    let n = raster.nx * raster.ny;
    let mut geo = vec![0.0f32; n * 4];
    let mut light = vec![0.0f32; n * 4];
    let (nxf, nyf) = (nx as f32, ny as f32);
    for idx in 0..n {
        let lat = raster.lat[idx];
        let lon = raster.lon[idx];
        let g = &mut geo[idx * 4..idx * 4 + 4];
        let l = &mut light[idx * 4..idx * 4 + 4];
        if !lat.is_finite() || !lon.is_finite() {
            // Space: sentinels off (< 0).
            g.copy_from_slice(&[-1.0, -1.0, -1.0, -1.0]);
            l.copy_from_slice(&[0.0, 0.0, 0.0, 0.0]);
            continue;
        }
        // Blue Marble UV (0 when absent; still >= 0 so this is not read as space).
        let (bm_u, bm_v) = match bm {
            Some(bm) if bm.lon_max > bm.lon_min && bm.lat_max > bm.lat_min => (
                ((lon - bm.lon_min) / (bm.lon_max - bm.lon_min)).clamp(0.0, 1.0),
                ((bm.lat_max - lat) / (bm.lat_max - bm.lat_min)).clamp(0.0, 1.0),
            ),
            _ => (0.0, 0.0),
        };
        // Domain UV (or -1 sentinel = outside the WRF domain).
        let fi = raster.grid_i[idx];
        let fj = raster.grid_j[idx];
        let (grid_u, grid_v) = if fi.is_finite() && fj.is_finite() {
            ((fi + 0.5) / nxf, (fj + 0.5) / nyf)
        } else {
            (-1.0, -1.0)
        };
        g.copy_from_slice(&[bm_u, bm_v, grid_u, grid_v]);

        let pos = solar.at(lat as f64, lon as f64);
        let d = pos.enu_direction();
        // w = local sun ELEVATION (deg): the M2 shader uses it for the finite-disk
        // terminator and the ambient-table lookup (M1 stored a 0/1 day flag).
        l.copy_from_slice(&[
            d[0] as f32,
            d[1] as f32,
            d[2] as f32,
            pos.elevation_deg as f32,
        ]);
    }
    (geo, light)
}

// ── M4 cloud GPU volume path (design section 2/4; vol3d 3-D upload pattern) ────

/// The cloud raymarch fragment shader (a superset of the surface pass). The GPU
/// render twin of `clouds.rs`; naga-validated headlessly, activated in M5.
pub const CLOUDS_WGSL: &str = include_str!("shaders/clouds.wgsl");
/// The sun optical-depth compute shader (the GPU twin of `clouds::accumulate_sun_od`).
pub const SUN_OD_WGSL: &str = include_str!("shaders/sun_od.wgsl");

/// GPU 3-D cloud volume resources (design section 2, "brick-to-GPU volume path"; the
/// `vol3d` 3-D texture upload pattern). Holds Texture A (the log-quantized extinction
/// volume ext_liquid/ext_ice/ext_precip/tau_up, `Rgba8Unorm` 3-D) and the occupancy
/// mip (`R8Unorm` 3-D), uploaded via the `write_texture` path. The per-volume
/// `LogQuant` scales are decoded in-shader from uniforms (no re-quantization).
///
/// M4 renders clouds on the CPU (`clouds::render_cloud_frame_rgba`, tested on the
/// headless nodes for correctness); these GPU resources + [`CLOUDS_WGSL`] /
/// [`SUN_OD_WGSL`] are the naga-validated interactive-GPU path activated in M5. They
/// live here so the 3-D volume upload is a real, public engine resource today.
/// Respects the wgpu 3-D limit (brick `<= 2048` per axis, already guaranteed at
/// ingest) and uses `rgba8`/`r8` only as storage formats (design section 8, f32 ALU).
pub struct CloudVolumeResources {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub occ_dims: (u32, u32, u32),
    volume_tex: wgpu::Texture,
    occupancy_tex: wgpu::Texture,
}

impl CloudVolumeResources {
    /// Create the 3-D textures for a brick of `nx*ny*nz` and its occupancy mip.
    pub fn new(
        device: &wgpu::Device,
        nx: u32,
        ny: u32,
        nz: u32,
        occ_dims: (u32, u32, u32),
    ) -> Self {
        let volume_tex = Self::create_3d(
            device,
            nx,
            ny,
            nz,
            wgpu::TextureFormat::Rgba8Unorm,
            "simsat-cloud-volume",
        );
        let occupancy_tex = Self::create_3d(
            device,
            occ_dims.0,
            occ_dims.1,
            occ_dims.2,
            wgpu::TextureFormat::R8Unorm,
            "simsat-cloud-occupancy",
        );
        Self {
            nx,
            ny,
            nz,
            occ_dims,
            volume_tex,
            occupancy_tex,
        }
    }

    fn create_3d(
        device: &wgpu::Device,
        w: u32,
        h: u32,
        d: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: d.max(1),
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    /// Upload Texture A (interleaved `Rgba8Unorm`, `clouds::pack_texture_a`) and the
    /// occupancy mip (`R8Unorm`, `OccupancyMip::to_r8_occupancy`).
    pub fn upload(&self, queue: &wgpu::Queue, texture_a: &[u8], occupancy: &[u8]) {
        Self::write_3d(
            queue,
            &self.volume_tex,
            self.nx,
            self.ny,
            self.nz,
            4,
            texture_a,
        );
        Self::write_3d(
            queue,
            &self.occupancy_tex,
            self.occ_dims.0,
            self.occ_dims.1,
            self.occ_dims.2,
            1,
            occupancy,
        );
    }

    fn write_3d(
        queue: &wgpu::Queue,
        tex: &wgpu::Texture,
        w: u32,
        h: u32,
        d: u32,
        bytes_per_texel: u32,
        data: &[u8],
    ) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w.max(1) * bytes_per_texel),
                rows_per_image: Some(h.max(1)),
            },
            wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: d.max(1),
            },
        );
    }

    /// A view of the extinction volume (Texture A) for a cloud-pass bind group.
    pub fn volume_view(&self) -> wgpu::TextureView {
        self.volume_tex.create_view(&Default::default())
    }

    /// A view of the occupancy mip for a cloud-pass bind group.
    pub fn occupancy_view(&self) -> wgpu::TextureView {
        self.occupancy_tex.create_view(&Default::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{GeoCamera, SatellitePreset, build_surface_raster};
    use crate::frame::{GridGeoref, MapProjection};

    #[test]
    fn cloud_shaders_validate() {
        // The M4 cloud raymarch pass + the sun-OD compute pass both naga-validate
        // (design section 9, test strategy 3: catch shader breakage headlessly).
        for (name, src) in [("clouds.wgsl", CLOUDS_WGSL), ("sun_od.wgsl", SUN_OD_WGSL)] {
            let module =
                naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("parse {name}: {e}"));
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            validator
                .validate(&module)
                .unwrap_or_else(|e| panic!("validate {name}: {e:?}"));
        }
    }

    #[test]
    fn every_shader_validates() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/shaders");
        let mut count = 0;
        for entry in std::fs::read_dir(&dir).expect("shaders dir") {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("wgsl") {
                continue;
            }
            let src = std::fs::read_to_string(&path).unwrap();
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            validator
                .validate(&module)
                .unwrap_or_else(|e| panic!("validate {}: {e:?}", path.display()));
            count += 1;
        }
        assert!(
            count >= 1,
            "expected at least one wgsl shader, found {count}"
        );
    }

    #[test]
    fn surface_wgsl_is_present_and_validates() {
        let module = naga::front::wgsl::parse_str(SURFACE_WGSL).expect("surface.wgsl parses");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&module).expect("surface.wgsl validates");
    }

    #[test]
    fn build_luts_marks_space_and_domain() {
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let (nx, ny) = (60usize, 45usize);
        let georef = GridGeoref::new(proj, 29.5, 22.0, 39.0, -97.5, 4000.0, 4000.0);
        let camera = GeoCamera::new(SatellitePreset::GoesEast);
        let raster = build_surface_raster(
            &camera,
            &georef,
            nx,
            ny,
            crate::camera::VISIBLE_PITCH_RAD,
            256,
        )
        .unwrap();
        let solar = SolarFrame::new(2025, 6, 21, 18.0);
        let (geo, light) = build_luts(&raster, None, nx, ny, &solar);
        assert_eq!(geo.len(), raster.nx * raster.ny * 4);
        assert_eq!(light.len(), geo.len());
        // Every pixel is either space (all sentinels < 0) or on-earth (bm_u >= 0).
        let mut space = 0;
        let mut earth = 0;
        for idx in 0..raster.nx * raster.ny {
            if geo[idx * 4] < 0.0 {
                space += 1;
            } else {
                earth += 1;
                // w carries the local sun elevation (deg) in a valid range.
                let f = light[idx * 4 + 3];
                assert!(f.is_finite() && (-90.5..=90.5).contains(&f));
            }
        }
        assert!(earth > 0, "domain should be on earth");
        let _ = space;
    }

    #[test]
    fn packers_encode_expected_bytes() {
        let up = normals_to_rgba8(&[[0.0, 0.0, 1.0]]);
        assert_eq!(up, vec![128, 128, 255, 255]);
        let lm = landmask_to_r8(&[1.0, 0.0, 0.7, 0.2]);
        assert_eq!(lm, vec![255, 0, 255, 0]);
    }

    #[test]
    fn align_up_rounds_to_256() {
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(800 * 4, 256), 3328);
    }

    #[test]
    fn surface_uniforms_pack_144_bytes_in_order() {
        let u = SurfaceUniforms {
            cam: [1.0, 2.0, 3.0],
            r_ground: 6_370_000.0,
            sun: [0.0, 0.0, 1.0],
            r_top: 6_470_000.0,
            ex: [-1.0, 0.0, 0.0],
            x_min: -0.01,
            ey: [0.0, 1.0, 0.0],
            y_max: 0.02,
            ez: [0.0, 0.0, 1.0],
            pitch_x: 2.8e-5,
            solar: [180.0, 188.0, 196.0],
            pitch_y: 2.8e-5,
            mie_sca: 7.5e-5,
            mie_ext: 8.3e-5,
            mie_g: 0.8,
            pw_ratio: 1.0,
            bm_present: 1.0,
            water_scale: 0.55,
            flat_albedo: 0.3,
            output_transform: 0.0,
            ambient_elev_min: -20.0,
            ambient_elev_max: 90.0,
            ambient_n: 48.0,
        };
        let bytes = u.to_bytes();
        assert_eq!(bytes.len(), 144);
        // Spot-check the layout: cam.x at [0], r_ground at [3*4], ambient_n at [34*4].
        assert_eq!(f32::from_le_bytes(bytes[0..4].try_into().unwrap()), 1.0);
        assert_eq!(
            f32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            6_370_000.0
        );
        assert_eq!(
            f32::from_le_bytes(bytes[136..140].try_into().unwrap()),
            48.0
        );
        // Trailing pad word is zero.
        assert_eq!(f32::from_le_bytes(bytes[140..144].try_into().unwrap()), 0.0);
    }
}
