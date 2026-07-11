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
use crate::bricks::{LogQuant, VolumeBrick};
use crate::camera::SurfaceRaster;
use crate::frame::{GridGeoref, MapProjection};
use crate::render::{
    FLAT_ALBEDO_SRGB, LAND_DARK_TOE_GAMMA, LAND_DARK_TOE_KNEE, LAND_DARK_TOE_MAX_GAIN,
    LAND_SZA_MAX_GAIN, LandAppearanceConfig,
};
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

/// A conservative quantized occupancy upload for the GPU cloud volume.
///
/// `r8` is byte-identical to
/// `OccupancyMip::build(DecodedVolume::from_brick_legacy(..)).to_r8_occupancy()`:
/// each source code is decoded through the brick's exact [`LogQuant`], the three
/// decoded f32 channels are summed with the reference's f64 arithmetic, and a
/// one-block 26-neighbourhood dilation is applied. Building only this binary field
/// avoids materializing four full f32 volumes merely to prepare a raw-u8 GPU upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedOccupancyUpload {
    pub dims: (u32, u32, u32),
    pub r8: Vec<u8>,
}

/// Build the GPU occupancy mip directly from a brick's quantized extinction codes.
///
/// This deliberately reproduces the decoded reference's edge semantics, including
/// degenerate/non-finite quantizers: a voxel is occupied only when the exact decoded
/// f32 channel sum is greater than zero. It is therefore safe to use for empty-space
/// skipping without constructing a [`crate::clouds::DecodedVolume`].
pub fn quantized_occupancy_upload(brick: &VolumeBrick, factor: usize) -> QuantizedOccupancyUpload {
    let factor = factor.max(1);
    let cells = brick
        .nx
        .checked_mul(brick.ny)
        .and_then(|n| n.checked_mul(brick.nz))
        .expect("cloud-volume dimensions overflow usize");
    assert_eq!(brick.ext_liquid.len(), cells, "ext_liquid length");
    assert_eq!(brick.ext_ice.len(), cells, "ext_ice length");
    assert_eq!(brick.ext_precip.len(), cells, "ext_precip length");

    let decode_lut =
        |quant: LogQuant| -> [f32; 256] { core::array::from_fn(|code| quant.decode(code as u8)) };
    let liquid_lut = decode_lut(brick.quant.get("ext_liquid"));
    let ice_lut = decode_lut(brick.quant.get("ext_ice"));
    let precip_lut = decode_lut(brick.quant.get("ext_precip"));

    let mx = brick.nx.div_ceil(factor);
    let my = brick.ny.div_ceil(factor);
    let mz = brick.nz.div_ceil(factor);
    let mut raw = vec![0u8; mx * my * mz];
    for k in 0..brick.nz {
        let kb = k / factor;
        for j in 0..brick.ny {
            let jb = j / factor;
            let row = (k * brick.ny + j) * brick.nx;
            for i in 0..brick.nx {
                let cell = row + i;
                // Match DecodedVolume::total_ext_cell exactly: each stored decoded
                // f32 is widened to f64, summed left-to-right, then narrowed to f32
                // by OccupancyMip::build before its `>` comparison.
                let ext = (liquid_lut[brick.ext_liquid[cell] as usize] as f64
                    + ice_lut[brick.ext_ice[cell] as usize] as f64
                    + precip_lut[brick.ext_precip[cell] as usize] as f64)
                    as f32;
                if ext > 0.0 {
                    raw[(kb * my + jb) * mx + i / factor] = 255;
                }
            }
        }
    }

    // Twin of OccupancyMip::build's one-block dilation. The upload is binary, so
    // the reference max over the 26-neighbourhood reduces to an any-occupied test.
    let mut r8 = vec![0u8; raw.len()];
    for kb in 0..mz {
        for jb in 0..my {
            for ib in 0..mx {
                'neighbours: for dk in -1i64..=1 {
                    let nk = kb as i64 + dk;
                    if nk < 0 || nk as usize >= mz {
                        continue;
                    }
                    for dj in -1i64..=1 {
                        let nj = jb as i64 + dj;
                        if nj < 0 || nj as usize >= my {
                            continue;
                        }
                        for di in -1i64..=1 {
                            let ni = ib as i64 + di;
                            if ni < 0 || ni as usize >= mx {
                                continue;
                            }
                            if raw[(nk as usize * my + nj as usize) * mx + ni as usize] != 0 {
                                r8[(kb * my + jb) * mx + ib] = 255;
                                break 'neighbours;
                            }
                        }
                    }
                }
            }
        }
    }

    QuantizedOccupancyUpload {
        dims: (
            u32::try_from(mx).expect("occupancy x dimension exceeds u32"),
            u32::try_from(my).expect("occupancy y dimension exceeds u32"),
            u32::try_from(mz).expect("occupancy z dimension exceeds u32"),
        ),
        r8,
    }
}

/// GPU resources created once (pipeline + layout + sampler). Per-frame textures
/// and bind groups are built in [`SurfaceResources::render`].
pub struct SurfaceResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

/// The packed per-frame uniform (design section 3/6, M2). Mirrors the WGSL
/// `Uniforms` (11 vec4 = 176 bytes). Built on the CPU from the camera geometry,
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
    /// Product-facing atmosphere correction flag (0 = raw physical airlight, 1 =
    /// corrected true-color veil). Packed in the formerly-unused `p2.w` lane.
    pub atmosphere_correction: f32,
    /// Finished-visible land appearance controls shared by both visible GPU paths.
    pub land_appearance: LandAppearanceConfig,
}

/// Sanitize and pack the two land-appearance quads shared by both visible WGSL paths.
///
/// The CPU reference performs the same finite checks and clamps at evaluation time.
/// Doing them before the f64 -> f32 conversion prevents non-finite shader inputs while
/// preserving the exact switches and every value representable by Studio's f32 controls.
pub fn land_appearance_uniform_quads(config: LandAppearanceConfig) -> [[f32; 4]; 2] {
    // Deliberately exhaustive: adding another CPU land operator must break this GPU
    // packer at compile time until the shader ABI and eligibility review are updated.
    let LandAppearanceConfig {
        sza_normalization,
        sza_max_gain,
        dark_toe,
        dark_toe_knee,
        dark_toe_gamma,
        dark_toe_max_gain,
    } = config;
    let finite_clamped = |value: f64, fallback: f64, lo: f64, hi: f64| {
        (if value.is_finite() {
            value.clamp(lo, hi)
        } else {
            fallback
        }) as f32
    };
    let sza_max_gain = finite_clamped(sza_max_gain, LAND_SZA_MAX_GAIN, 1.0, 4.0);
    let dark_toe_knee = finite_clamped(dark_toe_knee, LAND_DARK_TOE_KNEE, 1.0e-6, 1.0);
    let dark_toe_gamma = finite_clamped(dark_toe_gamma, LAND_DARK_TOE_GAMMA, 0.05, 1.0);
    let dark_toe_max_gain = finite_clamped(dark_toe_max_gain, LAND_DARK_TOE_MAX_GAIN, 1.0, 4.0);
    [
        [
            if sza_normalization { 1.0 } else { 0.0 },
            sza_max_gain,
            if dark_toe { 1.0 } else { 0.0 },
            dark_toe_knee,
        ],
        [dark_toe_gamma, dark_toe_max_gain, 0.0, 0.0],
    ]
}

impl SurfaceUniforms {
    /// The 11 packed vec4s of the WGSL surface `Uniforms`. The first nine retain the
    /// historical surface/cloud ABI; the two land-control quads are shared explicitly.
    pub fn to_vec4s(&self) -> [[f32; 4]; 11] {
        let land = land_appearance_uniform_quads(self.land_appearance);
        [
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
                self.atmosphere_correction,
            ],
            land[0],
            land[1],
        ]
    }

    /// Pack into the 176-byte uniform buffer the WGSL `Uniforms` expects (11 vec4).
    pub fn to_bytes(&self) -> [u8; 176] {
        let vec4s = self.to_vec4s();
        let mut out = [0u8; 176];
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

        // The 176-byte packed uniform (surface physics + land appearance controls).
        let uniform_bytes = inputs.uniforms.to_bytes();
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simsat-surface-uniforms"),
            size: uniform_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniforms, 0, &uniform_bytes);

        // Per-pixel LUT textures (Rgba32Float, textureLoad).
        let lut_geo = upload_rgba32f(device, queue, w, h, inputs.lut_geo, "lut-geo");
        let lut_light = upload_rgba32f(device, queue, w, h, inputs.lut_light, "lut-light");

        // Atmosphere LUT textures (Rgba32Float, manual-bilinear via textureLoad).
        let transmittance_tex = upload_rgba32f(
            device,
            queue,
            256,
            64,
            inputs.transmittance_lut,
            "transmittance",
        );
        let multiscatter_tex = upload_rgba32f(
            device,
            queue,
            32,
            32,
            inputs.multiscatter_lut,
            "multiscatter",
        );
        let ambient_tex = upload_rgba32f(
            device,
            queue,
            inputs.ambient_n.max(1),
            1,
            inputs.ambient_lut,
            "ambient",
        );

        // Domain textures (normals + landmask), always present.
        let normal_tex = upload_rgba8(
            device,
            queue,
            inputs.nx,
            inputs.ny,
            inputs.normals_rgba,
            "normals",
        );
        let landmask_tex = upload_r8(
            device,
            queue,
            inputs.nx,
            inputs.ny,
            inputs.landmask_r8,
            "landmask",
        );

        // Blue Marble crop (or a 1x1 gray dummy when absent).
        let bm_tex = match inputs.bluemarble {
            Some(bm) => upload_rgba8(device, queue, bm.width, bm.height, &bm.rgba, "bluemarble"),
            None => {
                let gray = (FLAT_ALBEDO_SRGB * 255.0) as u8;
                upload_rgba8(device, queue, 1, 1, &[gray, gray, gray, 255], "bm-dummy")
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
}

fn upload_rgba32f(
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
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    w: u32,
    h: u32,
    data: &[u8],
    label: &str,
) -> wgpu::Texture {
    upload_bytes(
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
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    w: u32,
    h: u32,
    data: &[u8],
    label: &str,
) -> wgpu::Texture {
    upload_bytes(
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

// ── GPU cloud-pass ACTIVATION (feat/gpu-clouds) ────────────────────────────────
//
// The long-deferred M5-GPU activation: [`CloudPassResources`] dispatches the
// naga-validated `sun_od.wgsl` compute over the uploaded [`CloudVolumeResources`]
// volume and then runs the `clouds.wgsl` full-screen cloud-march pass offscreen with
// readback — the SAME output contract as the CPU composite's RGBA. The CPU path
// (`clouds::render_cloud_frame_rgba`) remains the shipping default, the stored-frame
// path, and the parity ground truth; this is the EXPERIMENTAL interactive preview
// behind the studio's "GPU clouds" toggle, validated by the owner's A/B + the
// in-studio parity instrument (the nodes are headless — here we can only compile,
// naga-validate, and unit-test the pure packing/planning math below).
//
// TDR SAFETY (the M8 concern, minimally): one submit is never unbounded. The sun-OD
// compute is dispatched in ROW BANDS of at most [`SUN_OD_MAX_SAMPLES_PER_SUBMIT`]
// texel-samples (the shader reads its band's row offset from `vert.w`), and the cloud
// fragment pass renders in SCISSOR BANDS of at most [`CLOUD_TILE_MAX_PIXELS`] pixels,
// each band its own command submission, so a large raster cannot hang the device
// past a watchdog on an iGPU.

/// Cap on `dim * n_steps * rows` texel-samples per sun-OD compute submit (~32M).
pub const SUN_OD_MAX_SAMPLES_PER_SUBMIT: u64 = 32 * 1024 * 1024;
/// Cap on pixels marched per cloud-pass submit (one scissor band), 1M.
pub const CLOUD_TILE_MAX_PIXELS: u64 = 1 << 20;

/// Split `total` rows into `(offset, count)` bands of at most `step` rows.
/// Covers every row exactly once, in order; `step` is clamped to >= 1.
pub fn row_bands(total: u32, step: u32) -> Vec<(u32, u32)> {
    let step = step.max(1);
    let mut out = Vec::new();
    let mut y = 0u32;
    while y < total {
        let rows = step.min(total - y);
        out.push((y, rows));
        y += rows;
    }
    out
}

/// Rows per sun-OD compute band for a `dim`-square map marching `n_steps` samples per
/// texel: bounded by [`SUN_OD_MAX_SAMPLES_PER_SUBMIT`], floored at one workgroup (8
/// rows), rounded down to the 8-row workgroup height so bands never overlap.
pub fn sun_od_band_rows(dim: u32, n_steps: u32) -> u32 {
    let per_row = (dim.max(1) as u64) * (n_steps.max(1) as u64);
    let rows = (SUN_OD_MAX_SAMPLES_PER_SUBMIT / per_row.max(1)).clamp(8, dim.max(8) as u64) as u32;
    (rows / 8).max(1) * 8
}

/// Rows per cloud-pass scissor band for a raster `width` px wide: bounded by
/// [`CLOUD_TILE_MAX_PIXELS`] pixels per submit, floored at one row.
pub fn cloud_tile_rows(width: u32) -> u32 {
    (CLOUD_TILE_MAX_PIXELS / width.max(1) as u64).max(1) as u32
}

// Small [f64; 3] helpers (twins of the private clouds.rs ones — clouds.rs is owned by
// a parallel workstream, so the GPU planner carries its own copies).
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm3(a: [f64; 3]) -> [f64; 3] {
    let l = dot3(a, a).sqrt();
    if l > 0.0 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        a
    }
}

/// The four `geo0..geo3` uniform quads of the WGSL WRF projection forward
/// (`clouds.wgsl` / `sun_od.wgsl` `project` + `ecef_to_brick`). `GridGeoref`'s anchor
/// fields are private, so the anchor is RE-DERIVED at grid index (0, 0) from the pub
/// surface: `plane_uv(0,0)` gives the plane coords of cell (0,0) and the unit steps
/// give `dx`/`dy` — an equivalent anchoring (`fi = (u - u00) / dx`), tested against
/// `georef.forward` below.
#[derive(Debug, Clone, Copy)]
pub struct GeoQuads {
    pub geo0: [f32; 4],
    pub geo1: [f32; 4],
    pub geo2: [f32; 4],
    pub geo3: [f32; 4],
}

/// Build [`GeoQuads`] from a georef (see the struct doc). The WGSL projection kinds:
/// 0 = Lambert, 1 = polar stereographic, 2 = Mercator, 3 = geographic lat/lon.
pub fn geo_quads(georef: &GridGeoref) -> GeoQuads {
    let (u00, v00) = georef.plane_uv(0.0, 0.0);
    let (u10, _) = georef.plane_uv(1.0, 0.0);
    let (_, v01) = georef.plane_uv(0.0, 1.0);
    let dx = u10 - u00;
    let dy = v01 - v00;
    let (kind, cm, lambert_n, lambert_f, ps_k, merc_scale, south) = match georef.projection() {
        MapProjection::Lambert {
            n,
            f,
            stand_lon_deg,
        } => (0.0, stand_lon_deg, n, f, 0.0, 0.0, 0.0),
        MapProjection::PolarStereographic {
            k,
            central_meridian_deg,
            south_pole,
        } => (
            1.0,
            central_meridian_deg,
            0.0,
            0.0,
            k,
            0.0,
            if south_pole { 1.0 } else { 0.0 },
        ),
        MapProjection::Mercator {
            scale,
            central_meridian_deg,
        } => (2.0, central_meridian_deg, 0.0, 0.0, 0.0, scale, 0.0),
        MapProjection::LatLon {
            central_meridian_deg,
        } => (3.0, central_meridian_deg, 0.0, 0.0, 0.0, 0.0, 0.0),
        // The rotated lat-lon grid (GRIB RRFS) has NO WGSL forward — the GPU cloud
        // pass cannot march such a brick. Kind -1 is an explicit unsupported
        // sentinel; callers must gate on [`projection_supported`] and stay on the
        // CPU path (the studio does).
        MapProjection::RotatedLatLon { .. } => (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    };
    GeoQuads {
        geo0: [kind, 0.0, 0.0, dx as f32],
        geo1: [u00 as f32, v00 as f32, dy as f32, cm as f32],
        geo2: [
            lambert_n as f32,
            lambert_f as f32,
            ps_k as f32,
            merc_scale as f32,
        ],
        geo3: [south, 0.0, 0.0, 0.0],
    }
}

/// Whether the WGSL projection forward implements this georef's projection — the
/// GPU cloud/sun-OD passes only speak Lambert / polar stereographic / Mercator /
/// geographic lat/lon. A rotated lat-lon (GRIB RRFS) brick must render on the CPU
/// path; [`geo_quads`] packs the explicit -1 unsupported sentinel for it.
pub fn projection_supported(georef: &GridGeoref) -> bool {
    !matches!(georef.projection(), MapProjection::RotatedLatLon { .. })
}

/// The sun-OD dispatch plan: the sun-aligned orthographic frame + extents + step
/// schedule, mirroring the geometry half of `clouds::accumulate_sun_od_granulated`
/// (whose `SunOdMap` frame fields are private — the GPU pipeline computes its own,
/// self-consistent frame, and the SAME values feed both `sun_od.wgsl` and the
/// `clouds.wgsl` `sod_*` sampler uniforms, so producer and consumer always agree).
#[derive(Debug, Clone, Copy)]
pub struct SunOdPlan {
    pub center: [f64; 3],
    pub au: [f64; 3],
    pub av: [f64; 3],
    pub sun: [f64; 3],
    pub u_min: f64,
    pub u_max: f64,
    pub v_min: f64,
    pub v_max: f64,
    pub s_start: f64,
    pub s_len: f64,
    /// Along-sun samples per texel (0 for a degenerate plan -> an all-zero map).
    pub n_steps: usize,
    pub ds: f64,
    /// Square map side (texels).
    pub dim: usize,
}

/// Plan the sun-OD map for a brick + sun direction (the CPU-geometry mirror; see
/// [`SunOdPlan`]). `resolution` is the square map side (the studio uses 512).
#[allow(clippy::too_many_arguments)]
pub fn plan_sun_od(
    georef: &GridGeoref,
    nx: usize,
    ny: usize,
    nz: usize,
    z_min_m: f64,
    dz_m: f64,
    voxel_pitch_m: f64,
    sun_ecef: [f64; 3],
    resolution: usize,
) -> SunOdPlan {
    use crate::atmosphere::R_GROUND_M;
    let resolution = resolution.max(1);
    let sun = norm3(sun_ecef);
    // perp_basis twin (clouds.rs private): a right-handed basis perpendicular to sun.
    let seed = if sun[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let au = norm3(cross3(seed, sun));
    let av = cross3(sun, au);
    let ci = (nx.saturating_sub(1)) as f64 / 2.0;
    let cj = (ny.saturating_sub(1)) as f64 / 2.0;
    let ck = (nz.saturating_sub(1)) as f64 / 2.0;
    let center = crate::clouds::brick_to_ecef(georef, ci, cj, ck, z_min_m, dz_m)
        .unwrap_or([R_GROUND_M, 0.0, 0.0]);

    let (mut u_min, mut u_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut v_min, mut v_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut s_min, mut s_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &ki in &[0.0, (nz.saturating_sub(1)) as f64] {
        for &ji in &[0.0, (ny.saturating_sub(1)) as f64] {
            for &ii in &[0.0, (nx.saturating_sub(1)) as f64] {
                if let Some(p) = crate::clouds::brick_to_ecef(georef, ii, ji, ki, z_min_m, dz_m) {
                    let d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
                    let (u, v, s) = (dot3(d, au), dot3(d, av), dot3(d, sun));
                    u_min = u_min.min(u);
                    u_max = u_max.max(u);
                    v_min = v_min.min(v);
                    v_max = v_max.max(v);
                    s_min = s_min.min(s);
                    s_max = s_max.max(s);
                }
            }
        }
    }
    if !(u_min.is_finite() && v_min.is_finite() && s_min.is_finite()) {
        // Degenerate (projection failed at every corner): an all-zero map, the same
        // contract as the CPU's degenerate SunOdMap.
        return SunOdPlan {
            center,
            au,
            av,
            sun,
            u_min: -1.0,
            u_max: 1.0,
            v_min: -1.0,
            v_max: 1.0,
            s_start: 0.0,
            s_len: 0.0,
            n_steps: 0,
            ds: 1.0,
            dim: resolution,
        };
    }
    let pitch = voxel_pitch_m.max(1.0);
    let margin = pitch * 4.0;
    let s_start = s_max + margin;
    let s_len = (s_max - s_min) + 2.0 * margin;
    let n_steps = ((s_len / pitch).ceil() as usize).clamp(1, 1024);
    let ds = s_len / n_steps as f64;
    SunOdPlan {
        center,
        au,
        av,
        sun,
        u_min,
        u_max,
        v_min,
        v_max,
        s_start,
        s_len,
        n_steps,
        ds,
        dim: resolution,
    }
}

/// The march/appearance parameters of one GPU cloud render (the `MarchConfig` slice
/// the shader consumes; the GPU path always uses the INTERACTIVE schedule — see the
/// `clouds.wgsl` schedule note).
#[derive(Debug, Clone, Copy)]
pub struct CloudMarchParams {
    pub coarse_step_m: f32,
    pub fine_step_m: f32,
    pub max_steps: f32,
    /// Display exposure gain (`u.m1.x`; the CPU `radiance_to_rgba_softclip` seam).
    pub exposure: f32,
    /// Multi-scatter octave count (`u.m1.y`; 6 or 1 from the studio A/B).
    pub octaves: f32,
    pub beer_powder: bool,
    pub ground_albedo: f32,
    pub transmittance_floor: f32,
    /// Visible-cloud optical-depth QA/calibration multiplier. Packed validated in
    /// `u.frx2.w`; raw volume/COD and the thermal path remain unscaled.
    pub cloud_optical_depth_scale: f32,
    /// Zoom-out-margin edge-feather band (cells; 0 = no margin, the no-op).
    pub edge_feather_cells: f32,
    /// The sun-gated whole-surface daytime lift (`render::GROUND_DAY_LIFT` default).
    pub ground_day_lift: f32,
}

/// The CPU-side inputs of one GPU cloud-composited render: the surface-pass inputs
/// (raster LUTs, ground/domain textures, atmosphere LUTs, packed surface uniforms)
/// plus the cloud volume, quant scales, projection quads, march params, sun-OD plan,
/// froxel and SH-ambient uploads.
pub struct CloudFrameInputs<'a> {
    pub surface: SurfaceFrameInputs<'a>,
    /// Brick dims + Texture A (`clouds::pack_texture_a`) + occupancy mip.
    pub vol_nx: u32,
    pub vol_ny: u32,
    pub vol_nz: u32,
    pub texture_a: &'a [u8],
    pub occ_dims: (u32, u32, u32),
    pub occupancy: &'a [u8],
    /// LogQuant scales: `ext_liquid vmin,vmax ; ext_ice vmin,vmax`.
    pub ql: [f32; 4],
    /// LogQuant scales: `ext_precip vmin,vmax ; tau_up vmin,vmax`.
    pub qp: [f32; 4],
    /// Brick vertical extent + shell radii (m).
    pub z_min_m: f32,
    pub dz_m: f32,
    pub r_top_m: f32,
    pub r_bottom_m: f32,
    pub voxel_pitch_m: f32,
    pub geo: GeoQuads,
    pub march: CloudMarchParams,
    pub sun_od: SunOdPlan,
    /// Aerial-perspective froxel (`atmosphere::AerialFroxel`): `dim^3 * 4` f32 RGBA,
    /// index `4*((z*dim + y)*dim + x)` — uploaded as a `dim^3` Rgba32Float 3-D texture.
    pub froxel_dim: u32,
    pub froxel_data: &'a [f32],
    /// SH-2 sky-ambient table (`SkyShTable::to_rgba_f32`): 9 coef columns x `sh_rows`
    /// elevation rows of RGBA f32 (binding 14). Row count must equal the scalar
    /// ambient LUT's entry count (both come from the same table).
    pub sh_rows: u32,
    pub sh_data: &'a [f32],
    /// The scan-angle rect the froxel was built over (`x_min, x_max, y_min, y_max`).
    pub scan_rect: [f32; 4],
}

/// The 27 packed vec4s of the WGSL cloud `Uniforms` (the historical surface 9, the
/// cloud 16, then the shared land-appearance 2), in declaration order. Kept as a
/// pure function so the layout is unit-testable on the headless nodes.
pub fn cloud_uniform_quads(inputs: &CloudFrameInputs) -> [[f32; 4]; 27] {
    let s = inputs.surface.uniforms.to_vec4s();
    let m = &inputs.march;
    let so = &inputs.sun_od;
    let f3 = |v: [f64; 3]| [v[0] as f32, v[1] as f32, v[2] as f32];
    let (c, au, av) = (f3(so.center), f3(so.au), f3(so.av));
    [
        s[0],
        s[1],
        s[2],
        s[3],
        s[4],
        s[5],
        s[6],
        s[7],
        s[8],
        // dims: nx, ny, nz, voxel_pitch
        [
            inputs.vol_nx as f32,
            inputs.vol_ny as f32,
            inputs.vol_nz as f32,
            inputs.voxel_pitch_m,
        ],
        // vert: z_min, dz, r_top(brick), r_bottom(brick)
        [
            inputs.z_min_m,
            inputs.dz_m,
            inputs.r_top_m,
            inputs.r_bottom_m,
        ],
        inputs.geo.geo0,
        inputs.geo.geo1,
        inputs.geo.geo2,
        inputs.geo.geo3,
        inputs.ql,
        inputs.qp,
        // m0: coarse_step_m, fine_step_m, max_steps, unused
        [m.coarse_step_m, m.fine_step_m, m.max_steps, 0.0],
        // m1: exposure, octaves, beer_powder, ground_albedo
        [
            m.exposure,
            m.octaves,
            if m.beer_powder { 1.0 } else { 0.0 },
            m.ground_albedo,
        ],
        // sod_c: sun_od centre xyz, transmittance_floor
        [c[0], c[1], c[2], m.transmittance_floor],
        // sod_u: au xyz, u_min
        [au[0], au[1], au[2], so.u_min as f32],
        // sod_v: av xyz, u_max
        [av[0], av[1], av[2], so.u_max as f32],
        // sod_e: v_min, v_max, sunod_dim, clouds_enabled (this path is clouds-on)
        [so.v_min as f32, so.v_max as f32, so.dim as f32, 1.0],
        // frx: the froxel scan rect
        inputs.scan_rect,
        // frx2: froxel_dim, edge_feather_cells, ground_day_lift, visible cloud OD scale
        [
            inputs.froxel_dim as f32,
            m.edge_feather_cells,
            m.ground_day_lift,
            crate::clouds::validated_cloud_optical_depth_scale(m.cloud_optical_depth_scale),
        ],
        // land0/land1: exact twins of the clear-surface land controls. Appended to
        // preserve every historical cloud-uniform offset above.
        s[9],
        s[10],
    ]
}

/// The 13 packed vec4s of the WGSL `SunOdUniforms` for one row band (`ty_offset` is
/// the band's first row — the TDR chunking offset in `vert.w`).
pub fn sun_od_uniform_quads(inputs: &CloudFrameInputs, ty_offset: u32) -> [[f32; 4]; 13] {
    let so = &inputs.sun_od;
    let f3 = |v: [f64; 3]| [v[0] as f32, v[1] as f32, v[2] as f32];
    let (c, au, av, sun) = (f3(so.center), f3(so.au), f3(so.av), f3(so.sun));
    [
        [c[0], c[1], c[2], 0.0],
        [au[0], au[1], au[2], so.u_min as f32],
        [av[0], av[1], av[2], so.u_max as f32],
        [sun[0], sun[1], sun[2], so.v_min as f32],
        // extent: v_max, s_start, s_len, n_steps
        [
            so.v_max as f32,
            so.s_start as f32,
            so.s_len as f32,
            so.n_steps as f32,
        ],
        // dims: nx, ny, nz, map_dim
        [
            inputs.vol_nx as f32,
            inputs.vol_ny as f32,
            inputs.vol_nz as f32,
            so.dim as f32,
        ],
        // vert: z_min, dz, ds, ty_offset (row-band)
        [inputs.z_min_m, inputs.dz_m, so.ds as f32, ty_offset as f32],
        inputs.geo.geo0,
        inputs.geo.geo1,
        inputs.geo.geo2,
        inputs.geo.geo3,
        inputs.ql,
        inputs.qp,
    ]
}

/// Flatten packed vec4 quads to little-endian bytes for a uniform upload.
pub fn quads_to_bytes(quads: &[[f32; 4]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(quads.len() * 16);
    for v in quads.iter().flatten() {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// GPU resources of the ACTIVATED cloud pass: the `sun_od.wgsl` compute pipeline +
/// the `clouds.wgsl` full-screen march pipeline (created once; per-frame textures and
/// bind groups are built in [`CloudPassResources::render`]).
pub struct CloudPassResources {
    cloud_pipeline: wgpu::RenderPipeline,
    cloud_bgl: wgpu::BindGroupLayout,
    sunod_pipeline: wgpu::ComputePipeline,
    sunod_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl CloudPassResources {
    /// One-time GPU setup (call from the app constructor with the wgpu device).
    pub fn init(device: &wgpu::Device) -> Self {
        let cloud_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("simsat-clouds"),
            source: wgpu::ShaderSource::Wgsl(CLOUDS_WGSL.into()),
        });
        let sunod_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("simsat-sun-od"),
            source: wgpu::ShaderSource::Wgsl(SUN_OD_WGSL.into()),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("simsat-clouds-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let tex2d = |binding: u32, filterable: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let tex3d = |binding: u32, filterable: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        };
        let cloud_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("simsat-clouds-bgl"),
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
                tex2d(1, false), // lut_geo
                tex2d(2, false), // lut_light
                tex2d(3, true),  // bm_tex
                tex2d(4, true),  // normal_tex
                tex2d(5, true),  // landmask_tex
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                tex2d(7, false),  // transmittance_lut
                tex2d(8, false),  // multiscatter_lut
                tex2d(9, false),  // ambient_lut (scalar; layout-bound, shader-legacy)
                tex3d(10, true),  // volume (Rgba8Unorm, hardware trilinear)
                tex3d(11, true),  // occupancy (R8Unorm, hardware trilinear)
                tex2d(12, false), // sun_od (R32Float)
                tex3d(13, false), // froxel (Rgba32Float)
                tex2d(14, false), // sh_ambient (Rgba32Float)
                tex2d(15, false), // sun_od_dist (R32Float)
            ],
        });
        let cloud_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("simsat-clouds-pl"),
            bind_group_layouts: &[Some(&cloud_bgl)],
            immediate_size: 0,
        });
        let cloud_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("simsat-clouds-pipeline"),
            layout: Some(&cloud_pl),
            vertex: wgpu::VertexState {
                module: &cloud_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &cloud_shader,
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

        let sunod_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("simsat-sun-od-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let sunod_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("simsat-sun-od-pl"),
            bind_group_layouts: &[Some(&sunod_bgl)],
            immediate_size: 0,
        });
        let sunod_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("simsat-sun-od-pipeline"),
            layout: Some(&sunod_pl),
            module: &sunod_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            cloud_pipeline,
            cloud_bgl,
            sunod_pipeline,
            sunod_bgl,
            sampler,
        }
    }

    /// Render one cloud-composited frame: upload the volume, run the banded sun-OD
    /// compute, run the tiled cloud fragment pass offscreen, and read it back — the
    /// same RGBA output contract as the CPU composite (`clouds::render_cloud_frame_rgba`).
    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &CloudFrameInputs,
    ) -> RenderedFrame {
        let (w, h) = (inputs.surface.width.max(1), inputs.surface.height.max(1));

        // The 3-D cloud volume + occupancy mip (the M4 upload machinery, first live use).
        let volume = CloudVolumeResources::new(
            device,
            inputs.vol_nx.max(1),
            inputs.vol_ny.max(1),
            inputs.vol_nz.max(1),
            inputs.occ_dims,
        );
        volume.upload(queue, inputs.texture_a, inputs.occupancy);

        // Sun-OD map targets (od + occluder distance), storage-written then sampled.
        let dim = inputs.sun_od.dim.max(1) as u32;
        let sunod_usage =
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        let make_sunod_tex = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: dim,
                    height: dim,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: sunod_usage,
                view_formats: &[],
            })
        };
        let sun_od_tex = make_sunod_tex("simsat-sun-od");
        let sun_od_dist_tex = make_sunod_tex("simsat-sun-od-dist");

        // Banded sun-OD compute: one submit per row band (TDR bound). Each band gets
        // its own small uniform buffer carrying the band's row offset in vert.w.
        let sun_od_view = sun_od_tex.create_view(&Default::default());
        let sun_od_dist_view = sun_od_dist_tex.create_view(&Default::default());
        let vol_view = volume.volume_view();
        let band_rows = sun_od_band_rows(dim, inputs.sun_od.n_steps as u32);
        for (y0, rows) in row_bands(dim, band_rows) {
            let quads = sun_od_uniform_quads(inputs, y0);
            let bytes = quads_to_bytes(&quads);
            let ub = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("simsat-sun-od-uniforms"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&ub, 0, &bytes);
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("simsat-sun-od-bg"),
                layout: &self.sunod_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ub.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&vol_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&sun_od_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&sun_od_dist_view),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("simsat-sun-od-encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("simsat-sun-od-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sunod_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(dim.div_ceil(8), rows.div_ceil(8), 1);
            }
            queue.submit([encoder.finish()]);
        }

        // Per-frame 2-D textures (shared shapes with the surface pass).
        let s = &inputs.surface;
        let lut_geo = upload_rgba32f(device, queue, w, h, s.lut_geo, "lut-geo");
        let lut_light = upload_rgba32f(device, queue, w, h, s.lut_light, "lut-light");
        let transmittance_tex =
            upload_rgba32f(device, queue, 256, 64, s.transmittance_lut, "transmittance");
        let multiscatter_tex =
            upload_rgba32f(device, queue, 32, 32, s.multiscatter_lut, "multiscatter");
        let ambient_tex = upload_rgba32f(
            device,
            queue,
            s.ambient_n.max(1),
            1,
            s.ambient_lut,
            "ambient",
        );
        let normal_tex = upload_rgba8(device, queue, s.nx, s.ny, s.normals_rgba, "normals");
        let landmask_tex = upload_r8(device, queue, s.nx, s.ny, s.landmask_r8, "landmask");
        let bm_tex = match s.bluemarble {
            Some(bm) => upload_rgba8(device, queue, bm.width, bm.height, &bm.rgba, "bluemarble"),
            None => {
                let gray = (FLAT_ALBEDO_SRGB * 255.0) as u8;
                upload_rgba8(device, queue, 1, 1, &[gray, gray, gray, 255], "bm-dummy")
            }
        };

        // Froxel (3-D Rgba32Float) + the SH-2 sky-ambient table (9 x rows Rgba32Float).
        let fdim = inputs.froxel_dim.max(1);
        let froxel_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("simsat-froxel"),
            size: wgpu::Extent3d {
                width: fdim,
                height: fdim,
                depth_or_array_layers: fdim,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &froxel_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck_f32(inputs.froxel_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(fdim * 16),
                rows_per_image: Some(fdim),
            },
            wgpu::Extent3d {
                width: fdim,
                height: fdim,
                depth_or_array_layers: fdim,
            },
        );
        let sh_tex = upload_rgba32f(
            device,
            queue,
            9,
            inputs.sh_rows.max(1),
            inputs.sh_data,
            "sh-ambient",
        );

        // The 432-byte packed cloud uniform.
        let quads = cloud_uniform_quads(inputs);
        let uniform_bytes = quads_to_bytes(&quads);
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simsat-clouds-uniforms"),
            size: uniform_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniforms, 0, &uniform_bytes);

        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("simsat-clouds-bg"),
            layout: &self.cloud_bgl,
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
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(&volume.volume_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(&volume.occupancy_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(&sun_od_view),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::TextureView(&view(&froxel_tex)),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::TextureView(&view(&sh_tex)),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::TextureView(&sun_od_dist_view),
                },
            ],
        });

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("simsat-clouds-target"),
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

        // Tiled cloud pass: one scissor band per submit (TDR bound). The first band
        // clears the whole target; later bands load it back.
        for (i, (y0, rows)) in row_bands(h, cloud_tile_rows(w)).into_iter().enumerate() {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("simsat-clouds-encoder"),
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("simsat-clouds-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: if i == 0 {
                                wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                            } else {
                                wgpu::LoadOp::Load
                            },
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.cloud_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_scissor_rect(0, y0, w, rows);
                pass.draw(0..3, 0..1);
            }
            queue.submit([encoder.finish()]);
        }

        // Readback (256-byte aligned rows), then wait like the surface pass.
        let unpadded_bpr = w * 4;
        let padded_bpr = align_up(unpadded_bpr, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simsat-clouds-readback"),
            size: (padded_bpr * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("simsat-clouds-readback-encoder"),
        });
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::bricks::ChannelQuant;
    use crate::camera::{GeoCamera, SatellitePreset, build_surface_raster};
    use crate::frame::{GridGeoref, MapProjection};

    fn occupancy_test_brick(
        dims: (usize, usize, usize),
        ext_liquid: Vec<u8>,
        ext_ice: Vec<u8>,
        ext_precip: Vec<u8>,
        quantizers: (LogQuant, LogQuant, LogQuant),
    ) -> VolumeBrick {
        let (nx, ny, nz) = dims;
        let cells = nx * ny * nz;
        let plane = nx * ny;
        assert_eq!(ext_liquid.len(), cells);
        assert_eq!(ext_ice.len(), cells);
        assert_eq!(ext_precip.len(), cells);
        let zero = LogQuant {
            vmin: 0.0,
            vmax: 0.0,
        };
        let mut quant = BTreeMap::new();
        quant.insert("ext_liquid".to_string(), quantizers.0);
        quant.insert("ext_ice".to_string(), quantizers.1);
        quant.insert("ext_snow".to_string(), zero);
        quant.insert("ext_precip".to_string(), quantizers.2);
        quant.insert("tau_up".to_string(), zero);
        quant.insert("qvapor".to_string(), zero);
        VolumeBrick {
            nx,
            ny,
            nz,
            z_min_m: 0.0,
            dz_m: 250.0,
            time_iso: None,
            quant: ChannelQuant(quant),
            ext_liquid,
            ext_ice,
            ext_snow: vec![0; cells],
            ext_precip,
            tau_up: vec![0; cells],
            qvapor: vec![0; cells],
            cloud_fraction: vec![255; cells],
            has_cloud_fraction: false,
            temperature_f16: vec![0; cells],
            hgt: vec![0.0; plane],
            landmask: vec![1.0; plane],
            tsk: vec![300.0; plane],
            u10: vec![0.0; plane],
            v10: vec![0.0; plane],
            snowh: None,
            ivgtyp: None,
        }
    }

    fn decoded_occupancy_reference(brick: &VolumeBrick, factor: usize) -> QuantizedOccupancyUpload {
        let volume = crate::clouds::DecodedVolume::from_brick_legacy(brick, 1000.0);
        let mip = crate::clouds::OccupancyMip::build(&volume, factor);
        QuantizedOccupancyUpload {
            dims: (mip.mx as u32, mip.my as u32, mip.mz as u32),
            r8: mip.to_r8_occupancy(),
        }
    }

    fn assert_quantized_occupancy_matches(brick: &VolumeBrick, factor: usize, label: &str) {
        let got = quantized_occupancy_upload(brick, factor);
        let expected = decoded_occupancy_reference(brick, factor);
        assert_eq!(got.dims, expected.dims, "{label}: dimensions");
        assert_eq!(got.r8, expected.r8, "{label}: occupancy bytes");
    }

    #[test]
    fn quantized_occupancy_matches_every_code_and_logquant_edge() {
        let codes: Vec<u8> = (0u16..=255).map(|code| code as u8).collect();
        let zero_codes = vec![0; codes.len()];
        let quantizers = [
            (
                "normal",
                LogQuant {
                    vmin: 1.0e-12,
                    vmax: 1.0e-1,
                },
            ),
            (
                "zero",
                LogQuant {
                    vmin: 0.0,
                    vmax: 0.0,
                },
            ),
            (
                "negative-max",
                LogQuant {
                    vmin: -1.0,
                    vmax: -0.1,
                },
            ),
            (
                "degenerate-positive",
                LogQuant {
                    vmin: 0.25,
                    vmax: 0.25,
                },
            ),
            (
                "reversed-positive",
                LogQuant {
                    vmin: 1.0,
                    vmax: 0.25,
                },
            ),
            (
                "zero-min",
                LogQuant {
                    vmin: 0.0,
                    vmax: 1.0,
                },
            ),
            (
                "negative-min",
                LogQuant {
                    vmin: -1.0,
                    vmax: 1.0,
                },
            ),
            (
                "nan-min",
                LogQuant {
                    vmin: f64::NAN,
                    vmax: 1.0,
                },
            ),
            (
                "nan-max",
                LogQuant {
                    vmin: 1.0e-6,
                    vmax: f64::NAN,
                },
            ),
            (
                "infinite-max",
                LogQuant {
                    vmin: 1.0e-6,
                    vmax: f64::INFINITY,
                },
            ),
        ];
        let zero = LogQuant {
            vmin: 0.0,
            vmax: 0.0,
        };
        for &(quant_label, quant) in &quantizers {
            for channel in 0..3 {
                let mut channels = [zero_codes.clone(), zero_codes.clone(), zero_codes.clone()];
                channels[channel] = codes.clone();
                let mut scales = [zero; 3];
                scales[channel] = quant;
                let brick = occupancy_test_brick(
                    (16, 16, 1),
                    channels[0].clone(),
                    channels[1].clone(),
                    channels[2].clone(),
                    (scales[0], scales[1], scales[2]),
                );
                for factor in [1, 7, 64] {
                    assert_quantized_occupancy_matches(
                        &brick,
                        factor,
                        &format!("{quant_label}/channel-{channel}/factor-{factor}"),
                    );
                }
            }
        }
    }

    #[test]
    fn quantized_occupancy_matches_mixed_codes_dilation_and_awkward_dimensions() {
        let dims = (13usize, 11usize, 5usize);
        let cells = dims.0 * dims.1 * dims.2;
        let mut seed = 0x9e37_79b9u32;
        let mut next_code = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as u8
        };
        let liquid: Vec<u8> = (0..cells).map(|_| next_code()).collect();
        let ice: Vec<u8> = (0..cells).map(|_| next_code()).collect();
        let mut precip: Vec<u8> = (0..cells).map(|_| next_code()).collect();
        // Explicit isolated corner + interior probes exercise partial edge blocks and
        // the one-block dilation in addition to the deterministic mixed-code field.
        precip.fill(0);
        precip[0] = 1;
        precip[cells / 2] = 127;
        precip[cells - 1] = 255;
        let normal = LogQuant {
            vmin: 1.0e-10,
            vmax: 1.0e-2,
        };
        let nan_min = LogQuant {
            vmin: f64::NAN,
            vmax: 1.0,
        };
        let reversed = LogQuant {
            vmin: 0.5,
            vmax: 1.0e-4,
        };
        for (label, quantizers) in [
            ("all-normal", (normal, normal, normal)),
            ("nan-poisons-positive", (normal, nan_min, normal)),
            ("reversed-mix", (reversed, normal, reversed)),
        ] {
            let brick = occupancy_test_brick(
                dims,
                liquid.clone(),
                ice.clone(),
                precip.clone(),
                quantizers,
            );
            for factor in [0, 1, 2, 3, 8, 32] {
                assert_quantized_occupancy_matches(
                    &brick,
                    factor,
                    &format!("{label}/factor-{factor}"),
                );
            }
        }
    }

    #[test]
    fn quantized_occupancy_real_fixture_benchmark() {
        use sha2::{Digest, Sha256};

        let Ok(path) = std::env::var("SIMSAT_GPU_OCCUPANCY_BENCH_BRICK") else {
            eprintln!("SIMSAT_GPU_OCCUPANCY_BENCH_BRICK unset; skipping occupancy benchmark");
            return;
        };
        let mode = std::env::var("SIMSAT_GPU_OCCUPANCY_BENCH_MODE")
            .unwrap_or_else(|_| "compare".to_string());
        let read_started = std::time::Instant::now();
        let brick = crate::bricks::read_ssb(std::path::Path::new(&path))
            .unwrap_or_else(|error| panic!("read benchmark brick {path}: {error}"));
        let read_wall = read_started.elapsed();
        let prep_started = std::time::Instant::now();
        let upload = match mode.as_str() {
            "raw" => quantized_occupancy_upload(&brick, crate::clouds::OCCUPANCY_MIP_FACTOR),
            "decoded" => decoded_occupancy_reference(&brick, crate::clouds::OCCUPANCY_MIP_FACTOR),
            "compare" => {
                let raw = quantized_occupancy_upload(&brick, crate::clouds::OCCUPANCY_MIP_FACTOR);
                let decoded =
                    decoded_occupancy_reference(&brick, crate::clouds::OCCUPANCY_MIP_FACTOR);
                assert_eq!(raw, decoded, "real-fixture occupancy mismatch");
                raw
            }
            other => panic!("unknown SIMSAT_GPU_OCCUPANCY_BENCH_MODE={other}"),
        };
        let prep_wall = prep_started.elapsed();
        let digest = Sha256::digest(&upload.r8);
        let hash = format!("{digest:x}");
        let peak_rss = crate::platform::peak_rss_bytes()
            .map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string());
        eprintln!(
            "GPU_OCCUPANCY_BENCH mode={mode} brick_dims={}x{}x{} occ_dims={}x{}x{} \
             occ_bytes={} sha256={hash} read_wall_s={:.6} prep_wall_s={:.6} \
             peak_rss_bytes={peak_rss}",
            brick.nx,
            brick.ny,
            brick.nz,
            upload.dims.0,
            upload.dims.1,
            upload.dims.2,
            upload.r8.len(),
            read_wall.as_secs_f64(),
            prep_wall.as_secs_f64(),
        );
    }

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

    /// A test-fixture `SurfaceUniforms` with recognizable values.
    fn test_surface_uniforms() -> SurfaceUniforms {
        SurfaceUniforms {
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
            atmosphere_correction: 1.0,
            land_appearance: LandAppearanceConfig::identity(),
        }
    }

    /// A test-fixture `CloudFrameInputs` over empty texture slices (the packers only
    /// read the scalar fields).
    fn test_cloud_inputs<'a>(uniforms: SurfaceUniforms) -> CloudFrameInputs<'a> {
        CloudFrameInputs {
            surface: SurfaceFrameInputs {
                width: 4,
                height: 4,
                lut_geo: &[],
                lut_light: &[],
                nx: 2,
                ny: 2,
                normals_rgba: &[],
                landmask_r8: &[],
                bluemarble: None,
                transmittance_lut: &[],
                multiscatter_lut: &[],
                ambient_lut: &[],
                ambient_n: 48,
                uniforms,
            },
            vol_nx: 80,
            vol_ny: 60,
            vol_nz: 40,
            texture_a: &[],
            occ_dims: (10, 8, 5),
            occupancy: &[],
            ql: [1.0e-6, 1.0e-2, 2.0e-6, 2.0e-2],
            qp: [3.0e-6, 3.0e-2, 4.0e-4, 40.0],
            z_min_m: 100.0,
            dz_m: 250.0,
            r_top_m: 6_380_000.0,
            r_bottom_m: 6_370_100.0,
            voxel_pitch_m: 500.0,
            geo: GeoQuads {
                geo0: [0.0, 0.0, 0.0, 4000.0],
                geo1: [-1.0e6, -2.0e6, 4000.0, -97.5],
                geo2: [0.7, 1.9, 0.0, 0.0],
                geo3: [0.0, 0.0, 0.0, 0.0],
            },
            march: CloudMarchParams {
                coarse_step_m: 1000.0,
                fine_step_m: 250.0,
                max_steps: 192.0,
                exposure: 1.6,
                octaves: 6.0,
                beer_powder: false,
                ground_albedo: 0.3,
                transmittance_floor: 0.003,
                cloud_optical_depth_scale: 0.75,
                edge_feather_cells: 3.2,
                ground_day_lift: 2.0,
            },
            sun_od: SunOdPlan {
                center: [6.37e6, 1000.0, 2000.0],
                au: [0.0, 1.0, 0.0],
                av: [0.0, 0.0, 1.0],
                sun: [1.0, 0.0, 0.0],
                u_min: -50_000.0,
                u_max: 50_000.0,
                v_min: -40_000.0,
                v_max: 40_000.0,
                s_start: 60_000.0,
                s_len: 120_000.0,
                n_steps: 240,
                ds: 500.0,
                dim: 512,
            },
            froxel_dim: 32,
            froxel_data: &[],
            sh_rows: 48,
            sh_data: &[],
            scan_rect: [-0.01, 0.01, -0.008, 0.008],
        }
    }

    #[test]
    fn cloud_uniform_quads_pack_the_wgsl_layout() {
        let inputs = test_cloud_inputs(test_surface_uniforms());
        let quads = cloud_uniform_quads(&inputs);
        assert_eq!(quads.len(), 27);
        // The first 9 quads are the surface uniforms verbatim.
        assert_eq!(quads[0], [1.0, 2.0, 3.0, 6_370_000.0]);
        assert_eq!(quads[8], [-20.0, 90.0, 48.0, 1.0]);
        // dims: nx, ny, nz, voxel_pitch.
        assert_eq!(quads[9], [80.0, 60.0, 40.0, 500.0]);
        // vert: z_min, dz, r_top, r_bottom.
        assert_eq!(quads[10], [100.0, 250.0, 6_380_000.0, 6_370_100.0]);
        // m0 + m1: the march schedule and exposure/octaves/powder/albedo.
        assert_eq!(quads[17], [1000.0, 250.0, 192.0, 0.0]);
        assert_eq!(quads[18], [1.6, 6.0, 0.0, 0.3]);
        // sod_c.w = transmittance floor; sod_e = extents + dim + clouds-enabled.
        assert_eq!(quads[19][3], 0.003);
        assert_eq!(quads[22], [-40_000.0, 40_000.0, 512.0, 1.0]);
        // frx2: froxel dim, edge feather, ground lift, visible cloud OD scale.
        assert_eq!(quads[24], [32.0, 3.2, 2.0, 0.75]);
        // Appended land controls preserve every historical cloud offset above.
        assert_eq!(quads[25], [0.0, 1.6, 0.0, 0.08]);
        assert_eq!(quads[26], [0.65, 1.5, 0.0, 0.0]);
        let mut invalid = test_cloud_inputs(test_surface_uniforms());
        invalid.march.cloud_optical_depth_scale = f32::NAN;
        assert_eq!(
            cloud_uniform_quads(&invalid)[24][3],
            crate::clouds::DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE
        );
        invalid.march.cloud_optical_depth_scale = 99.0;
        assert_eq!(cloud_uniform_quads(&invalid)[24][3], 4.0);
        // 27 vec4 = 432 bytes, little-endian f32s in order.
        let bytes = quads_to_bytes(&quads);
        assert_eq!(bytes.len(), 432);
        assert_eq!(f32::from_le_bytes(bytes[0..4].try_into().unwrap()), 1.0);
        assert_eq!(
            f32::from_le_bytes(bytes[9 * 16..9 * 16 + 4].try_into().unwrap()),
            80.0
        );
    }

    fn wgsl_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// Executable f32 transcription of the marked WGSL helper. The exhaustive property
    /// test below anchors both shader copies to the public f64 CPU reference over the
    /// whole solar-elevation range and representative reflectance/control domains.
    fn wgsl_land_appearance_gain(
        config: LandAppearanceConfig,
        sun_elev: f32,
        albedo: [f32; 3],
    ) -> f32 {
        let land = land_appearance_uniform_quads(config);
        let (land0, land1) = (land[0], land[1]);
        if land0[0] <= 0.5 && land0[2] <= 0.5 {
            return 1.0;
        }

        let sza = if land0[0] <= 0.5
            || land0[1] == 1.0
            || sun_elev >= crate::render::LAND_SZA_REFERENCE_ELEV_DEG as f32
        {
            1.0
        } else {
            let mu_ref = (crate::render::LAND_SZA_REFERENCE_ELEV_DEG as f32)
                .to_radians()
                .sin();
            let mu_floor = (crate::render::AERIAL_VEIL_ELEV_LO_DEG as f32)
                .to_radians()
                .sin();
            let mu = sun_elev.clamp(0.0, 90.0).to_radians().sin();
            let target = (mu_ref / mu.max(mu_floor)).clamp(1.0, land0[1]);
            1.0 + wgsl_smoothstep(
                crate::render::AERIAL_VEIL_ELEV_LO_DEG as f32,
                crate::render::AERIAL_VEIL_ELEV_HI_DEG as f32,
                sun_elev,
            ) * (target - 1.0)
        };

        let y = (0.2126 * albedo[0] + 0.7152 * albedo[1] + 0.0722 * albedo[2]).max(0.0);
        let toe =
            if land0[2] <= 0.5 || y <= 0.0 || y >= land0[3] || land1[1] == 1.0 || land1[0] == 1.0 {
                1.0
            } else {
                let power_target = land0[3] * (y / land0[3]).powf(land1[0]);
                let w = wgsl_smoothstep(0.0, land0[3], y);
                let target = power_target * (1.0 - w) + y * w;
                let gain = (target / y).clamp(1.0, land1[1]);
                1.0 + wgsl_smoothstep(
                    crate::render::AERIAL_VEIL_ELEV_LO_DEG as f32,
                    crate::render::AERIAL_VEIL_ELEV_HI_DEG as f32,
                    sun_elev,
                ) * (gain - 1.0)
            };
        sza * toe
    }

    #[test]
    fn land_appearance_uniforms_match_cpu_sanitization_and_identity() {
        let identity = land_appearance_uniform_quads(LandAppearanceConfig::identity());
        assert_eq!(identity[0], [0.0, 1.6, 0.0, 0.08]);
        assert_eq!(identity[1], [0.65, 1.5, 0.0, 0.0]);
        let shipped = land_appearance_uniform_quads(LandAppearanceConfig::default());
        assert_eq!(shipped[0], [1.0, 1.6, 1.0, 0.08]);
        assert_eq!(shipped[1], [0.65, 1.5, 0.0, 0.0]);

        let invalid = land_appearance_uniform_quads(LandAppearanceConfig {
            sza_normalization: true,
            sza_max_gain: f64::NAN,
            dark_toe: true,
            dark_toe_knee: f64::INFINITY,
            dark_toe_gamma: -9.0,
            dark_toe_max_gain: 99.0,
        });
        assert_eq!(invalid[0], [1.0, 1.6, 1.0, 0.08]);
        assert_eq!(invalid[1], [0.05, 4.0, 0.0, 0.0]);

        for elev in [-90.0, 0.0, 20.0, 25.0, 30.0, 60.0, 90.0] {
            assert_eq!(
                wgsl_land_appearance_gain(
                    LandAppearanceConfig::identity(),
                    elev,
                    [0.01, 0.04, 0.02]
                )
                .to_bits(),
                1.0f32.to_bits(),
                "both-off must be an exact shader identity at {elev} deg"
            );
        }
    }

    #[test]
    fn wgsl_land_appearance_math_tracks_cpu_reference_over_domain() {
        let identity = LandAppearanceConfig::identity();
        let configs = [
            identity,
            LandAppearanceConfig::default(),
            LandAppearanceConfig {
                sza_normalization: true,
                ..identity
            },
            LandAppearanceConfig {
                dark_toe: true,
                ..identity
            },
            LandAppearanceConfig {
                sza_normalization: true,
                sza_max_gain: 2.35f32 as f64,
                dark_toe: true,
                dark_toe_knee: 0.12f32 as f64,
                dark_toe_gamma: 0.42f32 as f64,
                dark_toe_max_gain: 2.7f32 as f64,
            },
            LandAppearanceConfig {
                sza_normalization: true,
                sza_max_gain: f64::NAN,
                dark_toe: true,
                dark_toe_knee: f64::INFINITY,
                dark_toe_gamma: f64::NEG_INFINITY,
                dark_toe_max_gain: f64::NAN,
            },
        ];
        let albedos = [
            [0.0, 0.0, 0.0],
            [1.0e-6, 1.0e-6, 1.0e-6],
            [0.001, 0.004, 0.002],
            [0.01, 0.02, 0.04],
            [0.04, 0.04, 0.04],
            [0.079, 0.079, 0.079],
            [0.081, 0.081, 0.081],
            [0.2, 0.1, 0.05],
            [1.0, 1.0, 1.0],
        ];
        for config in configs {
            for elev_i in -90..=90 {
                let elev = elev_i as f32;
                for albedo in albedos {
                    let gpu = wgsl_land_appearance_gain(config, elev, albedo);
                    let cpu = crate::render::land_appearance_gain(
                        config,
                        elev as f64,
                        [albedo[0] as f64, albedo[1] as f64, albedo[2] as f64],
                    ) as f32;
                    let delta = (gpu - cpu).abs();
                    assert!(
                        delta <= 3.0e-5,
                        "CPU/WGSL land gain delta {delta} at elev={elev}, albedo={albedo:?}, config={config:?}: gpu={gpu}, cpu={cpu}"
                    );
                    if elev <= crate::render::AERIAL_VEIL_ELEV_LO_DEG as f32 {
                        assert_eq!(gpu.to_bits(), 1.0f32.to_bits(), "twilight identity");
                    }
                }
            }
        }
    }

    #[test]
    fn both_visible_shaders_share_identical_land_helper_and_land_only_callsite() {
        fn marked_helper(shader: &str) -> &str {
            shader
                .split_once("// LAND_APPEARANCE_TWIN_BEGIN")
                .and_then(|(_, rest)| rest.split_once("// LAND_APPEARANCE_TWIN_END"))
                .map(|(helper, _)| helper)
                .expect("marked land-appearance WGSL helper")
        }
        assert_eq!(marked_helper(SURFACE_WGSL), marked_helper(CLOUDS_WGSL));
        let call = "l_surf = l_surf * land_appearance_gain_gpu(sun_elev, albedo);";
        for (name, shader) in [("surface", SURFACE_WGSL), ("clouds", CLOUDS_WGSL)] {
            assert_eq!(
                shader.matches(call).count(),
                1,
                "{name} must apply the gain exactly once in its land surface branch"
            );
            let call_at = shader.find(call).unwrap();
            let water_at = shader[..call_at]
                .rfind("if (is_water)")
                .expect("water/land branch before land appearance call");
            let land_at = shader[water_at..call_at]
                .rfind("} else {")
                .expect("land branch before land appearance call");
            assert!(land_at > 0, "{name}: correction must be inside land branch");
        }
    }

    #[test]
    fn wgsl_highlight_calibration_matches_rust_defaults() {
        assert_eq!(crate::render::CLOUD_SOFTCLIP_KNEE, 0.65);
        assert_eq!(crate::render::RHO_HIGHLIGHT_MAX, 1.25);
        for (name, shader) in [
            ("clouds", include_str!("shaders/clouds.wgsl")),
            ("surface", include_str!("shaders/surface.wgsl")),
        ] {
            assert!(
                shader.contains("const CLOUD_SOFTCLIP_KNEE: f32 = 0.65;"),
                "{name} WGSL highlight knee drifted from Rust"
            );
            assert!(
                shader.contains("const RHO_HIGHLIGHT_MAX: f32 = 1.25;"),
                "{name} WGSL highlight ceiling drifted from Rust"
            );
        }
    }

    #[test]
    fn sun_od_uniform_quads_pack_the_wgsl_layout() {
        let inputs = test_cloud_inputs(test_surface_uniforms());
        let quads = sun_od_uniform_quads(&inputs, 64);
        assert_eq!(quads.len(), 13);
        // center / au / av / sun with the frame extents in the w lanes.
        assert_eq!(quads[0], [6.37e6, 1000.0, 2000.0, 0.0]);
        assert_eq!(quads[1], [0.0, 1.0, 0.0, -50_000.0]);
        assert_eq!(quads[2], [0.0, 0.0, 1.0, 50_000.0]);
        assert_eq!(quads[3], [1.0, 0.0, 0.0, -40_000.0]);
        // extent: v_max, s_start, s_len, n_steps.
        assert_eq!(quads[4], [40_000.0, 60_000.0, 120_000.0, 240.0]);
        // dims: brick dims + map dim.
        assert_eq!(quads[5], [80.0, 60.0, 40.0, 512.0]);
        // vert: z_min, dz, ds, ty_offset (the TDR row-band offset).
        assert_eq!(quads[6], [100.0, 250.0, 500.0, 64.0]);
        // 13 vec4 = 208 bytes.
        assert_eq!(quads_to_bytes(&quads).len(), 208);
    }

    /// A Rust twin of the WGSL `project` + `ecef_to_brick` anchor arithmetic, driven
    /// by the PACKED `GeoQuads` values (f32, like the shader reads them).
    fn wgsl_forward_twin(q: &GeoQuads, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        let kind = (q.geo0[0] + 0.5) as i32;
        let cm = q.geo1[3] as f64;
        let r: f64 = 6_370_000.0;
        let pi = std::f64::consts::PI;
        let phi = lat_deg.clamp(-89.999, 89.999).to_radians();
        let mut dlon = lon_deg - cm;
        dlon -= 360.0 * ((dlon + 180.0) / 360.0).floor();
        let dlon_r = dlon.to_radians();
        let (u, v) = match kind {
            0 => {
                let n = q.geo2[0] as f64;
                let f = q.geo2[1] as f64;
                let rho = r * f / (pi * 0.25 + phi * 0.5).tan().powf(n);
                let theta = n * dlon_r;
                (rho * theta.sin(), -rho * theta.cos())
            }
            1 => {
                let k = q.geo2[2] as f64;
                if q.geo3[0] > 0.5 {
                    let rho = 2.0 * r * k * (pi * 0.25 + phi * 0.5).tan();
                    (rho * dlon_r.sin(), rho * dlon_r.cos())
                } else {
                    let rho = 2.0 * r * k * (pi * 0.25 - phi * 0.5).tan();
                    (rho * dlon_r.sin(), -rho * dlon_r.cos())
                }
            }
            2 => {
                let scale = q.geo2[3] as f64;
                (
                    r * scale * dlon_r,
                    r * scale * (pi * 0.25 + phi * 0.5).tan().ln(),
                )
            }
            _ => (dlon, lat_deg.clamp(-89.999, 89.999)),
        };
        let fi = q.geo0[1] as f64 + (u - q.geo1[0] as f64) / q.geo0[3] as f64;
        let fj = q.geo0[2] as f64 + (v - q.geo1[1] as f64) / q.geo1[2] as f64;
        (fi, fj)
    }

    #[test]
    fn geo_quads_reproduce_georef_forward() {
        // The WGSL projection twin driven by the packed quads must reproduce
        // `georef.forward` for every projection kind (f32 packing tolerance).
        let cases: Vec<(GridGeoref, &str)> = vec![
            (
                GridGeoref::new(
                    MapProjection::lambert(30.0, 60.0, -97.5),
                    99.0,
                    99.0,
                    39.0,
                    -97.5,
                    4000.0,
                    4000.0,
                ),
                "lambert",
            ),
            (
                GridGeoref::new(
                    MapProjection::polar_stereographic(60.0, -150.0, false),
                    50.0,
                    40.0,
                    64.0,
                    -150.0,
                    3000.0,
                    3000.0,
                ),
                "polar",
            ),
            (
                GridGeoref::new(
                    MapProjection::mercator(20.0, -80.0),
                    60.0,
                    60.0,
                    25.0,
                    -80.0,
                    2000.0,
                    2000.0,
                ),
                "mercator",
            ),
            (
                GridGeoref::new(
                    MapProjection::LatLon {
                        central_meridian_deg: -97.5,
                    },
                    30.0,
                    30.0,
                    39.0,
                    -97.5,
                    0.03,
                    0.03,
                ),
                "latlon",
            ),
        ];
        for (georef, name) in &cases {
            let q = geo_quads(georef);
            for &(lat, lon) in &[(38.2, -99.1), (40.7, -96.0), (36.5, -98.4)] {
                // Shift the probe near each projection's own domain.
                let (lat, lon) = match *name {
                    "polar" => (lat + 25.0, lon - 52.0),
                    "mercator" => (lat - 13.0, lon + 18.0),
                    _ => (lat, lon),
                };
                let (fi_ref, fj_ref) = georef.forward(lat, lon);
                let (fi, fj) = wgsl_forward_twin(&q, lat, lon);
                assert!(
                    (fi - fi_ref).abs() < 1.0e-2 && (fj - fj_ref).abs() < 1.0e-2,
                    "{name} at ({lat}, {lon}): twin ({fi}, {fj}) vs georef ({fi_ref}, {fj_ref})"
                );
            }
        }
    }

    #[test]
    fn plan_sun_od_covers_the_brick_corners() {
        let georef = GridGeoref::new(
            MapProjection::lambert(30.0, 60.0, -97.5),
            3.5,
            3.5,
            39.0,
            -97.5,
            4000.0,
            4000.0,
        );
        let (nx, ny, nz) = (8usize, 8usize, 4usize);
        let (z_min, dz, pitch) = (100.0, 500.0, 4000.0);
        let sun = [0.5f64, -0.3, 0.8];
        let plan = plan_sun_od(&georef, nx, ny, nz, z_min, dz, pitch, sun, 512);
        assert_eq!(plan.dim, 512);
        assert!(plan.n_steps >= 1 && plan.n_steps <= 1024);
        assert!((plan.ds * plan.n_steps as f64 - plan.s_len).abs() < 1.0e-6 * plan.s_len);
        // Orthonormal sun frame.
        assert!((dot3(plan.sun, plan.sun) - 1.0).abs() < 1.0e-9);
        assert!(dot3(plan.sun, plan.au).abs() < 1.0e-9);
        assert!(dot3(plan.sun, plan.av).abs() < 1.0e-9);
        assert!(dot3(plan.au, plan.av).abs() < 1.0e-9);
        // Every brick corner projects inside the (u, v) extents.
        for &ki in &[0.0, (nz - 1) as f64] {
            for &ji in &[0.0, (ny - 1) as f64] {
                for &ii in &[0.0, (nx - 1) as f64] {
                    let p = crate::clouds::brick_to_ecef(&georef, ii, ji, ki, z_min, dz).unwrap();
                    let d = [
                        p[0] - plan.center[0],
                        p[1] - plan.center[1],
                        p[2] - plan.center[2],
                    ];
                    let (u, v) = (dot3(d, plan.au), dot3(d, plan.av));
                    assert!(u >= plan.u_min - 1.0e-6 && u <= plan.u_max + 1.0e-6);
                    assert!(v >= plan.v_min - 1.0e-6 && v <= plan.v_max + 1.0e-6);
                }
            }
        }
    }

    #[test]
    fn row_bands_cover_every_row_exactly_once() {
        let bands = row_bands(100, 32);
        assert_eq!(bands, vec![(0, 32), (32, 32), (64, 32), (96, 4)]);
        assert_eq!(row_bands(8, 8), vec![(0, 8)]);
        assert_eq!(row_bands(0, 8), Vec::<(u32, u32)>::new());
        // Contiguity + exact coverage for an awkward pair.
        let bands = row_bands(513, 64);
        let mut next = 0u32;
        for (y0, rows) in &bands {
            assert_eq!(*y0, next);
            next += rows;
        }
        assert_eq!(next, 513);
    }

    #[test]
    fn tdr_band_sizes_stay_bounded() {
        // Sun-OD: rows * dim * n_steps never exceeds the per-submit sample cap, rows
        // are a positive multiple of the 8-row workgroup height.
        for &(dim, n_steps) in &[(512u32, 1024u32), (512, 240), (1024, 1024), (16, 1)] {
            let rows = sun_od_band_rows(dim, n_steps);
            assert!(rows >= 8 && rows.is_multiple_of(8), "rows {rows}");
            assert!(
                rows as u64 * dim as u64 * n_steps as u64 <= SUN_OD_MAX_SAMPLES_PER_SUBMIT,
                "dim {dim} steps {n_steps} rows {rows}"
            );
        }
        // Cloud tiles: rows * width never exceeds the pixel cap; at least one row.
        for &w in &[1u32, 800, 4096, 5000] {
            let rows = cloud_tile_rows(w);
            assert!(rows >= 1);
            assert!(rows as u64 * w as u64 <= CLOUD_TILE_MAX_PIXELS.max(w as u64));
        }
    }

    #[test]
    fn surface_uniforms_pack_176_bytes_in_order() {
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
            atmosphere_correction: 1.0,
            land_appearance: LandAppearanceConfig::identity(),
        };
        let bytes = u.to_bytes();
        assert_eq!(bytes.len(), 176);
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
        // Trailing word is the product-facing atmosphere-correction flag.
        assert_eq!(f32::from_le_bytes(bytes[140..144].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_le_bytes(bytes[144..148].try_into().unwrap()), 0.0);
        assert_eq!(f32::from_le_bytes(bytes[148..152].try_into().unwrap()), 1.6);
    }
}
