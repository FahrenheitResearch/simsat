//! Env-gated real-fixture ingest test (design doc section 9, strategy 4).
//!
//! Set `SIMSAT_WRF_FIXTURE=<path to a wrfout file>` to exercise a full
//! single-timestep ingest: it asserts completion, that the brick dims are
//! consistent with the file, runs the projection ratchet on the real
//! `XLAT`/`XLONG`, and asserts peak RSS < 2.5 GB (the design contract). When the
//! variable is unset the test returns cleanly so plain `cargo test --workspace`
//! stays deterministic. Run release-profile on large grids:
//!   SIMSAT_WRF_FIXTURE=... cargo test -p simsat --release fixture -- --nocapture

use std::path::{Path, PathBuf};

use simsat::bricks::read_ssb;
use simsat::camera::{GeoCamera, SatellitePreset};
use simsat::ingest::{IngestConfig, ingest_timestep, read_grid_geometry};

/// Hard peak-RSS ceiling from the design contract.
const PEAK_RSS_LIMIT_BYTES: u64 = 2_500_000_000;

#[test]
fn optional_wrf_fixture_ingests_and_ratchets() {
    let Ok(path) = std::env::var("SIMSAT_WRF_FIXTURE") else {
        eprintln!("SIMSAT_WRF_FIXTURE unset; skipping real-fixture ingest test");
        return;
    };
    let path = Path::new(&path);
    assert!(path.is_file(), "fixture path does not exist: {path:?}");

    // Persist the brick when SIMSAT_CACHE_DIR is set (so ssb_preview can render
    // the same brick this single pass wrote); otherwise use a throwaway temp dir
    // that is cleaned up at the end.
    let keep = std::env::var("SIMSAT_CACHE_DIR")
        .ok()
        .filter(|d| !d.trim().is_empty());
    let cache = match &keep {
        Some(dir) => PathBuf::from(dir),
        None => std::env::temp_dir().join(format!("simsat-fixture-{}", std::process::id())),
    };
    let config = IngestConfig::new(cache.clone());

    let report = ingest_timestep(path, &config).expect("ingest should complete");
    println!("FIXTURE BRICK: {}", report.brick_path.display());

    println!(
        "FIXTURE INGEST: run={} dims={}x{}x{} hhmm={:04} wall={:.3}s peak_rss={} ssb_bytes={} ({:.2} MB)",
        report.run_id,
        report.nx,
        report.ny,
        report.nz,
        report.hhmm,
        report.wall.as_secs_f64(),
        report
            .peak_rss_bytes
            .map(|b| format!("{:.1} MB", b as f64 / (1024.0 * 1024.0)))
            .unwrap_or_else(|| "n/a".to_string()),
        report.ssb_bytes,
        report.ssb_bytes as f64 / (1024.0 * 1024.0),
    );

    assert!(report.ssb_bytes > 0, "brick should be non-empty");
    assert!(report.brick_path.is_file(), "brick file should exist");
    assert!(report.manifest_path.is_file(), "run.json should exist");

    if let Some(rss) = report.peak_rss_bytes {
        assert!(
            rss < PEAK_RSS_LIMIT_BYTES,
            "peak RSS {rss} bytes exceeds the 2.5 GB contract"
        );
    }

    // Brick dims consistent with the file, and it reads back cleanly.
    let geom = read_grid_geometry(path, config.timestep).expect("geometry");
    assert_eq!(report.nx, geom.nx);
    assert_eq!(report.ny, geom.ny);
    assert_eq!(report.nz, config.nz_brick);

    let brick = read_ssb(&report.brick_path).expect("read brick back");
    assert_eq!(brick.nx, geom.nx);
    assert_eq!(brick.ny, geom.ny);
    assert_eq!(brick.nz, config.nz_brick);
    assert_eq!(brick.hgt.len(), geom.nx * geom.ny);
    assert_eq!(brick.ext_liquid.len(), geom.nx * geom.ny * config.nz_brick);
    assert_eq!(brick.ext_snow.len(), geom.nx * geom.ny * config.nz_brick);
    assert_eq!(
        brick.cloud_fraction.len(),
        geom.nx * geom.ny * config.nz_brick
    );
    if !brick.has_cloud_fraction {
        assert!(
            brick.cloud_fraction.iter().all(|&v| v == 255),
            "unavailable coverage must use the full-cell fallback"
        );
    }
    assert_eq!(
        brick.temperature_f16.len(),
        geom.nx * geom.ny * config.nz_brick
    );

    // Projection ratchet on the real XLAT/XLONG: every Nth stored coord must
    // project back onto its own (i, j) within 0.05 cell.
    let georef = geom.georef().expect("georef");
    let step = 5usize;
    let mut worst = 0.0f64;
    for j in (0..geom.ny).step_by(step) {
        for i in (0..geom.nx).step_by(step) {
            let idx = j * geom.nx + i;
            let (fi, fj) = georef.forward(geom.xlat[idx] as f64, geom.xlong[idx] as f64);
            worst = worst.max((fi - i as f64).abs()).max((fj - j as f64).abs());
        }
    }
    println!(
        "PROJECTION RATCHET: map_proj={} worst error = {:.5} cells (limit 0.05)",
        geom.params.map_proj, worst
    );
    assert!(worst < 0.05, "ratchet worst {worst} cells exceeds 0.05");

    // Camera ratchet on the real XLAT/XLONG (M1): choose a satellite that sees the
    // domain (GOES-East for CONUS-class fixtures like Michael/Enderlin), then for
    // sampled stored coordinates assert the ported forward -> ported inverse
    // round-trips to < 1e-3 deg, and that scan -> lat/lon -> georef.forward lands
    // back on the same (i, j). Skips gracefully if the domain is off this disk.
    let camera = GeoCamera::new(SatellitePreset::GoesEast);
    let mut cam_worst_deg = 0.0f64;
    let mut cam_worst_cell = 0.0f64;
    let mut visible = 0usize;
    for j in (0..geom.ny).step_by(step) {
        for i in (0..geom.nx).step_by(step) {
            let idx = j * geom.nx + i;
            let (lat, lon) = (geom.xlat[idx] as f64, geom.xlong[idx] as f64);
            let Some((sx, sy)) = camera.forward(lat, lon) else {
                continue;
            };
            visible += 1;
            let (blat, blon) = camera.inverse(sx, sy).expect("inverse on-disk");
            cam_worst_deg = cam_worst_deg.max((blat - lat).abs());
            let dlon = (blon - lon + 180.0).rem_euclid(360.0) - 180.0;
            cam_worst_deg = cam_worst_deg.max(dlon.abs());
            // scan -> lat/lon -> (i, j) must return to the source cell.
            let (fi, fj) = georef.forward(blat, blon);
            cam_worst_cell = cam_worst_cell
                .max((fi - i as f64).abs())
                .max((fj - j as f64).abs());
        }
    }
    println!(
        "CAMERA RATCHET (GOES-East): visible={visible} worst lat/lon = {cam_worst_deg:.6} deg, \
         worst (i,j) = {cam_worst_cell:.5} cells"
    );
    assert!(
        visible > 0,
        "GOES-East should see a CONUS-class fixture domain"
    );
    assert!(
        cam_worst_deg < 1.0e-3,
        "camera round-trip {cam_worst_deg} deg"
    );
    assert!(cam_worst_cell < 0.05, "camera->grid {cam_worst_cell} cells");

    // M4/M5 cloud proof (design section 9): the ingested brick must render a NONZERO
    // cloud fraction and finite radiances via the CPU cloud march, and the M5 Wrenninge
    // multi-scatter octaves must LIFT the peak sunlit reflectance well above the fix2
    // single-scatter number (the brilliance payoff). Reduced resolution (strided rays +
    // a modest sun-OD map) keeps the cost bounded on the full domain.
    {
        use simsat::atmosphere::{
            AtmosphereLuts, AtmosphereParams, CameraGeometry, SkyShTable, sun_enu_to_ecef,
        };
        use simsat::camera::{MAX_AXIS, VISIBLE_PITCH_RAD, build_surface_raster};
        use simsat::clouds::{
            self, CloudScene, DecodedVolume, MarchConfig, OccupancyMip, StepQuality,
            accumulate_sun_od, cloud_frame_stats,
        };

        let horiz = geom.params.dx_m.min(geom.params.dy_m);
        let mut vol = DecodedVolume::from_brick(&brick, horiz);
        if brick.has_cloud_fraction {
            vol.apply_fractional_clouds();
        }
        let mip = OccupancyMip::build(&vol, clouds::OCCUPANCY_MIP_FACTOR);
        let camera_g = GeoCamera::new(SatellitePreset::GoesEast);
        let raster = build_surface_raster(
            &camera_g,
            &georef,
            geom.nx,
            geom.ny,
            VISIBLE_PITCH_RAD,
            MAX_AXIS,
        )
        .expect("Enderlin domain visible from GOES-East");
        let (la0, la1, lo0, lo1) = raster.lat_lon_bbox().expect("on-earth pixels");
        let clat = ((la0 + la1) * 0.5) as f64;
        let clon = ((lo0 + lo1) * 0.5) as f64;
        // A daylight sun ~50 deg over the domain centre (proof is cloud presence +
        // finiteness + the octave brightness gain, not the exact solar geometry).
        let e = 50f64.to_radians();
        let sun_ecef = sun_enu_to_ecef([0.0, e.cos(), e.sin()], clat, clon);
        let params = AtmosphereParams::default();
        let luts = AtmosphereLuts::build(&params);
        let sky_sh = SkyShTable::build(&luts, &params, 16);
        let sun_od = accumulate_sun_od(&vol, &georef, sun_ecef, 256);
        let cam_geo = CameraGeometry::from_sub_lon(SatellitePreset::GoesEast.sub_lon_deg());
        let stride = (raster.nx.max(raster.ny) / 128).max(1);

        let run_stats = |octaves: usize| {
            let cfg = MarchConfig {
                octaves,
                ..MarchConfig::new(StepQuality::Interactive, vol.voxel_pitch_m())
            };
            let scene = CloudScene {
                vol: &vol,
                mip: &mip,
                sun_od: &sun_od,
                georef: &georef,
                luts: &luts,
                sky_sh: &sky_sh,
                sun_ecef,
                cfg,
            };
            cloud_frame_stats(&scene, &cam_geo, &raster, stride, 0.98)
        };

        // Single scatter (octaves=1) == the fix2 baseline; multi-scatter (default N).
        // Time both to report the octave render-time delta (the octaves reuse the one
        // secondary sun march per sample and add only N cheap phase+exp evaluations, so
        // the delta is modest — the primary + secondary marches are unchanged).
        let t0 = std::time::Instant::now();
        let single = run_stats(1);
        let single_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = std::time::Instant::now();
        let multi = run_stats(clouds::DEFAULT_OCTAVES);
        let multi_ms = t1.elapsed().as_secs_f64() * 1000.0;
        println!(
            "CLOUD FRAME: raster={}x{} stride={stride} sampled={} cloudy={} frac={:.3} finite={}",
            raster.nx,
            raster.ny,
            multi.sampled,
            multi.cloudy,
            multi.cloud_fraction(),
            multi.all_finite
        );
        println!(
            "CLOUD RENDER TIME (full raster, Interactive steps): single-scatter {single_ms:.0} ms \
             -> multi-scatter (octaves={}) {multi_ms:.0} ms ({:+.0}% for the octaves)",
            clouds::DEFAULT_OCTAVES,
            (multi_ms / single_ms.max(1e-6) - 1.0) * 100.0
        );
        println!(
            "CLOUD BRIGHTNESS (beer-powder OFF default): SINGLE-scatter (octaves=1) \
             peak_reflectance={:.4} peak_SUN_reflectance={:.4}  ->  MULTI-scatter (octaves={}) \
             peak_reflectance={:.4} peak_SUN_reflectance={:.4}  (fix2 was ~0.10-0.16 total / \
             ~0.07 sun)",
            single.max_reflectance,
            single.max_sun_reflectance,
            clouds::DEFAULT_OCTAVES,
            multi.max_reflectance,
            multi.max_sun_reflectance
        );
        assert!(
            multi.all_finite && single.all_finite,
            "cloud radiances/transmittances must be finite"
        );
        assert!(multi.sampled > 0, "some in-domain pixels should be sampled");
        assert!(
            multi.cloud_fraction() > 0.0,
            "the Enderlin brick should render a nonzero cloud fraction"
        );
        // Energy plausibility: even the brightest sunlit anvil stays a physical
        // reflectance (<= 1).
        assert!(
            multi.max_reflectance <= 1.0,
            "peak cloud reflectance must stay physical (<= 1): {}",
            multi.max_reflectance
        );
        // The multi-scatter octaves must MATERIALLY brighten the sunlit face over
        // single scatter (the payoff); and the peak must clear the fix2 single-scatter
        // total (~0.156 beer-powder-off). The printed values are the acceptance
        // evidence for the 0.5-0.9 target.
        assert!(
            multi.max_sun_reflectance > single.max_sun_reflectance * 2.0,
            "octaves should multiply the sunlit-face sun term: multi {} vs single {}",
            multi.max_sun_reflectance,
            single.max_sun_reflectance
        );
        assert!(
            multi.max_reflectance > 0.5,
            "the multi-scatter sunlit anvil should read brilliant (order 0.5-0.9, real \
             convective-top reflectance), not the fix2 ~0.10-0.16 grey: {}",
            multi.max_reflectance
        );

        // NATIVE-resolution full-frame render timing (the studio path). Size the
        // raster to the WRF grid — one output pixel per cell — and time the parallel
        // `render_cloud_frame_rgba` the studio actually calls, at Offline (384-step)
        // full quality with the default multi-scatter octaves. This is the honest
        // stored-frame cost the owner accepts at native resolution (native-resolution
        // fix; see notes/res-fix-notes.md).
        {
            use simsat::atmosphere::{AERIAL_FROXEL_DIM, OutputTransform, build_aerial_froxel};
            use simsat::camera::{ResolutionMode, build_surface_raster_mode};
            use simsat::clouds::{render_cloud_frame_rgba, scan_rect_of};
            use simsat::render::{
                FLAT_ALBEDO_SRGB, FrameContext, SurfacePixel, WATER_ALBEDO_SCALE,
            };

            let native = build_surface_raster_mode(
                &camera_g,
                &georef,
                geom.nx,
                geom.ny,
                ResolutionMode::Native,
                0.0,
                MAX_AXIS,
            )
            .expect("domain visible from GOES-East");
            let scan_rect = scan_rect_of(&native.scan);
            let froxel = build_aerial_froxel(
                &luts,
                &params,
                &cam_geo,
                sun_ecef,
                scan_rect,
                AERIAL_FROXEL_DIM,
            );
            let scene_off = CloudScene {
                vol: &vol,
                mip: &mip,
                sun_od: &sun_od,
                georef: &georef,
                luts: &luts,
                sky_sh: &sky_sh,
                sun_ecef,
                cfg: MarchConfig::new(StepQuality::Offline, vol.voxel_pitch_m()),
            };
            let surf = FrameContext {
                luts: &luts,
                params: &params,
                sky_sh: &sky_sh,
                cam: cam_geo,
                sun_ecef,
                output_transform: OutputTransform::AbiReflectance,
                bm_present: false,
                water_scale: WATER_ALBEDO_SCALE as f64,
                flat_albedo_srgb: FLAT_ALBEDO_SRGB as f64,
                raymarch_steps: 16,
                exposure: 1.0,
                ground_day_lift: simsat::render::GROUND_DAY_LIFT,
                cloud_softclip_knee: simsat::render::CLOUD_SOFTCLIP_KNEE,
                cloud_highlight_max: simsat::render::RHO_HIGHLIGHT_MAX,
                synthetic_green: false,
                atmosphere_correction: true,
                terrain_atmosphere: true,
                land_appearance: simsat::render::LandAppearanceConfig::identity(),
            };
            let sun_enu = [0.0f32, e.cos() as f32, e.sin() as f32];
            let rnx = native.nx;
            let assemble = |px: usize, py: usize| -> SurfacePixel {
                let idx = py * rnx + px;
                SurfacePixel {
                    on_earth: native.lat[idx].is_finite(),
                    base_srgb: [0.5, 0.5, 0.5],
                    normal_enu: [0.0, 0.0, 1.0],
                    sun_enu,
                    sun_elev_deg: 50.0,
                    is_water: false,
                    view_dir: [0.0, 0.0, 1.0],
                    ..Default::default()
                }
            };
            let t0 = std::time::Instant::now();
            let rgba = render_cloud_frame_rgba(&scene_off, &surf, &froxel, &native, assemble);
            let wall = t0.elapsed();
            println!(
                "NATIVE RENDER: raster={}x{} ({} px, WRF grid {}x{}) Offline(384) parallel \
                 wall={:.3}s bytes={}",
                native.nx,
                native.ny,
                native.nx * native.ny,
                geom.nx,
                geom.ny,
                wall.as_secs_f64(),
                rgba.len(),
            );
            assert_eq!(rgba.len(), native.nx * native.ny * 4);
        }
    }

    if keep.is_none() {
        std::fs::remove_dir_all(&cache).ok();
    }
}
