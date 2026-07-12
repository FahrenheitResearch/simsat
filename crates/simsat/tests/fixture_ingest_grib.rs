//! Env-gated real-fixture GRIB2 ingest test (the SIMSAT_WRF_FIXTURE precedent,
//! GRIB edition).
//!
//! Set `SIMSAT_GRIB_FIXTURE=<dir>` to a directory holding the staged operational
//! fixtures from the private build-node cache; the tests ingest the
//! HRRR native-level file end to end and the RRFS rotated-grid file through the
//! crop path (asserting the full-NA and full-CONUS refusals first), assert the
//! memory/geometry contracts, ratchet the projections against grib-core's own
//! per-cell lat/lon, and sanity-check the brick content. Missing fixture files
//! are skipped individually. When the variable is unset the tests return
//! cleanly. Run release-profile — the HRRR grid is 1799x1059x50 and the RRFS
//! crop ~1300x1200x65:
//!   SIMSAT_GRIB_FIXTURE=... cargo test -p simsat --release fixture_grib -- --nocapture

use std::path::{Path, PathBuf};

use simsat::bricks::read_ssb;
use simsat::frame::MAP_PROJ_ROTATED_LATLON;
use simsat::ingest::IngestConfig;
use simsat::ingest_grib::{
    GribIngestOptions, ingest_grib_timestep, ingest_grib_timestep_with, is_grib_input, parse_crop,
    probe_grib, read_grib_geometry, read_grib_geometry_with,
};

/// Hard peak-RSS ceiling from the design contract (the wrfout fixture's limit).
const PEAK_RSS_LIMIT_BYTES: u64 = 2_500_000_000;

/// The staged HRRR fixture file name (cycle 2026-07-09 20z f00; see
/// notes/grib-ingest-notes.md for provenance + exact byte size).
const HRRR_FIXTURE: &str = "hrrr.t20z.wrfnatf00.grib2";

/// The staged RRFS native-level fixture (same cycle, f000; 9.29 GB, rotated
/// lat-lon 4881x2961x65 — see notes/grib-ingest-notes.md).
const RRFS_FIXTURE: &str = "rrfs.t20z.natlev.3km.f000.na.grib2";

/// The in-budget RRFS QA crop (central/eastern CONUS). A FULL-CONUS box hulls
/// to ~3.7M columns on the tilted rotated grid — over the 2.5 GB contract by
/// the brick write phase alone — so the admission gate refuses it (asserted
/// below) and QA uses this regional box.
const RRFS_QA_CROP: &str = "25,49.5,-110,-78";

fn fixture_dir() -> Option<PathBuf> {
    let dir = std::env::var("SIMSAT_GRIB_FIXTURE").ok()?;
    let dir = PathBuf::from(dir);
    assert!(dir.is_dir(), "fixture dir does not exist: {dir:?}");
    Some(dir)
}

#[test]
fn optional_grib_fixture_ingests_and_ratchets() {
    let Some(dir) = fixture_dir() else {
        eprintln!("SIMSAT_GRIB_FIXTURE unset; skipping real-fixture GRIB ingest test");
        return;
    };
    let hrrr = dir.join(HRRR_FIXTURE);
    if !hrrr.is_file() {
        eprintln!("HRRR fixture missing ({hrrr:?}); skipping");
        return;
    }
    ingest_and_check(&hrrr);
}

#[test]
fn optional_rrfs_fixture_crop_ingests_and_ratchets() {
    let Some(dir) = fixture_dir() else {
        eprintln!("SIMSAT_GRIB_FIXTURE unset; skipping RRFS fixture test");
        return;
    };
    let rrfs = dir.join(RRFS_FIXTURE);
    if !rrfs.is_file() {
        eprintln!("RRFS fixture missing ({rrfs:?}); skipping");
        return;
    }

    // Probe: the file's own full rotated grid.
    let probe = probe_grib(&rrfs).expect("rrfs probe");
    println!(
        "RRFS PROBE: {}x{}x{} messages={} valid={} run_id={}",
        probe.nx, probe.ny, probe.nz, probe.messages, probe.time_iso, probe.default_run_id
    );
    assert_eq!((probe.nx, probe.ny, probe.nz), (4881, 2961, 65));
    assert_eq!(probe.messages, 1631);

    let cache = std::env::var("SIMSAT_CACHE_DIR")
        .ok()
        .filter(|d| !d.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("simsat-rrfs-fixture-{}", std::process::id()))
        });
    let config = IngestConfig::new(cache);

    // The full-NA refusal (no crop): remedy names the crop option.
    let err = ingest_grib_timestep(&rrfs, &config).expect_err("full NA must be refused");
    println!("RRFS NO-CROP REFUSAL: {err}");
    assert!(
        err.to_string().contains("crop"),
        "refusal must name the remedy"
    );

    // A FULL-CONUS box on the tilted rotated grid busts the RSS budget: the
    // admission gate refuses it BEFORE allocating (the honest full-CONUS story).
    let conus = GribIngestOptions {
        crop: Some(parse_crop("conus").unwrap()),
    };
    let err = ingest_grib_timestep_with(&rrfs, &config, &conus)
        .expect_err("full-CONUS crop must be refused by the RSS admission gate");
    println!("RRFS CONUS-CROP REFUSAL: {err}");
    assert!(err.to_string().contains("estimated ingest peak"), "{err}");

    // The in-budget regional crop ingests end to end.
    let options = GribIngestOptions {
        crop: Some(parse_crop(RRFS_QA_CROP).unwrap()),
    };
    let report =
        ingest_grib_timestep_with(&rrfs, &config, &options).expect("rrfs crop ingest completes");
    println!(
        "RRFS FIXTURE INGEST: run={} dims={}x{}x{} hhmm={:04} wall={:.3}s peak_rss={} \
         ssb_bytes={} ({:.2} MB)",
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
    assert!(report.ssb_bytes > 0);
    assert!(report.brick_path.is_file());
    if let Some(rss) = report.peak_rss_bytes {
        assert!(
            rss < PEAK_RSS_LIMIT_BYTES,
            "peak RSS {rss} bytes exceeds the 2.5 GB contract"
        );
    }

    // Geometry + the rotated-projection ratchet on the cropped sub-grid.
    let geom = read_grib_geometry_with(&rrfs, &options).expect("rrfs geometry");
    assert_eq!((report.nx, report.ny), (geom.nx, geom.ny));
    assert_eq!(geom.params.map_proj, MAP_PROJ_ROTATED_LATLON);
    let georef = geom.georef().expect("rotated georef");
    let step = 5usize;
    let mut worst = 0.0f64;
    for j in (0..geom.ny).step_by(step) {
        for i in (0..geom.nx).step_by(step) {
            let idx = j * geom.nx + i;
            let (fi, fj) = georef.forward(geom.xlat[idx] as f64, geom.xlong[idx] as f64);
            worst = worst.max((fi - i as f64).abs()).max((fj - j as f64).abs());
        }
    }
    println!("RRFS PROJECTION RATCHET: worst error = {worst:.5} cells (limit 0.05)");
    assert!(worst < 0.05, "rotated ratchet worst {worst} cells");

    // Brick content sanity (July-afternoon central/eastern CONUS).
    let brick = read_ssb(&report.brick_path).expect("read rrfs brick back");
    assert_eq!((brick.nx, brick.ny), (geom.nx, geom.ny));
    let liquid = brick.quant.get("ext_liquid");
    let ice = brick.quant.get("ext_ice");
    let snow = brick.quant.get("ext_snow");
    let precip = brick.quant.get("ext_precip");
    let qv = brick.quant.get("qvapor");
    println!(
        "RRFS BRICK QUANT: liquid vmax={:.3e} ice vmax={:.3e} snow vmax={:.3e} \
         precip vmax={:.3e} qvapor vmax={:.3e}",
        liquid.vmax, ice.vmax, snow.vmax, precip.vmax, qv.vmax
    );
    assert!(liquid.vmax > 0.0 && ice.vmax > 0.0 && snow.vmax > 0.0 && precip.vmax > 0.0);
    assert!(brick.ext_snow.iter().any(|&v| v > 0));
    assert!(!brick.has_cloud_fraction);
    assert!(brick.cloud_fraction.iter().all(|&v| v == 255));
    assert!(qv.vmax > 1.0e-3);
    let (mut sum, mut n) = (0f64, 0u64);
    for &t in &brick.tsk {
        if t > 0.0 {
            sum += t as f64;
            n += 1;
        }
    }
    assert!(n > 0, "TSK should be populated");
    let tsk_mean = sum / n as f64;
    println!("RRFS BRICK TSK: mean {tsk_mean:.1} K over {n} cells");
    assert!((250.0..=340.0).contains(&tsk_mean));
    let land_frac =
        brick.landmask.iter().filter(|&&v| v > 0.5).count() as f64 / brick.landmask.len() as f64;
    println!("RRFS BRICK LAND FRACTION: {land_frac:.3}");
    assert!((0.3..=0.95).contains(&land_frac));
}

fn ingest_and_check(path: &Path) {
    assert!(is_grib_input(path), "fixture should classify as GRIB");

    // The metadata probe (the api/studio seam) agrees with the file's headers.
    let probe = probe_grib(path).expect("grib probe");
    println!(
        "GRIB PROBE: {}x{}x{} messages={} valid={} ref={} run_id={}",
        probe.nx,
        probe.ny,
        probe.nz,
        probe.messages,
        probe.time_iso,
        probe.reference_iso,
        probe.default_run_id
    );
    assert_eq!((probe.nx, probe.ny, probe.nz), (1799, 1059, 50));
    assert_eq!(probe.messages, 1133);
    assert!(
        probe
            .default_run_id
            .starts_with("hrrr_t20z_wrfnatf00_grib2_")
    );

    // Persist the brick when SIMSAT_CACHE_DIR is set (so the render harnesses can
    // consume the same run.json); otherwise a throwaway temp dir.
    let keep = std::env::var("SIMSAT_CACHE_DIR")
        .ok()
        .filter(|d| !d.trim().is_empty());
    let cache = match &keep {
        Some(dir) => PathBuf::from(dir),
        None => std::env::temp_dir().join(format!("simsat-grib-fixture-{}", std::process::id())),
    };
    let config = IngestConfig::new(cache.clone());

    let report = ingest_grib_timestep(path, &config).expect("grib ingest should complete");
    println!(
        "GRIB FIXTURE INGEST: run={} dims={}x{}x{} hhmm={:04} wall={:.3}s peak_rss={} \
         ssb_bytes={} ({:.2} MB)",
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

    // Geometry consistent with the file's own headers.
    let geom = read_grib_geometry(path).expect("grib geometry");
    assert_eq!(report.nx, geom.nx);
    assert_eq!(report.ny, geom.ny);
    assert_eq!(report.nz, config.nz_brick);
    // The staged HRRR CONUS fixture is the canonical 1799x1059x50 grid.
    assert_eq!((geom.nx, geom.ny, geom.nz), (1799, 1059, 50));

    // Projection ratchet: every Nth grib-core-computed cell coordinate must
    // project back onto its own (i, j) through OUR georef (the dx earth-radius
    // rescale correctness gate; same 0.05-cell limit as the wrfout fixture,
    // expected ~1e-3).
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
    println!("GRIB PROJECTION RATCHET: worst error = {worst:.5} cells (limit 0.05)");
    assert!(worst < 0.05, "ratchet worst {worst} cells exceeds 0.05");

    // Brick content sanity: reads back, dims match, and a July-afternoon CONUS
    // analysis must carry cloud somewhere + plausible surface fields.
    let brick = read_ssb(&report.brick_path).expect("read brick back");
    assert_eq!(
        (brick.nx, brick.ny, brick.nz),
        (geom.nx, geom.ny, config.nz_brick)
    );
    assert_eq!(brick.hgt.len(), geom.nx * geom.ny);

    let liquid_quant = brick.quant.get("ext_liquid");
    let ice_quant = brick.quant.get("ext_ice");
    let snow_quant = brick.quant.get("ext_snow");
    let precip_quant = brick.quant.get("ext_precip");
    let qv_quant = brick.quant.get("qvapor");
    println!(
        "GRIB BRICK QUANT: liquid vmax={:.3e} ice vmax={:.3e} snow vmax={:.3e} \
         precip vmax={:.3e} qvapor vmax={:.3e}",
        liquid_quant.vmax, ice_quant.vmax, snow_quant.vmax, precip_quant.vmax, qv_quant.vmax
    );
    assert!(
        liquid_quant.vmax > 0.0,
        "CONUS afternoon should have liquid cloud"
    );
    assert!(
        ice_quant.vmax > 0.0,
        "CONUS afternoon should have ice cloud"
    );
    assert!(
        precip_quant.vmax > 0.0,
        "CONUS afternoon should have precip"
    );
    assert!(qv_quant.vmax > 1.0e-3, "qvapor should be moist somewhere");
    assert!(
        snow_quant.vmax > 0.0,
        "HRRR SNMR should populate the SSB v6 snow auxiliary"
    );
    assert!(brick.ext_snow.iter().any(|&c| c > 0));
    assert!(
        brick.has_cloud_fraction,
        "HRRR wrfnat carries complete cc (0/6/32) hybrid levels"
    );
    assert!(brick.cloud_fraction.iter().any(|&c| c < 255));
    assert!(brick.cloud_fraction.iter().any(|&c| c > 0));
    assert!(brick.ext_liquid.iter().any(|&c| c > 0));
    assert!(brick.tau_up.iter().any(|&c| c > 0));

    // Terrain: the CONUS grid spans below-sea-level (Death Valley) to the Rockies.
    let hgt_max = brick.hgt.iter().copied().fold(f32::MIN, f32::max);
    let hgt_min = brick.hgt.iter().copied().fold(f32::MAX, f32::min);
    println!("GRIB BRICK TERRAIN: hgt {hgt_min:.0}..{hgt_max:.0} m");
    assert!(hgt_max > 2500.0, "Rockies should exceed 2500 m");
    assert!(hgt_min < 100.0, "coastal terrain should be near sea level");

    // Skin temperature: mean over non-zero cells in a plausible July range.
    let (mut sum, mut n) = (0f64, 0u64);
    for &t in &brick.tsk {
        if t > 0.0 {
            sum += t as f64;
            n += 1;
        }
    }
    assert!(n > 0, "TSK should be populated");
    let tsk_mean = sum / n as f64;
    println!("GRIB BRICK TSK: mean {tsk_mean:.1} K over {n} cells");
    assert!(
        (250.0..=340.0).contains(&tsk_mean),
        "TSK mean {tsk_mean} K implausible"
    );

    // Land mask present (CONUS is ~63% land on this grid).
    let land_frac =
        brick.landmask.iter().filter(|&&v| v > 0.5).count() as f64 / brick.landmask.len() as f64;
    println!("GRIB BRICK LAND FRACTION: {land_frac:.3}");
    assert!(
        (0.3..=0.9).contains(&land_frac),
        "land fraction {land_frac} implausible"
    );

    if keep.is_none() {
        std::fs::remove_dir_all(&cache).ok();
    }
}
