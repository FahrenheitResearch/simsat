//! Env-gated real-fixture CACHE-STALENESS test (WS3): prove that "re-run WRF over the
//! same wrfout path, render again" RE-INGESTS instead of silently rendering the OLD
//! cached brick. The re-run is simulated by touching the source file's mtime.
//!
//! Set `SIMSAT_WRF_FIXTURE=<path to a wrfout file>` (the ~85 MB Michael d01 fixture is
//! the intended size; run release-profile, per the fixture_ingest.rs conventions). The
//! fixture is COPIED into a temp dir first — the shared fixture file is never touched.
//! When the variable is unset the test returns cleanly so plain
//! `cargo test --workspace` stays deterministic.

use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use simsat::api::{self, Product, RenderParams};
use simsat::bricks::RunManifest;
use simsat::derived::DerivedField;
use simsat::ingest;

#[test]
fn optional_wrf_fixture_reingests_when_source_mtime_changes() {
    let Ok(fixture) = std::env::var("SIMSAT_WRF_FIXTURE") else {
        eprintln!("SIMSAT_WRF_FIXTURE unset; skipping cache-staleness fixture test");
        return;
    };
    let fixture = PathBuf::from(fixture);
    assert!(
        fixture.is_file(),
        "fixture path does not exist: {fixture:?}"
    );

    let dir = std::env::temp_dir().join(format!("simsat-staleness-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join(fixture.file_name().expect("fixture file name"));
    std::fs::copy(&fixture, &src).expect("copy the fixture (never touch the shared one)");
    let cache = dir.join("cache");

    // A cheap full-api product (per-column integrals; no sun/atmosphere/cloud march,
    // no Blue Marble) — the point is the resolve_wrfout ingest/cache path.
    let mut params = RenderParams::new(src.clone());
    params.cache = cache.clone();
    let product = Product::Derived {
        field: DerivedField::CloudOpticalDepth,
    };

    // 1. First render ingests and records the source identity + anchor.
    let r1 = api::render(&params, product).expect("first render (ingest)");
    assert!(r1.nx > 0 && r1.ny > 0);
    let run_id = ingest::default_run_id(&src);
    let manifest_path = RunManifest::path(&cache, &run_id);
    let m1 = RunManifest::load(&manifest_path).expect("manifest after first render");
    let e1 = m1.timesteps.first().expect("one timestep").clone();
    let mt1 = e1
        .source_mtime_unix
        .expect("source mtime recorded at ingest");
    assert!(e1.source_bytes.is_some(), "source bytes recorded at ingest");
    assert!(
        e1.anchor.is_some(),
        "per-timestep anchor recorded at ingest"
    );
    let brick_path = cache.join(&run_id).join(&e1.file);
    let brick_mtime_1 = std::fs::metadata(&brick_path).unwrap().modified().unwrap();

    // 2. Untouched source -> the second render is a cache HIT: the brick file is NOT
    //    rewritten (its own mtime is unchanged).
    let _ = api::render(&params, product).expect("second render (fresh cache)");
    let brick_mtime_2 = std::fs::metadata(&brick_path).unwrap().modified().unwrap();
    assert_eq!(
        brick_mtime_1, brick_mtime_2,
        "a fresh cache hit must not re-ingest/rewrite the brick"
    );

    // 3. Touch the source mtime (simulating a re-run WRF writing over the same path)
    //    and render again: the manifest's recorded identity MUST move to the new
    //    mtime — which can only happen via a re-ingest.
    let f = std::fs::File::options().write(true).open(&src).unwrap();
    f.set_modified(SystemTime::now() + Duration::from_secs(120))
        .expect("touch the copy's mtime");
    drop(f);
    let touched = std::fs::metadata(&src)
        .unwrap()
        .modified()
        .unwrap()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    assert_ne!(touched, mt1, "the touch must move the mtime");

    let _ = api::render(&params, product).expect("third render (stale cache)");
    let m3 = RunManifest::load(&manifest_path).unwrap();
    let e3 = m3.timesteps.first().unwrap();
    assert_eq!(
        e3.source_mtime_unix,
        Some(touched),
        "a stale cache must RE-INGEST and record the new source mtime (silent reuse of \
         the old brick is the bug this gate closes)"
    );
    let brick_mtime_3 = std::fs::metadata(&brick_path).unwrap().modified().unwrap();
    assert_ne!(
        brick_mtime_1, brick_mtime_3,
        "the stale path must actually rewrite the brick"
    );

    println!(
        "STALENESS FIXTURE: ingest -> fresh hit (no rewrite) -> touched mtime {mt1} -> \
         {touched} -> re-ingest confirmed ({}x{} raster)",
        r1.nx, r1.ny
    );
    std::fs::remove_dir_all(&dir).ok();
}
