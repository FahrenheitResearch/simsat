# Releasing SimSat

Preview v0.3.0-rc.1 is a Windows Studio release candidate. Source is tagged on the reviewed release branch; prereleases do not replace the stable v0.2.1 release.

The v0.3.0-rc.1 Windows release compiler is Rust 1.94.0. CI pins the same toolchain for workspace checks and native builds; the Python wheel smoke remains a separate portability check. Rust 1.98 introduced additional Clippy style warnings, so following an unpinned stable channel did not reproduce the reviewed gate. Upgrade the release compiler in a separate reviewed change.

Run from the repository root so `.cargo/config.toml` applies the static MSVC runtime. Keep six build jobs.

```
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -j 6 -- -D warnings
cargo test --workspace --locked -j 6
cargo build --release --workspace --bins --locked -j 6
```

Ship the normal `release` profile, never `release-fast`. Update the root workspace version and both manifests/lockfiles of the standalone Python package together. Verify the PE product version and imported DLLs, hash the final EXE, and test that same file through Studio's case render and PNG export. Test visible measured land, ordinary daytime land, and night IR. Keep the tested bytes unchanged when uploading.

Studio can run an exact repeatable case through its normal window/render/export pipeline:

```
simsat_studio --case case.json --render-and-save output.png
```

This explicit command starts ingestion/rendering, saves PNG, exits, and does not change the interactive settings profile. It requires an available desktop/GPU for the Studio window. It is not a headless replacement for the engine CLI. Nonzero exit means failure. Open cases interactively with `--case case.json` or Open > Test cases. See `docs/STUDIO-CASES.md`.

The manual release-artifact workflow builds Windows, Linux and a Python wheel without publishing. Dispatch CI and release artifacts on the exact candidate branch and inspect results. Publish only on an explicit owner release request; use a GitHub prerelease for RC tags. Attach the exact tested Windows EXE, portable Windows zip, checksums, notes, and the separate optional Earth-map pack. Keep local model files and personal case manifests out of public assets.

Regenerate the target runtime license inventory using cargo-about and preserve the existing special-case/font copyright appendices. The dependency lock versions in this candidate match v0.2.1 except SimSat's own package versions. The checked-in notices include the regenerated Windows inventory.

Earth-map pack: NASA BMNG 2004 Base Map, twelve `world.2004MM.3x21600x10800.jpg` files plus source URLs/SHA-256 manifest. Extract its `earth-basemap` directory beside the EXE. Cite NASA Earth Observatory; do not confuse these unshaded maps with the legacy topographically shaded pack. MODIS NBAR must be prepared for the exact source WRF grid/date and is an optional measured RGB display proxy, not an ABI spectral product. Case-specific NBAR/model inputs are separate from the general public package.
