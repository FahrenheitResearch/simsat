# Skip unused direct-sun cloud marches in Earth shadow

When the entire finite solar disk is behind Earth at a cloud sample, the existing direct-sun atmospheric transmittance is exactly zero. The secondary cloud-to-sun integral cannot affect that source. Hoist the existing solar-disk calculation and skip only that integral in this case. Sky ambient and every partially visible disk sample keep the previous path. No optical coefficient, opacity threshold, ray spacing, or image default changes. CPU and WGSL implement the same gate.

Verification: 618 Rust tests passed, 2 ignored, including shader validation; release render example builds. Eight before/after small-image fixtures spanning sun elevations -8, -3, 0, +17 degrees and legacy-octaves/delta-flux-v2 produced identical PNG bytes. Additionally the full native 474x380 21Z topdown display image AND its raw f32 RGB plane are byte-identical to the completed four-hour regression. These image fixtures test output stability, not independent ABI spectral accuracy. No controlled performance benchmark is claimed; concurrent wall-clock times cannot establish a speedup.

All measurements retain the old binaries and fresh output directories. The four-hour radiometric scoreboard is unchanged; the optimization does not address the warm IR residual or missing per-band transport.

Latest binary SHA-256: `7b498fa19053f6aa1324f1fd3e6c86b824f5f521d62fe8292c3a6e2b1bf9c0fe`.

21Z png old/new SHA-256: `cf0a170fa3b77709a5236a3c6804ae72fe4f5304f12a67f0ee3a83ee82f5e275`.

21Z bin old/new SHA-256: `33e2ede45e80964af3086552ba65fbf65a82a0974e787bf74ca78b6425dfc7d8`.

Evidence: `work/earth-shadow-tests.log`, `work/earth-shadow-release.log`, `work/earth-shadow-byte-ab/run.json`, `work/satellite-focus-runs/run.json` under the local Codex task directory.
