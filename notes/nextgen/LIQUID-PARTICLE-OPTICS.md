# Spectral liquid-droplet scattering

The experimental ABI operator now has a homogeneous-sphere solver and a size-distribution-to-model-mass interface. These components do not change the current display or IR renderer yet. Full scene integration, nonspherical ice optics, gas absorption, surface coupling, spatial sampling and multiple scattering remain required.

## Physical contract

The sphere solver accepts relative refractive index n + i k with nonnegative absorption, size parameter x = 2 pi n_medium r / lambda_vacuum in [1, 2048], real index [1, 2] and imaginary index [0, 0.1]. Its downward logarithmic derivative and Miller Riccati-Bessel recurrence use 128 additional orders. Efficiencies, absorption, asymmetry and unpolarized angular phase are calculated from the same complex coefficients. Phase has units sr^-1 and integrates to one over 4 pi. Index-matched particles have zero scattering and no normalized phase.

The mathematical formulation follows Bohren and Huffman (1983), also described in [Scott Prahl's algorithm documentation](https://miepython.readthedocs.io/en/latest/07_algorithm.html). The implementation is local Rust; no third-party scattering source was copied into the engine.

LiquidPopulationOptics consumes explicit radii and number weights, including quadrature/bin widths where appropriate. It reports the cross section divided by particle mass, effective radius M3/M2, effective variance M4 M2/M3^2 - 1, and scattering-weighted phase/asymmetry. Multiplication by model liquid water mass density in kg m^-3 yields extinction/scattering/absorption in m^-1. It does not apply display opacity, exposure or a default particle-size distribution. The moment convention is described by [Hansen and Travis (1974)](https://www.giss.nasa.gov/pubs/abs/ha09500o.html).

Pure-water material indices come from the previously pinned reference-condition Segelstein table. The relative index and size parameter both include the explicitly supplied real medium index. Liquid density is 1000 kg m^-3. Temperature-dependent supercooled-water properties and nonspherical ice are not implied by this model.

## Independent evidence

The unchanged BHMIE supplied with official libRadtran 2.0.6 was compiled externally with local LLVM Flang. Its public input/output and angular amplitudes are f32. Requests were quantized identically before the Rust calculation. A separate SciPy calculation evaluates spherical_jn and spherical_yn directly; it does not use the Rust recurrence or coefficients.

The broad comparison contains 68 spheres and 13 angles each: five visible water wavelengths at eight radii, plus weak-index, conservative and absorbing generic spheres up to x = 2048.

- All SciPy comparisons pass the declared tolerances: efficiencies/asymmetry 1e-12 + 2e-9 relative; phase 1e-10 sr^-1 + 1e-7 relative.
- The older BHMIE comparison fails 337 checks: 283 angular values and 18 each of extinction, scattering and asymmetry. Those failures are retained. Its largest efficiency relative difference is about 0.14%; individual angular differences can be much larger. This is not reported as an all-reference pass, and no vendor code or tolerances were changed to force agreement.
- The independent SciPy values are checked in as a Rust regression fixture, including all unfavorable BHMIE cases.
- Independent angular quadrature verifies phase normalization and the first moment. Conservation and 128/192-order convergence tests cover the large-particle regime.

An initial ordinary f32 angular recurrence had large near-forward/backward errors. The preserved failed GPU audit documents that. The corrected CPU and WGSL recurrences track the difference between neighboring angular polynomials near |cos(theta)| = 1 and use parity for negative cosines.

Actual RTX 3080 / Vulkan tests:

| Component | Cases | Largest relative discrepancy | Outcome |
|---|---:|---:|---|
| Individual sphere phase | 4575 | 2.617e-5 for phase > 1e-7 sr^-1 | pass |
| Population phase | 3660 | 7.670e-6 | pass |
| Population mass conversion | 3660 | 8.155e-8 | pass |

Both backends consume the same host-prepared f64 particle coefficients and population weights. GPU evaluation uses f32 angular sums and mass conversion; this is not an independent GPU Bessel solver.

## Reproduction

Run cargo run -p simsat --example audit_mie_sphere --locked --offline -j6 to produce the sphere request JSON. Build scripts/reference/simsat_bhmie_reference.f90 against the unchanged libsrc_f/bhmie.f from [official libRadtran 2.0.6](https://www.libradtran.org/doku.php?id=download). Run scripts/reference/audit_mie_sphere.py --simulated <requests.json> --bhmie-exe <exe> --bhmie-source <bhmie.f> --bhmie-wrapper <wrapper.f90> --output-dir <new-directory>. Dependencies are NumPy and SciPy. It exits nonzero when any declared reference tolerance fails; the retained older-BHMIE disagreements therefore produce a nonzero result.

Actual GPU examples: audit_mie_sphere_gpu and audit_liquid_population_gpu. Unit tests are in mie_sphere.rs and liquid_population.rs. Source/archive/executable hashes, input transcript, compact reference CSVs and GPU reports are adjacent to this note. The historical compiled Fortran wrapper hash refers to the scratch CRLF file; its checked-in LF copy has the same program text.

Current best imagery remains in the curated Downloads folder. These kernels are not represented as a new improved satellite image.


Checkpoint verification: 711 workspace Rust tests pass, 2 ignored; formatting, strict all-target lint and complete workspace/GUI build pass. Cargo jobs remains 6. No existing renderer behavior changed.
