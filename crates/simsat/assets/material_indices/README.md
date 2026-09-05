# Visible material refractive indices

These are reference material inputs for the ABI C01/C02/C03 operator. They are not cloud phase functions or a complete renderer. All native tabulation knots bracketing 0.4-1.0 micrometres are retained. Wavelength is in micrometres, refractive index is dimensionless, and the imaginary index is positive for the absorbing convention `m = n + i*k` with `exp(i*(k_wave*z - omega*t))`.

## Liquid water

`water-segelstein-1981-visible.txt` contains 121 rows from the Segelstein (1981) data tabulated in the official libRadtran 2.0.6 `libsrc_f/REFWAT.f`. It brackets the public 0.4-1.0 um contract with source knots 0.39994 through 1.0 um. This is pure liquid water, not seawater. The tabulation is a fixed reference spectrum; the source routine does not include temperature dependence below 10 um. It must not be represented as a temperature-resolved supercooled-liquid optical model.

Source: [official libRadtran distribution](https://libradtran.org/doku.php?id=download), archive `https://www.libradtran.org/download/libRadtran-2.0.6.tar.gz`, 154147176 bytes, SHA-256 `64930cc40b6e4a37aa220520974d330fc1563796f466a649b2238131f2d69840`. The selected source member's SHA-256 is `f80caefb0a926a2e258ca9761ba5f07f8e83867f032b0d033b57a226959d5a5f`. Original citation: D. Segelstein, *The Complex Refractive Index of Water*, M.S. thesis, University of Missouri-Kansas City, 1981. Only numerical tabulation data are extracted; vendor routines are not linked into SimSat.

## Ice Ih

`ice-ih-warren-brandt-2008-visible.txt` contains all 61 native knots from 0.4 through 1.0 um in the official [Warren and Brandt (2008) data](https://www.atmos.washington.edu/ice_optical_constants/). Source ASCII: `https://www.atmos.washington.edu/ice_optical_constants/IOP_2008_ASCIItable.dat`, 16514 bytes, SHA-256 `891d0e0690cc6c6ed2dfe81291fa20a3af177fa85ef93127c58c5f86d3c370af`. Citation: Warren, S. G., and R. E. Brandt (2008), *Optical constants of ice from the ultraviolet to the microwave: A revised compilation*, JGR 113 D14220, doi:10.1029/2007JD009744.

These are the reference optical constants of the material ice Ih. They do not specify crystal habit, roughness, orientation, particle-size distribution, or a temperature-resolved visible spectrum. A sphere solver using them would remain an ice-sphere approximation; it cannot be described as a realistic ice-crystal scattering model.

## Interpolation and reproduction

Both sources prescribe linear interpolation of real index in log wavelength and linear interpolation of log imaginary index in log wavelength. The Rust API enforces finite wavelengths in 0.4..=1.0 um and never extrapolates. Exact source knots return their stored values. The WGSL kernel consumes the same pair of adjacent native knots selected by the host and performs the same interpolation. There are no RGB index constants or independent GPU fits.

From the repository root, with the original files downloaded separately:

```text
python scripts/simsat-prepare-material-indices.py --water-fortran /path/to/REFWAT.f --ice-ascii /path/to/IOP_2008_ASCIItable.dat --output-dir /new/output/directory
```

Preparation checks both complete input hashes before extracting data; it does not execute Fortran. It rejects overlapping/missing arrays, nonpositive or unordered values, and source coverage insufficient to bracket the target range. Reproduction produces byte-identical material text files with canonical LF line endings; the repository .gitattributes rule preserves their pinned hashes across operating systems. The accompanying manifest records original provenance and artifact hashes.

Rust tests pin asset hashes, recover all source knots, verify geometric-midpoint interpolation, and exercise every official C01/C02/C03 response node plus rejected invalid wavelengths. Four component tests pass. The actual RTX 3080/Vulkan audit runs 2402 wavelength/material cases; maximum CPU/WGSL relative discrepancy is 8.652e-8 in real index and 9.882e-6 in imaginary index against a declared 2e-5 tolerance. This measures numerical interpolation agreement, not laboratory-data uncertainty or full-renderer equivalence. Audit result: `notes/nextgen/material-index-gpu-audit.json`.

Mie size-distribution integration for liquid cloud, nonspherical ice optics, native model size/phase mapping, gas absorption, and full spectral multiple scattering remain required. These new inputs do not change existing image defaults.
