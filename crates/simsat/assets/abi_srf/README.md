# GOES-R ABI FM4 spectral response assets

These are the official NOAA/NESDIS Calibration Working Group (CWG) pre-launch
GOES-R ABI FM4 (GOES-19) spectral response functions. The three reflective-band
files support the response-only registry in src/visible_sensor.rs. The existing
channel-13 asset remains owned by src/thermal_sensor.rs.

- [NOAA release page](https://ncc.nesdis.noaa.gov/GOESR/ABI.php)
- [Exact NOAA FM4 archive](https://ncc.nesdis.noaa.gov/GOESR/docs/GOES-R_ABI_FM4_SRF_CWG.zip)
- [CWG release description, reproduced by EUMETSAT NWP SAF](https://nwp-saf.eumetsat.int/downloads/rtcoef_info/visir_srf/rtcoef_goes_19_abi_srf.html)
- [NOAA CMIP ATBD v4, section 3.4.1.2](https://www.star.nesdis.noaa.gov/goesr/documents/ATBDs/Enterprise/ATBD_Enterprise_Cloud_and_Moisture_Imagery_Product_v4_2021-01-13.pdf)

Release date: 2016-03-10. Reflective assets retrieved: 2026-09-05 UTC.
Archive SHA-256:
B1482058CD63481F55E523C565A982BB2787F01088960E2833A1E4BF6286DD17

The archive hash matches the previously recorded thermal-registry archive.
The only transformation was CRLF to LF. No wavelength, response, header,
precision, ordering, or support limits were changed. The original files use
channel numbers without zero padding. CWG had already truncated the response
to the innermost 0.1% limits; this repository applies no additional truncation.

| File | Original NOAA entry SHA-256 | Vendored LF SHA-256 |
|---|---|---|
| GOES-R_ABI_FM4_SRF_CWG_ch1.txt | 940A9CB586E96BEE0B079CC3149D1AF6D9C3252B35E24D2FC7DB9BB3CD7C88DB | 8076BA1B487B706574B55158E02A69A9A7E6B457C6EC9D20C92180B5B32C47FE |
| GOES-R_ABI_FM4_SRF_CWG_ch2.txt | 800A98FE0AA63253883F9CCDD6BA6117594A91C26DF4743E0D8A7D5579C3A24F | 7B025F14CB49FD02E898F0AF3DC57155211F87A8FC3A41E27F1462A1C0A53DE6 |
| GOES-R_ABI_FM4_SRF_CWG_ch3.txt | 5D15E53230BFF18F6F8F8C45B8885468254D66CAE50BBF0573D8330C8E09E4E3 | BA962645BDB8CA434C8C474AE7D5D6CF0BB337FD96E0A21E55E7CA9B9F4E8B7E |

The module's tests hash the vendored bytes and reconstruct the original CRLF
bytes to verify both hashes. The source has two comment lines followed by
wavelength in micrometres, wavenumber in inverse centimetres, and dimensionless
relative response.

## Radiometric contract

The new module integrates the full tabulated response with trapezoidal
quadrature over wavelength in micrometres. For a spectral radiance L_lambda
in W m^-2 sr^-1 um^-1 and a solar spectral irradiance E_lambda in
W m^-2 um^-1 at the observation's Earth-Sun distance:

    rho_f = pi * integral(R * L_lambda * d_lambda)
                 / integral(R * E_lambda * d_lambda)

Equivalently, L_band = integral(R * L_lambda)/integral(R), with the same
definition for E_band. The arbitrary response normalization cancels.

NOAA CMI stores the reflectance factor rho_f, with no solar-zenith division.
A Lambertian surface without atmosphere gives rho_f = albedo * cos(SZA).
The solar irradiance input is normal to the solar beam, before local incidence
projection. The registry does not clip values above one, including directional
glint, or signed finite radiances.

If solar irradiance is instead tabulated at 1 AU, use the explicit
reflectance_factor_from_1au method with observation-time radiance and distance d
in AU. It applies E_observation = E_1au/d^2, matching
rho_f = pi*d^2*L_band/E_1au_band. No date or ephemeris is inferred.
Invalid/nonfinite inputs and zero in-band solar irradiance return errors.

Spectral densities per inverse centimetre require the coordinate Jacobian:
lambda_um = 10000/nu_cm1 and abs(d_lambda/d_nu) = 10000/nu_cm1^2.
Thermal wavenumber quadrature cannot be reused unchanged for solar per-um
radiance. An independent Jacobian test compares both quadrature coordinates,
allowing for NOAA's rounded wavelength column and discretization error.

For comparison with a particular NOAA acquisition, use its radiometric
metadata and a compatible calibrated solar spectrum. The SRF does not itself
provide solar irradiance. Substituting a different solar reference spectrum
requires recorded provenance and a calibration-consistency check.

## Missing transport inputs: this is not a rendered ABI product

No render intent, CLI product, existing RGB transport, WGSL code, or GUI is
changed. This CPU response-integration utility has no GPU transport twin.

A true ABI renderer still needs:

- Calibrated spectral surface albedo/BRDF, including the vegetation near-IR
  response sampled by C03. Blue Marble display sRGB cannot supply it.
- Wavelength-dependent atmospheric Rayleigh, aerosol and gas optics, including
  pressure, water-vapor and ozone treatment that is independent of display
  stylization.
- Defensible spectral hydrometeor extinction, single-scatter albedo and phase
  functions, with an explicit cloud-fraction/overlap treatment.
- A calibrated solar irradiance spectrum and radiative transport that supplies
  TOA L_lambda at the quadrature wavelengths.
- Actual satellite view geometry and documented spatial/temporal sampling,
  followed by per-band comparison with aligned C01/C02/C03 and appropriate
  validity/cloud/glint regimes.

The existing broad RGB renderer, its source illumination, and its output
channels must not be interpolated or relabeled as these ABI bands.
