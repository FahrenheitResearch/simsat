# Post-lighting terrain recovery controls

The broad post-light surface toe remains a default-off finished-display experiment.
The separately gated twilight recovery has completed owner cross-case review and is now
part of the shipped finished-visible preset.

The operator is applied only to the LAND surface contribution after direct and ambient
lighting and camera/view transmittance, but before additive atmospheric airlight and the
cloud composite. Ocean body colour, Fresnel sky reflection, Cox-Munk glint, cloud
radiance, raw RGB reflectance, thermal and derived products are unchanged.

The attenuated surface radiance is converted to solar-normalized linear reflectance
factor. Its luminance uses the existing smooth, colour-preserving dark-toe formulation:
black and values at/above the knee are identity, gamma 1 and max gain 1 are identity,
and the gain is bounded. The existing daylight ramp keeps the horizon/night exact.

Controls and safe bounds:

- `surface-postlight-toe` / `surface_postlight_toe`: default `off` / `False`
- knee: default `0.18`, bound `1e-6..=1.0`
- gamma: default `0.80`, bound `0.05..=1.0`
- max gain: default `1.35`, bound `1.0..=4.0`

The first validation grid is knee `0.15/0.18/0.22`, gamma `0.75/0.80/0.85`, and max
gain `1.25/1.35/1.45`. These are experiment coordinates, not calibration claims.

The Rust `RenderParams`, `SurfacePostlightToeConfig`, headless `render_frame` CLI,
Python visible-family functions, and persisted Studio controls carry the same state.
Sensor Fast Gray forces the operator off and records an intent substitution. CPU and
GPU visible paths share the same sanitized controls, land-only placement, and bounded
math, so GPU preview preserves an enabled request without a compatibility substitution.

A separate default-on-for-visible `twilight-surface-recovery` /
`twilight_surface_recovery` control uses the owner-selected D controls
(`0.30 / 0.50 / 4.0`) and a tight gate:
fade in from -6 to 0 degrees, remain full through +4, and fade to exact identity at +12.
When both experiments are enabled, their gains combine by maximum rather than multiply.
The switch and all three parameters remain exposed in Rust, the CLI, Python, and Studio.
Sensor QA, raw reflectance, thermal, derived, and cloud-only products force identity.
