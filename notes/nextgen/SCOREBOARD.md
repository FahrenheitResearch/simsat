# Verified 04 September 2026 diagnostic case

| UTC | IR clear bias, before -> after K | IR both-cloudy bias K | Visible display-intent clear bias, before -> after | Sensor v2 clear bias |
|---|---:|---:|---:|---:|
| 12 | -2.985 -> -2.985 | +7.597 | +0.0013 -> +0.0013 | +0.0130 |
| 15 | -2.643 -> -2.643 | +9.592 | -0.0289 -> -0.0278 | +0.0248 |
| 18 | -1.123 -> -1.123 | +10.803 | +0.3550 -> +0.0363 | +0.0799 |
| 21 | -1.824 -> -1.824 | +8.959 | -0.0055 -> -0.0051 | +0.0451 |

| UTC | Gray sensor clear bias v1 -> v2 | Gray overall bias v1 -> v2 | Gray correlation v1 -> v2 |
|---|---:|---:|---:|
| 12 | +0.0119 -> +0.0130 | +0.0071 -> +0.0087 | 0.3959 -> 0.3832 |
| 15 | +0.0281 -> +0.0248 | -0.0374 -> -0.0404 | 0.2175 -> 0.2140 |
| 18 | +0.3691 -> +0.0799 | +0.2679 -> -0.0223 | 0.0312 -> 0.1155 |
| 21 | +0.0486 -> +0.0451 | -0.0502 -> -0.0530 | 0.1661 -> 0.1652 |

This is a gray RGB diagnostic, not per-band ABI acceptance. See ../../NOTES-nextgen-abi.md.
