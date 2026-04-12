# HillshadeGenerator — Technical Notes

## Illumination Model

Lambertian model from Wan et al. 2021 (CAS paper) Eq. 3.

```
I(x,y) = L · (p·cos(τ)·cos(σ) + q·sin(τ)·cos(σ) + sin(σ)) · r(x,y)
```

Parameters:
- `τ` — solar azimuth (rad, from North, clockwise). Configurable per mission.
- `σ` — solar elevation above horizon (rad). Configurable per mission.
- `r(x,y)` = 1.0 — uniform Lambertian reflectance assumed for reference generation.

Default parameters (azimuth=315°, elevation=45°) match typical mid-morning
Indian subcontinent conditions.

## Multi-directional Hillshade

Used as the default for TRN reference tiles (`generate_multidirectional()`).
N azimuth directions equally spaced around the compass, averaged.

Mathematical basis: CAS paper §3.2 — the translation information in phase
correlation is in fringe density and orientation, not amplitude.
Multi-directional hillshade reduces amplitude modulation from a single sun
angle, which improves phase correlation robustness across:
- Different mission times of day
- Seasonal variation in sun angle
- Cloud shadow effects

Default N=8 (45° spacing). Increasing N improves illumination invariance
at the cost of processing time. For pre-generated tile libraries, N=16
is recommended.

## Future Extension

When the LWIR sensor is integrated (thermal night mode), replace
HillshadeGenerator with a thermal emission model. Interface is identical —
numpy array elevation in, numpy uint8 array out. The phase correlation
TRN engine does not change.

Candidate replacement: Planck emission model parameterised by surface
emissivity and skin temperature, integrated over LWIR bandpass (8–12 µm).
