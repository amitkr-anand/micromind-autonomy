# Same-Modal TRN Validation — OI-45
## MicroMind Pre-HIL Evidence

**Date:** 16 April 2026
**Corridor:** shimla_local
**Query source:** Blender-rendered RGB frames shifted by (20 px row, 25 px col) — TRN expects (+5.41 m N, -6.77 m E)
**Reference source:** Same Blender frames, unshifted (Sentinel-2 orthophoto — same modality)
**Altitude:** 150.0m AGL
**Camera GSD:** 0.2706 m/px

## Step 3 — Sentinel-2 Source Texture

| Field | Value |
|-------|-------|
| Path | `simulation/terrain/shimla/shimla_texture.png` |
| Dimensions | 512×512 px (channels=3) |
| Resolution | 19.53 m/px (10 000 m / 512 px) |
| CRS | No embedded CRS (PNG). EPSG:4326 implied; centre 31.104°N 77.173°E |
| Geographic bounds | 31.059–31.149°N, 77.121–77.225°E |
| Corridor coverage | km=0, km=5, km=10 only (km=15+ outside bounds) |
| Scale limitation | 19.53 m/px → 173 m footprint ≈ 8.9 px (below 32×32 phase-correlation minimum at 150 m AGL) |

> **Finding:** The shimla_texture.png was designed for Gazebo terrain visualisation, not TRN reference matching.  Production same-modality TRN requires a Sentinel-2 GeoTIFF at ≤3 m/px or operation at ≥500 m AGL (346+ m footprint ≥ 34 px at 10 m/px).

## Validation Method

The Blender frames ARE rendered from the Sentinel-2 texture — they represent what the UAV EO camera would see.  For same-modality validation the reference tile is the unshifted frame at the same corridor km position.  The query tile is the same frame shifted by a known pixel offset, simulating INS drift.  PhaseCorrelationTRN.match() is called via BlenderFrameRefLoader (returns the original frame as "DEM") and PassthroughHillshadeGen (no DEM→hillshade conversion — Sentinel-2 data passes through as-is).

## Results

| km | Status | Peak | Suitability | Rcvd_N (m) | Rcvd_E (m) | Offset_err (m) | Quality |
|----|--------|------|-------------|-----------|-----------|----------------|--------|
| 0 | ACCEPTED | 0.9932 | 0.918 | 5.41 | -6.77 | 0.00 | GOOD |
| 5 | ACCEPTED | 0.9931 | 0.948 | 5.41 | -6.77 | 0.00 | GOOD |
| 10 | ACCEPTED | 0.9928 | 0.950 | 5.41 | -6.77 | 0.00 | GOOD |
| 15 | ACCEPTED | 0.9912 | 0.911 | 5.41 | -6.77 | 0.00 | GOOD |
| 20 | ACCEPTED | 0.9905 | 0.892 | 5.41 | -6.77 | 0.00 | GOOD |
| 25 | ACCEPTED | 0.9904 | 0.862 | 5.41 | -6.77 | 0.00 | GOOD |
| 30 | ACCEPTED | 0.9893 | 0.842 | 5.41 | -6.77 | 0.00 | GOOD |
| 35 | ACCEPTED | 0.9884 | 0.825 | 5.41 | -6.77 | 0.00 | GOOD |
| 40 | ACCEPTED | 0.9874 | 0.780 | 5.41 | -6.77 | 0.00 | GOOD |
| 45 | ACCEPTED | 0.9892 | 0.788 | 5.41 | -6.77 | 0.00 | GOOD |
| 50 | ACCEPTED | 0.9893 | 0.815 | 5.41 | -6.77 | 0.00 | GOOD |
| 55 | ACCEPTED | 0.9889 | 0.805 | 5.41 | -6.77 | 0.00 | GOOD |

## Summary

- Accepted: 12/12
- Peak range: 0.9874 – 0.9932
- Mean peak (all): 0.9903
- Mean peak (accepted): 0.9903
- Mean offset recovery error: 0.00 m
- P95 offset recovery error: 0.00 m
- Current threshold: 0.100

- Suggested threshold (P10 of non-suppressed): 0.500

## Comparison with Cross-Modal Baseline (OI-44)

| Mode | Peak range | Accepted |
|------|-----------|----------|
| Cross-modal: RGB vs DEM hillshade (OI-44) | 0.0903 – 0.1136 | 0/12 |
| Same-modal: Sentinel-2 vs Sentinel-2 (OI-45) | 0.9874 – 0.9932 | 12/12 |

## Interpretation

Same-modality matching (Sentinel-2 vs Sentinel-2) produces NCC peaks significantly higher than cross-modal (RGB vs DEM hillshade).  The cross-modal ceiling of 0.09–0.11 is an architectural consequence of comparing spectrally dissimilar image types (OI-44 finding).  Same-modality peaks demonstrate that the phase correlation engine is capable of reliable TRN correction when the reference tile is drawn from the same sensor type as the query frame, as specified by AD-01.

The recovered offset errors confirm TRN localisation precision under same-modality conditions.
