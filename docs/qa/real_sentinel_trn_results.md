# Real Sentinel-2 TRN Validation — OI-46
## MicroMind Pre-HIL Evidence

**Date:** 16 April 2026
**Corridor:** shimla_local
**Query:** Blender-rendered RGB frames (shimla_texture.png — DEM hillshade-colour from OpenTopography)
**Reference:** Sentinel-2 TCI `T43RGQ_20251017T053241_TCI_10m.jp2`  
  CRS: EPSG:32643 (UTM 43N), 10 m/px, uint8 RGB  
  Scene date: 17 October 2025  
  WGS84: 77.086–78.264°E, 30.605–31.618°N
**Altitude:** 150.0m AGL
**TRN GSD:** 5.0 m/px
**Ref tile:** 34×34 px (173m footprint)

## Results

| km | Status | Peak | Suitability | Rec | RefTexVar | RefRelief | CorrMag (m) | Query |
|----|--------|------|-------------|-----|----------|-----------|-------------|-------|
| 0 | ACCEPTED | 0.1041 | 0.567 | CAUTION | 553.3 | 136.5 | 80.16 | GOOD |
| 5 | REJECTED | 0.0943 | 0.439 | CAUTION | 551.8 | 73.1 | 0.00 | GOOD |
| 10 | ACCEPTED | 0.1098 | 0.429 | CAUTION | 563.3 | 70.9 | 75.17 | GOOD |
| 15 | REJECTED | 0.0944 | 0.393 | CAUTION | 154.8 | 113.2 | 0.00 | GOOD |
| 20 | ACCEPTED | 0.1142 | 0.427 | CAUTION | 421.1 | 89.9 | 56.57 | GOOD |
| 25 | REJECTED | 0.0950 | 0.516 | CAUTION | 647.8 | 86.5 | 0.00 | GOOD |
| 30 | REJECTED | 0.0901 | 0.414 | CAUTION | 398.6 | 92.8 | 0.00 | GOOD |
| 35 | ACCEPTED | 0.1057 | 0.409 | CAUTION | 510.8 | 85.4 | 5.00 | GOOD |
| 40 | REJECTED | 0.0950 | 0.408 | CAUTION | 445.8 | 83.2 | 0.00 | GOOD |
| 45 | REJECTED | 0.0931 | 0.583 | CAUTION | 1489.6 | 25.8 | 0.00 | GOOD |
| 50 | REJECTED | 0.0973 | 0.515 | CAUTION | 349.9 | 131.9 | 0.00 | GOOD |
| 55 | SUPPRESSED | 0.0000 | 0.000 | SUPPRESS | 103.7 | 218.1 | 0.00 | GOOD |

## Summary

- Accepted: 4/12
- Rejected: 7/12
- Suppressed: 1/12
- Peak range: 0.0000 – 0.1142
- Mean peak (all): 0.0911
- Suggested threshold (P10 non-suppressed): 0.093
- Current threshold: 0.100

## Baseline Comparison

| Validation | Peak range | Accepted |
|-----------|-----------|----------|
| OI-44 Cross-modal: RGB vs DEM hillshade | 0.0903–0.1136 | 0/12 |
| OI-45 Same-modal self-offset | 0.9874–0.9932 | 12/12 |
| OI-46 Real Sentinel-2 TCI | 0.0000–0.1142 | 4/12 |

## Interpretation and OI-46 Finding

OI-46 PARTIAL: Real Sentinel-2 TCI reference yields moderate NCC peaks on some frames.  Frames with low peaks may reflect texture mismatch between shimla_texture.png (DEM hillshade-colour) and genuine Sentinel-2 imagery at those positions.  Recommend re-generating Blender terrain texture from the TCI tile crop to achieve full-corridor same-modality validation.

**shimla_texture.png finding:** The texture used to render the Blender frames is `viz.hh_hillshade-color.png` from OpenTopography — a terrain elevation visualisation product, not an optical satellite image.  Genuine Sentinel-2 TRN requires the reference and query images to be from the same sensor class.  The Sentinel-2 TCI tiles now available provide the correct reference source; the Blender frames must be re-rendered with a TCI-derived texture to complete the same-modality validation chain.
