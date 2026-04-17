# Real Sentinel-2 TRN Validation — OI-46
## MicroMind Pre-HIL Evidence

**Date:** 16 April 2026
**Corridor:** shimla_local
**Query:** Blender-rendered RGB frames (sentinel_tci_dem_extent.tif — Sentinel-2 TCI S2A T43RGQ 20251017)
**Reference:** Sentinel-2 TCI `T43RGQ_20251017T053241_TCI_10m.jp2`  
  CRS: EPSG:32643 (UTM 43N), 10 m/px, uint8 RGB  
  Scene date: 17 October 2025  
  WGS84: 77.086–78.264°E, 30.605–31.618°N
**Altitude:** 300.0m AGL
**TRN GSD:** 5.0 m/px
**Ref tile:** 69×69 px (346m footprint)

## Results

| km | Status | Peak | Suitability | Rec | RefTexVar | RefRelief | CorrMag (m) | Query |
|----|--------|------|-------------|-----|----------|-----------|-------------|-------|
| 0 | REJECTED | 0.0903 | 0.648 | ACCEPT | 349.8 | 196.0 | 0.00 | POOR |
| 5 | REJECTED | 0.0590 | 0.467 | CAUTION | 240.0 | 129.5 | 0.00 | POOR |
| 10 | REJECTED | 0.0950 | 0.431 | CAUTION | 244.2 | 118.6 | 0.00 | POOR |
| 15 | REJECTED | 0.0979 | 0.401 | CAUTION | 137.1 | 121.1 | 0.00 | POOR |
| 20 | ACCEPTED | 0.1310 | 0.529 | CAUTION | 96.8 | 188.6 | 135.00 | POOR |
| 25 | REJECTED | 0.0911 | 0.446 | CAUTION | 361.5 | 97.4 | 0.00 | POOR |
| 30 | REJECTED | 0.0766 | 0.448 | CAUTION | 183.7 | 131.4 | 0.00 | POOR |
| 35 | ACCEPTED | 0.2047 | 0.391 | CAUTION | 195.3 | 107.4 | 0.00 | POOR |
| 40 | ACCEPTED | 0.1988 | 0.401 | CAUTION | 305.7 | 94.1 | 98.49 | POOR |
| 45 | ACCEPTED | 0.1012 | 0.405 | CAUTION | 567.9 | 41.7 | 0.00 | POOR |
| 50 | REJECTED | 0.0878 | 0.547 | CAUTION | 194.7 | 172.8 | 0.00 | POOR |
| 55 | ACCEPTED | 0.1185 | 0.548 | CAUTION | 67.6 | 235.3 | 0.00 | POOR |

## Summary

- Accepted: 5/12
- Rejected: 7/12
- Suppressed: 0/12
- Peak range: 0.0590 – 0.2047
- Mean peak (all): 0.1127
- Suggested threshold (P10 non-suppressed): 0.078
- Current threshold: 0.100

## Baseline Comparison

| Validation | Peak range | Accepted |
|-----------|-----------|----------|
| OI-44 Cross-modal: RGB vs DEM hillshade | 0.0903–0.1136 | 0/12 |
| OI-45 Same-modal self-offset | 0.9874–0.9932 | 12/12 |
| OI-46 Real Sentinel-2 TCI | 0.0590–0.2047 | 5/12 |

## Interpretation and OI-46 Finding

OI-46 PARTIAL: Real Sentinel-2 TCI reference yields moderate NCC peaks on some frames.  Frames with low peaks may reflect texture mismatch between shimla_texture.png (DEM hillshade-colour) and genuine Sentinel-2 imagery at those positions.  Recommend re-generating Blender terrain texture from the TCI tile crop to achieve full-corridor same-modality validation.

**shimla_texture.png finding:** The texture used to render the Blender frames is `viz.hh_hillshade-color.png` from OpenTopography — a terrain elevation visualisation product, not an optical satellite image.  Genuine Sentinel-2 TRN requires the reference and query images to be from the same sensor class.  The Sentinel-2 TCI tiles now available provide the correct reference source; the Blender frames must be re-rendered with a TCI-derived texture to complete the same-modality validation chain.
