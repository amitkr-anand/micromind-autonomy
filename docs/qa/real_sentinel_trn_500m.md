# Real Sentinel-2 TRN Validation — OI-46
## MicroMind Pre-HIL Evidence

**Date:** 16 April 2026
**Corridor:** shimla_local
**Query:** Blender-rendered RGB frames (sentinel_tci_dem_extent.tif — Sentinel-2 TCI S2A T43RGQ 20251017)
**Reference:** Sentinel-2 TCI `T43RGQ_20251017T053241_TCI_10m.jp2`  
  CRS: EPSG:32643 (UTM 43N), 10 m/px, uint8 RGB  
  Scene date: 17 October 2025  
  WGS84: 77.086–78.264°E, 30.605–31.618°N
**Altitude:** 500.0m AGL
**TRN GSD:** 5.0 m/px
**Ref tile:** 115×115 px (577m footprint)

## Results

| km | Status | Peak | Suitability | Rec | RefTexVar | RefRelief | CorrMag (m) | Query |
|----|--------|------|-------------|-----|----------|-----------|-------------|-------|
| 0 | REJECTED | 0.0992 | 0.627 | ACCEPT | 258.4 | 239.8 | 0.00 | POOR |
| 5 | REJECTED | 0.0683 | 0.498 | CAUTION | 171.1 | 156.0 | 0.00 | POOR |
| 10 | ACCEPTED | 0.1775 | 0.470 | CAUTION | 213.7 | 136.3 | 0.00 | POOR |
| 15 | REJECTED | 0.0676 | 0.419 | CAUTION | 145.9 | 126.3 | 0.00 | POOR |
| 20 | REJECTED | 0.0589 | 0.536 | CAUTION | 113.3 | 183.6 | 0.00 | POOR |
| 25 | REJECTED | 0.0495 | 0.410 | CAUTION | 209.4 | 110.1 | 0.00 | POOR |
| 30 | REJECTED | 0.0523 | 0.485 | CAUTION | 67.6 | 169.3 | 0.00 | POOR |
| 35 | ACCEPTED | 0.1292 | 0.413 | CAUTION | 183.5 | 116.4 | 0.00 | POOR |
| 40 | ACCEPTED | 0.1230 | 0.548 | CAUTION | 53.5 | 233.6 | 0.00 | POOR |
| 45 | REJECTED | 0.0489 | 0.365 | CAUTION | 400.2 | 55.7 | 0.00 | POOR |
| 50 | REJECTED | 0.0714 | 0.532 | CAUTION | 137.7 | 176.8 | 0.00 | POOR |
| 55 | ACCEPTED | 0.1042 | 0.553 | CAUTION | 65.6 | 239.3 | 0.00 | POOR |

## Summary

- Accepted: 4/12
- Rejected: 8/12
- Suppressed: 0/12
- Peak range: 0.0489 – 0.1775
- Mean peak (all): 0.0875
- Suggested threshold (P10 non-suppressed): 0.050
- Current threshold: 0.100

## Baseline Comparison

| Validation | Peak range | Accepted |
|-----------|-----------|----------|
| OI-44 Cross-modal: RGB vs DEM hillshade | 0.0903–0.1136 | 0/12 |
| OI-45 Same-modal self-offset | 0.9874–0.9932 | 12/12 |
| OI-46 Real Sentinel-2 TCI | 0.0489–0.1775 | 4/12 |

## Interpretation and OI-46 Finding

OI-46 PARTIAL: Real Sentinel-2 TCI reference yields moderate NCC peaks on some frames.  Frames with low peaks may reflect texture mismatch between shimla_texture.png (DEM hillshade-colour) and genuine Sentinel-2 imagery at those positions.  Recommend re-generating Blender terrain texture from the TCI tile crop to achieve full-corridor same-modality validation.

**shimla_texture.png finding:** The texture used to render the Blender frames is `viz.hh_hillshade-color.png` from OpenTopography — a terrain elevation visualisation product, not an optical satellite image.  Genuine Sentinel-2 TRN requires the reference and query images to be from the same sensor class.  The Sentinel-2 TCI tiles now available provide the correct reference source; the Blender frames must be re-rendered with a TCI-derived texture to complete the same-modality validation chain.
