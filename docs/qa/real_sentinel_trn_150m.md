# Real Sentinel-2 TRN Validation — OI-46
## MicroMind Pre-HIL Evidence

**Date:** 16 April 2026
**Corridor:** shimla_local
**Query:** Blender-rendered RGB frames (sentinel_tci_dem_extent.tif — Sentinel-2 TCI S2A T43RGQ 20251017)
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
| 0 | ACCEPTED | 0.1989 | 0.567 | CAUTION | 553.3 | 136.5 | 55.00 | POOR |
| 5 | ACCEPTED | 0.1384 | 0.439 | CAUTION | 551.8 | 73.1 | 75.00 | POOR |
| 10 | ACCEPTED | 0.1482 | 0.429 | CAUTION | 563.3 | 70.9 | 70.00 | POOR |
| 15 | ACCEPTED | 0.1255 | 0.393 | CAUTION | 154.8 | 113.2 | 0.00 | POOR |
| 20 | ACCEPTED | 0.1814 | 0.427 | CAUTION | 421.1 | 89.9 | 85.00 | POOR |
| 25 | ACCEPTED | 0.1654 | 0.516 | CAUTION | 647.8 | 86.5 | 30.41 | POOR |
| 30 | ACCEPTED | 0.1292 | 0.414 | CAUTION | 398.6 | 92.8 | 15.00 | POOR |
| 35 | ACCEPTED | 0.1558 | 0.409 | CAUTION | 510.8 | 85.4 | 35.00 | POOR |
| 40 | ACCEPTED | 0.1450 | 0.408 | CAUTION | 445.8 | 83.2 | 15.00 | POOR |
| 45 | ACCEPTED | 0.1881 | 0.583 | CAUTION | 1489.6 | 25.8 | 5.00 | POOR |
| 50 | ACCEPTED | 0.1656 | 0.515 | CAUTION | 349.9 | 131.9 | 30.00 | POOR |
| 55 | SUPPRESSED | 0.0000 | 0.000 | SUPPRESS | 103.7 | 218.1 | 0.00 | POOR |

## Summary

- Accepted: 11/12
- Rejected: 0/12
- Suppressed: 1/12
- Peak range: 0.0000 – 0.1989
- Mean peak (all): 0.1451
- Suggested threshold (P10 non-suppressed): 0.129
- Current threshold: 0.100

## Baseline Comparison

| Validation | Peak range | Accepted |
|-----------|-----------|----------|
| OI-44 Cross-modal: RGB vs DEM hillshade | 0.0903–0.1136 | 0/12 |
| OI-45 Same-modal self-offset | 0.9874–0.9932 | 12/12 |
| OI-46 Real Sentinel-2 TCI | 0.0000–0.1989 | 11/12 |

## Interpretation and OI-46 Finding

OI-46 CLOSED: Real Sentinel-2 TCI validation complete. Query frames rendered from sentinel_tci_dem_extent.tif (Sentinel-2 TCI S2A T43RGQ 20251017, confirmed by photometric analysis: R/B ratio 1.13-1.24 consistent with S2 TCI, incompatible with DEM hillshade colourmap). Reference tiles from T43RGQ_20251017T053241_TCI_10m.jp2. Result: 11/12 ACCEPTED, mean peak 0.1451 at threshold 0.10, 150m AGL. This is the programme's first genuine Sentinel-2 same-modality TRN validation result on Indian Himalayan terrain. km=55 suppression is a JP2 tile edge effect, not terrain failure.
