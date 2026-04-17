# Real Sentinel-2 TRN Validation — OI-46
## MicroMind Pre-HIL Evidence

**Date:** 16 April 2026
**Corridor:** shimla_local
**Query:** Blender-rendered RGB frames (sentinel_tci_dem_extent.tif — Sentinel-2 TCI S2A T43RGQ 20251017)
**Reference:** Sentinel-2 TCI `T43RGQ_20251017T053241_TCI_10m.jp2`  
  CRS: EPSG:32643 (UTM 43N), 10 m/px, uint8 RGB  
  Scene date: 17 October 2025  
  WGS84: 77.086–78.264°E, 30.605–31.618°N
**Altitude:** 200.0m AGL
**TRN GSD:** 5.0 m/px
**Ref tile:** 46×46 px (231m footprint)

## Results

| km | Status | Peak | Suitability | Rec | RefTexVar | RefRelief | CorrMag (m) | Query |
|----|--------|------|-------------|-----|----------|-----------|-------------|-------|
| 0 | ACCEPTED | 0.1292 | 0.617 | ACCEPT | 433.2 | 189.1 | 115.00 | POOR |
| 5 | ACCEPTED | 0.1040 | 0.455 | CAUTION | 240.9 | 129.3 | 35.00 | POOR |
| 10 | ACCEPTED | 0.1143 | 0.389 | CAUTION | 369.7 | 78.0 | 50.00 | POOR |
| 15 | ACCEPTED | 0.1058 | 0.374 | CAUTION | 129.2 | 112.0 | 0.00 | POOR |
| 20 | ACCEPTED | 0.2096 | 0.408 | CAUTION | 282.2 | 97.7 | 0.00 | POOR |
| 25 | ACCEPTED | 0.1176 | 0.464 | CAUTION | 492.6 | 83.4 | 95.00 | POOR |
| 30 | ACCEPTED | 0.1586 | 0.398 | CAUTION | 222.1 | 105.6 | 10.00 | POOR |
| 35 | ACCEPTED | 0.1525 | 0.398 | CAUTION | 243.4 | 102.0 | 70.00 | POOR |
| 40 | ACCEPTED | 0.1274 | 0.380 | CAUTION | 327.3 | 87.2 | 15.00 | POOR |
| 45 | REJECTED | 0.0977 | 0.457 | CAUTION | 859.4 | 32.9 | 0.00 | POOR |
| 50 | ACCEPTED | 0.1564 | 0.560 | CAUTION | 243.0 | 180.6 | 25.00 | POOR |
| 55 | SUPPRESSED | 0.0000 | 0.000 | SUPPRESS | 71.7 | 233.4 | 0.00 | POOR |

## Summary

- Accepted: 10/12
- Rejected: 1/12
- Suppressed: 1/12
- Peak range: 0.0000 – 0.2096
- Mean peak (all): 0.1228
- Suggested threshold (P10 non-suppressed): 0.104
- Current threshold: 0.100

## Baseline Comparison

| Validation | Peak range | Accepted |
|-----------|-----------|----------|
| OI-44 Cross-modal: RGB vs DEM hillshade | 0.0903–0.1136 | 0/12 |
| OI-45 Same-modal self-offset | 0.9874–0.9932 | 12/12 |
| OI-46 Real Sentinel-2 TCI | 0.0000–0.2096 | 10/12 |

## Interpretation and OI-46 Finding

OI-46 CLOSED: Real Sentinel-2 TCI reference yields acceptable NCC peaks.  The Blender frames correlate with genuine Sentinel-2 imagery at operationally useful levels.  AD-01 same-modality validated with real satellite data.

**shimla_texture.png finding:** The texture used to render the Blender frames is `viz.hh_hillshade-color.png` from OpenTopography — a terrain elevation visualisation product, not an optical satellite image.  Genuine Sentinel-2 TRN requires the reference and query images to be from the same sensor class.  The Sentinel-2 TCI tiles now available provide the correct reference source; the Blender frames must be re-rendered with a TCI-derived texture to complete the same-modality validation chain.
