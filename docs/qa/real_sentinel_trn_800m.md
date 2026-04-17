# Real Sentinel-2 TRN Validation — OI-46
## MicroMind Pre-HIL Evidence

**Date:** 16 April 2026
**Corridor:** shimla_local
**Query:** Blender-rendered RGB frames (sentinel_tci_dem_extent.tif — Sentinel-2 TCI S2A T43RGQ 20251017)
**Reference:** Sentinel-2 TCI `T43RGQ_20251017T053241_TCI_10m.jp2`  
  CRS: EPSG:32643 (UTM 43N), 10 m/px, uint8 RGB  
  Scene date: 17 October 2025  
  WGS84: 77.086–78.264°E, 30.605–31.618°N
**Altitude:** 800.0m AGL
**TRN GSD:** 5.0 m/px
**Ref tile:** 184×184 px (924m footprint)

## Results

| km | Status | Peak | Suitability | Rec | RefTexVar | RefRelief | CorrMag (m) | Query |
|----|--------|------|-------------|-----|----------|-----------|-------------|-------|
| 0 | REJECTED | 0.0629 | 0.614 | ACCEPT | 209.8 | 247.9 | 0.00 | POOR |
| 5 | REJECTED | 0.0337 | 0.557 | CAUTION | 76.6 | 217.3 | 0.00 | POOR |
| 10 | REJECTED | 0.0344 | 0.456 | CAUTION | 143.5 | 141.8 | 0.00 | POOR |
| 15 | REJECTED | 0.0398 | 0.509 | CAUTION | 72.3 | 179.6 | 0.00 | POOR |
| 20 | REJECTED | 0.0279 | 0.526 | CAUTION | 74.9 | 186.3 | 0.00 | POOR |
| 25 | REJECTED | 0.0389 | 0.425 | CAUTION | 123.2 | 132.4 | 0.00 | POOR |
| 30 | REJECTED | 0.0292 | 0.495 | CAUTION | 70.0 | 173.4 | 0.00 | POOR |
| 35 | REJECTED | 0.0394 | 0.455 | CAUTION | 86.3 | 152.6 | 0.00 | POOR |
| 40 | REJECTED | 0.0397 | 0.548 | CAUTION | 53.9 | 242.0 | 0.00 | POOR |
| 45 | SUPPRESSED | 0.0000 | 0.000 | SUPPRESS | 22.6 | 247.9 | 0.00 | POOR |
| 50 | REJECTED | 0.0358 | 0.558 | CAUTION | 100.4 | 196.2 | 0.00 | POOR |
| 55 | SUPPRESSED | 0.0000 | 0.000 | SUPPRESS | 42.5 | 246.5 | 0.00 | POOR |

## Summary

- Accepted: 0/12
- Rejected: 10/12
- Suppressed: 2/12
- Peak range: 0.0000 – 0.0629
- Mean peak (all): 0.0318
- Suggested threshold (P10 non-suppressed): 0.050
- Current threshold: 0.100

## Baseline Comparison

| Validation | Peak range | Accepted |
|-----------|-----------|----------|
| OI-44 Cross-modal: RGB vs DEM hillshade | 0.0903–0.1136 | 0/12 |
| OI-45 Same-modal self-offset | 0.9874–0.9932 | 12/12 |
| OI-46 Real Sentinel-2 TCI | 0.0000–0.0629 | 0/12 |

## Interpretation and OI-46 Finding

OI-46 OPEN: Real Sentinel-2 NCC peaks are below operational threshold across most or all frames.  Root cause: shimla_texture.png is a DEM hillshade-colour map (viz.hh_hillshade-color.png), NOT actual Sentinel-2 imagery.  The Blender frames do not represent Sentinel-2 visual content.  Action required: replace shimla_texture.png with a TCI crop matched to the corridor extent, re-render Blender frames, re-run OI-46.

**shimla_texture.png finding:** The texture used to render the Blender frames is `viz.hh_hillshade-color.png` from OpenTopography — a terrain elevation visualisation product, not an optical satellite image.  Genuine Sentinel-2 TRN requires the reference and query images to be from the same sensor class.  The Sentinel-2 TCI tiles now available provide the correct reference source; the Blender frames must be re-rendered with a TCI-derived texture to complete the same-modality validation chain.
