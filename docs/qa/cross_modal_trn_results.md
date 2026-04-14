# Cross-Modal TRN Validation
## MicroMind Pre-HIL Evidence

**Date:** 15 April 2026
**Corridor:** shimla_local
**Query source:** Blender-rendered RGB frames (Sentinel-2 texture + GLO-30 DEM)
**Reference source:** DEM hillshade tiles (Copernicus GLO-30)
**Altitude:** 150.0m AGL
**Sun:** azimuth 135deg, elevation 45deg

## Results

| km | Status | Peak | Suitability | Error (m) | Quality |
|----|--------|------|-------------|-----------|--------|
| 0 | REJECTED | 0.0908 | 0.674 | 0.0 | GOOD |
| 5 | REJECTED | 0.1071 | 0.783 | 0.0 | GOOD |
| 10 | REJECTED | 0.1066 | 0.713 | 0.0 | GOOD |
| 15 | REJECTED | 0.0939 | 0.704 | 0.0 | GOOD |
| 20 | REJECTED | 0.1085 | 0.680 | 0.0 | GOOD |
| 25 | REJECTED | 0.1077 | 0.770 | 0.0 | GOOD |
| 30 | REJECTED | 0.0990 | 0.765 | 0.0 | GOOD |
| 35 | REJECTED | 0.1067 | 0.692 | 0.0 | GOOD |
| 40 | REJECTED | 0.0903 | 0.626 | 0.0 | GOOD |
| 45 | REJECTED | 0.1136 | 0.658 | 0.0 | GOOD |
| 50 | REJECTED | 0.1055 | 0.564 | 0.0 | MARGINAL |
| 55 | REJECTED | 0.0977 | 0.548 | 0.0 | GOOD |

## Summary

- Accepted: 0/12
- Mean peak (accepted): N/A
- P50 error: N/A
- P95 error: N/A
- P99 error: N/A
- Suggested threshold: 0.091
- Current threshold: 0.150

## Interpretation

Peak values in cross-modal matching (Blender RGB vs DEM hillshade) are lower than self-match (1.0) by design. The CAS paper reports 0.3–0.7 over textured terrain. Values above the acceptance threshold indicate reliable TRN corrections in operational conditions.
