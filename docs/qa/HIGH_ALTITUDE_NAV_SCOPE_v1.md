# High-Altitude Navigation Scope v1.0
Authority: Deputy 1 | Date: 26 April 2026

## AVP-03 Profile (Jammu-Leh Tactical)
- Range: 330km | Cruise: 300-500m AGL | GNSS-denied: km 30-330
- DEM: 3 tiles available in repo (TILE1/2/3 confirmed)
- Optical: Sentinel-2 acquisition needed for JL corridor

## AVP-02 Validated Specification (Reference)
P99 < 77m at 180km GNSS-denied (Shimla-Manali, confirmed Gate 6)

## AVP-03 Provisional Specification
At 500m AGL, EO camera GSD degrades to ~1.0-1.5 m/px.
TRN correction accuracy degrades to 50-100m CEP vs 15-25m at AVP-02.
Proposed AVP-03 provisional: **P99 < 150m at 330km GNSS-denied**

This accommodates GSD degradation while remaining operationally useful
for terminal guidance handoff. Requires formal SRS amendment (ADR required).

## Spiti Valley — AVP-04 Indicator
Spiti Valley DEM now confirmed available at:
micromind_data/raw/dem/Spiti Valley/
Spiti represents the extreme high-altitude case (3500-4500m MSL terrain,
5000m+ AGL operations). AVP-04 feasibility requires landmark density
analysis on this DEM before any specification can be proposed.

## Correction Interval Estimates
| Altitude AGL | Expected LightGlue ACCEPT interval | INS drift per interval | SAL-3 required? |
|---|---|---|---|
| 150m (AVP-02) | 1 per 5km (textured terrain) | 25-50m | No (nominal) |
| 300m (AVP-03 low) | 1 per 15km | 75-150m | YES |
| 500m (AVP-03 high) | 1 per 30-40km | 150-300m | YES — mandatory |
| 500m+ (AVP-04) | 1 per 50km+ | 250-500m | YES — insufficient alone |

## Confidence Model for AVP-03
| Source | CEP at 500m AGL | EKF Weight |
|---|---|---|
| LightGlue ACCEPT | 20-40m | Full weight |
| LightGlue CAUTION | 50-100m | 0.5x weight |
| SAL-3 ACCEPT (validated) | 30-80m | 0.3x weight |
| INS-only | Accumulates ~5m/km | Propagation only |

## Next Actions
1. Run Gate 6 equivalent Monte Carlo on Jammu-Leh corridor DEMs
2. Characterise terrain-class profile of JL corridor km by km
3. Download Sentinel-2 optical for JL corridor (coordinates TBD after Gate 6 JL run)
4. Formal ADR for AVP-03 P99 < 150m specification
