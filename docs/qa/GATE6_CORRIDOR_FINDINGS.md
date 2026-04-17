# Gate 6 — Jammu-Leh Corridor Terrain Findings
**Date:** 18 April 2026
**Commit:** 24b01e6

## Terrain Suitability Profile

| Segment | km range | Terrain | Suitability | TRN Status |
|---------|----------|---------|-------------|------------|
| Jammu approach | 0–30 | Shivalik foothills | GNSS zone | N/A |
| Udhampur–Ramban | 30–60 | Chenab gorge | CAUTION | Active |
| Banihal–Srinagar | 60–120 | Kashmir valley floor | SUPPRESS | VIO bridge |
| Sonamarg–Zoji La | 120–180 | Alpine ridge | CAUTION | Active |
| Drass–Kargil | 180–240 | High-altitude valley | CAUTION | Active |
| Kargil–Leh approach | 240–300 | Ladakh valley | CAUTION | Active |
| Leh terminal | 300–330 | Ladakh plateau desert | SUPPRESS | VIO bridge |

## Key Findings
- 60km suppression gap (km=60–120): Kashmir valley floor. VIO bridging required.
- Terminal suppression (km=300–330): High-altitude desert. Documented limitation.
- Zoji La (km=180) is strongest TRN point: relief 3931m, score 0.599.
- Suppression rate: 43.3% (26/60 correction opportunities suppressed).

## N=300 Monte Carlo Results (master_seed=42)
- INS-only P99 at km=330: 540.7m
- TRN-only P99 at km=330: 96.9m (82.1% reduction)
- VIO+TRN P99 at km=330: 84.9m
- TRN P99 at km=180 (Zoji La): 71.5m

## Gate 6 Acceptance Criteria (Deputy 1, 18 April 2026)
- C1 TRN P99 km=330 < 150m: PASS (96.9m)
- C2 TRN reduction ≥ 70%: PASS (82.1%)
- C3 TRN P99 km=180 < 100m: PASS (71.5m)
- C4 Non-monotonic TRN growth: PASS
