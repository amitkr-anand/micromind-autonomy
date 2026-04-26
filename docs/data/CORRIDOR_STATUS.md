# MicroMind Demonstration Corridor Status

Authority: Deputy 1 | Date: 26 April 2026 | Updated: 27 April 2026

---

## Corridor 1 — Shimla-Manali (PRIMARY)

**Status: VALIDATED — demonstration ready**

| Parameter | Value |
|-----------|-------|
| Range | ~180 km |
| GNSS-denied from | km 30 |
| DEM | `data/terrain/shimla_manali_corridor/` (COP30 ~30m) ✅ |
| Optical | `micromind_data/raw/imagery/sentinel/sentinel_tci_mosaic.tif` (UTM43N, 10m) ✅ |
| Nav performance | P99 < 77 m at 180 km (Monte Carlo N=300) ✅ |

**EC-13 Characterisation:** Run 3 complete (26 Apr 2026).
- Cross-modal ceiling 0.03–0.04 on winter Sentinel TCI (confirmed architectural finding).
- TerrainSuitabilityScorer ACCEPT on all 4 waypoints (scores 0.644–0.714).
- Suppression zone not exercised at adjusted WP2 (31.43°N valley floor — coordinate re-selection needed for SUPPRESS path exercise).

**Demonstration readiness: READY**

---

## Corridor 2 — Jammu-Sonamarg (SECONDARY)

**Status: CHARACTERISATION READY — imagery confirmed**

| Parameter | Value |
|-----------|-------|
| Range | ~230 km (Jammu to Sonamarg via NH-1) |
| GNSS-denied from | km 30 |
| DEM | `data/terrain/Jammu_leh_corridor_COP30/TILE1` + `TILE2` (COP30 ~28m) ✅ |
| Optical | T43SDS, T43SES, T43SDT, T43SET (Nov 2025, UTM43N 10m) ✅ — 6 confirmed-good WPs |
| Nav performance | NOT YET CHARACTERISED — Run 4 pending |

**Confirmed-good waypoints (JL-ROUTE-DESIGN, 27 Apr 2026):**

| ID | Name | lon | lat | Tile | km | lap_var | sat% | Quality |
|----|------|-----|-----|------|----|---------|------|---------|
| WP00 | Jammu departure | 74.86°E | 32.73°N | T43SDS | 0 | 1509 | 31.1% | GOOD |
| WP01 | Udhampur ridge | 75.14°E | 32.92°N | T43SES | 50 | 1326 | 24.7% | GOOD |
| WP02 | Ramban gorge | 75.24°E | 33.24°N | T43SES | 90 | 1359 | 27.3% | GOOD |
| WP03 | Banihal pass | 75.18°E | 33.40°N | T43SES | 120 | 1270 | 5.1% | GOOD — best imagery |
| WP_GAP | Kashmir valley entry | 75.10°E | 33.50°N | T43SET | 130 | — | 100% | SUPPRESS — INS-only |
| WP04 | Kangan / Sind Valley | 75.00°E | 34.19°N | T43SDT | 180 | 1269 | 34.0% | GOOD — TRN resumes |
| WP05 | Sonamarg | 75.29°E | 34.30°N | T43SET | 230 | 987 | 41.9% | GOOD |

**Corridor segments:**
- km 0–120: Western mountain approach — TRN ACTIVE (lap_var 1270–1509)
- km 120–180: Kashmir valley floor — TRN SUPPRESS, INS-only (expected, operationally representative)
- km 180–230: Sind Valley reconnect — TRN ACTIVE (lap_var 987–1269)

**Structural pattern:** Mirrors Shimla-Manali: mountain approach → valley suppression gap → mountain reconnect.

**EC-13 Characterisation:** READY FOR RUN 4. Corridor definition: `scenarios/nav02_char/jammu_sonamarg_corridor_definition.yaml`

**Demonstration readiness: READY FOR NAV02-CHAR-RUN4**

---

## Open Items

| ID | Description | Priority | Status |
|----|-------------|----------|--------|
| OI-JL-01 | Summer TCI re-acquisition (T43SFS/T43SFT/T43SGS/T43SGT) | HIGH | OPEN |
| OI-JL-02 | T43SDT (Jammu/Srinagar tile) | LOW | DEFERRED |
| OI-JL-03 | Kargil DEM gap — COP30 tile N33E076 | MEDIUM | OPEN |
| OI-JL-04 | NAV02-CHAR-RUN4 (Jammu-Sonamarg) | HIGH | READY — corridor definition complete |

---

## Acquisition Guidance for Summer Re-download

**Product:** Sentinel-2 L2A (MSIL2A)
**Tiles needed:** T43SFS, T43SFT, T43SGS, T43SGT
**Date range:** 15 Aug – 30 Sep (summer, pre-monsoon clearance in Ladakh)
**Cloud filter:** < 10%
**Source:** Copernicus Data Space (dataspace.copernicus.eu)
**Priority:** T43SFT (Kargil/Zoji La approach) and T43SGS (Ladakh plateau) most critical for corridor continuity.
**Bands required:** B04 (Red), B03 (Green), B02 (Blue) at R10m — same extraction method as current TCI set.
