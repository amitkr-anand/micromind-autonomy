# MicroMind Demonstration Corridor Status

Authority: Deputy 1 | Date: 26 April 2026

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

## Corridor 2 — Jammu-Leh (SECONDARY — IN PREPARATION)

**Status: DATA PREPARATION — not yet demonstration ready**

| Parameter | Value |
|-----------|-------|
| Range | ~330 km (Udhampur to Leh) |
| GNSS-denied from | km 30 |
| DEM | `data/terrain/Jammu_leh_corridor_COP30/TILE1-3` (COP30 ~28m) ✅ — gap at Kargil 76.10–76.18°E |
| Optical | Current acquisition (Nov 2025) **UNUSABLE** east of Kargil — winter whiteout. Re-acquisition target: Aug–Sep 2025 Sentinel-2 L2A <10% cloud cover (OI-JL-01). |
| Nav performance | NOT YET CHARACTERISED |

**DEM suitability (JL-TCI-VALIDATE, 26 Apr 2026):**
- WP_UDHAMPUR (75.15°E, 33.00°N): ACCEPT 0.6457, texture_var=271, relief=671 m ✓
- WP_KARGIL (76.18°E, 33.55°N): OUTSIDE DEM COVERAGE (OI-JL-03)
- WP_LEH (77.57°E, 34.17°N): ACCEPT 0.9497, texture_var=35899, relief=189 m ✓

**TCI quality (JL-TCI-VALIDATE, 26 Apr 2026):**

| Tile | Waypoint | lap_var | sat% | Usable |
|------|----------|---------|------|--------|
| T43SES | WP_UDHAMPUR | 1232 | 2% | YES |
| T43SFT | WP_KARGIL | 1.5 | 99.9% | NO — whiteout |
| T43SGT | WP_LEH | 166 | 83% | MARGINAL |

**EC-13 Characterisation:** BLOCKED on OI-JL-01 + OI-JL-03.

**Demonstration readiness: NOT READY — pending summer imagery and Kargil DEM tile**

---

## Open Items

| ID | Description | Priority | Status |
|----|-------------|----------|--------|
| OI-JL-01 | Summer TCI re-acquisition (T43SFS/T43SFT/T43SGS/T43SGT) | HIGH | OPEN |
| OI-JL-02 | T43SDT (Jammu/Srinagar tile) | LOW | DEFERRED |
| OI-JL-03 | Kargil DEM gap — COP30 tile N33E076 | MEDIUM | OPEN |
| OI-JL-04 | NAV02-CHAR-RUN4 | HIGH | BLOCKED |

---

## Acquisition Guidance for Summer Re-download

**Product:** Sentinel-2 L2A (MSIL2A)
**Tiles needed:** T43SFS, T43SFT, T43SGS, T43SGT
**Date range:** 15 Aug – 30 Sep (summer, pre-monsoon clearance in Ladakh)
**Cloud filter:** < 10%
**Source:** Copernicus Data Space (dataspace.copernicus.eu)
**Priority:** T43SFT (Kargil/Zoji La approach) and T43SGS (Ladakh plateau) most critical for corridor continuity.
**Bands required:** B04 (Red), B03 (Green), B02 (Blue) at R10m — same extraction method as current TCI set.
