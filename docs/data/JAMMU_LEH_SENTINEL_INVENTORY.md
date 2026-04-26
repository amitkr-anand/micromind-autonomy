# Jammu-Leh Sentinel-2 TCI Inventory

Authority: Deputy 1 | Created: 2026-04-26 | Updated: 2026-04-26 | Directives: JL-SENTINEL-EXTRACT, JL-TCI-VALIDATE

---

## 1. Purpose

RGB True-Colour Image (TCI) GeoTIFFs extracted from Sentinel-2 L2A SAFE ZIP archives
for Jammu-Leh corridor characterisation (NAV-02 / EC-13 future runs).
Source: `/home/mmuser/micromind_data/raw/imagery/sentinel2/Jammu_Leh/`
Output: `/home/mmuser/micromind_data/raw/imagery/sentinel2/Jammu_Leh_TCI/`

Extraction method: `gdalbuildvrt -separate` (B04/B03/B02) + `gdal_translate -ot Byte -scale 0 2000 0 255`
Format: GeoTIFF, DEFLATE compressed, tiled 256×256, INTERLEAVE=PIXEL, EPSG:32643

---

## 2. Extracted TCI Files — Priority 6 Tiles

| Tile    | File                   | Size  | Acq. Date  | UL (lon,lat)            | LR (lon,lat)            | Bands | Pixels      |
|---------|------------------------|-------|------------|-------------------------|-------------------------|-------|-------------|
| T43SES  | T43SES_TCI_10m.tif     | 261 MB | 2025-11-27 | 74°59'59"E 33°26'22"N  | 76°10'04"E 32°26'37"N  | 3     | 10980×10980 |
| T43SET  | T43SET_TCI_10m.tif     | 149 MB | 2025-11-27 | 74°59'59"E 34°20'30"N  | 76°10'47"E 33°20'45"N  | 3     | 10980×10980 |
| T43SFS  | T43SFS_TCI_10m.tif     | 137 MB | 2025-11-29 | 76°04'33"E 33°26'06"N  | 77°13'53"E 32°25'46"N  | 3     | 10980×10980 |
| T43SFT  | T43SFT_TCI_10m.tif     | 105 MB | 2025-11-27 | 76°05'14"E 34°20'13"N  | 77°15'15"E 33°19'51"N  | 3     | 10980×10980 |
| T43SGS  | T43SGS_TCI_10m.tif     |  53 MB | 2025-11-26 | 77°09'02"E 33°25'15"N  | 78°17'36"E 32°24'22"N  | 3     | 10980×10980 |
| T43SGT  | T43SGT_TCI_10m.tif     |  85 MB | 2025-11-26 | 77°10'24"E 34°19'21"N  | 78°19'36"E 33°18'24"N  | 3     | 10980×10980 |
| **TOTAL** |                      | **790 MB** | — | — | — | — | — |

---

## 3. Coverage Assessment (Step 4)

The 6 tiles form a **2-row × 3-column grid** (EPSG:32643, UTM Zone 43N):

```
Lat 34.3°N ┌─────────────┬─────────────┬─────────────┐
           │   T43SET    │   T43SFT    │   T43SGT    │
Lat 33.3°N └──────┬──────┴──────┬──────┴──────┬──────┘  (T-row overlap ~9km)
           ┌──────┴──────┬──────┴──────┬──────┴──────┐
           │   T43SES    │   T43SFS    │   T43SGS    │
Lat 32.4°N └─────────────┴─────────────┴─────────────┘
           75°E         76°E          77°E          78.3°E
           (W→E overlap ~6–8km at each column junction)
```

**Combined extent:** 74°59'59"E to 78°19'36"E × 32°24'22"N to 34°20'30"N

### Key waypoints:

| Location        | Lon (°E) | Lat (°N) | Covered by       | Status   |
|-----------------|----------|----------|------------------|----------|
| Jammu city      | 74.87    | 32.73    | T43SES (W edge at 75.00°E) | **PARTIAL** — 0.13° (~11 km) west gap |
| Banihal Pass    | 75.20    | 33.40    | T43SES / T43SET  | COVERED  |
| Srinagar        | 74.80    | 34.08    | T43SET (W edge at 75.00°E) | **PARTIAL** — 0.20° (~17 km) west gap |
| Zoji La Pass    | 75.47    | 34.27    | T43SET           | COVERED  |
| Kargil          | 76.13    | 34.55    | T43SFT           | COVERED  |
| Leh             | 77.57    | 34.17    | T43SGT           | COVERED  |

### Verdict: **CONTIGUOUS PARTIAL**
East-West coverage is contiguous (tiles overlap 6–8 km at junctions). Core Leh-approach
segment (75°E–78.3°E) is fully covered. Western corridor start (Jammu, Srinagar) falls
0.13°–0.20° west of tile coverage. Srinagar is outside T-row coverage — requires additional
western tile (T43SDT) for complete Jammu→Leh strip at northern latitudes.

---

## 4. Source ZIP Inventory (Full)

12 SAFE ZIP archives in `/home/mmuser/micromind_data/raw/imagery/sentinel2/Jammu_Leh/`:

| Tile    | ZIP filename                                                         | Size   | Priority |
|---------|----------------------------------------------------------------------|--------|----------|
| T43SDS  | S2C_MSIL2A_20251127T054211_N0511_R005_T43SDS_20251127T104409.zip   | 1.2 GB | secondary |
| T43SDT  | S2C_MSIL2A_20251127T054211_N0511_R005_T43SDT_20251127T104409.zip   | 930 MB | secondary |
| T43SDU  | S2A_MSIL2A_20251122T055251_N0511_R048_T43SDU_20251122T083810.zip   | 1.2 GB | secondary |
| T43SES  | S2C_MSIL2A_20251127T054211_N0511_R005_T43SES_20251127T104409.zip   | 1.2 GB | **primary** |
| T43SET  | S2C_MSIL2A_20251127T054211_N0511_R005_T43SET_20251127T104409.zip   | 1.2 GB | **primary** |
| T43SEU  | S2C_MSIL2A_20251127T054211_N0511_R005_T43SEU_20251127T104409.zip   | 1.2 GB | secondary |
| T43SFS  | S2A_MSIL2A_20251129T054251_N0511_R005_T43SFS_20251129T085708.zip   | 1.2 GB | **primary** |
| T43SFT  | S2C_MSIL2A_20251127T054211_N0511_R005_T43SFT_20251127T104409.zip   | 1.2 GB | **primary** |
| T43SFU  | S2C_MSIL2A_20251127T054211_N0511_R005_T43SFU_20251127T104409.zip   | 1.2 GB | secondary |
| T43SGS  | S2A_MSIL2A_20251126T053251_N0511_R105_T43SGS_20251126T083814.zip   | 1.2 GB | **primary** |
| T43SGT  | S2A_MSIL2A_20251126T053251_N0511_R105_T43SGT_20251126T083814.zip   | 1.1 GB | **primary** |
| T43SGU  | S2A_MSIL2A_20251126T053251_N0511_R105_T43SGU_20251126T083814.zip   | 1.2 GB | secondary |
| **TOTAL** |                                                                  | **14 GB** | — |

---

## 5. Disk Management Report (Step 5)

| Category                         | Count | Size    |
|----------------------------------|-------|---------|
| Raw SAFE ZIPs (all 12)           | 12    | 14.0 GB |
| Raw SAFE ZIPs (priority 6 only)  |  6    |  7.1 GB |
| Raw SAFE ZIPs (secondary 6)      |  6    |  6.9 GB |
| Extracted TCI GeoTIFFs           |  6    |  790 MB |
| Disk partition free              | —     | 1.2 TB  |

**Compression ratio:** Each 1.1–1.2 GB SAFE ZIP → 53–261 MB TCI GeoTIFF (RGB byte only).
TCI is 6–22× smaller than source ZIP because: (a) 3 bands only vs. ~13 bands in SAFE; (b)
DEFLATE-compressed byte vs. JP2-compressed uint16.

**Space recovery options (Deputy 1 authorisation required before any deletion):**
1. Delete secondary ZIPs (T43SDS, T43SDT, T43SDU, T43SEU, T43SFU, T43SGU): **~6.9 GB recovered**
   - Note: T43SDT needed for Srinagar western gap coverage if future run requires it
2. Delete all 12 ZIPs after verifying TCI correctness: **~14 GB recovered**
   - CAUTION: ZIPs are re-downloadable from Copernicus Data Space; TCI cannot be regenerated
     from TCI alone. Do not delete ZIPs until extraction quality is validated for NAV-02 use.

**Verdict: No urgent disk pressure.** 1.2 TB free. No action required at this time.

---

## 6. Extraction Log

| Tile    | Extracted      | Method                            | VRT cleaned |
|---------|----------------|-----------------------------------|-------------|
| T43SES  | 2026-04-26     | gdalbuildvrt + gdal_translate     | Yes         |
| T43SET  | 2026-04-26     | gdalbuildvrt + gdal_translate     | Yes         |
| T43SFS  | 2026-04-26     | gdalbuildvrt + gdal_translate     | Yes         |
| T43SFT  | 2026-04-26     | gdalbuildvrt + gdal_translate     | Yes         |
| T43SGS  | 2026-04-26     | gdalbuildvrt + gdal_translate     | Yes         |
| T43SGT  | 2026-04-26     | gdalbuildvrt + gdal_translate     | Yes         |

Failed attempt: `gdal_merge.py -separate -scale_1 ...` — flag not supported. Resolved by
switching to `gdalbuildvrt -separate` + `gdal_translate -scale`.

---

## 7. TCI Quality Validation (JL-TCI-VALIDATE, 2026-04-26)

### Step 1 — Tile-level statistics (Band 1 / Red channel)

| Tile    | Mean  | StdDev | Assessment                           |
|---------|-------|--------|--------------------------------------|
| T43SES  | 201.2 | 42.0   | MODERATE — usable texture            |
| T43SET  | 232.7 | 37.4   | BRIGHT — partial cloud/snow at elev. |
| T43SFS  | 225.2 | 57.0   | MODERATE-BRIGHT — best stddev, rock/vegetation mix |
| T43SFT  | 246.0 | 23.6   | VERY BRIGHT, low texture — significant cloud/snow |
| T43SGS  | 252.6 | 13.5   | NEAR SATURATED — cloud/snow dominant |
| T43SGT  | 250.4 | 15.7   | NEAR SATURATED — cloud/snow dominant |

### Step 2 — Tactical waypoint pixel windows (100×100px)

| Waypoint      | Lon (°E) | Lat (°N) | Tile    | mean_gray | std_gray | lap_var  | sat_frac | Assessment          |
|---------------|----------|----------|---------|-----------|----------|----------|----------|---------------------|
| WP_UDHAMPUR   | 75.15    | 33.00    | T43SES  | 212.8     | 27.0     | 1232.28  | 2.1%     | USABLE — good texture |
| WP_KARGIL     | 76.18    | 33.55    | T43SFT  | 255.0     | 0.3      | 1.49     | 99.9%    | WHITE OUT — cloud saturation |
| WP_LEH        | 77.57    | 34.17    | T43SGT  | 253.5     | 5.8      | 166.46   | 83.2%    | MARGINAL — high saturation |

### Step 3 — TerrainSuitabilityScorer at tactical waypoints

DEM tiles: TILE1 (EPSG:4326, 28.40m/px), TILE2 (EPSG:4326, 28.40m/px), TILE3 (EPSG:4326, 28.22m/px).
trn_gsd = max(10, 28.4×0.5) = 15.0m.

| Waypoint    | DEM Tile | DEM Coverage | suitability | score  | texture_var | relief_m |
|-------------|----------|--------------|-------------|--------|-------------|----------|
| WP_UDHAMPUR | TILE1    | ✓ COVERED    | ACCEPT      | 0.6457 | 271.31      | 670.63   |
| WP_KARGIL   | —        | ✗ OUTSIDE DEM COVERAGE (TILE2 east edge 76.10°E; Kargil at 76.18°E) | — | — | — | — |
| WP_LEH      | TILE3    | ✓ COVERED    | ACCEPT      | 0.9497 | 35899.29    | 189.46   |

DEM gap finding: No COP30 tile covers the Kargil region (76.18°E, 33.55°N). TILE2 ends at
76.10°E east and 33.55°N south. TILE3 starts at 34.05°N north. Gap is ~0.08° east and ~0.50°
latitude. A supplementary DEM tile centred on Kargil is needed for NAV02-CHAR-RUN4.

### Step 4 — Corridor Viability Verdict

```
┌──────────────┬────────────┬──────────────┬─────────────┬────────────┐
│   Waypoint   │ TCI Texture│ DEM Coverage │ Suitability │  Usable?   │
├──────────────┼────────────┼──────────────┼─────────────┼────────────┤
│ WP_UDHAMPUR  │ YES        │ TILE1 ✓      │ ACCEPT 0.65 │ YES        │
│ (75.15°E)    │ lap=1232   │              │             │            │
├──────────────┼────────────┼──────────────┼─────────────┼────────────┤
│ WP_KARGIL    │ NO         │ OUTSIDE ✗    │ UNKNOWN     │ NO         │
│ (76.18°E)    │ lap=1.5    │ DEM gap      │             │ (dual gap) │
│              │ 99.9% sat  │              │             │            │
├──────────────┼────────────┼──────────────┼─────────────┼────────────┤
│ WP_LEH       │ MARGINAL   │ TILE3 ✓      │ ACCEPT 0.95 │ MARGINAL   │
│ (77.57°E)    │ lap=166    │              │             │ (83% sat.) │
│              │ 83% sat    │              │             │            │
└──────────────┴────────────┴──────────────┴─────────────┴────────────┘
```

**NAV02-CHAR-RUN4 readiness:** NOT READY as currently configured.
- WP_UDHAMPUR: clear for characterisation.
- WP_KARGIL: requires (a) clear-air/summer TCI acquisition and (b) supplementary COP30 DEM tile.
- WP_LEH: DEM suitable (score 0.95), but TCI winter saturation at 83% will suppress matching.
  Acceptable for SUPPRESS/REJECT characterisation. For ACCEPT validation: clear-air acquisition needed.

**ZIP retention:** RECOMMENDED. Current TCI is Winter 2025 — unsuitable for eastern corridor
TRN matching. Do not delete ZIPs; re-acquisition from Copernicus for summer imagery is the path
forward. Deleting ZIPs does not change the TCI quality problem.

---

## 8. Open Items

| ID  | Item                                                                                 | Status |
|-----|--------------------------------------------------------------------------------------|--------|
| OI-JL-01 | TCI quality validation — complete (JL-TCI-VALIDATE 26 Apr 2026)             | **CLOSED** |
| OI-JL-02 | Extract T43SDT TCI to close Srinagar western gap                             | **DEFERRED** (Deputy 1) |
| OI-JL-03 | Assess mosaic feasibility across 6 tiles for contiguous corridor use        | OPEN   |
| OI-JL-04 | Supplementary COP30 DEM tile for Kargil gap (76.18°E, 33.55°N)             | OPEN — blocks NAV02-CHAR-RUN4 |
| OI-JL-05 | Summer/clear-air TCI acquisition for Kargil and Leh (eastern corridor)      | OPEN — blocks NAV02-CHAR-RUN4 |
| OI-JL-06 | Deputy 1 to authorise ZIP deletion — RECOMMENDATION: RETAIN (winter TCI insufficient for TRN, re-acquisition needed) | OPEN |
