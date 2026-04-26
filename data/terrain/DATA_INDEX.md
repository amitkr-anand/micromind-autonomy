# MicroMind Terrain Data Index
**Last updated:** 2026-04-26
**Maintainer:** Programme Director
**Governance:** All raw DEM files are Copernicus GLO-30 (30m, EPSG:4326).
Viz derivatives (hillshade, color-relief, slope, roughness) are generated outputs —
never commit derivatives to git. Canonical raw files only.

## Active Corridors

| Corridor | File | Path (repo-relative) | MD5 | Bounds (WGS84) | Gate |
|---|---|---|---|---|---|
| Shimla Local | SHIMLA-1_COP30.tif | data/terrain/shimla_corridor/ | 28f960b8 | 76.60–77.68°E, 30.93–31.44°N | Gates 1–4 |
| Shimla-Manali (south) | shimla_tile.tif *(symlink→shimla_corridor)* | data/terrain/shimla_manali_corridor/ | 28f960b8 | same as Shimla Local | Gate 5 |
| Shimla-Manali (north) | manali_tile.tif | data/terrain/shimla_manali_corridor/ | f06a726f | 76.88–77.28°E, 31.40–32.32°N | Gate 5 |
| Jammu-Leh TILE1 | output_hh.tif | data/terrain/Jammu_leh_corridor_COP30/TILE1/rasters_COP30/ | 78c095da | 74.80–75.80°E, 32.80–33.60°N | Active |
| Jammu-Leh TILE2 | output_hh.tif | data/terrain/Jammu_leh_corridor_COP30/TILE2/rasters_COP30/ | f51b1ed8 | 74.75–76.10°E, 33.55–34.52°N | Active |
| Jammu-Leh TILE3 | output_hh.tif | data/terrain/Jammu_leh_corridor_COP30/TILE3/rasters_COP30/ | 62388636 | 75.60–77.75°E, 34.05–34.70°N | Active |

## Future Corridors (not yet ingested to repo)

| Corridor | File | Canonical store path | MD5 | Notes |
|---|---|---|---|---|
| Srinagar | srinagar_dem.tif | micromind_data/raw/dem/srinagar/ | 7f9e3665 | Smaller tile (74.3–75.6°E) |
| Srinagar (ext) | srinagar_dem_ext.tif | micromind_data/raw/dem/srinagar/ | 96e3666e | Full coverage (74.3–78.9°E) — USE THIS for corridor ingestion |

## Optical Imagery

| File | Store path | MD5 | Coverage | Notes |
|---|---|---|---|---|
| sentinel_tci_mosaic.tif | micromind_data/raw/imagery/sentinel/ | 38393ece | 76.1–78.2°E, 30.6–31.6°N | Sentinel-2 TCI, 10m GSD |

## Derivative Rules

1. All viz derivatives live under `viz/` subdirectories alongside their source raw file
2. Viz derivatives are NOT committed to git (add to `.gitignore` if not already excluded)
3. To regenerate: use OpenTopography GDAL pipeline or `scripts/generate_viz_derivatives.sh` (TBD)
4. Symlinks: `shimla_tile.tif` and `simulation/terrain/shimla/SHIMLA-1_COP30.tif` are symlinks —
   do not replace with real files

## Housekeeping Log

| Date | Action | Recovery |
|---|---|---|
| 2026-04-26 | Phase 1+2: Downloads/Trash cleanup + sentinel_tci_mosaic_utm43 deletion; srinagar DEMs rescued to micromind_data/raw/dem/srinagar/; sentinel mosaic moved to micromind_data/raw/imagery/sentinel/ | ~2.51 GB |
| 2026-04-26 | Phase 3: Blender sandbox dedup — 14 Group C viz files + 6 Group E shimla_corridor files deleted | ~652 MB |
| 2026-04-26 | Phase 4: SHIMLA-1 symlinks — simulation/terrain/shimla/ + shimla_manali_corridor/shimla_tile.tif | ~58 MB |

---

## Sentinel-2 Imagery (L2A, 10m TCI)

| Tile | TCI File | Canonical Path | Acquisition | Coverage |
|------|----------|----------------|-------------|----------|
| T43RGQ | T43RGQ_20251017T053241_TCI_10m.jp2 | micromind_data/raw/imagery/sentinel2/shimla_corridor/T43RGQ_20251017.../GRANULE/.../IMG_DATA/R10m/ | 2025-10-17 | Eastern Shimla (77°E+), ~135 MB |
| T43RFQ | T43RFQ_20251017T053241_TCI_10m.jp2 | micromind_data/raw/imagery/sentinel2/shimla_corridor/T43RFQ_20251017.../GRANULE/.../IMG_DATA/R10m/ | 2025-10-17 | Western Shimla (76.8°E), ~135 MB |

Full SAFE packages retained (QI_DATA cloud/snow masks included).
Do NOT copy SAFE packages into the repo. Reference TCI JP2 by absolute path.

## UAV VisLoc Dataset — LightGlue Validation Corpus

**Canonical path:** `micromind_data/raw/imagery/uav_visloc/UAV_VisLoc/`
**Symlink (sandbox backward compat):** `micromind_blender_sandbox/matching_sandbox/data/uav_visloc/UAV_VisLoc/`
**Source:** Xu et al., arXiv:2405.11936, 2024. DO NOT DELETE.

| Site | File | Size | Status |
|------|------|------|--------|
| 01 | satellite01.tif | 0.81 GB | Active |
| 02 | satellite02.tif | 1.19 GB | Active |
| 03 | satellite03.tif | 2.58 GB | Active |
| 04 | satellite04.tif | 2.11 GB | **Primary HIL benchmark** (H-4/H-5/H-6). Orin copy at /home/mmuser-orin/hil_benchmark/ |
| 05 | satellite05.tif | 0.17 GB | Active |
| 06 | satellite06.tif | 0.25 GB | Active |
| 07 | — | — | UAV frames only, no satellite tile |
| 08 | satellite08.tif | 2.85 GB | Excluded QA-042 (temporal change — greenhouses post-acquisition) |
| 09 | satellite09_01-01/01-02/02-01/02-02.tif | 4.47 GB | 4 sub-tiles; excluded QA-042 (highway construction) |
| 10 | satellite10.tif | 0.10 GB | Active |
| 11 | satellite11.tif | 1.48 GB | Active |

## Other Imagery

| File | Canonical Path | Size | Notes |
|------|----------------|------|-------|
| ORTHOUV2018.tif | micromind_data/raw/imagery/colombia/ | 157 MB | Colombia orthophoto — speculative, not yet gate-tested |
| output_hh_utm43.tif | micromind_data/raw/dem/shimla_local/ | 32.7 MB | UTM43N reprojection of Shimla DEM raw (WGS84 source: output_hh.tif) |
| sentinel_tci_mosaic.tif | micromind_data/raw/imagery/sentinel/ | 691 MB | Sentinel-2 TCI mosaic, 10m GSD, UTM43N metric space |
| sentinel_tci_mosaic_utm43.tif | micromind_data/raw/imagery/sentinel/ | 691 MB | UTM43N reprojection — retain, different projection from mosaic |

## Imagery Housekeeping Log

| Date | Action | Recovery |
|------|--------|----------|
| 2026-04-26 | Phase I-1: Deleted S2 SAFE QI_DATA from repo | 5.4 MB |
| 2026-04-26 | Phase I-2: Moved S2 SAFE packages to imagery/sentinel2/ | structural |
| 2026-04-26 | Phase I-3: Moved UAV VisLoc (16 GB) to micromind_data/raw/imagery/; symlink at sandbox | structural |
| 2026-04-26 | Phase I-4: Moved Colombia orthophoto to micromind_data/raw/imagery/ | structural |
| 2026-04-26 | Phase I-5: Moved output_hh_utm43.tif to micromind_data/raw/dem/shimla_local/ | structural |
