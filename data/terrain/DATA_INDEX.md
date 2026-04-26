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
