# Terrain-Class Dataset Matrix v1.0
Authority: Deputy 1 | Date: 26 April 2026

## Available DEM and Optical Datasets

| Dataset | Path | Status | Terrain Class |
|---|---|---|---|
| Shimla-Manali COP30 | data/terrain/Shimla_Manali_Corridor/ | IN USE | Mountain structured + valley floor |
| Shimla-1 COP30 | data/terrain/shimla_corridor/ | IN USE | Mountain structured |
| Jammu-Leh TILE1/2/3 | data/terrain/Jammu_leh_corridor_COP30/ | AVAILABLE | Mountain pass + high-altitude desert |
| Spiti Valley COP30 | micromind_data/raw/dem/Spiti Valley/ | AVAILABLE-UNUSED | High-altitude desert/semi-arid |
| Low observability plains -1 | micromind_data/raw/dem/low_observability_plains_corridor -1/ | AVAILABLE-UNUSED | Flat agricultural — SUPPRESS expected |
| Low observability plains -2 | micromind_data/raw/dem/low_observability_plains_corridor -2/ | AVAILABLE-UNUSED | Flat agricultural — SUPPRESS expected |
| Mixed observability corridor | micromind_data/raw/dem/mixed_observability_operational_corridor/ | AVAILABLE-UNUSED | Mixed — CAUTION/ACCEPT transition |
| Srinagar DEM | Documents/MicroMind-X/.../DEM FILES/srinagar_dem.tif | AVAILABLE-UNUSED | Kashmir valley — SAL-3 suppression gap |
| Colorado USGS 10m | micromind_data/raw/dem/colorado_test/ | AVAILABLE-UNUSED | Algorithm testing only |
| Sentinel TCI mosaic | Downloads/sentinel_tci_mosaic.tif | AVAILABLE — coverage TBD | Optical landmark reference |
| UAV VisLoc (11 sites) | micromind_data/raw/imagery/uav_visloc/UAV_VisLoc/ | AVAILABLE-UNUSED | OI-07 VIO validation |
| INRIA Aerial (urban) | micromind_data/raw/inria/ | AVAILABLE-UNUSED | Urban landmark detection testing |
| Indus Basin PDF | micromind_data/raw/dem/ | REFERENCE | Hydrological landmark reference |

## Terrain-Class Profiles

| Terrain Class | Corridor Segment | SAL-2 Band | SAL-3 Required | Dataset Available |
|---|---|---|---|---|
| Mountain structured (ridges) | Shimla-Manali km 0-60 | ACCEPT | No | Shimla-Manali COP30 |
| Kashmir valley floor | Shimla-Manali km 60-120 | SUPPRESS | YES | Srinagar DEM + Sentinel TCI |
| Mountain pass (Rohtang) | Shimla-Manali km 100-120 | CAUTION-SUPPRESS | Boundary | Shimla-Manali COP30 |
| High-altitude desert | Jammu-Leh km 280-330 | SUPPRESS | YES | JL TILE3 |
| Plains corridor | low_observability -1/-2 | SUPPRESS | YES (stress test) | Available-unused |
| Mixed observability | mixed_observability corridor | CAUTION | Boundary | Available-unused |
| Spiti high altitude | Spiti Valley | SUPPRESS | YES (AVP-04) | Available-unused |

## Seasonality Risk Register
| Terrain | Risk | Mitigation |
|---|---|---|
| Kashmir valley | High (monsoon Jul-Sep) | Use Oct-Nov imagery — confirmed 2025-10-13 in Copernicus Browser |
| Riverbank corridors | High (flood season) | Acquisition_season tag mandatory on landmark map (SAL-3 prerequisite 8) |
| High-altitude passes | Medium (snow cover Nov-May) | Separate summer/winter landmark maps for Zoji La, Rohtang |
| Plains | Low | Crop rotation changes texture — bi-annual revalidation |

## Priority Actions
1. Characterise Sentinel TCI mosaic coverage (run gdal coordinates check)
2. Register Srinagar DEM to Shimla-Manali corridor coordinate system
3. Run landmark density analysis on Spiti Valley for AVP-04 feasibility
4. Use low_observability_plains corridors for SAL-3 SUPPRESS stress testing
