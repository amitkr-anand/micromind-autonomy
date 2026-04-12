# DEMLoader — Technical Notes

## Sensor Substitution Contract

| Field | Value |
|---|---|
| Current source | Copernicus GLO-30 GeoTIFF (OpenTopography, 30 m, WGS84) |
| HIL replacement | Onboard terrain package loaded from mission data card (same GeoTIFF format). Only file path changes. Architecture unchanged. |
| Interface contract | `docs/interfaces/dem_contract.yaml` |
| breaking_change_at_HIL | false |

## GSD Formulation

Based on CAS paper (Wan et al. 2021) Eq. 19. GSD varies with altitude and
terrain relief within the camera FOV:

```
GSD = 2 × (alt_m − min_terrain_m) / (sensor_pixels × tan(fov/2))
```

Current implementation accepts explicit `gsd_m` parameter in `get_tile()`.
Phase D will compute GSD from altitude and camera geometry model.

## Coordinate System

All coordinates in WGS84 (EPSG:4326).
Elevation in metres above WGS84 ellipsoid / EGM2008 geoid (EPSG:3855)
as per Copernicus GLO-30 specification.

## Resampling

`get_tile()` uses scipy bilinear zoom (order=1). For production HIL,
replace with rasterio's Resampling.bilinear if scipy is not available.
The interface is identical — numpy float32 array in, numpy float32 array out.

## Resolution Estimate

`_resolution_m` is an approximate average of X and Y pixel size in metres
at the DEM's central latitude. For Copernicus GLO-30 over mid-latitudes,
this is approximately 27–30 m. Used by `TerrainSuitabilityScorer` as the
reference DEM resolution for GSD validity checking.
