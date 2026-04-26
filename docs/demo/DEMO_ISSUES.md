# MicroMind Demonstration Issue Tracker
Authority: Deputy 1 | Created: 26 April 2026

| ID | Module | Description | Severity | Status |
|---|---|---|---|---|
| DI-01 | nav02_char harness | `osgeo/gdal` not installed in `micromind-autonomy` conda env. Harness falls back to `python3.12` (system, gdal 3.8.4). Rasters load correctly via system Python. Not blocking. Fix: `conda install -c conda-forge gdal` in micromind-autonomy env. | LOW | OPEN |
| DI-02 | nav02_char harness | `pyproj` not installed in `micromind-autonomy` conda env. UTM→WGS84 coordinate conversion uses Deputy 1 approved fallback formula (northing/111320, lon=-123+(E-500000)/(111320*cos(lat))). Not blocking for 6 km corridor. Fix: `conda install -c conda-forge pyproj`. | LOW | OPEN |
| DI-03 | nav02_char harness / PhaseCorrelationTRN | Direct PhaseCorrelationTRN.match() calls at camera GSD (0.60 m) bypass CrossModalEvaluator GSD clamping (trn_gsd = max(camera_gsd, dem_res×0.5)). Result: 10m DEM tile upsampled 16.7× → hillshade texture_variance=5.18 < threshold 50.0 → SUPPRESS all waypoints both modes. Production path (via CrossModalEvaluator) would clamp to trn_gsd=5.0m (10m DEM) or 15.0m (30m DEM) and avoid over-upsampling. Characterisation finding: valid architectural constraint. Deputy 1 to rule on whether char run 2 should call with clamped GSD. | MEDIUM | OPEN |
| DEMO-BUG-001 | PhaseCorrelationTRN | Direct match() calls bypass CrossModalEvaluator GSD clamping (max(camera_gsd, dem_res × 0.5)). At camera_gsd=0.60m and dem_res=10m, DEM is upsampled 16.7× producing smooth hillshade with texture_variance=5.18 < threshold 50.0 → all SUPPRESS. Fix: use production-clamped GSD in harness (5.0m for 10m DEM, 15.0m for 30m DEM) or call via CrossModalEvaluator. | MEDIUM | OPEN |
