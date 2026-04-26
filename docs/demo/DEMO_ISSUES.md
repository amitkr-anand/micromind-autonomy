# MicroMind Demonstration Issue Tracker
Authority: Deputy 1 | Created: 26 April 2026

| ID | Module | Description | Severity | Status |
|---|---|---|---|---|
| DI-01 | nav02_char harness | `osgeo/gdal` not installed in `micromind-autonomy` conda env. Harness falls back to `python3.12` (system, gdal 3.8.4). Rasters load correctly via system Python. Not blocking. Fix: `conda install -c conda-forge gdal` in micromind-autonomy env. | LOW | OPEN |
