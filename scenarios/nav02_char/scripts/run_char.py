"""
NAV-02 Characterisation Harness — Trinity Corridor
Scope: EC-13 partial characterisation only.
NAV-02 remains PARTIAL. No SRS closure.
Deputy 1 authorised: 26 April 2026.

Consumer of production modules — no production
file modifications permitted.
"""

import yaml
import numpy as np
from pathlib import Path

# Raster loading
try:
    from osgeo import gdal
    GDAL_OK = True
except ImportError:
    GDAL_OK = False
    print("WARNING: osgeo/gdal not available in this env")

CORRIDOR_DEF = Path(
    "scenarios/nav02_char/corridor_definition.yaml"
)

def load_corridor():
    with open(CORRIDOR_DEF) as f:
        return yaml.safe_load(f)

def load_raster_window(path, centre_e, centre_n,
                       half_width_m=500):
    """Load a square window from a raster centred on
    UTM coordinates. Returns numpy array or None."""
    if not GDAL_OK:
        return None
    ds = gdal.Open(str(path))
    if ds is None:
        return None
    gt = ds.GetGeoTransform()
    px = int((centre_e - gt[0]) / gt[1])
    py = int((centre_n - gt[3]) / gt[5])
    half_px = int(half_width_m / abs(gt[1]))
    data = ds.ReadAsArray(
        px - half_px, py - half_px,
        half_px * 2, half_px * 2
    )
    return data

def report_raster_coverage(path):
    """Report basic raster metadata."""
    if not GDAL_OK:
        print(f"  GDAL unavailable — cannot read {path}")
        return
    ds = gdal.Open(str(path))
    if ds is None:
        print(f"  Cannot open {path}")
        return
    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    print(f"  Bands: {ds.RasterCount}")
    print(f"  Size: {w} x {h} px")
    print(f"  Pixel: {abs(gt[1]):.2f} x {abs(gt[5]):.2f} m")
    print(f"  UL: ({gt[0]:.1f}, {gt[3]:.1f})")
    print(f"  LR: ({gt[0]+gt[1]*w:.1f}, "
          f"{gt[3]+gt[5]*h:.1f})")

def main():
    corridor = load_corridor()
    print(f"\nCorridor: {corridor['corridor_id']}")
    print(f"Length: {corridor['corridor_length_km']} km")
    print(f"Waypoints: {len(corridor['waypoints_utm'])}")
    print(f"\nDEM 10m:")
    report_raster_coverage(corridor['dem_10m'])
    print(f"\nDEM 30m:")
    report_raster_coverage(corridor['dem_30m'])
    print(f"\nOrthophoto RGB:")
    report_raster_coverage(corridor['orthophoto_rgb'])
    print("\nHarness ready. Awaiting Deputy 1 direction"
          " on matching integration.")

if __name__ == "__main__":
    main()
