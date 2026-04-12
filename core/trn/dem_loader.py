"""
core/trn/dem_loader.py
MicroMind / NanoCorteX — DEM Loader

NAV-02: Real terrain elevation data loader.
Supports Copernicus GLO-30 GeoTIFF format (WGS84 EPSG:4326, 30 m resolution).

Sensor substitution contract:
    Current source: Copernicus GLO-30 GeoTIFF (OpenTopography)
    Real replacement: Onboard terrain package (mission data card, same format)
    Interface: identical — only file path changes at HIL

References:
    SRS v1.3 NAV-02
    Part Two V7.2 §1.7.2
    CAS paper: Wan et al. 2021 (phase correlation TRN, Eq. 19 GSD formulation)
    docs/interfaces/dem_contract.yaml
"""
from __future__ import annotations

import math

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.transform import array_bounds, rowcol, xy
from rasterio.warp import reproject, Resampling


_WGS84_EPSG = 4326
# 1 degree of latitude ≈ 111 320 m (WGS-84 sphere approximation, sufficient for tile sizing)
_M_PER_DEG_LAT: float = 111_320.0


class DEMLoader:
    """
    NAV-02: Real terrain elevation data loader.
    Supports Copernicus GLO-30 GeoTIFF format (WGS84 EPSG:4326, 30 m resolution).

    Sensor substitution contract:
        Current source: Copernicus GLO-30 GeoTIFF
        Real replacement: Onboard terrain package (same GeoTIFF format, pre-loaded
            mission data card). Interface: identical — only file path changes.
        Interface contract: docs/interfaces/dem_contract.yaml
    """

    def __init__(self, dem_path: str) -> None:
        """
        Load DEM from GeoTIFF file.

        Uses rasterio for file access.
        Raises FileNotFoundError if path is invalid.
        Raises ValueError if CRS is not EPSG:4326.
        Stores elevation array, transform, and bounds in memory on load.
        """
        import os
        if not os.path.exists(dem_path):
            raise FileNotFoundError(f"DEM file not found: {dem_path}")

        with rasterio.open(dem_path) as ds:
            crs = ds.crs
            if crs is None:
                raise ValueError(
                    f"DEM file has no CRS — expected EPSG:{_WGS84_EPSG}"
                )
            if not crs.to_epsg() == _WGS84_EPSG:
                raise ValueError(
                    f"DEM CRS is {crs.to_string()} — expected EPSG:{_WGS84_EPSG}"
                )
            # Read the first band as float32
            self._elevation: np.ndarray = ds.read(1).astype(np.float32)
            self._transform = ds.transform
            self._nodata = ds.nodata
            self._crs = crs
            self._bounds = ds.bounds
            self._height, self._width = self._elevation.shape

        # Replace nodata with NaN for clean arithmetic
        if self._nodata is not None:
            self._elevation[self._elevation == self._nodata] = float("nan")

        # Approximate DEM pixel resolution in metres (X direction, along latitude midpoint)
        lat_mid = (self._bounds.top + self._bounds.bottom) / 2.0
        deg_per_pixel_x = abs(self._transform.a)    # longitude degrees per pixel
        deg_per_pixel_y = abs(self._transform.e)    # latitude  degrees per pixel
        m_per_deg_lon = _M_PER_DEG_LAT * math.cos(math.radians(lat_mid))
        self._resolution_m: float = (
            deg_per_pixel_x * m_per_deg_lon + deg_per_pixel_y * _M_PER_DEG_LAT
        ) / 2.0

    # ── Public API ────────────────────────────────────────────────────────────

    def get_elevation(self, lat: float, lon: float) -> float:
        """
        Return elevation in metres at WGS84 coordinate.
        Bilinear interpolation between grid cells.
        Returns NaN if coordinate is outside DEM bounds.
        """
        if not self.is_in_bounds(lat, lon):
            return float("nan")

        # Fractional row, col in pixel space
        row_f, col_f = rowcol(self._transform, lon, lat, offset="center")
        row_f = float(row_f)
        col_f = float(col_f)

        # Bilinear interpolation
        r0 = int(math.floor(row_f))
        c0 = int(math.floor(col_f))
        r1 = r0 + 1
        c1 = c0 + 1
        dr = row_f - r0
        dc = col_f - c0

        h, w = self._height, self._width

        def _safe(r: int, c: int) -> float:
            if 0 <= r < h and 0 <= c < w:
                return float(self._elevation[r, c])
            return float("nan")

        v00 = _safe(r0, c0)
        v01 = _safe(r0, c1)
        v10 = _safe(r1, c0)
        v11 = _safe(r1, c1)

        if any(math.isnan(v) for v in (v00, v01, v10, v11)):
            # Fall back to nearest valid sample
            for v in (v00, v01, v10, v11):
                if not math.isnan(v):
                    return v
            return float("nan")

        return (
            v00 * (1 - dr) * (1 - dc)
            + v01 * (1 - dr) * dc
            + v10 * dr * (1 - dc)
            + v11 * dr * dc
        )

    def get_tile(
        self,
        lat_centre: float,
        lon_centre: float,
        tile_size_m: float,
        gsd_m: float,
    ) -> np.ndarray:
        """
        NAV-02: Extract a terrain tile centred at the given coordinate.

        tile_size_m : physical size of tile in metres (e.g. 500 m × 500 m)
        gsd_m       : ground sample distance in metres — determines output pixel
                      resolution. Derived from flight altitude and camera geometry
                      (CAS paper Eq. 19). Accepted as explicit parameter here;
                      Phase D will compute from camera model.

        Returns: float32 numpy array of elevation values resampled to gsd_m
                 resolution. Shape (H, W) where H = W = int(tile_size_m / gsd_m).
        Returns array filled with NaN if centre is outside DEM bounds.
        """
        n_pixels = int(tile_size_m / gsd_m)
        if n_pixels < 1:
            raise ValueError(
                f"tile_size_m/gsd_m must be >= 1 (got {tile_size_m}/{gsd_m})"
            )

        if not self.is_in_bounds(lat_centre, lon_centre):
            return np.full((n_pixels, n_pixels), float("nan"), dtype=np.float32)

        lat_mid = lat_centre
        m_per_deg_lon_local = _M_PER_DEG_LAT * math.cos(math.radians(lat_mid))

        half_m = tile_size_m / 2.0
        half_deg_lat = half_m / _M_PER_DEG_LAT
        half_deg_lon = half_m / m_per_deg_lon_local if m_per_deg_lon_local > 0 else 0.0

        lat_north = lat_centre + half_deg_lat
        lat_south = lat_centre - half_deg_lat
        lon_west  = lon_centre - half_deg_lon
        lon_east  = lon_centre + half_deg_lon

        # Clamp to DEM bounds
        lat_north = min(lat_north, self._bounds.top)
        lat_south = max(lat_south, self._bounds.bottom)
        lon_west  = max(lon_west,  self._bounds.left)
        lon_east  = min(lon_east,  self._bounds.right)

        # Convert geographic extent to pixel window
        row_top, col_left   = rowcol(self._transform, lon_west,  lat_north)
        row_bot, col_right  = rowcol(self._transform, lon_east,  lat_south)

        row_top   = max(0, int(row_top))
        col_left  = max(0, int(col_left))
        row_bot   = min(self._height - 1, int(row_bot))
        col_right = min(self._width  - 1, int(col_right))

        if row_bot <= row_top or col_right <= col_left:
            return np.full((n_pixels, n_pixels), float("nan"), dtype=np.float32)

        patch = self._elevation[row_top:row_bot + 1, col_left:col_right + 1]

        # Resample patch to (n_pixels, n_pixels) using bilinear interpolation
        from scipy.ndimage import zoom
        if patch.shape[0] < 2 or patch.shape[1] < 2:
            return np.full((n_pixels, n_pixels), float("nan"), dtype=np.float32)

        zoom_r = n_pixels / patch.shape[0]
        zoom_c = n_pixels / patch.shape[1]
        resampled = zoom(patch, (zoom_r, zoom_c), order=1, mode="nearest")
        return resampled.astype(np.float32)

    def get_bounds(self) -> dict:
        """
        Return DEM geographic bounds:
            {'north': float, 'south': float, 'east': float, 'west': float,
             'resolution_m': float}
        """
        return {
            "north":        float(self._bounds.top),
            "south":        float(self._bounds.bottom),
            "east":         float(self._bounds.right),
            "west":         float(self._bounds.left),
            "resolution_m": float(self._resolution_m),
        }

    def is_in_bounds(self, lat: float, lon: float) -> bool:
        """Return True if coordinate is within loaded DEM extent."""
        return (
            self._bounds.bottom <= lat <= self._bounds.top
            and self._bounds.left  <= lon <= self._bounds.right
        )

    # ── Multi-tile constructor ─────────────────────────────────────────────────

    @classmethod
    def from_directory(cls, terrain_dir: str) -> "DEMLoader":
        """
        Load and merge all GeoTIFF files in terrain_dir into a single DEMLoader.

        Production path: at HIL the terrain package is a directory of tiles
        covering the full mission corridor. from_directory() loads them all
        without needing to know how many tiles are present.

        Uses rasterio.merge.merge() for multi-tile stitching.

        Raises FileNotFoundError if terrain_dir contains no .tif files.
        Raises ValueError if any tile CRS is not EPSG:4326.
        Raises ValueError if tiles have mismatched CRS.
        """
        import glob
        import os
        from rasterio.merge import merge

        tif_files = sorted(glob.glob(os.path.join(terrain_dir, "*.tif")))
        if not tif_files:
            raise FileNotFoundError(
                f"No .tif files found in terrain directory: {terrain_dir}"
            )

        # Single tile: fast path — delegate to __init__
        if len(tif_files) == 1:
            return cls(tif_files[0])

        # Multi-tile: verify CRS compatibility and merge
        datasets = []
        try:
            ref_epsg = None
            for path in tif_files:
                ds = rasterio.open(path)
                if ds.crs is None:
                    raise ValueError(f"DEM tile has no CRS — expected EPSG:{_WGS84_EPSG}: {path}")
                epsg = ds.crs.to_epsg()
                if epsg != _WGS84_EPSG:
                    raise ValueError(
                        f"DEM tile CRS is EPSG:{epsg} — expected EPSG:{_WGS84_EPSG}: {path}"
                    )
                if ref_epsg is None:
                    ref_epsg = epsg
                elif epsg != ref_epsg:
                    raise ValueError(
                        f"CRS mismatch between tiles: {path} (EPSG:{epsg}) vs "
                        f"first tile (EPSG:{ref_epsg})"
                    )
                datasets.append(ds)

            merged_data, merged_transform = merge(datasets)
            nodata = datasets[0].nodata
            crs    = datasets[0].crs

        finally:
            for ds in datasets:
                ds.close()

        # Build DEMLoader instance from merged in-memory data
        instance = cls.__new__(cls)
        instance._elevation = merged_data[0].astype(np.float32)
        instance._transform = merged_transform
        instance._nodata    = nodata
        instance._crs       = crs

        h, w = instance._elevation.shape
        instance._height = h
        instance._width  = w

        # Compute bounds from merged array dimensions + transform
        west, south, east, north = array_bounds(h, w, merged_transform)
        instance._bounds = BoundingBox(west, south, east, north)

        # Replace nodata with NaN
        if instance._nodata is not None:
            instance._elevation[instance._elevation == instance._nodata] = float("nan")

        # Approximate resolution
        lat_mid = (instance._bounds.top + instance._bounds.bottom) / 2.0
        deg_per_pixel_x = abs(merged_transform.a)
        deg_per_pixel_y = abs(merged_transform.e)
        m_per_deg_lon   = _M_PER_DEG_LAT * math.cos(math.radians(lat_mid))
        instance._resolution_m = (
            deg_per_pixel_x * m_per_deg_lon + deg_per_pixel_y * _M_PER_DEG_LAT
        ) / 2.0

        return instance
