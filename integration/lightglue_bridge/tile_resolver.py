"""
Maps GPS coordinates (lat, lon) to the satellite tile path on disk.

Coverage regions are derived from the programme terrain data directories.
Resolution falls back through: shimla-manali → shimla → jammu-leh.
Returns None if no tile covers the requested position.

Extra tiles (HIL assets, absolute paths) are registered via the
LIGHTGLUE_EXTRA_TILES environment variable as a JSON array:
  [{"name":"site04","lat_min":32.15,"lat_max":32.26,
    "lon_min":119.90,"lon_max":119.96,
    "tile_path":"/abs/path/satellite04.tif"}]
Extra tiles are checked first (higher priority than built-in regions).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import TILE_DIR


@dataclass(frozen=True)
class TileRegion:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    tile_path: str  # relative to TILE_DIR, or absolute if it starts with /


# Ordered from most-specific to least-specific.
_BUILT_IN_REGIONS: list[TileRegion] = [
    TileRegion(
        name="shimla_local",
        lat_min=30.9,
        lat_max=31.6,
        lon_min=76.9,
        lon_max=77.7,
        tile_path="shimla_corridor/SHIMLA-1_COP30.tif",
    ),
    TileRegion(
        name="shimla_manali",
        lat_min=31.0,
        lat_max=32.5,
        lon_min=76.9,
        lon_max=77.7,
        tile_path="shimla_manali_corridor/shimla_tile.tif",
    ),
    TileRegion(
        name="jammu_leh_tile1",
        lat_min=32.5,
        lat_max=33.6,
        lon_min=74.5,
        lon_max=76.0,
        tile_path="Jammu_leh_corridor_COP30/TILE1/rasters_COP30/output_hh.tif",
    ),
    TileRegion(
        name="jammu_leh_tile2",
        lat_min=33.5,
        lat_max=34.6,
        lon_min=74.5,
        lon_max=76.5,
        tile_path="Jammu_leh_corridor_COP30/TILE2/rasters_COP30/output_hh.tif",
    ),
    TileRegion(
        name="jammu_leh_tile3",
        lat_min=34.0,
        lat_max=35.0,
        lon_min=75.5,
        lon_max=78.0,
        tile_path="Jammu_leh_corridor_COP30/TILE3/rasters_COP30/output_hh.tif",
    ),
]


def _load_extra_regions() -> list[TileRegion]:
    """Parse LIGHTGLUE_EXTRA_TILES env var into TileRegion objects."""
    raw = os.environ.get("LIGHTGLUE_EXTRA_TILES", "").strip()
    if not raw:
        return []
    try:
        entries = json.loads(raw)
        return [
            TileRegion(
                name=e["name"],
                lat_min=float(e["lat_min"]),
                lat_max=float(e["lat_max"]),
                lon_min=float(e["lon_min"]),
                lon_max=float(e["lon_max"]),
                tile_path=e["tile_path"],
            )
            for e in entries
        ]
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "LIGHTGLUE_EXTRA_TILES parse error: %s", exc
        )
        return []


# Exposed for diagnostics (e.g. tile_resolver._REGIONS)
_REGIONS: list[TileRegion] = _load_extra_regions() + _BUILT_IN_REGIONS


def _candidate_path(region: TileRegion, base: Path) -> Path:
    """Return absolute Path for a region's tile, supporting both absolute and relative tile_path."""
    p = region.tile_path
    if os.path.isabs(p):
        return Path(p)
    return base / p


def resolve(lat: float, lon: float, tile_dir: Optional[str] = None) -> Optional[str]:
    """Return absolute path to the satellite tile covering (lat, lon), or None."""
    base = Path(tile_dir or TILE_DIR)
    for region in _REGIONS:
        if (region.lat_min <= lat <= region.lat_max and
                region.lon_min <= lon <= region.lon_max):
            candidate = _candidate_path(region, base)
            if candidate.exists():
                return str(candidate)
    return None


def region_name(lat: float, lon: float) -> Optional[str]:
    """Return the name of the matching region, or None."""
    for region in _REGIONS:
        if (region.lat_min <= lat <= region.lat_max and
                region.lon_min <= lon <= region.lon_max):
            return region.name
    return None
