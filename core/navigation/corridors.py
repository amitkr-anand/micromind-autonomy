"""
core/navigation/corridors.py
MicroMind / NanoCorteX — Named Mission Corridor Definitions

Defines mission corridors as structured data objects.
Not hardcoded in tests or scripts — all corridor-specific parameters
(waypoints, distances, terrain_dir, GNSS denial profile) live here.

At HIL: terrain_dir points to the onboard terrain package (mission data card).
At SIL: terrain_dir points to Copernicus GLO-30 GeoTIFF files in data/terrain/.

Req IDs: NAV-02, NAV-03, EC-09, EC-11
SRS ref: §2.2, §2.3, AD-16
Governance: Code Governance Manual v3.4
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Haversine utility
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points in kilometres."""
    R = 6371.0
    phi1  = math.radians(lat1)
    phi2  = math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    )
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


# ---------------------------------------------------------------------------
# MissionCorridor
# ---------------------------------------------------------------------------

@dataclass
class MissionCorridor:
    """
    Structured mission corridor definition.

    Attributes
    ----------
    name                  : Short identifier (used in log keys, report titles)
    waypoints             : Ordered (lat, lon) WGS84 waypoints defining the corridor axis.
                            First = launch point, last = terminal point.
    total_distance_km     : Authoritative corridor length in km.
                            May differ from Haversine sum of waypoints if the
                            route curves between waypoints.
    terrain_dir           : Directory containing Copernicus GLO-30 GeoTIFF tile(s).
                            Passed to DEMLoader.from_directory() at runtime.
    gnss_denial_start_km  : Mission km at which GNSS is denied.
                            0.0 = denied from launch.
    gnss_denial_end_km    : Mission km at which GNSS denial ends.
                            -1.0 = denied to end of mission.
    description           : Human-readable corridor summary.
    terrain_zones         : Optional list of terrain zone annotation dicts.
                            Each dict: {name, km_start, km_end, character,
                            expected_suitability, notes}.
    """
    name:                   str
    waypoints:              List[Tuple[float, float]]  # (lat, lon) WGS84
    total_distance_km:      float
    terrain_dir:            str
    gnss_denial_start_km:   float
    gnss_denial_end_km:     float   # -1.0 = denied to mission end
    description:            str
    terrain_zones:          List[dict] = field(default_factory=list)

    def position_at_km(self, km: float) -> Tuple[float, float]:
        """
        Return (lat, lon) at mission distance km by linear interpolation
        along the waypoint path.

        Waypoint-to-waypoint distances are computed via Haversine and then
        proportionally rescaled to sum to total_distance_km, so that the
        returned position is consistent with the authoritative distance field.

        Parameters
        ----------
        km : mission distance (0.0 = launch, total_distance_km = terminal)

        Returns
        -------
        (lat, lon) WGS84
        """
        if km <= 0.0:
            return self.waypoints[0]
        if km >= self.total_distance_km:
            return self.waypoints[-1]

        n = len(self.waypoints)
        if n == 1:
            return self.waypoints[0]

        # Compute raw segment distances
        raw_dists = [
            _haversine_km(
                self.waypoints[i][0], self.waypoints[i][1],
                self.waypoints[i + 1][0], self.waypoints[i + 1][1],
            )
            for i in range(n - 1)
        ]
        raw_total = sum(raw_dists) or 1.0  # guard against zero (co-located waypoints)

        # Scale each segment to be proportional within total_distance_km
        scale = self.total_distance_km / raw_total
        seg_kms = [d * scale for d in raw_dists]

        # Walk segments to find the one containing km
        cumulative = 0.0
        for i, seg_km in enumerate(seg_kms):
            next_cum = cumulative + seg_km
            if km <= next_cum:
                t = (km - cumulative) / seg_km if seg_km > 0.0 else 0.0
                lat0, lon0 = self.waypoints[i]
                lat1, lon1 = self.waypoints[i + 1]
                lat = lat0 + t * (lat1 - lat0)
                lon = lon0 + t * (lon1 - lon0)
                return (lat, lon)
            cumulative = next_cum

        # Numerical safety: return last waypoint if km slightly beyond total
        return self.waypoints[-1]

    def waypoint_bearing_deg(self, idx: int) -> float:
        """
        Initial bearing from waypoints[idx] to waypoints[idx+1] in degrees [0, 360).
        Raises IndexError if idx >= len(waypoints) - 1.
        """
        if idx >= len(self.waypoints) - 1:
            raise IndexError(
                f"idx={idx} has no successor in {len(self.waypoints)}-waypoint corridor"
            )
        lat1, lon1 = self.waypoints[idx]
        lat2, lon2 = self.waypoints[idx + 1]
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dlam = math.radians(lon2 - lon1)
        y = math.sin(dlam) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
        bearing = math.degrees(math.atan2(y, x))
        return bearing % 360.0


# ---------------------------------------------------------------------------
# Programme-defined corridors
# ---------------------------------------------------------------------------

SHIMLA_MANALI = MissionCorridor(
    name="shimla_manali",
    waypoints=[
        (31.104, 77.173),   # Shimla
        (31.250, 77.100),   # Fagu ridge
        (31.450, 77.050),   # Narkanda
        (31.700, 77.000),   # Rampur
        (31.900, 77.100),   # Tapri
        (32.050, 77.150),   # Karcham
        (32.200, 77.180),   # Kalpa ridge
        (32.240, 77.190),   # Manali approach
    ],
    total_distance_km=180.0,
    terrain_dir="data/terrain/shimla_manali_corridor/",
    gnss_denial_start_km=10.0,
    gnss_denial_end_km=-1.0,
    description=(
        "Shimla to Manali via Sutlej valley and Kullu. "
        "Three terrain zones: forested ridge (0–60 km), "
        "river gorge (60–120 km), high alpine (120–180 km)."
    ),
    terrain_zones=[
        {
            "name": "Shimla Ridge",
            "km_start": 0,
            "km_end": 60,
            "character": "forested_ridge",
            "expected_suitability": "ACCEPT",
            "notes": (
                "Dense Himalayan forest, high relief, strong TRN signal"
            ),
        },
        {
            "name": "Sutlej-Beas Gorge",
            "km_start": 60,
            "km_end": 120,
            "character": "river_gorge",
            "expected_suitability": "CAUTION",
            "notes": (
                "Narrow gorge, steep walls, river valley floor suppressed, "
                "wall faces variable"
            ),
        },
        {
            "name": "Kullu-Manali Alpine",
            "km_start": 120,
            "km_end": 180,
            "character": "high_alpine",
            "expected_suitability": "ACCEPT",
            "notes": (
                "High relief, open terrain, strong TRN signal. "
                "Snow cover may reduce reliability — noted as future "
                "thermal sensing use case per Addendum v2"
            ),
        },
    ],
)

SHIMLA_LOCAL = MissionCorridor(
    name="shimla_local",
    waypoints=[
        (31.104, 77.173),   # Shimla
        (31.250, 77.400),   # Rampur direction
    ],
    total_distance_km=55.0,
    terrain_dir="data/terrain/shimla_corridor/",
    gnss_denial_start_km=5.0,
    gnss_denial_end_km=-1.0,
    description="Shimla local corridor used in Gates 1–3.",
)

JAMMU_LEH = MissionCorridor(
    name="JAMMU_LEH",
    waypoints=[
        (32.73, 74.87),   # Jammu — corridor start
        (32.92, 75.13),   # Udhampur
        (33.17, 75.08),   # Ramban — Chenab gorge
        (33.50, 75.19),   # Banihal Pass (2,832m)
        (34.08, 74.80),   # Srinagar — valley floor
        (34.30, 75.29),   # Sonamarg (2,740m)
        (34.35, 75.47),   # Zoji La (3,528m) — tactical chokepoint
        (34.43, 75.76),   # Drass (3,280m)
        (34.56, 76.13),   # Kargil (2,676m)
        (34.17, 77.58),   # Leh (3,524m) — corridor end
    ],
    total_distance_km=330.0,
    terrain_dir="data/terrain/Jammu_leh_corridor_COP30/",
    gnss_denial_start_km=30.0,
    gnss_denial_end_km=330.0,
    description=(
        "Jammu to Leh via NH-1. "
        "Four terrain zones: Shivalik foothills (0–90 km), "
        "Pir Panjal / Banihal (90–150 km), "
        "Kashmir valley and Zoji La (150–210 km), "
        "Ladakh high plateau (210–330 km)."
    ),
    terrain_zones=[
        {
            "name": "Shivalik Foothills",
            "km_start": 0,
            "km_end": 90,
            "character": "forested_ridge",
            "expected_suitability": "ACCEPT",
            "notes": (
                "Shivalik range, moderate relief, forested ridgelines, "
                "strong TRN signal expected"
            ),
        },
        {
            "name": "Pir Panjal — Banihal",
            "km_start": 90,
            "km_end": 150,
            "character": "high_alpine",
            "expected_suitability": "ACCEPT",
            "notes": (
                "Chenab gorge and Banihal Pass, high relief, "
                "terrain well-differentiated"
            ),
        },
        {
            "name": "Kashmir Valley — Zoji La",
            "km_start": 150,
            "km_end": 210,
            "character": "river_gorge",
            "expected_suitability": "CAUTION",
            "notes": (
                "Kashmir valley floor low relief then Zoji La ascent; "
                "valley floor may suppress TRN score"
            ),
        },
        {
            "name": "Ladakh High Plateau",
            "km_start": 210,
            "km_end": 330,
            "character": "high_alpine",
            "expected_suitability": "ACCEPT",
            "notes": (
                "Drass, Kargil, Zanskar ranges, extreme relief, "
                "strong TRN signal expected; snow cover seasonal risk"
            ),
        },
    ],
)
