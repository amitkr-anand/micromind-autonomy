"""
integration/drivers/sim_gnss.py
MicroMind Pre-HIL — Phase 1 Driver Abstraction Layer

SimGNSSDriver: simulation GNSS driver wrapping sim/gnss_spoof_injector.py.

Uses GNSSSpoofInjector.generate() to produce GNSSMeasurement, then
converts to GNSSReading. Supports clean and spoofed operation via
the existing attack profile API.

SWaP note: MicroMind does not own the GNSS receiver. This sim driver
exists only for SIL/SITL testing. Real GNSS data arrives via
MAVLinkGNSSDriver (GPS_RAW_INT) in Phase 3.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from sim.gnss_spoof_injector import GNSSSpoofInjector, NominalGNSSState
from integration.drivers.base import DriverHealth, DriverReadError
from integration.drivers.gnss import GNSSDriver, GNSSReading, GNSSFixType

# WGS-84 reference point for ENU -> lat/lon/alt conversion.
# Default origin: Jaisalmer, Rajasthan (western corridor reference).
_DEFAULT_ORIGIN_LAT = 26.9157   # degrees
_DEFAULT_ORIGIN_LON = 70.9083   # degrees
_DEFAULT_ORIGIN_ALT = 225.0     # metres AMSL

_DEG_PER_M_LAT = 1.0 / 111_320.0
_R_EARTH_M     = 6_371_000.0


def _enu_to_geodetic(
    enu: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float,
) -> tuple[float, float, float]:
    """Convert ENU offset (metres) to geodetic (lat, lon, alt).

    Flat-earth approximation valid for corridors up to ~100 km.
    Accuracy <1m error at 50 km range — sufficient for SIL navigation.

    Args:
        enu:        ENU position offset in metres [East, North, Up].
        origin_lat: reference latitude in decimal degrees.
        origin_lon: reference longitude in decimal degrees.
        origin_alt: reference altitude in metres AMSL.

    Returns:
        (lat_deg, lon_deg, alt_m)
    """
    east_m, north_m, up_m = float(enu[0]), float(enu[1]), float(enu[2])
    lat = origin_lat + north_m * _DEG_PER_M_LAT
    lon_scale = math.cos(math.radians(origin_lat))
    lon = origin_lon + east_m * _DEG_PER_M_LAT / max(lon_scale, 1e-6)
    alt = origin_alt + up_m
    return lat, lon, alt


class SimGNSSDriver(GNSSDriver):
    """Simulation GNSS driver backed by sim/gnss_spoof_injector.py.

    Wraps GNSSSpoofInjector to produce GNSSReading at the rate requested
    by the caller. Attack profiles can be added via the injector property
    before or during a run, matching the pattern in bcmp1_runner.py.

    Args:
        nominal:           NominalGNSSState for the injector. If None,
                           injector uses its default nominal state.
        origin_lat:        geodetic reference latitude (decimal degrees).
        origin_lon:        geodetic reference longitude (decimal degrees).
        origin_alt:        geodetic reference altitude (metres AMSL).
        stale_threshold_s: staleness threshold (default 0.2s = 5Hz margin).
    """

    def __init__(
        self,
        nominal: Optional[NominalGNSSState] = None,
        origin_lat: float = _DEFAULT_ORIGIN_LAT,
        origin_lon: float = _DEFAULT_ORIGIN_LON,
        origin_alt: float = _DEFAULT_ORIGIN_ALT,
        stale_threshold_s: float = 0.2,
    ) -> None:
        super().__init__(stale_threshold_s)
        self._injector = GNSSSpoofInjector(nominal=nominal)
        self._origin_lat = origin_lat
        self._origin_lon = origin_lon
        self._origin_alt = origin_alt
        self._health_state = DriverHealth.DEGRADED
        self._closed = False
        self._mission_time_s: float = 0.0

    # ------------------------------------------------------------------
    # Public property — callers may add attack profiles directly
    # ------------------------------------------------------------------

    @property
    def injector(self) -> GNSSSpoofInjector:
        """Expose injector so attack profiles can be added externally."""
        return self._injector

    # ------------------------------------------------------------------
    # SensorDriver interface
    # ------------------------------------------------------------------

    def health(self) -> DriverHealth:
        return self._health_state

    def last_update_time(self) -> float:
        return self._last_update_time

    def is_stale(self) -> bool:
        return self._default_is_stale()

    def source_type(self) -> str:
        return 'sim'

    def read(
        self,
        true_position_enu: Optional[np.ndarray] = None,
        ew_jammer_confidence: float = 0.0,
        dt_s: float = 0.2,
    ) -> GNSSReading:
        """Generate and return the next simulated GNSS measurement.

        Advances internal mission time by dt_s on each call.

        Args:
            true_position_enu:   true ENU position for spoof injection.
                                 If None, injector uses its nominal state.
            ew_jammer_confidence: EW jammer confidence [0.0, 1.0].
            dt_s:                time step to advance mission clock (seconds).

        Returns:
            GNSSReading with lat, lon, alt, hdop, fix_type, timestamp.

        Raises:
            DriverReadError: if driver has been closed.
        """
        if self._closed:
            raise DriverReadError(
                "SimGNSSDriver: driver has been closed. Cannot read."
            )

        meas = self._injector.generate(
            mission_time_s=self._mission_time_s,
            true_position_enu=true_position_enu,
            ew_jammer_confidence=ew_jammer_confidence,
        )
        self._mission_time_s += dt_s

        # Convert ENU position to geodetic
        pos_enu = meas.gps_position_enu
        if pos_enu is not None and len(pos_enu) == 3:
            lat, lon, alt = _enu_to_geodetic(
                pos_enu, self._origin_lat, self._origin_lon, self._origin_alt
            )
        else:
            lat = self._origin_lat
            lon = self._origin_lon
            alt = self._origin_alt

        # Map pdop to hdop approximation (pdop >= hdop; use as conservative estimate)
        hdop = float(meas.pdop) if meas.pdop is not None else float('nan')

        # Infer fix type from tracked satellites and pdop
        tracked = meas.tracked_satellites if meas.tracked_satellites is not None else 0
        if tracked == 0:
            fix_type = GNSSFixType.NO_FIX
        elif tracked < 4:
            fix_type = GNSSFixType.FIX_2D
        else:
            fix_type = GNSSFixType.FIX_3D

        self._record_successful_read()
        self._health_state = DriverHealth.OK

        return GNSSReading(
            lat=lat,
            lon=lon,
            alt=alt,
            hdop=hdop,
            fix_type=fix_type,
            t=self._last_update_time,
        )

    def close(self) -> None:
        """Clear attack profiles and mark driver closed."""
        if not self._closed:
            self._injector.clear_attacks()
            self._closed = True
