"""
integration/drivers/sim_radalt.py
MicroMind Pre-HIL — Phase 1 Driver Abstraction Layer

SimRADALTDriver: simulation RADALT driver using DEMProvider terrain elevation.

Computes height above ground level as:
    alt_agl_m = vehicle_alt_amsl_m - dem.patch(north_m, east_m, 1, 1)[0, 0]

The vehicle altitude AMSL is supplied on each read() call. This matches
the pattern used by als250_nav_sim.py where altitude comes from the INS
state vector.

SWaP note: MicroMind does not own the RADALT. This sim driver exists only
for SIL/SITL testing. Real RADALT data arrives via MAVLinkRADALTDriver
(DISTANCE_SENSOR) in Phase 3.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from core.ins.trn_stub import DEMProvider
from integration.drivers.base import DriverHealth, DriverReadError
from integration.drivers.radalt import RADALTDriver, RADALTReading


class SimRADALTDriver(RADALTDriver):
    """Simulation RADALT driver backed by core/ins/trn_stub.DEMProvider.

    Queries terrain elevation at the current position and computes AGL
    altitude. Returns validity_flag=False when the vehicle altitude is
    below terrain (should not occur in normal SIL runs).

    Args:
        seed:              DEMProvider seed for terrain reproducibility.
        stale_threshold_s: staleness threshold (default 0.1s = 10Hz margin).
    """

    def __init__(
        self,
        seed: int = 42,
        stale_threshold_s: float = 0.1,
    ) -> None:
        super().__init__(stale_threshold_s)
        self._dem = DEMProvider(seed=seed)
        self._health_state = DriverHealth.DEGRADED
        self._closed = False

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
        north_m: float = 0.0,
        east_m: float = 0.0,
        vehicle_alt_amsl_m: float = DEMProvider._MEAN_ALT_M + 50.0,
    ) -> RADALTReading:
        """Compute and return simulated RADALT AGL altitude.

        Args:
            north_m:            vehicle north position in metres from DEM origin.
            east_m:             vehicle east position in metres from DEM origin.
            vehicle_alt_amsl_m: vehicle altitude above mean sea level in metres.
                                Defaults to mean terrain altitude + 50m.

        Returns:
            RADALTReading with alt_agl_m, validity_flag, and timestamp.
            validity_flag=False if computed AGL is negative (below terrain).

        Raises:
            DriverReadError: if driver has been closed.
        """
        if self._closed:
            raise DriverReadError(
                "SimRADALTDriver: driver has been closed. Cannot read."
            )

        terrain_elev_m = float(self._dem.patch(north_m, east_m, 1, 1)[0, 0])
        alt_agl_m = vehicle_alt_amsl_m - terrain_elev_m

        validity_flag = alt_agl_m >= 0.0

        self._record_successful_read()
        self._health_state = DriverHealth.OK

        return RADALTReading(
            alt_agl_m=alt_agl_m if validity_flag else float('nan'),
            validity_flag=validity_flag,
            t=self._last_update_time,
        )

    def close(self) -> None:
        """Release DEM reference and mark driver closed."""
        if not self._closed:
            self._dem = None
            self._closed = True
