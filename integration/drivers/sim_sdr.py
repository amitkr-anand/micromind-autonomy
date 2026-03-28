"""
integration/drivers/sim_sdr.py
MicroMind Pre-HIL — Phase 1 Driver Abstraction Layer

SimSDRDriver: synthetic SDR/EW observation generator for SIL testing.

Generates EWObservation lists using random draws, matching the pattern
used by bcmp1_ew_sim.py. Output feeds directly into EWEngine.process_observations().

SWaP note: MicroMind owns the SDR interface (USB3 or M.2 PCIe).
No MAVLink path exists for RF spectrum data. This sim driver produces
synthetic observations only. Real SDR data arrives via RealSDRDriver
in Phase 3.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.ew_engine.ew_engine import EWObservation
from integration.drivers.base import SensorDriver, DriverHealth, DriverReadError


class SDRReading:
    """Container for a batch of EW observations from one SDR snapshot.

    Attributes:
        observations: list of EWObservation for this snapshot.
        mission_time_s: mission time at snapshot (seconds).
        t: monotonic timestamp of this reading.
    """
    __slots__ = ('observations', 'mission_time_s', 't')

    def __init__(
        self,
        observations: list[EWObservation],
        mission_time_s: float,
        t: float,
    ) -> None:
        self.observations   = observations
        self.mission_time_s = mission_time_s
        self.t              = t


class SimSDRDriver(SensorDriver):
    """Synthetic SDR driver producing random EW observations.

    Generates between 0 and max_emitters EWObservation objects per read(),
    using a seeded RNG for reproducibility. Signal strengths and bearings
    are drawn from distributions matching bcmp1_ew_sim.py parameters.

    Args:
        seed:              RNG seed for reproducibility.
        max_emitters:      maximum emitters per snapshot (default 3).
        stale_threshold_s: staleness threshold (default 0.5s = 2Hz margin).
        observer_pos_enu:  fixed observer position for all observations.
                           In a live pipeline this would come from the INS state.
    """

    def __init__(
        self,
        seed: int = 42,
        max_emitters: int = 3,
        stale_threshold_s: float = 0.5,
        observer_pos_enu: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(stale_threshold_s)
        self._rng           = np.random.default_rng(seed)
        self._max_emitters  = max_emitters
        self._mission_time  = 0.0
        self._observer_pos  = (observer_pos_enu if observer_pos_enu is not None
                               else np.zeros(3, dtype=float))
        self._health_state  = DriverHealth.DEGRADED
        self._closed        = False

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

    def read(self, dt_s: float = 0.5) -> SDRReading:
        """Generate a synthetic EW observation batch.

        Randomly draws 0..max_emitters emitters with randomised bearing,
        signal strength, and range. Advances internal mission time by dt_s.

        Args:
            dt_s: time step to advance mission clock (seconds).

        Returns:
            SDRReading containing list of EWObservation and timestamps.

        Raises:
            DriverReadError: if driver has been closed.
        """
        if self._closed:
            raise DriverReadError(
                "SimSDRDriver: driver has been closed. Cannot read."
            )

        n_emitters = int(self._rng.integers(0, self._max_emitters + 1))
        observations: list[EWObservation] = []

        for _ in range(n_emitters):
            bearing_deg        = float(self._rng.uniform(0.0, 360.0))
            signal_strength_db = float(self._rng.uniform(-110.0, -60.0))
            estimated_range_m  = float(self._rng.uniform(500.0, 15_000.0))
            frequency_mhz      = float(self._rng.choice([1575.42, 1227.60, 1176.45]))

            observations.append(EWObservation(
                timestamp_s=self._mission_time,
                bearing_deg=bearing_deg,
                signal_strength_db=signal_strength_db,
                estimated_range_m=estimated_range_m,
                position_enu=self._observer_pos.copy(),
                frequency_mhz=frequency_mhz,
            ))

        self._mission_time += dt_s
        self._record_successful_read()
        self._health_state = DriverHealth.OK

        return SDRReading(
            observations=observations,
            mission_time_s=self._mission_time - dt_s,
            t=self._last_update_time,
        )

    def close(self) -> None:
        """Mark driver closed."""
        self._closed = True
