"""
integration/drivers/sim_imu.py
MicroMind Pre-HIL — Phase 1 Driver Abstraction Layer

SimIMUDriver: simulation IMU using the existing imu_model.py noise model.

Wraps core/ins/imu_model.py (S8-A). Uses get_imu_model() and
generate_imu_noise() — the same API used by als250_nav_sim.py.

SWaP note: MicroMind does not own the IMU. This sim driver exists only
for SIL and SITL testing. Real IMU data arrives via MAVLinkIMUDriver
(HIGHRES_IMU) in Phase 3.
"""

from __future__ import annotations

import math

from core.ins.imu_model import get_imu_model, generate_imu_noise, IMUNoiseOutput
from integration.drivers.base import DriverHealth, DriverReadError
from integration.drivers.imu import IMUDriver, IMUReading

_SUPPORTED_IMU_TYPES = frozenset({"STIM300", "ADIS16505_3", "BASELINE"})
_DT_200HZ = 1.0 / 200.0


class SimIMUDriver(IMUDriver):
    """Simulation IMU driver backed by core/ins/imu_model.py.

    Generates deterministic noisy IMU data using get_imu_model() and
    generate_imu_noise(). Seed is set at construction for reproducibility.
    Pre-generates n_steps of noise; wraps at exhaustion.

    Args:
        imu_type:          "STIM300", "ADIS16505_3", or "BASELINE".
        seed:              RNG seed for reproducibility.
        stale_threshold_s: staleness threshold (default 0.01s = 100Hz margin).
        n_steps:           steps to pre-generate (default 400000 ~= 2000s at 200Hz).
        dt:                timestep in seconds (default 0.005 = 200Hz).
    """

    def __init__(
        self,
        imu_type: str = "STIM300",
        seed: int = 42,
        stale_threshold_s: float = 0.01,
        n_steps: int = 400_000,
        dt: float = _DT_200HZ,
    ) -> None:
        super().__init__(stale_threshold_s)

        if imu_type not in _SUPPORTED_IMU_TYPES:
            raise ValueError(
                f"SimIMUDriver: unknown imu_type '{imu_type}'. "
                f"Supported: {sorted(_SUPPORTED_IMU_TYPES)}"
            )

        self._imu_type = imu_type
        self._seed = seed
        self._step = 0
        self._health_state = DriverHealth.DEGRADED
        self._closed = False

        model = get_imu_model(imu_type)
        noise: IMUNoiseOutput = generate_imu_noise(
            model=model, n_steps=n_steps, dt=dt, seed=seed
        )
        # Total effective noise per step: noise + bias (matching als250_nav_sim.py)
        self._gyro_total  = noise.gyro_noise_rads  + noise.gyro_bias_rads
        self._accel_total = noise.accel_noise_ms2  + noise.accel_bias_ms2
        self._n_steps = n_steps

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

    def read(self) -> IMUReading:
        """Return the next simulated IMU step.

        Cycles through pre-generated noise arrays. Wraps at n_steps.

        Returns:
            IMUReading with body-frame accel (m/s^2) and gyro (rad/s).

        Raises:
            DriverReadError: if driver has been closed.
        """
        if self._closed:
            raise DriverReadError(
                "SimIMUDriver: driver has been closed. Cannot read."
            )

        idx = self._step % self._n_steps
        gyro  = (float(self._gyro_total[idx, 0]),
                 float(self._gyro_total[idx, 1]),
                 float(self._gyro_total[idx, 2]))
        accel = (float(self._accel_total[idx, 0]),
                 float(self._accel_total[idx, 1]),
                 float(self._accel_total[idx, 2]))

        self._step += 1
        self._record_successful_read()
        self._health_state = DriverHealth.OK

        return IMUReading(
            accel_mss=accel,
            gyro_rads=gyro,
            temp_c=float('nan'),
            t=self._last_update_time,
        )

    def close(self) -> None:
        """Release pre-generated arrays and mark driver closed."""
        if not self._closed:
            self._gyro_total  = None
            self._accel_total = None
            self._closed = True
