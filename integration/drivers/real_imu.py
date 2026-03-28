"""
integration/drivers/real_imu.py
MicroMind Pre-HIL — Phase 1 Driver Abstraction Layer

RealIMUDriver: Real hardware stub for SPI /dev/spidev0.0.

Raises DriverReadError on first read() to signal that hardware is not
connected. The error message identifies the expected interface path so an
OEM integrator can diagnose without reading source code.

SIA v1.0 note: Phase 3 live implementation: MAVLinkIMUDriver (HIGHRES_IMU at 200 Hz). Direct SPI is upgrade path if P95 jitter > 1.0ms gate.

v1.2 §11.3: Observable failure behaviour is part of the integration claim.
A clean, informative error is evidence the interface is real and replaceable.
"""

from __future__ import annotations

from integration.drivers.imu import IMUDriver
from integration.drivers.base import DriverHealth, DriverReadError


class RealIMUDriver(IMUDriver):
    """RealIMUDriver — hardware stub raising DriverReadError on first read().

    Replace this stub with the real implementation once hardware is
    available. The real implementation must:
      - Inherit from IMUDriver
      - Implement all six SensorDriver abstract methods
      - Return source_type() == 'real'
      - Open SPI /dev/spidev0.0 in __init__() and close in close()
    """

    def __init__(self, stale_threshold_s: float = 0.01) -> None:
        super().__init__(stale_threshold_s)
        self._closed = False

    def health(self) -> DriverHealth:
        """Returns FAILED until hardware is connected and read() succeeds."""
        return DriverHealth.FAILED

    def last_update_time(self) -> float:
        return self._last_update_time

    def is_stale(self) -> bool:
        return True   # always stale until hardware is available

    def source_type(self) -> str:
        return 'real'

    def read(self):
        """Always raises DriverReadError — hardware not connected.

        Raises:
            DriverReadError: with interface path for OEM diagnosis.
        """
        raise DriverReadError(
            "RealIMUDriver: device not connected. Check SPI /dev/spidev0.0. Ensure IMU is powered and SPI interface is enabled."
        )

    def close(self) -> None:
        """No-op stub. Real implementation must release hardware resources."""
        self._closed = True
