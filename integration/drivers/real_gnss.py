"""
integration/drivers/real_gnss.py
MicroMind Pre-HIL — Phase 1 Driver Abstraction Layer

RealGNSSDriver: Real hardware stub for UART /dev/ttyUSB0.

Raises DriverReadError on first read() to signal that hardware is not
connected. The error message identifies the expected interface path so an
OEM integrator can diagnose without reading source code.

SIA v1.0 note: Phase 3 live implementation: MAVLinkGNSSDriver (GPS_RAW_INT at 5 Hz). Direct UART Y-cable is upgrade path for production spoof detection.

v1.2 §11.3: Observable failure behaviour is part of the integration claim.
A clean, informative error is evidence the interface is real and replaceable.
"""

from __future__ import annotations

from integration.drivers.gnss import GNSSDriver
from integration.drivers.base import DriverHealth, DriverReadError


class RealGNSSDriver(GNSSDriver):
    """RealGNSSDriver — hardware stub raising DriverReadError on first read().

    Replace this stub with the real implementation once hardware is
    available. The real implementation must:
      - Inherit from GNSSDriver
      - Implement all six SensorDriver abstract methods
      - Return source_type() == 'real'
      - Open UART /dev/ttyUSB0 in __init__() and close in close()
    """

    def __init__(self, stale_threshold_s: float = 0.2) -> None:
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
            "RealGNSSDriver: NMEA port not open. Check /dev/ttyUSB0. Ensure GNSS receiver is powered and UART baud rate is 115200."
        )

    def close(self) -> None:
        """No-op stub. Real implementation must release hardware resources."""
        self._closed = True
