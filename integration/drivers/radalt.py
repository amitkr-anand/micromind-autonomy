"""
integration/drivers/radalt.py
MicroMind Pre-HIL — Phase 1 Driver Abstraction Layer

RADALTDriver: ABC for radar altimeter drivers.
RADALTReading: dataclass returned by RADALTDriver.read().

ADR-0 v1.1 Section 6: integration/drivers/radalt.py
SIA v1.0: Phase 1 real implementation is MAVLinkRADALTDriver
          (DISTANCE_SENSOR at 10 Hz). Direct UART is the upgrade path
          for terminal phase scenarios requiring lower latency.

Tech Review v1.1 R-03: RADALT reclassified as MVP-required.
TRN receives validity_flag=False when RADALT is unavailable —
TRN corrections are suppressed, not corrupted.

SWaP note: MicroMind does not own the RADALT. The sensor belongs to
the host UAV platform. This driver subscribes to the platform stream.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from integration.drivers.base import SensorDriver, DriverHealth


@dataclass(frozen=True)
class RADALTReading:
    """Single radar altimeter measurement returned by RADALTDriver.read().

    Attributes:
        alt_agl_m:     height above ground level in metres.
                       Use float('nan') if measurement is invalid.
        validity_flag: True if alt_agl_m is a valid measurement.
                       False suppresses TRN correction for this cycle.
                       TRN must check this flag before using alt_agl_m.
        t:             monotonic timestamp (seconds) of this measurement.
    """
    alt_agl_m:     float
    validity_flag: bool
    t:             float


class RADALTDriver(SensorDriver):
    """Abstract base class for all RADALT driver implementations.

    Extends SensorDriver with RADALT-specific read() return type contract.
    All six SensorDriver abstract methods must still be implemented.

    Integration path (SIA v1.0):
        Sim:  SimRADALTDriver      — wraps DEMProvider altitude query
        Real: MAVLinkRADALTDriver  — subscribes to DISTANCE_SENSOR at 10 Hz
              RealRADALTDriver     — direct UART (upgrade path)

    TRN contract: TRN must not use alt_agl_m when validity_flag is False.
    When RealRADALTDriver raises DriverReadError (hardware absent), the
    calling layer must synthesise a RADALTReading with validity_flag=False
    so TRN corrections are suppressed cleanly rather than using stale data.
    """

    @abstractmethod
    def read(self) -> RADALTReading:
        """Read and return the latest RADALT measurement.

        Returns:
            RADALTReading with alt_agl_m, validity_flag, and timestamp.
            validity_flag=False when measurement is invalid or unavailable.

        Raises:
            DriverReadError: on unrecoverable fault or missing hardware.
        """
