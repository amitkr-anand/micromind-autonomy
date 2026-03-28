"""
integration/drivers/gnss.py
MicroMind Pre-HIL — Phase 1 Driver Abstraction Layer

GNSSDriver: ABC for GNSS receiver drivers.
GNSSReading: dataclass returned by GNSSDriver.read().

ADR-0 v1.1 Section 6: integration/drivers/gnss.py
SIA v1.0: Phase 1 real implementation is MAVLinkGNSSDriver (GPS_RAW_INT at 5 Hz).
          RealGNSSDriver (direct UART) is the upgrade path for production
          spoof detection requiring a Y-cable.

SWaP note: MicroMind does not own the GNSS receiver. The receiver belongs
to the host UAV platform. This driver subscribes to the platform GNSS stream.
BIM.update() is called from within the driver after each successful read.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import IntEnum

from integration.drivers.base import SensorDriver, DriverHealth


class GNSSFixType(IntEnum):
    """GNSS fix quality, aligned with MAVLink GPS_FIX_TYPE values."""
    NO_GPS    = 0
    NO_FIX    = 1
    FIX_2D    = 2
    FIX_3D    = 3
    DGPS      = 4
    RTK_FLOAT = 5
    RTK_FIXED = 6


@dataclass(frozen=True)
class GNSSReading:
    """Single GNSS measurement returned by GNSSDriver.read().

    Attributes:
        lat:      latitude in decimal degrees (WGS-84).
        lon:      longitude in decimal degrees (WGS-84).
        alt:      altitude above mean sea level in metres.
        hdop:     horizontal dilution of precision (dimensionless).
                  Use float('nan') if not available.
        fix_type: GNSSFixType enum value indicating fix quality.
        t:        monotonic timestamp (seconds) of this measurement.
    """
    lat:      float
    lon:      float
    alt:      float
    hdop:     float
    fix_type: GNSSFixType
    t:        float


class GNSSDriver(SensorDriver):
    """Abstract base class for all GNSS driver implementations.

    Extends SensorDriver with GNSS-specific read() return type contract.
    All six SensorDriver abstract methods must still be implemented.

    Integration path (SIA v1.0):
        Sim:  SimGNSSDriver        — wraps gnss_spoof_injector.py
        Real: MAVLinkGNSSDriver    — subscribes to GPS_RAW_INT at 5 Hz
              RealGNSSDriver       — direct UART NMEA (upgrade path)

    BIM hook: implementations must call BIM.update() after each successful
    read() so the trust scorer receives every GNSS measurement.
    The BIM instance is injected at construction by DriverFactory.
    """

    @abstractmethod
    def read(self) -> GNSSReading:
        """Read and return the latest GNSS measurement.

        Returns:
            GNSSReading with lat, lon, alt, hdop, fix_type, and timestamp.

        Raises:
            DriverReadError: on unrecoverable fault or missing hardware.
        """
