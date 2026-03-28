"""
integration/drivers/base.py
MicroMind Pre-HIL — Phase 1 Driver Abstraction Layer

SensorDriver: abstract base class for all MicroMind integration drivers.

Every driver in the integration layer — Sim or Real — inherits from this
class and must implement all six methods. No duck typing. DriverFactory
enforces this at construction time.

ADR-0 v1.1 D-3: Python ABCs for all driver classes.
ADR-0 v1.1 Section 3: six mandatory methods on every driver ABC.
v1.2 §10.4: every driver inherits from an abstract base class.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import Enum, auto


class DriverHealth(Enum):
    """Health state reported by a driver's health() method.

    OK       — driver is operating normally, data is fresh and valid.
    DEGRADED — driver is operational but data quality is reduced
               (transient fault, stale read, or partial data). The last
               valid reading is still returned by read().
    FAILED   — driver has encountered an unrecoverable fault. read()
               will raise DriverReadError. LivePipeline must be notified.
    """
    OK       = auto()
    DEGRADED = auto()
    FAILED   = auto()


class DriverReadError(RuntimeError):
    """Raised by read() when the driver cannot return any valid data.

    The message must identify the interface path so an OEM integrator
    can diagnose the connection without reading source code.

    Examples:
        "RealIMUDriver: device not connected. Check SPI /dev/spidev0.0."
        "RealGNSSDriver: NMEA port not open. Check /dev/ttyUSB0."
        "RealRADALTDriver: RADALT not responding. TRN corrections suppressed."
    """


class SensorDriver(ABC):
    """Abstract base class for all MicroMind Pre-HIL integration drivers.

    Subclasses must implement all six abstract methods. DriverFactory
    verifies conformance at construction; do not bypass this check.

    Thread safety: drivers are read from T-NAV at 200 Hz. Implementations
    must be thread-safe for concurrent calls to read() and health().
    close() is called from the main thread on any exit path.

    SWaP note: MicroMind does not own or provide sensors. Sensors belong
    to the host UAV platform. This driver layer is the listener/subscriber
    interface to platform sensor streams, not the sensor itself.
    """

    def __init__(self, stale_threshold_s: float) -> None:
        """Initialise the driver with a staleness threshold.

        Args:
            stale_threshold_s: seconds after which a reading is considered
                stale. is_stale() uses this value. Set by MissionConfig
                per driver type and passed in by DriverFactory.
        """
        self._stale_threshold_s: float = stale_threshold_s
        self._last_update_time: float = 0.0

    @abstractmethod
    def health(self) -> DriverHealth:
        """Return the current health state of the driver.

        Contract:
        - Must NEVER raise under any circumstances.
        - Must be callable at 10 Hz from LivePipeline health watchdog.
        - Must reflect the state of the most recent read() attempt.
        - If the driver has never been read, return DriverHealth.DEGRADED.
        """

    @abstractmethod
    def last_update_time(self) -> float:
        """Return the monotonic timestamp of the last successful read.

        Contract:
        - Returns time.monotonic() at the moment the last valid data was
          received from the sensor or simulation source.
        - Returns 0.0 if read() has never succeeded.
        - Used by is_stale() and BridgeLogger for timestamp annotation.
        """

    @abstractmethod
    def is_stale(self) -> bool:
        """Return True if the last reading is older than the stale threshold.

        Contract:
        - Computes (time.monotonic() - last_update_time()) > stale_threshold_s.
        - Returns True if read() has never succeeded (last_update_time = 0.0).
        - Must never raise.
        """

    @abstractmethod
    def source_type(self) -> str:
        """Return the source type identifier for this driver instance.

        Contract:
        - Returns exactly 'sim' for all SimXxxDriver implementations.
        - Returns exactly 'real' for all RealXxxDriver implementations.
        - Set at construction by DriverFactory; must not change at runtime.
        - Used by BridgeLogger to annotate all log entries with data source.
        """

    @abstractmethod
    def read(self):
        """Read and return the latest sensor data.

        Contract:
        - Returns a driver-specific dataclass (IMUReading, GNSSReading, etc.).
        - On transient fault: returns last valid data and sets health to DEGRADED.
        - On unrecoverable fault: raises DriverReadError with interface path.
        - On first call with hardware absent (Real drivers): raises
          DriverReadError immediately with a descriptive message identifying
          the expected hardware interface path.
        - Updates _last_update_time on every successful read.
        - Thread-safe: callable concurrently from T-NAV at 200 Hz.

        Raises:
            DriverReadError: on unrecoverable fault or missing hardware.
        """

    @abstractmethod
    def close(self) -> None:
        """Shut down the driver cleanly.

        Contract:
        - Must be idempotent: safe to call multiple times.
        - Must not raise under any circumstances.
        - Called by LivePipeline on every exit path, including exceptions.
        - Real drivers must release hardware resources.
        - Sim drivers must stop any background simulation threads.
        """

    def _record_successful_read(self) -> None:
        """Update _last_update_time to now. Call after every successful read()."""
        self._last_update_time = time.monotonic()

    def _default_is_stale(self) -> bool:
        """Default is_stale() implementation using _last_update_time."""
        if self._last_update_time == 0.0:
            return True
        return (time.monotonic() - self._last_update_time) > self._stale_threshold_s
