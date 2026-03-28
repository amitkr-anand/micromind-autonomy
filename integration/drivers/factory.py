"""
integration/drivers/factory.py
MicroMind Pre-HIL — Phase 1 Driver Abstraction Layer

DriverFactory: reads MissionConfig and constructs the correct driver
implementation for each sensor type.

All drivers are constructed here and nowhere else. No other module
should instantiate driver classes directly.
"""

from __future__ import annotations

from integration.config.mission_config import MissionConfig
from integration.drivers.base import SensorDriver
from integration.drivers.imu import IMUDriver
from integration.drivers.gnss import GNSSDriver
from integration.drivers.radalt import RADALTDriver
from integration.drivers.eoir import EOIRDriver


class DriverFactory:
    """Constructs driver instances from a MissionConfig.

    Usage:
        config  = MissionConfig(imu_source='sim', gnss_source='sim')
        factory = DriverFactory(config)
        imu     = factory.make_imu()
        gnss    = factory.make_gnss()

    All make_*() methods validate config on first call. Raises ValueError
    on invalid source strings before any driver is constructed.
    """

    def __init__(self, config: MissionConfig) -> None:
        config.validate()
        self._config = config

    # ------------------------------------------------------------------
    # Factory methods — one per driver type
    # ------------------------------------------------------------------

    def make_imu(self) -> IMUDriver:
        """Construct and return the configured IMU driver."""
        if self._config.imu_source == 'sim':
            from integration.drivers.sim_imu import SimIMUDriver
            return SimIMUDriver(
                imu_type=self._config.imu_type,
                seed=self._config.sim_seed,
                stale_threshold_s=self._config.stale.imu_s,
            )
        from integration.drivers.real_imu import RealIMUDriver
        return RealIMUDriver(stale_threshold_s=self._config.stale.imu_s)

    def make_gnss(self) -> GNSSDriver:
        """Construct and return the configured GNSS driver."""
        if self._config.gnss_source == 'sim':
            from integration.drivers.sim_gnss import SimGNSSDriver
            return SimGNSSDriver(stale_threshold_s=self._config.stale.gnss_s)
        from integration.drivers.real_gnss import RealGNSSDriver
        return RealGNSSDriver(stale_threshold_s=self._config.stale.gnss_s)

    def make_radalt(self) -> RADALTDriver:
        """Construct and return the configured RADALT driver."""
        if self._config.radalt_source == 'sim':
            from integration.drivers.sim_radalt import SimRADALTDriver
            return SimRADALTDriver(
                seed=self._config.sim_seed,
                stale_threshold_s=self._config.stale.radalt_s,
            )
        from integration.drivers.real_radalt import RealRADALTDriver
        return RealRADALTDriver(stale_threshold_s=self._config.stale.radalt_s)

    def make_eoir(self) -> EOIRDriver:
        """Construct and return the configured EO/IR camera driver."""
        if self._config.eoir_source == 'sim':
            from integration.drivers.sim_eoir import SimEOIRDriver
            return SimEOIRDriver(
                seed=self._config.sim_seed,
                stale_threshold_s=self._config.stale.eoir_s,
            )
        from integration.drivers.real_eoir import RealEOIRDriver
        return RealEOIRDriver(stale_threshold_s=self._config.stale.eoir_s)

    def make_sdr(self) -> SensorDriver:
        """Construct and return the configured SDR driver."""
        if self._config.sdr_source == 'sim':
            from integration.drivers.sim_sdr import SimSDRDriver
            return SimSDRDriver(
                seed=self._config.sim_seed,
                stale_threshold_s=self._config.stale.sdr_s,
            )
        from integration.drivers.real_sdr import RealSDRDriver
        return RealSDRDriver(stale_threshold_s=self._config.stale.sdr_s)
