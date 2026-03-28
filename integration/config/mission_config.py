"""
integration/config/mission_config.py
MicroMind Pre-HIL — Phase 1 Driver Abstraction Layer

MissionConfig: configuration dataclass governing driver source selection
and staleness thresholds. Read by DriverFactory at construction.

Source values: 'sim' selects the Sim driver; 'real' selects the Real stub.
All threshold values are in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StaleThresholds:
    """Per-driver staleness thresholds in seconds.

    A driver is considered stale if no successful read() has occurred
    within the threshold. LivePipeline health watchdog uses these values.
    """
    imu_s:    float = 0.01   # 100Hz margin — IMU at 200Hz
    gnss_s:   float = 0.20   # 5Hz margin   — GNSS at 5Hz
    radalt_s: float = 0.10   # 10Hz margin  — RADALT at 10Hz
    eoir_s:   float = 0.04   # 25Hz margin  — EOIR at 25Hz
    sdr_s:    float = 0.50   # 2Hz margin   — SDR at 2Hz


@dataclass
class MissionConfig:
    """Top-level mission configuration for Pre-HIL driver selection.

    Attributes:
        imu_source:    'sim' or 'real'. Selects SimIMUDriver or RealIMUDriver.
        gnss_source:   'sim' or 'real'. Selects SimGNSSDriver or RealGNSSDriver.
        radalt_source: 'sim' or 'real'. Selects SimRADALTDriver or RealRADALTDriver.
        eoir_source:   'sim' or 'real'. Selects SimEOIRDriver or RealEOIRDriver.
        sdr_source:    'sim' or 'real'. Selects SimSDRDriver or RealSDRDriver.
        px4_output:    'sim' or 'real'. 'real' enables MAVLink bridge to PX4 SITL.
        imu_type:      IMU model for SimIMUDriver ('STIM300', 'ADIS16505_3', 'BASELINE').
        sim_seed:      RNG seed for all Sim drivers.
        stale:         per-driver staleness threshold configuration.
    """
    imu_source:    str = 'sim'
    gnss_source:   str = 'sim'
    radalt_source: str = 'sim'
    eoir_source:   str = 'sim'
    sdr_source:    str = 'sim'
    px4_output:    str = 'sim'
    imu_type:      str = 'STIM300'
    sim_seed:      int = 42
    stale: StaleThresholds = field(default_factory=StaleThresholds)

    _VALID_SOURCES = frozenset({'sim', 'real'})
    _VALID_IMU_TYPES = frozenset({'STIM300', 'ADIS16505_3', 'BASELINE'})

    def validate(self) -> None:
        """Validate all config values. Raises ValueError on invalid input.

        Called by DriverFactory before constructing any driver.
        """
        for attr in ['imu_source', 'gnss_source', 'radalt_source',
                     'eoir_source', 'sdr_source', 'px4_output']:
            val = getattr(self, attr)
            if val not in self._VALID_SOURCES:
                raise ValueError(
                    f"MissionConfig.{attr}='{val}' is invalid. "
                    f"Must be one of {sorted(self._VALID_SOURCES)}."
                )
        if self.imu_type not in self._VALID_IMU_TYPES:
            raise ValueError(
                f"MissionConfig.imu_type='{self.imu_type}' is invalid. "
                f"Must be one of {sorted(self._VALID_IMU_TYPES)}."
            )
