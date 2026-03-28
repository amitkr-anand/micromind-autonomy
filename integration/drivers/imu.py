"""
integration/drivers/imu.py
MicroMind Pre-HIL — Phase 1 Driver Abstraction Layer

IMUDriver: ABC for inertial measurement unit drivers.
IMUReading: dataclass returned by IMUDriver.read().

ADR-0 v1.1 Section 6: integration/drivers/imu.py
SIA v1.0: Phase 1 real implementation is MAVLinkIMUDriver (HIGHRES_IMU).
          This ABC defines the interface that both SimIMUDriver and
          MAVLinkIMUDriver must satisfy.

SWaP note: MicroMind does not own the IMU. The IMU belongs to the host
UAV platform. This driver subscribes to the platform IMU data stream.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from integration.drivers.base import SensorDriver, DriverHealth


@dataclass(frozen=True)
class IMUReading:
    """Single IMU measurement returned by IMUDriver.read().

    All values use SI units. Timestamps are monotonic seconds from
    time.monotonic() at the moment the measurement was received.

    Attributes:
        accel_mss:  linear acceleration [x, y, z] in m/s^2, body frame.
        gyro_rads:  angular velocity [x, y, z] in rad/s, body frame.
        temp_c:     sensor temperature in degrees Celsius. Use float('nan')
                    if temperature is not available from this source.
        t:          monotonic timestamp (seconds) of this measurement.
    """
    accel_mss: tuple[float, float, float]
    gyro_rads: tuple[float, float, float]
    temp_c:    float
    t:         float


class IMUDriver(SensorDriver):
    """Abstract base class for all IMU driver implementations.

    Extends SensorDriver with IMU-specific read() return type contract.
    All six SensorDriver abstract methods must still be implemented.

    Integration path (SIA v1.0):
        Sim:  SimIMUDriver  — uses imu_model.py noise model
        Real: MAVLinkIMUDriver — subscribes to HIGHRES_IMU at 200 Hz
              RealIMUDriver     — direct SPI (upgrade path if jitter gate fails)

    The ESKF receives IMUReading from this driver at 200 Hz via T-NAV.
    Frame convention: body frame, matching PX4 and ESKF conventions.
    """

    @abstractmethod
    def read(self) -> IMUReading:
        """Read and return the latest IMU measurement.

        Returns:
            IMUReading with accel_mss, gyro_rads, temp_c, and timestamp.

        Raises:
            DriverReadError: on unrecoverable fault or missing hardware.
        """
