"""
integration/drivers/eoir.py
MicroMind Pre-HIL — Phase 1 Driver Abstraction Layer

EOIRDriver: ABC for electro-optical / infrared camera drivers.
EOIRFrame: dataclass returned by EOIRDriver.read().

ADR-0 v1.1 Section 6: integration/drivers/eoir.py
SIA v1.0: EO/IR is MicroMind-owned (direct MIPI CSI-2 or USB3).
          No MAVLink path exists for video frames.
          RealEOIRDriver (Phase 3) implements direct MIPI/USB3 frame receive.
          SimEOIRDriver uses synthetic frame generation (generate_synthetic_scene).

DMRL contract: DMRL receives EOIRFrame from this driver. When
validity_flag=False, DMRL must not attempt target acquisition.
Synthetic scene remains active until RealEOIRDriver is implemented.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from integration.drivers.base import SensorDriver, DriverHealth


@dataclass
class EOIRFrame:
    """Single EO/IR camera frame returned by EOIRDriver.read().

    Unlike other driver readings, EOIRFrame is NOT frozen because
    numpy arrays are not hashable and cannot be stored in frozen dataclasses.
    Callers must not mutate the frame_data array after receipt.

    Attributes:
        frame_data:    numpy array of shape (H, W) for thermal (uint16) or
                       (H, W, 3) for colour (uint8). None if validity_flag
                       is False.
        width:         frame width in pixels. 0 if validity_flag is False.
        height:        frame height in pixels. 0 if validity_flag is False.
        validity_flag: True if frame_data contains a valid frame.
                       False suppresses DMRL acquisition for this cycle.
        t:             monotonic timestamp (seconds) of frame capture.
    """
    frame_data:    Optional[np.ndarray]
    width:         int
    height:        int
    validity_flag: bool
    t:             float


class EOIRDriver(SensorDriver):
    """Abstract base class for all EO/IR camera driver implementations.

    Extends SensorDriver with EOIR-specific read() return type contract.
    All six SensorDriver abstract methods must still be implemented.

    Integration path (SIA v1.0):
        Sim:  SimEOIRDriver    — synthetic frame generation (no hardware)
        Real: RealEOIRDriver   — direct MIPI CSI-2 / USB3 frame receiver
                                 (no MAVLink path exists for video)

    DMRL contract: DMRL must check validity_flag before processing frame_data.
    When RealEOIRDriver raises DriverReadError (camera not initialised),
    the calling layer must synthesise an EOIRFrame with validity_flag=False
    so DMRL suppression is applied cleanly.

    Frame format contract:
        Thermal (LWIR): uint16 array, shape (H, W), radiometric counts.
        Colour (EO):    uint8 array,  shape (H, W, 3), RGB.
    """

    @abstractmethod
    def read(self) -> EOIRFrame:
        """Read and return the latest camera frame.

        Returns:
            EOIRFrame with frame_data, width, height, validity_flag, timestamp.
            validity_flag=False when no valid frame is available.

        Raises:
            DriverReadError: on unrecoverable camera fault or absent hardware.
        """
