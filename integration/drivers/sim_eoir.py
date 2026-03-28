"""
integration/drivers/sim_eoir.py
MicroMind Pre-HIL — Phase 1 Driver Abstraction Layer

SimEOIRDriver: synthetic EO/IR frame generator for SIL testing.

Renders thermal targets from generate_synthetic_scene() as Gaussian
blobs on a uint16 frame. This is the same scene used by dmrl_stub.py
internally. No camera hardware is required.

SWaP note: MicroMind does not own the EO/IR camera. The camera connects
directly to MicroMind compute via MIPI CSI-2 or USB3 (platform-owned
sensor, MicroMind-side interface). This sim driver produces synthetic
frames only for SIL/SITL testing. Real frames arrive via RealEOIRDriver
in Phase 3.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from core.dmrl.dmrl_stub import generate_synthetic_scene, ThermalTarget
from integration.drivers.base import DriverHealth, DriverReadError
from integration.drivers.eoir import EOIRDriver, EOIRFrame

_DEFAULT_WIDTH  = 320
_DEFAULT_HEIGHT = 256
_UINT16_MAX     = 65535


def _render_thermal_frame(
    targets: list[ThermalTarget],
    width: int,
    height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render a uint16 thermal frame with target blobs.

    Places each target as a 2D Gaussian blob scaled by thermal_signature.
    Background noise is added from rng. This matches the scene geometry
    assumed by dmrl_stub.py discriminator logic.

    Args:
        targets: list of ThermalTarget from generate_synthetic_scene().
        width:   frame width in pixels.
        height:  frame height in pixels.
        rng:     numpy Generator for reproducible background noise.

    Returns:
        uint16 numpy array of shape (height, width).
    """
    frame = rng.integers(800, 1200, size=(height, width), dtype=np.uint16)
    cx, cy = width // 2, height // 2

    for t in targets:
        # Place target offset from centre using bearing and a fixed scale
        angle_rad = math.radians(t.bearing_deg)
        scale_px  = min(width, height) * 0.3
        tx = int(cx + scale_px * math.sin(angle_rad))
        ty = int(cy - scale_px * math.cos(angle_rad))
        sigma = max(2, t.initial_roi_px // 4)
        peak  = int(t.thermal_signature * _UINT16_MAX * 0.8)

        # Gaussian blob
        for dy in range(-sigma * 3, sigma * 3 + 1):
            for dx in range(-sigma * 3, sigma * 3 + 1):
                px, py = tx + dx, ty + dy
                if 0 <= px < width and 0 <= py < height:
                    val = peak * math.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma))
                    frame[py, px] = min(_UINT16_MAX, frame[py, px] + int(val))

    return frame


class SimEOIRDriver(EOIRDriver):
    """Synthetic thermal camera driver for SIL/SITL testing.

    Generates a new scene on each read() call. The scene is rendered
    as a uint16 thermal frame using Gaussian blobs for each target.

    Args:
        width:             frame width in pixels (default 320).
        height:            frame height in pixels (default 256).
        n_targets:         number of real targets per scene (default 1).
        n_decoys:          number of decoy targets per scene (default 1).
        seed:              RNG seed. If None, non-deterministic.
        stale_threshold_s: staleness threshold (default 0.04s = 25Hz margin).
    """

    def __init__(
        self,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        n_targets: int = 1,
        n_decoys: int = 1,
        seed: Optional[int] = 42,
        stale_threshold_s: float = 0.04,
    ) -> None:
        super().__init__(stale_threshold_s)
        self._width     = width
        self._height    = height
        self._n_targets = n_targets
        self._n_decoys  = n_decoys
        self._seed      = seed
        self._rng       = np.random.default_rng(seed)
        self._health_state = DriverHealth.DEGRADED
        self._closed    = False
        self._frame_count = 0

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

    def read(self) -> EOIRFrame:
        """Generate and return a synthetic thermal frame.

        Each call generates a new scene with fresh target positions.
        The scene seed advances deterministically from the initial seed.

        Returns:
            EOIRFrame with uint16 thermal frame, validity_flag=True.

        Raises:
            DriverReadError: if driver has been closed.
        """
        if self._closed:
            raise DriverReadError(
                "SimEOIRDriver: driver has been closed. Cannot read."
            )

        # Generate scene with per-frame seed for determinism
        frame_seed = (self._seed + self._frame_count) if self._seed is not None else None
        targets = generate_synthetic_scene(
            n_targets=self._n_targets,
            n_decoys=self._n_decoys,
            seed=frame_seed,
        )

        frame = _render_thermal_frame(
            targets=targets,
            width=self._width,
            height=self._height,
            rng=self._rng,
        )

        self._frame_count += 1
        self._record_successful_read()
        self._health_state = DriverHealth.OK

        return EOIRFrame(
            frame_data=frame,
            width=self._width,
            height=self._height,
            validity_flag=True,
            t=self._last_update_time,
        )

    def close(self) -> None:
        """Mark driver closed."""
        self._closed = True
