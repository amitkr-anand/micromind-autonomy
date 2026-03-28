"""
integration/drivers/vio_driver.py
MicroMind Pre-HIL — Phase 3 Live Input Stubs + VIO Adapter

OfflineVIODriver: replays a pre-generated ENU position array as VIO updates.

Wraps the als250_nav_sim position output (.npy or in-memory array).
Applies ENU→NED rotation via frame_utils on each read().
Enforces IFM-01 monotonicity gate — non-monotonic timestamps are logged
and skipped; the ESKF is not updated.

LiveVIODriver: stub raising DriverReadError until real VIO hardware is
connected. Interface documents the expected input format.

Interface contract:
    read() → VIOReading(pos_ned_m, cov_ned, t, frame_index, valid)
    pos_ned_m : (3,) ndarray — NED position in metres, caller passes to
                update_vio(state, pos_ned_m, cov_ned)
    cov_ned   : (3,3) ndarray — position covariance in NED frame

SWaP note: MicroMind does not own the VIO camera. The camera is a
platform-owned sensor. MicroMind owns the VIO algorithm layer.
This driver replays offline data for SIL/SITL testing only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.fusion.frame_utils import rotate_pos_enu_to_ned, rotate_cov_enu_to_ned
from integration.drivers.base import SensorDriver, DriverHealth, DriverReadError


# ---------------------------------------------------------------------------
# VIOReading — output of read()
# ---------------------------------------------------------------------------

@dataclass
class VIOReading:
    """Single VIO position estimate in NED frame.

    Attributes:
        pos_ned_m:   (3,) NED position in metres. Pass to update_vio().
        cov_ned:     (3,3) NED position covariance. Pass to update_vio().
        t:           monotonic timestamp of this reading (seconds).
        frame_index: index into the source position array.
        valid:       False if IFM-01 monotonicity gate rejected this frame.
    """
    pos_ned_m:   np.ndarray
    cov_ned:     np.ndarray
    t:           float
    frame_index: int
    valid:       bool


# ---------------------------------------------------------------------------
# IFM-01 monotonicity guard
# ---------------------------------------------------------------------------

class _MonotonicityGuard:
    """IFM-01: enforce timestamp monotonicity. Log violations, do not crash.

    ADR-0 v1.1 / SIA v1.0 IFM-01: if a timestamp is not strictly greater
    than the previous one, the frame is rejected. The violation is logged
    with event ID and both timestamps. The ESKF is not updated.
    """

    def __init__(self) -> None:
        self._last_t:       float = -1.0
        self._violation_count: int = 0
        self._violations:  list  = []

    def check(self, t: float) -> bool:
        """Return True if t is monotonically greater than last seen t.

        Side effect: records violation if check fails.
        """
        if t <= self._last_t:
            event_id = f"IFM01-{self._violation_count:04d}"
            self._violations.append({
                "event_id":  event_id,
                "t_prev":    self._last_t,
                "t_current": t,
                "delta":     t - self._last_t,
            })
            self._violation_count += 1
            return False
        self._last_t = t
        return True

    @property
    def violation_count(self) -> int:
        return self._violation_count

    @property
    def violations(self) -> list:
        return list(self._violations)

    def last_violation(self) -> Optional[dict]:
        return self._violations[-1] if self._violations else None


# ---------------------------------------------------------------------------
# OfflineVIODriver
# ---------------------------------------------------------------------------

class OfflineVIODriver(SensorDriver):
    """Replays a pre-generated ENU position array as VIO updates.

    Advances one frame per read() call. Applies ENU→NED rotation.
    Wraps around to frame 0 when the array is exhausted (loop mode).

    Compatible with:
        - als250_nav_sim.run_als250_sim()["position"]  (N, 3) ENU ndarray
        - Any (N, 3) float64 ENU position array
        - Pre-saved .npy file of shape (N, 3)

    Args:
        position_enu:      (N, 3) ndarray of ENU positions, OR path to .npy.
        sigma_pos_m:       1-sigma position noise for covariance (default 0.1m).
        dt_s:              timestep between frames (default 0.04s = 25Hz).
        loop:              if True, wrap around at end of array (default True).
        stale_threshold_s: staleness threshold (default 0.08s = 25Hz margin).
    """

    def __init__(
        self,
        position_enu,
        sigma_pos_m:       float = 0.1,
        dt_s:              float = 0.04,
        loop:              bool  = True,
        stale_threshold_s: float = 0.08,
    ) -> None:
        super().__init__(stale_threshold_s)

        # Load array
        if isinstance(position_enu, (str,)):
            arr = np.load(position_enu)
        else:
            arr = np.asarray(position_enu, dtype=np.float64)

        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"OfflineVIODriver: position_enu must be (N, 3), got {arr.shape}"
            )

        self._pos_enu:  np.ndarray = arr
        self._n_frames: int        = len(arr)
        self._sigma:    float      = sigma_pos_m
        self._dt_s:     float      = dt_s
        self._loop:     bool       = loop

        # State
        self._frame_idx:   int   = 0
        self._mission_t:   float = 0.0
        self._health_state = DriverHealth.DEGRADED
        self._closed:      bool  = False
        self._exhausted:   bool  = False

        # IFM-01 monotonicity guard
        self._guard = _MonotonicityGuard()

        # Pre-compute NED covariance (constant diagonal)
        sigma2 = sigma_pos_m ** 2
        self._cov_enu_template = np.diag([sigma2, sigma2, sigma2])
        self._cov_ned_template = rotate_cov_enu_to_ned(self._cov_enu_template)

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

    def read(self) -> VIOReading:
        """Advance one frame and return VIOReading in NED frame.

        IFM-01: if the computed timestamp is non-monotonic (should not
        occur in normal replay), the frame is marked valid=False and
        the violation is logged. The caller must check valid before
        passing to update_vio().

        Returns:
            VIOReading with pos_ned_m, cov_ned, t, frame_index, valid.

        Raises:
            DriverReadError: if driver has been closed or array is
                             exhausted and loop=False.
        """
        if self._closed:
            raise DriverReadError(
                "OfflineVIODriver: driver has been closed. Cannot read."
            )
        if self._exhausted:
            raise DriverReadError(
                "OfflineVIODriver: position array exhausted (loop=False). "
                "Re-instantiate with a new array or set loop=True."
            )

        idx = self._frame_idx
        pos_enu = self._pos_enu[idx]
        t       = self._mission_t

        # IFM-01 monotonicity check
        valid = self._guard.check(t)

        # Rotate ENU → NED
        pos_ned = rotate_pos_enu_to_ned(pos_enu)

        # Advance state
        self._frame_idx  += 1
        self._mission_t  += self._dt_s

        if self._frame_idx >= self._n_frames:
            if self._loop:
                self._frame_idx = 0
            else:
                self._exhausted = True

        self._record_successful_read()
        self._health_state = DriverHealth.OK

        return VIOReading(
            pos_ned_m=pos_ned,
            cov_ned=self._cov_ned_template.copy(),
            t=t,
            frame_index=idx,
            valid=valid,
        )

    def close(self) -> None:
        """Mark driver closed."""
        self._closed = True

    # ------------------------------------------------------------------
    # IFM-01 diagnostics
    # ------------------------------------------------------------------

    @property
    def monotonicity_guard(self) -> _MonotonicityGuard:
        """Access to IFM-01 guard for test injection and inspection."""
        return self._guard

    @property
    def frame_index(self) -> int:
        """Current frame index."""
        return self._frame_idx

    @property
    def n_frames(self) -> int:
        """Total frames in position array."""
        return self._n_frames


# ---------------------------------------------------------------------------
# LiveVIODriver — stub
# ---------------------------------------------------------------------------

class LiveVIODriver(SensorDriver):
    """Live VIO driver stub — raises DriverReadError until hardware connected.

    Real implementation must:
      - Subscribe to VIO algorithm output (OpenVINS, ORB-SLAM3, or equivalent)
      - Apply ENU→NED rotation via frame_utils
      - Enforce IFM-01 monotonicity gate before passing to ESKF
      - Return VIOReading with valid=True only when position is trustworthy
      - Implement all six SensorDriver abstract methods
      - Return source_type() == 'real'

    Interface: same as OfflineVIODriver.read() → VIOReading
    """

    def __init__(self, stale_threshold_s: float = 0.04) -> None:
        super().__init__(stale_threshold_s)
        self._closed = False

    def health(self) -> DriverHealth:
        return DriverHealth.FAILED

    def last_update_time(self) -> float:
        return self._last_update_time

    def is_stale(self) -> bool:
        return True

    def source_type(self) -> str:
        return 'real'

    def read(self) -> VIOReading:
        """Always raises DriverReadError — VIO algorithm not connected.

        Raises:
            DriverReadError: with interface path for OEM diagnosis.
        """
        raise DriverReadError(
            "LiveVIODriver: VIO algorithm not connected. "
            "Implement this driver by subscribing to your VIO algorithm output "
            "(OpenVINS, ORB-SLAM3, or equivalent). Apply rotate_pos_enu_to_ned "
            "before passing to update_vio(). Enforce IFM-01 monotonicity gate."
        )

    def close(self) -> None:
        self._closed = True
