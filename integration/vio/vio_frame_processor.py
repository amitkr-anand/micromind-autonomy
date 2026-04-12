"""
integration/vio/vio_frame_processor.py
MicroMind / NanoCorteX — VIO Frame Processor (Case C: OpenVINS not available)

NAV-03: Lightweight ORB feature tracker + Lucas-Kanade optical flow as
VIO substitute. This is NOT a stub — it is a real feature tracker that
extracts ORB features, tracks them across frames via optical flow, and
estimates camera motion from tracked feature displacement.

Sensor substitution contract:
    Current:  ORB feature tracker on Gazebo rendered frames
    HIL:      OpenVINS on real EO camera frames
    Interface: identical VIOEstimate dataclass
    Breaking change at HIL: NO

Algorithm:
    1. Extract ORB keypoints on each new frame
    2. If prior frame exists: track keypoints using Lucas-Kanade optical flow
    3. Compute median pixel displacement (robust to outliers)
    4. Convert pixel displacement to metres using current GSD
    5. Confidence = f(tracked_feature_count, track_quality)

References:
    SRS v1.3 NAV-03
    S-NEP-09 OpenVINS validation
    docs/interfaces/eo_day_contract.yaml
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class VIOEstimate:
    """
    Output of VIOFrameProcessor.process_frame().

    Fields
    ------
    delta_north_m  : float — northward displacement since last frame (m)
    delta_east_m   : float — eastward displacement since last frame (m)
    delta_alt_m    : float — altitude change (0.0 — not observable from monocular nadir)
    confidence     : float [0.0, 1.0] — based on tracked feature count and track quality
    feature_count  : int   — number of successfully tracked features
    timestamp_ms   : int   — mission time of this estimate
    """
    delta_north_m: float
    delta_east_m:  float
    delta_alt_m:   float
    confidence:    float
    feature_count: int
    timestamp_ms:  int


# ---------------------------------------------------------------------------
# VIOFrameProcessor
# ---------------------------------------------------------------------------

class VIOFrameProcessor:
    """
    NAV-03: ORB feature tracker + LK optical flow VIO substitute.

    Produces VIOEstimate from consecutive camera frames. Confidence score
    is based on tracked feature count and optical flow consistency.

    At HIL: replace this class with OpenVINS bridge. VIOEstimate interface
    is identical — no ESKF changes required.
    """

    # Confidence thresholds
    _CONF_FULL_AT_FEATURES:   int   = 200   # feature count for confidence = 1.0
    _CONF_ZERO_AT_FEATURES:   int   = 20    # below this: confidence contribution = 0
    _FLOW_CONSISTENCY_WEIGHT: float = 0.3   # weight of consistency score in confidence
    _FEATURE_COUNT_WEIGHT:    float = 0.7   # weight of feature count score in confidence

    def __init__(
        self,
        min_features: int = 50,
        max_features: int = 500,
        clock_fn:     Optional[Callable[[], int]] = None,
        gsd_m:        float = 5.0,
    ) -> None:
        """
        min_features : minimum tracked features needed for a valid estimate
        max_features : maximum features to extract per frame (ORB cap)
        clock_fn     : callable returning mission time in ms (int)
        gsd_m        : ground sample distance in metres (used to convert px→m).
                       Should be updated per-frame when altitude changes.
        """
        import cv2
        self._cv2         = cv2
        self._min_features = min_features
        self._max_features = max_features
        self._clock_fn    = clock_fn
        self._gsd_m       = gsd_m

        # Shi-Tomasi corner detection (more robust on gradient images than ORB)
        # CLAHE is applied before detection to improve contrast on smooth DEM tiles.
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # LK optical flow parameters
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        # State
        self._prev_gray:       Optional[np.ndarray] = None
        self._prev_keypoints:  Optional[np.ndarray] = None  # (N,1,2) float32
        self._prev_ts_ms:      Optional[int]        = None
        self._frame_count:     int                  = 0

        # Structured event log
        self._event_log: list = []

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def gsd_m(self) -> float:
        return self._gsd_m

    @gsd_m.setter
    def gsd_m(self, value: float) -> None:
        """Update GSD when altitude changes."""
        self._gsd_m = float(value)

    def process_frame(
        self,
        frame:        np.ndarray,
        timestamp_ms: int,
    ) -> Optional[VIOEstimate]:
        """
        Process a camera frame and return a VIOEstimate, or None if
        insufficient features for a reliable estimate.

        Parameters
        ----------
        frame        : uint8 numpy array (H,W) or (H,W,3)
        timestamp_ms : mission time at frame capture

        Returns
        -------
        VIOEstimate or None
        """
        cv2 = self._cv2
        self._frame_count += 1

        # Convert to grayscale
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame.astype(np.uint8)

        # First frame — just store and return None
        if self._prev_gray is None:
            self._prev_gray, self._prev_keypoints, self._prev_ts_ms = (
                self._init_frame(gray, timestamp_ms)
            )
            return None

        # No keypoints to track
        if self._prev_keypoints is None or len(self._prev_keypoints) == 0:
            self._prev_gray, self._prev_keypoints, self._prev_ts_ms = (
                self._init_frame(gray, timestamp_ms)
            )
            self._log_insufficient(timestamp_ms, 0)
            return None

        # Track features via LK optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray,
            self._prev_keypoints,
            None,
            **self._lk_params,
        )

        if curr_pts is None or status is None:
            self._prev_gray, self._prev_keypoints, self._prev_ts_ms = (
                self._init_frame(gray, timestamp_ms)
            )
            self._log_insufficient(timestamp_ms, 0)
            return None

        # Filter to successfully tracked points
        mask       = (status.ravel() == 1)
        tracked_n  = int(mask.sum())

        if tracked_n < self._min_features:
            self._prev_gray, self._prev_keypoints, self._prev_ts_ms = (
                self._init_frame(gray, timestamp_ms)
            )
            self._log_insufficient(timestamp_ms, tracked_n)
            return None

        prev_good = self._prev_keypoints[mask]   # (K,1,2)
        curr_good = curr_pts[mask]               # (K,1,2)

        # Compute pixel displacement for each tracked point
        disp = curr_good.reshape(-1, 2) - prev_good.reshape(-1, 2)  # (K,2)

        # Robust estimate: median displacement
        median_disp = np.median(disp, axis=0)   # (2,) [dx_col, dy_row]

        # Flow consistency: fraction of points within 2px of median
        deviations = np.linalg.norm(disp - median_disp, axis=1)
        consistency = float(np.mean(deviations < 2.0))

        # Convert pixel displacement to metres
        # Image convention: col (+right) = east, row (+down) = south
        # For nadir camera looking down: camera moves east → scene moves west → col offset positive
        # VIO convention: delta_east = +median_disp[0] * gsd_m
        #                 delta_north = -median_disp[1] * gsd_m  (row+ = south = north-)
        delta_east_m  = float( median_disp[0] * self._gsd_m)
        delta_north_m = float(-median_disp[1] * self._gsd_m)

        # Confidence score
        feat_score  = np.clip(
            (tracked_n - self._CONF_ZERO_AT_FEATURES) /
            (self._CONF_FULL_AT_FEATURES - self._CONF_ZERO_AT_FEATURES),
            0.0, 1.0
        )
        confidence = float(
            self._FEATURE_COUNT_WEIGHT  * feat_score +
            self._FLOW_CONSISTENCY_WEIGHT * consistency
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Update state for next frame
        self._prev_gray, self._prev_keypoints, self._prev_ts_ms = (
            self._init_frame(gray, timestamp_ms)
        )

        estimate = VIOEstimate(
            delta_north_m=delta_north_m,
            delta_east_m=delta_east_m,
            delta_alt_m=0.0,   # monocular nadir: altitude not observable
            confidence=confidence,
            feature_count=tracked_n,
            timestamp_ms=timestamp_ms,
        )

        self._log_estimate(estimate)
        return estimate

    @property
    def event_log(self) -> list:
        return list(self._event_log)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_frame(
        self,
        gray:         np.ndarray,
        timestamp_ms: int,
    ):
        """
        Detect Shi-Tomasi corners on a CLAHE-enhanced grayscale frame.

        Shi-Tomasi (goodFeaturesToTrack) is preferred over ORB for gradient images
        such as DEM hillshades and real nadir EO imagery — it produces more corners
        on smooth terrain and tracks well with LK optical flow.
        CLAHE improves local contrast before detection.
        """
        cv2 = self._cv2
        enhanced = self._clahe.apply(gray)
        pts = cv2.goodFeaturesToTrack(
            enhanced,
            maxCorners=self._max_features,
            qualityLevel=0.01,
            minDistance=5,
        )
        if pts is None:
            pts = np.zeros((0, 1, 2), dtype=np.float32)
        else:
            pts = pts.reshape(-1, 1, 2).astype(np.float32)
        return gray, pts, timestamp_ms

    def _log_estimate(self, est: VIOEstimate) -> None:
        self._event_log.append({
            "event":        "VIO_ESTIMATE_PRODUCED",
            "module_name":  "VIOFrameProcessor",
            "req_id":       "NAV-03",
            "severity":     "INFO",
            "timestamp_ms": est.timestamp_ms,
            "payload": {
                "confidence":    round(est.confidence, 4),
                "feature_count": est.feature_count,
                "delta_north_m": round(est.delta_north_m, 3),
                "delta_east_m":  round(est.delta_east_m, 3),
            },
        })

    def _log_insufficient(self, timestamp_ms: int, count: int) -> None:
        self._event_log.append({
            "event":        "VIO_INSUFFICIENT_FEATURES",
            "module_name":  "VIOFrameProcessor",
            "req_id":       "NAV-03",
            "severity":     "WARNING",
            "timestamp_ms": timestamp_ms,
            "payload": {
                "feature_count": count,
                "minimum":       self._min_features,
                "timestamp_ms":  timestamp_ms,
            },
        })
