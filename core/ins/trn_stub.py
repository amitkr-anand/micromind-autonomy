"""
core/ins/trn_stub.py
MicroMind / NanoCorteX — Terrain-Referenced Navigation Stub
Sprint S3 Deliverable 1 of 3

Implements a TRN correction layer that uses Normalised Cross-Correlation (NCC)
between a live radar-altimeter terrain strip and a DEM patch to estimate a
horizontal position fix, then injects a Kalman correction into the INS state.

Design references:
    Part Two V7  §1.7.2  (TRN correction algorithm requirements)
    FR-107       Drift < 2 % of distance over any 5 km GNSS-denied segment
    NAV-01       TRN corrections shall bound INS drift, min 1 correction / 2 km

Architecture:
    ┌──────────────┐   radar strip   ┌─────────────┐
    │ RadarAltimSim│ ──────────────> │  TRNStub    │
    └──────────────┘                 │  NCC match  │
    ┌──────────────┐   DEM patch     │  Kalman fix │
    │  DEMProvider │ ──────────────> │             │
    └──────────────┘                 └──────┬──────┘
                                            │ position correction Δ(N,E)
                                            ▼
                                      INS state updated

NCC algorithm:
    r(dx,dy) = Σ[ (s-μs)(t-μt) ] / ( σs·σt·N )
    peak of r gives sub-pixel position fix ± 1 pixel (≈ 5 m)
    correction applied only if r_peak ≥ NCC_THRESHOLD (0.45)

Kalman correction:
    Standard EKF measurement update step:
        K  = P H^T (H P H^T + R)^-1
        x' = x + K (z - Hx)
        P' = (I - KH) P
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NCC_THRESHOLD       = 0.45          # minimum peak NCC score to accept fix
CORRECTION_INTERVAL = 1500.0        # metres — min ground-track between corrections (NAV-01: ≤2 km)
DEM_PIXEL_SIZE      = 5.0           # metres per pixel in synthetic DEM
STRIP_WIDTH_PX      = 20            # radar swath width (pixels)
STRIP_LEN_PX        = 40            # strip length used for correlation template
SEARCH_PAD_PX       = 25            # search half-width — 125 m radius to cover INS drift
R_TRN_NORTH         = 15.0 ** 2     # TRN measurement noise variance (m²) — north
R_TRN_EAST          = 15.0 ** 2     # TRN measurement noise variance (m²) — east


# ---------------------------------------------------------------------------
# Synthetic DEM provider
# ---------------------------------------------------------------------------

class DEMProvider:
    """
    Generates deterministic synthetic Digital Elevation Model data.

    In production this would wrap a geo-referenced DEM raster (e.g. SRTM).
    For SIL we create a reproducible terrain surface with rolling hills,
    ridges, and valley features suitable for NCC correlation.
    """

    def __init__(self, seed: int = 7):
        self._rng = np.random.default_rng(seed)
        # Pre-generate a large terrain grid (500 × 500 pixels = 2.5 × 2.5 km)
        base   = self._rng.uniform(0, 1, (500, 500))
        # Low-frequency terrain features
        xx, yy = np.meshgrid(np.linspace(0, 6*math.pi, 500),
                             np.linspace(0, 4*math.pi, 500))
        self._dem = (
            80.0 * np.sin(xx * 0.3) * np.cos(yy * 0.4) +
            40.0 * np.sin(xx * 0.9 + 1.2) +
            30.0 * np.cos(yy * 0.7 + 0.5) +
            15.0 * base +
            200.0   # mean altitude 200 m AGL
        ).astype(np.float32)

    def patch(self, north_m: float, east_m: float,
              h_px: int, w_px: int) -> np.ndarray:
        """
        Return a DEM patch centred on (north_m, east_m) with size (h_px × w_px).
        Wraps around the synthetic terrain grid.
        """
        r0 = int(north_m / DEM_PIXEL_SIZE) % self._dem.shape[0]
        c0 = int(east_m  / DEM_PIXEL_SIZE) % self._dem.shape[1]
        # Extract with wrap-around
        rows = np.arange(r0, r0 + h_px) % self._dem.shape[0]
        cols = np.arange(c0, c0 + w_px) % self._dem.shape[1]
        return self._dem[np.ix_(rows, cols)]


# ---------------------------------------------------------------------------
# Radar altimeter simulator
# ---------------------------------------------------------------------------

class RadarAltimeterSim:
    """
    Simulates a forward-looking radar altimeter terrain strip.

    Samples the DEM at the *true* aircraft position (with noise) to produce
    the strip that TRN will correlate against the DEM.
    """

    NOISE_STD = 1.5  # m — radar ranging noise 1σ

    def __init__(self, dem: DEMProvider, seed: int = 99):
        self._dem = dem
        self._rng = np.random.default_rng(seed)

    def acquire_strip(self,
                      true_north_m: float,
                      true_east_m:  float) -> np.ndarray:
        """
        Return a (STRIP_LEN_PX × STRIP_WIDTH_PX) terrain strip centred on
        the true aircraft position with additive radar noise.
        """
        patch = self._dem.patch(true_north_m, true_east_m,
                                STRIP_LEN_PX, STRIP_WIDTH_PX)
        noise = self._rng.normal(0, self.NOISE_STD, patch.shape).astype(np.float32)
        return patch + noise


# ---------------------------------------------------------------------------
# NCC correlation engine
# ---------------------------------------------------------------------------

def _normalised_cross_correlation(template: np.ndarray,
                                  search_area: np.ndarray) -> Tuple[float, int, int]:
    """
    Slide template over search_area using NCC.

    Returns (peak_score, row_offset, col_offset) in pixels.
    row_offset / col_offset are the displacement of the best match
    relative to the centre of the search area.
    """
    th, tw = template.shape
    sh, sw = search_area.shape

    mu_t  = template.mean()
    sig_t = template.std()
    if sig_t < 1e-6:   # flat terrain — no correlation possible
        return 0.0, 0, 0

    best_score = -1.0
    best_dr = best_dc = 0

    for r in range(0, sh - th + 1):
        for c in range(0, sw - tw + 1):
            window = search_area[r:r+th, c:c+tw]
            mu_w   = window.mean()
            sig_w  = window.std()
            if sig_w < 1e-6:
                continue
            ncc = float(
                np.sum((template - mu_t) * (window - mu_w)) /
                (sig_t * sig_w * template.size)
            )
            if ncc > best_score:
                best_score = ncc
                # offset = displacement from search area centre
                centre_r = (sh - th) // 2
                centre_c = (sw - tw) // 2
                best_dr = r - centre_r
                best_dc = c - centre_c

    return best_score, best_dr, best_dc


# ---------------------------------------------------------------------------
# INS state (minimal, for TRN correction)
# ---------------------------------------------------------------------------

@dataclass
class INSState:
    """
    Minimal INS navigation state consumed and updated by TRNStub.

    north_m, east_m   — estimated position in local NED frame (m)
    vn, ve            — velocity components (m/s)
    P                 — 2×2 position covariance matrix (m²)
    """
    north_m:  float = 0.0
    east_m:   float = 0.0
    vn:       float = 0.0
    ve:       float = 0.0
    P:        np.ndarray = field(
        default_factory=lambda: np.diag([100.0**2, 100.0**2])  # 100 m initial 1σ
    )


# ---------------------------------------------------------------------------
# TRN correction record
# ---------------------------------------------------------------------------

@dataclass
class TRNCorrection:
    """Record of a single TRN position correction."""
    timestamp_s:     float
    ground_track_m:  float      # total ground track at time of fix
    ncc_score:       float
    delta_north_m:   float      # correction applied Δ north
    delta_east_m:    float      # correction applied Δ east
    accepted:        bool       # True if NCC score ≥ threshold


# ---------------------------------------------------------------------------
# TRN Stub — main class
# ---------------------------------------------------------------------------

class TRNStub:
    """
    Terrain-Referenced Navigation stub.

    Usage:
        dem   = DEMProvider()
        radar = RadarAltimeterSim(dem)
        trn   = TRNStub(dem, radar)

        ins   = INSState(north_m=1000, east_m=500, vn=50, ve=10)

        # Call each navigation tick (e.g. 1 Hz):
        trn.update(
            ins=ins,
            true_north_m=1052.0,   # ground truth (from sim)
            true_east_m=512.0,
            dt=1.0,
            ground_track_m=250.0
        )

        # inspect:
        print(trn.last_correction)
        print(trn.drift_m)         # current INS position error vs ground truth
    """

    def __init__(self,
                 dem:   DEMProvider,
                 radar: RadarAltimeterSim,
                 ncc_threshold: float = NCC_THRESHOLD):
        self._dem   = dem
        self._radar = radar
        self._ncc_threshold       = ncc_threshold
        self._last_correction_gt  = 0.0   # ground track at last correction (m)
        self._corrections: List[TRNCorrection] = []
        self.last_correction: Optional[TRNCorrection] = None
        self._drift_m = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def corrections(self) -> List[TRNCorrection]:
        return list(self._corrections)

    @property
    def correction_count(self) -> int:
        return len(self._corrections)

    @property
    def last_ncc_score(self) -> float:
        return self.last_correction.ncc_score if self.last_correction else 0.0

    @property
    def trn_correlation_valid(self) -> bool:
        """True when last accepted NCC score ≥ threshold (FSM guard input)."""
        if not self.last_correction:
            return False
        return self.last_correction.accepted

    @property
    def drift_m(self) -> float:
        """Absolute INS position error at time of last correction (m)."""
        return self._drift_m

    def update(self,
               ins:            INSState,
               true_north_m:   float,
               true_east_m:    float,
               dt:             float,
               ground_track_m: float,
               timestamp_s:    float = 0.0) -> Optional[TRNCorrection]:
        """
        Attempt a TRN position fix and apply Kalman correction to ins.

        A fix attempt is made only when enough ground track has accumulated
        since the last accepted correction (NAV-01: ≥ 1 correction / 2 km).

        Args:
            ins            : INS state (mutated in-place on accepted fix)
            true_north_m   : ground-truth north position (simulation only)
            true_east_m    : ground-truth east position  (simulation only)
            dt             : time step (s)
            ground_track_m : accumulated ground track from mission start (m)
            timestamp_s    : simulation clock time

        Returns:
            TRNCorrection record, or None if no fix attempted this tick.
        """
        # Propagate INS covariance with process noise (random walk model)
        Q = np.diag([
            (5.0 * dt) ** 2,   # 5 m/s INS north drift 1σ
            (5.0 * dt) ** 2,
        ])
        ins.P = ins.P + Q

        # Check if we should attempt a fix
        since_last = ground_track_m - self._last_correction_gt
        if since_last < CORRECTION_INTERVAL:
            return None

        # Acquire radar strip at TRUE position (what the sensor sees)
        strip = self._radar.acquire_strip(true_north_m, true_east_m)

        # Build DEM search area CENTRED on INS estimate (extends ±SEARCH_PAD in each direction)
        search_h = STRIP_LEN_PX + 2 * SEARCH_PAD_PX
        search_w = STRIP_WIDTH_PX + 2 * SEARCH_PAD_PX
        search_area = self._dem.patch(
            ins.north_m - SEARCH_PAD_PX * DEM_PIXEL_SIZE,
            ins.east_m  - SEARCH_PAD_PX * DEM_PIXEL_SIZE,
            search_h, search_w,
        )

        # NCC correlation
        score, dr, dc = _normalised_cross_correlation(strip, search_area)

        # Convert pixel offset to metres
        delta_north_m = dr * DEM_PIXEL_SIZE  # + north means INS estimated too far south
        delta_east_m  = dc * DEM_PIXEL_SIZE

        accepted = score >= self._ncc_threshold

        if accepted:
            # Kalman measurement update
            # Measurement model: z = [north_meas, east_meas]
            #                    z = ins.position + correction
            z  = np.array([ins.north_m + delta_north_m,
                           ins.east_m  + delta_east_m])
            H  = np.eye(2)
            R  = np.diag([R_TRN_NORTH, R_TRN_EAST])
            S  = H @ ins.P @ H.T + R
            K  = ins.P @ H.T @ np.linalg.inv(S)
            x  = np.array([ins.north_m, ins.east_m])
            innov = z - H @ x
            x_new = x + K @ innov
            ins.north_m = float(x_new[0])
            ins.east_m  = float(x_new[1])
            ins.P = (np.eye(2) - K @ H) @ ins.P

            self._last_correction_gt = ground_track_m

        # Record position error vs ground truth
        self._drift_m = math.hypot(
            ins.north_m - true_north_m,
            ins.east_m  - true_east_m
        )

        corr = TRNCorrection(
            timestamp_s    = timestamp_s,
            ground_track_m = ground_track_m,
            ncc_score      = score,
            delta_north_m  = delta_north_m,
            delta_east_m   = delta_east_m,
            accepted       = accepted,
        )
        self._corrections.append(corr)
        self.last_correction = corr
        return corr

    def reset(self) -> None:
        """Reset state (e.g. on GNSS recovery — TRN re-initialised from GNSS fix)."""
        self._last_correction_gt = 0.0
        self._corrections.clear()
        self.last_correction = None
        self._drift_m = 0.0
