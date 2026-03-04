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
    For SIL we use a procedural terrain model — elevation is computed
    analytically from (north_m, east_m) coordinates using a sum of
    sinusoids with wavelengths of 3–20 km. This produces unique terrain
    at every position along corridors up to 500+ km with no tiling,
    no wrapping, and no aliasing. NCC correlation remains valid at any
    range because each 2.5 km window is genuinely distinct.

    The pre-allocated grid approach used previously had a 2.5 km × 2.5 km
    coverage that repeated 60× over a 150 km corridor, causing NCC to
    return high-scoring but positionally meaningless fixes beyond 2.5 km.

    patch() interface is unchanged — all callers are unaffected.
    """

    # Procedural terrain parameters — wavelengths >> patch size, << corridor
    # Chosen to produce terrain variability at the 200–5000 m scale that
    # radar altimeters can discriminate (FR-107 / NAV-01 compliance).
    _COMPONENTS = [
        # (amplitude_m, wavelength_N_m, wavelength_E_m, phase_N, phase_E)
        # Long-wavelength base terrain (macro topography)
        (60.0,  8_000,  7_000, 0.00, 0.00),
        (40.0, 12_000,  5_000, 1.10, 2.30),
        (30.0,  4_000, 15_000, 0.70, 1.80),
        # Mid-wavelength features (ridges, valleys — NCC discriminability)
        (20.0,    800,    600, 2.50, 0.40),
        (15.0,    500,    900, 1.40, 3.10),
        (12.0,    700,    400, 0.30, 2.70),
        # Short-wavelength texture (unique within search window)
        ( 8.0,    200,    300, 3.00, 0.90),
        ( 6.0,    150,    250, 0.80, 1.50),
        ( 5.0,    300,    200, 2.10, 0.60),
    ]
    _MEAN_ALT_M = 200.0   # mean terrain altitude AGL

    def __init__(self, seed: int = 7):
        # Seed controls small-scale noise amplitude offsets (±10%)
        rng = np.random.default_rng(seed)
        self._amp_jitter = rng.uniform(0.9, 1.1, len(self._COMPONENTS))

    def _elevation(self, north_m: np.ndarray, east_m: np.ndarray) -> np.ndarray:
        """
        Compute terrain elevation (m) at arbitrary (north_m, east_m) coordinates.
        Inputs may be scalars or arrays of any shape (must be broadcastable).
        Returns float32 array of same shape.
        """
        z = np.full_like(north_m, self._MEAN_ALT_M, dtype=np.float64)
        for i, (amp, wl_n, wl_e, ph_n, ph_e) in enumerate(self._COMPONENTS):
            a = amp * self._amp_jitter[i]
            z += a * np.sin(2 * math.pi * north_m / wl_n + ph_n)                    * np.cos(2 * math.pi * east_m  / wl_e + ph_e)
        return z.astype(np.float32)

    def patch(self, north_m: float, east_m: float,
              h_px: int, w_px: int) -> np.ndarray:
        """
        Return a DEM patch of size (h_px × w_px) with its top-left corner
        at (north_m, east_m). Each pixel covers DEM_PIXEL_SIZE metres.
        No wrapping — terrain is unique at every coordinate.
        """
        rows = north_m + np.arange(h_px) * DEM_PIXEL_SIZE   # (h_px,)
        cols = east_m  + np.arange(w_px) * DEM_PIXEL_SIZE   # (w_px,)
        nn, ee = np.meshgrid(rows, cols, indexing='ij')      # (h_px, w_px)
        return self._elevation(nn, ee)


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
                 dem:            DEMProvider,
                 radar:          RadarAltimeterSim,
                 ncc_threshold:  float = NCC_THRESHOLD,
                 search_pad_px:  int   = SEARCH_PAD_PX):
        self._dem             = dem
        self._radar           = radar
        self._ncc_threshold   = ncc_threshold
        self._search_pad_px       = search_pad_px
        self._last_correction_gt  = 0.0
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
               ins_north_m:    float,
               ins_east_m:     float,
               true_north_m:   float,
               true_east_m:    float,
               ground_track_m: float,
               timestamp_s:    float = 0.0,
               ) -> Optional[TRNCorrection]:
        """
        S9-1: Measurement provider only. No internal Kalman. No state mutation.

        Performs NCC correlation, applies threshold gate, returns raw offset
        via TRNCorrection. All corrections applied exclusively through the
        15-state ESKF in the calling simulation loop.

        Args:
            ins_north_m    : INS estimated north position (m)
            ins_east_m     : INS estimated east  position (m)
            true_north_m   : ground-truth north (simulation only)
            true_east_m    : ground-truth east  (simulation only)
            ground_track_m : accumulated ground track from mission start (m)
            timestamp_s    : simulation clock time (s)

        Returns:
            TRNCorrection if fix attempted, else None.
            Caller must check .accepted before injecting into ESKF.
        """
        # Gate: only attempt a fix at CORRECTION_INTERVAL spacing
        since_last = ground_track_m - self._last_correction_gt
        if since_last < CORRECTION_INTERVAL:
            return None
        # Acquire radar strip at TRUE position (what the sensor sees)
        strip = self._radar.acquire_strip(true_north_m, true_east_m)
        # Build DEM search area centred on INS estimate
        pad = self._search_pad_px
        search_h = STRIP_LEN_PX + 2 * pad
        search_w = STRIP_WIDTH_PX + 2 * pad
        search_area = self._dem.patch(
            ins_north_m - pad * DEM_PIXEL_SIZE,
            ins_east_m  - pad * DEM_PIXEL_SIZE,
            search_h, search_w,
        )
        # NCC correlation
        score, dr, dc = _normalised_cross_correlation(strip, search_area)
        # Convert pixel offset to metres
        delta_north_m = dr * DEM_PIXEL_SIZE
        delta_east_m  = dc * DEM_PIXEL_SIZE
        accepted = score >= self._ncc_threshold
        if accepted:
            self._last_correction_gt = ground_track_m
            # Diagnostic: residual drift after applying this fix
            self._drift_m = math.hypot(
                ins_north_m + delta_north_m - true_north_m,
                ins_east_m  + delta_east_m  - true_east_m,
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
