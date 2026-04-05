"""
core/ins/orthophoto_matching_stub.py
MicroMind / NanoCorteX — Orthophoto Matching Stub

Implements L2 Absolute Reset per Part Two V7.2 §1.7.2
Replaces RADALT-NCC TRN stub (AD-01, 03 April 2026)
Measurement-provider-only pattern (AD-03)
SIL stub — satellite tile provider is synthetic
OI-05 resolved by this file

Architecture:
    SatelliteTileProvider generates synthetic texture scores and
    match confidence values that mimic the statistics of a real
    orthophoto matcher operating against preloaded satellite tiles.

    OrthophotoMatchingStub.update() is a measurement provider only —
    it returns an OMCorrection dataclass. The caller applies the
    correction via eskf.update_vio() or equivalent. No internal
    Kalman filter (AD-03).

    Texture model:
        High sigma_terrain (>= 30 m) → high match confidence (~0.82)
        Medium sigma_terrain (10–30 m) → marginal confidence (~0.60)
        Featureless (< 10 m) → suppressed confidence (~0.25)

    R matrix:
        OM_R_NORTH = OM_R_EAST = 9.0**2 = 81 m²
        Reflects orthophoto MAE < 7 m (NAV-02 v1.3).
        Previous RADALT-NCC R was 15.0**2 = 225 m².

References:
    Part Two V7.2 §1.7.2   L2 Absolute Reset specification
    SRS v1.3 NAV-02         Orthophoto matching requirements
    AD-01 (2026-04-03)      Navigation architecture decision
    AD-03                   Measurement-provider-only pattern
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Named constants — do not use magic numbers elsewhere in this file
# ---------------------------------------------------------------------------

OM_MATCH_THRESHOLD              = 0.65   # minimum confidence to accept a fix (NAV-02 v1.3, calibration placeholder)
OM_CORRECTION_INTERVAL_MIN_KM   = 2.0    # minimum km between corrections (NAV-02)
OM_CORRECTION_INTERVAL_MAX_KM   = 5.0    # maximum expected gap on textured terrain
OM_MAX_CONSECUTIVE_SUPPRESSED   = 3      # alert threshold for consecutive misses
OM_FEATURELESS_SIGMA_THRESHOLD  = 10.0   # m — below this: featureless, match suppressed
OM_PREFERRED_SIGMA_THRESHOLD    = 30.0   # m — above this: high confidence match

OM_R_NORTH = 9.0 ** 2  # m² — north measurement noise variance (NAV-02 v1.3)
OM_R_EAST  = 9.0 ** 2  # m² — east  measurement noise variance (NAV-02 v1.3)

# mu_confidence values per texture band
_MU_HIGH_TEXTURE    = 0.82   # sigma >= OM_PREFERRED_SIGMA_THRESHOLD
_MU_MEDIUM_TEXTURE  = 0.60   # OM_FEATURELESS_SIGMA_THRESHOLD <= sigma < OM_PREFERRED_SIGMA_THRESHOLD
_MU_FEATURELESS     = 0.25   # sigma < OM_FEATURELESS_SIGMA_THRESHOLD
_CONFIDENCE_NOISE_STD = 0.05  # Gaussian noise on sampled confidence

# Correction residual noise for SIL (simulates imperfect match even when confident)
_CORRECTION_NOISE_STD_M = 3.0  # m — 1σ of position residual after accepted fix

# Sentinel value for om_last_fix_km_ago when no fix has ever been achieved
_NO_FIX_SENTINEL = 999.0


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class OMCorrection:
    """
    Output of OrthophotoMatchingStub.update().
    Consumed by caller via eskf.update_vio() or equivalent.

    Fields
    ------
    timestamp_s               : simulation clock time (s)
    correction_north_m        : north correction to apply (0.0 if not applied)
    correction_east_m         : east  correction to apply (0.0 if not applied)
    match_confidence          : sampled confidence score, 0.0–1.0
    correction_applied        : True if confidence >= OM_MATCH_THRESHOLD and interval gate passed
    consecutive_suppressed_count : number of consecutive suppressed fixes (confidence below threshold)
    om_last_fix_km_ago        : km since last accepted fix (999.0 if no fix ever)
    sigma_terrain             : terrain texture sigma of current tile (m)
    r_matrix                  : 2×2 measurement noise matrix, diag[OM_R_NORTH, OM_R_EAST]
    """
    timestamp_s:                  float
    correction_north_m:           float
    correction_east_m:            float
    match_confidence:             float
    correction_applied:           bool
    consecutive_suppressed_count: int
    om_last_fix_km_ago:           float
    sigma_terrain:                float
    r_matrix:                     np.ndarray


# ---------------------------------------------------------------------------
# Synthetic satellite tile provider
# ---------------------------------------------------------------------------

class SatelliteTileProvider:
    """
    Synthetic preloaded tile provider for SIL testing.

    Produces tiles characterised by sigma_terrain (texture score).
    High sigma_terrain = textured terrain = high match probability.
    Low sigma_terrain = featureless terrain = match suppressed.

    In production this would wrap a geo-referenced raster tile store
    keyed by (north_m, east_m, zoom_level). For SIL we sample a
    Gaussian distribution whose mean is determined by the texture band,
    analogous to how the prior sinusoidal terrain model served NCC SIL.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def _mu_confidence(self, sigma_terrain: float) -> float:
        """Return mean match confidence for the given terrain texture band."""
        if sigma_terrain >= OM_PREFERRED_SIGMA_THRESHOLD:
            return _MU_HIGH_TEXTURE
        elif sigma_terrain >= OM_FEATURELESS_SIGMA_THRESHOLD:
            return _MU_MEDIUM_TEXTURE
        else:
            return _MU_FEATURELESS

    def sample_confidence(self, sigma_terrain: float) -> float:
        """
        Sample match confidence for a tile with the given texture score.

        Returns a value in [0.0, 1.0] drawn from:
            N(mu_confidence(sigma_terrain), _CONFIDENCE_NOISE_STD)

        Parameters
        ----------
        sigma_terrain : float
            Terrain texture standard deviation (m).

        Returns
        -------
        float
            Clipped match confidence in [0.0, 1.0].
        """
        mu = self._mu_confidence(sigma_terrain)
        raw = self._rng.normal(mu, _CONFIDENCE_NOISE_STD)
        return float(np.clip(raw, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Orthophoto Matching Stub — main class
# ---------------------------------------------------------------------------

class OrthophotoMatchingStub:
    """
    Orthophoto Matching Stub — L2 Absolute Reset measurement provider.

    Usage
    -----
        provider = SatelliteTileProvider(seed=42)
        om = OrthophotoMatchingStub(provider)

        correction = om.update(
            pos_north_m  = 15_230.0,
            pos_east_m   = 420.0,
            mission_km   = 3.2,
            sigma_terrain= 45.0,    # from route planner terrain model
        )

        if correction.correction_applied:
            # inject via eskf.update_vio()
            eskf.update_vio(
                delta_north_m = correction.correction_north_m,
                delta_east_m  = correction.correction_east_m,
                R             = correction.r_matrix,
            )

    Design notes
    ------------
    - Measurement provider only (AD-03). No internal Kalman filter.
    - The correction is the residual from accumulated drift simulation:
      in SIL the "true" position is the input; the correction is a
      small Gaussian noise term representing the imperfect match residual.
    - consecutive_suppressed_count resets to 0 on each accepted fix.
    - om_last_fix_km_ago is 999.0 until the first fix is accepted.
    """

    def __init__(
        self,
        tile_provider: Optional[SatelliteTileProvider] = None,
        seed: int = 42,
    ):
        if tile_provider is None:
            tile_provider = SatelliteTileProvider(seed=seed)
        self._provider = tile_provider
        self._rng = np.random.default_rng(seed)

        self._last_fix_km: Optional[float] = None
        self._consecutive_suppressed: int = 0
        self._r_matrix = np.diag([OM_R_NORTH, OM_R_EAST])

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def consecutive_suppressed_count(self) -> int:
        """Number of consecutive below-threshold matches since last accepted fix."""
        return self._consecutive_suppressed

    @property
    def last_fix_km(self) -> Optional[float]:
        """Mission km at which last fix was accepted, or None if never."""
        return self._last_fix_km

    def update(
        self,
        pos_north_m:   float,
        pos_east_m:    float,
        mission_km:    float,
        sigma_terrain: float,
    ) -> OMCorrection:
        """
        Attempt an orthophoto position fix at the current mission position.

        Parameters
        ----------
        pos_north_m : float
            Current INS-estimated north position (m).
        pos_east_m : float
            Current INS-estimated east position (m).
        mission_km : float
            Accumulated mission distance (km) from start.
        sigma_terrain : float
            Terrain texture standard deviation (m) at current position,
            as provided by the route planner terrain model.

        Returns
        -------
        OMCorrection
            Populated correction record. Caller checks .correction_applied
            before injecting into ESKF.
        """
        timestamp_s = datetime.now(timezone.utc).timestamp()

        # -- 1. Correction interval gate --
        if self._last_fix_km is not None:
            km_since_last = mission_km - self._last_fix_km
            if km_since_last < OM_CORRECTION_INTERVAL_MIN_KM:
                return OMCorrection(
                    timestamp_s=timestamp_s,
                    correction_north_m=0.0,
                    correction_east_m=0.0,
                    match_confidence=0.0,
                    correction_applied=False,
                    consecutive_suppressed_count=self._consecutive_suppressed,
                    om_last_fix_km_ago=km_since_last,
                    sigma_terrain=sigma_terrain,
                    r_matrix=self._r_matrix.copy(),
                )

        # -- 2. Sample match confidence from tile provider --
        confidence = self._provider.sample_confidence(sigma_terrain)

        # -- 3 & 4. Apply threshold gate --
        if confidence >= OM_MATCH_THRESHOLD:
            # Accepted fix: correction is small Gaussian residual (SIL)
            correction_north_m = float(
                self._rng.normal(0.0, _CORRECTION_NOISE_STD_M)
            )
            correction_east_m = float(
                self._rng.normal(0.0, _CORRECTION_NOISE_STD_M)
            )
            self._consecutive_suppressed = 0
            prev_fix_km = self._last_fix_km
            self._last_fix_km = mission_km
            km_ago = (mission_km - prev_fix_km) if prev_fix_km is not None else _NO_FIX_SENTINEL

            return OMCorrection(
                timestamp_s=timestamp_s,
                correction_north_m=correction_north_m,
                correction_east_m=correction_east_m,
                match_confidence=confidence,
                correction_applied=True,
                consecutive_suppressed_count=0,
                om_last_fix_km_ago=km_ago,
                sigma_terrain=sigma_terrain,
                r_matrix=self._r_matrix.copy(),
            )
        else:
            # Suppressed fix
            self._consecutive_suppressed += 1
            km_ago = (
                (mission_km - self._last_fix_km)
                if self._last_fix_km is not None
                else _NO_FIX_SENTINEL
            )
            return OMCorrection(
                timestamp_s=timestamp_s,
                correction_north_m=0.0,
                correction_east_m=0.0,
                match_confidence=confidence,
                correction_applied=False,
                consecutive_suppressed_count=self._consecutive_suppressed,
                om_last_fix_km_ago=km_ago,
                sigma_terrain=sigma_terrain,
                r_matrix=self._r_matrix.copy(),
            )

    def reset(self) -> None:
        """Reset stub state (e.g. on re-initialisation from GNSS fix)."""
        self._last_fix_km = None
        self._consecutive_suppressed = 0
