"""
core/trn/terrain_suitability.py
MicroMind / NanoCorteX — Terrain Suitability Scorer

NAV-02: Assess terrain suitability for TRN correction before each match attempt.

A system that applies corrections without knowing when terrain is reliable is
not a product. This module is what distinguishes MicroMind from a stub.

Scores terrain on four axes:
    1. Texture variance   — featureless terrain produces unreliable phase correlation
    2. Relief magnitude   — flat terrain produces weak spectral structure
    3. GSD validity       — if current GSD > DEM resolution, correction is suppressed
    4. Composite score    — weighted combination → ACCEPT / CAUTION / SUPPRESS

References:
    SRS v1.3 NAV-02
    Part Two V7.2 §1.7.2
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Default thresholds — match config/tunable_mission.yaml entries
# These values must not be hardcoded at call sites; pass via scorer constructor.
# ---------------------------------------------------------------------------
_DEFAULT_TEXTURE_VAR_MIN:  float = 50.0   # Laplacian variance units; below = LOW texture
_DEFAULT_RELIEF_MAG_MIN_M: float = 20.0   # metres; below = LOW relief
_DEFAULT_GSD_MAX_RATIO:    float = 2.0    # gsd / dem_resolution; above = SUPPRESS


@dataclass
class TerrainSuitabilityResult:
    """
    Output of TerrainSuitabilityScorer.score().

    Fields
    ------
    score                 : float [0.0, 1.0] — 1.0 = highly suitable, 0.0 = suppress
    texture_variance      : float — Laplacian variance of hillshade tile
    relief_magnitude_m    : float — max − min elevation in tile (metres)
    gsd_ratio             : float — gsd_m / dem_resolution_m
    recommendation        : str — 'ACCEPT' | 'CAUTION' | 'SUPPRESS'
    reason                : str — human-readable explanation
    """
    score:              float
    texture_variance:   float
    relief_magnitude_m: float
    gsd_ratio:          float
    recommendation:     str    # 'ACCEPT' | 'CAUTION' | 'SUPPRESS'
    reason:             str


class TerrainSuitabilityScorer:
    """
    NAV-02: Assess terrain suitability for TRN correction before each match attempt.

    Scores terrain on four axes:
        1. Texture variance  — featureless terrain produces unreliable phase correlation
        2. Relief magnitude  — flat terrain produces weak spectral structure
        3. GSD validity      — if GSD is coarser than DEM resolution, suppress
        4. Terrain class     — (future) known unreliable classes flagged explicitly

    Thresholds sourced from config/tunable_mission.yaml; constructor accepts them
    as explicit parameters so they are never magic numbers at call sites.
    """

    def __init__(
        self,
        texture_var_min:  float = _DEFAULT_TEXTURE_VAR_MIN,
        relief_mag_min_m: float = _DEFAULT_RELIEF_MAG_MIN_M,
        gsd_max_ratio:    float = _DEFAULT_GSD_MAX_RATIO,
    ) -> None:
        """
        texture_var_min  : Laplacian variance below which texture is LOW (suppress)
        relief_mag_min_m : Relief magnitude (m) below which terrain is flat (suppress)
        gsd_max_ratio    : gsd_m / dem_resolution_m above which matching is suppressed
        """
        self.TEXTURE_VAR_MIN  = texture_var_min
        self.RELIEF_MAG_MIN_M = relief_mag_min_m
        self.GSD_MAX_RATIO    = gsd_max_ratio

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        elevation_tile:  np.ndarray,
        hillshade_tile:  np.ndarray,
        gsd_m:           float,
        dem_resolution_m: float,
    ) -> TerrainSuitabilityResult:
        """
        Assess terrain suitability and return a TerrainSuitabilityResult.

        Parameters
        ----------
        elevation_tile   : float32 (H,W) elevation array from DEMLoader.get_tile()
        hillshade_tile   : uint8  (H,W) hillshade from HillshadeGenerator
        gsd_m            : current ground sample distance (m)
        dem_resolution_m : DEM pixel resolution (m) from DEMLoader.get_bounds()

        Returns
        -------
        TerrainSuitabilityResult with score, recommendation, and diagnostics.
        """
        texture_var    = self._compute_texture_variance(hillshade_tile)
        relief_mag_m   = self._compute_relief_magnitude(elevation_tile)
        gsd_ratio      = gsd_m / dem_resolution_m if dem_resolution_m > 0 else float("inf")
        gsd_valid      = self._check_gsd_validity(gsd_m, dem_resolution_m)

        # ── Hard SUPPRESS conditions ─────────────────────────────────────────
        suppress_reasons = []
        if not gsd_valid:
            suppress_reasons.append(
                f"GSD ratio {gsd_ratio:.2f} > {self.GSD_MAX_RATIO:.2f} "
                f"(GSD {gsd_m:.1f} m > DEM {dem_resolution_m:.1f} m × {self.GSD_MAX_RATIO})"
            )
        if texture_var < self.TEXTURE_VAR_MIN:
            suppress_reasons.append(
                f"texture_variance {texture_var:.1f} < min {self.TEXTURE_VAR_MIN:.1f} (featureless)"
            )
        if relief_mag_m < self.RELIEF_MAG_MIN_M:
            suppress_reasons.append(
                f"relief {relief_mag_m:.1f} m < min {self.RELIEF_MAG_MIN_M:.1f} m (flat terrain)"
            )

        if suppress_reasons:
            return TerrainSuitabilityResult(
                score=0.0,
                texture_variance=texture_var,
                relief_magnitude_m=relief_mag_m,
                gsd_ratio=gsd_ratio,
                recommendation="SUPPRESS",
                reason="; ".join(suppress_reasons),
            )

        # ── Compute composite score [0, 1] ────────────────────────────────────
        # Normalise texture variance against a reasonable upper bound (1000 units)
        # and relief magnitude against a high-confidence bound (200 m).
        # These are soft signals — hard thresholds already applied above.
        _TEXTURE_SCALE  = 1000.0
        _RELIEF_SCALE_M = 200.0

        tex_score    = min(texture_var    / _TEXTURE_SCALE,  1.0)
        relief_score = min(relief_mag_m   / _RELIEF_SCALE_M, 1.0)
        # GSD score: 1.0 when GSD = 0, degrades linearly to 0 at GSD_MAX_RATIO
        gsd_score    = max(0.0, 1.0 - (gsd_ratio / self.GSD_MAX_RATIO))

        # Weighted combination: texture and relief carry equal weight; GSD is a
        # soft penalty on top.
        composite = (0.45 * tex_score + 0.45 * relief_score + 0.10 * gsd_score)
        composite = float(np.clip(composite, 0.0, 1.0))

        if composite >= 0.60:
            recommendation = "ACCEPT"
            reason = (
                f"texture_variance={texture_var:.1f}, "
                f"relief={relief_mag_m:.1f} m, gsd_ratio={gsd_ratio:.2f}"
            )
        else:
            recommendation = "CAUTION"
            reason = (
                f"marginal suitability: texture_variance={texture_var:.1f}, "
                f"relief={relief_mag_m:.1f} m, gsd_ratio={gsd_ratio:.2f}"
            )

        return TerrainSuitabilityResult(
            score=composite,
            texture_variance=texture_var,
            relief_magnitude_m=relief_mag_m,
            gsd_ratio=gsd_ratio,
            recommendation=recommendation,
            reason=reason,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_texture_variance(self, hillshade: np.ndarray) -> float:
        """
        Laplacian variance of hillshade tile.
        High = rich texture, Low = featureless terrain.

        Uses the discrete Laplacian kernel:
            [[ 0, -1,  0],
             [-1,  4, -1],
             [ 0, -1,  0]]
        Variance of the response is the texture richness metric.
        """
        arr = hillshade.astype(np.float64)
        # Apply discrete Laplacian via slicing (avoids scipy dependency)
        lap = (
              arr[:-2, 1:-1]   # top
            + arr[2:,  1:-1]   # bottom
            + arr[1:-1, :-2]   # left
            + arr[1:-1, 2:]    # right
            - 4.0 * arr[1:-1, 1:-1]  # centre × 4
        )
        return float(np.var(lap))

    def _compute_relief_magnitude(self, elevation: np.ndarray) -> float:
        """
        Max − min elevation in tile (metres).
        Low relief = flat terrain = weak TRN spectral structure.
        """
        valid = elevation[~np.isnan(elevation)]
        if valid.size == 0:
            return 0.0
        return float(np.max(valid) - np.min(valid))

    def _check_gsd_validity(
        self,
        gsd_m: float,
        dem_resolution_m: float,
    ) -> bool:
        """
        Return False if gsd_m > dem_resolution_m × GSD_MAX_RATIO.
        Matching at GSD coarser than DEM resolution is meaningless —
        the reference tile contains no more information than the sensor can resolve.
        """
        if dem_resolution_m <= 0:
            return False
        return gsd_m <= dem_resolution_m * self.GSD_MAX_RATIO
