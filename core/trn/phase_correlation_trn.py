"""
core/trn/phase_correlation_trn.py
MicroMind / NanoCorteX — Phase Correlation TRN Engine

NAV-02: Terrain-Relative Navigation using frequency-domain phase correlation.

Algorithm: Wan et al. 2021 (CAS paper).
Normalised cross-power spectrum between UAV nadir camera tile and DEM hillshade
reference tile. Translation estimated via inverse Fourier transform peak location.

This module replaces the SIL stub (core/ins/orthophoto_matching_stub.py).
The stub is preserved as the frozen fallback interface contract reference.

Sensor substitution contract:
    Current camera tile source: Gazebo rendered frame (Gate 2)
    Real replacement: EO camera nadir frame (uint8 numpy array)
    Interface: identical — no architecture change at HIL

CALLER RESPONSIBILITY:
    This module MUST NOT be called if TerrainSuitabilityScorer returns SUPPRESS.
    The caller (NavigationManager) checks suitability before calling match().

References:
    SRS v1.3 NAV-02
    Part Two V7.2 §1.7.2
    CAS paper: Wan et al. 2021
    docs/interfaces/trn_contract.yaml
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from core.trn.dem_loader import DEMLoader
from core.trn.hillshade_generator import HillshadeGenerator
from core.trn.terrain_suitability import TerrainSuitabilityScorer, TerrainSuitabilityResult

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Approximate metres per degree of latitude (WGS-84 sphere)
_M_PER_DEG_LAT: float = 111_320.0

import math


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class TRNMatchResult:
    """
    Output of PhaseCorrelationTRN.match().

    Fields
    ------
    status                  : 'ACCEPTED' | 'REJECTED' | 'SUPPRESSED' | 'OUTSIDE_COVERAGE'
    correction_north_m      : north correction (m) — valid only when ACCEPTED
    correction_east_m       : east correction (m)  — valid only when ACCEPTED
    confidence              : float [0, 1] — phase correlation peak value
    suitability_score       : float [0, 1] from TerrainSuitabilityScorer
    suitability_recommendation : str 'ACCEPT' | 'CAUTION' | 'SUPPRESS'
    latency_ms              : int — processing latency from clock_fn
    mission_time_ms         : int — mission time at call
    """
    status:                     str   # 'ACCEPTED' | 'REJECTED' | 'SUPPRESSED' | 'OUTSIDE_COVERAGE'
    correction_north_m:         float
    correction_east_m:          float
    confidence:                 float
    suitability_score:          float
    suitability_recommendation: str
    latency_ms:                 int
    mission_time_ms:            int


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class PhaseCorrelationTRN:
    """
    NAV-02: Terrain-Relative Navigation using frequency-domain phase correlation.

    Algorithm: Wan et al. 2021 (CAS paper).
    Normalised cross-power spectrum between UAV nadir camera tile and DEM
    hillshade reference tile. Translation estimated via IFT peak location.

    Sensor substitution contract:
        Current camera tile source: Gazebo rendered frame (Gate 2)
        Real replacement: EO camera nadir frame
        Interface: numpy uint8 array, same size as reference tile
    """

    def __init__(
        self,
        dem_loader:          DEMLoader,
        hillshade_gen:       HillshadeGenerator,
        suitability_scorer:  TerrainSuitabilityScorer,
        tile_size_m:         float = 500.0,
        min_peak_value:      float = 0.15,
        clock_fn:            Optional[Callable[[], int]] = None,
    ) -> None:
        """
        dem_loader          : DEMLoader instance (pre-loaded DEM)
        hillshade_gen       : HillshadeGenerator instance
        suitability_scorer  : TerrainSuitabilityScorer instance
        tile_size_m         : physical size of correlation tile (metres)
        min_peak_value      : minimum phase correlation peak to accept match [0,1]
        clock_fn            : callable returning mission time in ms (int).
                              No time.time() calls permitted — inject clock.
        """
        self._dem          = dem_loader
        self._hs_gen       = hillshade_gen
        self._scorer       = suitability_scorer
        self._tile_size_m  = tile_size_m
        self._min_peak     = min_peak_value
        self._clock_fn     = clock_fn

        # Structured event log (list of dicts — programme log schema)
        self._event_log: list = []

    # ── Public API ────────────────────────────────────────────────────────────

    def match(
        self,
        camera_tile:     np.ndarray,
        lat_estimate:    float,
        lon_estimate:    float,
        alt_m:           float,
        gsd_m:           float,
        mission_time_ms: int,
    ) -> TRNMatchResult:
        """
        Attempt phase correlation match between camera_tile and DEM reference tile
        centred at (lat_estimate, lon_estimate).

        Steps:
            1. Check DEM bounds — return OUTSIDE_COVERAGE if not in DEM
            2. Extract DEM elevation tile
            3. Score terrain suitability
            4. If SUPPRESS: return SUPPRESSED with suitability result
            5. Generate hillshade reference tile
            6. Run phase correlation: Q(u,v) = F1·conj(F2) / |F1·conj(F2)|
               peak = |IFT(Q)|; offset = argmax(IFT(Q))
            7. Validate peak value against min_peak_value
            8. Convert pixel offset to metres using gsd_m
            9. Convert metres to lat/lon delta
           10. Log TRN_CORRECTION_ACCEPTED or TRN_CORRECTION_REJECTED

        Returns TRNMatchResult.
        """
        t_start_ms = self._clock_ms()

        # Step 1 — bounds check
        if not self._dem.is_in_bounds(lat_estimate, lon_estimate):
            latency_ms = self._clock_ms() - t_start_ms
            return TRNMatchResult(
                status="OUTSIDE_COVERAGE",
                correction_north_m=0.0,
                correction_east_m=0.0,
                confidence=0.0,
                suitability_score=0.0,
                suitability_recommendation="SUPPRESS",
                latency_ms=latency_ms,
                mission_time_ms=mission_time_ms,
            )

        # Step 2 — extract DEM elevation tile
        elevation_tile = self._dem.get_tile(
            lat_estimate, lon_estimate, self._tile_size_m, gsd_m
        )

        bounds = self._dem.get_bounds()
        dem_resolution_m = bounds["resolution_m"]

        # Step 3 — generate hillshades:
        #   single-direction for texture suitability scoring (higher Laplacian variance)
        #   multi-directional for correlation reference (illumination-invariant)
        hs_single = self._hs_gen.generate(elevation_tile, gsd_m)
        hs_multi  = self._hs_gen.generate_multidirectional(elevation_tile, gsd_m)

        # Score terrain suitability using single-direction hillshade for texture
        suit: TerrainSuitabilityResult = self._scorer.score(
            elevation_tile, hs_single, gsd_m, dem_resolution_m
        )

        # Step 4 — suppress if suitability says so
        if suit.recommendation == "SUPPRESS":
            latency_ms = self._clock_ms() - t_start_ms
            self._log_suppressed(
                mission_time_ms, suit, latency_ms,
                lat_estimate, lon_estimate, alt_m
            )
            return TRNMatchResult(
                status="SUPPRESSED",
                correction_north_m=0.0,
                correction_east_m=0.0,
                confidence=0.0,
                suitability_score=suit.score,
                suitability_recommendation=suit.recommendation,
                latency_ms=latency_ms,
                mission_time_ms=mission_time_ms,
            )

        # Step 5 — use multi-directional hillshade as correlation reference tile
        # (illumination-invariant, CAS paper §3.2)
        reference_tile = hs_multi

        # Ensure camera_tile and reference_tile are same size
        ref_h, ref_w = reference_tile.shape
        if camera_tile.shape != reference_tile.shape:
            # Resize camera tile to match reference (bilinear)
            from scipy.ndimage import zoom as _zoom
            zr = ref_h / camera_tile.shape[0]
            zc = ref_w / camera_tile.shape[1]
            camera_resized = _zoom(
                camera_tile.astype(np.float32), (zr, zc), order=1
            ).astype(np.uint8)
        else:
            camera_resized = camera_tile

        # Step 6 — phase correlation
        peak_value, (row_offset, col_offset) = self._run_phase_correlation(
            reference_tile, camera_resized
        )

        # Step 7 — validate peak
        latency_ms = self._clock_ms() - t_start_ms

        if peak_value < self._min_peak:
            self._log_rejected(
                mission_time_ms, peak_value, self._min_peak,
                "peak_value below min_peak_value", latency_ms,
                lat_estimate, lon_estimate, alt_m
            )
            return TRNMatchResult(
                status="REJECTED",
                correction_north_m=0.0,
                correction_east_m=0.0,
                confidence=float(peak_value),
                suitability_score=suit.score,
                suitability_recommendation=suit.recommendation,
                latency_ms=latency_ms,
                mission_time_ms=mission_time_ms,
            )

        # Step 8 — pixel offset → metres
        # Positive row_offset = image moved down = camera is north of estimate
        # Positive col_offset = image moved right = camera is east of estimate
        # Sign convention: correction is added to INS estimate to move toward truth
        correction_north_m = float(-row_offset * gsd_m)
        correction_east_m  = float(-col_offset * gsd_m)

        # Step 9 — log accepted
        self._log_accepted(
            mission_time_ms, peak_value, suit, correction_north_m,
            correction_east_m, lat_estimate, lon_estimate, alt_m, latency_ms
        )

        return TRNMatchResult(
            status="ACCEPTED",
            correction_north_m=correction_north_m,
            correction_east_m=correction_east_m,
            confidence=float(peak_value),
            suitability_score=suit.score,
            suitability_recommendation=suit.recommendation,
            latency_ms=latency_ms,
            mission_time_ms=mission_time_ms,
        )

    # ── Phase correlation core ────────────────────────────────────────────────

    def _run_phase_correlation(
        self,
        reference: np.ndarray,
        query:     np.ndarray,
    ) -> Tuple[float, Tuple[int, int]]:
        """
        Normalised phase correlation between reference and query tiles.

        Q(u,v) = F1 · conj(F2) / |F1 · conj(F2)|
        peak    = max(|IFT(Q)|)
        offset  = argmax(IFT(Q))

        Uses numpy.fft — no scipy dependency.

        Returns (peak_value: float, (row_offset: int, col_offset: int)).
        peak_value is in [0, 1] (normalised).
        row_offset: shift in rows (pixels); col_offset: shift in cols (pixels).
        Offsets are wrapped around half the tile size ([-H/2, H/2], [-W/2, W/2]).
        """
        ref_f = reference.astype(np.float64)
        qry_f = query.astype(np.float64)

        F1 = np.fft.fft2(ref_f)
        F2 = np.fft.fft2(qry_f)

        cross_power = F1 * np.conj(F2)
        magnitude   = np.abs(cross_power)
        # Avoid division by zero on zero-frequency component or flat tiles
        magnitude   = np.where(magnitude < 1e-12, 1e-12, magnitude)
        Q           = cross_power / magnitude

        ift = np.abs(np.fft.ifft2(Q))

        # Normalise IFT to [0, 1]
        ift_max = float(np.max(ift))
        if ift_max < 1e-12:
            return 0.0, (0, 0)

        # numpy.fft.ifft2 normalises by 1/N² so the peak of an all-unit-amplitude
        # spectrum is 1.0 at perfect match. No additional normalisation needed.
        peak_value = float(np.clip(ift_max, 0.0, 1.0))

        # argmax — use IFT before normalising to preserve location
        flat_idx  = int(np.argmax(ift))
        H, W      = ift.shape
        row_peak  = flat_idx // W
        col_peak  = flat_idx %  W

        # Wrap offsets to [-H/2, H/2] and [-W/2, W/2]
        if row_peak > H // 2:
            row_peak -= H
        if col_peak > W // 2:
            col_peak -= W

        return peak_value, (int(row_peak), int(col_peak))

    # ── Structured event logging ──────────────────────────────────────────────

    def _clock_ms(self) -> int:
        """Return current mission time in ms. Uses injected clock_fn."""
        if self._clock_fn is not None:
            return int(self._clock_fn())
        return 0

    def _log_accepted(
        self,
        mission_time_ms: int,
        confidence: float,
        suit: TerrainSuitabilityResult,
        correction_north_m: float,
        correction_east_m: float,
        lat: float,
        lon: float,
        alt_m: float,
        latency_ms: int,
    ) -> None:
        entry = {
            "event":        "TRN_CORRECTION_ACCEPTED",
            "module_name":  "PhaseCorrelationTRN",
            "req_id":       "NAV-02",
            "severity":     "INFO",
            "timestamp_ms": mission_time_ms,
            "payload": {
                "confidence":         float(confidence),
                "suitability_score":  float(suit.score),
                "correction_north_m": float(correction_north_m),
                "correction_east_m":  float(correction_east_m),
                "lat":                float(lat),
                "lon":                float(lon),
                "alt_m":              float(alt_m),
                "latency_ms":         int(latency_ms),
            },
        }
        self._event_log.append(entry)

    def _log_rejected(
        self,
        mission_time_ms: int,
        peak_value: float,
        min_required: float,
        reason: str,
        latency_ms: int,
        lat: float,
        lon: float,
        alt_m: float,
    ) -> None:
        entry = {
            "event":        "TRN_CORRECTION_REJECTED",
            "module_name":  "PhaseCorrelationTRN",
            "req_id":       "NAV-02",
            "severity":     "WARNING",
            "timestamp_ms": mission_time_ms,
            "payload": {
                "peak_value":   float(peak_value),
                "min_required": float(min_required),
                "reason":       reason,
                "lat":          float(lat),
                "lon":          float(lon),
                "alt_m":        float(alt_m),
                "latency_ms":   int(latency_ms),
            },
        }
        self._event_log.append(entry)

    def _log_suppressed(
        self,
        mission_time_ms: int,
        suit: TerrainSuitabilityResult,
        latency_ms: int,
        lat: float,
        lon: float,
        alt_m: float,
    ) -> None:
        entry = {
            "event":        "TRN_CORRECTION_SUPPRESSED",
            "module_name":  "PhaseCorrelationTRN",
            "req_id":       "NAV-02",
            "severity":     "INFO",
            "timestamp_ms": mission_time_ms,
            "payload": {
                "suitability_score":   float(suit.score),
                "reason":              suit.reason,
                "texture_variance":    float(suit.texture_variance),
                "relief_magnitude_m":  float(suit.relief_magnitude_m),
                "lat":                 float(lat),
                "lon":                 float(lon),
                "alt_m":               float(alt_m),
                "latency_ms":          int(latency_ms),
            },
        }
        self._event_log.append(entry)

    @property
    def event_log(self) -> list:
        """Read-only view of the structured event log."""
        return list(self._event_log)
