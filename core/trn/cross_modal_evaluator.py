"""
core/trn/cross_modal_evaluator.py
MicroMind / NanoCorteX — Cross-Modal TRN Evaluator

NAV-02: Evaluate phase correlation TRN performance in the cross-modal scenario:
    Query:     Blender-rendered RGB frame (simulates real EO camera)
    Reference: DEM hillshade tile (pre-loaded on vehicle)

This is the definitive pre-HIL TRN validation. Self-match tests (peak=1.0)
proved the algorithm. Cross-modal tests prove operational performance.

Expected outcome based on CAS paper (Wan et al. 2021):
    Peak value 0.3–0.7 over textured terrain (ridges, forest).
    Peak < 0.15 over flat valley floors — correctly rejected.
    Average localisation error < 1 pixel at 5m GSD = < 5m position
    error from TRN alone.

Req IDs: NAV-02, AD-01, EC-13
Governance: Code Governance Manual v3.4
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from core.trn.dem_loader import DEMLoader
from core.trn.hillshade_generator import HillshadeGenerator
from core.trn.phase_correlation_trn import PhaseCorrelationTRN
from core.trn.terrain_suitability import TerrainSuitabilityScorer

if TYPE_CHECKING:
    from core.trn.blender_frame_ingestor import BlenderFrameIngestor


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CrossModalResult:
    """
    Result for one Blender frame evaluated through TRN.

    Fields
    ------
    km                          : corridor km of this frame
    status                      : 'ACCEPTED' | 'REJECTED' | 'SUPPRESSED' | 'OUTSIDE_COVERAGE'
    peak_value                  : phase correlation peak [0, 1]
    suitability_score           : terrain suitability score [0, 1]
    suitability_recommendation  : 'ACCEPT' | 'CAUTION' | 'SUPPRESS'
    correction_north_m          : north correction (m) — valid when ACCEPTED
    correction_east_m           : east correction (m) — valid when ACCEPTED
    localisation_error_m        : error vs known offset (m); NaN if not provided
    localisation_error_pixels   : error in pixels; NaN if not provided
    quality                     : 'GOOD' | 'MARGINAL' | 'POOR' from frame validator
    terrain_zone                : terrain zone name derived from km position
    """
    km:                         float
    status:                     str
    peak_value:                 float
    suitability_score:          float
    suitability_recommendation: str
    correction_north_m:         float
    correction_east_m:          float
    localisation_error_m:       float  # NaN if known_offset not provided
    localisation_error_pixels:  float  # NaN if known_offset not provided
    quality:                    str
    terrain_zone:               str


@dataclass
class CrossModalCorridorResult:
    """
    Aggregated result for a full corridor evaluation.

    Fields
    ------
    n_frames            : total frames evaluated
    n_accepted          : frames with ACCEPTED TRN correction
    n_rejected          : frames REJECTED (low peak)
    n_suppressed        : frames SUPPRESSED (terrain suitability)
    mean_peak_accepted  : mean peak value for accepted frames; NaN if none
    peak_values         : all peak values (all frames)
    localisation_errors_m: localisation errors for accepted frames (m)
    p50_error_m         : median localisation error (m)
    p95_error_m         : 95th-percentile localisation error (m)
    p99_error_m         : 99th-percentile localisation error (m)
    per_frame           : list of CrossModalResult (one per frame)
    threshold_calibration: suggested min_peak_value calibration dict
    """
    n_frames:               int
    n_accepted:             int
    n_rejected:             int
    n_suppressed:           int
    mean_peak_accepted:     float
    peak_values:            list[float]
    localisation_errors_m:  list[float]
    p50_error_m:            float
    p95_error_m:            float
    p99_error_m:            float
    per_frame:              list[CrossModalResult]
    threshold_calibration:  dict


# ---------------------------------------------------------------------------
# CrossModalEvaluator
# ---------------------------------------------------------------------------

class CrossModalEvaluator:
    """
    NAV-02: Evaluate phase correlation TRN performance in the cross-modal
    scenario.

    Query:     Blender-rendered RGB frame (simulates real EO camera)
    Reference: DEM hillshade tile (pre-loaded on vehicle)

    This is the definitive pre-HIL TRN validation. Self-match tests
    (peak=1.0) proved the algorithm. Cross-modal tests prove operational
    performance.

    Expected outcome based on CAS paper:
        Peak value 0.3–0.7 over textured terrain (ridges, forest).
        Peak < 0.15 over flat valley floors — correctly rejected.
        Average localisation error < 1 pixel at 5m GSD = < 5m position
        error from TRN alone.
    """

    def __init__(
        self,
        dem_loader:         DEMLoader,
        hillshade_gen:      HillshadeGenerator,
        suitability_scorer: TerrainSuitabilityScorer,
        trn:                PhaseCorrelationTRN,
    ) -> None:
        self._dem      = dem_loader
        self._hs_gen   = hillshade_gen
        self._scorer   = suitability_scorer
        self._trn      = trn

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate_frame(
        self,
        frame_rgb: np.ndarray,
        lat: float,
        lon: float,
        alt_m: float,
        gsd_m: float,
        mission_time_ms: int,
        known_offset_pixels: tuple[int, int] = (0, 0),
    ) -> CrossModalResult:
        """
        Run TRN match on one Blender frame.

        known_offset_pixels : if the frame was rendered at a position
            deliberately offset from the DEM tile centre, provide the
            offset so localisation error can be computed.
            (0,0) means frame rendered at exact DEM tile centre — TRN
            should return near-zero correction.

        Returns CrossModalResult.
        """
        # Convert BGR frame to grayscale for phase correlation
        if frame_rgb.ndim == 3 and frame_rgb.shape[2] == 3:
            camera_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        else:
            camera_gray = frame_rgb.astype(np.uint8)

        # Run TRN match
        match_result = self._trn.match(
            camera_tile=camera_gray,
            lat_estimate=lat,
            lon_estimate=lon,
            alt_m=alt_m,
            gsd_m=gsd_m,
            mission_time_ms=mission_time_ms,
        )

        # Frame quality (use simple Laplacian variance on colour frame)
        quality = self._assess_frame_quality(frame_rgb)

        # Localisation error vs known offset
        if known_offset_pixels == (0, 0):
            # Frame rendered at tile centre — correction should be near-zero
            # Error is the magnitude of the returned correction
            error_north_px = match_result.correction_north_m / gsd_m if gsd_m > 0 else 0.0
            error_east_px  = match_result.correction_east_m  / gsd_m if gsd_m > 0 else 0.0
        else:
            # True offset was (row_off, col_off) pixels
            true_north_corr_m = known_offset_pixels[0] * gsd_m
            true_east_corr_m  = known_offset_pixels[1] * gsd_m
            error_north_px = (
                (match_result.correction_north_m - true_north_corr_m) / gsd_m
                if gsd_m > 0 else 0.0
            )
            error_east_px = (
                (match_result.correction_east_m - true_east_corr_m) / gsd_m
                if gsd_m > 0 else 0.0
            )

        loc_error_px = math.sqrt(error_north_px**2 + error_east_px**2)
        loc_error_m  = loc_error_px * gsd_m

        return CrossModalResult(
            km=0.0,  # caller sets km
            status=match_result.status,
            peak_value=match_result.confidence,
            suitability_score=match_result.suitability_score,
            suitability_recommendation=match_result.suitability_recommendation,
            correction_north_m=match_result.correction_north_m,
            correction_east_m=match_result.correction_east_m,
            localisation_error_m=loc_error_m,
            localisation_error_pixels=loc_error_px,
            quality=quality,
            terrain_zone='',  # set by evaluate_corridor
        )

    def evaluate_corridor(
        self,
        ingestor: 'BlenderFrameIngestor',
    ) -> CrossModalCorridorResult:
        """
        Evaluate all frames in corridor.

        Returns CrossModalCorridorResult with full statistics.
        """
        from core.trn.blender_frame_ingestor import BlenderFrameIngestor as _BFI

        all_frames = ingestor.load_all_frames()
        per_frame: list[CrossModalResult] = []

        mission_time_ms = 0

        # The camera GSD (ingestor._gsd_m) may be MUCH finer than the DEM
        # native resolution (e.g. 0.27m camera vs 30m DEM). Using the raw
        # camera GSD for TRN matching causes the DEM to be over-upsampled,
        # producing near-zero texture → terrain suitability SUPPRESS on all frames.
        #
        # Operational fix (CAS paper §2): TRN correlation is performed at the
        # DEM's native resolution, not the camera's pixel pitch. The camera frame
        # is downscaled inside PhaseCorrelationTRN.match() to match the reference
        # tile. We therefore clamp gsd_m to at least half the DEM native resolution
        # so the reference hillshade always has meaningful texture variance.
        dem_res    = self._dem.get_bounds()['resolution_m']
        camera_gsd = ingestor._gsd_m
        trn_gsd    = max(camera_gsd, dem_res * 0.5)

        for km, frame_bgr, lat, lon, _camera_gsd in all_frames:
            result = self.evaluate_frame(
                frame_rgb=frame_bgr,
                lat=lat,
                lon=lon,
                alt_m=ingestor._altitude_m,
                gsd_m=trn_gsd,
                mission_time_ms=mission_time_ms,
                known_offset_pixels=(0, 0),
            )
            result.km = km
            result.terrain_zone = self._get_terrain_zone(
                km, ingestor._corridor
            )
            per_frame.append(result)
            mission_time_ms += 5000  # 5 s between frames at 100 km/h, 5 km spacing

        # Aggregate statistics
        n_frames    = len(per_frame)
        n_accepted  = sum(1 for r in per_frame if r.status == 'ACCEPTED')
        n_rejected  = sum(1 for r in per_frame if r.status == 'REJECTED')
        n_suppressed = sum(1 for r in per_frame if r.status == 'SUPPRESSED')

        peak_values = [r.peak_value for r in per_frame]

        accepted_peaks = [r.peak_value for r in per_frame if r.status == 'ACCEPTED']
        mean_peak_accepted = float(np.mean(accepted_peaks)) if accepted_peaks else float('nan')

        loc_errors = [r.localisation_error_m for r in per_frame
                      if r.status == 'ACCEPTED']
        if loc_errors:
            p50  = float(np.percentile(loc_errors, 50))
            p95  = float(np.percentile(loc_errors, 95))
            p99  = float(np.percentile(loc_errors, 99))
        else:
            p50 = p95 = p99 = float('nan')

        threshold_calib = self.calibrate_threshold(
            CrossModalCorridorResult(
                n_frames=n_frames,
                n_accepted=n_accepted,
                n_rejected=n_rejected,
                n_suppressed=n_suppressed,
                mean_peak_accepted=mean_peak_accepted,
                peak_values=peak_values,
                localisation_errors_m=loc_errors,
                p50_error_m=p50,
                p95_error_m=p95,
                p99_error_m=p99,
                per_frame=per_frame,
                threshold_calibration={},
            )
        )

        return CrossModalCorridorResult(
            n_frames=n_frames,
            n_accepted=n_accepted,
            n_rejected=n_rejected,
            n_suppressed=n_suppressed,
            mean_peak_accepted=mean_peak_accepted,
            peak_values=peak_values,
            localisation_errors_m=loc_errors,
            p50_error_m=p50,
            p95_error_m=p95,
            p99_error_m=p99,
            per_frame=per_frame,
            threshold_calibration=threshold_calib,
        )

    def calibrate_threshold(
        self,
        results: CrossModalCorridorResult,
    ) -> float:
        """
        Suggest optimal min_peak_value threshold based on observed peak
        distribution.

        Strategy: find the peak value that maximises accepted corrections
        while minimising false positives.

        A false positive is a correction accepted when localisation error
        is large.

        Returns a dict:
            {
                'suggested_threshold': float,
                'n_peaks_observed': int,
                'peak_min': float,
                'peak_max': float,
                'peak_mean': float,
                'rationale': str,
            }
        """
        non_suppressed_peaks = [
            r.peak_value for r in results.per_frame
            if r.status != 'SUPPRESSED' and r.status != 'OUTSIDE_COVERAGE'
        ]

        if not non_suppressed_peaks:
            return {
                'suggested_threshold': 0.15,
                'n_peaks_observed': 0,
                'peak_min': float('nan'),
                'peak_max': float('nan'),
                'peak_mean': float('nan'),
                'rationale': 'No non-suppressed frames; using default 0.15',
            }

        peak_arr  = np.array(non_suppressed_peaks)
        peak_min  = float(np.min(peak_arr))
        peak_max  = float(np.max(peak_arr))
        peak_mean = float(np.mean(peak_arr))

        # Strategy: use p10 of non-suppressed peaks as threshold
        # so that the bottom 10% (likely flat/featureless) are rejected
        # while the bulk are accepted.
        p10 = float(np.percentile(peak_arr, 10))

        # Clamp to [0.05, 0.50] — below 0.05 is noise; above 0.50 too aggressive
        suggested = float(np.clip(p10, 0.05, 0.50))

        return {
            'suggested_threshold': suggested,
            'n_peaks_observed': len(non_suppressed_peaks),
            'peak_min': peak_min,
            'peak_max': peak_max,
            'peak_mean': peak_mean,
            'rationale': (
                f'P10 of {len(non_suppressed_peaks)} non-suppressed peaks = '
                f'{p10:.3f}; clamped to [{0.05:.2f}, {0.50:.2f}]'
            ),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _assess_frame_quality(frame: np.ndarray) -> str:
        """Classify frame quality from Laplacian variance and corners."""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.astype(np.uint8)

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=1000, qualityLevel=0.01, minDistance=10
        )
        n_corners = len(corners) if corners is not None else 0

        if lap_var > 200 and n_corners > 100:
            return 'GOOD'
        elif lap_var >= 50 and n_corners >= 30:
            return 'MARGINAL'
        return 'POOR'

    @staticmethod
    def _get_terrain_zone(km: float, corridor: object) -> str:
        """Look up terrain zone name for given km from corridor definition."""
        zones = getattr(corridor, 'terrain_zones', [])
        for zone in zones:
            if zone.get('km_start', 0) <= km < zone.get('km_end', float('inf')):
                return zone.get('name', '')
        return ''
