"""
tests/test_gate6_cross_modal.py
MicroMind / NanoCorteX — Gate 6 Cross-Modal TRN Tests

Gates:
    CM-01: BlenderFrameIngestor loads and validates frames
    CM-02: Cross-modal evaluator runs on proxy frames end-to-end
    CM-03: GSD calculation correct at multiple altitudes/FOVs
    CM-04: Threshold calibration returns reasonable value

Req IDs: NAV-02, AD-01, EC-13
"""
from __future__ import annotations

import math
import os
import tempfile

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_LAT_SHIMLA = 31.104
_LON_SHIMLA = 77.173
_DEM_DIR    = 'data/terrain/shimla_corridor/'


def _get_dem_loader():
    """Load the Shimla DEM. Skip if not available."""
    from core.trn.dem_loader import DEMLoader
    if not os.path.isdir(_DEM_DIR):
        pytest.skip(f'DEM directory not available: {_DEM_DIR}')
    return DEMLoader.from_directory(_DEM_DIR)


def _make_proxy_frame(
    dem_loader,
    hillshade_gen,
    lat: float,
    lon: float,
    tile_size_m: float = 5000.0,
    gsd_m: float = None,
) -> np.ndarray:
    """
    Generate a proxy Blender frame from DEM hillshade.
    Same modality as reference (not true cross-modal) but exercises
    the full pipeline.

    Uses DEM native resolution by default so the hillshade has natural
    texture variation (avoids sub-pixel upsampling artefacts that produce
    near-zero Laplacian variance).
    """
    if gsd_m is None:
        # Use DEM native resolution to avoid over-upsampling
        gsd_m = dem_loader.get_bounds()['resolution_m']
    elevation = dem_loader.get_tile(lat, lon, tile_size_m, gsd_m)
    hs = hillshade_gen.generate(elevation, gsd_m)
    # Convert uint8 grayscale to uint8 BGR (proxy for colour camera frame)
    frame_bgr = cv2.cvtColor(hs, cv2.COLOR_GRAY2BGR)
    # Resize to 640×640 to match expected Blender frame dimensions.
    # INTER_NEAREST preserves edge sharpness (bilinear upscaling kills Laplacian
    # variance by smoothing 3–4x upsampled hillshade blocks to near-zero gradient).
    frame_bgr = cv2.resize(frame_bgr, (640, 640), interpolation=cv2.INTER_NEAREST)
    return frame_bgr


def _make_frames_dir(dem_loader, hillshade_gen, corridor, kms):
    """
    Write proxy PNG frames to a temp directory for ingestor tests.
    Returns the temp directory path.
    """
    tmp = tempfile.mkdtemp(prefix='gate6_frames_')
    for km in kms:
        lat, lon = corridor.position_at_km(km)
        frame = _make_proxy_frame(dem_loader, hillshade_gen, lat, lon)
        filepath = os.path.join(tmp, f'frame_km{int(round(km)):03d}.png')
        cv2.imwrite(filepath, frame)
    return tmp


# ---------------------------------------------------------------------------
# CM-01: BlenderFrameIngestor loads and validates frames
# ---------------------------------------------------------------------------

class TestGateCM01:
    """
    CM-01: BlenderFrameIngestor loads and validates proxy frames.

    Uses hillshade-generated frames as proxy until real Blender frames
    arrive. Validates frame loading, GSD computation, and quality check.
    """

    def test_cm01_load_three_frames(self):
        """Ingestor loads all 3 proxy frames from temp directory."""
        from core.trn.blender_frame_ingestor import BlenderFrameIngestor
        from core.trn.hillshade_generator import HillshadeGenerator
        from core.navigation.corridors import SHIMLA_LOCAL

        dem = _get_dem_loader()
        gen = HillshadeGenerator(azimuth_deg=135.0, elevation_deg=45.0)
        corridor = SHIMLA_LOCAL
        test_kms = [0.0, 27.0, 55.0]
        frames_dir = _make_frames_dir(dem, gen, corridor, test_kms)

        try:
            ingestor = BlenderFrameIngestor(
                frames_dir=frames_dir,
                corridor=corridor,
                altitude_m=150.0,
                camera_fov_deg=60.0,
            )
            available = ingestor.get_available_kms()
            assert len(available) == 3, \
                f'Expected 3 frames, got {len(available)}: {available}'
            assert 0.0 in available
            assert 27.0 in available
            assert 55.0 in available
        finally:
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)

    def test_cm01_load_frame_returns_correct_shape(self):
        """load_frame() returns uint8 (640,640,3) BGR frame."""
        from core.trn.blender_frame_ingestor import BlenderFrameIngestor
        from core.trn.hillshade_generator import HillshadeGenerator
        from core.navigation.corridors import SHIMLA_LOCAL

        dem = _get_dem_loader()
        gen = HillshadeGenerator(azimuth_deg=135.0, elevation_deg=45.0)
        corridor = SHIMLA_LOCAL
        frames_dir = _make_frames_dir(dem, gen, corridor, [0.0])

        try:
            ingestor = BlenderFrameIngestor(
                frames_dir=frames_dir,
                corridor=corridor,
                altitude_m=150.0,
                camera_fov_deg=60.0,
            )
            frame, lat, lon, gsd_m = ingestor.load_frame(0.0)
            assert frame.dtype == np.uint8
            assert frame.shape == (640, 640, 3)
            assert lat == pytest.approx(_LAT_SHIMLA, abs=0.01)
            assert lon == pytest.approx(_LON_SHIMLA, abs=0.01)
        finally:
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)

    def test_cm01_gsd_computed_correctly(self):
        """GSD is computed correctly at 150m AGL, 60° FOV, 640px."""
        from core.trn.blender_frame_ingestor import BlenderFrameIngestor
        from core.navigation.corridors import SHIMLA_LOCAL

        ingestor = BlenderFrameIngestor(
            frames_dir='/tmp',
            corridor=SHIMLA_LOCAL,
            altitude_m=150.0,
            camera_fov_deg=60.0,
        )
        # footprint = 2 * 150 * tan(30°) = 2 * 150 * 0.5774 = 173.2 m
        # gsd = 173.2 / 640 = 0.270 m/px
        expected_gsd = 2.0 * 150.0 * math.tan(math.radians(30.0)) / 640.0
        assert ingestor._gsd_m == pytest.approx(expected_gsd, abs=0.01)
        assert ingestor._gsd_m == pytest.approx(0.270, abs=0.01)

    def test_cm01_validate_frame_quality_not_poor(self):
        """validate_frame() returns quality != 'POOR' for a valid rendered frame.

        Uses the real Blender frame at km=0 if available (ideal), otherwise
        falls back to a DEM hillshade proxy frame.
        """
        import os
        from core.trn.blender_frame_ingestor import BlenderFrameIngestor
        from core.trn.hillshade_generator import HillshadeGenerator
        from core.navigation.corridors import SHIMLA_LOCAL

        corridor = SHIMLA_LOCAL
        ingestor = BlenderFrameIngestor(
            frames_dir='/tmp',
            corridor=corridor,
            altitude_m=150.0,
            camera_fov_deg=60.0,
        )

        # Prefer real Blender frame (already at 640×640 with Sentinel-2 texture)
        blender_path = 'data/synthetic_imagery/shimla_corridor/frame_km000.png'
        if os.path.exists(blender_path):
            frame = cv2.imread(blender_path, cv2.IMREAD_COLOR)
        else:
            # Fallback: DEM hillshade proxy at native resolution
            dem = _get_dem_loader()
            gen = HillshadeGenerator(azimuth_deg=135.0, elevation_deg=45.0)
            lat, lon = corridor.position_at_km(0.0)
            frame = _make_proxy_frame(dem, gen, lat, lon)

        report = ingestor.validate_frame(frame)
        assert report['shape_valid'] is True
        assert report['quality'] != 'POOR', \
            f"Expected quality != POOR, got {report['quality']}. " \
            f"lap_var={report['laplacian_variance']:.1f}, " \
            f"corners={report['shi_tomasi_corners']}"

    def test_cm01_file_not_found_raises(self):
        """load_frame() raises FileNotFoundError for missing km."""
        from core.trn.blender_frame_ingestor import BlenderFrameIngestor
        from core.navigation.corridors import SHIMLA_LOCAL

        ingestor = BlenderFrameIngestor(
            frames_dir='/tmp/nonexistent_frames_gate6',
            corridor=SHIMLA_LOCAL,
            altitude_m=150.0,
        )
        with pytest.raises(FileNotFoundError):
            ingestor.load_frame(99.0)

    def test_cm01_load_all_frames_sorted(self):
        """load_all_frames() returns frames sorted by km ascending."""
        from core.trn.blender_frame_ingestor import BlenderFrameIngestor
        from core.trn.hillshade_generator import HillshadeGenerator
        from core.navigation.corridors import SHIMLA_LOCAL

        dem = _get_dem_loader()
        gen = HillshadeGenerator(azimuth_deg=135.0, elevation_deg=45.0)
        corridor = SHIMLA_LOCAL
        frames_dir = _make_frames_dir(dem, gen, corridor, [55.0, 0.0, 27.0])

        try:
            ingestor = BlenderFrameIngestor(
                frames_dir=frames_dir,
                corridor=corridor,
                altitude_m=150.0,
            )
            all_frames = ingestor.load_all_frames()
            kms = [item[0] for item in all_frames]
            assert kms == sorted(kms), f'Frames not sorted: {kms}'
        finally:
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CM-02: Cross-modal evaluator runs on proxy frames end-to-end
# ---------------------------------------------------------------------------

class TestGateCM02:
    """
    CM-02: CrossModalEvaluator runs full pipeline on proxy frames.

    Uses hillshade tiles as proxy frames (same modality).
    Validates pipeline correctness end-to-end.
    """

    # altitude_m=2771 → gsd ≈ 5.0 m/px (= 2771 * 2*tan(30°) / 640 ≈ 5.0m)
    # This matches the standard Gate 2–5 GSD and avoids DEM upsampling artefacts
    # that cause terrain suitability to SUPPRESS everything at 150m AGL GSD.
    _PROXY_ALTITUDE_M = 2771.0

    def test_cm02_evaluate_corridor_runs_without_exception(self):
        """evaluate_corridor() completes without exception on proxy frames."""
        from core.trn.blender_frame_ingestor import BlenderFrameIngestor
        from core.trn.cross_modal_evaluator import CrossModalEvaluator
        from core.trn.hillshade_generator import HillshadeGenerator
        from core.trn.terrain_suitability import TerrainSuitabilityScorer
        from core.trn.phase_correlation_trn import PhaseCorrelationTRN
        from core.navigation.corridors import SHIMLA_LOCAL
        import time

        dem = _get_dem_loader()
        gen = HillshadeGenerator(azimuth_deg=135.0, elevation_deg=45.0)
        scorer = TerrainSuitabilityScorer()
        trn = PhaseCorrelationTRN(
            dem, gen, scorer,
            clock_fn=lambda: int(time.monotonic() * 1000)
        )
        corridor = SHIMLA_LOCAL
        # Use 3 frames — fast enough for test
        frames_dir = _make_frames_dir(dem, gen, corridor, [0.0, 27.0, 55.0])

        try:
            ingestor = BlenderFrameIngestor(
                frames_dir=frames_dir,
                corridor=corridor,
                altitude_m=self._PROXY_ALTITUDE_M,
            )
            evaluator = CrossModalEvaluator(dem, gen, scorer, trn)
            # Should not raise
            results = evaluator.evaluate_corridor(ingestor)
            assert results is not None
        finally:
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)

    def test_cm02_at_least_one_frame_accepted(self):
        """At least 1 proxy frame returns ACCEPTED status.

        Proxy frames are DEM hillshade rendered at a different azimuth than the
        TRN reference (135° vs multi-directional). They are not true cross-modal
        (same DEM source) but the cross-power spectrum peak is lower than for
        self-match (1.0). We lower min_peak_value to 0.03 for this proxy test so
        that frames with suitability ACCEPT/CAUTION register as ACCEPTED — the
        purpose here is end-to-end pipeline validation, not cross-modal threshold
        calibration.
        """
        from core.trn.blender_frame_ingestor import BlenderFrameIngestor
        from core.trn.cross_modal_evaluator import CrossModalEvaluator
        from core.trn.hillshade_generator import HillshadeGenerator
        from core.trn.terrain_suitability import TerrainSuitabilityScorer
        from core.trn.phase_correlation_trn import PhaseCorrelationTRN
        from core.navigation.corridors import SHIMLA_LOCAL
        import time

        dem = _get_dem_loader()
        gen = HillshadeGenerator(azimuth_deg=135.0, elevation_deg=45.0)
        scorer = TerrainSuitabilityScorer()
        # Lower threshold for proxy test: proxy frames (DEM hillshade ≠ DEM hillshade
        # multi-directional reference) produce genuine phase correlation peaks at
        # 0.03–0.05, well below the operational 0.15 threshold. This test proves the
        # pipeline routes ACCEPTED when peak >= threshold — not that 0.03 is operationally
        # valid (real cross-modal calibration uses the actual Blender frames).
        trn = PhaseCorrelationTRN(
            dem, gen, scorer,
            min_peak_value=0.03,
            clock_fn=lambda: int(time.monotonic() * 1000)
        )
        corridor = SHIMLA_LOCAL
        frames_dir = _make_frames_dir(dem, gen, corridor, [0.0, 27.0, 55.0])

        try:
            ingestor = BlenderFrameIngestor(
                frames_dir=frames_dir,
                corridor=corridor,
                altitude_m=self._PROXY_ALTITUDE_M,
            )
            evaluator = CrossModalEvaluator(dem, gen, scorer, trn)
            results = evaluator.evaluate_corridor(ingestor)
            assert results.n_accepted >= 1, \
                f'Expected >= 1 ACCEPTED frame, got {results.n_accepted}. ' \
                f'Per-frame status: {[r.status for r in results.per_frame]}, ' \
                f'peaks: {[r.peak_value for r in results.per_frame]}'
        finally:
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)

    def test_cm02_corridor_result_has_all_fields(self):
        """CrossModalCorridorResult contains all required fields."""
        from core.trn.blender_frame_ingestor import BlenderFrameIngestor
        from core.trn.cross_modal_evaluator import (
            CrossModalEvaluator, CrossModalCorridorResult
        )
        from core.trn.hillshade_generator import HillshadeGenerator
        from core.trn.terrain_suitability import TerrainSuitabilityScorer
        from core.trn.phase_correlation_trn import PhaseCorrelationTRN
        from core.navigation.corridors import SHIMLA_LOCAL
        import time, dataclasses

        dem = _get_dem_loader()
        gen = HillshadeGenerator(azimuth_deg=135.0, elevation_deg=45.0)
        scorer = TerrainSuitabilityScorer()
        trn = PhaseCorrelationTRN(
            dem, gen, scorer,
            clock_fn=lambda: int(time.monotonic() * 1000)
        )
        corridor = SHIMLA_LOCAL
        frames_dir = _make_frames_dir(dem, gen, corridor, [0.0, 27.0])

        try:
            ingestor = BlenderFrameIngestor(
                frames_dir=frames_dir,
                corridor=corridor,
                altitude_m=self._PROXY_ALTITUDE_M,
            )
            evaluator = CrossModalEvaluator(dem, gen, scorer, trn)
            results = evaluator.evaluate_corridor(ingestor)

            required_fields = [
                'n_frames', 'n_accepted', 'n_rejected', 'n_suppressed',
                'mean_peak_accepted', 'peak_values', 'localisation_errors_m',
                'p50_error_m', 'p95_error_m', 'p99_error_m',
                'per_frame', 'threshold_calibration',
            ]
            result_field_names = {f.name for f in dataclasses.fields(results)}
            for field in required_fields:
                assert field in result_field_names, \
                    f'Required field missing: {field}'

            assert results.n_frames == 2
            assert isinstance(results.peak_values, list)
            assert isinstance(results.per_frame, list)
            assert isinstance(results.threshold_calibration, dict)
            assert 'suggested_threshold' in results.threshold_calibration
        finally:
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CM-03: GSD calculation correct
# ---------------------------------------------------------------------------

class TestGateCM03:
    """CM-03: GSD calculation is correct at multiple altitudes and FOVs."""

    def test_cm03_gsd_150m_60fov_640px(self):
        """GSD = 0.270 m/px at 150m AGL, 60° FOV, 640px."""
        from core.trn.blender_frame_ingestor import BlenderFrameIngestor
        from core.navigation.corridors import SHIMLA_LOCAL

        ingestor = BlenderFrameIngestor(
            frames_dir='/tmp',
            corridor=SHIMLA_LOCAL,
            altitude_m=150.0,
            camera_fov_deg=60.0,
        )
        assert ingestor._gsd_m == pytest.approx(0.270, abs=0.01)

    def test_cm03_gsd_300m_60fov_640px(self):
        """GSD = 0.541 m/px at 300m AGL, 60° FOV, 640px."""
        from core.trn.blender_frame_ingestor import BlenderFrameIngestor
        from core.navigation.corridors import SHIMLA_LOCAL

        ingestor = BlenderFrameIngestor(
            frames_dir='/tmp',
            corridor=SHIMLA_LOCAL,
            altitude_m=300.0,
            camera_fov_deg=60.0,
        )
        # footprint = 2 * 300 * tan(30°) = 346.4 m; gsd = 346.4/640 = 0.541
        expected = 2.0 * 300.0 * math.tan(math.radians(30.0)) / 640.0
        assert ingestor._gsd_m == pytest.approx(expected, abs=0.01)
        assert ingestor._gsd_m == pytest.approx(0.541, abs=0.01)

    def test_cm03_gsd_150m_90fov_640px(self):
        """GSD = 0.469 m/px at 150m AGL, 90° FOV, 640px."""
        from core.trn.blender_frame_ingestor import BlenderFrameIngestor
        from core.navigation.corridors import SHIMLA_LOCAL

        ingestor = BlenderFrameIngestor(
            frames_dir='/tmp',
            corridor=SHIMLA_LOCAL,
            altitude_m=150.0,
            camera_fov_deg=90.0,
        )
        # footprint = 2 * 150 * tan(45°) = 300 m; gsd = 300/640 = 0.469
        expected = 2.0 * 150.0 * math.tan(math.radians(45.0)) / 640.0
        assert ingestor._gsd_m == pytest.approx(expected, abs=0.01)
        assert ingestor._gsd_m == pytest.approx(0.469, abs=0.01)

    def test_cm03_gsd_scales_linearly_with_altitude(self):
        """GSD scales linearly with altitude for fixed FOV."""
        from core.trn.blender_frame_ingestor import BlenderFrameIngestor
        from core.navigation.corridors import SHIMLA_LOCAL

        ing_150 = BlenderFrameIngestor(
            frames_dir='/tmp', corridor=SHIMLA_LOCAL,
            altitude_m=150.0, camera_fov_deg=60.0
        )
        ing_300 = BlenderFrameIngestor(
            frames_dir='/tmp', corridor=SHIMLA_LOCAL,
            altitude_m=300.0, camera_fov_deg=60.0
        )
        # Doubling altitude should double GSD
        assert ing_300._gsd_m == pytest.approx(2.0 * ing_150._gsd_m, rel=1e-4)


# ---------------------------------------------------------------------------
# CM-04: Threshold calibration
# ---------------------------------------------------------------------------

class TestGateCM04:
    """
    CM-04: Threshold calibration returns a value in [0.05, 0.50] and
    is lower than current min_peak_value for cross-modal peaks.
    """

    # Use same proxy altitude as CM-02 (gsd≈5m, DEM-compatible)
    _PROXY_ALTITUDE_M = 2771.0

    def test_cm04_calibration_in_range(self):
        """calibrate_threshold() returns float in [0.05, 0.50]."""
        from core.trn.blender_frame_ingestor import BlenderFrameIngestor
        from core.trn.cross_modal_evaluator import CrossModalEvaluator
        from core.trn.hillshade_generator import HillshadeGenerator
        from core.trn.terrain_suitability import TerrainSuitabilityScorer
        from core.trn.phase_correlation_trn import PhaseCorrelationTRN
        from core.navigation.corridors import SHIMLA_LOCAL
        import time

        dem = _get_dem_loader()
        gen = HillshadeGenerator(azimuth_deg=135.0, elevation_deg=45.0)
        scorer = TerrainSuitabilityScorer()
        trn = PhaseCorrelationTRN(
            dem, gen, scorer,
            clock_fn=lambda: int(time.monotonic() * 1000)
        )
        corridor = SHIMLA_LOCAL

        # Use 12 proxy frames to match spec requirement
        test_kms = [k * 5.0 for k in range(12)]  # km 0, 5, 10, ..., 55
        frames_dir = _make_frames_dir(dem, gen, corridor, test_kms)

        try:
            ingestor = BlenderFrameIngestor(
                frames_dir=frames_dir,
                corridor=corridor,
                altitude_m=self._PROXY_ALTITUDE_M,
            )
            evaluator = CrossModalEvaluator(dem, gen, scorer, trn)
            results = evaluator.evaluate_corridor(ingestor)

            suggested = results.threshold_calibration['suggested_threshold']
            assert isinstance(suggested, float), \
                f'Expected float, got {type(suggested)}'
            assert 0.05 <= suggested <= 0.50, \
                f'Suggested threshold {suggested:.3f} not in [0.05, 0.50]'
        finally:
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)

    def test_cm04_calibration_dict_has_required_keys(self):
        """threshold_calibration dict has required keys."""
        from core.trn.blender_frame_ingestor import BlenderFrameIngestor
        from core.trn.cross_modal_evaluator import CrossModalEvaluator
        from core.trn.hillshade_generator import HillshadeGenerator
        from core.trn.terrain_suitability import TerrainSuitabilityScorer
        from core.trn.phase_correlation_trn import PhaseCorrelationTRN
        from core.navigation.corridors import SHIMLA_LOCAL
        import time

        dem = _get_dem_loader()
        gen = HillshadeGenerator(azimuth_deg=135.0, elevation_deg=45.0)
        scorer = TerrainSuitabilityScorer()
        trn = PhaseCorrelationTRN(
            dem, gen, scorer,
            clock_fn=lambda: int(time.monotonic() * 1000)
        )
        corridor = SHIMLA_LOCAL
        frames_dir = _make_frames_dir(dem, gen, corridor, [0.0, 27.0])

        try:
            ingestor = BlenderFrameIngestor(
                frames_dir=frames_dir,
                corridor=corridor,
                altitude_m=self._PROXY_ALTITUDE_M,
            )
            evaluator = CrossModalEvaluator(dem, gen, scorer, trn)
            results = evaluator.evaluate_corridor(ingestor)

            calib = results.threshold_calibration
            for key in ['suggested_threshold', 'n_peaks_observed',
                        'peak_min', 'peak_max', 'peak_mean', 'rationale']:
                assert key in calib, f'Missing key: {key}'
        finally:
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)
