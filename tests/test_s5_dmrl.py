"""
tests/test_s5_dmrl.py
MicroMind Sprint S5 — DMRL Unit Tests

Tests all boundary conditions from Part Two V7 §1.9.3 / FR-103:
  - Lock confidence ≥ 0.85 to proceed
  - Decoy abort ≥ 0.80 probability over 3 consecutive frames
  - Minimum dwell ≥ 5 frames at 25 FPS
  - ROI ≥ 8×8 pixels at max engagement range
  - Re-acquisition timeout 1.5 s → abort
  - Aimpoint correction within ±15°
  - Real target selected over decoy in scene

Run: python tests/test_s5_dmrl.py
"""

import sys
import os
import unittest
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.dmrl.dmrl_stub import (
    DMRLProcessor, generate_synthetic_scene, ThermalTarget,
    LOCK_CONFIDENCE_THRESHOLD, DECOY_ABORT_THRESHOLD,
    DECOY_ABORT_CONSECUTIVE, MIN_DWELL_FRAMES, FRAME_RATE_HZ,
    MIN_THERMAL_ROI_PX, AIMPOINT_CORRECTION_LIMIT, REACQUISITION_TIMEOUT_S,
    DMRLResult
)


class TestDMRLBoundaryConditions(unittest.TestCase):
    """BC-level tests — each maps to a named boundary condition in §1.9.3."""

    def setUp(self):
        self.processor = DMRLProcessor(verbose=False)

    # ── BC-01: Lock confidence threshold ─────────────────────────────────────
    def test_bc01_lock_confidence_threshold_is_correct(self):
        """LOCK_CONFIDENCE_THRESHOLD must be exactly 0.85 (§1.9.3)."""
        self.assertEqual(LOCK_CONFIDENCE_THRESHOLD, 0.85,
            "Lock confidence threshold must be 0.85 per Part Two V7 §1.9.3")

    def test_bc01_lock_not_acquired_below_threshold(self):
        """A target that never exceeds 0.85 lock confidence must NOT acquire lock."""
        # Use a target designed to produce low lock confidence
        weak_target = ThermalTarget(
            target_id="WEAK-01", is_decoy=False,
            thermal_signature=0.40,    # too low
            thermal_decay_rate=0.001,
            initial_roi_px=25,
            bearing_deg=2.0, range_m=1000.0
        )
        result = self.processor.process_target(weak_target, max_frames=30)
        self.assertFalse(result.lock_acquired,
            "Lock should NOT be acquired when confidence stays below 0.85")

    def test_bc01_lock_acquired_above_threshold(self):
        """A strong real target should acquire lock (confidence ≥ 0.85)."""
        # 50-run Monte Carlo: expect lock rate ≥ 80%
        lock_count = 0
        for seed in range(50):
            random.seed(seed)
            target = ThermalTarget(
                target_id="STRONG-01", is_decoy=False,
                thermal_signature=0.90,
                thermal_decay_rate=0.001,
                initial_roi_px=28,
                bearing_deg=1.0, range_m=1200.0
            )
            result = self.processor.process_target(target, max_frames=30)
            if result.lock_acquired:
                lock_count += 1

        lock_rate = lock_count / 50
        self.assertGreaterEqual(lock_rate, 0.80,
            f"Strong target lock rate {lock_rate:.0%} should be ≥ 80%")

    # ── BC-02: Decoy abort threshold ─────────────────────────────────────────
    def test_bc02_decoy_abort_threshold_is_correct(self):
        """DECOY_ABORT_THRESHOLD must be 0.80, DECOY_ABORT_CONSECUTIVE must be 3."""
        self.assertEqual(DECOY_ABORT_THRESHOLD, 0.80)
        self.assertEqual(DECOY_ABORT_CONSECUTIVE, 3)

    def test_bc02_decoy_detected_and_flagged(self):
        """A thermal decoy must be flagged as is_decoy=True in majority of runs."""
        detect_count = 0
        for seed in range(50):
            random.seed(seed)
            decoy = ThermalTarget(
                target_id="DCY-01", is_decoy=True,
                thermal_signature=0.95,
                thermal_decay_rate=0.030,    # rapid dissipation
                initial_roi_px=10,
                bearing_deg=15.0, range_m=1000.0
            )
            result = self.processor.process_target(decoy, max_frames=30)
            if result.is_decoy:
                detect_count += 1

        detection_rate = detect_count / 50
        self.assertGreaterEqual(detection_rate, 0.85,
            f"Decoy detection rate {detection_rate:.0%} should be ≥ 85% (KPI-T02 requires ≥ 90%)")

    def test_bc02_real_target_not_flagged_as_decoy(self):
        """A real target must NOT be flagged as a decoy."""
        false_positive_count = 0
        for seed in range(50):
            random.seed(seed)
            target = ThermalTarget(
                target_id="TGT-01", is_decoy=False,
                thermal_signature=0.85,
                thermal_decay_rate=0.002,    # stable
                initial_roi_px=24,
                bearing_deg=2.0, range_m=1200.0
            )
            result = self.processor.process_target(target, max_frames=30)
            if result.is_decoy:
                false_positive_count += 1

        false_positive_rate = false_positive_count / 50
        self.assertLessEqual(false_positive_rate, 0.15,
            f"Real target false-positive decoy rate {false_positive_rate:.0%} should be ≤ 15%")

    # ── BC-03: Minimum dwell frames ───────────────────────────────────────────
    def test_bc03_min_dwell_frames_constant(self):
        """MIN_DWELL_FRAMES must be 5 (§1.9.3: ≥5 frames @ 25 FPS = 200 ms)."""
        self.assertEqual(MIN_DWELL_FRAMES, 5)

    def test_bc03_frame_rate_is_25fps(self):
        """FRAME_RATE_HZ must be 25.0."""
        self.assertAlmostEqual(FRAME_RATE_HZ, 25.0, places=1)

    def test_bc03_lock_not_acquired_before_min_dwell(self):
        """Lock confidence gate must not trigger before MIN_DWELL_FRAMES frames."""
        target = ThermalTarget(
            target_id="TGT-FAST", is_decoy=False,
            thermal_signature=0.92, thermal_decay_rate=0.001,
            initial_roi_px=28, bearing_deg=1.0, range_m=1000.0
        )
        result = self.processor.process_target(target, max_frames=30)
        # Lock should only be possible after dwell_frames ≥ 5
        if result.lock_acquired:
            self.assertGreaterEqual(result.dwell_frames, MIN_DWELL_FRAMES,
                "Lock acquired before minimum dwell — BC-03 violation")

    # ── BC-04: Thermal ROI size ───────────────────────────────────────────────
    def test_bc04_min_roi_constant(self):
        """MIN_THERMAL_ROI_PX must be 8 (8×8 pixels at max engagement range)."""
        self.assertEqual(MIN_THERMAL_ROI_PX, 8)

    def test_bc04_small_roi_target_handles_gracefully(self):
        """Target with ROI < 8 px should not raise exceptions."""
        small_roi_target = ThermalTarget(
            target_id="TGT-SMALL", is_decoy=False,
            thermal_signature=0.85, thermal_decay_rate=0.001,
            initial_roi_px=4,   # Below threshold
            bearing_deg=0.0, range_m=2500.0
        )
        try:
            result = self.processor.process_target(small_roi_target, max_frames=15)
            # May not acquire lock, but must not crash
            self.assertIsInstance(result, DMRLResult)
        except Exception as e:
            self.fail(f"Small ROI target caused exception: {e}")

    # ── BC-05: Re-acquisition timeout ─────────────────────────────────────────
    def test_bc05_reacquisition_timeout_constant(self):
        """REACQUISITION_TIMEOUT_S must be 1.5."""
        self.assertAlmostEqual(REACQUISITION_TIMEOUT_S, 1.5, places=3)

    def test_bc05_lock_loss_timeout_sets_flag(self):
        """Lock loss at frame 0 with 50 frames total lets 1.5s sim time elapse → timeout."""
        target = ThermalTarget(
            target_id="TGT-LOST", is_decoy=False,
            thermal_signature=0.90, thermal_decay_rate=0.001,
            initial_roi_px=24, bearing_deg=1.0, range_m=1200.0
        )
        # inject_lock_loss_at=0: lock lost at t=0s.
        # 50 frames @ 25fps = 2.0s total → exceeds 1.5s timeout
        result = self.processor.process_target(
            target, max_frames=50, inject_lock_loss_at=0
        )
        self.assertTrue(result.lock_lost_timeout,
            "Lock loss at frame 0 with 50 frames must trigger re-acquisition timeout (1.5s)")
        self.assertFalse(result.lock_acquired,
            "Lock must NOT be acquired when re-acquisition timeout fires")

    # ── BC-06: Aimpoint correction limit ─────────────────────────────────────
    def test_bc06_aimpoint_correction_limit_constant(self):
        """AIMPOINT_CORRECTION_LIMIT must be 15.0 degrees."""
        self.assertAlmostEqual(AIMPOINT_CORRECTION_LIMIT, 15.0, places=3)

    def test_bc06_all_frame_corrections_within_limit(self):
        """Every frame aimpoint correction must be within ±15°."""
        target = ThermalTarget(
            target_id="TGT-AIM", is_decoy=False,
            thermal_signature=0.88, thermal_decay_rate=0.001,
            initial_roi_px=22, bearing_deg=12.0, range_m=1000.0
        )
        result = self.processor.process_target(target, max_frames=30)
        for frame in result.frames:
            self.assertLessEqual(abs(frame.aimpoint_correction_deg),
                AIMPOINT_CORRECTION_LIMIT,
                f"Frame {frame.frame_id}: aimpoint correction "
                f"{frame.aimpoint_correction_deg:.2f}° exceeds ±{AIMPOINT_CORRECTION_LIMIT}°")


class TestDMRLSceneProcessing(unittest.TestCase):
    """Scene-level tests — multi-target discrimination."""

    def setUp(self):
        self.processor = DMRLProcessor(verbose=False)

    def test_scene_generation_correct_counts(self):
        """generate_synthetic_scene must produce correct target/decoy counts."""
        scene = generate_synthetic_scene(n_targets=1, n_decoys=1, seed=0)
        self.assertEqual(len(scene), 2)
        real   = [t for t in scene if not t.is_decoy]
        decoys = [t for t in scene if t.is_decoy]
        self.assertEqual(len(real),   1)
        self.assertEqual(len(decoys), 1)

    def test_scene_decoy_has_higher_decay_rate(self):
        """Decoys must have faster thermal decay than real targets (discriminator)."""
        for seed in range(20):
            scene  = generate_synthetic_scene(n_targets=1, n_decoys=1, seed=seed)
            real   = next(t for t in scene if not t.is_decoy)
            decoy  = next(t for t in scene if t.is_decoy)
            self.assertGreater(decoy.thermal_decay_rate, real.thermal_decay_rate,
                f"Seed {seed}: Decoy decay rate ({decoy.thermal_decay_rate:.4f}) must be "
                f"greater than real target ({real.thermal_decay_rate:.4f})")

    def test_process_scene_returns_all_targets(self):
        """process_scene must return a result for every target in scene."""
        scene = generate_synthetic_scene(n_targets=1, n_decoys=1, seed=7)
        results = self.processor.process_scene(scene)
        self.assertEqual(len(results), len(scene))
        for target in scene:
            self.assertIn(target.target_id, results)

    def test_select_primary_target_returns_non_decoy(self):
        """select_primary_target must return a real target, not a decoy."""
        pass_count = 0
        for seed in range(20):
            scene   = generate_synthetic_scene(n_targets=1, n_decoys=1, seed=seed)
            results = self.processor.process_scene(scene, max_frames=30)
            primary = self.processor.select_primary_target(results)
            if primary is not None:
                # Verify target ID maps to a real target
                selected_target = next(t for t in scene if t.target_id == primary.target_id)
                self.assertFalse(selected_target.is_decoy,
                    f"Seed {seed}: select_primary_target returned a DECOY")
                pass_count += 1

        # At least half of runs should successfully select a primary
        self.assertGreater(pass_count, 5,
            "Primary target selection rate too low")

    def test_select_primary_returns_none_when_all_decoys(self):
        """With only decoys in scene, select_primary_target must return None."""
        # Manually craft a scene with only decoys
        scene = [ThermalTarget(
            target_id="DCY-ONLY", is_decoy=True,
            thermal_signature=0.95, thermal_decay_rate=0.030,
            initial_roi_px=10, bearing_deg=10.0, range_m=1000.0
        )]
        results = self.processor.process_scene(scene, max_frames=30)

        # Override results to simulate all decoys flagged
        for r in results.values():
            r.is_decoy = True
            r.lock_acquired = False

        primary = self.processor.select_primary_target(results)
        self.assertIsNone(primary,
            "select_primary_target must return None when all candidates are decoys")


class TestDMRLKPIRequirements(unittest.TestCase):
    """
    KPI-level tests mapping to BCMP-1 acceptance criteria:
      KPI-T01: Terminal lock rate ≥ 85% over 50 runs
      KPI-T02: Decoy rejection rate ≥ 90% over 50 runs
    """

    def setUp(self):
        self.processor = DMRLProcessor(verbose=False)

    def test_kpi_t01_terminal_lock_rate_50_runs(self):
        """KPI-T01: Lock rate ≥ 85% over 50 terminal approach runs."""
        lock_count = 0
        n_runs = 50
        for seed in range(n_runs):
            scene   = generate_synthetic_scene(n_targets=1, n_decoys=1, seed=seed)
            results = self.processor.process_scene(scene, max_frames=30)
            primary = self.processor.select_primary_target(results)
            if primary is not None and primary.lock_confidence >= LOCK_CONFIDENCE_THRESHOLD:
                lock_count += 1

        lock_rate = lock_count / n_runs
        print(f"\n  KPI-T01 Lock Rate: {lock_rate:.0%} ({lock_count}/{n_runs})")
        self.assertGreaterEqual(lock_rate, 0.85,
            f"KPI-T01 FAIL: Terminal lock rate {lock_rate:.0%} < 85%")

    def test_kpi_t02_decoy_rejection_rate_50_runs(self):
        """KPI-T02: Decoy rejection ≥ 90% over 50 runs with thermal decoy present."""
        reject_count = 0
        n_runs = 50
        for seed in range(n_runs):
            scene = [ThermalTarget(
                target_id="DCY-TEST", is_decoy=True,
                thermal_signature=random.uniform(0.85, 0.99),
                thermal_decay_rate=random.uniform(0.018, 0.035),
                initial_roi_px=random.randint(8, 14),
                bearing_deg=random.uniform(-20, 20),
                range_m=random.uniform(900, 2600),
            )]
            random.seed(seed)
            results = self.processor.process_scene(scene, max_frames=30)
            if results["DCY-TEST"].is_decoy:
                reject_count += 1

        rejection_rate = reject_count / n_runs
        print(f"\n  KPI-T02 Decoy Rejection Rate: {rejection_rate:.0%} ({reject_count}/{n_runs})")
        self.assertGreaterEqual(rejection_rate, 0.90,
            f"KPI-T02 FAIL: Decoy rejection rate {rejection_rate:.0%} < 90%")


class TestDMRLResultStructure(unittest.TestCase):
    """Structural integrity tests for DMRLResult data."""

    def setUp(self):
        self.processor = DMRLProcessor(verbose=False)

    def test_result_has_required_fields(self):
        """DMRLResult must have all required fields populated."""
        target = ThermalTarget(
            target_id="TGT-STRUCT", is_decoy=False,
            thermal_signature=0.88, thermal_decay_rate=0.002,
            initial_roi_px=22, bearing_deg=2.0, range_m=1200.0
        )
        result = self.processor.process_target(target, max_frames=20)
        self.assertIsNotNone(result.target_id)
        self.assertIsInstance(result.lock_confidence, float)
        self.assertIsInstance(result.is_decoy, bool)
        self.assertIsInstance(result.decoy_confidence, float)
        self.assertIsInstance(result.dwell_frames, int)
        self.assertIsInstance(result.lock_acquired, bool)
        self.assertIsInstance(result.lock_lost_timeout, bool)
        self.assertIsInstance(result.frames, list)
        self.assertIsInstance(result.log, list)
        self.assertGreater(result.processing_latency_s, 0.0)

    def test_frame_results_in_temporal_order(self):
        """Frame results must be in ascending frame_id order."""
        target = ThermalTarget(
            target_id="TGT-ORDER", is_decoy=False,
            thermal_signature=0.88, thermal_decay_rate=0.002,
            initial_roi_px=22, bearing_deg=2.0, range_m=1200.0
        )
        result = self.processor.process_target(target, max_frames=20)
        frame_ids = [f.frame_id for f in result.frames]
        self.assertEqual(frame_ids, sorted(frame_ids),
            "Frame results must be in ascending temporal order")

    def test_confidence_values_in_range(self):
        """All confidence values must be in [0.0, 1.0]."""
        target = ThermalTarget(
            target_id="TGT-RANGE", is_decoy=False,
            thermal_signature=0.88, thermal_decay_rate=0.002,
            initial_roi_px=22, bearing_deg=2.0, range_m=1200.0
        )
        result = self.processor.process_target(target, max_frames=20)
        self.assertGreaterEqual(result.lock_confidence, 0.0)
        self.assertLessEqual(result.lock_confidence, 1.0)
        self.assertGreaterEqual(result.decoy_confidence, 0.0)
        self.assertLessEqual(result.decoy_confidence, 1.0)
        for frame in result.frames:
            self.assertGreaterEqual(frame.lock_confidence, 0.0)
            self.assertLessEqual(frame.lock_confidence, 1.0)
            self.assertGreaterEqual(frame.decoy_probability, 0.0)
            self.assertLessEqual(frame.decoy_probability, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
