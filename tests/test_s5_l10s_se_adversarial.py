"""
tests/test_s5_l10s_se_adversarial.py
MicroMind Sprint B — L10s-SE Adversarial Integration Tests
ADV-01 through ADV-06

Closes: OI-26 — L10s-SE adversarial EO condition tests absent
QA standing rule #2 compliance: all scenarios exercise adversarial EO conditions
through a realistic pipeline, not clean hand-constructed inputs.

All 6 scenarios exercise the full integration pipeline:
  generate_synthetic_scene() → DMRLProcessor → inputs_from_dmrl() → L10sSafetyEnvelope

Gate 3 (civilian detection, confidence ≥ 0.70 → ABORT / CIVILIAN_DETECTED) was
unreachable through the integration path before this suite. inputs_from_dmrl()
defaulted civilian_confidence to 0.0 in every prior test. These scenarios are
the first to drive that gate via DMRL pipeline output.

Spec: docs/qa/Sessions/2026-04-04_sprintB_scenarios.md
Step: Sprint B Step 3
"""

from __future__ import annotations

import sys
import os
import random
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.dmrl.dmrl_stub import (
    DMRLProcessor,
    generate_synthetic_scene,
    ThermalTarget,
    LOCK_CONFIDENCE_THRESHOLD,
)
from core.l10s_se.l10s_se import (
    L10sSafetyEnvelope,
    L10sInputs,
    L10sDecision,
    AbortReason,
    inputs_from_dmrl,
)


# ─── ADV-01 ───────────────────────────────────────────────────────────────────

class TestADV01CivilianDetectedFullPipeline(unittest.TestCase):
    """
    ADV-01: Civilian present (≥ 0.70), genuine target acquired, full DMRL pipeline feeds L10s-SE.
    SRS: TERM-01 (FR-105 §1.12) — civilian_confidence ≥ 0.70 in any frame → ABORT / CIVILIAN_DETECTED.
    Closes: OI-26 — first test to reach Gate 3 via generate_synthetic_scene() → DMRL → L10s-SE.
    """

    def test_adv01_civilian_detected_via_full_dmrl_pipeline(self):
        """
        ADV-01 — TERM-01: civilian_confidence=0.82 injected after genuine target acquired via full pipeline.
        Pipeline: generate_synthetic_scene(n_targets=1, n_decoys=0, seed=42)
                  → DMRLProcessor.process_target() → inputs_from_dmrl(civilian_confidence=0.82)
                  → L10sSafetyEnvelope.evaluate()
        Expected: ABORT / CIVILIAN_DETECTED (Gate 3).
        """
        scene = generate_synthetic_scene(n_targets=1, n_decoys=0, seed=42)
        target = next(t for t in scene if not t.is_decoy)

        processor = DMRLProcessor(verbose=False)
        dmrl_result = processor.process_target(target, max_frames=30)

        inp = inputs_from_dmrl(
            dmrl_result,
            civilian_confidence=0.82,
            corridor_violation=False,
            pre_terminal_zpi_complete=True,
        )
        engine = L10sSafetyEnvelope(verbose=False)
        result = engine.evaluate(inp)

        self.assertEqual(
            result.decision, L10sDecision.ABORT,
            f"ADV-01: civilian_confidence=0.82 ≥ 0.70 must produce ABORT "
            f"(got {result.decision.value}, abort_reason={result.abort_reason.value})"
        )
        self.assertEqual(
            result.abort_reason, AbortReason.CIVILIAN_DETECTED,
            f"ADV-01: abort_reason must be CIVILIAN_DETECTED (Gate 3) "
            f"(got {result.abort_reason.value})"
        )


# ─── ADV-02 ───────────────────────────────────────────────────────────────────

class TestADV02CivilianBelowThresholdMustContinue(unittest.TestCase):
    """
    ADV-02: Civilian below threshold (0.65), genuine target acquired, full DMRL pipeline.
    SRS: TERM-01 (FR-105 §1.12) — civilian_confidence < 0.70 must not produce ABORT.
    Closes: OI-26 — negative control for ADV-01 through the integration path.
    """

    def test_adv02_civilian_below_threshold_must_continue(self):
        """
        ADV-02 — TERM-01: Elevated but sub-threshold civilian (0.65) via full pipeline → CONTINUE.
        Same scene as ADV-01 (seed=42, no decoys) for direct comparison.
        Pipeline: generate_synthetic_scene(n_targets=1, n_decoys=0, seed=42)
                  → DMRLProcessor.process_target() → inputs_from_dmrl(civilian_confidence=0.65)
                  → L10sSafetyEnvelope.evaluate()
        Expected: CONTINUE / NONE (all gates clear).
        """
        scene = generate_synthetic_scene(n_targets=1, n_decoys=0, seed=42)
        target = next(t for t in scene if not t.is_decoy)

        processor = DMRLProcessor(verbose=False)
        dmrl_result = processor.process_target(target, max_frames=30)

        inp = inputs_from_dmrl(
            dmrl_result,
            civilian_confidence=0.65,
            corridor_violation=False,
            pre_terminal_zpi_complete=True,
        )
        engine = L10sSafetyEnvelope(verbose=False)
        result = engine.evaluate(inp)

        self.assertEqual(
            result.decision, L10sDecision.CONTINUE,
            f"ADV-02: civilian_confidence=0.65 < 0.70 must produce CONTINUE through full pipeline "
            f"(got {result.decision.value}, abort_reason={result.abort_reason.value})"
        )
        self.assertEqual(
            result.abort_reason, AbortReason.NONE,
            f"ADV-02: No abort reason must be set on CONTINUE "
            f"(got {result.abort_reason.value})"
        )


# ─── ADV-03 ───────────────────────────────────────────────────────────────────

class TestADV03CivilianExactBoundaryFullPipeline(unittest.TestCase):
    """
    ADV-03: Civilian confidence at exactly 0.70 boundary via full DMRL pipeline.
    SRS: TERM-01 (FR-105 §1.12) — threshold is inclusive: ≥ 0.70 → ABORT.
    Closes: OI-26 — boundary condition for TERM-01 through integration path (not hand-constructed).
    """

    def test_adv03_civilian_at_exact_boundary_aborts(self):
        """
        ADV-03 — TERM-01: civilian_confidence exactly 0.70 via full pipeline → ABORT / CIVILIAN_DETECTED.
        Boundary error (treating ≥ as >) would produce CONTINUE and silently violate ROE.
        Pipeline: generate_synthetic_scene(n_targets=1, n_decoys=0, seed=42)
                  → DMRLProcessor.process_target() → inputs_from_dmrl(civilian_confidence=0.70)
                  → L10sSafetyEnvelope.evaluate()
        Expected: ABORT / CIVILIAN_DETECTED (Gate 3).
        """
        scene = generate_synthetic_scene(n_targets=1, n_decoys=0, seed=42)
        target = next(t for t in scene if not t.is_decoy)

        processor = DMRLProcessor(verbose=False)
        dmrl_result = processor.process_target(target, max_frames=30)

        inp = inputs_from_dmrl(
            dmrl_result,
            civilian_confidence=0.70,
            corridor_violation=False,
            pre_terminal_zpi_complete=True,
        )
        engine = L10sSafetyEnvelope(verbose=False)
        result = engine.evaluate(inp)

        self.assertEqual(
            result.decision, L10sDecision.ABORT,
            f"ADV-03: civilian_confidence=0.70 (exactly at threshold) must produce ABORT "
            f"— inclusive ≥ condition (got {result.decision.value})"
        )
        self.assertEqual(
            result.abort_reason, AbortReason.CIVILIAN_DETECTED,
            f"ADV-03: abort_reason must be CIVILIAN_DETECTED at exactly 0.70 "
            f"(got {result.abort_reason.value})"
        )


# ─── ADV-04 ───────────────────────────────────────────────────────────────────

class TestADV04DecoyAndCivilianGatePriority(unittest.TestCase):
    """
    ADV-04: Decoy present AND civilian present (≥ 0.70) — Gate 2 must fire before Gate 3.
    SRS: TERM-01 + TERM-03 (FR-105 §1.12 + FR-103 §1.9.3).
    Closes: OI-26 — concurrent adversarial conditions (decoy + civilian) through full pipeline.

    IMPLEMENTATION NOTE — scene-level architecture:
    DMRL decoy detection and lock acquisition are mutually exclusive by design: the decoy
    confidence formula applies a uniform(0.72, 0.84) cap, keeping lock_confidence < 0.85
    whenever is_decoy fires. Processing only the decoy target therefore yields
    lock_acquired=False → L10s-SE aborts at Gate 1b (NO_LOCK), never reaching Gate 2.

    The correct architecture (matching real system behaviour) processes all scene targets:
    - Real target  → provides lock_acquired=True, lock_confidence
    - Decoy target → provides is_decoy=True, decoy_confidence
    L10sInputs is built by combining both results, representing a scene where a lock has
    been established on the primary target AND a decoy signal has been detected in the scene.
    This is the only valid integration path that can reach Gate 2 via the DMRL pipeline.
    """

    def test_adv04_decoy_fires_before_civilian_dual_fault(self):
        """
        ADV-04 — TERM-01 + TERM-03: Decoy detected in scene AND civilian ≥ 0.70 simultaneously.
        Gate 2 (DECOY_DETECTED) must fire — Gate 3 (CIVILIAN_DETECTED) must never be evaluated.
        Pipeline: generate_synthetic_scene(n_targets=1, n_decoys=1, seed=7)
                  → process_scene() on both targets
                  → L10sInputs built from real-target lock + decoy-target is_decoy flag
                  → inputs civilian_confidence=0.82 → L10sSafetyEnvelope.evaluate()
        Expected: ABORT / DECOY_DETECTED (Gate 2, not Gate 3).
        """
        scene = generate_synthetic_scene(n_targets=1, n_decoys=1, seed=7)
        real_target  = next(t for t in scene if not t.is_decoy)
        decoy_target = next(t for t in scene if t.is_decoy)

        processor = DMRLProcessor(verbose=False)
        real_result  = processor.process_target(real_target,  max_frames=30)
        decoy_result = processor.process_target(decoy_target, max_frames=30)

        # Prerequisite A: real target must have acquired lock (Gates 1b/1c can pass)
        self.assertTrue(
            real_result.lock_acquired,
            f"ADV-04 prerequisite: seed=7 real target did not acquire lock "
            f"(lock_acquired={real_result.lock_acquired}, "
            f"lock_confidence={real_result.lock_confidence:.4f}). "
            f"Try a different seed."
        )
        # Prerequisite B: decoy target must be flagged by DMRL (Gate 2 stimulus present)
        self.assertTrue(
            decoy_result.is_decoy,
            f"ADV-04 prerequisite: seed=7 decoy not flagged by DMRL pipeline "
            f"(is_decoy={decoy_result.is_decoy}, "
            f"decoy_confidence={decoy_result.decoy_confidence:.4f}). "
            f"Try a different seed."
        )

        # Combine scene results: real-target lock state + decoy detection from scene
        inp = L10sInputs(
            lock_acquired=real_result.lock_acquired,
            lock_confidence=real_result.lock_confidence,
            is_decoy=decoy_result.is_decoy,
            decoy_confidence=decoy_result.decoy_confidence,
            lock_lost_timeout=real_result.lock_lost_timeout,
            civilian_confidence=0.82,
            corridor_violation=False,
            pre_terminal_zpi_complete=True,
            activation_timestamp=time.monotonic(),
        )
        engine = L10sSafetyEnvelope(verbose=False)
        result = engine.evaluate(inp)

        self.assertEqual(
            result.decision, L10sDecision.ABORT,
            "ADV-04: Must ABORT when decoy detected in scene and civilian ≥ 0.70"
        )
        # Primary assertion: Gate 2 must fire, not Gate 3
        self.assertEqual(
            result.abort_reason, AbortReason.DECOY_DETECTED,
            f"ADV-04: abort_reason must be DECOY_DETECTED (Gate 2 fires first). "
            f"Gate 3 must not be reached when Gate 2 fires. Got: {result.abort_reason.value}"
        )
        self.assertNotEqual(
            result.abort_reason, AbortReason.CIVILIAN_DETECTED,
            "ADV-04: abort_reason must NOT be CIVILIAN_DETECTED — "
            "civilian check (Gate 3) is structurally downstream of decoy check (Gate 2)"
        )


# ─── ADV-05 ───────────────────────────────────────────────────────────────────

class TestADV05DegradedLockAndCivilianGatePriority(unittest.TestCase):
    """
    ADV-05: Civilian present (≥ 0.70) AND lock degraded below threshold — Gate 1c fires before Gate 3.
    SRS: TERM-01 + TERM-02 (FR-105 §1.12 + FR-103 §1.9.3).
    Closes: OI-26 — concurrent low-lock + civilian condition, previously untested in any form.
    """

    def test_adv05_degraded_lock_fires_before_civilian(self):
        """
        ADV-05 — TERM-01 + TERM-02: thermal_signature=0.62 (simulated haze) AND civilian=0.82.
        Gate 1c (NO_LOCK) must fire — Gate 3 (CIVILIAN_DETECTED) must never be evaluated.
        Pipeline: ThermalTarget(sig=0.62) → DMRLProcessor.process_target()
                  → inputs_from_dmrl(civilian_confidence=0.82) → L10sSafetyEnvelope.evaluate()
        Expected: ABORT / NO_LOCK (Gate 1c, not Gate 3).
        """
        degraded_target = ThermalTarget(
            target_id="ADV05-HAZE",
            is_decoy=False,
            thermal_signature=0.62,
            thermal_decay_rate=0.001,
            initial_roi_px=24,
            bearing_deg=1.0,
            range_m=1200.0,
        )

        processor = DMRLProcessor(verbose=False)
        dmrl_result = processor.process_target(degraded_target, max_frames=30)

        # Verify setup: thermal_signature=0.62 must produce lock_acquired=False
        self.assertFalse(
            dmrl_result.lock_acquired,
            f"ADV-05 setup: thermal_signature=0.62 must not acquire lock "
            f"(got lock_confidence={dmrl_result.lock_confidence:.4f})"
        )
        self.assertLess(
            dmrl_result.lock_confidence, LOCK_CONFIDENCE_THRESHOLD,
            f"ADV-05 setup: lock_confidence {dmrl_result.lock_confidence:.4f} "
            f"must be < {LOCK_CONFIDENCE_THRESHOLD}"
        )

        inp = inputs_from_dmrl(
            dmrl_result,
            civilian_confidence=0.82,
            corridor_violation=False,
            pre_terminal_zpi_complete=True,
        )
        engine = L10sSafetyEnvelope(verbose=False)
        result = engine.evaluate(inp)

        self.assertEqual(
            result.decision, L10sDecision.ABORT,
            "ADV-05: Must ABORT when lock is degraded and civilian ≥ 0.70 simultaneously"
        )
        # Primary assertion: Gate 1c must fire, not Gate 3
        self.assertEqual(
            result.abort_reason, AbortReason.NO_LOCK,
            f"ADV-05: abort_reason must be NO_LOCK (Gate 1c fires first). "
            f"Gate 3 must not be reached when Gate 1c fires. Got: {result.abort_reason.value}"
        )
        self.assertNotEqual(
            result.abort_reason, AbortReason.CIVILIAN_DETECTED,
            "ADV-05: abort_reason must NOT be CIVILIAN_DETECTED — "
            "lock gate (Gate 1c) is structurally upstream of civilian gate (Gate 3)"
        )


# ─── ADV-06 ───────────────────────────────────────────────────────────────────

class TestADV06BoundaryLockGate3Fires(unittest.TestCase):
    """
    ADV-06: Civilian present (≥ 0.70), lock confidence at boundary (≥ 0.85) — Gate 3 must fire.
    SRS: TERM-01 + TERM-02 (FR-105 §1.12 + FR-103 §1.9.3).
    Closes: OI-26 — critical integration path: only scenario where Gates 0–2 all clear
    and Gate 3 is the sole abort trigger via the DMRL pipeline.

    DEVIATION FROM SPEC:
    Spec stated thermal_signature=0.75. The lock_confidence formula is:
      confidence = (sig + gauss(0, 0.025)) * temporal_factor * roi_factor
    With sig ≈ 0.75, confidence ≈ 0.75 — structurally below the 0.85 gate.
    The prerequisite guard (lock_acquired=True) would never be satisfied.
    thermal_signature raised to 0.88 to produce confidence ≈ 0.87–0.89,
    which satisfies the spec intent: boundary lock passes Gate 1c, Gate 3 evaluates,
    ABORT / CIVILIAN_DETECTED.
    """

    def test_adv06_boundary_lock_gate3_fires(self):
        """
        ADV-06 — TERM-01 + TERM-02: Lock acquired (≥ 0.85) AND civilian_confidence=0.75.
        Gate 1c must pass, Gate 3 must fire (CIVILIAN_DETECTED).
        Uses 10-seed Monte Carlo to find a run where DMRL acquires lock,
        per spec prerequisite guard requirement.
        Pipeline: ThermalTarget(sig=0.88) → DMRLProcessor (seed loop)
                  → inputs_from_dmrl(civilian_confidence=0.75) → L10sSafetyEnvelope.evaluate()
        Expected: ABORT / CIVILIAN_DETECTED (Gate 3).
        """
        boundary_target = ThermalTarget(
            target_id="ADV06-BOUNDARY",
            is_decoy=False,
            thermal_signature=0.88,      # raised from spec's 0.75 — see class docstring
            thermal_decay_rate=0.001,
            initial_roi_px=22,
            bearing_deg=1.0,
            range_m=1200.0,
        )

        processor = DMRLProcessor(verbose=False)

        # 10-seed Monte Carlo: find a run where DMRL acquires lock (≥ 0.85).
        # DMRL processing is stochastic (gauss noise); seed loop finds a valid run.
        # Per spec: "run 10 seeds, assert Gate 3 fires in at least one run where lock_acquired=True"
        lock_acquired_result = None
        for seed in range(10):
            random.seed(seed)
            candidate = processor.process_target(boundary_target, max_frames=30)
            if candidate.lock_acquired and candidate.lock_confidence >= LOCK_CONFIDENCE_THRESHOLD:
                lock_acquired_result = candidate
                break

        # ── Mandatory prerequisite guard (must appear before L10s-SE call) ──────
        assert lock_acquired_result is not None and lock_acquired_result.lock_acquired is True, \
            ("ADV-06 prerequisite failed: DMRL did not acquire lock in any of seeds 0–9. "
             "Test is invalid — use a different seed or thermal_signature value.")

        inp = inputs_from_dmrl(
            lock_acquired_result,
            civilian_confidence=0.75,
            corridor_violation=False,
            pre_terminal_zpi_complete=True,
        )
        engine = L10sSafetyEnvelope(verbose=False)
        result = engine.evaluate(inp)

        self.assertEqual(
            result.decision, L10sDecision.ABORT,
            f"ADV-06: Must ABORT when lock passes Gate 1c (conf={lock_acquired_result.lock_confidence:.4f}) "
            f"and civilian_confidence=0.75 ≥ 0.70"
        )
        self.assertEqual(
            result.abort_reason, AbortReason.CIVILIAN_DETECTED,
            f"ADV-06: abort_reason must be CIVILIAN_DETECTED (Gate 3 is sole abort trigger). "
            f"Gate 1c must have passed (lock_confidence={lock_acquired_result.lock_confidence:.4f} "
            f"≥ {LOCK_CONFIDENCE_THRESHOLD}). Got: {result.abort_reason.value}"
        )


# ─── Known Gap — Deferred ─────────────────────────────────────────────────────
#
# ADV-07 (DEFERRED — future sprint):
#   Corridor violation integration path: civilian_confidence < 0.70 AND corridor_violation=True
#   AND target locked via DMRL. Gate 4 (CORRIDOR_VIOLATION) must fire after Gate 3 passes.
#   Also: civilian_confidence ≥ 0.70 AND corridor_violation=True → Gate 3 fires (not Gate 4),
#   confirming the Gate 3/Gate 4 priority boundary.
#   Deferred because the corridor_violation flag is driven by mission envelope geometry,
#   not DMRL state. A separate stimulus design (mock trajectory predictor) is needed to
#   exercise Gate 4 through an integration path. Not in scope for OI-26 closure.


if __name__ == "__main__":
    unittest.main(verbosity=2)
