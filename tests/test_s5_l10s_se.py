"""
tests/test_s5_l10s_se.py
MicroMind Sprint S5 — L10s-SE Unit Tests

Tests all decision tree gates and timing compliance (FR-105):
  - Decision within ≤ 2 s of activation
  - All gate conditions: lock, decoy, civilian, corridor
  - ZPI pre-terminal burst mandatory
  - Audit log completeness
  - KPI-T03: 100% L10s-SE compliance

Run: python tests/test_s5_l10s_se.py
"""

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.l10s_se.l10s_se import (
    L10sSafetyEnvelope, L10sInputs, L10sDecision, AbortReason,
    inputs_from_dmrl, L10sLogEntry,
    DECISION_TIMEOUT_S, L10S_WINDOW_S,
    LOCK_CONFIDENCE_THRESHOLD, CIVILIAN_DETECT_THRESHOLD, REACQUISITION_TIMEOUT_S,
)


def _valid_inputs(**overrides) -> L10sInputs:
    """Return a fully valid L10sInputs that should produce CONTINUE."""
    base = dict(
        lock_acquired=True,
        lock_confidence=0.92,
        is_decoy=False,
        decoy_confidence=0.10,
        lock_lost_timeout=False,
        civilian_confidence=0.05,
        corridor_violation=False,
        pre_terminal_zpi_complete=True,
        activation_timestamp=time.monotonic(),
    )
    base.update(overrides)
    return L10sInputs(**base)


class TestL10sBoundaryConstants(unittest.TestCase):
    """Verify boundary constants match Part Two V7 §1.12."""

    def test_decision_timeout_is_2s(self):
        self.assertAlmostEqual(DECISION_TIMEOUT_S, 2.0, places=3,
            msg="Decision timeout must be 2.0 s per §1.12")

    def test_window_is_10s(self):
        self.assertAlmostEqual(L10S_WINDOW_S, 10.0, places=3,
            msg="L10s-SE window must be 10.0 s per §1.12")

    def test_lock_threshold_matches_dmrl(self):
        """L10s-SE lock threshold must match DMRL lock threshold (0.85)."""
        self.assertAlmostEqual(LOCK_CONFIDENCE_THRESHOLD, 0.85, places=3)

    def test_civilian_threshold_is_0_7(self):
        self.assertAlmostEqual(CIVILIAN_DETECT_THRESHOLD, 0.70, places=3,
            msg="Civilian detection threshold must be 0.70 per §1.12")

    def test_reacquisition_timeout_is_1_5s(self):
        self.assertAlmostEqual(REACQUISITION_TIMEOUT_S, 1.5, places=3)


class TestL10sGate0ZPI(unittest.TestCase):
    """Gate 0: Pre-terminal ZPI burst (DD-02 mandatory)."""

    def setUp(self):
        self.engine = L10sSafetyEnvelope(verbose=False)

    def test_gate0_abort_if_zpi_not_complete(self):
        """ZPI burst not confirmed → ABORT regardless of all other conditions."""
        inp = _valid_inputs(pre_terminal_zpi_complete=False)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.ABORT,
            "ZPI not confirmed must produce ABORT")

    def test_gate0_pass_if_zpi_complete(self):
        """ZPI confirmed → Gate 0 passes; decision depends on other gates."""
        inp = _valid_inputs(pre_terminal_zpi_complete=True)
        result = self.engine.evaluate(inp)
        # With all other gates clear, should CONTINUE
        self.assertEqual(result.decision, L10sDecision.CONTINUE)


class TestL10sGate1LockAcquired(unittest.TestCase):
    """Gate 1: EO lock checks."""

    def setUp(self):
        self.engine = L10sSafetyEnvelope(verbose=False)

    def test_gate1_abort_if_no_lock(self):
        inp = _valid_inputs(lock_acquired=False, lock_confidence=0.70)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.ABORT)
        self.assertEqual(result.abort_reason, AbortReason.NO_LOCK)

    def test_gate1_abort_if_lock_confidence_below_threshold(self):
        inp = _valid_inputs(lock_acquired=True, lock_confidence=0.84)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.ABORT)
        self.assertEqual(result.abort_reason, AbortReason.NO_LOCK)

    def test_gate1_abort_exactly_at_boundary_minus_epsilon(self):
        """lock_confidence = 0.8499 (just below 0.85) → ABORT."""
        inp = _valid_inputs(lock_acquired=True, lock_confidence=0.8499)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.ABORT)

    def test_gate1_continue_at_threshold(self):
        """lock_confidence = 0.85 (exactly at threshold) → CONTINUE."""
        inp = _valid_inputs(lock_acquired=True, lock_confidence=0.850)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.CONTINUE)

    def test_gate1_abort_if_lock_lost_timeout(self):
        inp = _valid_inputs(lock_lost_timeout=True, lock_acquired=True, lock_confidence=0.95)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.ABORT)
        self.assertEqual(result.abort_reason, AbortReason.LOCK_LOST_TIMEOUT)

    def test_gate1_lock_lost_timeout_takes_precedence_over_good_lock(self):
        """lock_lost_timeout=True must abort even if lock_acquired=True and conf=0.99."""
        inp = _valid_inputs(lock_acquired=True, lock_confidence=0.99, lock_lost_timeout=True)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.ABORT)
        self.assertEqual(result.abort_reason, AbortReason.LOCK_LOST_TIMEOUT)


class TestL10sGate2Decoy(unittest.TestCase):
    """Gate 2: Decoy detection."""

    def setUp(self):
        self.engine = L10sSafetyEnvelope(verbose=False)

    def test_gate2_abort_if_decoy_detected(self):
        inp = _valid_inputs(is_decoy=True, decoy_confidence=0.88)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.ABORT)
        self.assertEqual(result.abort_reason, AbortReason.DECOY_DETECTED)

    def test_gate2_continue_if_no_decoy(self):
        inp = _valid_inputs(is_decoy=False, decoy_confidence=0.10)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.CONTINUE)

    def test_gate2_abort_even_with_high_lock_confidence(self):
        """Decoy flag overrides high lock confidence — abort required."""
        inp = _valid_inputs(is_decoy=True, decoy_confidence=0.95, lock_confidence=0.97)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.ABORT)
        self.assertEqual(result.abort_reason, AbortReason.DECOY_DETECTED)


class TestL10sGate3Civilian(unittest.TestCase):
    """Gate 3: Civilian presence detection (confidence ≥ 0.70 → abort)."""

    def setUp(self):
        self.engine = L10sSafetyEnvelope(verbose=False)

    def test_gate3_abort_if_civilian_confidence_at_threshold(self):
        inp = _valid_inputs(civilian_confidence=0.70)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.ABORT)
        self.assertEqual(result.abort_reason, AbortReason.CIVILIAN_DETECTED)

    def test_gate3_abort_if_civilian_confidence_above_threshold(self):
        inp = _valid_inputs(civilian_confidence=0.85)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.ABORT)
        self.assertEqual(result.abort_reason, AbortReason.CIVILIAN_DETECTED)

    def test_gate3_continue_below_civilian_threshold(self):
        inp = _valid_inputs(civilian_confidence=0.69)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.CONTINUE)

    def test_gate3_abort_at_0_699_boundary(self):
        """0.699 (just below 0.70) must CONTINUE."""
        inp = _valid_inputs(civilian_confidence=0.699)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.CONTINUE)

    def test_gate3_abort_at_exactly_0_700(self):
        """Exactly 0.700 must ABORT."""
        inp = _valid_inputs(civilian_confidence=0.700)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.ABORT)


class TestL10sGate4Corridor(unittest.TestCase):
    """Gate 4: Corridor hard enforcement."""

    def setUp(self):
        self.engine = L10sSafetyEnvelope(verbose=False)

    def test_gate4_abort_if_corridor_violation(self):
        inp = _valid_inputs(corridor_violation=True)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.ABORT)
        self.assertEqual(result.abort_reason, AbortReason.CORRIDOR_VIOLATION)

    def test_gate4_continue_if_corridor_clear(self):
        inp = _valid_inputs(corridor_violation=False)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.CONTINUE)

    def test_gate4_corridor_abort_overrides_high_confidence(self):
        """Corridor violation aborts even with perfect DMRL output."""
        inp = _valid_inputs(corridor_violation=True, lock_confidence=0.99,
                            is_decoy=False, civilian_confidence=0.01)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.abort_reason, AbortReason.CORRIDOR_VIOLATION)


class TestL10sTimingCompliance(unittest.TestCase):
    """
    Timing compliance tests for KPI-T03.
    Decision latency must be ≤ 2 s from activation.
    """

    def setUp(self):
        self.engine = L10sSafetyEnvelope(verbose=False)

    def test_decision_latency_within_2s_continue(self):
        """CONTINUE path decision latency must be ≤ 2 s."""
        inp = _valid_inputs()
        result = self.engine.evaluate(inp)
        self.assertLessEqual(result.decision_latency_s, DECISION_TIMEOUT_S,
            f"CONTINUE decision latency {result.decision_latency_s*1000:.2f}ms exceeds 2000ms")
        self.assertTrue(result.decision_compliance)

    def test_decision_latency_within_2s_abort_no_lock(self):
        """ABORT (no lock) decision latency must be ≤ 2 s."""
        inp = _valid_inputs(lock_acquired=False)
        result = self.engine.evaluate(inp)
        self.assertLessEqual(result.decision_latency_s, DECISION_TIMEOUT_S)
        self.assertTrue(result.decision_compliance)

    def test_decision_latency_within_2s_abort_decoy(self):
        """ABORT (decoy) decision latency must be ≤ 2 s."""
        inp = _valid_inputs(is_decoy=True, decoy_confidence=0.92)
        result = self.engine.evaluate(inp)
        self.assertLessEqual(result.decision_latency_s, DECISION_TIMEOUT_S)
        self.assertTrue(result.decision_compliance)

    def test_l10s_compliant_flag_true_when_timing_met(self):
        """l10s_compliant must be True when all timing requirements are met."""
        inp = _valid_inputs()
        result = self.engine.evaluate(inp)
        self.assertTrue(result.l10s_compliant,
            "l10s_compliant must be True when decision made within 2 s")

    def test_100_consecutive_runs_all_timing_compliant(self):
        """KPI-T03: 100% compliance over 100 consecutive evaluations."""
        compliant_count = 0
        n_runs = 100
        for _ in range(n_runs):
            inp = _valid_inputs()
            result = self.engine.evaluate(inp)
            if result.l10s_compliant:
                compliant_count += 1

        compliance_rate = compliant_count / n_runs
        print(f"\n  KPI-T03 compliance rate: {compliance_rate:.0%} ({compliant_count}/{n_runs})")
        self.assertEqual(compliance_rate, 1.0,
            f"KPI-T03 FAIL: L10s-SE compliance {compliance_rate:.0%} < 100%")


class TestL10sAuditLog(unittest.TestCase):
    """Secure audit log integrity tests (§1.12: all events logged)."""

    def setUp(self):
        self.engine = L10sSafetyEnvelope(verbose=False)

    def test_secure_log_not_empty(self):
        """Secure log must contain at least one entry per evaluation."""
        inp = _valid_inputs()
        result = self.engine.evaluate(inp)
        self.assertGreater(len(result.secure_log), 0)

    def test_secure_log_has_timestamps(self):
        """All log entries must have a timestamp."""
        inp = _valid_inputs()
        result = self.engine.evaluate(inp)
        for entry in result.secure_log:
            self.assertIsInstance(entry, L10sLogEntry)
            self.assertIsNotNone(entry.timestamp)
            self.assertGreater(entry.timestamp, 0.0)

    def test_secure_log_has_event_strings(self):
        """All log entries must have non-empty event description."""
        inp = _valid_inputs()
        result = self.engine.evaluate(inp)
        for entry in result.secure_log:
            self.assertIsInstance(entry.event, str)
            self.assertGreater(len(entry.event), 0)

    def test_continue_log_contains_activation_record(self):
        """CONTINUE path log must record activation event."""
        inp = _valid_inputs()
        result = self.engine.evaluate(inp)
        events = [e.event for e in result.secure_log]
        activation_recorded = any("ACTIVATED" in e for e in events)
        self.assertTrue(activation_recorded,
            "Secure log must record L10s-SE activation event")

    def test_abort_log_records_reason(self):
        """ABORT path log must record the abort reason."""
        inp = _valid_inputs(is_decoy=True, decoy_confidence=0.90)
        result = self.engine.evaluate(inp)
        events = [e.event for e in result.secure_log]
        abort_recorded = any("DECOY" in e.upper() or "ABORT" in e.upper() for e in events)
        self.assertTrue(abort_recorded,
            "Abort log must record the reason for abort")

    def test_sensor_snapshot_in_activation_log(self):
        """Activation log entry must include sensor state snapshot."""
        inp = _valid_inputs()
        result = self.engine.evaluate(inp)
        # First entry should have the snapshot
        activation_entry = result.secure_log[0]
        self.assertIsNotNone(activation_entry.sensor_state_snapshot)
        snap = activation_entry.sensor_state_snapshot
        self.assertIn("lock_acquired", snap)
        self.assertIn("lock_confidence", snap)
        self.assertIn("is_decoy", snap)
        self.assertIn("civilian_confidence", snap)
        self.assertIn("corridor_violation", snap)

    def test_log_independent_between_evaluations(self):
        """Each evaluation must produce an independent log (no cross-contamination)."""
        inp1 = _valid_inputs()
        r1 = self.engine.evaluate(inp1)

        inp2 = _valid_inputs(is_decoy=True)
        r2 = self.engine.evaluate(inp2)

        # Logs should not share entries
        self.assertIsNot(r1.secure_log, r2.secure_log)


class TestL10sDecisionPriority(unittest.TestCase):
    """
    Verify gate evaluation order.
    ZPI → Lock Lost → No Lock → Decoy → Civilian → Corridor → Continue
    """

    def setUp(self):
        self.engine = L10sSafetyEnvelope(verbose=False)

    def test_zpi_abort_beats_all_other_conditions(self):
        """ZPI absent must produce abort regardless of any other condition."""
        inp = _valid_inputs(
            pre_terminal_zpi_complete=False,
            lock_acquired=True, lock_confidence=0.99,
            is_decoy=False, civilian_confidence=0.0, corridor_violation=False
        )
        result = self.engine.evaluate(inp)
        self.assertEqual(result.decision, L10sDecision.ABORT)

    def test_lock_lost_timeout_beats_decoy_flag(self):
        """lock_lost_timeout fires at Gate 1 before Gate 2 (decoy) is checked."""
        inp = _valid_inputs(lock_lost_timeout=True, is_decoy=True, decoy_confidence=0.95)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.abort_reason, AbortReason.LOCK_LOST_TIMEOUT)

    def test_decoy_abort_beats_civilian_detection(self):
        """Decoy check (Gate 2) fires before civilian check (Gate 3)."""
        inp = _valid_inputs(is_decoy=True, decoy_confidence=0.90, civilian_confidence=0.80)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.abort_reason, AbortReason.DECOY_DETECTED)

    def test_civilian_abort_beats_corridor(self):
        """Civilian check (Gate 3) fires before corridor check (Gate 4)."""
        inp = _valid_inputs(civilian_confidence=0.85, corridor_violation=True)
        result = self.engine.evaluate(inp)
        self.assertEqual(result.abort_reason, AbortReason.CIVILIAN_DETECTED)


class TestInputsFromDMRL(unittest.TestCase):
    """Test the inputs_from_dmrl helper function."""

    def test_helper_builds_valid_inputs(self):
        """inputs_from_dmrl must produce valid L10sInputs from a mock DMRLResult."""

        class MockDMRL:
            lock_acquired    = True
            lock_confidence  = 0.91
            is_decoy         = False
            decoy_confidence = 0.08
            lock_lost_timeout = False

        mock = MockDMRL()
        inp = inputs_from_dmrl(mock, civilian_confidence=0.04, corridor_violation=False)

        self.assertTrue(inp.lock_acquired)
        self.assertAlmostEqual(inp.lock_confidence, 0.91, places=3)
        self.assertFalse(inp.is_decoy)
        self.assertFalse(inp.corridor_violation)
        self.assertTrue(inp.pre_terminal_zpi_complete)

    def test_helper_defaults_are_safe(self):
        """Default parameters must produce safe (conservative) inputs."""

        class MockDMRL:
            lock_acquired     = False
            lock_confidence   = 0.0
            is_decoy          = False
            decoy_confidence  = 0.0
            lock_lost_timeout = True

        mock = MockDMRL()
        inp = inputs_from_dmrl(mock)
        engine = L10sSafetyEnvelope(verbose=False)
        result = engine.evaluate(inp)
        # Should abort (lock not acquired and lock_lost_timeout=True)
        self.assertEqual(result.decision, L10sDecision.ABORT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
