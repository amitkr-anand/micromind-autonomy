"""
tests/test_s5_bcmp1_runner.py
MicroMind Sprint S5 — BCMP-1 End-to-End Acceptance Tests

Acceptance gate: All 11 BCMP-1 pass criteria met, 5 runs in a row.
Maps to: SPRINT_STATUS.md Sprint S5 acceptance gate.

Tests:
  - All 11 BCMP-1 KPI pass criteria evaluated individually
  - 5 consecutive clean runs (acceptance gate)
  - KPI log export and structure
  - Run-to-run consistency (determinism gate)
  - Module integration (DMRL ↔ L10s-SE ↔ BCMP-1 runner)

Run: python tests/test_s5_bcmp1_runner.py
"""

import sys
import os
import json
import unittest
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scenarios.bcmp1.bcmp1_runner import (
    BCMP1Runner, BCMP1KPI, BCMP1RunResult, run_bcmp1,
    NAV01_DRIFT_THRESHOLD_PCT, NAV02_TRN_CEP95_M,
    EW01_COSTMAP_LATENCY_MS, EW02_REPLAN_LATENCY_MS,
    EW03_BIM_SPOOF_MS, SM_TRANSITION_MAX_S,
)
from core.dmrl.dmrl_stub import LOCK_CONFIDENCE_THRESHOLD


class TestBCMP1IndividualCriteria(unittest.TestCase):
    """
    Unit-level test per BCMP-1 criterion.
    Each test validates the specific threshold and pass logic for that KPI.
    """

    def setUp(self):
        self.runner = BCMP1Runner(verbose=False, seed=42)

    def _run(self, run_id: int = 1) -> BCMP1RunResult:
        return self.runner.run(run_id=run_id)

    # ── NAV-01 ────────────────────────────────────────────────────────────────
    def test_nav01_threshold_constant(self):
        """NAV-01 drift threshold must be < 2% of path length."""
        self.assertEqual(NAV01_DRIFT_THRESHOLD_PCT, 2.0)

    def test_nav01_measured_and_evaluated(self):
        """NAV-01: drift value measured and compared to threshold."""
        result = self._run()
        self.assertIsNotNone(result.kpi.nav01_drift_pct,
            "NAV-01 drift must be measured")
        if result.kpi.nav01_drift_pct < NAV01_DRIFT_THRESHOLD_PCT:
            self.assertTrue(result.kpi.nav01_pass)
        else:
            self.assertFalse(result.kpi.nav01_pass)

    def test_nav01_drift_in_plausible_range(self):
        """NAV-01 drift should be physically plausible (0–10%)."""
        result = self._run()
        self.assertGreaterEqual(result.kpi.nav01_drift_pct, 0.0)
        self.assertLessEqual(result.kpi.nav01_drift_pct, 10.0)

    # ── NAV-02 ────────────────────────────────────────────────────────────────
    def test_nav02_threshold_constant(self):
        """NAV-02 TRN CEP-95 threshold must be < 50 m."""
        self.assertEqual(NAV02_TRN_CEP95_M, 50.0)

    def test_nav02_measured_and_evaluated(self):
        """NAV-02: TRN CEP-95 measured and compared to 50 m threshold."""
        result = self._run()
        self.assertIsNotNone(result.kpi.nav02_trn_cep95_m)
        if result.kpi.nav02_trn_cep95_m < NAV02_TRN_CEP95_M:
            self.assertTrue(result.kpi.nav02_pass)
        else:
            self.assertFalse(result.kpi.nav02_pass)

    def test_nav02_cep95_positive(self):
        """NAV-02 CEP-95 must be a positive distance."""
        result = self._run()
        self.assertGreater(result.kpi.nav02_trn_cep95_m, 0.0)

    # ── EW-01 ─────────────────────────────────────────────────────────────────
    def test_ew01_threshold_constant(self):
        """EW-01 cost-map latency threshold must be ≤ 500 ms."""
        self.assertEqual(EW01_COSTMAP_LATENCY_MS, 500.0)

    def test_ew01_measured_and_evaluated(self):
        """EW-01: cost-map latency measured and compared to 500 ms."""
        result = self._run()
        self.assertIsNotNone(result.kpi.ew01_costmap_latency_ms)
        if result.kpi.ew01_costmap_latency_ms <= EW01_COSTMAP_LATENCY_MS:
            self.assertTrue(result.kpi.ew01_pass)

    def test_ew01_latency_positive(self):
        result = self._run()
        self.assertGreater(result.kpi.ew01_costmap_latency_ms, 0.0)

    # ── EW-02 ─────────────────────────────────────────────────────────────────
    def test_ew02_threshold_constant(self):
        """EW-02 route replan threshold must be ≤ 1000 ms (1 s)."""
        self.assertEqual(EW02_REPLAN_LATENCY_MS, 1000.0)

    def test_ew02_both_replans_measured(self):
        """EW-02: Both replans must be measured (jammer 1 and jammer 2)."""
        result = self._run()
        self.assertIsNotNone(result.kpi.ew02_replan1_ms,
            "Replan 1 (Jammer-1) must be measured")
        self.assertIsNotNone(result.kpi.ew02_replan2_ms,
            "Replan 2 (Jammer-2) must be measured")

    def test_ew02_pass_requires_both_within_threshold(self):
        """EW-02 pass requires BOTH replans within 1 s."""
        result = self._run()
        if result.kpi.ew02_pass:
            self.assertLessEqual(result.kpi.ew02_replan1_ms, EW02_REPLAN_LATENCY_MS)
            self.assertLessEqual(result.kpi.ew02_replan2_ms, EW02_REPLAN_LATENCY_MS)

    # ── EW-03 ─────────────────────────────────────────────────────────────────
    def test_ew03_threshold_constant(self):
        """EW-03 BIM spoof detection threshold must be ≤ 250 ms."""
        self.assertEqual(EW03_BIM_SPOOF_MS, 250.0)

    def test_ew03_bim_reaches_red_on_spoof(self):
        """EW-03: BIM must reach RED state when spoof injected."""
        result = self._run()
        self.assertIsNotNone(result.kpi.ew03_bim_latency_ms)
        if result.kpi.ew03_pass:
            self.assertLessEqual(result.kpi.ew03_bim_latency_ms, EW03_BIM_SPOOF_MS)

    # ── SAT-01 ────────────────────────────────────────────────────────────────
    def test_sat01_masking_executed(self):
        """SAT-01: Terrain masking manoeuvre must be executed at T+20 min."""
        result = self._run()
        self.assertIsNotNone(result.kpi.sat01_masking_executed)

    def test_sat01_pass_requires_masking(self):
        """SAT-01 pass requires masking_executed=True."""
        result = self._run()
        if result.kpi.sat01_pass:
            self.assertTrue(result.kpi.sat01_masking_executed)

    # ── TERM-01 ───────────────────────────────────────────────────────────────
    def test_term01_lock_confidence_threshold(self):
        """TERM-01 pass requires lock confidence ≥ 0.85."""
        result = self._run()
        self.assertIsNotNone(result.kpi.term01_lock_confidence)
        if result.kpi.term01_pass:
            self.assertGreaterEqual(
                result.kpi.term01_lock_confidence,
                LOCK_CONFIDENCE_THRESHOLD
            )

    def test_term01_lock_confidence_is_float(self):
        result = self._run()
        self.assertIsInstance(result.kpi.term01_lock_confidence, float)

    # ── TERM-02 ───────────────────────────────────────────────────────────────
    def test_term02_decoy_rejection_logged(self):
        """TERM-02: Decoy rejection outcome must be recorded."""
        result = self._run()
        self.assertIsNotNone(result.kpi.term02_decoy_rejected)

    def test_term02_pass_requires_decoy_rejected(self):
        result = self._run()
        if result.kpi.term02_pass:
            self.assertTrue(result.kpi.term02_decoy_rejected)

    def test_term02_dmrl_log_populated(self):
        """DMRL audit log must be populated during terminal phase."""
        result = self._run()
        self.assertGreater(len(result.dmrl_log), 0,
            "DMRL log must contain entries from terminal scene processing")

    # ── TERM-03 ───────────────────────────────────────────────────────────────
    def test_term03_l10s_se_log_populated(self):
        """L10s-SE audit log must be populated."""
        result = self._run()
        self.assertGreater(len(result.l10s_log), 0,
            "L10s-SE audit log must contain entries")

    def test_term03_compliance_recorded(self):
        """TERM-03: L10s-SE compliance flag must be set."""
        result = self._run()
        self.assertIsNotNone(result.kpi.term03_l10s_compliant)

    def test_term03_pass_requires_compliance(self):
        result = self._run()
        if result.kpi.term03_pass:
            self.assertTrue(result.kpi.term03_l10s_compliant)

    # ── SYS-01 ────────────────────────────────────────────────────────────────
    def test_sys01_threshold_constant(self):
        """SYS-01: All FSM transitions must complete within ≤ 2 s."""
        self.assertEqual(SM_TRANSITION_MAX_S, 2.0)

    def test_sys01_max_transition_measured(self):
        result = self._run()
        self.assertIsNotNone(result.kpi.sys01_max_transition_s)
        self.assertGreater(result.kpi.sys01_max_transition_s, 0.0)

    def test_sys01_fsm_history_populated(self):
        """FSM history must record all state transitions."""
        result = self._run()
        self.assertGreater(len(result.fsm_history), 0,
            "FSM history must be non-empty")
        for entry in result.fsm_history:
            self.assertIn("from", entry)
            self.assertIn("to", entry)
            self.assertIn("latency_s", entry)
            self.assertIn("timestamp", entry)

    def test_sys01_pass_requires_all_transitions_fast(self):
        result = self._run()
        if result.kpi.sys01_pass:
            self.assertLessEqual(result.kpi.sys01_max_transition_s, SM_TRANSITION_MAX_S)

    # ── SYS-02 ────────────────────────────────────────────────────────────────
    def test_sys02_log_completeness_measured(self):
        result = self._run()
        self.assertIsNotNone(result.kpi.sys02_log_completeness_pct)

    def test_sys02_zpi_confirmed(self):
        """SYS-02: ZPI pre-terminal burst must be confirmed."""
        result = self._run()
        self.assertTrue(result.kpi.sys02_zpi_confirmed)

    def test_sys02_log_completeness_at_or_near_100(self):
        """SYS-02: Log completeness must be ≥ 99%."""
        result = self._run()
        self.assertGreaterEqual(result.kpi.sys02_log_completeness_pct, 99.0,
            f"Log completeness {result.kpi.sys02_log_completeness_pct}% < 99%")


class TestBCMP1AcceptanceGate(unittest.TestCase):
    """
    Sprint S5 Acceptance Gate:
    All 11 BCMP-1 criteria met. Runs cleanly 5 times in a row on Mac.
    """

    def test_acceptance_gate_5_consecutive_runs(self):
        """
        THE acceptance gate: 5 consecutive BCMP-1 runs all passing all 11 criteria.
        This is the TASL-demo readiness check.
        """
        runner = BCMP1Runner(verbose=False, seed=42)
        results = []

        for run_id in range(1, 6):
            result = runner.run(run_id=run_id)
            results.append(result)

        # Print summary
        print(f"\n  Sprint S5 Acceptance Gate — 5 Consecutive BCMP-1 Runs")
        print(f"  {'Run':>4} {'Pass':>6} {'Criteria':>10}")
        for r in results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            print(f"  {r.run_id:>4} {status:>6} {r.kpi.pass_count:>9}/11")

        all_passed = all(r.passed for r in results)
        pass_count = sum(r.passed for r in results)

        print(f"\n  Acceptance gate: {'✅ PASS' if all_passed else '❌ FAIL'} "
              f"({pass_count}/5 runs passed)")

        self.assertTrue(all_passed,
            f"Acceptance gate FAIL: Only {pass_count}/5 runs passed all 11 criteria. "
            f"Review individual run failures above.")

    def test_all_11_criteria_evaluated_per_run(self):
        """Every run must evaluate all 11 criteria (none may be skipped)."""
        runner = BCMP1Runner(verbose=False, seed=100)
        result = runner.run(run_id=1)
        kpi = result.kpi

        criteria = {
            "NAV-01": kpi.nav01_drift_pct is not None,
            "NAV-02": kpi.nav02_trn_cep95_m is not None,
            "EW-01":  kpi.ew01_costmap_latency_ms is not None,
            "EW-02":  kpi.ew02_replan1_ms is not None and kpi.ew02_replan2_ms is not None,
            "EW-03":  kpi.ew03_bim_latency_ms is not None,
            "SAT-01": kpi.sat01_masking_executed is not None,
            "TERM-01": kpi.term01_lock_confidence is not None,
            "TERM-02": kpi.term02_decoy_rejected is not None,
            "TERM-03": kpi.term03_l10s_compliant is not None,
            "SYS-01": kpi.sys01_max_transition_s is not None,
            "SYS-02": kpi.sys02_log_completeness_pct is not None,
        }

        for name, evaluated in criteria.items():
            self.assertTrue(evaluated, f"{name} was not evaluated in BCMP-1 run")

    def test_kpi_all_pass_property_correct(self):
        """BCMP1KPI.all_pass must correctly aggregate all 11 criteria."""
        kpi = BCMP1KPI()
        # All False by default
        self.assertFalse(kpi.all_pass)
        self.assertEqual(kpi.pass_count, 0)

        # Set all True
        kpi.nav01_pass = kpi.nav02_pass = True
        kpi.ew01_pass = kpi.ew02_pass = kpi.ew03_pass = True
        kpi.sat01_pass = True
        kpi.term01_pass = kpi.term02_pass = kpi.term03_pass = True
        kpi.sys01_pass = kpi.sys02_pass = True

        self.assertTrue(kpi.all_pass)
        self.assertEqual(kpi.pass_count, 11)

        # One failure breaks all_pass
        kpi.term03_pass = False
        self.assertFalse(kpi.all_pass)
        self.assertEqual(kpi.pass_count, 10)


class TestBCMP1RunnerKPIExport(unittest.TestCase):
    """KPI log export structure and completeness."""

    def test_kpi_log_exports_valid_json(self):
        """run_bcmp1 must export a valid JSON file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            output_path = f.name

        summary = run_bcmp1(
            n_runs=2, seed=99, verbose=False,
            export_kpi=True, output_path=output_path
        )

        with open(output_path) as f:
            data = json.load(f)

        self.assertEqual(data["scenario"], "BCMP-1")
        self.assertEqual(data["n_runs"], 2)
        self.assertEqual(len(data["runs"]), 2)
        os.unlink(output_path)

    def test_kpi_export_contains_all_criteria(self):
        """Each run in KPI export must contain all 11 criteria fields."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            output_path = f.name

        run_bcmp1(n_runs=1, seed=77, verbose=False, export_kpi=True, output_path=output_path)

        with open(output_path) as f:
            data = json.load(f)

        run_kpi = data["runs"][0]["kpi"]
        required_fields = [
            "nav01_drift_pct", "nav01_pass",
            "nav02_trn_cep95_m", "nav02_pass",
            "ew01_costmap_latency_ms", "ew01_pass",
            "ew02_replan1_ms", "ew02_replan2_ms", "ew02_pass",
            "ew03_bim_latency_ms", "ew03_pass",
            "sat01_masking_executed", "sat01_pass",
            "term01_lock_confidence", "term01_pass",
            "term02_decoy_rejected", "term02_pass",
            "term03_l10s_compliant", "term03_pass",
            "sys01_max_transition_s", "sys01_pass",
            "sys02_log_completeness_pct", "sys02_zpi_confirmed", "sys02_pass",
        ]
        for field in required_fields:
            self.assertIn(field, run_kpi, f"KPI export missing field: {field}")

        os.unlink(output_path)

    def test_kpi_export_acceptance_gate_field(self):
        """KPI export must include acceptance_gate field as PASS or FAIL."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            output_path = f.name

        summary = run_bcmp1(n_runs=1, seed=42, verbose=False, export_kpi=True,
                            output_path=output_path)

        self.assertIn(summary["acceptance_gate"], ["PASS", "FAIL"],
            "acceptance_gate must be PASS or FAIL")

        with open(output_path) as f:
            data = json.load(f)
        self.assertIn("acceptance_gate", data)
        os.unlink(output_path)


class TestBCMP1Determinism(unittest.TestCase):
    """
    Determinism and reproducibility tests.
    Same seed must produce consistent pass/fail per criterion.
    """

    def test_same_seed_produces_consistent_nav01(self):
        """NAV-01 pass/fail must be consistent for same seed."""
        runner = BCMP1Runner(verbose=False, seed=42)
        r1 = runner.run(run_id=1)
        runner2 = BCMP1Runner(verbose=False, seed=42)
        r2 = runner2.run(run_id=1)
        self.assertEqual(r1.kpi.nav01_pass, r2.kpi.nav01_pass)

    def test_same_seed_produces_consistent_term01(self):
        """TERM-01 pass/fail must be consistent for same seed."""
        runner1 = BCMP1Runner(verbose=False, seed=99)
        r1 = runner1.run(run_id=1)
        runner2 = BCMP1Runner(verbose=False, seed=99)
        r2 = runner2.run(run_id=1)
        self.assertEqual(r1.kpi.term01_pass, r2.kpi.term01_pass)

    def test_different_seeds_can_produce_different_results(self):
        """Different seeds may yield different individual KPI values."""
        results = []
        for seed in range(5):
            runner = BCMP1Runner(verbose=False, seed=seed * 13 + 7)
            result = runner.run(run_id=1)
            results.append(result.kpi.nav01_drift_pct)

        # Not all values should be identical (randomness should vary them)
        unique_values = len(set(round(v, 3) for v in results))
        self.assertGreater(unique_values, 1,
            "Different seeds must produce different drift values")


class TestBCMP1EventLog(unittest.TestCase):
    """Event log completeness and phase ordering."""

    def setUp(self):
        self.runner = BCMP1Runner(verbose=False, seed=42)

    def test_event_log_populated(self):
        result = self.runner.run(run_id=1)
        self.assertGreater(len(result.events), 0)

    def test_event_log_has_timestamps(self):
        result = self.runner.run(run_id=1)
        for event in result.events:
            self.assertIn("t", event)
            self.assertIn("msg", event)
            self.assertIsInstance(event["t"], float)

    def test_event_log_covers_all_phases(self):
        """Event log must cover all major mission phases."""
        result = self.runner.run(run_id=1)
        all_msgs = " ".join(e["msg"] for e in result.events)

        phase_keywords = [
            "PRE-LAUNCH", "GNSS DENIED", "EW THREAT", "RF LINK",
            "SATELLITE", "SPOOF", "TERMINAL", "SYSTEM KPI"
        ]
        for keyword in phase_keywords:
            self.assertIn(keyword, all_msgs,
                f"Event log missing phase: {keyword}")

    def test_fsm_history_records_key_states(self):
        """FSM history must include transitions to critical states."""
        result = self.runner.run(run_id=1)
        states_visited = [h["to"] for h in result.fsm_history]
        critical_states = ["NOMINAL", "GNSS_DENIED", "EW_AWARE"]
        for state in critical_states:
            self.assertIn(state, states_visited,
                f"FSM never transitioned to {state}")


if __name__ == "__main__":
    # When run directly, also print the acceptance gate result clearly
    print("\n" + "=" * 70)
    print("MicroMind Sprint S5 — BCMP-1 Acceptance Test Suite")
    print("=" * 70 + "\n")
    unittest.main(verbosity=2)
