"""
BCMP-2 Acceptance Test AT-2 — Nominal 150 km Dual-Track.

Tests that a full 150 km dual-track run with the canonical seeds produces
the expected comparative outcome:

  Vehicle A: INS-only after GNSS denial at km 30 — drift accumulates
             per C-2 envelope, corridor breached by P4 for stressed seeds.
  Vehicle B: MicroMind full stack — structured KPI output, mission log
             present, dual-track JSON complete and serialisable.

AT-2 pass criteria (per architecture doc §9):
  1. Both vehicle tracks produce KPI logs for all three canonical seeds.
  2. Vehicle A C-2 gates PASS for all three seeds (drift within envelope).
  3. Vehicle A corridor breach occurs by km 150 for at least one seed
     (demonstrates the failure mode that MicroMind prevents).
  4. Dual-track JSON is complete, serialisable, and contains disturbance
     schedule at top level (C-4 traceability).
  5. HTML report generates without error, business block appears first.

Note: Vehicle B outcome is "PARTIAL" in SIL mode because the BCMP-1
runner returns stub KPIs without full navigation state propagation.
The AT-2 test does not assert on Vehicle B's absolute mission success —
that is AT-5 (terminal integrity). AT-2 asserts on structural completeness
and Vehicle A failure demonstration.

Seeds: 42 (nominal), 101 (stressed — breach expected by km 150).
Seed 303 excluded from AT-2 because runtime >60s; included in AT-6
repeatability.
"""

import json
import os
import tempfile
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scenarios.bcmp2.bcmp2_runner import run_bcmp2, BCMP2RunConfig
from scenarios.bcmp2.bcmp2_report import BCMPReport, generate_report
from scenarios.bcmp2.bcmp2_drift_envelopes import PHASE_ENVELOPES


# ---------------------------------------------------------------------------
# Fixtures — run once per module (full 150 km takes ~27s per seed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run_seed42():
    config = BCMP2RunConfig(seed=42, max_km=150.0, verbose=False)
    return run_bcmp2(config)


@pytest.fixture(scope="module")
def run_seed101():
    config = BCMP2RunConfig(seed=101, max_km=150.0, verbose=False)
    return run_bcmp2(config)


# ---------------------------------------------------------------------------
# AT-2-A: Structural completeness (both seeds)
# ---------------------------------------------------------------------------

class TestAT2Structure:

    def test_top_level_keys_seed42(self, run_seed42):
        for key in ["disturbance_schedule", "vehicle_a", "vehicle_b",
                    "comparison", "bcmp2_version", "seed"]:
            assert key in run_seed42, f"Missing key: {key}"

    def test_top_level_keys_seed101(self, run_seed101):
        for key in ["disturbance_schedule", "vehicle_a", "vehicle_b",
                    "comparison", "bcmp2_version", "seed"]:
            assert key in run_seed101, f"Missing key: {key}"

    def test_disturbance_schedule_top_level_seed42(self, run_seed42):
        """C-4: schedule must be at top level, not nested inside vehicle data."""
        sched = run_seed42["disturbance_schedule"]
        assert "gnss_denial" in sched
        assert sched["seed"] == 42

    def test_disturbance_schedule_top_level_seed101(self, run_seed101):
        sched = run_seed101["disturbance_schedule"]
        assert "gnss_denial" in sched
        assert sched["seed"] == 101

    def test_vehicle_a_has_drift_values_seed42(self, run_seed42):
        va = run_seed42["vehicle_a"]
        for key in ["drift_at_km60_m", "drift_at_km100_m", "drift_at_km120_m"]:
            assert va.get(key) is not None, f"Missing: {key}"
            assert va[key] > 0, f"Drift must be positive: {key}"

    def test_vehicle_a_has_drift_values_seed101(self, run_seed101):
        va = run_seed101["vehicle_a"]
        for key in ["drift_at_km60_m", "drift_at_km100_m", "drift_at_km120_m"]:
            assert va.get(key) is not None, f"Missing: {key}"

    def test_vehicle_b_has_kpi_fields(self, run_seed42):
        vb = run_seed42["vehicle_b"]
        # At minimum these fields must exist (may be None in SIL mode)
        for key in ["nav01_pass", "all_criteria_met"]:
            assert key in vb, f"Missing Vehicle B KPI field: {key}"

    def test_json_serialisable_seed42(self, run_seed42):
        s = json.dumps(run_seed42, default=str)
        restored = json.loads(s)
        assert "comparison" in restored

    def test_json_serialisable_seed101(self, run_seed101):
        s = json.dumps(run_seed101, default=str)
        restored = json.loads(s)
        assert "comparison" in restored

    def test_seeds_differ(self, run_seed42, run_seed101):
        """Different seeds must produce different Vehicle A drift values."""
        d42  = run_seed42["vehicle_a"].get("drift_at_km120_m")
        d101 = run_seed101["vehicle_a"].get("drift_at_km120_m")
        assert d42 != d101, "Seeds must produce distinct drift trajectories"

    def test_run_duration_positive(self, run_seed42):
        assert run_seed42["run_duration_s"] > 0

    def test_imu_model_recorded(self, run_seed42):
        assert "STIM300" in run_seed42.get("imu_model", ""), \
            f"Expected STIM300 in imu_model, got: {run_seed42.get('imu_model')}"


# ---------------------------------------------------------------------------
# AT-2-B: Vehicle A C-2 gate compliance (both seeds)
# ---------------------------------------------------------------------------

class TestAT2VehicleAC2Gates:

    def test_c2_gates_all_pass_seed42(self, run_seed42):
        gates = run_seed42["vehicle_a"]["c2_gates"]
        failures = []
        for km, g in gates.items():
            if g.get("observed_m") is not None and not g.get("passed"):
                obs = g["observed_m"]
                fl  = PHASE_ENVELOPES[km]["floor"]
                ce  = PHASE_ENVELOPES[km]["ceiling"]
                failures.append(f"km{km}: {obs:.0f}m not in [{fl},{ce}]")
        assert not failures, f"C-2 gate failures (seed 42): {failures}"

    def test_c2_gates_all_pass_seed101(self, run_seed101):
        gates = run_seed101["vehicle_a"]["c2_gates"]
        failures = []
        for km, g in gates.items():
            if g.get("observed_m") is not None and not g.get("passed"):
                obs = g["observed_m"]
                fl  = PHASE_ENVELOPES[km]["floor"]
                ce  = PHASE_ENVELOPES[km]["ceiling"]
                failures.append(f"km{km}: {obs:.0f}m not in [{fl},{ce}]")
        assert not failures, f"C-2 gate failures (seed 101): {failures}"

    def test_drift_grows_with_distance_seed42(self, run_seed42):
        """Drift must increase monotonically with mission distance."""
        va   = run_seed42["vehicle_a"]
        d60  = va.get("drift_at_km60_m",  0)
        d100 = va.get("drift_at_km100_m", 0)
        d120 = va.get("drift_at_km120_m", 0)
        assert d60 < d100 < d120, \
            f"Drift must grow: {d60:.0f} < {d100:.0f} < {d120:.0f}"

    def test_drift_grows_with_distance_seed101(self, run_seed101):
        va   = run_seed101["vehicle_a"]
        d60  = va.get("drift_at_km60_m",  0)
        d100 = va.get("drift_at_km100_m", 0)
        d120 = va.get("drift_at_km120_m", 0)
        assert d60 < d100 < d120, \
            f"Drift must grow: {d60:.0f} < {d100:.0f} < {d120:.0f}"

    def test_seed42_drift_above_floor(self, run_seed42):
        """Even the low-drift seed must be above C-2 minimum floor at km 60."""
        d60 = run_seed42["vehicle_a"].get("drift_at_km60_m", 0)
        floor = PHASE_ENVELOPES[60]["floor"]
        assert d60 >= floor, \
            f"Seed 42 drift {d60:.1f}m below floor {floor}m — simulation bug"

    def test_seed101_corridor_breach_by_km150(self, run_seed101):
        """
        AT-2 failure demonstration: stressed seed (101) must breach the
        corridor by km 150, proving the failure mode that MicroMind prevents.
        """
        breach_km = run_seed101["vehicle_a"].get("first_corridor_violation_km")
        assert breach_km is not None, \
            "Seed 101 must produce a corridor breach — failure mode not demonstrated"
        assert breach_km <= 150.0, \
            f"Breach at km {breach_km:.1f} — expected by km 150"

    def test_seed42_final_drift_exceeds_50m(self, run_seed42):
        """
        Seed 42 drift at km 120 must exceed 50 m — enough to show meaningful
        navigation degradation even in the nominal case.
        """
        d120 = run_seed42["vehicle_a"].get("drift_at_km120_m", 0)
        assert d120 >= 50, \
            f"Seed 42 drift at km120 = {d120:.1f}m — expected ≥ 50m for meaningful comparison"


# ---------------------------------------------------------------------------
# AT-2-C: Comparison block correctness
# ---------------------------------------------------------------------------

class TestAT2Comparison:

    def test_comparison_has_gnss_denial_km(self, run_seed42):
        comp = run_seed42["comparison"]
        assert "gnss_denial_km" in comp
        assert comp["gnss_denial_km"] > 0

    def test_comparison_has_mission_results(self, run_seed42):
        comp = run_seed42["comparison"]
        assert "vehicle_a_mission_result" in comp
        assert "vehicle_b_mission_result" in comp

    def test_seed101_vehicle_a_result_is_failed(self, run_seed101):
        """Seed 101 breaches corridor — Vehicle A must be marked FAILED."""
        result = run_seed101["comparison"]["vehicle_a_mission_result"]
        assert result == "FAILED", \
            f"Seed 101 Vehicle A result should be FAILED, got {result!r}"

    def test_causal_chains_present(self, run_seed42):
        comp = run_seed42["comparison"]
        assert len(comp.get("vehicle_a_causal_chain", [])) >= 3
        assert len(comp.get("vehicle_b_causal_chain", [])) >= 3

    def test_c2_gates_summary_in_comparison(self, run_seed42):
        comp = run_seed42["comparison"]
        assert "vehicle_a_c2_gates_all_passed" in comp


# ---------------------------------------------------------------------------
# AT-2-D: Report generation
# ---------------------------------------------------------------------------

class TestAT2Report:

    def test_html_report_generates_seed42(self, run_seed42):
        report = BCMPReport(run_seed42, run_date="2026-03-30")
        html   = report.to_html()
        assert len(html) > 2000, "HTML report too short"
        assert "Mission Outcome" in html
        assert "Without MicroMind" in html
        assert "With MicroMind" in html

    def test_business_block_before_technical_seed42(self, run_seed42):
        """Architecture doc §8.3: business block must precede technical tables."""
        report = BCMPReport(run_seed42, run_date="2026-03-30")
        html   = report.to_html()
        business_pos  = html.index("Mission Outcome")
        technical_pos = html.index("Technical Evidence")
        assert business_pos < technical_pos, \
            "Business comparison block must appear before technical tables"

    def test_html_report_generates_seed101(self, run_seed101):
        report = BCMPReport(run_seed101, run_date="2026-03-30")
        html   = report.to_html()
        assert "FAILED" in html, "Seed 101 FAILED outcome must appear in report"

    def test_json_report_round_trips(self, run_seed42):
        report  = BCMPReport(run_seed42, run_date="2026-03-30")
        j       = report.to_json()
        restored = json.loads(j)
        assert restored["seed"] == 42
        assert "vehicle_a" in restored

    def test_generate_report_writes_files(self, run_seed42):
        with tempfile.TemporaryDirectory() as tmp:
            jp, hp = generate_report(run_seed42, output_dir=tmp,
                                     run_date="20260330")
            assert os.path.exists(jp)
            assert os.path.exists(hp)
            assert os.path.getsize(hp) > 2000
            assert os.path.getsize(jp) > 200
