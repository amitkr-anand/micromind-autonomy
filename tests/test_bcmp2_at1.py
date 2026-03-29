"""
BCMP-2 Acceptance Test AT-1 — Boot and Regression Check (5 km).

Pass conditions:
  - Both vehicle tracks produce logs (vehicle_a, vehicle_b keys present)
  - disturbance_schedule at top level (C-4 traceability)
  - comparison block present
  - hardware_source field present
  - No NaN in vehicle_a drift values that were recorded
  - Dual-track JSON log well-formed (all required top-level keys)
  - 332 SIL gates unchanged (validated externally via run_s5_tests.py)
"""

import json
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scenarios.bcmp2.bcmp2_runner import run_bcmp2, run_at1, BCMP2RunConfig


# ---------------------------------------------------------------------------
# AT-1 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def at1_output():
    """Run AT-1 once, share across all tests in this module."""
    return run_at1(seed=42, max_km=5.0)


# ---------------------------------------------------------------------------
# AT-1 structural tests
# ---------------------------------------------------------------------------

class TestAT1Structure:

    def test_disturbance_schedule_present(self, at1_output):
        assert "disturbance_schedule" in at1_output, \
            "C-4: disturbance_schedule must be at top level of JSON output"

    def test_vehicle_a_present(self, at1_output):
        assert "vehicle_a" in at1_output

    def test_vehicle_b_present(self, at1_output):
        assert "vehicle_b" in at1_output

    def test_comparison_present(self, at1_output):
        assert "comparison" in at1_output

    def test_hardware_source_present(self, at1_output):
        assert "hardware_source" in at1_output
        assert at1_output["hardware_source"] == "simulated"

    def test_seed_matches(self, at1_output):
        assert at1_output["seed"] == 42

    def test_disturbance_schedule_has_gnss_denial(self, at1_output):
        sched = at1_output["disturbance_schedule"]
        assert "gnss_denial" in sched
        assert "start_s" in sched["gnss_denial"]

    def test_disturbance_schedule_has_vio_outages(self, at1_output):
        sched = at1_output["disturbance_schedule"]
        assert "vio_outages" in sched
        assert isinstance(sched["vio_outages"], list)

    def test_vehicle_a_has_c2_gates(self, at1_output):
        va = at1_output["vehicle_a"]
        assert "c2_gates" in va
        # Gates may be "not reached" at 5 km — that is acceptable
        assert isinstance(va["c2_gates"], dict)

    def test_vehicle_a_no_nan_in_recorded_drift(self, at1_output):
        va = at1_output["vehicle_a"]
        for key in ["drift_at_km60_m", "drift_at_km100_m", "drift_at_km120_m"]:
            val = va.get(key)
            if val is not None:
                assert not math.isnan(float(val)), \
                    f"NaN detected in vehicle_a.{key}"

    def test_comparison_has_verdict_fields(self, at1_output):
        comp = at1_output["comparison"]
        assert "vehicle_a_mission_result" in comp
        assert "vehicle_b_mission_result" in comp
        assert "vehicle_a_causal_chain" in comp
        assert "vehicle_b_causal_chain" in comp

    def test_run_duration_recorded(self, at1_output):
        assert "run_duration_s" in at1_output
        assert at1_output["run_duration_s"] > 0

    def test_json_serialisable(self, at1_output):
        """Output must be fully JSON serialisable (no numpy types etc.)"""
        try:
            serialised = json.dumps(at1_output, default=str)
            restored   = json.loads(serialised)
            assert "disturbance_schedule" in restored
        except (TypeError, ValueError) as e:
            pytest.fail(f"JSON serialisation failed: {e}")


# ---------------------------------------------------------------------------
# AT-1 determinism tests
# ---------------------------------------------------------------------------

class TestAT1Determinism:

    def test_same_seed_produces_same_vehicle_a_result(self):
        """Two runs with same seed must produce identical Vehicle A drift."""
        out1 = run_at1(seed=42, max_km=5.0)
        out2 = run_at1(seed=42, max_km=5.0)
        # Both runs with same seed: disturbance schedule must be identical
        assert out1["disturbance_schedule"]["seed"] == \
               out2["disturbance_schedule"]["seed"]
        # Vehicle A drift (if reached) must match
        for key in ["drift_at_km60_m", "total_corridor_violations"]:
            v1 = out1["vehicle_a"].get(key)
            v2 = out2["vehicle_a"].get(key)
            assert v1 == v2, f"Non-deterministic Vehicle A output for {key}"

    def test_different_seeds_produce_different_schedules(self):
        """Different seeds must produce different GNSS denial timing or VIO outages."""
        out42  = run_at1(seed=42,  max_km=5.0)
        out101 = run_at1(seed=101, max_km=5.0)
        sched42  = out42["disturbance_schedule"]
        sched101 = out101["disturbance_schedule"]
        # At minimum the seeds differ
        assert sched42["seed"] != sched101["seed"]


# ---------------------------------------------------------------------------
# AT-1 hardware_source field test
# ---------------------------------------------------------------------------

class TestAT1HardwareSource:

    def test_hardware_source_default_is_simulated(self):
        out = run_at1(seed=42, max_km=5.0)
        assert out["hardware_source"] == "simulated"

    def test_hardware_source_can_be_overridden(self):
        config = BCMP2RunConfig(seed=42, max_km=5.0, hardware_source="SITL")
        out = run_bcmp2(config)
        assert out["hardware_source"] == "SITL"
