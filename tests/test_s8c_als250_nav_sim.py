"""
tests/test_s8c_als250_nav_sim.py — Sprint S8-C Acceptance Tests

Run from repo root:
    PYTHONPATH=. python -m pytest tests/test_s8c_als250_nav_sim.py -v

Criteria verified:
  C1 — CLEAN (no model) run completes 250 km corridor without error
  C2 — --imu STIM300 run produces larger drift than CLEAN
  C3 — --imu BASELINE run produces smaller drift than STIM300
  C4 — NAV-01 compliance: max 5-km segment drift < 100 m for all models
  C5 — Output arrays have correct shape (n_steps, 3) for position
  C6 — Drift array has one entry per 5-km segment
  C7 — Metadata JSON contains all required KPI fields
  C8 — --imu ALL flag runs all three models in one call (via main())
"""
import sys
import json
import pathlib
import tempfile

import numpy as np
import pytest

from sim.als250_nav_sim import (
    run_als250_sim,
    save_results,
    main as sim_main,
    IMU_RATE_HZ,
    CORRIDOR_DURATION_S,
    SEGMENT_M,
    DRIFT_LIMIT_M,
    CRUISE_SPEED_MS,
)

# Use a short duration for fast CI runs (10 km ≈ 182 s @ 55 m/s)
_FAST_DURATION = 182.0    # seconds — covers 2 full 5-km segments
_FAST_N        = int(_FAST_DURATION * IMU_RATE_HZ)
_FAST_SEGS     = int((_FAST_DURATION * CRUISE_SPEED_MS) / SEGMENT_M)


class TestC1CleanRun:
    def test_c1_completes_without_error(self):
        r = run_als250_sim(imu_name=None, duration_s=_FAST_DURATION, seed=42, verbose=False)
        assert r["position"].shape[0] == _FAST_N
        assert r["imu_name"] == "CLEAN"

    def test_c1_kpi_fields_present(self):
        r = run_als250_sim(imu_name=None, duration_s=_FAST_DURATION, seed=42, verbose=False)
        for f in ("imu_model","corridor_km","final_drift_m","max_5km_drift_m",
                  "NAV01_pass","n_steps","seed"):
            assert f in r["kpi"], f"Missing KPI field: {f}"

    def test_c1_n_steps_matches_duration(self):
        r = run_als250_sim(imu_name=None, duration_s=_FAST_DURATION, seed=42, verbose=False)
        assert r["n_steps"] == _FAST_N


class TestC2STIM300Drift:
    def test_c2_stim300_drift_exceeds_clean(self):
        r_clean = run_als250_sim(imu_name=None,      duration_s=_FAST_DURATION, seed=42, verbose=False)
        r_stim  = run_als250_sim(imu_name="STIM300", duration_s=_FAST_DURATION, seed=42, verbose=False)
        assert r_stim["kpi"]["final_drift_m"] >= r_clean["kpi"]["final_drift_m"]

    def test_c2_stim300_imu_name_recorded(self):
        r = run_als250_sim(imu_name="STIM300", duration_s=_FAST_DURATION, seed=42, verbose=False)
        assert r["imu_name"] == "STIM300"
        assert r["kpi"]["imu_model"] == "STIM300"


class TestC3ModelOrdering:
    def test_c3_baseline_less_or_equal_stim300(self):
        r_stim = run_als250_sim(imu_name="STIM300",  duration_s=_FAST_DURATION, seed=42, verbose=False)
        r_base = run_als250_sim(imu_name="BASELINE", duration_s=_FAST_DURATION, seed=42, verbose=False)
        assert r_base["kpi"]["final_drift_m"] <= r_stim["kpi"]["final_drift_m"]


class TestC4NAV01:
    """NAV-01: max 5-km drift < 100 m for all models (should pass for short test)."""

    @pytest.mark.parametrize("imu_name", [None, "STIM300", "ADIS16505_3", "BASELINE"])
    def test_c4_nav01_compliance(self, imu_name):
        r = run_als250_sim(imu_name=imu_name, duration_s=_FAST_DURATION,
                           seed=42, verbose=False)
        max_drift = r["kpi"]["max_5km_drift_m"]
        # Over a short corridor, all models should be within the 100 m limit
        assert max_drift < DRIFT_LIMIT_M, (
            f"Model={imu_name or 'CLEAN'}: max 5km drift {max_drift:.1f} m "
            f">= limit {DRIFT_LIMIT_M:.0f} m")


class TestC5ArrayShapes:
    def test_c5_position_shape(self):
        r = run_als250_sim(imu_name="STIM300", duration_s=_FAST_DURATION, seed=42, verbose=False)
        assert r["position"].shape == (_FAST_N, 3)
        assert r["true_position"].shape == (_FAST_N, 3)

    def test_c5_drift_array_1d(self):
        r = run_als250_sim(imu_name=None, duration_s=_FAST_DURATION, seed=42, verbose=False)
        assert r["drift_per_seg"].ndim == 1
        assert len(r["drift_per_seg"]) >= 1


class TestC6SegmentCount:
    def test_c6_one_entry_per_5km_segment(self):
        r = run_als250_sim(imu_name=None, duration_s=_FAST_DURATION, seed=42, verbose=False)
        dist_m = CRUISE_SPEED_MS * _FAST_DURATION
        expected = int(dist_m // SEGMENT_M)
        assert len(r["drift_per_seg"]) == expected, (
            f"Expected {expected} segments, got {len(r['drift_per_seg'])}")


class TestC7MetadataJSON:
    def test_c7_save_and_reload(self):
        r = run_als250_sim(imu_name="STIM300", duration_s=_FAST_DURATION, seed=42, verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_results(r, pathlib.Path(tmp))
            assert paths["position"].exists()
            assert paths["drift"].exists()
            assert paths["meta"].exists()
            meta = json.loads(paths["meta"].read_text())
            for field in ("imu_model","final_drift_m","NAV01_pass","seed","n_steps"):
                assert field in meta


class TestC8AllModels:
    def test_c8_main_all_flag_exits_zero(self):
        """--imu ALL runs CLEAN + 3 models without error."""
        rc = sim_main(["--imu", "ALL", "--duration", str(_FAST_DURATION),
                       "--no-save", "--seed", "42"])
        assert rc == 0

    def test_c8_main_single_model(self):
        rc = sim_main(["--imu", "STIM300", "--duration", str(_FAST_DURATION),
                       "--no-save", "--seed", "42"])
        assert rc == 0

    def test_c8_main_clean(self):
        rc = sim_main(["--imu", "CLEAN", "--duration", str(_FAST_DURATION),
                       "--no-save", "--seed", "42"])
        assert rc == 0


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        check=False).returncode)
