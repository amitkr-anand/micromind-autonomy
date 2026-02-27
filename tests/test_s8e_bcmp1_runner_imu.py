"""
tests/test_s8e_bcmp1_runner_imu.py — Sprint S8-E Acceptance Tests

Run from repo root:
    PYTHONPATH=. python -m pytest tests/test_s8e_bcmp1_runner_imu.py -v

Criteria:
  E1 — run_bcmp1(seed=42, corridor_km=5.0) with no imu_model is identical signature to S5
  E2 — run_bcmp1(seed=42, corridor_km=5.0, imu_model=None) → same result as S5 (backward compat)
  E3 — run_bcmp1 with STIM300 completes and returns BCMPResult
  E4 — run_bcmp1 with BASELINE completes and passes all 11 criteria
  E5 — BCMPResult.imu_model_name is populated correctly
  E6 — KPI JSON includes imu_model field
  E7 — All 147 existing tests unaffected (signature check; run separately)
  E8 — CLI --imu-model flag runs without error
"""
import sys
import inspect
import json
import pathlib
import tempfile

import pytest

from scenarios.bcmp1.bcmp1_runner import run_bcmp1, BCMPResult, main as runner_main
from core.ins.imu_model import get_imu_model


class TestE1E2BackwardCompat:
    def test_e1_signature_preserved(self):
        """run_bcmp1(seed, kpi_log_path) positional args still work."""
        sig = inspect.signature(run_bcmp1)
        params = list(sig.parameters.keys())
        assert "seed" in params
        assert "kpi_log_path" in params

    def test_e2_imu_model_defaults_to_none(self):
        p = inspect.signature(run_bcmp1).parameters["imu_model"]
        assert p.default is None

    def test_e2_no_model_returns_bcmpresult(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = str(pathlib.Path(tmp) / "kpi.json")
            r = run_bcmp1(seed=42, corridor_km=5.0, kpi_log_path=log)
            assert isinstance(r, BCMPResult)

    def test_e2_no_model_imu_name_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = str(pathlib.Path(tmp) / "kpi.json")
            r = run_bcmp1(seed=42, corridor_km=5.0, kpi_log_path=log)
            assert r.imu_model_name == "NONE"


class TestE3STIM300Run:
    def test_e3_stim300_completes(self):
        model = get_imu_model("STIM300")
        with tempfile.TemporaryDirectory() as tmp:
            log = str(pathlib.Path(tmp) / "kpi.json")
            r = run_bcmp1(seed=42, corridor_km=5.0, kpi_log_path=log, imu_model=model)
            assert isinstance(r, BCMPResult)

    def test_e3_stim300_imu_name_recorded(self):
        model = get_imu_model("STIM300")
        with tempfile.TemporaryDirectory() as tmp:
            log = str(pathlib.Path(tmp) / "kpi.json")
            r = run_bcmp1(seed=42, corridor_km=5.0, kpi_log_path=log, imu_model=model)
            assert r.imu_model_name == "STIM300"

    def test_e3_stim300_has_11_criteria(self):
        model = get_imu_model("STIM300")
        with tempfile.TemporaryDirectory() as tmp:
            log = str(pathlib.Path(tmp) / "kpi.json")
            r = run_bcmp1(seed=42, corridor_km=5.0, kpi_log_path=log, imu_model=model)
            assert len(r.criteria) == 11


class TestE4BaselineAllPass:
    def test_e4_baseline_all_criteria_pass(self):
        """BASELINE model (zero noise) should not degrade any criterion."""
        model = get_imu_model("BASELINE")
        with tempfile.TemporaryDirectory() as tmp:
            log = str(pathlib.Path(tmp) / "kpi.json")
            r = run_bcmp1(seed=42, corridor_km=5.0, kpi_log_path=log, imu_model=model)
            failed = [k for k, v in r.criteria.items() if not v]
            assert r.passed, f"BASELINE run failed criteria: {failed}"


class TestE5E6KPIFields:
    def test_e5_imu_name_in_kpi(self):
        model = get_imu_model("ADIS16505_3")
        with tempfile.TemporaryDirectory() as tmp:
            log = str(pathlib.Path(tmp) / "kpi.json")
            r = run_bcmp1(seed=42, corridor_km=5.0, kpi_log_path=log, imu_model=model)
            assert r.kpi["imu_model"] == "ADIS16505_3"

    def test_e6_kpi_json_has_imu_field(self):
        model = get_imu_model("STIM300")
        with tempfile.TemporaryDirectory() as tmp:
            log = str(pathlib.Path(tmp) / "kpi.json")
            run_bcmp1(seed=42, corridor_km=5.0, kpi_log_path=log, imu_model=model)
            kpi = json.loads(pathlib.Path(log).read_text())
            assert "imu_model" in kpi
            assert kpi["imu_model"] == "STIM300"

    def test_e6_kpi_json_clean_run_imu_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = str(pathlib.Path(tmp) / "kpi.json")
            run_bcmp1(seed=42, corridor_km=5.0, kpi_log_path=log)
            kpi = json.loads(pathlib.Path(log).read_text())
            assert kpi["imu_model"] == "NONE"


class TestE7SignatureCompat:
    def test_e7_existing_caller_pattern_works(self):
        """Simulate S5-style call: run_bcmp1(seed=42, corridor_km=5.0, kpi_log_path=...) — must work."""
        with tempfile.TemporaryDirectory() as tmp:
            log = str(pathlib.Path(tmp) / "kpi.json")
            r = run_bcmp1(seed=42, corridor_km=5.0, kpi_log_path=log)
            assert isinstance(r, BCMPResult)
            assert len(r.criteria) == 11


class TestE8CLI:
    def test_e8_cli_no_imu(self):
        """CLI with no --imu-model flag (S5-compatible)."""
        # Redirect KPI log to /tmp
        import os
        orig = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            rc = runner_main(["--corridor-km", "5"])
            os.chdir(orig)
        assert rc == 0

    def test_e8_cli_with_imu_model(self):
        import os
        orig = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            rc = runner_main(["--imu-model", "BASELINE", "--seed", "42", "--corridor-km", "5"])
            os.chdir(orig)
        assert rc == 0


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        check=False).returncode)
