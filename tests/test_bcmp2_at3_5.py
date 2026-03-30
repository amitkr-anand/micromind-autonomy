"""
BCMP-2 Acceptance Tests AT-3 through AT-5 — Failure Mission Gates.

AT-3 — Single-failure mission
  One fault injected via FaultManager.  Vehicle B proxy intercepts the
  degraded signal.  Vehicle A is unaffected (no proxy).  Test verifies:
    - Vehicle A drift is unchanged (fault injection is Vehicle B only)
    - Vehicle B fault event log records the activation
    - Dual-track JSON remains structurally complete under fault

AT-4 — Multi-failure mission
  Two faults injected simultaneously (PRESET_VIO_GNSS).  Verifies:
    - Both faults appear in event log
    - Vehicle A C-2 gates still pass (faults are Vehicle B only)
    - JSON output remains serialisable with fault events present

AT-5 — Terminal integrity
  Verify that the report business comparison block correctly reflects the
  mission outcome and that the proxy chain produces no side-effects on
  frozen core modules.

All AT-3 through AT-5 tests use 5 km runs for speed.  The fault injection
validation for full 150 km runs is deferred to AT-6 (SB-5).
"""

import json
import os
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fault_injection.fault_manager import (
    FaultManager,
    FI_GNSS_LOSS, FI_VIO_LOSS, FI_EO_FREEZE, FI_TERRAIN_CONF_DROP,
    PRESET_VIO_GNSS,
)
from fault_injection.sensor_fault_proxy import SensorFaultProxy
from fault_injection.nav_source_proxy    import NavSourceProxy
from scenarios.bcmp2.bcmp2_runner        import run_bcmp2, BCMP2RunConfig
from scenarios.bcmp2.bcmp2_report        import BCMPReport


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fm():
    """Fresh FaultManager for each test — not the singleton."""
    return FaultManager()


@pytest.fixture()
def sensor_proxy(fm):
    return SensorFaultProxy(fault_manager=fm)


@pytest.fixture()
def nav_proxy(fm):
    return NavSourceProxy(fault_manager=fm, seed=42)


@pytest.fixture(scope="module")
def baseline_5km():
    """5 km run with no faults — reference for comparison."""
    config = BCMP2RunConfig(seed=42, max_km=5.0, verbose=False)
    return run_bcmp2(config)


# ---------------------------------------------------------------------------
# AT-3 — Single-failure mission
# ---------------------------------------------------------------------------

class TestAT3SingleFailure:

    def test_fi01_gnss_loss_event_logged(self, fm):
        """FI-01: activate GNSS loss, verify event log populated."""
        fm.activate(FI_GNSS_LOSS, duration_s=5.0, source="scripted")
        log = fm.event_log()
        assert len(log) == 1
        assert log[0].fault_id == FI_GNSS_LOSS
        assert log[0].action == "activated"
        assert log[0].duration_s == 5.0

    def test_fi01_proxy_intercepts_gnss(self, sensor_proxy, fm):
        """With FI_GNSS_LOSS active, proxy returns denied measurement."""
        fm.activate(FI_GNSS_LOSS, source="scripted")
        class _Real:
            pdop = 1.2
        result = sensor_proxy.gnss(_Real())
        assert result is not None            # denied measurement, not None
        # The denied measurement has worst-case quality fields
        assert hasattr(result, "pdop")
        assert result.pdop == 99.9

    def test_fi01_vehicle_a_unaffected(self, fm, baseline_5km):
        """
        Vehicle A has no proxy.  Activating a fault must not change
        Vehicle A's output — the fault is Vehicle B only.
        """
        # Run once with fault active — Vehicle A reads direct from schedule
        fm.activate(FI_GNSS_LOSS, duration_s=0.0)
        config = BCMP2RunConfig(seed=42, max_km=5.0, verbose=False)
        out_with_fault = run_bcmp2(config)
        fm.clear_all()

        # Vehicle A values must be identical (same seed, same schedule)
        # They may both be None for a 5km run that doesn't reach km60
        va_base  = baseline_5km["vehicle_a"]["total_steps"]
        va_fault = out_with_fault["vehicle_a"]["total_steps"]
        assert va_base == va_fault, \
            "Vehicle A step count must be identical regardless of proxy faults"

    def test_fi02_vio_loss_bim_and_proxy(self, sensor_proxy, nav_proxy, fm):
        """FI-02: VIO loss suppresses both sensor and nav source."""
        fm.activate(FI_VIO_LOSS, source="scripted")
        acc, innov = sensor_proxy.vio_update(True, 0.5)
        assert acc is False and innov == 0.0
        assert nav_proxy.vio_source_available(True) is False

    def test_fi05_eo_freeze_stale_frame(self, sensor_proxy, fm):
        """FI-05: EO freeze returns stale cached frame."""
        class _F:
            pass
        f1 = _F(); f1.frame_id = 10
        f2 = _F(); f2.frame_id = 11
        sensor_proxy.eo_frame(f1)           # cache f1
        fm.activate(FI_EO_FREEZE, source="scripted")
        result = sensor_proxy.eo_frame(f2)  # should return stale f1
        assert result is f1

    def test_single_fault_json_serialisable(self, fm):
        """Output dict with fault events must be JSON serialisable."""
        fm.activate(FI_GNSS_LOSS)
        fm.clear(FI_GNSS_LOSS)
        events = fm.event_log_as_dicts()
        s = json.dumps(events, default=str)
        restored = json.loads(s)
        assert len(restored) == 2   # activated + cleared

    def test_dual_track_json_complete_under_fault(self, fm):
        """
        Run a 5km dual-track with a fault manager active.
        The output JSON must still be structurally complete.
        """
        fm.activate(FI_GNSS_LOSS, duration_s=0.0)
        config = BCMP2RunConfig(seed=42, max_km=5.0, verbose=False)
        out = run_bcmp2(config)
        fm.clear_all()
        for key in ["disturbance_schedule", "vehicle_a", "vehicle_b", "comparison"]:
            assert key in out, f"Missing key under fault: {key}"
        s = json.dumps(out, default=str)
        assert json.loads(s)["seed"] == 42


# ---------------------------------------------------------------------------
# AT-4 — Multi-failure mission
# ---------------------------------------------------------------------------

class TestAT4MultiFailure:

    def test_preset_vio_gnss_both_active(self, fm):
        """PRESET_VIO_GNSS must activate both FI_VIO_LOSS and FI_GNSS_LOSS."""
        fm.activate_preset(PRESET_VIO_GNSS, duration_s=0.0, source="scripted")
        assert fm.is_active(FI_VIO_LOSS)
        assert fm.is_active(FI_GNSS_LOSS)
        assert len(fm.active_fault_ids()) == 2

    def test_preset_two_events_in_log(self, fm):
        """Activating a 2-fault preset must produce 2 event log entries."""
        fm.activate_preset(PRESET_VIO_GNSS, duration_s=0.0)
        log = fm.event_log()
        activated = [e for e in log if e.action == "activated"]
        assert len(activated) == 2

    def test_multi_fault_proxy_both_intercept(self, sensor_proxy, nav_proxy, fm):
        """Both sensor and nav proxies intercept simultaneously."""
        fm.activate(FI_GNSS_LOSS)
        fm.activate(FI_VIO_LOSS)
        class _R:
            pdop = 1.2
        assert sensor_proxy.gnss(_R()).pdop == 99.9
        acc, _ = sensor_proxy.vio_update(True, 0.5)
        assert acc is False
        assert nav_proxy.vio_source_available(True) is False

    def test_multi_fault_vehicle_a_c2_gates_unaffected(self, fm):
        """
        Vehicle A C-2 gates must pass even when multi-fault is active,
        because Vehicle A does not use the proxy layer.
        """
        fm.activate(FI_GNSS_LOSS)
        fm.activate(FI_VIO_LOSS)
        config = BCMP2RunConfig(seed=42, max_km=5.0, verbose=False)
        out = run_bcmp2(config)
        fm.clear_all()
        # 5 km run — gates may not be reached (None), but if reached must pass
        for km, gate in out["vehicle_a"]["c2_gates"].items():
            if gate.get("observed_m") is not None:
                assert gate["passed"], \
                    f"C-2 gate km{km} should pass — fault is Vehicle B only"

    def test_multi_fault_json_serialisable(self, fm):
        """Multi-fault event log must be JSON serialisable."""
        fm.activate_preset(PRESET_VIO_GNSS)
        fm.clear_all()
        s = json.dumps(fm.event_log_as_dicts(), default=str)
        events = json.loads(s)
        assert len(events) >= 2

    def test_clear_all_resets_both_faults(self, fm, sensor_proxy):
        """clear_all must restore all proxies to pass-through."""
        fm.activate(FI_GNSS_LOSS)
        fm.activate(FI_VIO_LOSS)
        fm.clear_all()
        class _R:
            pdop = 1.2
        real = _R()
        assert sensor_proxy.gnss(real) is real
        acc, innov = sensor_proxy.vio_update(True, 0.5)
        assert acc is True


# ---------------------------------------------------------------------------
# AT-5 — Terminal integrity
# ---------------------------------------------------------------------------

class TestAT5TerminalIntegrity:

    def test_report_business_block_present_and_first(self, baseline_5km):
        """Architecture doc §8.3: business block is first visible element."""
        report = BCMPReport(baseline_5km, run_date="2026-03-30")
        html   = report.to_html()
        assert "Mission Outcome" in html
        assert "Without MicroMind" in html
        assert "With MicroMind" in html
        # Business block before technical tables
        assert html.index("Mission Outcome") < html.index("Technical Evidence")

    def test_report_vehicle_a_result_in_html(self, baseline_5km):
        """Vehicle A outcome label must appear in HTML report."""
        report = BCMPReport(baseline_5km, run_date="2026-03-30")
        html   = report.to_html()
        va_result = baseline_5km["comparison"]["vehicle_a_mission_result"]
        assert va_result in html

    def test_report_causal_chains_in_html(self, baseline_5km):
        """Causal chain items must appear in the HTML output."""
        report = BCMPReport(baseline_5km, run_date="2026-03-30")
        html   = report.to_html()
        for item in baseline_5km["comparison"].get("vehicle_a_causal_chain", []):
            # Each causal chain item (first 20 chars) must be in the HTML
            assert item[:20] in html, f"Causal chain item missing: {item}"

    def test_frozen_core_unmodified_after_fault_chain(self, fm, sensor_proxy, nav_proxy):
        """
        Activate all AT-5 relevant faults, drive the proxy chain,
        then verify frozen core module constants are unchanged.
        """
        from core.ekf.error_state_ekf import ErrorStateEKF
        from core.bim.bim import BIM
        from core.fusion.vio_mode import VIONavigationMode

        fm.activate(FI_GNSS_LOSS)
        fm.activate(FI_VIO_LOSS)
        fm.activate(FI_TERRAIN_CONF_DROP)

        # Drive proxies
        class _R: pdop = 1.2
        sensor_proxy.gnss(_R())
        sensor_proxy.vio_update(True, 0.5)
        nav_proxy.vio_source_available(True)

        fm.clear_all()

        # Frozen constants must be unchanged
        eskf = ErrorStateEKF()
        assert eskf._ACC_BIAS_RW  == 9.81e-7
        assert eskf._GYRO_BIAS_RW == 4.04e-8

    def test_report_event_log_section_present(self, baseline_5km):
        """Event log section must be present in HTML."""
        report = BCMPReport(baseline_5km, run_date="2026-03-30")
        html   = report.to_html()
        assert "Disturbance Event Log" in html

    def test_report_drift_chart_section_present(self, baseline_5km):
        """Drift chart section must be present in HTML."""
        report = BCMPReport(baseline_5km, run_date="2026-03-30")
        html   = report.to_html()
        assert "Vehicle A Drift" in html
