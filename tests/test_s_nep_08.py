"""
tests/test_s_nep_08.py
======================
S-NEP-08 acceptance gates G-01 through G-08.

All tests are purely unit-level — no EuRoC data, no network access,
no ESKF state. Uses synthetic VIO sequences injected into
VIONavigationMode and FusionLogger directly.

Gates:
    G-01  NOMINAL → OUTAGE transition fires at threshold
    G-02  OUTAGE → RESUMPTION transition fires on first accepted update
    G-03  RESUMPTION → NOMINAL transition fires after required cycles
    G-04  Innovation spike alert fires when innov_mag > threshold
    G-04b No spike alert during normal NOMINAL operation
    G-05  Drift envelope grows monotonically during OUTAGE; None outside
    G-06  vel_err_m_s absent from log entries (default emit_vel_diagnostic=False)
    G-07  state_machine.py is NOT imported by vio_mode or fusion_logger
    G-08  current_mode accessible without importing ESKF
"""

import sys
import json
import importlib
import math
import tempfile
from pathlib import Path

import pytest

# Ensure micromind-autonomy is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fusion.vio_mode import (
    VIONavigationMode, VIOMode,
    VIO_OUTAGE_THRESHOLD_S,
    VIO_INNOVATION_SPIKE_THRESHOLD_M,
    VIO_DRIFT_RATE_CONSERVATIVE_M_S,
    VIO_RESUMPTION_CYCLES,
)
from core.fusion.fusion_logger import FusionLogger


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_nav(
    threshold_s: float = VIO_OUTAGE_THRESHOLD_S,
    spike_m: float = VIO_INNOVATION_SPIKE_THRESHOLD_M,
    cycles: int = VIO_RESUMPTION_CYCLES,
) -> VIONavigationMode:
    return VIONavigationMode(
        outage_threshold_s=threshold_s,
        spike_threshold_m=spike_m,
        resumption_cycles=cycles,
    )


def _tick_to_outage(nav: VIONavigationMode, threshold_s: float = VIO_OUTAGE_THRESHOLD_S,
                    dt: float = 0.01) -> None:
    """Advance clock until OUTAGE is declared."""
    steps = math.ceil(threshold_s / dt) + 1
    for _ in range(steps):
        nav.tick(dt)


# ─────────────────────────────────────────────────────────────────────────────
# G-01: NOMINAL → OUTAGE
# ─────────────────────────────────────────────────────────────────────────────

class TestG01NominalToOutage:
    def test_stays_nominal_below_threshold(self):
        nav = _make_nav(threshold_s=2.0)
        for _ in range(190):       # 1.9 s at 100Hz
            nav.tick(0.01)
        assert nav.current_mode is VIOMode.NOMINAL

    def test_transitions_at_threshold(self):
        nav = _make_nav(threshold_s=2.0)
        _tick_to_outage(nav, threshold_s=2.0)
        assert nav.current_mode is VIOMode.OUTAGE

    def test_outage_event_counter_increments(self):
        nav = _make_nav(threshold_s=2.0)
        assert nav.n_outage_events == 0
        _tick_to_outage(nav, threshold_s=2.0)
        assert nav.n_outage_events == 1

    def test_custom_threshold_respected(self):
        nav = _make_nav(threshold_s=0.5)
        for _ in range(45):        # 0.45 s — below threshold
            nav.tick(0.01)
        assert nav.current_mode is VIOMode.NOMINAL
        for _ in range(10):        # push past 0.5 s
            nav.tick(0.01)
        assert nav.current_mode is VIOMode.OUTAGE


# ─────────────────────────────────────────────────────────────────────────────
# G-02: OUTAGE → RESUMPTION
# ─────────────────────────────────────────────────────────────────────────────

class TestG02OutageToResumption:
    def test_transitions_on_first_accepted_update(self):
        nav = _make_nav()
        _tick_to_outage(nav)
        assert nav.current_mode is VIOMode.OUTAGE
        nav.on_vio_update(accepted=True, innov_mag=0.05)
        assert nav.current_mode is VIOMode.RESUMPTION

    def test_rejected_update_does_not_trigger_resumption(self):
        nav = _make_nav()
        _tick_to_outage(nav)
        nav.on_vio_update(accepted=False, innov_mag=0.05)
        assert nav.current_mode is VIOMode.OUTAGE

    def test_dt_since_vio_resets_on_resumption(self):
        nav = _make_nav()
        _tick_to_outage(nav)
        nav.on_vio_update(accepted=True, innov_mag=0.05)
        assert nav.dt_since_vio == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# G-03: RESUMPTION → NOMINAL
# ─────────────────────────────────────────────────────────────────────────────

class TestG03ResumptionToNominal:
    def test_transitions_after_required_cycles_default_1(self):
        nav = _make_nav(cycles=1)
        _tick_to_outage(nav)
        nav.on_vio_update(accepted=True, innov_mag=0.05)   # → RESUMPTION
        assert nav.current_mode is VIOMode.RESUMPTION
        # First accepted update in RESUMPTION counts; second completes the cycle
        nav.on_vio_update(accepted=True, innov_mag=0.05)   # completes cycle 1
        assert nav.current_mode is VIOMode.NOMINAL

    def test_transitions_after_required_cycles_custom_3(self):
        nav = _make_nav(cycles=3)
        _tick_to_outage(nav)
        nav.on_vio_update(accepted=True, innov_mag=0.05)   # → RESUMPTION, count=1
        assert nav.current_mode is VIOMode.RESUMPTION
        nav.on_vio_update(accepted=True, innov_mag=0.05)   # count=2
        assert nav.current_mode is VIOMode.RESUMPTION
        nav.on_vio_update(accepted=True, innov_mag=0.05)   # count=3 → NOMINAL
        assert nav.current_mode is VIOMode.NOMINAL

    def test_in_outage_flag_clears_on_nominal(self):
        nav = _make_nav(cycles=1)
        _tick_to_outage(nav)
        nav.on_vio_update(accepted=True, innov_mag=0.05)
        nav.on_vio_update(accepted=True, innov_mag=0.05)
        assert nav.current_mode is VIOMode.NOMINAL
        assert nav.in_outage is False

    def test_multiple_outage_cycles_counted(self):
        nav = _make_nav(threshold_s=2.0, cycles=1)
        # First outage/recovery
        _tick_to_outage(nav, threshold_s=2.0)
        nav.on_vio_update(accepted=True, innov_mag=0.05)
        nav.on_vio_update(accepted=True, innov_mag=0.05)
        assert nav.current_mode is VIOMode.NOMINAL
        assert nav.n_outage_events == 1
        # Second outage/recovery
        _tick_to_outage(nav, threshold_s=2.0)
        assert nav.n_outage_events == 2


# ─────────────────────────────────────────────────────────────────────────────
# G-04: Innovation spike alert
# ─────────────────────────────────────────────────────────────────────────────

class TestG04InnovationSpikeAlert:
    def test_spike_fires_when_innov_above_threshold(self):
        nav = _make_nav(spike_m=1.0)
        _tick_to_outage(nav)
        _, spike = nav.on_vio_update(accepted=True, innov_mag=1.5)
        assert spike is True
        assert nav.n_spike_alerts == 1

    def test_spike_does_not_fire_when_innov_below_threshold(self):
        nav = _make_nav(spike_m=1.0)
        _tick_to_outage(nav)
        _, spike = nav.on_vio_update(accepted=True, innov_mag=0.5)
        assert spike is False
        assert nav.n_spike_alerts == 0

    def test_spike_fires_at_exact_threshold_boundary(self):
        # Threshold is > (strict), so exactly at threshold should NOT fire
        nav = _make_nav(spike_m=1.0)
        _tick_to_outage(nav)
        _, spike = nav.on_vio_update(accepted=True, innov_mag=1.0)
        assert spike is False

    def test_spike_only_on_first_post_outage_update(self):
        nav = _make_nav(spike_m=1.0, cycles=3)
        _tick_to_outage(nav)
        _, spike1 = nav.on_vio_update(accepted=True, innov_mag=2.0)  # first
        assert spike1 is True
        _, spike2 = nav.on_vio_update(accepted=True, innov_mag=2.0)  # second — no spike
        assert spike2 is False


# ─────────────────────────────────────────────────────────────────────────────
# G-04b: No spike alert in NOMINAL
# ─────────────────────────────────────────────────────────────────────────────

class TestG04bNoSpikeInNominal:
    def test_no_spike_during_nominal_operation(self):
        nav = _make_nav(spike_m=1.0)
        # Many updates in NOMINAL with large innov — no spike ever
        for _ in range(100):
            _, spike = nav.on_vio_update(accepted=True, innov_mag=5.0)
            assert spike is False
        assert nav.n_spike_alerts == 0


# ─────────────────────────────────────────────────────────────────────────────
# G-05: Drift envelope
# ─────────────────────────────────────────────────────────────────────────────

class TestG05DriftEnvelope:
    def test_envelope_is_none_in_nominal(self):
        nav = _make_nav()
        assert nav.drift_envelope_m is None

    def test_envelope_is_none_in_resumption(self):
        nav = _make_nav()
        _tick_to_outage(nav)
        nav.on_vio_update(accepted=True, innov_mag=0.05)
        assert nav.current_mode is VIOMode.RESUMPTION
        assert nav.drift_envelope_m is None

    def test_envelope_grows_monotonically_during_outage(self):
        nav = _make_nav(threshold_s=0.1)
        for _ in range(15):
            nav.tick(0.01)
        assert nav.current_mode is VIOMode.OUTAGE

        values = []
        for _ in range(20):
            nav.tick(0.01)
            env = nav.drift_envelope_m
            assert env is not None
            values.append(env)

        # Must be monotonically non-decreasing
        for i in range(1, len(values)):
            assert values[i] >= values[i-1], \
                f"Drift envelope decreased: {values[i-1]:.4f} → {values[i]:.4f}"

    def test_envelope_formula(self):
        nav = _make_nav(threshold_s=1.0)
        # Tick exactly 1.5 s past threshold
        for _ in range(250):    # 2.5 s total
            nav.tick(0.01)
        assert nav.current_mode is VIOMode.OUTAGE
        expected = VIO_DRIFT_RATE_CONSERVATIVE_M_S * nav.dt_since_vio
        assert abs(nav.drift_envelope_m - expected) < 1e-9

    def test_envelope_resets_on_resumption(self):
        nav = _make_nav(threshold_s=0.5)
        for _ in range(60):
            nav.tick(0.01)
        assert nav.current_mode is VIOMode.OUTAGE
        assert nav.drift_envelope_m > 0
        nav.on_vio_update(accepted=True, innov_mag=0.05)
        assert nav.current_mode is VIOMode.RESUMPTION
        assert nav.drift_envelope_m is None


# ─────────────────────────────────────────────────────────────────────────────
# G-06: vel_err_m_s absent from log entries by default
# ─────────────────────────────────────────────────────────────────────────────

class TestG06VelErrAbsent:
    def test_vel_err_absent_from_vio_update_entry_default(self):
        with tempfile.TemporaryDirectory() as td:
            logger = FusionLogger(Path(td) / "test.json", emit_vel_diagnostic=False)
            logger.log_vio_update(
                t=1.0, nis=0.5, innov_mag=0.08, trace_P=0.01,
                vio_mode="NOMINAL", dt_since_vio=0.0,
                drift_envelope_m=None, innovation_spike_alert=False,
                error_m=0.09, ba_est=[0.0, 0.0, 0.0],
                vel_err_diagnostic=1.5,   # provided but should be suppressed
            )
            logger.close()
            data = json.loads((Path(td) / "test.json").read_text())
            entry = data["time_series"][0]
            assert "vel_err_m_s" not in entry, \
                "vel_err_m_s must not appear in schema 08.1 entries"
            assert "vel_err_diagnostic" not in entry, \
                "vel_err_diagnostic must not appear when emit_vel_diagnostic=False"

    def test_vel_err_diagnostic_present_when_flag_set(self):
        with tempfile.TemporaryDirectory() as td:
            logger = FusionLogger(Path(td) / "test.json", emit_vel_diagnostic=True)
            logger.log_vio_update(
                t=1.0, nis=0.5, innov_mag=0.08, trace_P=0.01,
                vio_mode="NOMINAL", dt_since_vio=0.0,
                drift_envelope_m=None, innovation_spike_alert=False,
                vel_err_diagnostic=1.5,
            )
            logger.close()
            data = json.loads((Path(td) / "test.json").read_text())
            entry = data["time_series"][0]
            assert "vel_err_diagnostic" in entry
            assert entry["vel_err_diagnostic"] == pytest.approx(1.5, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# G-07: state_machine.py not imported by fusion modules
# ─────────────────────────────────────────────────────────────────────────────

class TestG07StateMachineNotImported:
    """
    G-07: vio_mode and fusion_logger must not import state_machine.
    Checked via module source inspection, not sys.modules, to avoid
    false positives from test session pollution (other test files
    legitimately import state_machine for S1/S3 tests).
    """

    def _get_source(self, module_path: str) -> str:
        return Path(module_path).read_text()

    def test_vio_mode_does_not_import_state_machine(self):
        import core.fusion.vio_mode as vm
        src = self._get_source(vm.__file__)
        # Check for actual import statements, not documentary mentions in docstrings
        assert "import state_machine" not in src,             "vio_mode.py must not import state_machine"
        assert "from core.state_machine" not in src,             "vio_mode.py must not import from core.state_machine"

    def test_fusion_logger_does_not_import_state_machine(self):
        import core.fusion.fusion_logger as fl
        src = self._get_source(fl.__file__)
        assert "import state_machine" not in src,             "fusion_logger.py must not import state_machine"
        assert "from core.state_machine" not in src,             "fusion_logger.py must not import from core.state_machine"

    def test_vio_mode_does_not_import_ekf(self):
        import core.fusion.vio_mode as vm
        src = self._get_source(vm.__file__)
        assert "from core.ekf" not in src,             "vio_mode.py must not import from core.ekf"
        assert "import error_state_ekf" not in src,             "vio_mode.py must not import error_state_ekf"


# ─────────────────────────────────────────────────────────────────────────────
# G-08: current_mode accessible without importing ESKF internals
# ─────────────────────────────────────────────────────────────────────────────

class TestG08ModeAccessible:
    def test_current_mode_is_string_accessible(self):
        nav = _make_nav()
        mode_str = nav.current_mode.name
        assert mode_str in ("NOMINAL", "OUTAGE", "RESUMPTION")

    def test_mode_transitions_visible_as_string(self):
        nav = _make_nav(threshold_s=0.1)
        assert nav.current_mode.name == "NOMINAL"
        for _ in range(15):
            nav.tick(0.01)
        assert nav.current_mode.name == "OUTAGE"
        nav.on_vio_update(accepted=True, innov_mag=0.05)
        assert nav.current_mode.name == "RESUMPTION"

    def test_in_outage_flag_usable_by_mission_layer(self):
        """Mission layer can suppress position-dependent functions using
        in_outage without any knowledge of ESKF state."""
        nav = _make_nav(threshold_s=0.5)
        assert nav.in_outage is False
        for _ in range(60):
            nav.tick(0.01)
        assert nav.in_outage is True     # mission layer: suppress functions
        nav.on_vio_update(accepted=True, innov_mag=0.05)
        assert nav.in_outage is True     # still suppressed: RESUMPTION
        nav.on_vio_update(accepted=True, innov_mag=0.05)
        assert nav.in_outage is False    # NOMINAL: functions may resume

    def test_summary_statistics_accessible(self):
        nav = _make_nav(threshold_s=0.5, spike_m=1.0, cycles=1)
        # One full outage cycle with spike
        for _ in range(60):
            nav.tick(0.01)
        nav.on_vio_update(accepted=True, innov_mag=2.0)
        nav.on_vio_update(accepted=True, innov_mag=0.05)
        assert nav.n_outage_events == 1
        assert nav.n_spike_alerts == 1
        assert nav.max_dt_since_vio > 0.5
        assert nav.max_drift_envelope_m > 0.0
