"""
tests/test_s_nep_09.py
======================
S-NEP-09 acceptance gates — unit level.
Validates determinism, boundary conditions, and no regression.
All tests are synthetic (no EuRoC data). Fast and fully isolated.

Gates:
    G-09-01  Threshold precision
    G-09-02  Determinism
    G-09-03  No NOMINAL spike alerts
    G-09-05  Rejection holds OUTAGE
    G-09-06  Repeated cycles stable
    G-09-08  Transition latency bounded

G-09-04 (envelope conservatism) and G-09-07 (no regression) are
validated by the runners — G-09-07 is enforced by running the full
pytest suite at session close.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fusion.vio_mode import (
    VIONavigationMode, VIOMode,
    VIO_OUTAGE_THRESHOLD_S,
    VIO_INNOVATION_SPIKE_THRESHOLD_M,
    VIO_RESUMPTION_CYCLES,
)

IMU_DT = 0.005  # 200 Hz


def make_nav(**kwargs):
    return VIONavigationMode(**kwargs)


def tick_s(nav, seconds, dt=IMU_DT):
    n = int(seconds / dt)
    for _ in range(n):
        nav.tick(dt)


def force_outage(nav):
    """Drive nav into OUTAGE state."""
    tick_s(nav, VIO_OUTAGE_THRESHOLD_S + 0.1)
    assert nav.current_mode is VIOMode.OUTAGE, \
        f"Setup failed: expected OUTAGE, got {nav.current_mode}"


def full_recovery(nav, innov_mag=0.08):
    """Complete OUTAGE→RESUMPTION→NOMINAL cycle."""
    nav.on_vio_update(accepted=True, innov_mag=innov_mag)
    nav.on_vio_update(accepted=True, innov_mag=innov_mag)


# ─────────────────────────────────────────────────────────────────────────────
# G-09-01: Threshold precision
# ─────────────────────────────────────────────────────────────────────────────

class TestG0901ThresholdPrecision:
    def test_below_threshold_stays_nominal(self):
        """1.9s gap — must not trigger OUTAGE."""
        nav = make_nav()
        tick_s(nav, 1.9)
        assert nav.current_mode is VIOMode.NOMINAL

    def test_at_threshold_triggers_outage(self):
        """Gap reaches threshold — OUTAGE must be declared.
        Note: int(threshold/dt) ticks may fall short by ~2e-14 due to fp
        accumulation (400 × 0.005 = 1.9999...998, not 2.0 exactly).
        One additional tick guarantees threshold is crossed. This is a
        recorded behavioural observation: threshold fires at n+1 ticks,
        not n, when threshold / dt is not exactly representable.
        """
        nav = make_nav()
        n = int(VIO_OUTAGE_THRESHOLD_S / IMU_DT) + 1  # +1 for fp accumulation
        for _ in range(n):
            nav.tick(IMU_DT)
        assert nav.current_mode is VIOMode.OUTAGE

    def test_above_threshold_triggers_outage(self):
        """2.1s gap — OUTAGE must be declared."""
        nav = make_nav()
        tick_s(nav, 2.1)
        assert nav.current_mode is VIOMode.OUTAGE

    def test_custom_threshold_respected(self):
        nav = make_nav(outage_threshold_s=0.5)
        tick_s(nav, 0.49)
        assert nav.current_mode is VIOMode.NOMINAL
        tick_s(nav, 0.02)
        assert nav.current_mode is VIOMode.OUTAGE


# ─────────────────────────────────────────────────────────────────────────────
# G-09-02: Determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestG0902Determinism:
    def _mode_sequence(self, outage_s=5.0, n_post=10):
        """Run a fixed scenario and return the mode sequence."""
        nav = make_nav()
        seq = []

        # 5s NOMINAL with VIO
        vio_interval = int((1.0/124.0) / IMU_DT)
        for i in range(int(5.0 / IMU_DT)):
            nav.tick(IMU_DT)
            if i % vio_interval == 0:
                nav.on_vio_update(accepted=True, innov_mag=0.08)
            seq.append(nav.current_mode.name)

        # Outage
        for _ in range(int(outage_s / IMU_DT)):
            nav.tick(IMU_DT)
            seq.append(nav.current_mode.name)

        # Recovery
        for _ in range(n_post):
            nav.on_vio_update(accepted=True, innov_mag=0.08)
            seq.append(nav.current_mode.name)

        return seq

    def test_identical_inputs_produce_identical_sequences(self):
        s1 = self._mode_sequence(outage_s=5.0)
        s2 = self._mode_sequence(outage_s=5.0)
        assert s1 == s2, "Mode sequence not deterministic for identical inputs"

    def test_deterministic_across_three_runs(self):
        seqs = [self._mode_sequence(outage_s=10.0) for _ in range(3)]
        assert seqs[0] == seqs[1] == seqs[2], \
            "Mode sequence differs across 3 identical runs"

    def test_different_outage_durations_produce_different_sequences(self):
        s5  = self._mode_sequence(outage_s=5.0)
        s10 = self._mode_sequence(outage_s=10.0)
        # Lengths differ — sequences must differ
        assert s5 != s10


# ─────────────────────────────────────────────────────────────────────────────
# G-09-03: No NOMINAL spike alerts
# ─────────────────────────────────────────────────────────────────────────────

class TestG0903NoNominalSpikeAlerts:
    def test_no_spike_during_continuous_nominal(self):
        nav = make_nav()
        # 300 VIO updates in NOMINAL — no spike regardless of innov_mag
        for _ in range(300):
            nav.tick(IMU_DT)
            _, spike = nav.on_vio_update(accepted=True, innov_mag=5.0)
            assert spike is False, \
                "Spike alert fired during NOMINAL — must not occur"
        assert nav.n_spike_alerts == 0

    def test_no_spike_on_large_nominal_innov(self):
        """Large innov in NOMINAL (e.g. initial large correction) — no spike."""
        nav = make_nav()
        for _ in range(100):
            _, spike = nav.on_vio_update(accepted=True, innov_mag=10.0)
            assert spike is False


# ─────────────────────────────────────────────────────────────────────────────
# G-09-05: Rejection holds OUTAGE
# ─────────────────────────────────────────────────────────────────────────────

class TestG0905RejectionHoldsOutage:
    def test_rejected_update_does_not_advance_mode(self):
        nav = make_nav()
        force_outage(nav)
        _, spike = nav.on_vio_update(accepted=False, innov_mag=0.5)
        assert nav.current_mode is VIOMode.OUTAGE, \
            f"Mode advanced on rejection: {nav.current_mode}"
        assert spike is False

    def test_multiple_rejections_keep_outage(self):
        nav = make_nav()
        force_outage(nav)
        for _ in range(20):
            nav.on_vio_update(accepted=False, innov_mag=2.0)
        assert nav.current_mode is VIOMode.OUTAGE

    def test_accepted_after_rejections_advances_mode(self):
        nav = make_nav()
        force_outage(nav)
        for _ in range(5):
            nav.on_vio_update(accepted=False, innov_mag=0.5)
        assert nav.current_mode is VIOMode.OUTAGE
        nav.on_vio_update(accepted=True, innov_mag=0.5)
        assert nav.current_mode is VIOMode.RESUMPTION

    def test_no_spike_on_rejection(self):
        nav = make_nav()
        force_outage(nav)
        _, spike = nav.on_vio_update(accepted=False, innov_mag=5.0)
        assert spike is False


# ─────────────────────────────────────────────────────────────────────────────
# G-09-06: Repeated cycles stable
# ─────────────────────────────────────────────────────────────────────────────

class TestG0906RepeatedCyclesStable:
    def test_five_cycles_correct_count(self):
        nav = make_nav()
        for cycle in range(5):
            tick_s(nav, VIO_OUTAGE_THRESHOLD_S + 0.5)
            assert nav.current_mode is VIOMode.OUTAGE
            full_recovery(nav)
            assert nav.current_mode is VIOMode.NOMINAL
        assert nav.n_outage_events == 5

    def test_five_cycles_no_state_accumulation(self):
        """dt_since_vio must reset to 0 after each recovery."""
        nav = make_nav()
        for _ in range(5):
            tick_s(nav, VIO_OUTAGE_THRESHOLD_S + 0.5)
            full_recovery(nav)
            assert nav.current_mode is VIOMode.NOMINAL
            assert nav.dt_since_vio == 0.0, \
                f"dt_since_vio not reset after recovery: {nav.dt_since_vio}"

    def test_ten_cycles_stable(self):
        nav = make_nav()
        for _ in range(10):
            tick_s(nav, VIO_OUTAGE_THRESHOLD_S + 0.1)
            full_recovery(nav)
        assert nav.n_outage_events == 10
        assert nav.current_mode is VIOMode.NOMINAL

    def test_spike_count_matches_expected(self):
        """One spike per outage cycle (assuming innov > threshold)."""
        nav = make_nav(spike_threshold_m=0.5)
        for _ in range(5):
            tick_s(nav, VIO_OUTAGE_THRESHOLD_S + 0.5)
            nav.on_vio_update(accepted=True, innov_mag=2.0)  # spike
            nav.on_vio_update(accepted=True, innov_mag=0.08) # complete
        assert nav.n_spike_alerts == 5


# ─────────────────────────────────────────────────────────────────────────────
# G-09-08: Transition latency bounded (one update cycle)
# ─────────────────────────────────────────────────────────────────────────────

class TestG0908TransitionLatency:
    def test_nominal_to_outage_fires_on_threshold_tick(self):
        """OUTAGE declared in same tick that crosses threshold.
        Fp note: int(threshold/dt) ticks = 1.9999...998 (short by ~2e-14).
        The n+1-th tick crosses threshold and fires OUTAGE in that same tick.
        Confirmed: zero-lag between crossing and transition.
        """
        nav = make_nav()
        n = int(VIO_OUTAGE_THRESHOLD_S / IMU_DT) + 1  # n+1 for fp accumulation
        for i in range(n - 1):
            nav.tick(IMU_DT)
        # One tick before n+1: still NOMINAL
        assert nav.current_mode is VIOMode.NOMINAL
        # The n+1-th tick crosses threshold — OUTAGE fires in same tick
        nav.tick(IMU_DT)
        assert nav.current_mode is VIOMode.OUTAGE, \
            f"OUTAGE not declared in same tick as threshold crossing: {nav.current_mode}"

    def test_outage_to_resumption_fires_on_first_accepted_update(self):
        """RESUMPTION must be entered in the same call as first accepted update."""
        nav = make_nav()
        force_outage(nav)
        assert nav.current_mode is VIOMode.OUTAGE
        nav.on_vio_update(accepted=True, innov_mag=0.08)
        assert nav.current_mode is VIOMode.RESUMPTION, \
            f"RESUMPTION not reached in same call: {nav.current_mode}"

    def test_resumption_to_nominal_fires_on_required_cycle(self):
        """NOMINAL must be entered on the call that completes VIO_RESUMPTION_CYCLES."""
        nav = make_nav(resumption_cycles=1)
        force_outage(nav)
        nav.on_vio_update(accepted=True, innov_mag=0.08)  # → RESUMPTION, count=1
        assert nav.current_mode is VIOMode.RESUMPTION
        # Second call completes cycle
        nav.on_vio_update(accepted=True, innov_mag=0.08)
        assert nav.current_mode is VIOMode.NOMINAL, \
            f"NOMINAL not reached after required cycles: {nav.current_mode}"

    def test_no_delayed_transitions(self):
        """After threshold crossing, no additional ticks should be needed."""
        nav = make_nav()
        # Tick to threshold
        for _ in range(int(VIO_OUTAGE_THRESHOLD_S / IMU_DT) + 1):
            nav.tick(IMU_DT)
        # Must be OUTAGE immediately — no further ticks needed
        assert nav.current_mode is VIOMode.OUTAGE
        # Additional ticks must not change mode (stays OUTAGE)
        for _ in range(10):
            nav.tick(IMU_DT)
        assert nav.current_mode is VIOMode.OUTAGE
