"""
tests/test_sprint_s1_acceptance.py
MicroMind / NanoCorteX — Sprint S1 Acceptance Gate

Acceptance gate criteria (SPRINT_STATUS.md):
  ✅  State machine runs through all 7 states without error
  ✅  All transitions logged with timestamp and guard evaluation result
  ✅  NFR-002: all transition latencies ≤ 2000 ms
  ✅  NFR-013: log completeness ≥ 99%
  ✅  SimClock monotonic and deterministic
  ✅  BCMP-1 scenario instantiates cleanly; all 11 criteria present

Run:
    conda activate micromind
    cd /Users/amitanand/micromind-autonomy
    python -m pytest tests/test_sprint_s1_acceptance.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

from core.clock.sim_clock import SimClock
from core.state_machine.state_machine import (
    NCState,
    NanoCorteXFSM,
    SystemInputs,
)
from logs.mission_log_schema import (
    BIMState,
    LogCategory,
    MissionLog,
    MissionLogEntry,
    NavigationRecord,
    NavMode,
    BIMRecord,
    EWObservation,
)
from scenarios.bcmp1.bcmp1_scenario import BCMP1, PassCriteriaID, BCMP1Scenario


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fsm(mission_id: str = "S1-TEST-001"):
    clock = SimClock(dt=0.01)
    log   = MissionLog(mission_id=mission_id)
    fsm   = NanoCorteXFSM(clock=clock, log=log, mission_id=mission_id)
    clock.start()
    fsm.start()
    return clock, log, fsm


def nominal_inputs(**kwargs) -> SystemInputs:
    """Fully nominal system inputs — override specific fields via kwargs."""
    base = SystemInputs(
        bim_trust_score       = 1.0,
        bim_state             = BIMState.GREEN,
        bim_green_sample_count = 0,
        ew_jammer_confidence  = 0.0,
        ew_cost_map_active    = False,
        vio_feature_count     = 80,
        trn_correlation_valid = True,
        terminal_zone_entered = False,
        l10s_active           = False,
        eo_lock_confidence    = 0.0,
        key_mismatch          = False,
        tamper_detected       = False,
        corridor_violation    = False,
        l10s_abort_commanded  = False,
    )
    for k, v in kwargs.items():
        setattr(base, k, v)
    return base


# ---------------------------------------------------------------------------
# 1. SimClock tests
# ---------------------------------------------------------------------------

def test_clock_monotonic():
    """Clock time must never go backward."""
    clock = SimClock(dt=0.01)
    clock.start()
    times = []
    for _ in range(100):
        clock.step()
        times.append(clock.now())
    for i in range(1, len(times)):
        assert times[i] > times[i-1], f"Clock went backward at step {i}"
    print("  ✅ Clock monotonic over 100 steps")


def test_clock_deterministic():
    """Same dt sequence must produce identical timestamps."""
    def run_clock():
        c = SimClock(dt=0.01, start_time=0.0)
        c.start()
        for _ in range(50):
            c.step()
        return c.now()
    assert run_clock() == run_clock(), "Clock not deterministic"
    print("  ✅ Clock deterministic")


def test_clock_step_to():
    clock = SimClock(dt=0.01)
    clock.start()
    steps = clock.step_to(1.0, label="TARGET_REACHED")
    assert clock.now() >= 1.0, "step_to did not advance to target"
    assert steps > 0
    print(f"  ✅ step_to(1.0s) completed in {steps} steps, now={clock.now():.3f}s")


# ---------------------------------------------------------------------------
# 2. Full 7-state traversal (S1 acceptance gate — PRIMARY)
# ---------------------------------------------------------------------------

def test_all_7_states_traversed():
    """
    Drive the FSM through all 7 states in the canonical BCMP-1 sequence:
    NOMINAL → EW_AWARE → GNSS_DENIED → SILENT_INGRESS → SHM_ACTIVE → ABORT
    Then separately verify MISSION_FREEZE from NOMINAL.
    """
    clock, log, fsm = make_fsm("S1-TRAVERSAL-001")

    states_visited = set()
    states_visited.add(fsm.state)

    def tick(inputs):
        clock.step()
        return fsm.evaluate(inputs)

    # --- NOMINAL (start state) ---
    assert fsm.state == NCState.NOMINAL
    states_visited.add(NCState.NOMINAL)

    # --- NOMINAL → EW_AWARE ---
    result = tick(nominal_inputs(ew_jammer_confidence=0.75))
    assert fsm.state == NCState.EW_AWARE, f"Expected EW_AWARE, got {fsm.state}"
    assert result is not None and result.succeeded
    states_visited.add(NCState.EW_AWARE)
    print(f"  ✅ NOMINAL → EW_AWARE  (trigger={result.trigger}, latency={result.latency_ms:.3f}ms)")

    # Feed 0 Green samples to reset counter, keep jammer confidence high
    # Then drive BIM to RED → GNSS_DENIED
    for _ in range(3):
        tick(nominal_inputs(
            ew_jammer_confidence=0.75,
            bim_state=BIMState.AMBER,
            bim_trust_score=0.5,
        ))

    result = tick(nominal_inputs(
        ew_jammer_confidence=0.75,
        bim_state=BIMState.RED,
        bim_trust_score=0.1,
        vio_feature_count=60,
    ))
    assert fsm.state == NCState.GNSS_DENIED, f"Expected GNSS_DENIED, got {fsm.state}"
    states_visited.add(NCState.GNSS_DENIED)
    print(f"  ✅ EW_AWARE → GNSS_DENIED  (trigger={result.trigger}, latency={result.latency_ms:.3f}ms)")

    # --- GNSS_DENIED → SILENT_INGRESS ---
    result = tick(nominal_inputs(
        bim_state=BIMState.RED,
        bim_trust_score=0.05,
        terminal_zone_entered=True,
        vio_feature_count=60,
    ))
    assert fsm.state == NCState.SILENT_INGRESS, f"Expected SILENT_INGRESS, got {fsm.state}"
    states_visited.add(NCState.SILENT_INGRESS)
    print(f"  ✅ GNSS_DENIED → SILENT_INGRESS  (trigger={result.trigger}, latency={result.latency_ms:.3f}ms)")

    # --- SILENT_INGRESS → SHM_ACTIVE ---
    result = tick(nominal_inputs(
        bim_state=BIMState.RED,
        bim_trust_score=0.05,
        terminal_zone_entered=True,
        l10s_active=True,
        eo_lock_confidence=0.92,
    ))
    assert fsm.state == NCState.SHM_ACTIVE, f"Expected SHM_ACTIVE, got {fsm.state}"
    states_visited.add(NCState.SHM_ACTIVE)
    print(f"  ✅ SILENT_INGRESS → SHM_ACTIVE  (trigger={result.trigger}, latency={result.latency_ms:.3f}ms)")

    # --- SHM_ACTIVE → ABORT (EO lock loss) ---
    result = tick(nominal_inputs(
        bim_state=BIMState.RED,
        l10s_active=True,
        eo_lock_confidence=0.1,  # below 0.3 threshold
    ))
    assert fsm.state == NCState.ABORT, f"Expected ABORT, got {fsm.state}"
    states_visited.add(NCState.ABORT)
    print(f"  ✅ SHM_ACTIVE → ABORT  (trigger={result.trigger}, latency={result.latency_ms:.3f}ms)")

    # --- MISSION_FREEZE (separate FSM run from NOMINAL) ---
    clock2, log2, fsm2 = make_fsm("S1-FREEZE-001")
    result = fsm2.evaluate(nominal_inputs(key_mismatch=True))
    assert fsm2.state == NCState.MISSION_FREEZE
    states_visited.add(NCState.MISSION_FREEZE)
    print(f"  ✅ ANY → MISSION_FREEZE  (trigger={result.trigger}, latency={result.latency_ms:.3f}ms)")

    # Verify all 7 states visited
    all_states = set(NCState)
    assert states_visited == all_states, (
        f"Not all states visited. Missing: {all_states - states_visited}"
    )
    print(f"\n  ✅ ALL 7 STATES TRAVERSED: {[s.value for s in states_visited]}")

    # Also verify NOMINAL → EW_AWARE → NOMINAL recovery path
    clock3, log3, fsm3 = make_fsm("S1-RECOVERY-001")
    fsm3.evaluate(nominal_inputs(ew_jammer_confidence=0.75))
    assert fsm3.state == NCState.EW_AWARE
    # Feed 3 consecutive Green samples with low jammer confidence
    for i in range(3):
        fsm3.evaluate(nominal_inputs(
            ew_jammer_confidence=0.3,
            bim_state=BIMState.GREEN,
            bim_trust_score=0.95,
        ))
        clock3.step()
    result = fsm3.evaluate(nominal_inputs(
        ew_jammer_confidence=0.3,
        bim_state=BIMState.GREEN,
        bim_trust_score=0.95,
    ))
    assert fsm3.state == NCState.NOMINAL, f"Recovery to NOMINAL failed: {fsm3.state}"
    print("  ✅ EW_AWARE → NOMINAL recovery path confirmed")



# ---------------------------------------------------------------------------
# 3. Transition logging (S1 acceptance gate — MANDATORY)
# ---------------------------------------------------------------------------

def test_transitions_logged_with_guards():
    """Every transition must be logged with timestamp, trigger, guard results."""
    clock, log, fsm = make_fsm("S1-LOG-001")

    # Drive through NOMINAL → EW_AWARE → GNSS_DENIED
    clock.step()
    fsm.evaluate(nominal_inputs(ew_jammer_confidence=0.75))
    clock.step()
    fsm.evaluate(nominal_inputs(
        ew_jammer_confidence=0.75,
        bim_state=BIMState.RED,
        bim_trust_score=0.1,
    ))

    transitions = log.transitions()
    assert len(transitions) >= 2, f"Expected ≥2 transitions, got {len(transitions)}"

    for entry in transitions:
        assert entry.category == LogCategory.STATE_TRANSITION
        assert entry.timestamp_s > 0, "Transition timestamp not set"
        assert entry.from_state is not None, "from_state not logged"
        assert entry.to_state is not None, "to_state not logged"
        assert entry.transition_trigger is not None, "trigger not logged"
        assert entry.transition_latency_ms is not None, "latency not logged"
        assert len(entry.guards) > 0, f"No guard evaluations logged for {entry.to_state}"
        # Each guard must have a name and a result
        for g in entry.guards:
            assert g.guard_name, "Guard missing name"
            assert isinstance(g.guard_result, bool), "Guard result must be bool"

    print(f"  ✅ {len(transitions)} transitions logged with timestamps and guard evaluations")
    for t in transitions:
        guards_str = "; ".join(f"{g.guard_name}={g.guard_result}" for g in t.guards)
        print(f"     {t.from_state} → {t.to_state} | t={t.timestamp_s:.3f}s | guards=[{guards_str}]")


# ---------------------------------------------------------------------------
# 4. NFR-002 timing
# ---------------------------------------------------------------------------

def test_nfr_002_timing():
    """All state transitions must complete in ≤ 2000 ms (NFR-002)."""
    clock, log, fsm = make_fsm("S1-NFR002-001")

    # Drive through full canonical sequence
    inputs_sequence = [
        nominal_inputs(ew_jammer_confidence=0.75),
        nominal_inputs(ew_jammer_confidence=0.75, bim_state=BIMState.RED, bim_trust_score=0.1, vio_feature_count=50),
        nominal_inputs(bim_state=BIMState.RED, bim_trust_score=0.05, terminal_zone_entered=True),
        nominal_inputs(bim_state=BIMState.RED, bim_trust_score=0.05, terminal_zone_entered=True, l10s_active=True, eo_lock_confidence=0.9),
        nominal_inputs(bim_state=BIMState.RED, l10s_active=True, eo_lock_confidence=0.1),
    ]

    for inp in inputs_sequence:
        clock.step()
        fsm.evaluate(inp)

    report = log.transition_timing_report()
    assert report["nfr_002_pass"], (
        f"NFR-002 VIOLATIONS: {report['nfr_002_violations']} transitions exceeded 2000 ms"
    )
    print(f"  ✅ NFR-002 PASS — {report['transition_count']} transitions, "
          f"max latency = {report['max_latency_ms']:.3f} ms, "
          f"mean = {report['mean_latency_ms']:.3f} ms")


# ---------------------------------------------------------------------------
# 5. Log completeness (NFR-013)
# ---------------------------------------------------------------------------

def test_nfr_013_log_completeness():
    """Log completeness must be ≥ 99% (NFR-013)."""
    clock, log, fsm = make_fsm("S1-NFR013-001")

    # Add some enriched entries to simulate real module outputs
    clock.step()
    fsm.evaluate(nominal_inputs(ew_jammer_confidence=0.75))

    # Simulate BIM update entry
    log.append(MissionLogEntry(
        timestamp_s = clock.now(),
        tick        = clock.tick(),
        category    = LogCategory.BIM_UPDATE,
        mission_id  = "S1-NFR013-001",
        state       = NCState.EW_AWARE.value,
        bim         = BIMRecord(
            trust_score        = 0.85,
            bim_state          = BIMState.AMBER,
            pdop               = 2.1,
            constellation_count = 6,
            hysteresis_count   = 1,
        ),
    ))

    # Simulate nav record
    log.append(MissionLogEntry(
        timestamp_s = clock.now(),
        tick        = clock.tick(),
        category    = LogCategory.NAVIGATION,
        mission_id  = "S1-NFR013-001",
        state       = NCState.EW_AWARE.value,
        navigation  = NavigationRecord(
            nav_mode               = NavMode.GNSS_PRIMARY,
            position_enu           = [1000.0, 500.0, 3200.0],
            velocity_enu           = [50.0, 85.0, 0.0],
            position_covariance_m2 = 4.2,
            gnss_trust             = 0.85,
        ),
    ))

    fsm.log_mission_end("Test run complete")

    report = log.completeness_report()
    assert report["nfr_013_pass"], (
        f"NFR-013 FAIL: mean completeness = {report['mean_completeness']:.4f} "
        f"({report['entries_below_99pct']} entries below 99%)"
    )
    print(f"  ✅ NFR-013 PASS — {report['entry_count']} entries, "
          f"mean completeness = {report['mean_completeness']*100:.1f}%")


# ---------------------------------------------------------------------------
# 6. BCMP-1 scenario integrity
# ---------------------------------------------------------------------------

def test_bcmp1_scenario():
    """BCMP-1 scenario must instantiate cleanly with all 11 criteria present."""
    scenario = BCMP1Scenario()

    assert len(scenario.pass_criteria) == 11, (
        f"Expected 11 pass criteria, got {len(scenario.pass_criteria)}"
    )
    assert len(scenario.jammer_nodes) == 3, "Expected 3 jammer/spoofer nodes"
    assert len(scenario.satellite_overpasses) == 1
    assert scenario.decoy_count == 1
    assert scenario.primary_target is not None

    # All 11 criteria IDs present
    for cid in PassCriteriaID:
        assert cid in scenario.pass_criteria, f"Missing criterion {cid}"

    # Timeline sanity checks
    tl = scenario.timeline
    assert tl.gnss_denial_start_s < tl.rf_link_lost_s
    assert tl.rf_link_lost_s < tl.terminal_zone_entry_s
    assert tl.terminal_zone_entry_s < tl.l10s_se_activation_s

    # Route has sufficient waypoints
    assert len(scenario.route) >= 5

    print(f"  ✅ BCMP-1 scenario OK — {len(scenario.pass_criteria)} criteria, "
          f"{len(scenario.jammer_nodes)} jammers, "
          f"{len(scenario.route)} waypoints")
    print()
    print(scenario.summary())


# ---------------------------------------------------------------------------
# 7. MISSION_FREEZE irreversibility
# ---------------------------------------------------------------------------

def test_mission_freeze_is_terminal():
    """MISSION_FREEZE must accept no further transitions."""
    clock, log, fsm = make_fsm("S1-FREEZE-TERMINAL-001")
    fsm.evaluate(nominal_inputs(tamper_detected=True))
    assert fsm.state == NCState.MISSION_FREEZE
    # Any subsequent input should not change state
    for _ in range(5):
        clock.step()
        result = fsm.evaluate(nominal_inputs())
        assert result is None, "MISSION_FREEZE should not transition"
        assert fsm.state == NCState.MISSION_FREEZE
    print("  ✅ MISSION_FREEZE is terminal — no further transitions accepted")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PASS = 0
    FAIL = 0

    tests = [
        ("SimClock — monotonic",           test_clock_monotonic),
        ("SimClock — deterministic",        test_clock_deterministic),
        ("SimClock — step_to",              test_clock_step_to),
        ("FSM — all 7 states traversed",    test_all_7_states_traversed),
        ("FSM — transitions logged",        test_transitions_logged_with_guards),
        ("NFR-002 — transition timing",     test_nfr_002_timing),
        ("NFR-013 — log completeness",      test_nfr_013_log_completeness),
        ("BCMP-1 — scenario integrity",     test_bcmp1_scenario),
        ("MISSION_FREEZE — terminal",       test_mission_freeze_is_terminal),
    ]

    print("\n" + "="*65)
    print("  MicroMind / NanoCorteX — Sprint S1 Acceptance Gate")
    print("="*65)

    for name, fn in tests:
        print(f"\n▶ {name}")
        try:
            fn()
            PASS += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback; traceback.print_exc()
            FAIL += 1

    print("\n" + "="*65)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"  🟢 SPRINT S1 ACCEPTANCE GATE: PASSED  ({PASS}/{total} tests)")
    else:
        print(f"  🔴 SPRINT S1 ACCEPTANCE GATE: FAILED  ({FAIL}/{total} tests failed)")
    print("="*65 + "\n")

    sys.exit(0 if FAIL == 0 else 1)
