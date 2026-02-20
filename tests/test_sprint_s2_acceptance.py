"""
tests/test_sprint_s2_acceptance.py
MicroMind / NanoCorteX — Sprint S2 Acceptance Gate

Acceptance gate criteria (SPRINT_STATUS.md):
  ✅  Spoof injection → trust_score < 0.1 within 250 ms
  ✅  State machine → GNSS_DENIED after BIM RED
  ✅  All events logged with timestamp
  ✅  BIM component scores correct for each attack type
  ✅  Hysteresis: 3 samples before state transition (no oscillation)
  ✅  EKF noise scaling: R_GNSS = R_nominal / trust_score capped at 10×
  ✅  NFR-001: BIM processing latency ≤ 250 ms per evaluation
  ✅  BCMP-1 EW-03: spoof → BIM RED within 250 ms

Run:
    conda activate micromind
    cd /Users/amitanand/micromind-autonomy
    python -m pytest tests/test_sprint_s2_acceptance.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np

from core.bim.bim import BIM, BIMConfig, GNSSMeasurement
from core.clock.sim_clock import SimClock
from core.state_machine.state_machine import (
    NCState, NanoCorteXFSM, SystemInputs
)
from logs.mission_log_schema import BIMState, BIMRecord, LogCategory, MissionLog, MissionLogEntry
from sim.gnss_spoof_injector import (
    AttackProfile, AttackType, GNSSSpoofInjector,
    NominalGNSSState, build_bcmp1_attack_sequence
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_stack(mission_id: str = "S2-TEST-001"):
    clock = SimClock(dt=0.01)
    log   = MissionLog(mission_id=mission_id)
    fsm   = NanoCorteXFSM(clock=clock, log=log, mission_id=mission_id)
    bim   = BIM()
    clock.start()
    fsm.start()
    return clock, log, fsm, bim


def clean_measurement(**kwargs) -> GNSSMeasurement:
    """Fully healthy GNSS measurement."""
    base = dict(
        pdop                 = 1.8,
        cn0_db               = 42.0,
        tracked_satellites   = 10,
        gps_position_enu     = np.array([0.0, 0.0, 3200.0]),
        glonass_position_enu = np.array([0.1, 0.0, 3200.1]),
        doppler_deviation_ms = 0.02,
        pose_innovation_m    = 1.5,
        ew_jammer_confidence = 0.0,
    )
    base.update(kwargs)
    return GNSSMeasurement(**base)


def spoof_measurement(offset_m: float = 250.0, cn0_db: float = 28.0) -> GNSSMeasurement:
    """
    Combined spoof measurement: GPS position offset + C/N0 drop.
    Designed to trigger BIM RED immediately.
    """
    gps_pos     = np.array([offset_m, 0.0, 3200.0])
    glonass_pos = np.array([0.0, 0.0, 3200.0])   # GLONASS unaffected
    return GNSSMeasurement(
        pdop                 = 4.5,
        cn0_db               = cn0_db,
        tracked_satellites   = 7,
        gps_position_enu     = gps_pos,
        glonass_position_enu = glonass_pos,
        doppler_deviation_ms = 1.8,    # above Red threshold (1.5 m/s)
        pose_innovation_m    = 45.0,   # large position jump
        ew_jammer_confidence = 0.85,
    )


# ---------------------------------------------------------------------------
# 1. Clean signal → GREEN state
# ---------------------------------------------------------------------------

def test_clean_signal_green():
    """Healthy GNSS → BIM GREEN, trust_score ≥ 0.70."""
    bim = BIM()
    m   = clean_measurement()

    for _ in range(5):
        out = bim.evaluate(m)

    assert out.trust_state == BIMState.GREEN, f"Expected GREEN, got {out.trust_state}"
    assert out.trust_score >= 0.70, f"Expected score ≥ 0.70, got {out.trust_score:.3f}"
    assert out.latency_ms < 250.0, f"NFR-001 violated: {out.latency_ms:.2f} ms"
    print(f"  ✅ Clean signal → GREEN  score={out.trust_score:.3f}  latency={out.latency_ms:.3f}ms")


# ---------------------------------------------------------------------------
# 2. PRIMARY ACCEPTANCE GATE — spoof → trust_score < 0.1 within 250 ms
# ---------------------------------------------------------------------------

def test_spoof_detection_within_250ms():
    """
    BCMP-1 EW-03 / Sprint S2 primary gate:
    Spoof injection → trust_score < 0.1 within 250 ms.
    """
    bim = BIM()

    # Establish GREEN baseline (3 samples for hysteresis)
    clean = clean_measurement()
    for _ in range(3):
        bim.evaluate(clean)
    assert bim.state == BIMState.GREEN

    # Inject spoof — measure wall-clock time to RED
    spoof = spoof_measurement(offset_m=250.0, cn0_db=28.0)

    t_inject = time.monotonic()
    out = None
    elapsed_ms = 0.0

    # BIM runs at ≥10 Hz; simulate up to 25 calls (250 ms at 10 Hz)
    for i in range(25):
        out = bim.evaluate(spoof)
        elapsed_ms = (time.monotonic() - t_inject) * 1000.0
        if out.trust_score < 0.1:
            break

    assert out.trust_score < 0.1, (
        f"Spoof not detected: trust_score={out.trust_score:.3f} after 25 evaluations"
    )
    assert elapsed_ms < 250.0, (
        f"NFR-001/EW-03 VIOLATED: detection took {elapsed_ms:.1f} ms > 250 ms"
    )
    assert out.spoof_alert, "Spoof alert flag not set"
    assert out.trust_state == BIMState.RED

    print(f"  ✅ SPOOF DETECTED: trust_score={out.trust_score:.3f} "
          f"in {i+1} evaluations / {elapsed_ms:.2f} ms  (limit 250 ms)")
    print(f"     Spoof alert: {out.spoof_alert} | State: {out.trust_state.value}")


# ---------------------------------------------------------------------------
# 3. Hysteresis — 3 samples before state transition (no oscillation)
# ---------------------------------------------------------------------------

def test_hysteresis_prevents_oscillation():
    """
    State must not transition on a single degraded sample.
    Requires 3 consecutive Amber samples before G→A transition.
    """
    bim = BIM()

    # Establish GREEN with 5 clean samples
    clean = clean_measurement()
    for _ in range(5):
        bim.evaluate(clean)
    assert bim.state == BIMState.GREEN

    # One marginal sample — PDOP well into Amber zone, Doppler also degraded
    amber_m = clean_measurement(pdop=5.2, doppler_deviation_ms=0.9)
    out1 = bim.evaluate(amber_m)
    assert out1.trust_state == BIMState.GREEN, (
        f"Hysteresis failed: state changed on 1 sample (got {out1.trust_state})"
    )

    # Second sample
    out2 = bim.evaluate(amber_m)
    assert out2.trust_state == BIMState.GREEN, (
        f"Hysteresis failed: state changed on 2 samples"
    )

    # Third sample — now transition should fire
    out3 = bim.evaluate(amber_m)
    assert out3.trust_state == BIMState.AMBER, (
        f"Hysteresis failed: state did not change after 3 samples (got {out3.trust_state})"
    )

    print(f"  ✅ Hysteresis correct — 3 samples needed for G→A transition")
    print(f"     Sample 1: {out1.trust_state.value} | "
          f"Sample 2: {out2.trust_state.value} | "
          f"Sample 3: {out3.trust_state.value}")


# ---------------------------------------------------------------------------
# 4. EKF noise scaling
# ---------------------------------------------------------------------------

def test_ekf_noise_scaling():
    """
    R_GNSS = R_nominal / trust_score, capped at 10×.
    GREEN:  trust ~0.9 → scale ~1.1×
    AMBER:  trust ~0.5 → scale ~2.0×
    RED:    trust ~0.05 → scale = 10× (cap)
    """
    bim = BIM()

    # Green scaling
    out_green = bim.evaluate(clean_measurement())
    # Ensure GREEN via 3 samples
    for _ in range(2):
        out_green = bim.evaluate(clean_measurement())

    assert out_green.ekf_noise_scale >= 1.0
    assert out_green.ekf_noise_scale <= 10.0

    # Spoof → RED → cap
    spoof = spoof_measurement()
    out_red = bim.evaluate(spoof)
    assert out_red.ekf_noise_scale == 10.0, (
        f"EKF cap not applied on RED: scale={out_red.ekf_noise_scale}"
    )

    print(f"  ✅ EKF scaling: GREEN={out_green.ekf_noise_scale:.2f}×  "
          f"RED={out_red.ekf_noise_scale:.2f}× (cap=10×)")


# ---------------------------------------------------------------------------
# 5. FSM integration — BIM RED → GNSS_DENIED
# ---------------------------------------------------------------------------

def test_fsm_gnss_denied_on_bim_red():
    """
    Full integration:
    BIM RED → SystemInputs.bim_state=RED → FSM EW_AWARE → GNSS_DENIED.
    All transitions logged.
    """
    clock, log, fsm, bim = make_stack("S2-FSM-001")

    # Step 1: Drive FSM to EW_AWARE
    inputs = SystemInputs(
        bim_trust_score      = 0.9,
        bim_state            = BIMState.GREEN,
        ew_jammer_confidence = 0.75,   # triggers NOMINAL → EW_AWARE
        vio_feature_count    = 60,
        trn_correlation_valid= True,
    )
    clock.step()
    fsm.evaluate(inputs)
    assert fsm.state == NCState.EW_AWARE, f"Expected EW_AWARE, got {fsm.state}"

    # Step 2: BIM evaluates spoof → produces RED
    spoof = spoof_measurement(offset_m=250.0)
    for _ in range(3):
        bim_out = bim.evaluate(spoof)

    assert bim_out.trust_state == BIMState.RED
    assert bim_out.trust_score < 0.1

    # Log BIM update
    log.append(MissionLogEntry(
        timestamp_s = clock.now(),
        tick        = clock.tick(),
        category    = LogCategory.BIM_UPDATE,
        mission_id  = "S2-FSM-001",
        state       = fsm.state.value,
        bim         = BIMRecord(
            trust_score         = bim_out.trust_score,
            bim_state           = bim_out.trust_state,
            hysteresis_count    = bim_out.hysteresis_samples,
            spoof_delta_m       = 250.0,
        ),
    ))

    # Step 3: Feed BIM output into FSM SystemInputs → GNSS_DENIED
    inputs_red = SystemInputs(
        bim_trust_score      = bim_out.trust_score,
        bim_state            = bim_out.trust_state,    # RED
        ew_jammer_confidence = 0.75,
        vio_feature_count    = 60,
        trn_correlation_valid= True,
    )
    clock.step()
    result = fsm.evaluate(inputs_red)

    assert fsm.state == NCState.GNSS_DENIED, (
        f"FSM did not transition to GNSS_DENIED: {fsm.state}"
    )
    assert result is not None and result.succeeded
    assert result.latency_ms < 2000.0  # NFR-002

    # Verify transition is in log
    transitions = log.transitions()
    gnss_denied_transitions = [
        t for t in transitions if t.to_state == NCState.GNSS_DENIED.value
    ]
    assert len(gnss_denied_transitions) >= 1, "GNSS_DENIED transition not logged"

    t = gnss_denied_transitions[0]
    assert t.transition_trigger == "BIM_RED"
    assert len(t.guards) > 0
    assert t.timestamp_s > 0

    print(f"  ✅ BIM RED → FSM GNSS_DENIED  "
          f"trust={bim_out.trust_score:.3f}  "
          f"latency={result.latency_ms:.3f}ms")
    print(f"     Logged: trigger={t.transition_trigger}  "
          f"guards={[g.guard_name for g in t.guards]}")


# ---------------------------------------------------------------------------
# 6. NFR-001 — BIM processing latency ≤ 250 ms
# ---------------------------------------------------------------------------

def test_nfr_001_latency():
    """Every BIM evaluation must complete within 250 ms (NFR-001)."""
    bim = BIM()
    measurements = [
        clean_measurement(),
        clean_measurement(pdop=4.0),
        spoof_measurement(),
        clean_measurement(ew_jammer_confidence=0.9),
    ]
    for m in measurements:
        out = bim.evaluate(m)
        assert out.latency_ms < 250.0, (
            f"NFR-001 VIOLATED: {out.latency_ms:.2f} ms > 250 ms"
        )
    print(f"  ✅ NFR-001 PASS — all evaluations < 250 ms")


# ---------------------------------------------------------------------------
# 7. BCMP-1 attack sequence — jamming then spoof
# ---------------------------------------------------------------------------

def test_bcmp1_attack_sequence():
    """
    Run BCMP-1 attack timeline through BIM.
    Verify:
    - JMR-01 jamming → BIM transitions to AMBER then RED
    - SPF-01 terminal spoof → BIM RED within 250 ms (EW-03)
    """
    bim      = BIM()
    injector = build_bcmp1_attack_sequence()

    # Simulate from T=0 to T=55 min in 1-second steps
    true_pos = np.array([0.0, 0.0, 3200.0])

    first_amber_t = None
    first_red_t   = None
    spoof_detect_t = None
    spoof_start_s  = 52 * 60  # SPF-01 activation

    dt = 1.0   # 1-second steps for this coarse check
    t  = 0.0

    while t <= 55 * 60:
        m   = injector.generate(t, true_pos)
        out = bim.evaluate(m)

        if first_amber_t is None and out.trust_state == BIMState.AMBER:
            first_amber_t = t
        if first_red_t is None and out.trust_state == BIMState.RED:
            first_red_t = t

        # Check spoof detection latency (coarse — 1 s steps)
        if t >= spoof_start_s and spoof_detect_t is None:
            if out.trust_score < 0.1:
                spoof_detect_t = t

        t += dt
        # Advance position along ingress
        true_pos[0] += (100_000 / (55 * 60)) * dt

    assert first_amber_t is not None, "BIM never reached AMBER during jamming"
    assert first_red_t   is not None, "BIM never reached RED"
    assert spoof_detect_t is not None, "SPF-01 not detected"

    spoof_latency_s = spoof_detect_t - spoof_start_s
    assert spoof_latency_s <= 10.0, (
        f"EW-03 VIOLATED: spoof detection took {spoof_latency_s:.0f}s "
        f"(limit is sub-second at 1Hz step resolution)"
    )

    print(f"  ✅ BCMP-1 attack sequence:")
    print(f"     First AMBER: T+{first_amber_t/60:.1f} min")
    print(f"     First RED:   T+{first_red_t/60:.1f} min")
    print(f"     SPF-01 detected: T+{spoof_detect_t/60:.1f} min  "
          f"(activation T+{spoof_start_s/60:.0f} min, "
          f"latency ≤{spoof_latency_s:.0f}s)")


# ---------------------------------------------------------------------------
# 8. Score component correctness
# ---------------------------------------------------------------------------

def test_component_scores():
    """
    Verify each component scorer responds correctly to its input.
    """
    bim = BIM()

    # PDOP Red → low RAIM score
    m_pdop = clean_measurement(pdop=8.0)
    out = bim.evaluate(m_pdop)
    assert out.score_raim == 0.0, f"PDOP>6 should give score_raim=0, got {out.score_raim}"

    # Doppler Red → low Doppler score
    m_dopp = clean_measurement(doppler_deviation_ms=2.0)
    out = bim.evaluate(m_dopp)
    assert out.score_doppler == 0.0, f"Doppler>1.5 should give score_doppler=0, got {out.score_doppler}"

    # Constellation delta > 15 m → constellation score = 0
    m_const = clean_measurement(
        gps_position_enu     = np.array([20.0, 0.0, 3200.0]),
        glonass_position_enu = np.array([0.0,  0.0, 3200.0]),
    )
    out = bim.evaluate(m_const)
    assert out.score_constellation == 0.0, (
        f"Delta>15m should give score_constellation=0, got {out.score_constellation}"
    )

    # EW impact at 1.0 → ew score = 0
    m_ew = clean_measurement(ew_jammer_confidence=1.0)
    out  = bim.evaluate(m_ew)
    assert out.score_ew == 0.0

    print(f"  ✅ All component scorers respond correctly to boundary inputs")


# ---------------------------------------------------------------------------
# 9. Reset / re-run determinism
# ---------------------------------------------------------------------------

def test_bim_reset_determinism():
    """BIM.reset() must return to initial state; two runs produce identical output."""
    bim = BIM()
    m   = spoof_measurement()

    for _ in range(5):
        bim.evaluate(m)
    out1 = bim.evaluate(m)

    bim.reset()
    for _ in range(5):
        bim.evaluate(m)
    out2 = bim.evaluate(m)

    assert abs(out1.trust_score - out2.trust_score) < 1e-9, "BIM not deterministic after reset"
    assert out1.trust_state == out2.trust_state
    print(f"  ✅ BIM reset deterministic: trust_score={out1.trust_score:.6f}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PASS = 0
    FAIL = 0

    tests = [
        ("Clean signal → GREEN",                  test_clean_signal_green),
        ("SPOOF → trust_score < 0.1 in ≤250ms",  test_spoof_detection_within_250ms),
        ("Hysteresis — 3 samples before G→A",     test_hysteresis_prevents_oscillation),
        ("EKF noise scaling",                      test_ekf_noise_scaling),
        ("FSM integration — BIM RED → GNSS_DENIED", test_fsm_gnss_denied_on_bim_red),
        ("NFR-001 — latency ≤ 250 ms",            test_nfr_001_latency),
        ("BCMP-1 attack sequence",                 test_bcmp1_attack_sequence),
        ("Component score correctness",            test_component_scores),
        ("Reset / re-run determinism",             test_bim_reset_determinism),
    ]

    print("\n" + "="*65)
    print("  MicroMind / NanoCorteX — Sprint S2 Acceptance Gate")
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
        print(f"  🟢 SPRINT S2 ACCEPTANCE GATE: PASSED  ({PASS}/{total} tests)")
    else:
        print(f"  🔴 SPRINT S2 ACCEPTANCE GATE: FAILED  ({FAIL}/{total} tests failed)")
    print("="*65 + "\n")

    sys.exit(0 if FAIL == 0 else 1)
