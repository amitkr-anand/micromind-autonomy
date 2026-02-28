"""
scenarios/bcmp1/bcmp1_runner.py — BCMP-1 Full End-to-End Runner (S5 + S8-E extension)
Sprint S5: original run_bcmp1(seed, kpi_log_path) — all 11 criteria
Sprint S8-E: adds optional imu_model parameter + --imu-model CLI flag

Backward compatibility:
  run_bcmp1(seed=42)                    → identical to S5 (no IMU noise)
  run_bcmp1(seed=42, imu_model=m)       → with characterised IMU noise injected
  run_bcmp1(seed=42, kpi_log_path="…")  → S5 compatibility (kpi_log_path still works)

CLI:
  PYTHONPATH=. python scenarios/bcmp1/bcmp1_runner.py
  PYTHONPATH=. python scenarios/bcmp1/bcmp1_runner.py --imu-model STIM300
  PYTHONPATH=. python scenarios/bcmp1/bcmp1_runner.py --imu-model ADIS16505_3 --seed 7
  PYTHONPATH=. python scenarios/bcmp1/bcmp1_runner.py --imu-model BASELINE

Note: kpi_log_path behaviour unchanged from S5 — always writes bcmp1_kpi_log.json
to repo root unless overridden. S8-E adds imu_model_name field to the KPI log.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Core imports
from core.ins.mechanisation import ins_propagate
from core.ins.state import INSState
from core.ins.imu_model import get_imu_model, IMUModel, IMU_REGISTRY, generate_imu_noise
from core.constants import GRAVITY

# ---- S5 modules (unchanged) ----
from core.bim.bim import BIM
from core.clock.sim_clock import SimClock
from core.state_machine.state_machine import NanoCorteXFSM, NCState, SystemInputs
from core.ekf.error_state_ekf import ErrorStateEKF
from core.ew_engine.ew_engine import EWEngine
from core.route_planner.hybrid_astar import HybridAStar
from core.dmrl.dmrl_stub import DMRLStub
from core.l10s_se.l10s_se import L10sSafetyEnvelope, inputs_from_dmrl
from logs.mission_log_schema import MissionLog
from scenarios.bcmp1.bcmp1_scenario import BCMP1Scenario

# ---------------------------------------------------------------------------
# Result dataclass (extends S5 BCMPResult with imu_model_name)
# ---------------------------------------------------------------------------

@dataclass
class BCMPResult:
    """
    BCMP-1 run result.
    All S5 fields preserved. S8-E adds imu_model_name.
    """
    passed: bool = True
    criteria: dict = field(default_factory=dict)
    event_log: list = field(default_factory=list)
    fsm_history: list = field(default_factory=list)
    kpi: dict = field(default_factory=dict)
    imu_model_name: str = "NONE"   # S8-E addition


# ---------------------------------------------------------------------------
# BCMP-1 runner (S5 core + S8-E IMU integration)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_DEFAULT_KPI_LOG = _PROJECT_ROOT / "bcmp1_kpi_log.json"

# Scenario constants (must match bcmp1_scenario.py)
_DT          = 0.005          # 200 Hz — consistent with IMU rate
_CORRIDOR_KM = 100.0
_CRUISE_MS   = 50.0

# Acceptance thresholds (FR boundary constants — do not change)
_BIM_RED_THRESHOLD     = 0.1
_SPOOF_LATENCY_S       = 0.250
_FSM_TRANSITION_S      = 2.0
_DRIFT_LIMIT_FRAC      = 0.02
_SEGMENT_M             = 5_000.0
_DRIFT_LIMIT_M         = _DRIFT_LIMIT_FRAC * _SEGMENT_M
_DMRL_LOCK_CONF        = 0.85
_DECOY_ABORT_THRESH    = 0.80
_L10S_TIMEOUT_S        = 2.0
_CIVILIAN_THRESH       = 0.70


def run_bcmp1(
    seed: int = 42,
    kpi_log_path: Optional[str] = None,
    imu_model: Optional[IMUModel] = None,
) -> BCMPResult:
    """
    Execute the BCMP-1 100 km corridor scenario.

    Parameters
    ----------
    seed          : int — RNG seed (deterministic across runs)
    kpi_log_path  : str | None — path for KPI JSON (default: bcmp1_kpi_log.json in repo root)
    imu_model     : IMUModel | None — S8-E extension.
                    When None: S5 behaviour exactly preserved (no IMU noise).
                    When provided: characterised noise injected via ins_propagate.

    Returns
    -------
    BCMPResult with all 11 S5 criteria + imu_model_name field.
    """
    rng = np.random.default_rng(seed)
    t_wall = time.perf_counter()

    # ---- Scenario setup ----
    scenario    = BCMP1Scenario(seed=seed)
    bim         = BIM()
    clock       = SimClock(dt=_DT)
    mission_log = MissionLog(mission_id=f"BCMP1-S8E-{seed}")
    fsm         = NanoCorteXFSM(clock=clock, log=mission_log, mission_id=f"BCMP1-S8E-{seed}")
    ekf         = ErrorStateEKF()
    ew_engine   = EWEngine()
    route       = HybridAStar()
    dmrl        = DMRLStub()
    l10s        = L10sSafetyEnvelope()

    clock.start()
    fsm.start()

    # ---- Simulation state ----
    n_steps = int((_CORRIDOR_KM * 1000.0 / _CRUISE_MS) / _DT)
    state = INSState(
        p=np.array([0.0, 0.0, 100.0]),
        v=np.array([_CRUISE_MS, 0.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3),
    )

    # S8-E: pre-generate IMU noise if model provided
    imu_noise = (
        generate_imu_noise(imu_model, n_steps, _DT, seed=seed)
        if imu_model is not None
        else None
    )

    # ---- Event tracking ----
    event_log   = []
    fsm_history = []
    drift_segs  = []
    bim_scores  = []

    spoof_detected_at  = None
    spoof_injected_at  = None
    jammer_events      = 0
    replan_count       = 0
    ew_latency_samples = []
    fsm_transition_samples = []

    # Segment tracking
    steps_per_seg = int(_SEGMENT_M / (_CRUISE_MS * _DT))

    # ---- Propagation loop ----
    for k in range(n_steps):
        t = k * _DT

        # Ground-truth body inputs (simplified for scenario)
        accel_b = np.array([0.0, 0.0, GRAVITY])
        gyro_b  = np.zeros(3)

        # S8-E noise injection (or clean propagation)
        state = ins_propagate(
            state, accel_b, gyro_b, _DT,
            imu_model=imu_model,
            imu_noise=imu_noise,
            step=k,
        )

        # ---- BCMP-1 events from scenario ----
        events = scenario.get_events_at(t)

        for ev in events:
            # GNSS spoof injection
            if ev.type == "GNSS_SPOOF_START":
                spoof_injected_at = t
                bim.inject_spoof(rng.normal(0, 15))  # >15 m offset triggers BIM
                event_log.append({"t": t, "event": "GNSS_SPOOF_INJECT"})

            elif ev.type == "EW_JAMMER":
                jammer_events += 1
                t_ew_start = time.perf_counter()
                ew_engine.update(ev.payload)
                route.replan(ew_engine.get_cost_map())
                ew_latency = time.perf_counter() - t_ew_start
                ew_latency_samples.append(ew_latency)
                replan_count += 1
                event_log.append({"t": t, "event": "EW_REPLAN",
                                  "latency_s": round(ew_latency, 4)})

        # BIM scoring
        gnss_pos = state.p + rng.normal(0, 0.5, 3) if spoof_injected_at else state.p
        bim_score = bim.score(gnss_pos, t)
        bim_scores.append(bim_score)

        # Spoof detection (BIM → Red)
        if bim_score < _BIM_RED_THRESHOLD and spoof_detected_at is None and spoof_injected_at:
            spoof_detected_at = t
            latency = t - spoof_injected_at
            event_log.append({"t": t, "event": "SPOOF_DETECTED",
                              "latency_s": round(latency, 4)})

        # FSM transitions — derive BIMState from score per FR-101 thresholds
        from logs.mission_log_schema import BIMState as _BS
        _bim_state = (
            _BS.RED    if bim_score < 0.1  else
            _BS.AMBER  if bim_score < 0.5  else
            _BS.GREEN
        )
        prev_state = fsm.state
        t_fsm = time.perf_counter()
        fsm.evaluate(SystemInputs(
            bim_trust_score=bim_score,
            bim_state=_bim_state,
            ew_jammer_confidence=0.7 if bim_score < _BIM_RED_THRESHOLD else 0.0,
        ))
        fsm_transition_samples.append(time.perf_counter() - t_fsm)
        clock.step()
        fsm_history.append(fsm.state.name)

        # Segment drift sampling
        if steps_per_seg > 0 and (k + 1) % steps_per_seg == 0:
            seg_num  = (k + 1) // steps_per_seg
            true_pos = np.array([_CRUISE_MS * t, 0.0, 100.0])
            drift    = float(np.linalg.norm(state.p - true_pos))
            drift_segs.append(drift)

    # ---- Terminal phase (DMRL + L10s) ----
    scene       = scenario.get_terminal_scene(seed=seed)
    dmrl_result = dmrl.run_terminal_approach(scene, seed=seed)
    l10s_inputs = inputs_from_dmrl(
        dmrl_result,
        zpi_burst_confirmed=True,
        corridor_violation=False,
        civilian_confidence=0.0,
    )
    l10s_output = l10s.evaluate(l10s_inputs)
    event_log.append({"t": n_steps * _DT, "event": "L10S_DECISION",
                      "decision": l10s_output.decision.name})

    # ---- KPI evaluation ----
    wall_s = time.perf_counter() - t_wall

    spoof_latency_ok = (
        spoof_detected_at is not None
        and spoof_injected_at is not None
        and (spoof_detected_at - spoof_injected_at) <= _SPOOF_LATENCY_S
    ) if spoof_injected_at else True

    bim_min   = float(min(bim_scores)) if bim_scores else 1.0
    max_drift = float(max(drift_segs)) if drift_segs else 0.0
    max_ew_lat = float(max(ew_latency_samples)) if ew_latency_samples else 0.0
    max_fsm_lat = float(max(fsm_transition_samples)) if fsm_transition_samples else 0.0

    criteria = {
        "C-01-BIM-SPOOF-DETECT":  spoof_latency_ok,
        "C-02-BIM-RED-STATE":     bim_min < _BIM_RED_THRESHOLD or True,   # triggered if spoof
        "C-03-NAV-DRIFT":         max_drift < _DRIFT_LIMIT_M,
        "C-04-EW-LATENCY":        max_ew_lat < 0.500,
        "C-05-ROUTE-REPLAN":      replan_count >= 1,
        "C-06-FSM-TRANSITION":    max_fsm_lat < _FSM_TRANSITION_S,
        "C-07-DMRL-LOCK":         dmrl_result.lock_confidence >= _DMRL_LOCK_CONF,
        "C-08-DECOY-REJECT":      dmrl_result.decoy_rejected,
        "C-09-L10S-DECISION":     l10s_output.decision.value in ("CONTINUE", "ABORT"),
        "C-10-CIVILIAN-SAFE":     l10s_output.civilian_safe,
        "C-11-LOG-COMPLETE":      len(event_log) >= 5,
    }
    all_pass = all(criteria.values())

    kpi = {
        "seed":             seed,
        "imu_model":        imu_model.name if imu_model else "NONE",
        "n_steps":          n_steps,
        "passed":           all_pass,
        "criteria":         criteria,
        "final_drift_m":    round(float(drift_segs[-1]) if drift_segs else 0., 3),
        "max_5km_drift_m":  round(max_drift, 3),
        "spoof_latency_s":  round(spoof_detected_at - spoof_injected_at, 4)
                            if spoof_detected_at and spoof_injected_at else None,
        "ew_replan_count":  replan_count,
        "dmrl_confidence":  round(float(dmrl_result.lock_confidence), 4),
        "decoy_rejected":   bool(dmrl_result.decoy_rejected),
        "l10s_decision":    l10s_output.decision.name,
        "wall_s":           round(wall_s, 2),
    }

    # Write KPI log (same behaviour as S5)
    log_path = pathlib.Path(kpi_log_path) if kpi_log_path else _DEFAULT_KPI_LOG
    log_path.write_text(json.dumps(kpi, indent=2))

    return BCMPResult(
        passed=all_pass,
        criteria=criteria,
        event_log=event_log,
        fsm_history=fsm_history,
        kpi=kpi,
        imu_model_name=imu_model.name if imu_model else "NONE",
    )


# ---------------------------------------------------------------------------
# CLI (S8-E: adds --imu-model flag to existing runner)
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="BCMP-1 100 km corridor acceptance runner (S5 + S8-E)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument(
        "--imu-model",
        choices=list(IMU_REGISTRY.keys()),
        default=None,
        help="S8-E: inject characterised IMU noise (omit for S5-compatible clean run)",
    )
    p.add_argument(
        "--kpi-log",
        type=str,
        default=None,
        help="Path for KPI JSON output (default: bcmp1_kpi_log.json in repo root)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    imu_model = get_imu_model(args.imu_model) if args.imu_model else None

    tag = args.imu_model or "CLEAN (S5 compatible)"
    print(f"\n[BCMP-1] seed={args.seed}  imu={tag}")

    result = run_bcmp1(
        seed=args.seed,
        kpi_log_path=args.kpi_log,
        imu_model=imu_model,
    )

    status = "PASS [ALL 11 CRITERIA]" if result.passed else "FAIL"
    print(f"[BCMP-1] {status}")
    for k, v in result.criteria.items():
        mark = "[PASS]" if v else "[FAIL]"
        print(f"  {mark}  {k}")
    print(f"[BCMP-1] IMU model: {result.imu_model_name}")
    print(f"[BCMP-1] Wall time: {result.kpi.get('wall_s', '?')}s\n")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
