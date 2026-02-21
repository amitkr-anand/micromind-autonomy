"""
sim/nav_scenario.py
MicroMind / NanoCorteX — Navigation Scenario Simulation
Sprint S3 Deliverable 2 of 3

Simulates a 50 km straight-corridor flight at constant speed with:
    • Phase 1 (0–20 km)    : GNSS PRIMARY — clean signal, BIM GREEN
    • Phase 2 (20–45 km)   : GNSS DENIED  — signal jammed, BIM → RED,
                             FSM → GNSS_DENIED, TRN corrections active
    • Phase 3 (45–50 km)   : GNSS RECOVERY — signal restored, BIM → GREEN,
                             FSM holds GNSS_DENIED (design: stays until
                             explicit operator recovery command)

Acceptance criteria validated by this scenario (Sprint S3 gate):
    • FSM transitions NOMINAL → EW_AWARE → GNSS_DENIED on GNSS loss
    • TRN corrections delivered ≥ 1 per 2 km (NAV-01)
    • Drift over 5 km GNSS-denied segment < 2 % = 100 m (FR-107)
    • Navigation mode logged correctly at each phase transition

Output:
    NavScenarioResult  — structured record consumable by dashboard & tests
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.bim.bim import BIM, GNSSMeasurement, TrustState
from core.clock.sim_clock import SimClock
from core.ins.trn_stub import DEMProvider, INSState, RadarAltimeterSim, TRNStub
from core.state_machine.state_machine import (
    NanoCorteXFSM, NCState, SystemInputs,
)
from logs.mission_log_schema import BIMState, MissionLog
from sim.gnss_spoof_injector import GNSSSpoofInjector, SpoofConfig, SpoofProfile


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

TOTAL_DISTANCE_M    = 50_000.0   # 50 km corridor
AIRSPEED_MS         = 50.0       # 180 km/h — representative UAV speed
HEADING_DEG         = 45.0       # north-east heading for varied terrain coverage
DT                  = 1.0        # simulation time step (s)
GNSS_LOSS_START_M   = 20_000.0   # ground track where GNSS jamming begins
GNSS_LOSS_END_M     = 45_000.0   # ground track where GNSS recovers
DRIFT_GATE_START_M  = 20_000.0   # 5 km segment start for FR-107 check
DRIFT_GATE_END_M    = 25_000.0   # 5 km segment end
FR107_DRIFT_LIMIT_M = 100.0      # 2 % of 5 000 m


# ---------------------------------------------------------------------------
# Snapshot logged every tick
# ---------------------------------------------------------------------------

@dataclass
class NavTick:
    """One simulation tick record (all SI units)."""
    time_s:          float
    ground_track_m:  float
    # True state
    true_north_m:    float
    true_east_m:     float
    # INS estimate
    ins_north_m:     float
    ins_east_m:      float
    # Navigation error
    drift_m:         float
    drift_pct:       float         # drift / ground_track_m × 100
    # BIM
    bim_trust:       float
    bim_state:       str
    # FSM
    fsm_state:       str
    # TRN
    trn_ncc_score:   float
    trn_corrections: int
    # GNSS
    gnss_active:     bool
    nav_mode:        str           # "GNSS_PRIMARY" | "VIO_TRN"


# ---------------------------------------------------------------------------
# Scenario result
# ---------------------------------------------------------------------------

@dataclass
class NavScenarioResult:
    """Full result of running the navigation scenario."""
    ticks:                  List[NavTick]
    fsm_transitions:        list          # list[TransitionResult]
    trn_corrections:        list          # list[TRNCorrection]
    gnss_loss_start_t:      Optional[float]
    gnss_loss_end_t:        Optional[float]
    drift_at_5km_gate_m:    float         # FR-107 compliance value
    fr107_pass:             bool
    nav01_pass:             bool          # ≥1 TRN correction per 2 km denied
    total_time_s:           float
    mission_id:             str

    @property
    def ticks_np(self) -> dict:
        """Return tick data as numpy arrays (for plotting)."""
        return {
            "time_s":          np.array([t.time_s          for t in self.ticks]),
            "ground_track_m":  np.array([t.ground_track_m  for t in self.ticks]),
            "true_north_m":    np.array([t.true_north_m    for t in self.ticks]),
            "true_east_m":     np.array([t.true_east_m     for t in self.ticks]),
            "ins_north_m":     np.array([t.ins_north_m     for t in self.ticks]),
            "ins_east_m":      np.array([t.ins_east_m      for t in self.ticks]),
            "drift_m":         np.array([t.drift_m         for t in self.ticks]),
            "drift_pct":       np.array([t.drift_pct       for t in self.ticks]),
            "bim_trust":       np.array([t.bim_trust       for t in self.ticks]),
            "bim_state":       np.array([t.bim_state       for t in self.ticks]),
            "fsm_state":       np.array([t.fsm_state       for t in self.ticks]),
            "trn_ncc_score":   np.array([t.trn_ncc_score   for t in self.ticks]),
            "trn_corrections": np.array([t.trn_corrections for t in self.ticks]),
            "gnss_active":     np.array([t.gnss_active     for t in self.ticks]),
            "nav_mode":        np.array([t.nav_mode        for t in self.ticks]),
        }


# ---------------------------------------------------------------------------
# INS propagation (simple dead-reckoning for SIL)
# ---------------------------------------------------------------------------

def _propagate_ins(ins: INSState, dt: float, noise_std_m: float = 2.0) -> None:
    """
    Propagate INS estimate with dead-reckoning and additive drift noise.
    Uses global numpy random state (seeded at scenario start).
    """
    ins.north_m += ins.vn * dt + np.random.normal(0, noise_std_m)
    ins.east_m  += ins.ve * dt + np.random.normal(0, noise_std_m)


# ---------------------------------------------------------------------------
# GNSS measurement builder
# ---------------------------------------------------------------------------

def _make_gnss_measurement(
    true_north_m: float,
    true_east_m:  float,
    spoof_injector: GNSSSpoofInjector,
    ground_track_m: float,
    gnss_active: bool,
) -> GNSSMeasurement:
    """Build a GNSSMeasurement tick for the BIM."""
    rng = np.random.default_rng(int(ground_track_m) % 2**31)

    if gnss_active:
        return GNSSMeasurement(
            pdop                = 1.2 + rng.uniform(0, 0.3),
            constellation_count = 3,
            doppler_deviation_ms= rng.uniform(0, 0.05),
            cn0_db              = 42.0 + rng.normal(0, 1.0),
            cn0_nominal_db      = 45.0,
            pose_innovation_m   = rng.uniform(0, 1.5),
            ew_alert_active     = False,
            ew_confidence       = 0.0,
        )
    else:
        return GNSSMeasurement(
            pdop                = 8.0 + rng.uniform(0, 4.0),
            constellation_count = 1,
            doppler_deviation_ms= rng.uniform(0.3, 0.8),
            gps_glonass_delta_m = 60.0 + rng.uniform(0, 40.0),
            cn0_db              = 22.0 + rng.normal(0, 3.0),   # ~20 dB drop under jamming
            cn0_nominal_db      = 45.0,
            pose_innovation_m   = rng.uniform(40.0, 120.0),
            ew_alert_active     = True,
            ew_confidence       = 0.85,
        )


# ---------------------------------------------------------------------------
# Main scenario runner
# ---------------------------------------------------------------------------

def run_nav_scenario(
    mission_id: str = "S3-NAV-50K-001",
    seed: int = 42,
    verbose: bool = False,
) -> NavScenarioResult:
    """
    Run the full 50 km navigation scenario.

    Returns a NavScenarioResult containing all telemetry, FSM transitions,
    TRN corrections and acceptance-gate values.
    """
    np.random.seed(seed)

    # --- Infrastructure
    clock  = SimClock(dt=DT, start_time=0.0)
    log    = MissionLog(mission_id=mission_id)
    clock.start()

    # --- Navigation subsystems
    dem   = DEMProvider(seed=7)
    radar = RadarAltimeterSim(dem, seed=99)
    trn   = TRNStub(dem, radar)

    # --- BIM & FSM
    bim = BIM()
    fsm = NanoCorteXFSM(clock=clock, log=log, mission_id=mission_id)
    fsm.start()

    # --- GNSS spoof injector (JAMMING_ONLY for GNSS-denied phase simulation)
    spoof_cfg = SpoofConfig(profile=SpoofProfile.JAMMING_ONLY)
    spoof = GNSSSpoofInjector(config=spoof_cfg, seed=seed)

    # --- Heading decomposition
    heading_rad = math.radians(HEADING_DEG)
    vn = AIRSPEED_MS * math.cos(heading_rad)   # northward component
    ve = AIRSPEED_MS * math.sin(heading_rad)   # eastward component

    # --- State
    true_north = 0.0
    true_east  = 0.0
    ins = INSState(north_m=0.0, east_m=0.0, vn=vn, ve=ve)

    ground_track_m   = 0.0
    ticks: List[NavTick] = []
    fsm_transitions  = []
    gnss_loss_start_t: Optional[float] = None
    gnss_loss_end_t:   Optional[float] = None
    drift_at_5km_gate  = 0.0
    nav_mode = "GNSS_PRIMARY"

    # -----------------------------------------------------------------------
    # Simulation loop
    # -----------------------------------------------------------------------

    while ground_track_m <= TOTAL_DISTANCE_M:
        t = clock.now()
        gnss_active = not (GNSS_LOSS_START_M <= ground_track_m < GNSS_LOSS_END_M)

        # --- Mark phase transitions
        if not gnss_active and gnss_loss_start_t is None:
            gnss_loss_start_t = t
            if verbose:
                print(f"  [T={t:.0f}s] GNSS LOST at {ground_track_m/1000:.1f} km")

        if gnss_active and gnss_loss_start_t is not None and gnss_loss_end_t is None:
            gnss_loss_end_t = t
            if verbose:
                print(f"  [T={t:.0f}s] GNSS RECOVERED at {ground_track_m/1000:.1f} km")

        # --- GNSS measurement → BIM
        meas = _make_gnss_measurement(true_north, true_east,
                                      spoof, ground_track_m, gnss_active)
        bim_output = bim.evaluate(meas)
        bim_state_fsm = BIMState[bim_output.trust_state.value]

        # --- TRN update (only in GNSS-denied phase)
        trn_corr = None
        if not gnss_active:
            nav_mode = "VIO_TRN"
            _propagate_ins(ins, DT, noise_std_m=1.5)   # tactical-grade IMU drift
            trn_corr = trn.update(
                ins           = ins,
                true_north_m  = true_north,
                true_east_m   = true_east,
                dt            = DT,
                ground_track_m= ground_track_m,
                timestamp_s   = t,
            )
        else:
            nav_mode = "GNSS_PRIMARY"
            # GNSS-aided INS — position error bounded by GNSS
            ins.north_m = true_north + np.random.normal(0, 1.5)
            ins.east_m  = true_east  + np.random.normal(0, 1.5)

        # --- FSM inputs
        sys_inputs = SystemInputs(
            bim_trust_score       = bim_output.trust_score,
            bim_state             = bim_state_fsm,
            ew_jammer_confidence  = 0.8 if not gnss_active else 0.0,
            vio_feature_count     = 120,
            trn_correlation_valid = trn.trn_correlation_valid,
            terminal_zone_entered = False,
        )

        result = fsm.evaluate(sys_inputs)
        if result:
            fsm_transitions.append(result)
            if verbose:
                print(f"  [T={t:.0f}s] FSM {result.from_state.value} → "
                      f"{result.to_state.value}  trigger={result.trigger}")

        # --- Position error
        drift = math.hypot(ins.north_m - true_north, ins.east_m - true_east)
        drift_pct = (drift / max(ground_track_m, 1.0)) * 100.0

        # --- FR-107 gate: drift at end of first 5 km GNSS-denied segment
        if abs(ground_track_m - DRIFT_GATE_END_M) < AIRSPEED_MS:
            drift_at_5km_gate = drift

        # --- Log tick
        ticks.append(NavTick(
            time_s         = t,
            ground_track_m = ground_track_m,
            true_north_m   = true_north,
            true_east_m    = true_east,
            ins_north_m    = ins.north_m,
            ins_east_m     = ins.east_m,
            drift_m        = drift,
            drift_pct      = drift_pct,
            bim_trust      = bim_output.trust_score,
            bim_state      = bim_output.trust_state.value,
            fsm_state      = fsm.state.value,
            trn_ncc_score  = trn.last_ncc_score,
            trn_corrections= trn.correction_count,
            gnss_active    = gnss_active,
            nav_mode       = nav_mode,
        ))

        # --- Advance truth state
        true_north     += vn * DT
        true_east      += ve * DT
        ground_track_m += AIRSPEED_MS * DT
        clock.step()

    fsm.log_mission_end(notes="50 km corridor scenario complete")

    # -----------------------------------------------------------------------
    # Acceptance gate values
    # -----------------------------------------------------------------------

    denied_km = (GNSS_LOSS_END_M - GNSS_LOSS_START_M) / 1000.0
    expected_corrections = int(denied_km / 2.0)   # NAV-01: 1 per 2 km
    accepted_corrections = sum(1 for c in trn.corrections if c.accepted)

    return NavScenarioResult(
        ticks              = ticks,
        fsm_transitions    = fsm_transitions,
        trn_corrections    = trn.corrections,
        gnss_loss_start_t  = gnss_loss_start_t,
        gnss_loss_end_t    = gnss_loss_end_t,
        drift_at_5km_gate_m= drift_at_5km_gate,
        fr107_pass         = drift_at_5km_gate <= FR107_DRIFT_LIMIT_M,
        nav01_pass         = accepted_corrections >= expected_corrections,
        total_time_s       = clock.now(),
        mission_id         = mission_id,
    )


# ---------------------------------------------------------------------------
# Quick-run entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MicroMind S3 — 50 km Navigation Scenario")
    print("=" * 60)
    t0 = time.perf_counter()
    result = run_nav_scenario(verbose=True)
    elapsed = time.perf_counter() - t0

    print(f"\n{'─'*60}")
    print(f"  Mission ID      : {result.mission_id}")
    print(f"  Total sim time  : {result.total_time_s:.0f} s  ({result.total_time_s/60:.1f} min)")
    print(f"  Wall time       : {elapsed:.2f} s")
    print(f"  FSM transitions : {len(result.fsm_transitions)}")
    for t in result.fsm_transitions:
        print(f"    {t.from_state.value:15s} → {t.to_state.value:15s}  [{t.trigger}]")
    print(f"  TRN corrections : {len(result.trn_corrections)} total, "
          f"{sum(1 for c in result.trn_corrections if c.accepted)} accepted")
    print(f"  Drift @ 5 km    : {result.drift_at_5km_gate_m:.1f} m  "
          f"(limit {100.0} m)  {'PASS ✓' if result.fr107_pass else 'FAIL ✗'}")
    print(f"  FR-107          : {'PASS ✓' if result.fr107_pass else 'FAIL ✗'}")
    print(f"  NAV-01          : {'PASS ✓' if result.nav01_pass else 'FAIL ✗'}")
    print(f"{'─'*60}")
