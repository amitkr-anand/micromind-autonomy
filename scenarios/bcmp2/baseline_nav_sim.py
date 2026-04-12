"""
BCMP-2 Vehicle A Baseline Navigation Simulator.

Constraints C-1, C-2
---------------------
C-1: Same IMU model class and seed as Vehicle B.
C-2: Simulated drift must fall within pre-calculated STIM300 envelopes.

Navigation model — cross-track error propagation
-------------------------------------------------
Vehicle A lateral (cross-track) drift is modelled directly, matching the
C-2 analytical derivation exactly.

During GNSS-available phase (km 0–30):
  Cross-track error stays near zero (GNSS correction each second).

After GNSS denial (km 30 onward):
  heading_error(t) = gyro_bias_z * t + ARW_random_walk(t)
  cross_track(t)  ≈ 0.5 * heading_error(t) * v * t_local
                    + VRW_lateral_random_walk(t)
                    + 0.5 * accel_bias_e * t_local^2

where t_local is seconds since GNSS denial.

This is the same model as C-2. The gyro_bias_z is drawn once from the
STIM300 in-run bias instability distribution (C-1 compatible).

Along-track position is propagated from the planned route with no error
(Vehicle A follows the corridor centreline in heading, but drifts laterally).

The simulated cross-track error is what feeds the C-2 gate checker and
the KPI log. It represents the 2D lateral distance from the planned route.

JOURNAL
-------
Built: 29 March 2026, micromind-node01.  SB-1 Step 4.
Revised to direct cross-track error propagation after diagnosing that
route heading (6.12 deg NNE) caused systematic east-component drift in
the 2D DR position model.  Cross-track model matches C-2 directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.ins.imu_model import IMUModel, IMUNoiseOutput, generate_imu_noise, STIM300

from scenarios.bcmp2.bcmp2_drift_envelopes import (
    VEHICLE_SPEED_MS,
    STIM300_ARW_DEG_SQRTH,
    STIM300_GYRO_BIAS_INSTAB_DPH,
    STIM300_VRW_MS_SQRTH,
    STIM300_ACCEL_BIAS_INSTAB_MG,
    check_vehicle_a_drift,
    PHASE_ENVELOPES,
    CORRIDOR_BREACH_KM_1SIGMA,
)
from scenarios.bcmp2.bcmp2_scenario import (
    BCMP2Scenario,
    DisturbanceSchedule,
    CORRIDOR_HALF_WIDTH_M,
    GNSS_DENIAL_KM,
)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

SIM_DT_S  = 0.005   # 200 Hz
GNSS_DT_S = 1.0     # GNSS correction rate

# STIM300 parameters in SI for heading/lateral error model (from C-2 derivation)
_ARW_RAD_SQRTS  = math.radians(STIM300_ARW_DEG_SQRTH)  / math.sqrt(3600.0)
_GYRO_BIAS_RADS = math.radians(STIM300_GYRO_BIAS_INSTAB_DPH / 3600.0)
_ACCEL_BIAS_MS2 = STIM300_ACCEL_BIAS_INSTAB_MG * 1e-3 * 9.81
_VRW_MS_SQRTS   = STIM300_VRW_MS_SQRTH / math.sqrt(3600.0)


# ---------------------------------------------------------------------------
# State record
# ---------------------------------------------------------------------------

@dataclass
class VehicleAState:
    t_s:              float
    mission_km:       float
    phase:            int
    gnss_available:   bool
    # Cross-track (lateral) error from planned route
    cross_track_m:    float
    # Along-track position on planned route
    along_track_km:   float
    # For route map: estimated absolute position
    north_m:          float
    east_m:           float
    true_north_m:     float
    true_east_m:      float
    heading_error_rad: float


@dataclass
class BaselineRunResult:
    seed:            int
    imu_model_name:  str
    total_steps:     int
    dt_s:            float
    states:          List[VehicleAState] = field(default_factory=list)

    drift_at_km60:   Optional[float] = None
    drift_at_km100:  Optional[float] = None
    drift_at_km120:  Optional[float] = None

    first_corridor_violation_km: Optional[float] = None
    total_corridor_violations:   int = 0
    c2_gates:                    dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Baseline simulator — cross-track error propagation
# ---------------------------------------------------------------------------

class BaselineNavSim:
    """Vehicle A: cross-track error propagation matching C-2 derivation."""

    def __init__(
        self,
        scenario:   BCMP2Scenario,
        imu_model:  IMUModel = STIM300,
    ):
        self.scenario  = scenario
        self.imu_model = imu_model

    def run(
        self,
        seed:           int,
        schedule:       DisturbanceSchedule,
        noise:          IMUNoiseOutput,
        max_km:         float = 150.0,
        record_every_n: int   = 40,
    ) -> BaselineRunResult:
        """Simulate Vehicle A cross-track drift to max_km."""
        dt   = SIM_DT_S
        v_ms = VEHICLE_SPEED_MS

        result = BaselineRunResult(
            seed=seed,
            imu_model_name=self.imu_model.name,
            total_steps=0,
            dt_s=dt,
        )

        # ── Per-run bias draw from STIM300 distribution (C-1 compatible) ──
        rng_bias = np.random.default_rng(seed)
        gyro_bias_z  = rng_bias.normal(0.0, _GYRO_BIAS_RADS)
        accel_bias_e = rng_bias.normal(0.0, _ACCEL_BIAS_MS2)

        # Stochastic noise RNGs
        rng_arw = np.random.default_rng(seed + 1)
        rng_vrw = np.random.default_rng(seed + 2)

        # ── State ─────────────────────────────────────────────────────────
        cross_track   = 0.0   # metres, lateral deviation from planned route
        hdg_err       = 0.0   # accumulated heading error (rad)
        lat_vel_err   = 0.0   # lateral velocity error (m/s)
        t_since_denial = 0.0  # seconds of INS-only flight

        t          = 0.0
        gnss_timer = 0.0
        denial_s   = (GNSS_DENIAL_KM * 1000.0) / v_ms
        max_steps  = int((max_km * 1000.0 / v_ms) / dt) + 1

        for step in range(max_steps):
            mission_km = t * v_ms / 1000.0
            phase      = schedule.phase_at_km(mission_km)
            gnss_ok    = schedule.gnss_available(t)

            wp         = self.scenario.interpolate_waypoint(mission_km)
            true_north = wp.north_m
            true_east  = wp.east_m

            if gnss_ok:
                # GNSS phase: correct cross-track each second
                gnss_timer += dt
                if gnss_timer >= GNSS_DT_S:
                    gnss_timer = 0.0
                    # GNSS resets cross-track to near-zero (2.5m noise)
                    cross_track = rng_arw.normal(0.0, 2.5)
                    hdg_err    *= 0.05
                    lat_vel_err *= 0.10
                    t_since_denial = 0.0
                else:
                    # Between GNSS updates: small drift
                    hdg_err    += gyro_bias_z * dt
                    arw_rate    = rng_arw.normal(0.0, _ARW_RAD_SQRTS / math.sqrt(dt))
                    hdg_err    += arw_rate * dt
                    vrw_dv      = rng_vrw.normal(0.0, _VRW_MS_SQRTS * math.sqrt(dt))
                    lat_vel_err += vrw_dv + accel_bias_e * dt
                    cross_track += (v_ms * math.sin(hdg_err) + lat_vel_err) * dt
            else:
                # INS-only: heading error and lateral velocity accumulate
                t_since_denial += dt
                hdg_err    += gyro_bias_z * dt * schedule.imu_noise_scale
                arw_rate    = rng_arw.normal(0.0, _ARW_RAD_SQRTS / math.sqrt(dt))
                hdg_err    += arw_rate * dt
                vrw_dv      = rng_vrw.normal(0.0, _VRW_MS_SQRTS * math.sqrt(dt))
                lat_vel_err += vrw_dv + accel_bias_e * dt * schedule.imu_noise_scale
                cross_track += (v_ms * math.sin(hdg_err) + lat_vel_err) * dt

            lateral_err = abs(cross_track)

            # Absolute position for route map (perpendicular to route heading)
            route_bearing = math.atan2(true_east, true_north) if true_north > 0 else 0.0
            perp = route_bearing + math.pi / 2
            est_north = true_north + cross_track * math.cos(perp)
            est_east  = true_east  + cross_track * math.sin(perp)

            # ── Corridor violation ─────────────────────────────────────────
            if lateral_err > CORRIDOR_HALF_WIDTH_M:
                result.total_corridor_violations += 1
                if result.first_corridor_violation_km is None:
                    result.first_corridor_violation_km = mission_km

            # ── Phase-boundary snapshots ───────────────────────────────────
            prev_km = (t - dt) * v_ms / 1000.0
            for gate_km in (60, 100, 120):
                if prev_km < gate_km <= mission_km:
                    setattr(result, f"drift_at_km{gate_km}", lateral_err)

            # ── Record ─────────────────────────────────────────────────────
            if step % record_every_n == 0:
                result.states.append(VehicleAState(
                    t_s=t,
                    mission_km=mission_km,
                    phase=phase,
                    gnss_available=gnss_ok,
                    cross_track_m=cross_track,
                    along_track_km=mission_km,
                    north_m=est_north,
                    east_m=est_east,
                    true_north_m=true_north,
                    true_east_m=true_east,
                    heading_error_rad=hdg_err,
                ))

            t += dt
            if mission_km >= max_km:
                break

        result.total_steps = step + 1
        result.c2_gates    = self.check_c2_gates(result)
        return result

    @staticmethod
    def check_c2_gates(result: BaselineRunResult) -> dict:
        gates = {}
        for gate_km, attr in [(60,  "drift_at_km60"),
                               (100, "drift_at_km100"),
                               (120, "drift_at_km120")]:
            observed = getattr(result, attr, None)
            if observed is None:
                gates[gate_km] = {"passed": False,
                                   "reason": f"Not reached km {gate_km}",
                                   "observed_m": None}
            else:
                gates[gate_km] = check_vehicle_a_drift(gate_km, observed)
        return gates

    def to_kpi_dict(self, result: BaselineRunResult) -> dict:
        # Serialise position log — every recorded VehicleAState.
        # Used by demo_data_pipeline.get_vehicle_tracks() to build the
        # Vehicle A animation track without re-running the simulation.
        # Record cadence: every 40 steps at 200 Hz = every 0.2 s sim time
        # ≈ every 5.56 m at VEHICLE_SPEED_MS (27.78 m/s).
        position_log = [
            {
                "sim_timestamp_ms": int(round(s.t_s * 1000)),
                "mission_km":       round(s.mission_km, 4),
                "north_m":          round(s.north_m, 2),
                "east_m":           round(s.east_m, 2),
                "true_north_m":     round(s.true_north_m, 2),
                "true_east_m":      round(s.true_east_m, 2),
                "cross_track_m":    round(s.cross_track_m, 3),
                "phase":            s.phase,
                "gnss_available":   s.gnss_available,
                "nav_mode":         "GNSS_AIDED" if s.gnss_available else "INS_ONLY",
            }
            for s in result.states
        ]
        return {
            "vehicle":                     "A_baseline",
            "seed":                        result.seed,
            "imu_model":                   result.imu_model_name,
            "total_steps":                 result.total_steps,
            "drift_at_km60_m":             result.drift_at_km60,
            "drift_at_km100_m":            result.drift_at_km100,
            "drift_at_km120_m":            result.drift_at_km120,
            "first_corridor_violation_km": result.first_corridor_violation_km,
            "total_corridor_violations":   result.total_corridor_violations,
            "c2_gates":                    result.c2_gates,
            "final_lateral_error_m": (
                result.states[-1].cross_track_m if result.states else None
            ),
            "position_log":                position_log,
        }


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, ".")

    from scenarios.bcmp2.bcmp2_scenario import BCMP2Scenario, generate_disturbance_schedule

    print("BCMP-2 Baseline Nav Sim — C-2 drift envelope validation")
    print("=" * 60)
    print(f"Expected corridor breach (1σ): ~km {CORRIDOR_BREACH_KM_1SIGMA:.1f}")
    print()

    all_passed = True
    for seed in [42, 101, 303]:
        sc    = BCMP2Scenario()
        sched = generate_disturbance_schedule(seed)
        n     = int(150000 / VEHICLE_SPEED_MS / SIM_DT_S) + 100
        noise = generate_imu_noise(STIM300, n_steps=n, dt=SIM_DT_S, seed=seed)

        sim    = BaselineNavSim(sc, STIM300)
        t0     = time.time()
        result = sim.run(seed=seed, schedule=sched, noise=noise, max_km=150.0)
        elapsed = time.time() - t0

        print(f"Seed {seed} ({elapsed:.1f}s, {result.total_steps/elapsed:.0f} steps/s):")
        for km, gate in result.c2_gates.items():
            obs    = gate.get("observed_m")
            fl, ce = PHASE_ENVELOPES[km]["floor"], PHASE_ENVELOPES[km]["ceiling"]
            status = "PASS" if gate["passed"] else "FAIL"
            obs_s  = f"{obs:.0f} m" if obs is not None else "N/A"
            print(f"  km {km:>3}: {obs_s:>8}  [{fl:.0f}–{ce:.0f}]  {status}")
            if not gate["passed"]:
                all_passed = False
        vcv = result.first_corridor_violation_km
        print(f"  Breach: km {vcv:.1f}" if vcv else "  Breach: none")
        print()

    print("Overall:", "PASS ✓" if all_passed else "FAIL ✗")
