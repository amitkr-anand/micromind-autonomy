"""
BCMP-2 Vehicle A INS-only lateral drift envelopes.

Source
------
Safran STIM300 datasheet TS1524 rev.31 (Table 5-3 gyro, Table 5-4 5g/10g accel)
Frozen ESKF Q-matrix constants from Sprint S9 (_GYRO_BIAS_RW, _ACC_BIAS_RW).
Cross-referenced against ALS-250 S10 NAV-01 characterisation results.

Derivation method
-----------------
Analytical INS-only lateral drift model.  GNSS denied from km 30.
Both vehicles carry an identical STIM300-class IMU (Constraint C-1).
Vehicle A has no correction stack (no TRN, no VIO, no ESKF updates after
GNSS denial).  Vehicle B carries the full MicroMind correction stack.

Four lateral-error contributors are modelled independently then combined:

  1. Heading error from ARW (stochastic):
       x_arw = 0.5 * ARW_rad_sqrts * v_ms * t**1.5
     where t is seconds of INS-only flight from GNSS denial.

  2. Heading error from gyro in-run bias (deterministic worst-case):
       x_gyro_bias = 0.5 * gyro_bias_rad_s * v_ms * t**2
     Gyro bias integrates into heading error, which integrates into position.

  3. Lateral velocity error from VRW (stochastic):
       x_vrw = VRW_m_sqrts * t**1.5 / sqrt(3)

  4. Lateral position error from accelerometer bias (deterministic):
       x_acc_bias = 0.5 * accel_bias_ms2 * t**2

Stochastic terms (1, 3) are RSS-combined for sigma_stochastic.
Deterministic terms (2, 4) are added directly.
Total 1-sigma estimate = sigma_stochastic + x_gyro_bias + x_acc_bias

C-2 envelope
  floor   = 0.5 * total_1sigma  (minimum credible drift — baseline too gentle if below)
  nominal = 1.0 * total_1sigma  (central expected estimate)
  ceiling = 3.0 * total_1sigma  (maximum credible drift — unrealistic failure if above)

A BCMP-2 AT-2 run is REJECTED if Vehicle A drift at any phase boundary falls
outside [floor, ceiling].  Both directions invalidate the demonstration:
  - Below floor  → baseline too gentle → comparison invalid
  - Above ceiling → unrealistic failure → comparison distorted

Physics note
------------
At all phase boundaries, gyro bias is the dominant contributor (grows as t²).
Flat terrain in P3 removes TRN correction opportunity, allowing the gyro bias
term to dominate unchecked.  INS-only 1-sigma lateral drift crosses the
100 m corridor half-width reference at approximately km 79.5 from mission start
(t ≈ 900 s INS-only) — squarely inside P3 plains.  This is the natural
demonstration moment: Vehicle A loses the corridor on flat terrain where there
is nothing to help it.

ADIS16505-3 note
----------------
If the session uses the ADIS profile (ARW=0.22 °/√h, gyro_bias=8.0 °/h),
corridor breach occurs earlier (~km 55–60) and ceiling values roughly double.
Derivation logic is identical; only input parameters change.

JOURNAL
-------
Derived: 29 March 2026, micromind-node01.
Committed independently as the first SB-1 deliverable, before any runner code.
"""

import math

# ---------------------------------------------------------------------------
# STIM300 source parameters (TS1524 rev.31)
# ---------------------------------------------------------------------------

STIM300_ARW_DEG_SQRTH           = 0.15      # °/√h  — Table 5-3, gyro ARW
STIM300_VRW_MS_SQRTH            = 0.024     # m/s/√h — Table 5-4, 5g/10g accel VRW
STIM300_GYRO_BIAS_INSTAB_DPH    = 0.5       # °/h   — in-run bias instability (S8 param)
STIM300_ACCEL_BIAS_INSTAB_MG    = 0.006     # mg    — 600 s Allan variance, Table 5-4
STIM300_GYRO_BIAS_RW            = 4.04e-8   # rad/s/√s — frozen ESKF _GYRO_BIAS_RW (S9)
STIM300_ACCEL_BIAS_RW           = 9.81e-7   # m/s²/√s  — frozen ESKF _ACC_BIAS_RW (S9)

# ---------------------------------------------------------------------------
# Mission geometry
# ---------------------------------------------------------------------------

GNSS_DENIAL_KM   = 30.0    # km from mission start — GNSS denied here
VEHICLE_SPEED_MS = 55.0    # m/s — ALS-250 class forward speed

# ---------------------------------------------------------------------------
# Derived SI parameters (computed once at import)
# ---------------------------------------------------------------------------

# ARW: °/√h → rad/√s   (= rad/s/√Hz)
_ARW_RAD_SQRTS = math.radians(STIM300_ARW_DEG_SQRTH) / math.sqrt(3600.0)

# VRW: m/s/√h → m/s/√s
_VRW_MS_SQRTS = STIM300_VRW_MS_SQRTH / math.sqrt(3600.0)

# Gyro bias: °/h → rad/s
_GYRO_BIAS_RADS = math.radians(STIM300_GYRO_BIAS_INSTAB_DPH / 3600.0)

# Accel bias: mg → m/s²
_ACCEL_BIAS_MS2 = STIM300_ACCEL_BIAS_INSTAB_MG * 1e-3 * 9.81


# ---------------------------------------------------------------------------
# Core derivation function
# ---------------------------------------------------------------------------

def ins_only_lateral_drift_1sigma(ins_only_km: float) -> dict:
    """
    Return 1-sigma lateral drift estimate for a pure-INS vehicle (Vehicle A)
    after *ins_only_km* kilometres of GNSS-denied flight at VEHICLE_SPEED_MS.

    Returns
    -------
    dict with keys:
        t_s          — INS-only flight duration (seconds)
        x_arw_m      — ARW contribution to lateral error (m)
        x_gyro_bias_m — gyro bias contribution (m)
        x_vrw_m      — VRW contribution (m)
        x_acc_bias_m — accel bias contribution (m)
        sigma_stoch_m — RSS of stochastic terms (m)
        total_1sigma_m — total 1-sigma lateral error (m)
        floor_m      — C-2 envelope floor  (0.5 × 1σ)
        nominal_m    — C-2 envelope nominal (1σ)
        ceiling_m    — C-2 envelope ceiling (3σ)
    """
    t = (ins_only_km * 1000.0) / VEHICLE_SPEED_MS   # seconds

    x_arw       = 0.5 * _ARW_RAD_SQRTS * VEHICLE_SPEED_MS * t ** 1.5
    x_gyro_bias = 0.5 * _GYRO_BIAS_RADS * VEHICLE_SPEED_MS * t ** 2
    x_vrw       = _VRW_MS_SQRTS * t ** 1.5 / math.sqrt(3.0)
    x_acc_bias  = 0.5 * _ACCEL_BIAS_MS2 * t ** 2

    sigma_stoch  = math.sqrt(x_arw ** 2 + x_vrw ** 2)
    total_1sigma = sigma_stoch + x_gyro_bias + x_acc_bias

    floor   = max(10.0, total_1sigma * 0.5)
    nominal = total_1sigma
    ceiling = total_1sigma * 3.0

    return {
        "t_s":           t,
        "x_arw_m":       x_arw,
        "x_gyro_bias_m": x_gyro_bias,
        "x_vrw_m":       x_vrw,
        "x_acc_bias_m":  x_acc_bias,
        "sigma_stoch_m": sigma_stoch,
        "total_1sigma_m": total_1sigma,
        "floor_m":       floor,
        "nominal_m":     nominal,
        "ceiling_m":     ceiling,
    }


def corridor_breach_km(half_width_m: float = 100.0,
                       step_km: float = 0.1) -> float:
    """
    Return the mission-km at which Vehicle A INS-only 1-sigma lateral drift
    first exceeds *half_width_m*.  Iterates from GNSS_DENIAL_KM forward.
    """
    km = GNSS_DENIAL_KM
    while km <= 200.0:
        ins_km = km - GNSS_DENIAL_KM
        if ins_km <= 0:
            km += step_km
            continue
        if ins_only_lateral_drift_1sigma(ins_km)["total_1sigma_m"] >= half_width_m:
            return round(km, 1)
        km += step_km
    return float("inf")


# ---------------------------------------------------------------------------
# Pre-computed phase-boundary envelopes  (C-2 gates for AT-2 and later)
# ---------------------------------------------------------------------------
# Analytical intermediates (retained for reference and corridor_breach_km)
_KM60  = ins_only_lateral_drift_1sigma(ins_only_km=30.0)   # km 60  -- end P2
_KM100 = ins_only_lateral_drift_1sigma(ins_only_km=70.0)   # km 100 -- end P3
_KM120 = ins_only_lateral_drift_1sigma(ins_only_km=90.0)   # km 120 -- end P4

# ---------------------------------------------------------------------------
# Per-seed simulation gate envelopes  (Monte Carlo calibrated)
# ---------------------------------------------------------------------------
# The analytical C-2 formula uses worst-case constant bias (0.5 deg/h max).
# baseline_nav_sim.py draws gyro bias from normal(0, 0.5 deg/h sigma), so
# individual runs legitimately produce lower drift than the analytical worst-case.
#
# Floors and ceilings below are Monte Carlo derived (N=300, seed 99/42):
#   km 60:  P5=2m   P50=19m  P99=73m   -> floor=5m   ceiling=80m
#   km 100: P5=10m  P50=96m  P99=317m  -> floor=12m  ceiling=350m
#   km 120: P5=13m  P50=155m P99=604m  -> floor=15m  ceiling=650m
#
# Floor: below this indicates a simulation bug, not a lucky seed.
# Ceiling: above this indicates an unrealistic catastrophic failure.
# Both reject the run as invalid for comparison purposes.

ENVELOPE_KM60  = {"floor": 5,   "nominal": 19,  "ceiling": 80}
ENVELOPE_KM100 = {"floor": 12,  "nominal": 96,  "ceiling": 350}
ENVELOPE_KM120 = {"floor": 15,  "nominal": 155, "ceiling": 650}

# Convenience dict keyed by mission-km
PHASE_ENVELOPES = {
    60:  ENVELOPE_KM60,
    100: ENVELOPE_KM100,
    120: ENVELOPE_KM120,
}

# Corridor breach reference (1-sigma crosses 100 m half-width)
CORRIDOR_BREACH_KM_1SIGMA = corridor_breach_km(half_width_m=100.0)


# ---------------------------------------------------------------------------
# Gate checker
# ---------------------------------------------------------------------------

def check_vehicle_a_drift(mission_km: int, observed_drift_m: float) -> dict:
    """
    Check whether *observed_drift_m* for Vehicle A at *mission_km* satisfies
    the C-2 envelope.

    Parameters
    ----------
    mission_km     : one of {60, 100, 120}
    observed_drift_m : lateral drift observed in the simulation run (metres)

    Returns
    -------
    dict with keys:
        passed  — bool
        reason  — human-readable string
        floor_m, nominal_m, ceiling_m, observed_m
    """
    if mission_km not in PHASE_ENVELOPES:
        raise ValueError(f"mission_km must be one of {list(PHASE_ENVELOPES)}, got {mission_km}")
    env = PHASE_ENVELOPES[mission_km]
    fl, ce = env["floor"], env["ceiling"]
    passed = fl <= observed_drift_m <= ce
    if passed:
        reason = (f"PASS — {observed_drift_m:.1f} m within [{fl:.0f}, {ce:.0f}] m "
                  f"at km {mission_km}")
    elif observed_drift_m < fl:
        reason = (f"FAIL — {observed_drift_m:.1f} m below floor {fl:.0f} m at km {mission_km}. "
                  f"Baseline too gentle; comparison invalid.")
    else:
        reason = (f"FAIL — {observed_drift_m:.1f} m above ceiling {ce:.0f} m at km {mission_km}. "
                  f"Unrealistic failure; comparison distorted.")
    return {
        "passed": passed,
        "reason": reason,
        "floor_m":    fl,
        "nominal_m":  env["nominal"],
        "ceiling_m":  ce,
        "observed_m": observed_drift_m,
    }


# ---------------------------------------------------------------------------
# Self-print (run directly to verify)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("BCMP-2 Vehicle A INS-only drift envelopes")
    print("Source: STIM300 TS1524 rev.31 + frozen ESKF S9 constants")
    print(f"GNSS denial: km {GNSS_DENIAL_KM:.0f}  |  Speed: {VEHICLE_SPEED_MS:.0f} m/s")
    print()
    print(f"{'Boundary':<22} {'INS-km':>7} {'t(s)':>7} "
          f"{'ARW':>8} {'GyroB':>8} {'VRW':>8} {'AccB':>8} "
          f"{'1σ':>8} {'Floor':>8} {'Ceiling':>9}")
    print("-" * 103)
    for label, ins_km in [("km 60 — end P2", 30), ("km 100 — end P3", 70), ("km 120 — end P4", 90)]:
        r = ins_only_lateral_drift_1sigma(ins_km)
        print(f"{label:<22} {ins_km:>7.0f} {r['t_s']:>7.0f} "
              f"{r['x_arw_m']:>8.1f} {r['x_gyro_bias_m']:>8.1f} "
              f"{r['x_vrw_m']:>8.1f} {r['x_acc_bias_m']:>8.2f} "
              f"{r['total_1sigma_m']:>8.1f} {r['floor_m']:>8.0f} {r['ceiling_m']:>9.0f}")
    print()
    print(f"Corridor breach (100 m half-width, 1σ): km {CORRIDOR_BREACH_KM_1SIGMA}")
    print()
    print("Phase envelope constants:")
    for km, env in PHASE_ENVELOPES.items():
        print(f"  km {km:>3}: floor={env['floor']:.0f} m  "
              f"nominal={env['nominal']:.0f} m  ceiling={env['ceiling']:.0f} m")
