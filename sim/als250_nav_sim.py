"""
sim/als250_nav_sim.py — ALS-250 Navigation Simulation (Sprint S8-C)
250 km GNSS-denied corridor at 200 Hz with selectable IMU noise model.

Usage:
    PYTHONPATH=. python sim/als250_nav_sim.py
    PYTHONPATH=. python sim/als250_nav_sim.py --imu STIM300
    PYTHONPATH=. python sim/als250_nav_sim.py --imu ADIS16505_3 --seed 7 --out results/
    PYTHONPATH=. python sim/als250_nav_sim.py --imu BASELINE --duration 600

Outputs (all in --out directory, default: sim/):
    als250_nav_<imu>_<seed>_position.npy   — (N,3) position array [m, ENU]
    als250_nav_<imu>_<seed>_drift.npy      — (M,) per-segment drift [m] at every 5 km
    als250_nav_<imu>_<seed>_meta.json      — run metadata (params, KPIs, spec compliance)

The position arrays from all three IMU models are consumed by S8-D
(als250_drift_chart.py) to produce the three-curve TASL chart.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Optional

import numpy as np

# Core imports — repo must be on PYTHONPATH
from core.ins.mechanisation import ins_propagate
from core.ins.state import INSState
from core.ins.imu_model import get_imu_model, IMUModel, IMU_REGISTRY, generate_imu_noise
from core.constants import GRAVITY

# ---------------------------------------------------------------------------
# Scenario constants (ALS-250 normative corridor)
# ---------------------------------------------------------------------------

CORRIDOR_LENGTH_M  = 250_000.0     # 250 km GNSS-denied corridor
CRUISE_SPEED_MS    = 55.0          # m/s (≈ 200 km/h, ALS-250 cruise)
CRUISE_ALT_M       = 120.0         # m AGL
IMU_RATE_HZ        = 200           # Hz — matches S8-A spec
DT                 = 1.0 / IMU_RATE_HZ
SEGMENT_KM         = 5.0           # drift sampled every 5 km
SEGMENT_M          = SEGMENT_KM * 1_000.0
DRIFT_LIMIT_FRAC   = 0.02          # NAV-01: < 2% per 5 km (= 100 m max)
DRIFT_LIMIT_M      = DRIFT_LIMIT_FRAC * SEGMENT_M   # 100 m

# Total corridor flight time (pure straight-line, no wind)
CORRIDOR_DURATION_S = CORRIDOR_LENGTH_M / CRUISE_SPEED_MS   # ≈ 4545 s

# For rapid validation runs, --duration overrides corridor length
# (used by S8-D three-curve chart to keep runtime manageable)

# ---------------------------------------------------------------------------
# Scenario ground-truth trajectory
# ---------------------------------------------------------------------------

def _build_ground_truth(n_steps: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Straight-line corridor: vehicle travels East at CRUISE_SPEED_MS,
    constant altitude CRUISE_ALT_M. Minor sinusoidal perturbation in
    accel_b to exercise the INS (not a flat-line trivial case).

    Returns
    -------
    true_pos : (n_steps, 3) — ground-truth ENU position [m]
    true_vel : (n_steps, 3) — ground-truth ENU velocity [m/s]
    """
    t = np.arange(n_steps) * dt

    # Gentle sinusoidal altitude variation ± 5 m (terrain following)
    altitude = CRUISE_ALT_M + 5.0 * np.sin(2.0 * np.pi * t / 120.0)
    d_alt_dt = 5.0 * (2.0 * np.pi / 120.0) * np.cos(2.0 * np.pi * t / 120.0)

    # Gentle course deviation ± 50 m over corridor (yaw perturbation)
    north_dev  = 50.0 * np.sin(2.0 * np.pi * t / 600.0)
    d_north_dt = 50.0 * (2.0 * np.pi / 600.0) * np.cos(2.0 * np.pi * t / 600.0)

    true_pos = np.column_stack([
        CRUISE_SPEED_MS * t,   # East
        north_dev,             # North
        altitude,              # Up
    ])
    true_vel = np.column_stack([
        np.full(n_steps, CRUISE_SPEED_MS),
        d_north_dt,
        d_alt_dt,
    ])
    return true_pos, true_vel


def _body_inputs_from_truth(
    true_pos: np.ndarray,
    true_vel: np.ndarray,
    k: int,
    dt: float,
    attitude_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute true specific force and angular rate in body frame
    consistent with the ground-truth trajectory at step k.
    """
    # True acceleration in nav frame
    if k + 1 < len(true_vel):
        accel_nav = (true_vel[k + 1] - true_vel[k]) / dt
    else:
        accel_nav = np.zeros(3)

    # GRAVITY = [0, 0, -9.80665] in ENU (constants.py) — use directly
    # Specific force in nav frame (what an accelerometer measures)
    f_nav = accel_nav - GRAVITY  # = accel - gravity for ENU specific force

    # Rotate to body frame using current quaternion (q rotates nav→body via inverse)
    # body_vec = R^T @ nav_vec  (R is nav←body rotation)
    w, x, y, z = attitude_q
    # Transposed rotation: R^T = R(q^{-1}) = R(q conjugate)
    R_T = np.array([
        [1 - 2*(y*y + z*z),  2*(x*y + z*w),  2*(x*z - y*w)],
        [2*(x*y - z*w),  1 - 2*(x*x + z*z),  2*(y*z + x*w)],
        [2*(x*z + y*w),  2*(y*z - x*w),  1 - 2*(x*x + y*y)],
    ])
    accel_b = R_T @ f_nav

    # Angular rate: tiny yaw rate from north deviation sinusoid
    # d²north/dt² slope change gives a yaw perturbation
    omega_z = 0.0
    if k + 1 < len(true_pos):
        dx = true_pos[k + 1, 0] - true_pos[k, 0]
        dy = true_pos[k + 1, 1] - true_pos[k, 1]
        if dx > 1e-9:
            omega_z = dy / (dx * dt + 1e-12) * dt  # approximate yaw rate
    gyro_b = np.array([0.0, 0.0, omega_z])

    return accel_b, gyro_b


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_als250_sim(
    imu_name: Optional[str] = None,
    duration_s: float = CORRIDOR_DURATION_S,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run the ALS-250 corridor simulation.

    Parameters
    ----------
    imu_name  : str or None — IMU model key ("STIM300", "ADIS16505_3", "BASELINE")
                or None for clean (no-noise) baseline
    duration_s : float — simulation duration in seconds (default: full 250 km)
    seed      : int — RNG seed for noise generation
    verbose   : bool — print progress

    Returns
    -------
    dict with keys:
        position      — (N, 3) ndarray, INS position history [m, ENU]
        true_position — (N, 3) ndarray, ground-truth position
        drift_per_seg — (M,)  ndarray, position error at each 5-km segment boundary
        imu_name      — str
        n_steps       — int
        duration_s    — float
        kpi           — dict with NAV-01 compliance and summary statistics
    """
    n_steps = int(duration_s * IMU_RATE_HZ)
    if n_steps < 1:
        raise ValueError(f"duration_s={duration_s} → n_steps={n_steps}, too short")

    t_start = time.perf_counter()

    # ---- IMU model ----
    imu_model = get_imu_model(imu_name) if imu_name else None
    imu_noise = generate_imu_noise(imu_model, n_steps, DT, seed=seed) if imu_model else None

    # ---- Ground truth ----
    true_pos, true_vel = _build_ground_truth(n_steps, DT)

    # ---- Initial state (aligned to ground truth) ----
    state = INSState(
        p=true_pos[0].copy(),
        v=true_vel[0].copy(),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3),
    )

    # ---- Storage ----
    pos_hist = np.empty((n_steps, 3))

    # Segment drift sampling
    steps_per_seg = int(SEGMENT_M / (CRUISE_SPEED_MS * DT))
    seg_steps = list(range(steps_per_seg - 1, n_steps, steps_per_seg))
    drift_per_seg = np.empty(len(seg_steps))
    seg_idx = 0

    # ---- Propagation loop ----
    for k in range(n_steps):
        accel_b, gyro_b = _body_inputs_from_truth(
            true_pos, true_vel, k, DT, state.q
        )

        state = ins_propagate(
            state, accel_b, gyro_b, DT,
            imu_model=imu_model,
            imu_noise=imu_noise,
            step=k,
        )
        pos_hist[k] = state.p

        # Sample drift at segment boundaries
        if seg_idx < len(seg_steps) and k == seg_steps[seg_idx]:
            drift_per_seg[seg_idx] = np.linalg.norm(
                state.p - true_pos[k]
            )
            seg_idx += 1

    elapsed = time.perf_counter() - t_start

    # ---- KPIs ----
    final_drift_m = float(np.linalg.norm(pos_hist[-1] - true_pos[-1]))
    max_seg_drift  = float(drift_per_seg.max()) if len(drift_per_seg) > 0 else 0.0
    nav01_pass     = bool(max_seg_drift < DRIFT_LIMIT_M)
    dist_km        = (CRUISE_SPEED_MS * duration_s) / 1_000.0

    kpi = {
        "imu_model":       imu_name or "CLEAN",
        "corridor_km":     round(dist_km, 1),
        "duration_s":      round(duration_s, 1),
        "n_steps":         n_steps,
        "final_drift_m":   round(final_drift_m, 3),
        "max_5km_drift_m": round(max_seg_drift, 3),
        "drift_limit_m":   DRIFT_LIMIT_M,
        "NAV01_pass":      nav01_pass,
        "sim_wall_s":      round(elapsed, 2),
        "seed":            seed,
    }

    if verbose:
        tag   = imu_name or "CLEAN"
        mark  = "PASS ✅" if nav01_pass else "FAIL ❌"
        print(f"[ALS-250] IMU={tag:<14} dist={dist_km:6.0f} km  "
              f"final_drift={final_drift_m:8.1f} m  "
              f"max_5km={max_seg_drift:7.1f} m  NAV-01={mark}  "
              f"wall={elapsed:.1f}s")

    return {
        "position":       pos_hist,
        "true_position":  true_pos,
        "drift_per_seg":  drift_per_seg,
        "imu_name":       imu_name or "CLEAN",
        "n_steps":        n_steps,
        "duration_s":     duration_s,
        "kpi":            kpi,
    }


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_results(result: dict, out_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """
    Save position array, drift array, and metadata JSON to out_dir.
    Returns dict of {key: path}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tag  = result["imu_name"]
    seed = result["kpi"]["seed"]
    stem = f"als250_nav_{tag}_{seed}"

    pos_path   = out_dir / f"{stem}_position.npy"
    drift_path = out_dir / f"{stem}_drift.npy"
    meta_path  = out_dir / f"{stem}_meta.json"

    np.save(pos_path,   result["position"])
    np.save(drift_path, result["drift_per_seg"])
    meta_path.write_text(json.dumps(result["kpi"], indent=2))

    return {"position": pos_path, "drift": drift_path, "meta": meta_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="ALS-250 250 km corridor INS simulation (S8-C)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--imu",
        choices=list(IMU_REGISTRY.keys()) + ["ALL", "CLEAN"],
        default="ALL",
        help="IMU model to use. 'ALL' runs CLEAN + all registered models. "
             "'CLEAN' runs no-noise baseline only.",
    )
    p.add_argument("--seed",     type=int,   default=42,  help="RNG seed")
    p.add_argument("--duration", type=float, default=None,
                   help="Simulation duration in seconds (default: full 250 km corridor)")
    p.add_argument("--out",      type=str,   default="sim",
                   help="Output directory for .npy and .json files")
    p.add_argument("--no-save",  action="store_true",
                   help="Skip saving output files (useful for quick smoke tests)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    duration_s = args.duration if args.duration else CORRIDOR_DURATION_S
    out_dir    = pathlib.Path(args.out)

    # Determine which IMU models to run
    if args.imu == "ALL":
        imu_names = [None] + list(IMU_REGISTRY.keys())   # None = CLEAN baseline
    elif args.imu == "CLEAN":
        imu_names = [None]
    else:
        imu_names = [args.imu]

    print(f"\n{'='*72}")
    print(f"  ALS-250 NAV SIM — Sprint S8-C")
    print(f"  Corridor: {duration_s * CRUISE_SPEED_MS / 1000:.1f} km  "
          f"| Duration: {duration_s:.0f}s  | Rate: {IMU_RATE_HZ} Hz  | Seed: {args.seed}")
    print(f"{'='*72}")

    results = []
    for imu_name in imu_names:
        r = run_als250_sim(
            imu_name=imu_name,
            duration_s=duration_s,
            seed=args.seed,
            verbose=True,
        )
        results.append(r)
        if not args.no_save:
            paths = save_results(r, out_dir)

    print(f"{'='*72}")
    all_pass = all(r["kpi"]["NAV01_pass"] for r in results)
    print(f"  NAV-01 overall: {'PASS ✅' if all_pass else 'FAIL ❌'}")
    if not args.no_save:
        print(f"  Outputs → {out_dir}/")
    print(f"{'='*72}\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
