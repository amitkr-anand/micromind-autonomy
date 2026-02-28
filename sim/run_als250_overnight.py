#!/usr/bin/env python3
"""
sim/run_als250_overnight.py
MicroMind S8-D — ALS-250 Full 250 km Overnight Run

Runs the full 250 km corridor simulation for all four IMU configurations:
  - CLEAN    (no noise model — baseline reference)
  - STIM300  (Safran STIM300 — tactical grade)
  - ADIS16505_3 (Analog Devices — MEMS grade)
  - BASELINE (S0–S7 simplified model)

Results saved to sim/als250_results/ as .npy + .json files.
Run als250_drift_chart.py next session to render the three-curve TASL chart.

Usage:
    PYTHONPATH=. python sim/run_als250_overnight.py
    PYTHONPATH=. python sim/run_als250_overnight.py --seed 7
    PYTHONPATH=. python sim/run_als250_overnight.py --out-dir results/als250/

Estimated runtime: ~4–6 hours (depends on Mac performance)
"""

import sys
import os
import time
import argparse
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

from sim.als250_nav_sim import run_als250_sim, CORRIDOR_DURATION_S, CRUISE_SPEED_MS

MODELS = ["CLEAN", "STIM300", "ADIS16505_3", "BASELINE"]


def main():
    parser = argparse.ArgumentParser(
        description="ALS-250 full 250 km overnight run — all IMU models"
    )
    parser.add_argument("--seed",    type=int, default=42,
                        help="RNG seed (default: 42)")
    parser.add_argument("--out-dir", type=str, default="sim/als250_results",
                        help="Output directory for .npy + .json files")
    args = parser.parse_args()

    out_dir = os.path.join(_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    corridor_km = CORRIDOR_DURATION_S * CRUISE_SPEED_MS / 1000.0
    print("=" * 70)
    print(f"  MicroMind ALS-250 — Full Overnight Run")
    print(f"  Corridor: {corridor_km:.0f} km  |  Seed: {args.seed}")
    print(f"  Output:   {out_dir}")
    print(f"  Models:   {', '.join(MODELS)}")
    print("=" * 70)

    summary = {}
    total_start = time.time()

    for imu_name in MODELS:
        print(f"\n[{imu_name}] Starting ...")
        t0 = time.time()

        result = run_als250_sim(
            imu_name=None if imu_name == "CLEAN" else imu_name,
            duration_s=CORRIDOR_DURATION_S,
            seed=args.seed,
            verbose=True,
        )
        # Save outputs manually
        import numpy as np
        tag = imu_name.replace("/", "_")
        np.save(f"{out_dir}/als250_nav_{tag}_{args.seed}_position.npy", result["position"])
        np.save(f"{out_dir}/als250_nav_{tag}_{args.seed}_drift.npy", result["drift_per_seg"])
        import json as _json
        meta = {k: v for k, v in result.items() if k not in ("position", "true_position", "drift_per_seg")}
        with open(f"{out_dir}/als250_nav_{tag}_{args.seed}_meta.json", "w") as _f:
            _json.dump(meta, _f, indent=2, default=str)

        elapsed = time.time() - t0
        nav01 = result.get("NAV01_pass", False)
        max_drift = max(result.get("drift_m", [0]))
        dist_km = result.get("dist_km", corridor_km)

        print(f"[{imu_name}] DONE  {elapsed/60:.1f} min  |  "
              f"NAV-01: {'PASS' if nav01 else 'FAIL'}  |  "
              f"Max drift: {max_drift:.1f} m over {dist_km:.0f} km")

        summary[imu_name] = {
            "elapsed_min": round(elapsed / 60, 1),
            "NAV01_pass":  nav01,
            "max_drift_m": round(float(max_drift), 2),
            "dist_km":     round(float(dist_km), 1),
            "n_steps":     result.get("n_steps", 0),
        }

    total_elapsed = time.time() - total_start

    # Write summary
    summary_path = os.path.join(out_dir, f"overnight_summary_seed{args.seed}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  ALL MODELS COMPLETE — {total_elapsed/3600:.1f} hours total")
    print(f"  Summary: {summary_path}")
    print("=" * 70)
    print(f"\n{'Model':<15} {'NAV-01':>8} {'Max drift (m)':>14} {'Time (min)':>11}")
    print("-" * 55)
    for name, s in summary.items():
        nav = "PASS" if s["NAV01_pass"] else "FAIL"
        print(f"  {name:<13} {nav:>8} {s['max_drift_m']:>14.1f} {s['elapsed_min']:>11.1f}")

    print(f"\nNext step:")
    print(f"  PYTHONPATH=. python dashboard/als250_drift_chart.py "
          f"--results-dir {out_dir}")


if __name__ == "__main__":
    main()
