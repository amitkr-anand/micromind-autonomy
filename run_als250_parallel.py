#!/usr/bin/env python3
"""
run_als250_parallel.py — S10-4: Parallel ALS-250 Multi-IMU Runner
================================================================
Runs STIM300, ADIS16505_3, and BASELINE simultaneously using one
subprocess per IMU. No shared MKL state, no fork, no mutex inheritance.

Usage (from repo root, inside tmux):
    PYTHONPATH=. python run_als250_parallel.py --corridor-km 50 --seed 42
    PYTHONPATH=. python run_als250_parallel.py --corridor-km 250 --seed 42
    PYTHONPATH=. python run_als250_parallel.py --corridor-km 250 --imu STIM300 ADIS16505_3

Exit codes:
    0 — all IMUs completed, all NAV01 pass
    1 — all completed, at least one NAV01 fail
    2 — at least one subprocess failed (crash/error)
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import List

DEFAULT_IMUS   = ["STIM300", "ADIS16505_3", "BASELINE"]
DEFAULT_OUT    = "sim/als250_results"
DEFAULT_SEED   = 42
DEFAULT_KM     = 250

_WORKER_SCRIPT = """\
import sys, os, time, json
os.environ["MKL_NUM_THREADS"] = "1"

imu_name    = sys.argv[1]
corridor_km = float(sys.argv[2])
seed        = int(sys.argv[3])
out_dir     = sys.argv[4]

import pathlib
from sim.als250_nav_sim import run_als250_sim, save_results

t0 = time.perf_counter()
print(f"[{imu_name}] Starting {corridor_km:.0f} km corridor, seed={seed}", flush=True)

result = run_als250_sim(
    imu_name=imu_name,
    corridor_km=corridor_km,
    seed=seed,
    verbose=False,
)

wall_s = time.perf_counter() - t0
kpi    = result["kpi"]
paths  = save_results(result, pathlib.Path(out_dir))
status = "PASS" if kpi["NAV01_pass"] else "FAIL"

print(
    f"[{imu_name}] {status}  "
    f"max_drift={kpi['max_5km_drift_m']:.1f}m  "
    f"wall={wall_s/60:.1f}min",
    flush=True,
)

summary = {
    "imu":         imu_name,
    "NAV01_pass":  kpi["NAV01_pass"],
    "max_drift_m": kpi["max_5km_drift_m"],
    "corridor_km": corridor_km,
    "seed":        seed,
    "wall_s":      round(wall_s, 1),
    "outputs":     {k: str(v) for k, v in paths.items()},
}
print("RESULT:" + json.dumps(summary), flush=True)
"""


class IMUWorker:
    def __init__(self, imu: str, corridor_km: float, seed: int,
                 out_dir: pathlib.Path):
        self.imu            = imu
        self.corridor_km    = corridor_km
        self.seed           = seed
        self.out_dir        = out_dir
        self.start_time     = None
        self.proc           = None
        self.stdout_lines: List[str] = []
        self.result_summary: dict    = {}
        self.returncode: int         = -1

    def start(self) -> None:
        cmd = [
            sys.executable, "-c", _WORKER_SCRIPT,
            self.imu, str(self.corridor_km), str(self.seed), str(self.out_dir),
        ]
        self.start_time = time.perf_counter()
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        print(f"[{self.imu}] Launched PID {self.proc.pid}", flush=True)

    def collect(self) -> bool:
        stdout, stderr = self.proc.communicate()
        self.returncode    = self.proc.returncode
        self.stdout_lines  = stdout.splitlines()

        for line in self.stdout_lines:
            print(f"  {line}", flush=True)
        if stderr.strip():
            print(f"  [{self.imu}] STDERR:", flush=True)
            for line in stderr.strip().splitlines():
                print(f"    {line}", flush=True)

        if self.returncode != 0:
            print(f"[{self.imu}] FAILED — exit code {self.returncode}", flush=True)
            return False

        for line in reversed(self.stdout_lines):
            if line.startswith("RESULT:"):
                try:
                    self.result_summary = json.loads(line[len("RESULT:"):])
                except json.JSONDecodeError:
                    pass
                break
        return True

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.start_time if self.start_time else 0.0


def run_parallel(imu_list: List[str], corridor_km: float, seed: int,
                 out_dir: pathlib.Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = [IMUWorker(imu, corridor_km, seed, out_dir) for imu in imu_list]
    for w in workers:
        w.start()

    print(f"\n{len(workers)} workers running. Waiting for completion...\n", flush=True)

    wall_start = time.perf_counter()
    results, failed = [], []

    for w in workers:
        success = w.collect()
        if success and w.result_summary:
            results.append(w.result_summary)
            status = "PASS" if w.result_summary.get("NAV01_pass") else "FAIL"
            print(f"[{w.imu}] Collected — {status}  wall={w.elapsed_s/60:.1f}min",
                  flush=True)
        else:
            failed.append(w.imu)
            print(f"[{w.imu}] Collected — FAILED  wall={w.elapsed_s/60:.1f}min",
                  flush=True)

    total_wall = time.perf_counter() - wall_start

    manifest = {
        "corridor_km":  corridor_km,
        "seed":         seed,
        "out_dir":      str(out_dir),
        "total_wall_s": round(total_wall, 1),
        "imu_results":  results,
        "failed_imus":  failed,
    }

    print()
    print("=" * 65)
    print(f"{'IMU':<16} {'NAV01':>6} {'Max drift':>12} {'Wall time':>12}")
    print("-" * 65)
    for r in results:
        status = "PASS" if r["NAV01_pass"] else "FAIL"
        print(f"{r['imu']:<16} {status:>6} {r['max_drift_m']:>10.1f}m"
              f" {r['wall_s']/60:>10.1f}min")
    for imu in failed:
        print(f"{imu:<16} {'ERROR':>6} {'---':>12} {'---':>12}")
    print("-" * 65)
    print(f"{'Total parallel wall time':>40}  {total_wall/60:.1f}min")
    print("=" * 65)

    if failed:
        print(f"\nWARNING: {len(failed)} IMU(s) failed: {', '.join(failed)}")

    manifest_path = out_dir / "als250_parallel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest → {manifest_path}")

    return manifest


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="S10-4: Parallel ALS-250 multi-IMU runner (subprocess.Popen, no MKL deadlock)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--imu", nargs="+", default=DEFAULT_IMUS, metavar="IMU",
                    help=f"IMU model(s) (default: {' '.join(DEFAULT_IMUS)})")
    ap.add_argument("--corridor-km", type=float, default=DEFAULT_KM,
                    help=f"Corridor length in km (default: {DEFAULT_KM})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"Random seed (default: {DEFAULT_SEED})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Output directory (default: {DEFAULT_OUT})")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    out_dir = pathlib.Path(args.out)

    print("=" * 65)
    print("MicroMind ALS-250 Parallel IMU Runner — S10-4")
    print("=" * 65)
    print(f"IMU models  : {', '.join(args.imu)}")
    print(f"Corridor    : {args.corridor_km:.0f} km")
    print(f"Seed        : {args.seed}")
    print(f"Output dir  : {out_dir}")
    print(f"Strategy    : subprocess.Popen per IMU (no MKL fork deadlock)")
    print("=" * 65)
    print()

    manifest   = run_parallel(args.imu, args.corridor_km, args.seed, out_dir)
    any_failed = bool(manifest["failed_imus"])
    all_pass   = all(r.get("NAV01_pass") for r in manifest["imu_results"])

    if any_failed:
        sys.exit(2)
    elif not all_pass:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
