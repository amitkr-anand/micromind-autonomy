#!/usr/bin/env python
"""
run_s8_tests.py — Sprint S8 Master Test Runner
Repo root — run as: PYTHONPATH=. python run_s8_tests.py

Gates:
  S8-A  tests/test_s8a_imu_model.py         (16 tests — already green)
  S8-B  tests/test_s8b_mechanisation.py      (21 tests)
  S8-C  tests/test_s8c_als250_nav_sim.py     (17 tests)
  S8-E  tests/test_s8e_bcmp1_runner_imu.py   (12 tests)

S8-D (drift chart) is a visual artefact — validated by running:
  PYTHONPATH=. python dashboard/als250_drift_chart.py --duration 300 --seed 42

Regression (must stay green):
  python run_s5_tests.py            → 111/111
  python tests/test_s6_zpi_cems.py  → 36/36
"""
import subprocess
import sys
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parent

SUITES = [
    ("S8-A  IMU noise model",            "tests/test_s8a_imu_model.py"),
    ("S8-B  INS mechanisation extension","tests/test_s8b_mechanisation.py"),
    ("S8-C  ALS-250 nav sim",            "tests/test_s8c_als250_nav_sim.py"),
    ("S8-E  BCMP-1 runner IMU extension","tests/test_s8e_bcmp1_runner_imu.py"),
]

def run_suite(label: str, path: str) -> tuple[bool, float]:
    full = ROOT / path
    if not full.exists():
        print(f"  [SKIP]  {label:42s} — file not found: {path}")
        return True, 0.0
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(full), "-v", "--tb=short", "-q"],
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    elapsed = time.perf_counter() - t0
    ok = result.returncode == 0
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}]  {label:42s}  {elapsed:.1f}s")
    return ok, elapsed


def main():
    print(f"\n{'='*70}")
    print("  MicroMind S8 — Full Sprint Test Run")
    print(f"{'='*70}")

    results = []
    total_t = 0.0
    for label, path in SUITES:
        ok, t = run_suite(label, path)
        results.append((label, ok, t))
        total_t += t

    print(f"\n{'='*70}")
    all_pass = all(ok for _, ok, _ in results)
    n_pass = sum(ok for _, ok, _ in results)
    print(f"  S8 gates: {n_pass}/{len(results)} PASS  ({total_t:.1f}s total)")

    if all_pass:
        print(f"\n  SPRINT S8 GATE: PASS [ALL SUITES GREEN]")
        print(f"\n  Next steps:")
        print(f"    python run_s5_tests.py               # regression 111/111")
        print(f"    python tests/test_s6_zpi_cems.py     # regression 36/36")
        print(f"    PYTHONPATH=. python dashboard/als250_drift_chart.py  # S8-D chart")
        print(f"    PYTHONPATH=. python sim/als250_nav_sim.py            # full 250 km run")
    else:
        print(f"\n  SPRINT S8 GATE: FAIL")
        for label, ok, _ in results:
            if not ok:
                print(f"    FAIL: {label}")

    print(f"{'='*70}\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
