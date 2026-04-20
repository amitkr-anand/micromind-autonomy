"""
BCMP-2 Regression Harness.

Wraps bcmp2_runner.run_bcmp2() into stable acceptance test entry points:
  AT-1 — 5 km smoke test
  AT-2 — canonical 150 km seed sweep
  AT-3 — nav transition / corridor check

This file should not modify bcmp2_runner.py behaviour.
It only orchestrates repeatable regression runs and summarizes results.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

from scenarios.bcmp2.bcmp2_runner import (
    BCMP2RunConfig,
    run_bcmp2,
    run_at1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_gate_pass(gates: dict, km: int) -> bool:
    gate = gates.get(km, {})
    return bool(gate.get("passed", False))


# ---------------------------------------------------------------------------
# AT-1: Smoke test
# ---------------------------------------------------------------------------

def run_at1_smoke(seed: int = 42) -> dict:
    """
    5 km boot smoke test.
    Verifies runner returns all required top-level keys.
    """
    out = run_at1(seed=seed, max_km=5.0)

    return {
        "test_id": "AT-1",
        "seed": seed,
        "passed": all([
            "disturbance_schedule" in out,
            "vehicle_a" in out,
            "vehicle_b" in out,
            "comparison" in out,
        ]),
        "vehicle_a_result": out.get("comparison", {}).get("vehicle_a_mission_result"),
        "vehicle_b_result": out.get("comparison", {}).get("vehicle_b_mission_result"),
        "raw": out,
    }


# ---------------------------------------------------------------------------
# AT-2: Canonical seed sweep
# ---------------------------------------------------------------------------

def run_at2_seed_sweep(seeds=(42, 101, 303), max_km: float = 150.0) -> dict:
    """
    Canonical NAV-01 regression sweep.
    Validates km60 / km100 / km120 C-2 gates across 3 seeds.
    """
    runs = []

    for seed in seeds:
        cfg = BCMP2RunConfig(
            seed=seed,
            max_km=max_km,
            verbose=False,
        )

        out = run_bcmp2(cfg)
        gates = out["vehicle_a"].get("c2_gates", {})

        run_result = {
            "seed": seed,
            "km60_pass": _safe_gate_pass(gates, 60),
            "km100_pass": _safe_gate_pass(gates, 100),
            "km120_pass": _safe_gate_pass(gates, 120),
            "all_gates_pass": all([
                _safe_gate_pass(gates, 60),
                _safe_gate_pass(gates, 100),
                _safe_gate_pass(gates, 120),
            ]),
            "vehicle_a_result": out["comparison"].get("vehicle_a_mission_result"),
            "vehicle_b_result": out["comparison"].get("vehicle_b_mission_result"),
            "vehicle_a_drift_km120_m": out["comparison"].get("vehicle_a_drift_km120_m"),
            "vehicle_b_max_5km_drift_m": out["comparison"].get("vehicle_b_max_5km_drift_m"),
            "raw": out,
        }

        runs.append(run_result)

    overall_pass = all(r["all_gates_pass"] for r in runs)

    return {
        "test_id": "AT-2",
        "overall_pass": overall_pass,
        "seed_count": len(runs),
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# AT-3: Nav transition / corridor test
# ---------------------------------------------------------------------------

def run_at3_nav_transition(seed: int = 42, max_km: float = 120.0) -> dict:
    """
    Focused transition test for GNSS-denied segment.
    """
    cfg = BCMP2RunConfig(
        seed=seed,
        max_km=max_km,
        verbose=False,
    )

    out = run_bcmp2(cfg)
    comp = out["comparison"]

    breach_km = comp.get("vehicle_a_first_corridor_violation_km")

    return {
        "test_id": "AT-3",
        "seed": seed,
        "passed": breach_km is None or breach_km > 76.9,
        "corridor_breach_km": breach_km,
        "vehicle_a_result": comp.get("vehicle_a_mission_result"),
        "vehicle_b_result": comp.get("vehicle_b_mission_result"),
        "raw": out,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_results(results: list[dict]) -> None:
    print("\nBCMP-2 Regression Summary")
    print("=" * 60)

    for r in results:
        print(f"{r['test_id']}: {'PASS' if r.get('passed', r.get('overall_pass')) else 'FAIL'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    at1 = run_at1_smoke()
    at2 = run_at2_seed_sweep()
    at3 = run_at3_nav_transition()

    results = [
        {
            "test_id": "AT-1",
            "passed": at1["passed"],
        },
        {
            "test_id": "AT-2",
            "passed": at2["overall_pass"],
        },
        {
            "test_id": "AT-3",
            "passed": at3["passed"],
        },
    ]

    summarize_results(results)

    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/bcmp2_regression_results.json"

    with open(out_path, "w") as f:
        json.dump(
            {
                "generated_at": time.time(),
                "elapsed_s": round(time.time() - t0, 2),
                "at1": at1,
                "at2": at2,
                "at3": at3,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\nResults written to: {out_path}")
