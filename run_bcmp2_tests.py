#!/usr/bin/env python
"""
BCMP-2 test runner.

Mirrors run_s5_tests.py pattern.  Run from repo root:

    conda activate micromind-autonomy
    python run_bcmp2_tests.py 2>&1 | tee bcmp2_test_results.txt

Exits 0 on all pass, 1 on any failure.
"""

import subprocess
import sys
import time


def main():
    suites = [
        ("AT-1 Boot & Regression",       "tests/test_bcmp2_at1.py"),
        ("SB-2 Fault Injection Proxies", "tests/test_bcmp2_sb2.py"),
        ("AT-2 Nominal 150 km",          "tests/test_bcmp2_at2.py"),
        ("AT-3/4/5 Failure Missions",    "tests/test_bcmp2_at3_5.py"),
    ]

    print("=" * 60)
    print("BCMP-2 Test Runner")
    print("=" * 60)

    total_passed = 0
    total_failed = 0
    t_start = time.time()

    for name, path in suites:
        print(f"\n[{name}]")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", path, "-v", "--tb=short"],
            capture_output=False,
        )
        if result.returncode == 0:
            print(f"  {name}: PASS")
            total_passed += 1
        else:
            print(f"  {name}: FAIL (exit code {result.returncode})")
            total_failed += 1

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"BCMP-2 Results: {total_passed} suites passed, "
          f"{total_failed} failed  ({elapsed:.1f}s)")
    print("=" * 60)

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
