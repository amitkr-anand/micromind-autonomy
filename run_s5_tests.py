#!/usr/bin/env python3
"""
run_s5_tests.py
MicroMind Sprint S5 — Master Test Runner

Runs all Sprint S5 test suites in sequence:
  1. test_s5_dmrl.py       — DMRL boundary conditions + KPI-T01, KPI-T02
  2. test_s5_l10s_se.py    — L10s-SE decision tree + KPI-T03
  3. test_s5_bcmp1_runner.py — Full BCMP-1 acceptance gate (5× clean runs)

Usage:
  cd s5
  python run_s5_tests.py

Exit code 0 = all tests pass = Sprint S5 acceptance gate MET.
Exit code 1 = one or more failures.
"""

import sys
import os
import time
import unittest

# Add s5 root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_s5_dmrl         import (TestDMRLBoundaryConditions,
                                          TestDMRLSceneProcessing,
                                          TestDMRLKPIRequirements,
                                          TestDMRLResultStructure)
from tests.test_s5_l10s_se      import (TestL10sBoundaryConstants,
                                          TestL10sGate0ZPI,
                                          TestL10sGate1LockAcquired,
                                          TestL10sGate2Decoy,
                                          TestL10sGate3Civilian,
                                          TestL10sGate4Corridor,
                                          TestL10sTimingCompliance,
                                          TestL10sAuditLog,
                                          TestL10sDecisionPriority,
                                          TestInputsFromDMRL)
from tests.test_s5_bcmp1_runner import (TestBCMP1IndividualCriteria,
                                          TestBCMP1AcceptanceGate,
                                          TestBCMP1RunnerKPIExport,
                                          TestBCMP1Determinism,
                                          TestBCMP1EventLog)


BANNER = "=" * 70


def section(title: str):
    print(f"\n{BANNER}")
    print(f"  {title}")
    print(BANNER)


if __name__ == "__main__":
    t_start = time.perf_counter()

    print(f"\n{BANNER}")
    print("  MicroMind Sprint S5 — Full Test Suite")
    print("  DMRL (FR-103) + L10s-SE (FR-105) + BCMP-1 Acceptance Gate")
    print(BANNER)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    # ── Suite 1: DMRL ─────────────────────────────────────────────────────────
    section("Suite 1/3 — DMRL Boundary Conditions & KPIs")
    for cls in [TestDMRLBoundaryConditions, TestDMRLSceneProcessing,
                TestDMRLKPIRequirements, TestDMRLResultStructure]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    # ── Suite 2: L10s-SE ──────────────────────────────────────────────────────
    section("Suite 2/3 — L10s-SE Decision Tree & Timing Compliance")
    for cls in [TestL10sBoundaryConstants, TestL10sGate0ZPI,
                TestL10sGate1LockAcquired, TestL10sGate2Decoy,
                TestL10sGate3Civilian, TestL10sGate4Corridor,
                TestL10sTimingCompliance, TestL10sAuditLog,
                TestL10sDecisionPriority, TestInputsFromDMRL]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    # ── Suite 3: BCMP-1 Runner ────────────────────────────────────────────────
    section("Suite 3/3 — BCMP-1 End-to-End Acceptance Gate (5× runs)")
    for cls in [TestBCMP1IndividualCriteria, TestBCMP1AcceptanceGate,
                TestBCMP1RunnerKPIExport, TestBCMP1Determinism,
                TestBCMP1EventLog]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    # ── Run ───────────────────────────────────────────────────────────────────
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    elapsed = time.perf_counter() - t_start
    total   = result.testsRun
    passed  = total - len(result.failures) - len(result.errors)
    failed  = len(result.failures) + len(result.errors)

    print(f"\n{BANNER}")
    print(f"  Sprint S5 Test Summary")
    print(f"  Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Sprint S5 Acceptance Gate: {'✅ PASS' if result.wasSuccessful() else '❌ FAIL'}")
    print(BANNER + "\n")

    sys.exit(0 if result.wasSuccessful() else 1)
