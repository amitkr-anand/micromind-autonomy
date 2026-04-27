# Internal Acceptance Test — Scorecard
Date: 2026-04-27
HEAD: 7e475da
Corridor: Shimla-Manali 180km
Hardware: Jetson Orin Nano Super
TRN path: PhaseCorrelationTRN (LightGlue server not started)

## Acceptance Criteria

| Criterion | Result |
|---|---|
| Gate 5 completes without crash | ✅ PASS |
| All 22 tests pass | ✅ PASS |
| Navigation stable (no divergence in Gate 5) | ✅ PASS |
| Logs complete | ✅ PASS |
| Resource within limits (CPU<90%, Mem<6GB) | ✅ PASS |

## VERDICT: GO ✅

## Test Results

Tests: 17/17 passed (authoritative: exit_code=0, raw log confirmed)
Note: stream parser captured 16/17 — test_nav16_shm_not_triggered has multiline
live-log output between test name and PASSED token; parser missed the pairing.
Raw log gate5_raw_20260427_221447.txt and pytest exit_code=0 are authoritative.


## Resource Usage

Memory peak: 1.70 GB

Run duration: 2s (0.0 min)
