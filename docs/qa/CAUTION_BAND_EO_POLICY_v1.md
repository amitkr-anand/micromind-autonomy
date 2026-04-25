# CAUTION-Band EO Correction Policy v1.0
Authority: Deputy 1 | Date: 26 April 2026

## Current State (SAL-2)
SAL-2 establishes per-class LightGlue confidence thresholds.
CAUTION band currently: single observation injects with 0.5x EKF weight.
No temporal validation requirement.

## Revised Policy (aligned with SAL-3 Baseline v1.0)

### Gate 1 — Temporal consistency (NEW):
Minimum two CAUTION observations within CAUTION_CONSISTENCY_WINDOW_KM
must be directionally consistent (bearing within CAUTION_MAX_BEARING_DEVIATION_DEG)
before first injection. Single observation = candidate only, not correction.
Identical principle to SAL-3 Gate 1.

### Gate 2 — Residual character:
Correction vector must not directionally oppose established INS drift vector.
Identical principle to SAL-3 Gate 3.

### Weight assignment:
CAUTION corrections use CAUTION_EKF_WEIGHT (candidate: 0.4x).
Named constant in config — not hardcoded.
Distinct from ACCEPT (1.0x) and SAL-3 (0.3x).

## Source Priority Hierarchy (revised)
LightGlue ACCEPT → LightGlue CAUTION (2-obs gate) → 
SAL-3 (validated) → INS-only with drift monitor

## Implementation Note
This policy must be implemented in NavigationManager BEFORE
SAL-3 implementation begins. This is SAL-3 prerequisite 9.
The temporal gate adds ~1 correction interval latency for first
CAUTION injection but eliminates single-observation false positives.

## Parameters Requiring Definition Before Implementation
- CAUTION_CONSISTENCY_WINDOW_KM (candidate: 3km)
- CAUTION_MAX_BEARING_DEVIATION_DEG (candidate: 30 degrees)
- CAUTION_EKF_WEIGHT (candidate: 0.4)
