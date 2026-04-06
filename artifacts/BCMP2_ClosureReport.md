# BCMP-2 Closure Report
**Programme:** MicroMind / NanoCorteX  
**Classification:** Programme Confidential  
**Date:** 06 April 2026  
**Author:** QA Agent (reviewed by Programme Director)  
**Status:** CLOSED — all 17 AT-6 gates PASS

---

## Executive Summary

BCMP-2 (Baseline Comparison Mission Profile 2) is hereby declared closed. The dual-track mission simulation framework has completed all acceptance testing across sprints SB-1 through SB-5. All 17 AT-6 gates pass, including a 4-hour overnight endurance run at 1483 missions with zero crashes, RSS slope 1.135 MB/hr, and log completeness 1.0000. The BCMP-2 framework provides the evidential baseline for pre-HIL programme review.

---

## Dual-Track Architecture Summary

BCMP-2 runs Vehicle A (baseline INS+GNSS → INS-only) and Vehicle B (full MicroMind stack) simultaneously from a shared seed, terrain, IMU noise model, and disturbance schedule. The comparative outcome — Vehicle A exceeding the C-2 drift corridor while Vehicle B maintains it — is the programme's primary SIL evidence for GNSS-denied navigation superiority.

Three canonical seeds were validated:
- Seed 42: nominal disturbance schedule
- Seed 101: stressed alternate weather profile
- Seed 303: virgin stress seed (not used in AT-1 through AT-5 or N=300 Monte Carlo)

All three seeds produce identical FSM phase transition chains (NOMINAL → EW_AWARE) and pass all C-2 envelope gates.

---

## C-2 Drift Envelope Results

C-2 envelopes were established by Monte Carlo N=300 simulation at the BASELINE IMU noise model. See SIL caveat BASELINE below.

| Boundary | Floor (P5) | Nominal | Ceiling (P99) | Seed 42 | Seed 101 | Seed 303 |
|---|---|---|---|---|---|---|
| km 60 | 5 m | 19 m | 80 m | PASS | PASS | PASS |
| km 100 | 12 m | 96 m | 350 m | PASS | PASS | PASS |
| km 120 | 15 m | 155 m | 650 m | PASS | PASS | PASS |

All 9 drift envelope gates (G-01 through G-09) PASS across all three seeds.

---

## SIL Limitations and Caveats

The following caveats are mandatory disclosures for any external use of BCMP-2 results. All five must be understood before presenting these results to TASL or any programme stakeholder.

**Caveat 1 — BASELINE IMU calibration**
The C-2 drift envelopes were calibrated using the BASELINE IMU noise model (ARW 0.05 °/√hr), not the STIM300 flight hardware (ARW 0.15 °/√hr). Results presented against these envelopes carry an implicit assumption that flight hardware noise will not exceed the calibration baseline. This must be re-validated when STIM300 hardware data is available. See OI-03.

**Caveat 2 — RADALT navigation correction stub**
Vehicle B navigation correction during ingress uses the legacy RADALT-NCC stub, not the orthophoto image matching mechanism adopted in AD-01 (03 April 2026). The production L2 correction mechanism (orthophoto matching, MAE < 7 m) is implemented as a measurement provider stub (Sprint C, commit 96bf98a) but is not yet integrated into the BCMP-2 mission runner. BCMP-2 drift results therefore do not reflect the navigation performance of the production architecture. See OI-05.

**Caveat 3 — DMRL decoy rejection is a rule-based stub**
The DMRL (Decoy-Resilient Multi-frame Lock) module used in all BCMP-2 terminal guidance results is a deterministic rule-based stub, not the CNN-based classifier specified for production. All terminal lock confidence values and decoy rejection results are stub outputs. No neural network inference occurs in any BCMP-2 result. See OI-06.

**Caveat 4 — Vehicle A is an illustrative drift model (AD-15)**
Vehicle A (baseline) uses a simplified INS drift model for comparative illustration. It is not a precision mechanisation of any specific inertial navigation unit. The Vehicle A trajectory is intended to demonstrate the consequence of GNSS denial on an unaugmented INS, not to characterise the performance of a specific baseline platform. See AD-15.

**Caveat 5 — EuRoC OpenVINS validation is indoor only**
The OpenVINS VIO component (Stage-2 GO verdict, 21 March 2026) was validated on EuRoC indoor datasets (MH_03, V1_01). Drift performance of 0.94–1.01 m/km was confirmed over short indoor sequences. Outdoor and km-scale validation is pending. EuRoC results must not be presented as evidence of mission-scale VIO performance. See OI-07.

---

## Programme State at Closure

| Sprint | Status | Gates |
|---|---|---|
| SB-1 | ✅ CLOSED | 17/17 AT-1 |
| SB-2 | ✅ CLOSED | 25/25 |
| SB-3 | ✅ CLOSED | 29/29 AT-2 + 19/19 AT-3/4/5 |
| SB-4 | ✅ CLOSED | Dashboard + Replay |
| SB-5 | ✅ CLOSED | 17/17 AT-6 |

**Total BCMP-2 gates:** 107/107  
**Regression baseline at closure:** 290 tests green (119 S5 + 68 S8 + 90 BCMP-2 + 7 RC + 6 ADV)  
**HEAD at closure:** to be filled by commit

---

## AT-6 Gate Evidence Summary

| Gate | Description | Result |
|---|---|---|
| G-01 | Seed 42 drift within C-2 at km 60 | ✅ PASS |
| G-02 | Seed 42 drift within C-2 at km 100 | ✅ PASS |
| G-03 | Seed 42 drift within C-2 at km 120 | ✅ PASS |
| G-04 | Seed 101 drift within C-2 at km 60 | ✅ PASS |
| G-05 | Seed 101 drift within C-2 at km 100 | ✅ PASS |
| G-06 | Seed 101 drift within C-2 at km 120 | ✅ PASS |
| G-07 | Seed 303 drift within C-2 at km 60 | ✅ PASS |
| G-08 | Seed 303 drift within C-2 at km 100 | ✅ PASS |
| G-09 | Seed 303 drift within C-2 at km 120 | ✅ PASS |
| G-10 | Seed 42 FSM phase chain matches canonical reference | ✅ PASS |
| G-11 | Seed 101 FSM phase chain matches canonical reference | ✅ PASS |
| G-12 | Seed 303 FSM phase chain matches canonical reference | ✅ PASS |
| G-13 | Zero crashes over 4-hour endurance (1483 missions) | ✅ PASS |
| G-14 | RSS slope 1.135 MB/hr ≤ 25 MB/hr over 4 hours | ✅ PASS |
| G-15 | Log completeness 1.0000 (1483/1483 missions) | ✅ PASS |
| G-16 | HTML reports rendered for all 3 seeds | ✅ PASS |
| G-17 | This closure report present with all 5 SIL caveats | ✅ PASS |

---

*BCMP-2 programme closed 06 April 2026. Next programme milestone: S-NEP-03 (EuRoC end-to-end with real MetricSet).*
