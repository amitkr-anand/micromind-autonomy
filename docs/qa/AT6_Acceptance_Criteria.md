# AT-6 Acceptance Test Specification
**Document:** AT6_Acceptance_Criteria.md  
**Governing sprint:** BCMP-2 SB-5  
**Supersedes:** "TBD" placeholder in 
  MicroMind_BCMP2_Implementation_Architecture_v1_1.md  
**Date:** 04 April 2026  
**Status:** APPROVED — governs SB-5 execution  
**Total gates:** 17

---

## 1. Purpose

AT-6 closes BCMP-2 with three-seed repeatability and overnight 
endurance evidence. It answers two questions:

1. Does the full MicroMind stack produce consistent, 
   deterministic results across three canonical seeds 
   representing nominal, alternate, and stress conditions?

2. Does the stack remain stable under 4-hour sustained 
   operation with no memory growth, no crashes, and no 
   log completeness failures?

AT-6 does not introduce new scenarios or new fault injection 
patterns. It re-runs the AT-2 structure (150 km dual-track 
mission) across seeds 42, 101, and 303, then executes an 
overnight endurance run.

---

## 2. Entry Criteria

All of the following must be confirmed before AT-6 execution 
begins. Do not proceed if any entry criterion is not met.

| # | Criterion | Verification method |
|---|---|---|
| EC-1 | `run_bcmp2_tests.py` green: 90/90 gates | Run and confirm output |
| EC-2 | `run_s5_tests.py` green: 111/111 gates | Run and confirm output |
| EC-3 | `run_s8_tests.py` green: 68/68 gates | Run and confirm output |
| EC-4 | SB-3 tag `sb3-full-mission-reports` present in git log | `git tag -l` |
| EC-5 | SB-4 tag `sb4-dashboard-replay` present in git log | `git tag -l` |
| EC-6 | Seeds 42 and 101 pass AT-2 C-2 gates individually (established in SB-3) | BCMP2_STATUS.md AT-2 results |
| EC-7 | Seed 303 has not been used in any prior calibration or test run | Confirm in BCMP2_JOURNAL.md |
| EC-8 | micromind-node01 has ≥ 10 GB free disk space for overnight log retention | `df -h /home` |
| EC-9 | UPS operational — overnight run must not be interrupted by power event | Visual confirmation |

---

## 3. Test Stimuli

### 3.1 Three-Seed Repeatability Run (AT-6-A)
Run the full BCMP-2 dual-track mission for each of the three 
canonical seeds in sequence:

| Seed | Profile | Purpose |
|---|---|---|
| 42 | Nominal | Baseline — established in SB-3 |
| 101 | Alternate weather | Stress — established in SB-3 |
| 303 | Stress | New — not used in any prior calibration |

Parameters for each run: `max_km=150`, `imu_name=STIM300`, 
fault injection disabled. Same parameters as AT-2.

### 3.2 Overnight Endurance Run (AT-6-B)
Run the full BCMP-2 dual-track mission continuously for 4 hours 
using seed 42 (nominal). The runner loops the 150 km mission 
repeatedly. RSS memory is sampled every 60 seconds.

---

## 4. Procedure

### AT-6-A: Three-Seed Repeatability
```bash
conda activate micromind-autonomy

# Confirm entry criteria
python run_s5_tests.py
python run_s8_tests.py
python run_bcmp2_tests.py

# Run AT-6-A
PYTHONPATH=. python tests/test_bcmp2_at6.py --mode repeatability \
    --seeds 42 101 303 \
    --imu STIM300 \
    --output artifacts/at6_repeatability_results.json
```

For each seed the runner must:
1. Execute the full 150 km dual-track mission
2. Record drift at km 60, 100, 120 for both Vehicle A and 
   Vehicle B
3. Record the phase transition chain for Vehicle B
4. Confirm C-2 envelope gates at each boundary
5. Generate the standard mission report via `bcmp2_report.py`

### AT-6-B: Overnight Endurance
```bash
PYTHONPATH=. python tests/test_bcmp2_at6.py --mode endurance \
    --seed 42 \
    --duration-hours 4 \
    --rss-interval-s 60 \
    --output artifacts/at6_endurance_results.json
```

The endurance runner must:
1. Loop the 150 km mission for 4 continuous hours
2. Sample RSS every 60 seconds and write to log
3. Detect and report any crash or OOM kill
4. Confirm log completeness ≥ 99% across the full run
5. Terminate cleanly at 4 hours and write final report

---

## 5. Gate Definitions

### Group 1 — Drift Envelope Gates (9 gates)
One gate per seed per km boundary. Each gate passes if 
Vehicle B cross-track error falls within the C-2 envelope 
for that boundary.

| Gate | Seed | Boundary | Floor | Ceiling | Pass condition |
|---|---|---|---|---|---|
| G-01 | 42 | km 60 | 5 m | 80 m | drift_at_km60_m ∈ [5, 80] |
| G-02 | 42 | km 100 | 12 m | 350 m | drift_at_km100_m ∈ [12, 350] |
| G-03 | 42 | km 120 | 15 m | 650 m | drift_at_km120_m ∈ [15, 650] |
| G-04 | 101 | km 60 | 5 m | 80 m | drift_at_km60_m ∈ [5, 80] |
| G-05 | 101 | km 100 | 12 m | 350 m | drift_at_km100_m ∈ [12, 350] |
| G-06 | 101 | km 120 | 15 m | 650 m | drift_at_km120_m ∈ [15, 650] |
| G-07 | 303 | km 60 | 5 m | 80 m | drift_at_km60_m ∈ [5, 80] |
| G-08 | 303 | km 100 | 12 m | 350 m | drift_at_km100_m ∈ [12, 350] |
| G-09 | 303 | km 120 | 15 m | 650 m | drift_at_km120_m ∈ [15, 650] |

**Note:** Vehicle A is expected to breach the corridor by km 150 
for seeds 101 and 303 (stress profiles). Vehicle A breach is a 
PASS signal for the dual-track comparison — it demonstrates 
the correction value. Vehicle A breach does not fail any gate.

### Group 2 — Phase Transition Consistency Gates (3 gates)
One gate per seed. The Vehicle B phase transition chain 
(sequence of FSM state transitions during the 150 km mission) 
must be identical across all three seeds. Seed 42 is the 
reference chain established in SB-3.

| Gate | Seed | Pass condition |
|---|---|---|
| G-10 | 42 | Phase chain matches SB-3 reference (identity check) |
| G-11 | 101 | Phase chain matches seed 42 chain for this run |
| G-12 | 303 | Phase chain matches seed 42 chain for this run |

**Note:** "Identical" means the same sequence of state IDs in 
the same order. Timing differences are permitted — the chain 
structure must match, not the timestamps.

### Group 3 — Endurance Stability Gates (3 gates)

| Gate | Metric | Pass condition | Fail condition |
|---|---|---|---|
| G-13 | Process stability | Zero crashes and zero OOM kills across 4-hour run | Any crash or OOM kill at any point |
| G-14 | Memory growth | RSS linear regression slope ≤ 25 MB/hour over 4-hour run | Slope > 50 MB/hour, or any OOM kill |
| G-15 | Log completeness | log_completeness ≥ 99% measured at end of run | log_completeness < 95% at any 60-second sample |

### Group 4 — Report and Closure Gates (2 gates)

| Gate | Requirement | Pass condition |
|---|---|---|
| G-16 | Final HTML report | Report generated for all three seeds; business comparison block appears before technical tables (§8.3 ordering); file < 5 MB; no external dependencies |
| G-17 | BCMP-2 Closure Report | Document exists at `artifacts/BCMP2_ClosureReport.md`; contains: executive summary, dual-track comparison result, C-2 envelope summary for all three seeds, SIL completeness caveats per QA log, and programme state at closure |

---

## 6. Pass and Fail Conditions

### Overall AT-6 Pass
All 17 gates (G-01 through G-17) must be in PASS state.  
AT-6 pass is the SB-5 exit condition.

### Overall AT-6 Fail
Any single gate in FAIL state fails AT-6.  
No partial pass is defined — gates are binary.

### Retry Policy
- **Group 1 and 2 gates (G-01 through G-12):** No retry. 
  A failing seed result requires root-cause investigation 
  before re-run. The failing seed, gate, observed value, 
  and C-2 envelope must be logged.
- **Group 3 gates (G-13 through G-15):** No retry for 
  crashes or OOM kills. Memory growth slope failure: 
  investigate and fix before re-run.
- **Group 4 gates (G-16, G-17):** Fix the report generation 
  and re-run report only — no need to re-run missions.

---

## 7. Required Logs and Artefacts

All of the following must be present and committed before 
SB-5 is declared closed:

| Artefact | Path | Contents |
|---|---|---|
| Repeatability results | `artifacts/at6_repeatability_results.json` | Gate results for all 9 drift gates and 3 chain gates, per seed |
| Endurance results | `artifacts/at6_endurance_results.json` | RSS trace (60 s intervals), crash log (empty if clean), log_completeness trace |
| Mission reports | `artifacts/bcmp2_report_seed42_at6.html` | Standard mission report, seed 42 |
| Mission reports | `artifacts/bcmp2_report_seed101_at6.html` | Standard mission report, seed 101 |
| Mission reports | `artifacts/bcmp2_report_seed303_at6.html` | Standard mission report, seed 303 |
| Closure report | `artifacts/BCMP2_ClosureReport.md` | Full programme closure document (G-17) |
| AT-6 run log | `artifacts/at6_run.log` | Full console output from both AT-6-A and AT-6-B runs |

---

## 8. SIL Caveats — Mandatory in Closure Report

The BCMP-2 Closure Report (G-17) must include the following 
caveats. Omitting any caveat fails G-17.

1. **IMU baseline:** C-2 drift envelopes were calibrated on the 
   BASELINE IMU noise model (ARW 0.05 °/√hr). STIM300 typical 
   ARW is 0.15 °/√hr. Re-calibration with STIM300 profile is 
   pending (OI-03). All envelope results must be read as 
   BASELINE-calibrated, not STIM300-calibrated.

2. **Navigation correction mechanism:** The TRN/L2 correction 
   stub currently implements RADALT-NCC (superseded by AD-01). 
   Orthophoto matching implementation is pending (OI-05). 
   All AT-2 and AT-6 navigation results reflect the 
   RADALT-NCC correction mechanism, not the adopted 
   orthophoto architecture.

3. **DMRL is a rule-based stub:** All terminal guidance 
   results in BCMP-1 and BCMP-2 use a rule-based DMRL stub 
   with synthetic thermal inputs. CNN implementation is 
   Phase-2. No CNN-based decoy rejection has been tested.

4. **Vehicle A is illustrative:** Vehicle A drift results 
   represent an illustrative INS-only propagation model, 
   not a precision simulation of a specific airframe's 
   inertial performance (AD-15).

5. **OpenVINS validation scope:** VIO performance results 
   are based on indoor EuRoC sequences (≤ 130 m). 
   Km-scale and outdoor validation are pending (OI-07). 
   EuRoC results must not be presented as mission-scale 
   evidence.

---

## 9. Post-Test State

On AT-6 PASS:
- Tag the repository: `sb5-bcmp2-closure`
- Update `BCMP2_STATUS.md`: SB-5 status → ✅ CLOSED
- Update `MICROMIND_PROJECT_CONTEXT.md` Section 6: 
  SB-5 → ✅ CLOSED with tag
- Update `MICROMIND_PROJECT_CONTEXT.md` Section 8: 
  OI-19 → CLOSED
- Append QA log entry QA-003

On AT-6 FAIL:
- Do not tag
- Log the failing gate(s), observed values, and 
  expected range in BCMP2_JOURNAL.md
- Investigate root cause before re-run
- Do not update programme state until pass is achieved

---

## 10. Relationship to Prior Acceptance Tests

| AT | Seeds used | Purpose | Relation to AT-6 |
|---|---|---|---|
| AT-1 | 42 | Boot + 5 km regression | Entry gate for SB-1; not repeated |
| AT-2 | 42, 101 | 150 km dual-track, nominal + alternate | AT-6 re-runs this structure for all 3 seeds |
| AT-3 | 42 | Single-failure mission | Not repeated in AT-6 |
| AT-4 | 42 | Multi-failure mission | Not repeated in AT-6 |
| AT-5 | 42 | Terminal integrity | Not repeated in AT-6 |
| **AT-6** | **42, 101, 303** | **Repeatability + endurance** | **SB-5 closure gate** |

Seed 303 has not been used in AT-1 through AT-5 or in the 
N=300 Monte Carlo calibration. It is reserved exclusively 
for AT-6 stress validation.

---
*This specification is the governing document for AT-6. 
Any deviation from these gate definitions requires 
Programme Director approval and a documented rationale 
in BCMP2_JOURNAL.md before the test run begins.*
