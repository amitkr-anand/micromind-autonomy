# HANDOFF — S-NEP-06 Closure
**Date:** 22 March 2026
**Status:** S-NEP-06 COMPLETE — analytically closed
**From:** S-NEP-06 execution and analysis session (22 March 2026)
**To:** Next programme phase session

---

## Critical Architecture Reminder

Two repositories. S-NEP-06 added artifacts to nep-vio-sandbox only.

| Repo | Latest commit | State |
|---|---|---|
| `micromind-autonomy` | ea1fe3e | Unchanged from S-NEP-04 — no modifications |
| `nep-vio-sandbox` | 4dd3a76 + S-NEP-06 logs | S-NEP-06 runners and logs added |

---

## Programme Invariants (Do Not Violate)

1. **Frozen constants are immutable**
   - `_ACC_BIAS_RW`, `_GYRO_BIAS_RW`, `_POS_DRIFT_PSD`, `_GNSS_R_NOMINAL`
   - Any modification requires explicit Technical Director approval

2. **EKF interface stability**
   - `update_vio()` is additive only
   - No changes to `propagate()`, `inject()`, or `update_gnss()` without approval

3. **Reproducibility is mandatory**
   - All evaluations must be runnable via committed scripts
   - All results must map to committed logs

4. **NIS interpretation constraint**
   NIS band [0.5, 8.0] remains valid for its intended estimator regime. PF-03 establishes that the current operating regime (high-rate VIO) does not satisfy the assumptions required for this band. NIS must not be tuned, reinterpreted, or adjusted. NIS is a diagnostic signal only under current architecture.

5. **Baseline anchoring**
   - All future evaluations must reference S-NEP-04, S-NEP-05, and S-NEP-06 logs
   - Any behavioural change must be compared against these baselines

6. **Analytical standard (established S-NEP-06)**
   - All conclusions must be derived from data, supported by explicit computation, and free from unverified assumptions
   - Statements not supportable by data must be declared explicitly as such

---

## Repo State at Closure

### micromind-autonomy (unchanged from S-NEP-04)

```
Latest commit: ea1fe3e
Branch: main (clean)
Test suite: 241 passed (222 S0–S10 + 19 S-NEP-04-A)
```

### nep-vio-sandbox

```
Latest commit on main: 4dd3a76 (S-NEP-05 logs)
S-NEP-06 artifacts: committed to logs/ directory
Branch: main
Test suite: 443/443
```

**S-NEP-06 artifacts (in fusion/logs/):**
```
Original outage runs (15 logs):
  s06_baseline_run{1,2,3}_06.json    — no-outage baseline
  s06_2s_run{1,2,3}_06.json
  s06_5s_run{1,2,3}_06.json
  s06_10s_run{1,2,3}_06.json
  s06_20s_run{1,2,3}_06.json

Controlled experiment logs:
  A_v101_low_o10s_ctrl2.json         — Block A: V1_01, velocity isolation
  A_mh01_mid_o10s_ctrl2.json         — Block A: MH_01
  A_mh03_high_o10s_ctrl2.json        — Block A: MH_03
  B_v101_str_o10s_ctrl2.json         — Block B: geometry isolation
  B_mh01_mix_o10s_ctrl2.json
  B_mh03_curv_o10s_ctrl2.json
  C_mh01_o{2,5,10,15,20,30}s_ctrl3.json  — Block C: regime mapping

Runners:
  fusion/run_06_vio_outage.py         — original outage runner (schema 06.1)
  fusion/run_06_euroc_controlled.py   — controlled segment runner (rev 2, schema 06e.2)
```

**Closure document:**
```
MicroMind_SNEP06_ClosureReport.docx  — uploaded to project knowledge
```

---

## S-NEP-06 Results Summary

### Original Runs — Outage and Recovery (full sequence, correct frame)

| Outage | Suppression | Gap detected | First err post | Recovery (τ=111mm) |
|---|---|---|---|---|
| 2s | ✅ | 2.03s | 34.9mm | Within 5s sustained |
| 5s | ✅ | 5.03s | 73.0mm | Within 5s sustained |
| 10s | ✅ | 10.03s | 66.0mm | Within 5s sustained |
| 20s | ✅ | 20.03s | 27.9mm | Within 5s sustained |

All original runs: no NaN, no divergence, full determinism (3/3 identical per configuration).

### Controlled Experiments — Analytically Defensible Conclusions

| ID | Conclusion | Evidence |
|---|---|---|
| C-01 | Loopback determined by trajectory geometry, not speed | V1_01 Block A vs B: same speed, different loopback |
| C-02 | Initial drift rate independent of outage duration | Std = 29 mm/s across 2s–30s early slopes |
| C-03 | Peak drift saturates at ~1230 mm for outages ≥10s | Block C: 1228–1232 mm for 10s–30s |
| C-04 | 5s→10s transition is trajectory-dependent, not filter regime | Filter parameters unchanged throughout |
| C-05 | Innovation at resumption is non-monotonic with duration | Loopback reduces INS-VIO gap at some endpoints |
| C-06 | MH_03: 7.06 m incr drift, linear model, R²=0.941, +800 mm/s | Block A direct measurement |
| C-07 | Recovery within 5s for all full-sequence outage durations | Causal rolling mean metric, original runs |
| C-08 | Outage suppression accurate within 30ms | Gap measurements across all configurations |

### Mandatory Non-Conclusions

- vel_err_m_s is not a valid steady-state observable in controlled runs (velocity state unconverged from zero-init)
- drift = velocity error × time is rejected (slope/vel_err ratios 0.09–0.70, inconsistent)
- The 5s→10s transition is not a filter regime change
- Recovery timing is not evaluable from controlled segment runs (origin mismatch)

---

## Key Programme Finding — PF-03 (unchanged)

NIS measurement model limitation persists. NIS is a diagnostic signal only. No corrective action within current architecture. Any change requires TD approval.

## Key Programme Finding — Velocity Observability (new, S-NEP-06)

The ESKF velocity state is weakly constrained by position-only VIO. Convergence is occurring (decreasing vel_err over time) but requires longer windows than the controlled segment runs provided (~60s from zero-init insufficient). This is expected behaviour, not a defect. Quantifying the steady-state velocity estimation error requires a dedicated longer-horizon experiment.

---

## Environment Notes

```
micromind-node01:
  Python: python3 → 3.12.3 (system)
  Repos: ~/micromind/repos/micromind-autonomy
         ~/micromind/repos/nep-vio-sandbox
  VIO benchmark arrays: ~/micromind/repos/nep-vio-sandbox/benchmark/
    MH_01: est_ts.npy, est_aligned.npy, sig_pos.npy (no prefix)
    MH_03: mh03_* prefix
    V1_01: v101_* prefix
  IMU/GT data: datasets/data/euroc/machine_hall/ and vicon_room1/
```

---

## Entry Checklist for Next Session

```bash
# micromind-autonomy
cd ~/micromind/repos/micromind-autonomy
git log --oneline main | head -3   # expect ea1fe3e at top
python3 -m pytest tests/ -q 2>&1 | tail -3  # expect 241 passed

# nep-vio-sandbox
cd ~/micromind/repos/nep-vio-sandbox
git log --oneline main | head -3   # expect 4dd3a76 at top
python3 -m pytest tests/ -q 2>&1 | tail -3  # expect 443/443
ls fusion/logs/ | grep ctrl | wc -l  # expect ≥ 12 ctrl log files
```

---

## Forward Look — Next Phase

S-NEP-06 closes the outage characterisation phase. The remaining open question identified by the analytical re-evaluation:

```
Velocity observability — quantifying steady-state velocity estimation error
under position-only VIO after sufficient convergence time.
```

This requires a longer-horizon experiment (convergence window >> 60s) with the velocity state pre-warmed. This is a candidate for a future programme phase, not a correction to S-NEP-06.

Beyond this, the programme questions for BCMP-1 integration and two-theatre architecture remain gated on the TASL partnership decision.

**Status:**
```
S-NEP-06: CLOSED — ANALYTICAL STANDARD ESTABLISHED
Next phase: TO BE DEFINED
```

