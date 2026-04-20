# HANDOFF — S-NEP-05 Closure
**Date:** 22 March 2026
**Status:** S-NEP-05 COMPLETE — mission-scale validation closed
**From:** S-NEP-05 execution session (22 March 2026)
**To:** Next programme phase session

---

## Critical Architecture Reminder

Two repositories. S-NEP-05 added artifacts to nep-vio-sandbox only.

| Repo | Latest commit | State |
|---|---|---|
| `micromind-autonomy` | ea1fe3e | Unchanged from S-NEP-04 — no modifications |
| `nep-vio-sandbox` | 4dd3a76 | S-NEP-05 runner + 3 logs added |

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
   NIS band [0.5, 8.0] remains valid for its intended estimator regime. PF-03 establishes that the current operating regime (high-rate VIO) does not satisfy the assumptions required for this band. Therefore:
   - NIS must NOT be tuned to satisfy the band
   - NIS must NOT be reinterpreted
   - NIS must be treated as a diagnostic signal only under current architecture
   - Any modification to R, measurement model, or NIS applicability requires explicit Technical Director approval

5. **Baseline anchoring**
   - All future evaluations must reference S-NEP-04 logs (04-B, 04-C, 04-D) and S-NEP-05 logs
   - Any behavioural change must be compared against these baselines

6. **Bounded behaviour scope constraint**
   - "Bounded within 138s observation window" is the correct statement
   - "Estimator is globally bounded" must NOT be stated
   - This distinction is preserved for future phases

---

## Repo State at Closure

### micromind-autonomy (unchanged)

```
Latest commit: ea1fe3e
Branch: main (clean)
Test suite: 241 passed (222 S0–S10 + 19 S-NEP-04-A)
```

### nep-vio-sandbox

```
Latest commit: 4dd3a76
Branch: main (clean)
Test suite: 443/443
```

**S-NEP-05 artifacts:**
```
fusion/
  run_05_mission_scale.py     ← runner rev 2 (aligned ATE + R² gating)
  logs/
    mh01_s05_run1_05.json     ← run 1 (schema 05.2)
    mh01_s05_run2_05.json     ← run 2
    mh01_s05_run3_05.json     ← run 3
```

**Log schema:** `05.2` — contains aligned ATE per window, unaligned error time series,
drift classification with R² values, per-update signals (t, type, error_m, innov_mag, trace_P, nis).

---

## S-NEP-05 Results Summary

### Anchor (baseline consistency)

| Metric | Value |
|---|---|
| S-NEP-05 aligned ATE RMSE (steady) | 0.0800 m |
| S-NEP-04 04-C baseline | 0.0795 m |
| Delta | +0.0005 m (+0.6%) |
| Status | **CONSISTENT** |

### Drift Classification (authoritative — full window, 10s–138s)

| Metric | Value |
|---|---|
| Regime | **BOUNDED (within observed window)** |
| Confidence | Moderate |
| R²_linear | 0.149 (below floor 0.30 — no trend detectable) |
| Drift rate | −27.3 mm/min (negative — correction dominant) |
| Running max growth | 123.6mm |

### EC Summary

| ID | Condition | Result |
|---|---|---|
| EC-01 | Stability — no NaN/reset/divergence | ✅ PASS |
| EC-02 | Drift behaviour — bounded within window | ✅ PASS (moderate confidence) |
| EC-03 | Correction effectiveness | ✅ PASS — negative drift slope confirms active correction |
| EC-04 | Covariance — stable grow/contract | ✅ PASS |
| EC-05 | Repeatability — deterministic | ✅ PASS — 3/3 identical |
| EC-06 | ATE within S-NEP-04 envelope | ✅ PASS |

### Programme-Level Statement

The estimator demonstrates stable, non-accumulating behaviour over mission-scale duration (138s) with preserved accuracy and full determinism.

Bounded behaviour is established only within the 138s observation window. No globally bounded claim is made.

---

## Classifier Design (for reference)

The corrected drift classifier introduced in S-NEP-05 uses R² gating:

```
R² floor = 0.30 (moderate confidence threshold)
R² strong = 0.50

Rules:
  No model meets R² floor → BOUNDED (default)
  Linear R² >= floor AND slope > 0.5mm/min → LINEAR
  Quad R² >= floor AND F-test AND linear base meets floor → ACCELERATING

F-test only applied if linear model already meets floor.
```

This classifier is committed in `run_05_mission_scale.py` and must be reused in future phases for consistency.

---

## Known Limitations

**Observation window:** 138s is the full extent of the MH_01_easy dataset. Drift regime classification confidence is moderate — the window may be insufficient to detect slow linear trends. A longer dataset would increase classification confidence.

**PF-03 persists:** NIS remains structurally suppressed under high-rate VIO. Not a defect — documented limitation of the measurement model.

---

## Entry Checklist for Next Session

```bash
# micromind-autonomy
cd ~/micromind/repos/micromind-autonomy
git log --oneline main | head -3    # expect ea1fe3e at top
grep -n "def update_vio" core/ekf/error_state_ekf.py  # expect line ~208
python3 -m pytest tests/ -q 2>&1 | tail -3  # expect 241 passed

# nep-vio-sandbox
cd ~/micromind/repos/nep-vio-sandbox
git log --oneline main | head -3    # expect 4dd3a76 at top
python3 -m pytest tests/ -q 2>&1 | tail -3  # expect 443/443
```

---

## Forward Look — Next Phase

S-NEP-05 closes the mission-scale validation phase. The next programme question is:

```
What happens when VIO reliability degrades or drops out?
```

This question probes the estimator's robustness under degraded sensor conditions — a necessary step before declaring the system mission-ready. Scope and definition for this phase are to be determined.

**Status:**
```
Next phase: TO BE DEFINED
S-NEP-05: CLOSED — STABLE MISSION-SCALE BEHAVIOUR CONFIRMED
```
