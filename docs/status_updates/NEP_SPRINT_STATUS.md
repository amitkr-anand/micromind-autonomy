# NEP VIO Sandbox — Sprint Status
**Last Updated:** 25 March 2026 (S-NEP-10 enforcement COMPLETE)
**Active Sprint:** None — enforcement layer committed. Next phase: BCMP-1 two-theatre integration.
**Current Step:** —
**GitHub:** amitkr-anand/nep-vio-sandbox  |  amitkr-anand/micromind-autonomy
**Branch:** main (both repos)
**Latest commit (micromind-autonomy):** d3afdd1 (S-NEP-10 enforcement: E-1 through E-5 in bcmp1_runner.py)
**Latest commit (nep-vio-sandbox):** 4fcf231 (S-NEP-09: robustness/signal/boundary runners + all execution logs)
**Environment:** Python 3.12.3 (system) / Ubuntu 24.04.4 / micromind-node01

---

## Sprint History

| Sprint | Commit | Gate | Delivered |
|---|---|---|---|
| S-NEP-01 | c3d5b45 | ✅ 413/413 | Fix 10 failing tests |
| S-NEP-02 | f3c83a3 | ✅ 424/424 | DatasetManager.build_dataset_ref() implemented |
| S-NEP-03 | 750660c | ✅ 443/443 | OpenVINS end-to-end pipeline on EuRoC MH_01_easy |
| S-NEP-04 | 154c860 | ✅ CLOSED | VIO integration with MicroMind fusion layer |
| S-NEP-05 | 4dd3a76 | ✅ CLOSED | Mission-scale validation — stable bounded behaviour confirmed |
| S-NEP-06 | d090851 | ✅ CLOSED | VIO outage and recovery — controlled characterisation, analytical standard established |
| S-NEP-07 | — (doc) | ✅ BASELINED | Engineering implications and decision framework — Rev 3 approved |
| S-NEP-08 | 542867e / 30c2d56 | ✅ CLOSED | VIO navigation mode framework — L-07/L-08/L-05, 311/311 passing |
| S-NEP-09 | 7c8c92d / 4fcf231 | ✅ CLOSED | Operational behaviour validation — 332/332 passing, 4 findings logged |
| S-NEP-10 (doc) | — | ✅ BASELINED (Rev 2) | Mission & system integration rules — 26 deterministic rules across 8 sections |
| S-NEP-10 (impl) | d3afdd1 | ✅ CLOSED | E-1 through E-5 enforcement in bcmp1_runner.py — 332/332 passing |

---

## VIO Selection Programme — Final State

**Governing standard:** MicroMind VIO Selection Standard v1.2
**Programme phase:** Selection COMPLETE → Endurance Validation COMPLETE → Integration COMPLETE

### Candidate Final Status

| Candidate | Classification | Reason |
|---|---|---|
| OpenVINS | **GO — System-Level Clearance** | All Stage-1, Stage-2, and S-NEP-04 integration criteria met. |
| RTAB-Map | NO-GO — Stage-1 Disqualified | RPE 5s failure + covariance type mismatch (PF-01) |
| ORB-SLAM3 | NOT READY — Covariance Gate Failed | No covariance by design |
| VINS-Fusion | DEFERRED — Environment Mismatch | No stable ROS2 Humble port |
| Kimera-VIO | DEFERRED — Pacing Rule | Integration exceeds 2-session budget |

### Programme Findings (Active)

- **PF-01:** Covariance type requirement — global state covariance required, not incremental
- **PF-02:** Covariance availability gap — most open-source VO systems not fusion-ready under v1.2
- **PF-03:** NIS measurement model limitation — absolute R (σ ≈ 87mm) vs relative inter-frame innovation (~2–66mm at high VIO rates). NIS band [0.5, 8.0] remains valid for its intended estimator regime; PF-03 establishes that the current operating regime (high-rate VIO) does not satisfy the assumptions required for this band. NIS responds correctly to rate changes (monotonic, confirmed 04-D). **NIS must not be tuned, reinterpreted, or adjusted. NIS is a diagnostic signal only under current architecture. Any modification to R, measurement model, or NIS applicability requires explicit Technical Director approval.**

---

## OpenVINS Stage-2 Results (Complete)

### Cross-Sequence Summary

| Metric | MH_01_easy (S1) | MH_03_medium (S2) | V1_01_easy (S2) | GO Limit |
|---|---|---|---|---|
| ATE RMSE (m) | 0.087 | 0.135 | 0.061 | < 1.0 |
| Drift proxy (m/km) | 1.01 | 0.94 | 0.97 | < 10 |
| Drift R² | — | 0.064 | 0.003 | — |
| RPE 5s mean (m) | 0.068 | 0.124 | 0.055 | < 0.3 |
| NIS mean | 0.426 CONSISTENT | 1.247 CONSISTENT | 0.357 CONSISTENT | [0.3–3.0] |
| Update rate (Hz) | 126.0 | 121.9 | 122.9 | ≥ 20 |
| Tracking loss | 3.3% | 3.5% | 3.0% | < 5% |
| FM events | 0 + 0 | 0 + 0 | 0 + 0 | 0 |

**Primary Stage-2 finding:** Drift 0.94–1.01 m/km across three sequences, two environments — 3.6% variance. Cross-sequence consistency is the primary GO signal per v1.2.

### Stage-2 Benchmark Files (in repo)

```
benchmark/
  openvins_metrics.json       — MH_01_easy Stage-1 metrics
  mh03_metrics.json           — MH_03_medium Stage-2 metrics
  v101_metrics.json           — V1_01_easy Stage-2 metrics
  plots_mh03/                 — 4 plots: trajectory, error, drift, NIS
  plots_v101/                 — 5 plots: trajectory, error, drift, NIS, cross-sequence
  [arrays].npy                — est_aligned, gt_interp, errors, est_ts, nis, sig_pos, gt_arc
                                per sequence (prefix: none / mh03_ / v101_)
```

---

## S-NEP-04 — CLOSED

**Objective:** Establish clean, stable, and verifiable integration of OpenVINS into the MicroMind navigation pipeline.

**Verdict: COMPLETE — Functional baseline established. All ATE and stability criteria met.**

### Step Summary

| Step | Status | Commit (nep-vio-sandbox) | Commit (micromind-autonomy) | Delivered |
|---|---|---|---|---|
| 04-A | ✅ CLOSED | — | ea1fe3e | update_vio(), core/fusion/, T-01+T-02 gates (19/19) |
| 04-B | ✅ CLOSED | b7adf11 | — | Offline replay, ATE 0.0865 m, EC-01 PASS, zero IFM |
| 04-C | ✅ CLOSED | 154c860 | — | IMU+VIO fusion, ATE 0.0795 m, NIS limitation identified and classified |
| 04-D | ✅ CLOSED | c875356 | — | Rate-response validation, NIS monotonic behaviour confirmed |

### Final Results

| Phase | Config | ATE (steady) | NIS mean (steady) | IFM events | EC-01 |
|---|---|---|---|---|---|
| 04-B | Offline replay, no IMU | 0.0865 m | Not evaluable (no propagation) | 0 | ✅ PASS |
| 04-C | IMU+VIO, ~124 Hz | 0.0795 m | 0.000353 — NOT APPLICABLE (see PF-03) | 0 | ✅ PASS |
| 04-D stride 6 | IMU+VIO, ~21 Hz | 0.0805 m | 0.016983 — 48× 04-C | 0 | ✅ PASS |
| 04-D stride 12 | IMU+VIO, ~10 Hz | 0.0856 m | 0.033515 — 95× 04-C | 0 | ✅ PASS |

### EC Summary

| ID | Condition | Verdict |
|---|---|---|
| EC-01 | ATE ≤ 0.174 m | ✅ PASS — all phases |
| EC-02 | NIS [0.5, 8.0] | NOT APPLICABLE — PF-03; rate-response correct |
| EC-03 | Covariance grow/contract | ✅ PASS — confirmed 04-C diagnostic |
| EC-04 | IFM-01/04 = 0 | ✅ PASS — zero IFM events across all runs |
| EC-05 | No NaN / reset | ✅ PASS |
| EC-06 | std(ATE) < 5% | ✅ PASS — perfect determinism, std = 0.000 m |

### Closure Statement

S-NEP-04 establishes a verified fusion baseline.

The estimator has been validated across:
- measurement-dominant regime (high-rate VIO)
- prediction-influenced regime (downsampled VIO)

All core behaviours (propagation, update, covariance evolution, stability, determinism) have been confirmed through controlled experiments.

No open defects remain.

### Deliverables

**micromind-autonomy (ea1fe3e):**
- `core/ekf/error_state_ekf.py` — `update_vio()` added (additive, frozen constants untouched)
- `core/fusion/` — `__init__.py`, `vio_covariance_error.py`, `frame_utils.py`, `fusion_logger.py`
- `tests/test_s_nep_04a_interface.py` — 19/19 (T-01 frame sanity + T-02 covariance extraction)

**nep-vio-sandbox (154c860):**
- `fusion/run_04b_offline.py` — offline replay runner
- `fusion/run_04c_imu_vio.py` — IMU+VIO fusion runner (200Hz IMU + 124Hz VIO)
- `fusion/run_04d_downsampled.py` — stride-based downsampling runner
- `fusion/logs/` — 12 JSON observability logs (mh01 ×3 per phase: 04-B, 04-C, 04-D ×2)

---

## Dataset State

| Dataset | ASL | rosbag2 | DatasetManager | Path |
|---|---|---|---|---|
| EuRoC MH_01_easy | ✅ | ✅ | ✅ | datasets/data/euroc/MH_01_easy |
| EuRoC MH_03_medium | ✅ | ✅ | ✅ | datasets/data/euroc/MH_03_medium |
| EuRoC V1_01_easy | ✅ | ✅ | ✅ | datasets/data/euroc/V1_01_easy |

**Pose capture paths:**
```
MH_01:  datasets/data/euroc/machine_hall/MH_01_easy/MH_01_easy_openvins_poses.txt
MH_03:  datasets/data/euroc/machine_hall/MH_03_medium/MH_03_openvins_poses.txt
V1_01:  datasets/data/euroc/vicon_room1/vicon_room1/V1_01_easy/V1_01_openvins_poses.txt
```

---

## Key Documents (Project Knowledge)

| Document | Purpose | Status |
|---|---|---|
| MicroMind_VIO_Selection_Programme_v1_2.docx | Governing standard | Baseline — do not modify |
| OpenVINS_Stage1_Qualified_v1_2.docx | Stage-1 evidence record | Complete |
| OpenVINS_Endurance_Validation_Report_Stage2_v4.docx | Stage-2 GO record | Complete (images to insert offline) |
| RTABMap_Stage1_Evaluation_Record.docx | RTAB-Map NO-GO record | Complete |
| MicroMind_VIO_Stage1_Consolidated_Stage2_Transition_v2.docx | Programme state record | Complete |
| MicroMind_VIO_Integration_Plan_S-NEP-04.docx | Integration plan | **Active — drives execution** |

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

## S-NEP-05 — CLOSED

**Objective:** Mission-scale validation — stable estimator behaviour under sustained operation.

**Verdict: COMPLETE — Stable, non-accumulating behaviour confirmed within 138s observation window.**

### Results

| Metric | Value | Status |
|---|---|---|
| Aligned ATE RMSE (steady) | 0.0800 m | ✅ EC-06 PASS — +0.6% vs 04-C baseline |
| 04-C baseline ATE | 0.0795 m | Reference |
| Drift regime (full window) | BOUNDED (within observed window) | ✅ EC-02 |
| R²_linear | 0.149 (below floor 0.30) | No trend detectable |
| Drift rate | −27.3 mm/min (negative — correction dominant) | |
| IFM events | 0 | ✅ EC-01 |
| NaN / divergence | None | ✅ EC-01 |
| Repeatability | All 3 runs identical | ✅ EC-05 |

### Programme-Level Statement

The estimator demonstrates stable, non-accumulating behaviour over mission-scale duration (138s) with preserved accuracy and full determinism.

### Classification Basis

No linear or quadratic trend detected within the 138s window. Negative drift slope (−27.3 mm/min) indicates VIO corrections are dominant — error is decreasing on average, not accumulating. Error envelope oscillates without systematic growth.

**Scope constraint (mandatory):** Bounded behaviour is established only within the 138s observation window. The statement "estimator is globally bounded" must not be made. This distinction is preserved for future phases.

### EC Summary

| ID | Condition | Result |
|---|---|---|
| EC-01 | Stability — no NaN/reset/divergence | ✅ PASS |
| EC-02 | Drift behaviour — bounded within window | ✅ PASS (moderate confidence) |
| EC-03 | Correction effectiveness — VIO active | ✅ PASS — negative drift slope |
| EC-04 | Covariance — stable grow/contract | ✅ PASS (architecture unchanged from 04-C) |
| EC-05 | Repeatability — deterministic | ✅ PASS — 3/3 runs identical |
| EC-06 | ATE within S-NEP-04 envelope | ✅ PASS — 0.0800m vs 0.0795m |

### Deliverable

```
fusion/run_05_mission_scale.py    — runner (rev 2: aligned ATE, R² gating)
fusion/logs/mh01_s05_run1_05.json
fusion/logs/mh01_s05_run2_05.json
fusion/logs/mh01_s05_run3_05.json
```

### Forward Look

Next phase question (not yet scoped):
```
What happens when VIO reliability degrades or drops out?
```

**Status:**
```
S-NEP-05: CLOSED — STABLE MISSION-SCALE BEHAVIOUR CONFIRMED
```

---

## S-NEP-06 — CLOSED

**Objective:** Characterise estimator behaviour under VIO outage — drift during outage, recovery at resumption, and causal attribution of drift to trajectory geometry vs velocity.

**Verdict: ANALYTICALLY CLOSED — Mathematically defensible conclusions established. Analytical standard set for future phases.**

### Programme Standard (Established in S-NEP-06)

All conclusions in this and future phases must be:
- derived from data
- supported by explicit computation
- free from unverified assumptions

Statements not supportable by data must be declared explicitly as such.

### Experiments Executed

| Block | Purpose | Sequences | Outage |
|---|---|---|---|
| Original runs | Outage/recovery baseline | MH_01_easy | 2s, 5s, 10s, 20s × 3 runs each |
| Block A (ctrl) | Velocity isolation | V1_01, MH_01, MH_03 | 10s |
| Block B (ctrl) | Geometry isolation | V1_01, MH_01, MH_03 | 10s |
| Block C (ctrl) | Regime mapping | MH_01 | 2s, 5s, 10s, 15s, 20s, 30s |

### Analytically Defensible Conclusions

| ID | Conclusion |
|---|---|
| C-01 | Loopback is determined by trajectory geometry (which interval the outage covers), not speed alone |
| C-02 | Initial drift rate is independent of outage duration (std = 29 mm/s across 2s–30s) |
| C-03 | Peak incremental drift saturates at ~1230 mm for outages ≥10s at the tested placement |
| C-04 | The 5s→10s behavioural transition is trajectory-dependent, not a filter regime change |
| C-05 | Innovation at resumption is non-monotonic with outage duration (loopback reduces the gap) |
| C-06 | MH_03 shows 7.06 m incremental drift under 10s outage (linear, R²=0.941, +800 mm/s) |
| C-07 | In full-sequence runs, causal rolling mean returned within τ=111 mm within 5s for all durations |
| C-08 | Outage suppression operates correctly (gap accuracy within 30ms of commanded duration) |

### Mandatory Non-Conclusions (Explicitly Not Supported)

- vel_err_m_s is not a valid steady-state velocity observable (velocity state unconverged in controlled runs)
- δv/v ratios reported in earlier analysis are invalid — dominated by zero-initialisation
- drift = velocity error × time is rejected (slope/vel_err ratios: 0.09–0.70, inconsistent)
- Velocity contribution to drift cannot be isolated or quantified from current data
- The 5s→10s transition is not a filter regime change
- Recovery timing cannot be inferred from controlled segment runs (origin mismatch)

### Key Finding — Velocity State

The ESKF velocity state (state.v) is weakly constrained by position-only VIO. It is unconverged throughout the controlled segment runs (~60s window from zero-initialisation). This is not a defect — it is consistent with theoretical expectation for position-only observation. Velocity observability requires a longer converged window or direct velocity measurement.

### Key Finding — Geometry

Trajectory geometry (the specific interval of the physical trajectory covered by the outage window) determines whether error exhibits loopback (peaks then falls) or monotonic rise. This is confirmed by controlled comparison: same sequence (V1_01), similar speed (0.37–0.39 m/s), different outage placement → different loopback outcome. Speed alone does not predict loopback.

### Deliverables

```
nep-vio-sandbox:
  fusion/run_06_vio_outage.py          — original outage runner
  fusion/run_06_euroc_controlled.py    — controlled segment runner (rev 2)
  fusion/logs/s06_*_06.json           — 15 original run logs (baseline + 2s/5s/10s/20s × 3)
  fusion/logs/*_ctrl2.json            — Block A and B controlled logs
  fusion/logs/C_mh01_o*_ctrl3.json    — Block C controlled logs
  MicroMind_SNEP06_ClosureReport.docx — Final analytical closure report
```

### Entry Checklist for Next Session

```bash
cd ~/micromind/repos/nep-vio-sandbox
git log --oneline main | head -3
python3 -m pytest tests/ -q 2>&1 | tail -3  # expect 443/443

cd ~/micromind/repos/micromind-autonomy
git log --oneline main | head -3  # expect ea1fe3e at top
python3 -m pytest tests/ -q 2>&1 | tail -3  # expect 241 passed
```

**Status:**
```
S-NEP-06: CLOSED — ANALYTICAL STANDARD ESTABLISHED
```

---

## S-NEP-07 — BASELINED (decision document)

**Objective:** Translate S-NEP-06 validated findings into engineering decisions for the MicroMind Navigation System.

**Verdict: BASELINED — Rev 3 approved by Technical Director. No implementation activities in S-NEP-07 itself.**

### Document
`MicroMind_SNEP07_EngineeringDecisions_Rev3.docx` — uploaded to project knowledge.

### Key Decisions (implemented in S-NEP-08)

| Decision | Lever | Implemented |
|---|---|---|
| D-03 | L-07 — Outage mode switching | ✅ S-NEP-08 |
| D-03 | L-08 — Innovation spike alerting | ✅ S-NEP-08 |
| D-04 | L-05 — Drift envelope logging | ✅ S-NEP-08 |
| D-05 | vel_err_m_s demoted | ✅ S-NEP-08 |
| D-06 | NIS not used as health gate | ✅ Enforced by architecture |

### Architectural Directive (binding)
- `VIONavigationMode` is a fusion-layer construct
- Mission FSM (`core/state_machine/`) is NOT modified
- Mission layer consumes navigation mode as an input signal only
- Strict separation: estimation produces mode, mission consumes mode

**Status:**
```
S-NEP-07: BASELINED — ENGINEERING DECISIONS APPROVED
```

---

## S-NEP-08 — CLOSED

**Objective:** Implement the VIO navigation mode framework from S-NEP-07 decisions D-03, D-04, D-05.

**Verdict: COMPLETE — 311/311 tests passing. All pipeline gates confirmed.**

### Architecture

`VIONavigationMode` is a fusion-layer construct in `core/fusion/vio_mode.py`. Three states: NOMINAL, OUTAGE, RESUMPTION. Transitions triggered by `dt_since_last_vio` only — no trajectory knowledge required. Mission FSM untouched.

### Deliverables

**micromind-autonomy (542867e):**
```
core/fusion/vio_mode.py          — VIONavigationMode: NOMINAL/OUTAGE/RESUMPTION
core/fusion/fusion_logger.py     — schema 08.1: mode fields, drift envelope, spike alert
core/ekf/error_state_ekf.py      — update_vio() now returns (nis, rejected, innov_mag)
tests/test_s_nep_08.py           — 30 acceptance gates G-01..G-08
tests/test_s_nep_04a_interface.py — updated for schema 08.1 and 3-tuple return
tests/test_sprint_s3_acceptance.py — pre-existing TRNStub calling convention fix
tests/test_s9_nav01_pass.py       — pre-existing _get_Q() propagate fix
sim/nav_scenario.py               — pre-existing TRNStub calling convention fix
```

**nep-vio-sandbox (30c2d56):**
```
fusion/run_08_mode_validation.py  — pipeline validation runner (schema 08.1)
fusion/logs/run_08_a_5s_08.json   — 5s outage validation log
fusion/logs/run_08_b_20s_08.json  — 20s outage validation log
```

### Configurable Constants (core/fusion/vio_mode.py)

| Constant | Value | Source |
|---|---|---|
| `VIO_OUTAGE_THRESHOLD_S` | 2.0 s | S-NEP-07 L-07 |
| `VIO_INNOVATION_SPIKE_THRESHOLD_M` | 1.0 m | S-NEP-07 L-08 |
| `VIO_DRIFT_RATE_CONSERVATIVE_M_S` | 0.800 m/s | S-NEP-06 C-07 |
| `VIO_RESUMPTION_CYCLES` | 1 | S-NEP-07 Section 4D doctrine minimum |

### Acceptance Gates

| Gate | Description | Result |
|---|---|---|
| G-01 | NOMINAL→OUTAGE at threshold | ✅ |
| G-02 | OUTAGE→RESUMPTION on first accepted update | ✅ |
| G-03 | RESUMPTION→NOMINAL after required cycles | ✅ |
| G-04 | Spike alert fires when innov_mag > 1m | ✅ |
| G-04b | No spike alert in NOMINAL | ✅ |
| G-05 | Drift envelope monotonic in OUTAGE; None elsewhere | ✅ |
| G-06 | vel_err_m_s absent from operational log | ✅ |
| G-07 | state_machine.py not imported by fusion modules | ✅ |
| G-08 | current_mode accessible without ESKF import | ✅ |

### Pipeline Validation

| Run | Outage | Transitions confirmed | Spike alert | Max drift envelope |
|---|---|---|---|---|
| run_08_a | 5s | NOMINAL→OUTAGE→RESUMPTION→NOMINAL | Yes (innov > 1m) | 4.02 m |
| run_08_b | 20s | NOMINAL→OUTAGE→RESUMPTION→NOMINAL | Yes (innov > 1m) | 16.02 m |

### Pre-existing Fixes Included in This Commit
- `test_sprint_s3_acceptance.py` — TRNStub.update() calling convention (old `ins=`, `dt=` kwargs)
- `test_s9_nav01_pass.py` — `_get_Q()` must call `propagate()` before reading Q matrix
- `sim/nav_scenario.py` — same TRNStub calling convention fix

### Schema Change
- **Old:** schema `06e.3` — no mode fields, vel_err_m_s as operational signal
- **New:** schema `08.1` — vio_mode, dt_since_vio, drift_envelope_m, innovation_spike_alert per entry; vel_err_diagnostic default off; run-level summaries: n_outage_events, n_spike_alerts, max_dt_since_vio, max_drift_envelope_m
- Old logs (S-NEP-04 through S-NEP-06) are NOT affected

### Entry Checklist for Next Session

```bash
cd ~/micromind/repos/micromind-autonomy
git log --oneline main | head -3   # expect 542867e at top
python3 -m pytest tests/ -q 2>&1 | tail -3  # expect 311 passed

cd ~/micromind/repos/nep-vio-sandbox
git log --oneline main | head -3   # expect 30c2d56 at top
python3 -m pytest tests/ -q 2>&1 | tail -3  # expect 443/443
```

**Status:**
```
S-NEP-08: CLOSED — VIO NAVIGATION MODE FRAMEWORK IMPLEMENTED
311/311 tests passing
Next phase: S-NEP-09 — Operational Behaviour Validation (scope defined)
```

---

## Programme Invariants (Active — All Sprints)

The following invariants apply to all future phases. Any code or test that violates these is inadmissible.

1. **Estimator constants frozen** — `_ACC_BIAS_RW`, `_GYRO_BIAS_RW`, `_POS_DRIFT_PSD`, `_GNSS_R_NOMINAL`. TD approval required.
2. **EKF interface stability** — `update_vio()` returns `(nis, rejected, innov_mag)`. `propagate()`, `inject()`, `update_gnss()` signatures unchanged.
3. **Fusion/mission separation** — `VIONavigationMode` is fusion-layer only. `state_machine.py` zero-modification constraint. Mission reads mode as string signal only.
4. **NIS diagnostic only** — Not a health gate. Must not be tuned. PF-03 applies.
5. **Drift envelope is a confidence signal, not a bound** — `drift_envelope_m` is a conservative over-estimate. No logic may treat it as a guaranteed maximum.
6. **Velocity is not an operational signal** — `state.v` must not be used as a primary input to any control, planning, or decision logic (S-NEP-07 Section 1.8 System Rule).
7. **Baseline anchoring** — All evaluations reference S-NEP-04 through S-NEP-08 logs. Schema 08.1 for new runs.
8. **Mode Integrity Invariant (structural constraint)**
   - Navigation mode transitions are determined solely by `VIONavigationMode` inputs (`dt_since_vio`, VIO update events)
   - Transitions must be deterministic given identical input streams
   - No external module (including mission FSM) may override or force mode transitions
   - Mission layer consumes `current_mode` as a read-only signal only
   - Any code writing to `VIONavigationMode` internal state outside `tick()` and `on_vio_update()` is a violation

---

## S-NEP-09 — SCOPED

**Title:** S-NEP-09 — Operational Behaviour Validation

**Principle:** The system must be understood in operation before it is improved. This phase is behaviour validation, not optimisation.

**Objective:** Validate system behaviour under realistic operating variability — rapid dropouts, intermittent updates, repeated cycles, boundary conditions, and signal usefulness — without modifying the estimator, EKF, or any operational parameters.

---

### A — Mode Transition Robustness

Validate that `VIONavigationMode` transitions correctly under conditions beyond the single-outage tests of S-NEP-08.

| Scenario | Description |
|---|---|
| A-01 Rapid dropout | VIO dropouts < 2s (below threshold) — mode must stay NOMINAL |
| A-02 Threshold jitter | VIO gaps of 1.9s, 2.0s, 2.1s — confirm transition fires at and only at threshold |
| A-03 Burst recovery | VIO drops out, recovers for 1 update, drops out again within 1s — correct OUTAGE handling |
| A-04 Repeated cycles | 5 sequential outage/recovery cycles — no state accumulation, counters correct |
| A-05 Long outage | 60s outage — filter stability, envelope growth, correct resumption |
| A-06 Transition latency | Measure elapsed time between: (a) threshold crossing and OUTAGE transition, (b) first accepted VIO update and RESUMPTION transition, (c) RESUMPTION entry and NOMINAL transition. Confirm all transitions occur within one processing/update cycle of the triggering condition. Timing validation only — no behavioural change. |

---

### B — Signal Validity

Evaluate the usefulness and reliability of the three new observability signals.

**B-01 `drift_envelope_m`**
- Does it grow monotonically during all tested outages?
- Is it always ≥ actual incremental drift (conservative)?
- Cases where actual drift exceeds envelope (if any) must be identified and logged
- If `drift_envelope_m` < actual incremental drift: log full context (timestamp, dt_since_vio, mode, sequence); classify as a diagnostic finding; do not treat as automatic failure unless the pattern is systematic

**B-02 `innovation_spike_alert`**
- False positive rate: alerts firing during NOMINAL (must be zero by construction — verify)
- False negative rate: large innovations at resumption that do not trigger (innov ≤ 1m but position has jumped significantly)
- Characterise the distribution of `innov_mag` at resumption across outage durations
- Distribution stability: verify that innovation spike behaviour (frequency and magnitude distribution) is consistent across repeated identical runs (same dataset, same seed)

**B-03 `vio_mode` transitions**
- Confirm mode sequence is deterministic across repeated identical runs (same seed, same data)
- Confirm no spurious OUTAGE transitions during continuous VIO operation

---

### C — Downstream Behaviour (read-only observation)

Assess how the system state evolves through OUTAGE and RESUMPTION without modifying the mission FSM.

**C-01 Position state during OUTAGE**
- Track `error_m` vs `drift_envelope_m` during outage for multiple durations
- Establish the relationship between envelope and actual error across scenarios

**C-02 State discontinuity at RESUMPTION**
- Measure `innov_mag` at first post-outage VIO update across all scenarios
- Characterise the step size distribution — is 1.0m threshold appropriate?
- Log `outage_duration_s` and `mode_at_trigger` context fields

**C-03 Post-resumption stability**
- How many VIO update cycles before position error returns to pre-outage baseline?
- Does `VIO_RESUMPTION_CYCLES=1` produce correct behaviour observationally?

---

### D — Boundary Conditions

**D-01 Zero-length outage** — VIO gap exactly equal to `dt` (one IMU step) — must not trigger OUTAGE
**D-02 Simultaneous threshold crossing** — VIO update arrives at exactly `t_outage_threshold` — deterministic handling
**D-03 Back-to-back large innovations** — two sequential large corrections — second must not trigger spike alert (not first post-outage)
**D-04 Rejection at resumption** — first post-outage VIO update is rejected by NIS gate — OUTAGE must persist, mode must not advance

---

### Deliverables

| File | Repo | Description |
|---|---|---|
| `fusion/run_09_robustness.py` | nep-vio-sandbox | Scenarios A-01..A-05 automated runner |
| `fusion/run_09_signal_validity.py` | nep-vio-sandbox | B-01..B-03 signal analysis runner |
| `fusion/run_09_boundary.py` | nep-vio-sandbox | D-01..D-04 boundary condition runner |
| `tests/test_s_nep_09.py` | micromind-autonomy | Unit gates for determinism and boundary conditions |
| `fusion/logs/s09_*.json` | nep-vio-sandbox | All run logs (schema 08.1) |
| `MicroMind_SNEP09_ValidationReport.docx` | project | Findings document |

---

### Acceptance Gates

| ID | Gate | Pass Condition |
|---|---|---|
| G-09-01 | Threshold precision | OUTAGE fires at ≥ 2.0s, not before |
| G-09-02 | Determinism | Identical input → identical mode sequence, 3 runs |
| G-09-03 | No NOMINAL spike alerts | `innovation_spike_alert=False` for all NOMINAL updates |
| G-09-04 | Envelope conservatism | `drift_envelope_m ≥ actual_incr_drift` for all measured outages |
| G-09-05 | Rejection holds OUTAGE | Rejected first post-outage update does not advance mode |
| G-09-06 | Repeated cycles stable | 5 outage cycles — no state accumulation, all transitions correct |
| G-09-07 | 311/311 tests still passing | No regression introduced by new test files |
| G-09-08 | Transition latency bounded | Mode transitions must occur within one update cycle of the triggering condition. No delayed, skipped, or accumulated transitions permitted. |

---

### Constraints

- No EKF modifications
- No covariance changes
- No parameter tuning
- No new sensors
- No heuristic fixes
- `VIONavigationMode` constants (`VIO_OUTAGE_THRESHOLD_S` etc.) must NOT be changed during this sprint
- `state_machine.py` must not be touched
- All results must be reproducible via committed scripts

### Out of Scope

- EKF tuning or Q/R modification
- `VIO_OUTAGE_THRESHOLD_S` adjustment (observe only — threshold selection is a separate programme decision)
- Hardware integration
- TASL demonstration preparation
- Any implementation changes based on findings — findings feed the next decision cycle, not this sprint

**Status:**
```
S-NEP-09: SCOPED (Rev 2) — AWAITING EXECUTION
```

---

## S-NEP-09 — CLOSED

**Title:** S-NEP-09 — Operational Behaviour Validation

**Commits:** `7c8c92d` (micromind-autonomy) / `4fcf231` (nep-vio-sandbox)
**Test suite:** **332/332 passed** (21 new unit gates added)
**Logs:** 21 JSON files in `fusion/logs/s09_*.json`

### Execution Summary

All A-series (A-01..A-06), B-series (B-01..B-03), and D-series (D-01..D-04) scenarios executed. All acceptance gates evaluated.

### Gate Results

| Gate | Description | Result |
|---|---|---|
| G-09-01 | Threshold precision | ❌ DIAGNOSTIC — FP-01 (fp accumulation, test-construction artefact) |
| G-09-02 | Determinism | ✅ PASS |
| G-09-03 | No NOMINAL spike alerts | ✅ PASS |
| G-09-04 | Envelope conservatism | ✅ PASS (0 underruns) |
| G-09-05 | Rejection holds OUTAGE | ✅ PASS |
| G-09-06 | Repeated cycles stable | ✅ PASS |
| G-09-07 | 332/332 no regression | ✅ PASS |
| G-09-08 | Transition latency bounded | ✅ PASS (all 3 transitions: 1 step) |

### Four Findings

| ID | Finding | Classification | Disposition |
|---|---|---|---|
| FP-01 | 2.0s gap at 200Hz fires on tick 401 not 400 (fp accumulation) | Test construction artefact | DO NOTHING |
| FP-02 | Envelope over-conservative in loopback conditions (actual drift ≈ 0, envelope = 1.6–4.0m) | Structural — by design | DOCUMENT |
| FP-03 | Spike innov_mag non-monotonic with outage duration | Trajectory-dependent | DOCUMENT |
| FP-04 | D-02 mode at threshold boundary already RESUMPTION — correct zero-latency behaviour | Correct behaviour | DO NOTHING |

### Deliverables

```
micromind-autonomy:   tests/test_s_nep_09.py (21 unit gates)
nep-vio-sandbox:      fusion/run_09_robustness.py
                      fusion/run_09_signal_validity.py
                      fusion/run_09_boundary.py
                      fusion/logs/s09_*.json (21 log files)
project knowledge:    MicroMind_SNEP09_AnalysisReport_Rev2.docx
```

### Entry Checklist

```bash
cd ~/micromind/repos/micromind-autonomy
git log --oneline main | head -3   # expect d3afdd1 at top
python3 -m pytest tests/ -q 2>&1 | grep -E "passed|failed"  # expect 332 passed

cd ~/micromind/repos/nep-vio-sandbox
git log --oneline main | head -3   # expect 4fcf231 at top
```

**Status:**
```
S-NEP-09: CLOSED — OPERATIONAL BEHAVIOUR VALIDATED
332/332 tests passing
```

---

## S-NEP-10 — CLOSED

**Title:** S-NEP-10 — Mission & System Integration Rules + Enforcement Implementation

**Commits:** `d3afdd1` (micromind-autonomy — enforcement)
**Documents:** `MicroMind_SNEP10_IntegrationRules_Rev2.docx` (26 rules, 8 sections)
**Test suite:** **332/332 passed**

### Document Summary

S-NEP-10 Rev 2 defines 26 deterministic rules across 8 sections governing mission-layer consumption of three navigation signals (`vio_mode`, `drift_envelope_m`, `innovation_spike_alert`). All rules depend only on observable signals. No rule depends on unobservable quantities (actual position error, trajectory geometry, velocity state, NIS).

| Section | Content |
|---|---|
| 0 | 10 governing constraints (C-01..C-10) including Mode Integrity Invariant and asynchronous transition timing |
| 1 | 5 observable signals — explicit exclusion of vel_err, NIS, innov_mag magnitude |
| 2 | Function classification: PRECISION / DEGRADED-CONTINUE / ALWAYS-CONTINUE |
| 3 | Mode rules N-01..N-03, O-01..O-05, R-01..R-04, R-04a |
| 4 | Envelope rules E-01..E-04 including HIGH band non-negotiable suppression |
| 5 | Spike rules S-01..S-04 |
| 6 | Signal fusion rules F-01..F-05 including fail-safe F-04 |
| 7 | 7 explicit exclusions (E_WARN value, duration budget, spike severity, trajectory-aware behaviour, velocity, NIS, post-resumption timing) |

### Enforcement Implementation (E-1 through E-5)

Minimal enforcement layer applied to `scenarios/bcmp1/bcmp1_runner.py` — 61 insertions, 6 deletions.

| Point | Location | Action | Prevented Failure |
|---|---|---|---|
| E-1 | `_phase_terminal()` before DMRL call | Suppress DMRL if `in_outage`; emit `DMRL_SUPPRESSED` | DMRL executes on uncertain position |
| E-2 | `_phase_terminal()` before `inputs_from_dmrl()` | `corridor_violation=True` if `in_outage` | False corridor compliance from drifted state.p |
| E-3 | `_RoutePlannerStub.replan()` | `uncertain_position` flag widens clearance | Route planned with nominal margin under ~2.44m uncertainty |
| E-4 | `run()` before `_phase_terminal()` call | Unknown mode → `in_outage=True`; emit `VIO_MODE_FAULT` | Silent PRECISION execution on mode fault |
| E-5 | `run()` at terminal phase call site | Defer terminal phase if `in_outage`; re-evaluate at NOMINAL; emit single `TERMINAL_ZONE_ENTERED` | Zone entry logged from uncertain position; TERM KPIs contaminated |

### Validation Log (clean run, seed=42)

```
TERMINAL_ZONE_ENTERED source=nominal_evaluation vio_mode=NOMINAL
=== TERMINAL PHASE: SHM ACTIVE (T+28 min) ===
term01_pass: True  (lock_conf=0.9091)
term02_pass: True  (decoy_rejected=True)
term03_pass: True  (l10s_compliant=True)
all_pass:    True
```

### What Is NOT Done (safely deferred)

| Item | Reason |
|---|---|
| Spike hold (1 VIO cycle, ~8ms) | Below BCMP-1 KPI resolution |
| `E_WARN` threshold value | Programme decision — requires mission-specific characterisation |
| DEGRADED-CONTINUE margin quantification | Structural flag present; specific value is programme decision |
| Two-theatre BCMP-1 integration | Next phase — gates on this enforcement being complete |

### Entry Checklist

```bash
cd ~/micromind/repos/micromind-autonomy
git log --oneline main | head -3   # expect d3afdd1 at top
python3 -m pytest tests/ -q 2>&1 | grep -E "passed|failed"  # expect 332 passed

# Verify enforcement is live
python3 -c "
from scenarios.bcmp1.bcmp1_runner import BCMP1Runner
r = BCMP1Runner(verbose=False, seed=42).run(1)
msgs = [e['msg'] for e in r.events]
assert any('TERMINAL_ZONE_ENTERED' in m for m in msgs), 'E-5 not firing'
assert r.passed, 'BCMP-1 11/11 not passing'
print('Enforcement verified. BCMP-1 passing.')
"
```

**Status:**
```
S-NEP-10: CLOSED — INTEGRATION RULES BASELINED AND ENFORCEMENT IMPLEMENTED
332/332 tests passing
BCMP-1 11/11 criteria met with S-NEP-10-compliant enforcement active
Next phase: BCMP-1 two-theatre integration (TRN-primary eastern corridor,
            VIO-primary western plains corridor)
```
