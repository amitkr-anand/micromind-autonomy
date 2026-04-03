# MicroMind Status Digest — 36 Milestone Files
**Date:** 03 April 2026  
**Compiled by:** Claude Code QA Agent  
**Files read:** 36 of 36 (plus NEP_SPRINT_STATUS.md read in two passes; VIO_Phase0_Architecture_v1_2.md and NEP_PlatformContractSpec_v1_1-1.md exceed single-pass token limits — key content captured via targeted reads)  
**Source directory:** `docs/status_updates/`

---

## Section 1 — Chronological Development Arc

- **S0 (early 2026, macOS):** ESKF V2 baseline. 15-state error-state filter, quaternion math, INS strapdown mechanisation, WGS-84 constants. Development on 2017 MacBook Pro.
- **S1:** NanoCorteX 7-state FSM, SimClock, MissionLogSchema (DD-02 Phase-1), BCMP-1 scenario definition. 9/9 gates.
- **S2:** BIM trust scorer (GNSS spoof detection ≤250 ms, Green/Amber/Red states, hysteresis). GNSS spoof injector. 9/9 gates.
- **S3:** TRN stub (RADALT + DEM + 1D NCC), short-corridor nav scenario, Plotly mission dashboard. 8/8 gates. TRN wired but drift accumulation hidden at 10 km test length.
- **S4:** EW engine (DBSCAN jamming detection, cost map ≤500 ms), Hybrid A\* route planner (replan ≤1 s), BCMP-1 EW sim. 8/8 gates.
- **S5 (21 February 2026):** DMRL stub, L10s-SE, full BCMP-1 runner. 111/111 gates. Milestones M1 and M4 closed. Development still on macOS (2017 MacBook Pro retired at S8 due to thermal throttling).
- **S6:** ZPI burst scheduler, CEMS cooperative EW sharing (immediately frozen as Phase-2). 36/36 gates.
- **S7:** 9-panel BCMP-1 dashboard, HTML debrief report generator.
- **Demo Fork decision (28 February 2026):** Phase-1 scope formally bounded. CEMS, predictive EW, FR-108/FR-109–112 reclassified to Phase-2. All completed sprint work retained valid. Scope isolation exercise, not redesign.
- **S8 (prior to 28 February — completed, gated 28 February):** IMU fidelity models (STIM300, ADIS16505-3, BASELINE). ALS-250 250 km corridor sim. BCMP-1 IMU integration. 68/68 gates. **Platform transition: macOS retired, Azure VM (Standard_D8s_v5, 8 vCPU, 32 GB RAM, Ubuntu 24 LTS, Python 3.10) adopted.** Azure cost to S10: ₹3,423 (₹9,405 forecast pre-perf-fix).
- **S9 (4 March 2026, Azure VM):** Five-root-cause architectural correction — TRN internal Kalman removed, ESKF Q-matrix fixed (gyro bias RW was 247× too large), position PSD added, propagation order corrected, NAV-01 drift metric fixed to 2D horizontal. NAV-01 PASS at 150 km (14.5 m max 5-km drift, 7× margin). 215/215 gates.
- **TRN Sandbox (March 2026, Azure VM):** Pre-S10 forced question — 2D patch NCC vs 1D profile NCC on 150 km Himalayan manoeuvring corridor with real SRTM DEM. 1D NCC failed (NAV-01 = 421.9 m); 2D patch NCC achieved 33.3 m (3× margin). Architecture decision taken: 2D patch NCC is the production TRN method.
- **S10 (12 March 2026, Azure VM):** NCC vectorised (2D numpy), S9 regression gates, parallel IMU runner (subprocess.Popen; MKL fork deadlock resolved), S8-D TASL drift chart (three IMUs 250 km). Critical O(n²) perf bug found and fixed (134× speedup, 14,000 s → 105 s). 222/222 gates. Milestones M2 and M5 closed. **Phase-1 demonstration package complete.**
- **S-NEP VIO Selection Programme (March 2026):** Parallel programme on nep-vio-sandbox. Five candidates evaluated. RTABMap eliminated Stage-1 (RPE failure + covariance type incompatibility). ORB-SLAM3 not ready (no global covariance). VINS-Fusion/Kimera deferred. **OpenVINS selected: GO verdict.** S-NEP-01 through S-NEP-10 completed (332/332 gates on autonomy stack, 443/443 on sandbox).
- **Platform transition to micromind-node01 (mid-to-late March 2026):** Ryzen 7 9700X, RTX 5060 Ti 16 GB, Ubuntu 24.04.4, Python 3.12.3. Azure VM retired for development work. BCMP-2 sprints and Pre-HIL integration conducted on this hardware. Pace accelerated: SB-1 through SB-3 (all BCMP-2 acceptance tests) completed in a single 30 March session. CP-0 through CP-2 (Pre-HIL integration) completed 29 March.
- **Pre-HIL Integration Sprint (29 March 2026, micromind-node01):** PX4 SITL wired, 23 new files under integration/, MAVLink OFFBOARD bridge validated. S-PX4-01 through S-PX4-09 all PASS. ESKF P95 latency 0.085 ms (118× gate margin), E2E P95 0.363 ms (138× margin). 332/332 SIL gates unchanged throughout.
- **BCMP-2 SB-1 through SB-3 (29–30 March 2026, micromind-node01):** Dual-track acceptance regime implemented. Vehicle A (INS-only), Vehicle B (MicroMind full stack). 90/90 gates. Fault injection proxies (SB-2), full 150 km mission + reports (SB-3).
- **03 April 2026:** AD-01 (orthophoto matching replaces RADALT-NCC), AD-02 (IMU ARW floor correction) documented. OI-04 and OI-05 formally logged. This assessment session.

---

## Section 2 — Architecture Decisions Taken Along the Way

| ID | Decision | Why (at the time) | Sprint / Date | In MICROMIND_DECISIONS.md? |
|---|---|---|---|---|
| **SAD AD-01** | Single ESKF — TRN internal Kalman removed; TRN is measurement provider only | Two independent estimators receiving same measurements corrupt each other | S9 (4 March 2026) | Not in DECISIONS.md (this AD-01 is from the SAD, distinct from the DECISIONS.md AD-01 which is the orthophoto decision) |
| **SAD AD-02** | 2D NCC over 1D NCC for TRN | 2D NCC: 2 DEM reads + one numpy matrix op; 1D: ~33,640 Python DEM lookups; 2D NAV-01: 33.3 m vs 1D failure: 421.9 m | Pre-S10 sandbox (March 2026) | No |
| **SAD AD-03** | subprocess.Popen for parallelism, not multiprocessing | numpy/BLAS MKL fork deadlock on Linux; subprocess gives isolated Python interpreters | S10 (March 2026) | No |
| **SAD AD-04** | IMU noise array caching in mechanisation loop | total_gyro()/total_accel() are pure functions; calling per-step is O(n²); cache on first call is safe, 134× speedup | S10 (March 2026) | No |
| **SAD AD-05** | Bounded deterministic state machine — no learned policy | Tactical systems require deterministic, auditable behaviour; Demo Fork constraint (28 Feb 2026) | S1 / Demo Fork | No |
| **SAD AD-06** | Autonomy as payload — no autopilot modification | Airframe-agnostic design; TASL hardware decision deferred post-SIL | S0 | No |
| **Pre-HIL D-1** | Direct Python + pymavlink over UDP, no ROS2 in bridge layer | Previous December integration failure partly attributed to middleware complexity; ROS2 bridge optional addendum only | Pre-HIL v1.2 (26 March 2026) | No |
| **Pre-HIL D-2** | MicroMind owns monotonic mission clock; SYSTEM_TIME sent as courtesy only | Prevents IFM-01 (timestamp misalignment) — highest-consequence silent failure mode | ADR-0 (28 March 2026) | No |
| **Pre-HIL D-3** | Python ABCs for all driver types; DriverFactory selects at startup | Driver replaceability without core code changes; RC-4 pass criterion | ADR-0 (28 March 2026) | No |
| **Pre-HIL D-4** | Seven PX4 failure modes (FM-1–FM-7) as governing constraints | December integration failed — root causes enumerated and fixed as design rules | ADR-0 (28 March 2026) | No |
| **SIA** | Listener-first sensor integration (not listener-only) | Pure listener sacrifices performance where cost is unacceptable; direct ownership only where required | SIA v1.0 (28 March 2026) | No |
| **SIA** | EO camera direct MIPI/USB3, SDR direct USB3/PCIe | No MAVLink path for video; no autopilot interface with SDR | SIA v1.0 | No |
| **DEMO FORK** | Bounded Phase-1 scope — CEMS, predictive EW, FR-108, FR-109–112 to Phase-2 | Programme control; founder-led, resource-constrained; 100% of bounded core > 60% of broader stack | 28 February 2026 | No |
| **BCMP-2 architecture** | Cross-track error propagation model for Vehicle A (not full 3D mechanisation) | Full 3D mechanisation requires true specific force including gravity; simplified model matches C-2 analytical derivation | SB-1 (29 March 2026) | No |
| **BCMP-2 architecture** | Monte Carlo (N=300) calibration for C-2 drift envelopes (not analytical) | Analytical derivation uses worst-case constant bias; individual seeds draw from distribution, P5 far below analytical floor | SB-1 (29 March 2026) | No |
| **BCMP-2 architecture** | Dashboard in matplotlib, not Plotly/Dash | Plotly/Dash not installed on micromind-node01; matplotlib is air-gap safe; matches existing programme pattern | SB-3/SB-4 boundary (30 March 2026) | No |
| **S-NEP** | OpenVINS selected as VIO system | Stage-1 and Stage-2 GO criteria met; drift 0.94–1.01 m/km (3.6% variance), zero FM events; RTABMap disqualified PF-01 covariance type | S-NEP Stage-2 GO (21 March 2026) | No |
| **S-NEP PF-03** | NIS is diagnostic only — must not be tuned or reinterpreted; requires TD approval to change | Measurement model limitation at high VIO rate (absolute R vs relative inter-frame innovation) | S-NEP-04 (22 March 2026) | No |
| **S-NEP-07** | Velocity state must not be used as primary input (system rule 1.8) | VIO provides position-only updates; velocity weakly constrained, unconverged below 5-min window | S-NEP-07 Rev 3 (22 March 2026) | No |
| **AD-01 (DECISIONS.md)** | Orthophoto image matching replaces RADALT-NCC as primary L2 ingress correction | RADALT beam footprint spans 9–13 DEM cells at AVP cruise altitudes; orthophoto MAE < 7 m, no additional hardware | 03 April 2026 | **Yes** |
| **AD-02 (DECISIONS.md)** | IMU ARW floor: 0.1 → 0.2 °/√hr | STIM300 typical is 0.15 °/√hr, exceeds 0.1 floor; BASELINE model (0.05) is unreachable by any candidate sensor | Flagged S8 (27 Feb), formalised 03 April 2026 | **Yes** |

**Observation:** Only 2 of the 21 decisions catalogued above appear in MICROMIND_DECISIONS.md. The remaining 19 are documented in status files, handoff notes, or SAD but not in the formal decisions register. The ADR-0 documents capture 4 of these in a structured format (D-1 through D-4).

---

## Section 3 — Current Validated State

### Code-complete and tested with passing gate count

| Module | File | Sprint | Gate count | Notes |
|---|---|---|---|---|
| ESKF (15-state) | `core/ekf/error_state_ekf.py` | S0+S9+S10+S-NEP-04 | Included in 332/332 | Frozen. Q-matrix STIM300-calibrated. update_vio() additive. |
| INS mechanisation | `core/ins/mechanisation.py` | S0+S8+S10 | Included in 332 | S10 noise-cache perf fix applied. |
| BIM trust scorer | `core/bim/bim.py` | S2 | 9/9 (S2) | Frozen. Spoof detect ≤250 ms validated. |
| TRN stub | `core/ins/trn_stub.py` | S3+S9+S10 | Included in 332 | 2D NCC, measurement-provider-only. **Sensor/map model wrong vs AD-01** (OI-05). |
| IMU models | `core/ins/imu_model.py` | S8-A | 16/16 (S8-A) | STIM300, ADIS16505-3, BASELINE all validated. |
| NanoCorteX FSM | `core/state_machine/state_machine.py` | S1 | 9/9 (S1) | 7-state deterministic. |
| EW engine | `core/ew_engine/ew_engine.py` | S4 | 8/8 (S4) | Reactive cost map only. Predictive branch present but not exercised. |
| Hybrid A\* | `core/route_planner/hybrid_astar.py` | S4 | 8/8 (S4) | Route replan ≤1 s confirmed. Texture cost term missing (OI-08). |
| DMRL stub | `core/dmrl/dmrl_stub.py` | S5 | In 111/111 (S5) | Rule-based. 100% lock rate / 100% decoy reject on synthetic inputs (OI-06). |
| L10s-SE | `core/l10s_se/l10s_se.py` | S5 | In 111/111 (S5) | Deterministic decision tree. |
| ZPI | `core/zpi/zpi.py` | S6 | 36/36 (S6, combined) | Phase-2 frozen. Not exercised in BCMP-1 or BCMP-2 missions. |
| CEMS | `core/cems/cems.py` | S6 | 36/36 (S6, combined) | Phase-2 frozen. Not exercised in any mission scenario. |
| VIO mode manager | `core/fusion/vio_mode.py` | S-NEP-08 | In 332/332 | Frozen. Three states: NOMINAL/OUTAGE/RESUMPTION. |
| Frame utilities | `core/fusion/frame_utils.py` | S-NEP-04 | In 332/332 | Frozen. ENU→NED rotation, covariance extraction. |
| Fault injection layer | `fault_injection/` | BCMP-2 SB-2 | 25/25 (SB-2) | FI-01 through FI-13 + presets. Thread-safe singleton. |
| BCMP-2 scenario | `scenarios/bcmp2/` | SB-1 through SB-3 | 90/90 (SB-1–3) | C-1 through C-4 constraints. Seeds 42/101/303. |
| ALS-250 corridor sim | `sim/als250_nav_sim.py` | S8-C+S9 | 68/68 (S8) | 250 km NAV-01 PASS all three IMUs. search_pad_px=25 critical. |
| MAVLink bridge | `integration/mavlink_bridge.py` | Pre-HIL | 206/206 integration | 5-thread architecture. S-PX4-09: 60 s OFFBOARD held. Frozen SIL core untouched. |
| Driver abstraction | `integration/drivers/` | Pre-HIL | 206/206 integration | DriverFactory. Sim→real swap confirmed without core change (RC-4). |

### Code-complete but with known gaps or deferred items

| Module | Gap | OI reference |
|---|---|---|
| TRN stub | Sensor model (RADALT+DEM) conflicts with AD-01 (camera+orthophoto). Tests will be invalidated when OI-05 is resolved. | OI-05 |
| VIO integration | Outdoor / km-scale validation pending. Indoor EuRoC only. | OI-07 |
| DMRL stub | Rule-based classifier. CNN required for real operation. All results are synthetic inputs. | OI-06 |
| Live logger | BridgeLogger T-LOG queue tested in unit tests; formal 60 s drop-rate test (RC-8) pending Phase 3. | — |
| Timestamp monotonicity | IFM-01 guard implemented; formal injection test under live timing (RC-7) pending Phase 3. | — |
| Control loop OUTAGE | RC-11 (VIO OUTAGE + RESUMPTION during live SITL) pending Phase 3. Five sub-criteria RC-11a–e. | — |
| Gazebo rendering | Physics correct, telemetry valid; Gazebo GUI blank on micromind-node01 X11 session. Non-blocking for gate criteria but critical for OEM demo. | — |
| BCMP-2 dashboard | SB-4 tagged (`sb4-dashboard-replay`, 31 March 2026) but context file shows PENDING. Dashboard files present in repo (untracked in git status). SB-4 appears complete — context file not updated. | — |

### Specified but not yet implemented

| Item | Source specification | OI / phase |
|---|---|---|
| Orthophoto image matching (L2 correction) | AD-01 (03 April 2026) | OI-05, Phase-2 |
| Route planner terrain-texture cost term | AD-01 consequences | OI-08 |
| SRS §10.2 Mission Envelope Schema — AVP speed/altitude fields | OI-09 | OI-09 |
| BCMP-1 pass criteria ↔ SRS test ID traceability table | OI-10 | OI-10 |
| DMRL CNN upgrade | Phase-2 | OI-06 |
| PQC cryptography stack (FR-109–112) | Phase-2 | — |
| ROS2 / PX4 SITL full integration | Phase-3 | — |
| Satellite masking / FR-108 | Phase-2 | — |
| CEMS active use in mission | Phase-2 | — |
| Cross-mission learning pipeline (DD-02 Phase-2) | Phase-3 | — |
| Velocity observability improvement (L-02 wheel odometry/baro) | S-NEP-07 | — |
| Trajectory-aware VIO outage classification (L-09) | S-NEP-07 | — |
| run_demo.sh (single-command OEM demo) | Pre-HIL v1.2 §Part 9 | Phase-4 |

---

## Section 4 — SB-5 Demo Requirements (verbatim extracts)

Sources: `BCMP2_STATUS.md`, `MicroMind_BCMP2_Implementation_Architecture_v1_1.md`

**From BCMP2_STATUS.md:**

> **SB-5 — Repeatability and Closure ⏳ PENDING**
>
> **Deliverables:** `tests/test_bcmp2_at6.py` (seeds 42/101/303), overnight stress, final HTML report, BCMP-2 Closure Report.

**From MicroMind_BCMP2_Implementation_Architecture_v1_1.md (extracted via agent):**

AT-6 acceptance test definition:

> **AT-6: Three-seed repeatability / endurance**
> - Three canonical seeds: 42 (nominal), 101 (alternate weather), 303 (stress)
> - Four-hour overnight endurance run
> - No memory leak over sustained operation
> - Identical phase transition chains across all three seeds
> - Acceptance: all gates 42/101/303 pass C-2 envelope criteria

**From BCMP2_STATUS.md acceptance test summary:**

> | AT | Purpose | Gates | Status |
> |---|---|---|---|
> | AT-6 | 3× repeatability / endurance | TBD | ⏳ SB-5 |

**SB-5 exit conditions (from architecture document):**

> - AT-6: seeds 42/101/303 all pass AT-2 structure (C-2 envelopes, drift monotonicity, report structure)
> - Overnight stress: 4-hour run, no memory leak, no divergence
> - Final HTML report: business comparison block first (§8.3 ordering enforced)
> - BCMP-2 Closure Report: full programme closure document

**Note:** The architecture document (v1.1) specifies AT-6 gates as TBD. Exact gate count and acceptance criteria for AT-6 are not yet defined in any committed file. The SB-5 exit criteria are less precisely specified than SB-1 through SB-3.

---

## Section 5 — Open Items and Deferred Work

Items already captured in MICROMIND_PROJECT_CONTEXT.md (OI-01 through OI-10):

| ID | Item | Source file(s) |
|---|---|---|
| OI-01 | V7 IMU ARW floor 0.1 → 0.2 °/√hr | MicroMind_S9_TD_Update.md, MicroMind_DemoEdition_RealignmentAnalysis.md, TRN_Sandbox_ClosureReport_S10Handoff.md |
| OI-02 | `bcmp2_report.py` L420 `datetime.utcnow()` deprecation | BCMP2_STATUS.md, BCMP2_JOURNAL.md |
| OI-03 | ALS-250 overnight run → als250_drift_chart.py | HANDOFF_S10_to_S11.md, MicroMind_Phase1_ClosureReport.md |
| OI-04 | OpenVINS → ESKF interface spec not documented | NEP_SPRINT_STATUS.md (MICROMIND_PROJECT_CONTEXT.md §8) |
| OI-05 | TRN stub implies RADALT-NCC; must update to orthophoto | MICROMIND_DECISIONS.md AD-01 consequences |
| OI-06 | DMRL is rule-based stub; all terminal results are stub-based | README_2026-02-21_S5_Complete.md, MicroMind_SoftwareArchitecture_v1_0.md, MicroMind_Phase1_ClosureReport.md |
| OI-07 | Outdoor / km-scale OpenVINS validation pending | NEP_SPRINT_STATUS.md, OpenVINS_Endurance_Validation_Report_Stage2_v4.md (L1, L3) |
| OI-08 | Route planner terrain-texture cost not implemented | MICROMIND_DECISIONS.md AD-01 consequences |
| OI-09 | SRS §10.2 Mission Envelope Schema missing AVP fields | MICROMIND_PROJECT_CONTEXT.md §8 |
| OI-10 | BCMP-1 pass criteria ↔ SRS traceability missing | MICROMIND_PROJECT_CONTEXT.md §8 |

Items found in status files **not** captured in OI-01 through OI-10:

| New item | Source file | Priority |
|---|---|---|
| Gazebo rendering blank on micromind-node01 (X11 / OGRE2 backend issue). Non-blocking for gate criteria but blocks OEM demo per Pre-HIL §Part 9. | MicroMind_PreHIL_Part12_ProgrammeAssessment_20260329.md, MicroMind_PreHIL_TD_Update_20260329.md | HIGH — before any OEM meeting |
| RC-11 (control loop independence during VIO OUTAGE/RESUMPTION) — five sub-criteria RC-11a through RC-11e, all PENDING Phase 3. RC-11b (ESKF finite-value assertion during OUTAGE) is highest-risk. | MicroMind_PreHIL_Part12_ProgrammeAssessment_20260329.md | HIGH — before CP-3 |
| RC-7 (timestamp monotonicity injection test under live timing) — PENDING Phase 3. | MicroMind_PreHIL_TD_Update_20260329.md | HIGH — before CP-3 |
| RC-8 (live logger non-blocking at 200 Hz, 60 s formal drop-rate test) — PENDING Phase 3. Also: Jetson Orin risk variant (renice +10 or 2-core taskset) not yet characterised. | MicroMind_PreHIL_Part12_ProgrammeAssessment_20260329.md | HIGH — before CP-3 |
| CP-3 readiness report not yet committed (requires RC-7, RC-8, RC-11 all PASS). Pre-HIL cannot be declared complete without it. | MicroMind_PreHIL_Part12_ProgrammeAssessment_20260329.md | HIGH |
| mark_send not natively integrated into mavlink_bridge._setpoint_loop — CP-2 latency result has an asterisk (measured on patched instrumented variant, not canonical). | MicroMind_PreHIL_Part12_ProgrammeAssessment_20260329.md | MEDIUM — before CP-3 |
| SB-5 AT-6 gate count and exact acceptance criteria not defined in any committed file. | BCMP2_STATUS.md, MicroMind_BCMP2_Implementation_Architecture_v1_1.md | MEDIUM |
| BCMP-2 context file discrepancy: MICROMIND_PROJECT_CONTEXT.md shows SB-4 ⏳ PENDING, but git log shows `c183b9c SB-4 final` (31 March 2026) and tag `sb4-dashboard-replay`. Context file appears not updated after SB-4 closed. | BCMP2_JOURNAL.md, git history | LOW — context maintenance |
| NIS is a diagnostic signal only (PF-03) — must not be tuned, reinterpreted, or adjusted without TD approval. Not documented outside NEP status files. | NEP_SPRINT_STATUS.md, HANDOFF_S-NEP-04_CLOSURE.md | MEDIUM — document in any external report using NIS |
| System rule 1.8 (S-NEP-07): "No control, planning, or decision logic shall rely on state.v as a primary input." Not enforced in BCMP-2 or BCMP-1 runner explicitly. | MicroMind_SNEP07_EngineeringDecisions_Rev3.md | MEDIUM — before any velocity-dependent feature |
| Velocity state weakly constrained by position-only VIO — requires >60 s from zero-initialisation to converge. vel_err_m_s must not be used as operational health signal. | MicroMind_SNEP06_ClosureReport.md, HANDOFF_S-NEP-05_CLOSURE.md | MEDIUM |
| Drift envelope (drift_envelope_m) is a confidence degradation signal, not a hard error bound. Over-conserves by 3.3–9.8× on diverging trajectories; infinite over-estimation on loopback. Documented in S-NEP-09 but not in any external-facing document. | MicroMind_SNEP09_AnalysisReport_Rev2.md | MEDIUM — document in reports |
| Night-time orthophoto matching (LWIR vs visible-light tiles) — cross-spectral matching mentioned in AD-01 as risk but not yet validated in MicroMind SIL or referenced in any plan. | MICROMIND_DECISIONS.md AD-01 | MEDIUM |
| Phase-2 deferred items not in any OI: cybersecurity stack (FR-109–112), satellite masking (FR-108), CEMS active use, predictive EW, DMRL CNN, cross-mission learning. | MicroMind_DemoEdition_RealignmentAnalysis.md, S10_ScopeReview_DemoFork_Updated.md | LOW — roadmap items |
| run_demo.sh (single-command OEM demo) not yet written. Required for DR-3 and OEM presentation. | MicroMind_PreHIL_Part12_ProgrammeAssessment_20260329.md §DR-3 | HIGH — before any OEM meeting |
| Jetson Orin profiling not done — latency margins measured on Ryzen 7 9700X (138× gate margin). Jetson margins unknown. | MicroMind_PreHIL_TD_Update_20260329.md | MEDIUM — before HIL |
| V-9 from ADR-0 v1.1 (sensor stream rate verification: HIGHRES_IMU 200 Hz, GPS_RAW_INT 5 Hz, DISTANCE_SENSOR 10 Hz) still marked PENDING. | MicroMind_PreHIL_ADR0_v1_1.md | MEDIUM |

---

## Section 6 — Candidate Improvements (observations only)

These are observations from reading the full history. No changes are recommended here.

| Candidate | Rationale for reviewing |
|---|---|
| **The decisions register (MICROMIND_DECISIONS.md) contains only 2 of ~21 significant decisions.** The remaining 19 are scattered across SAD, handoff documents, ADR-0 files, and sprint status files. A future developer or TASL reviewer cannot reconstruct the programme's design rationale from DECISIONS.md alone. | Decision traceability |
| **SB-5 AT-6 gate count is "TBD"** in the governing architecture document. Every previous sprint closed against explicitly numbered gates. SB-5 is the first with undefined acceptance criteria — worth clarifying before the sprint starts. | Sprint hygiene |
| **C-2 drift envelopes were calibrated on the BASELINE IMU noise model** (0.05 °/√hr ARW), not the STIM300 (0.15 °/√hr). The context records this (AD-02, OI-03) but the consequence — that external BCMP-2 results must note the calibration baseline — is mentioned only in AD-02, not in any test or report template. | Evidence integrity |
| **The GPU NCC optimisation** was reclassified Phase-2 in S10 scope review. The TRN Sandbox closure report notes that 2D numpy NCC already completes 150 km in ~1 s and is "potentially sufficient for real-time operation on a mid-tier embedded processor." The Phase-2 classification may be more conservative than needed. | Resource allocation |
| **BCMP-1 is tested with synthetic thermal scenes** (generate_synthetic_scene() in dmrl_stub). The QA standing rule requires L10s-SE tests to verify decision tree under adversarial EO conditions, not clean synthetic inputs. No test currently exercises this. | QA standing rule #2 compliance |
| **The Pre-HIL integration produced 23 new files under integration/** without modifying any existing core file — clean separation. However, the integration layer is not covered by any test in the 332 SIL gates or 90 BCMP-2 gates. The 206 integration gates are a separate (and non-trivial) suite. Clarity on which gates cover what layer would simplify future regression claims. | Test architecture |
| **Demo Fork (28 Feb 2026) prevented scope creep across S9 and S10.** The sprint discipline is credited explicitly in MicroMind_Phase1_ClosureReport.md Lesson 8. The same discipline was not applied to the Pre-HIL phase, where Gazebo rendering, run_demo.sh, and VIO outage demonstration all entered scope during S-PX4 execution. | Programme control |
| **The ESKF position process noise PSD (1.0 m/√s)** was set empirically during S9 to "bring P_pos to approximately 27 m²" — the justification is functional (Kalman gain ~0.5) rather than physically derived from INS drift characterisation. This may need revisiting against STIM300 datasheet or Monte Carlo drift data before HIL. | Filter tuning rigour |

---

## Section 7 — SIL Completeness Assessment

### ESKF

| Dimension | Status |
|---|---|
| **SIL-tested?** | Yes. Propagation, GNSS update, TRN update, VIO update, Q-matrix constants, filter stability, determinism. |
| **Coverage gaps** | RC-11b: ESKF finite-value assertion during VIO OUTAGE under live timing (integration layer) not covered by any of the 332 SIL gates. Q-matrix PSD (1.0 m/√s) is empirically set, not derived from STIM300 datasheet. No adversarial multi-sensor fusion test (simultaneous spoofed GNSS + VIO). |
| **SIL-complete condition** | RC-11b verification added to integration test suite; position PSD documented against measured drift data. |

### BIM

| Dimension | Status |
|---|---|
| **SIL-tested?** | Yes. Trust scoring, spoof detection, hysteresis, GREEN/AMBER/RED states, ≤250 ms detection latency. |
| **Coverage gaps** | Sophisticated pre-filter spoofing (spoofed signals that pass PX4's GPS integrity checks) not testable via GPS_RAW_INT listener path (noted in SIA v1.0). No adversarial multi-constellation scenario. |
| **SIL-complete condition** | For SIL: adequate. For operational clearance: direct GNSS UART Y-cable required for production spoof testing. |

### TRN / L2-correction

| Dimension | Status |
|---|---|
| **SIL-tested?** | Yes, for the RADALT+DEM+NCC mechanism: 250 km linear NAV-01 PASS, 150 km manoeuvring Himalayan PASS. Sensor model and reference map are wrong relative to AD-01. |
| **Coverage gaps** | The entire L2 mechanism will change to orthophoto image matching (OI-05). Zero SIL tests exist for orthophoto matching. Current TRN SIL tests will be invalidated when OI-05 is resolved. Cross-spectral matching (night LWIR vs visible-light tiles) untested. Featureless terrain (σ_terrain <10 m) causes TRN suppression — no test for route planner response to long gaps. |
| **SIL-complete condition** | New orthophoto matching stub implemented, SIL tests rewritten for new sensor model, cross-spectral case documented, route planner texture cost tested. |

### VIO integration

| Dimension | Status |
|---|---|
| **SIL-tested?** | Yes. ENU→NED frame rotation, covariance extraction, update_vio(), VIONavigationMode transitions, OUTAGE/RESUMPTION characterisation (S-NEP-05 through S-NEP-09), enforcement E-1..E-5 (S-NEP-10). 332/332 gates. |
| **Coverage gaps** | Outdoor / km-scale validation (OI-07, L1/L3 from Stage-2 report). RC-11 (VIO OUTAGE + setpoint continuity during live SITL) pending. Run-to-run variance of OpenVINS not characterised (single run per sequence, L2). Velocity state unconverged below 5-min window. System rule 1.8 (no control logic using state.v) not enforced in BCMP runners. |
| **SIL-complete condition** | Outdoor km-scale validation complete; RC-11 verified; second run per sequence benchmarked; system rule 1.8 asserted in all runner code paths. |

### EW Engine

| Dimension | Status |
|---|---|
| **SIL-tested?** | Yes. Reactive path: jam detection → cost map update ≤500 ms → Hybrid A\* replan ≤1 s. Validated via BCMP-1 EW sim. |
| **Coverage gaps** | Predictive jammer velocity branch present in code but not exercised in any test (Phase-2). Multi-node jammer clustering not explicitly tested beyond BCMP-1 scenario. No SDR hardware path. |
| **SIL-complete condition** | For Phase-1: adequate (reactive path tested). Phase-2 would require predictive branch test suite. |

### Route Planner

| Dimension | Status |
|---|---|
| **SIL-tested?** | Yes. Hybrid A\* route planning against EW cost map, replan ≤1 s. |
| **Coverage gaps** | Terrain-texture cost term not implemented (OI-08). No test for featureless terrain avoidance (the failure mode identified in AD-01). |
| **SIL-complete condition** | Texture cost term implemented and tested against route that would otherwise traverse σ_terrain <10 m zone. |

### DMRL

| Dimension | Status |
|---|---|
| **SIL-tested?** | Yes, rule-based stub. 100% lock rate, 100% decoy rejection on synthetic inputs, 50/50 each (FR-103, KPI-T01, KPI-T02). |
| **Coverage gaps** | CNN not implemented (OI-06). All inputs are synthetic thermal scenes generated by generate_synthetic_scene(). No adversarial EO test. QA standing rule #2 requires verification under adversarial EO conditions. L10s-SE decisions in all BCMP results are downstream of stub-only DMRL — must be caveated in any external report. |
| **SIL-complete condition** | CNN implementation required for operational clearance. Rule-based stub SIL is complete for Phase-1 gate purposes only. |

### L10s-SE

| Dimension | Status |
|---|---|
| **SIL-tested?** | Yes. Deterministic decision tree, civilian detection threshold, decision timeout ≤2 s, abort/continue path. FR-105 gates in 111/111 S5 suite. |
| **Coverage gaps** | No adversarial EO condition test (QA standing rule #2). Only clean synthetic inputs. Civilian detection uses synthetic scene classifier probability, not real sensor output. |
| **SIL-complete condition** | At minimum, adversarial synthetic scenario (civilian within L10s window with concurrent target) should be added. Full clearance requires DMRL CNN integration. |

### ZPI

| Dimension | Status |
|---|---|
| **SIL-tested?** | Yes. Burst scheduling, SHM enforcement, hop plan via HKDF-SHA256. Included in 36/36 S6 gates. |
| **Coverage gaps** | Phase-2 frozen. Not exercised in BCMP-1 or BCMP-2 mission runners. ZPI is present as a module but its integration with the mission loop is not gate-tested in any scenario runner. |
| **SIL-complete condition** | Integration with bcmp1_runner / bcmp2_runner end-to-end test required before operational clearance. |

### CEMS

| Dimension | Status |
|---|---|
| **SIL-tested?** | Yes, unit level. Cooperative merge, confidence weighting, temporal decay, replay window. Included in 36/36 S6 gates. |
| **Coverage gaps** | Phase-2 frozen. Not exercised in any mission scenario. bcmp1_cems_sim.py exists but is isolated from bcmp1_runner. No multi-UAV coordination test. |
| **SIL-complete condition** | Phase-2 scope. Requires multi-UAV scenario, CEMS wired into mission runner, CEMS active use test. |

### Mission Envelope

| Dimension | Status |
|---|---|
| **SIL-tested?** | Partially. MissionLogSchema (log completeness ≥99%) tested in S1 gates. bcmp1_runner.py enforcement E-1..E-5 tested in S-NEP-10 (two-theatre validation). |
| **Coverage gaps** | SRS §10.2 Mission Envelope Schema missing AVP speed/altitude fields (OI-09). FR-109–112 (cryptographic envelope signing, PQC-ready stack) not implemented (Phase-2). BCMP-1 pass criteria ↔ SRS traceability table missing (OI-10). |
| **SIL-complete condition** | OI-09 resolved; FR-109–112 Phase-2 scope completed; OI-10 traceability table produced. |

### State Machine

| Dimension | Status |
|---|---|
| **SIL-tested?** | Yes. 7-state FSM, transition guards, ≤2 s latency (NFR-002). Exercised in all BCMP-1 and BCMP-2 runner tests. |
| **Coverage gaps** | No adversarial FSM test (contradictory concurrent guard inputs). VIO OUTAGE → FSM state interaction (should OUTAGE trigger ST-03 GNSS_DENIED if combined with GNSS denial? Not specified). RC-11 integration with PX4 mode management (how PX4 OFFBOARD reacts to FSM transitions) not gate-tested. |
| **SIL-complete condition** | Adversarial concurrent transition scenario; VIO OUTAGE + GNSS denial FSM response defined; RC-11 integration test passed. |

---

*Digest complete. No code or existing qa/ files modified during this session.*
