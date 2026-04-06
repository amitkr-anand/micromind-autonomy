# MicroMind QA Log
**Format:** One entry per session. Append; never delete. Most recent at top.  
**Owner:** QA Agent (Claude) + Programme Director (Amit)

---

## Entry QA-007 — 06 April 2026
**Session Type:** Sprint
**Focus:** BCMP-2 SB-5 — AT-6 repeatability and endurance (16/17 gates)

**Actions completed:**
1. Entry check: 290/290 tests green, HEAD e703486, all 9 SB-5 entry criteria satisfied including EC-5 tag `sb4-dashboard-replay` confirmed present.
2. Runner extension (67ebe5d): `_extract_bcmp1_kpis()` extended to surface `fsm_history` as `phase_sequence`. Handles both list-of-dicts (S5) and list-of-strings (S8-E) formats. `vehicle_b_phase_sequence` added to `run_bcmp2()` top-level result dict. No frozen files touched. 90/90 bcmp2 tests held green throughout.
3. `test_bcmp2_at6.py` written (67ebe5d): 17 gates, 4 groups. pytest `scope="module"` fixtures for seeds 42/101/303. Endurance gates marked `@pytest.mark.endurance`, configurable via `AT6_ENDURANCE_HOURS` env var.
4. G-14 RSS warmup fix: initial sampling position caused startup module-load spike to distort linear regression. Fix: sample RSS post-mission (not pre), excluding cold-start import overhead from slope. Post-fix slope confirmed −23.8 MB/hr (stable) on 5-minute CI run.
5. 5-minute CI endurance run confirmed green: G-13 zero crashes, G-14 slope=−23.8 MB/hr, G-15 completeness=1.0.
6. Overnight 4-hour run launched in tmux session `at6_overnight`. Log at `logs/at6_endurance_overnight_*.log`.
7. Context file and OI register updated (f9ee7d4). OI-29 added for pytest.ini endurance marker warning.

**Key QA findings:**
- [MEDIUM — pre-code] G-10/11/12 phase chain had no implementation in runner — `fsm_history` existed in `bcmp1_runner` but was silently dropped by `_extract_bcmp1_kpis()`. Surfaced without touching frozen files.
- [MEDIUM — pre-code] Seed 303 is virgin (not used in AT-1 through AT-5). G-10/11/12 for seed 303 are verified against canonical chain reference, not self-comparison. On record.
- [HIGH — resolved] G-14 RSS slope inconsistent between runs — diagnosed as startup allocation artefact, not leak. Process stable at 231 MB across 31 missions. Warmup filter fix correct and verified.
- [LOW — OI-29] pytest.ini missing endurance marker registration. Cosmetic warning only.

**Gate summary — all 17 PASS:**
- G-01–G-09 (drift envelope, 3 seeds): ✅ PASS
- G-10–G-12 (phase chain, 3 seeds): ✅ PASS
- G-13–G-15 (endurance, 4-hour overnight): ✅ PASS — 1483 missions, 0 crashes, slope 1.135 MB/hr, completeness 1.0000
- G-16 (HTML reports, 3 seeds): ✅ PASS
- G-17 (closure report): ✅ PASS — `artifacts/BCMP2_ClosureReport.md` committed

**Overnight endurance evidence (micromind-node01, tmux at6_overnight):**
- Duration: 14407 s (4.000 h)
- G-13: missions=1483, crashes=0
- G-14: RSS slope=1.135 MB/hr over 213 samples
- G-15: log_completeness=1.0000 (1483/1483)
- Log: `logs/at6_endurance_overnight_*.log`

**Regression baseline:** 290 tests green + 17/17 AT-6 gates = **307 total gates PASS**

**OI status:** OI-29 opened (LOW). All prior OIs unchanged.

**SB-5 declared CLOSED. Tag: `sb5-bcmp2-closure`. BCMP-2 fully closed: 107/107 gates.**

**Next milestone:** S-NEP-03 (EuRoC end-to-end with real MetricSet).

---

## Entry QA-007b — 06 April 2026 (SB-5 Closure)
**Session Type:** Sprint — continuation
**Focus:** BCMP-2 SB-5 final closure — overnight endurance results, closure report, G-17

**Actions completed:**
1. Overnight 4-hour endurance results confirmed: G-13 zero crashes (1483 missions), G-14 RSS slope 1.135 MB/hr (22× margin), G-15 completeness 1.0000 (1483/1483). All three gates pass at full duration.
2. BCMP2_ClosureReport.md authored and committed (e9e8cb0). All 5 mandatory SIL caveats present: BASELINE, RADALT, DMRL, AD-15, EuRoC. All 5 required section headers present.
3. G-17 passes. 17/17 AT-6 gates green. 107/107 total BCMP-2 gates across SB-1 through SB-5.
4. Tag sb5-bcmp2-closure applied. Full sprint tag chain SB-1 through SB-5 intact.
5. Context file updated: SB-5 row changed to CLOSED.

**Key QA note — G-14 fix clarification (verified against committed code):**
The G-14 fix moves RSS sampling to post-mission position (after run_bcmp2()
returns) rather than pre-mission. This single change excludes the one-time
Python module-load allocation from the linear regression. _WARMUP_S and
warm_trace were designed in session but not committed — post-mission sampling
alone was sufficient. Confirmed: grep of test_bcmp2_at6.py shows no _WARMUP_S
or warm_trace. Overnight evidence: 1.135 MB/hr across 213 samples over 4 hours
on post-mission sampling only.

**Gate summary — final:**
- G-01–G-09 (drift envelope, 3 seeds): ✅ PASS
- G-10–G-12 (phase chain, 3 seeds): ✅ PASS
- G-13–G-15 (endurance, 4-hour): ✅ PASS
- G-16 (HTML reports): ✅ PASS
- G-17 (closure report): ✅ PASS

**Regression baseline at closure:** 290 tests green + 17/17 AT-6 gates
**Tag:** sb5-bcmp2-closure (e9e8cb0)

**Next programme milestone:** S-NEP-03 — EuRoC end-to-end with real MetricSet.

---

## Entry QA-006 — 05 April 2026
**Session Type:** Sprint
**Focus:** Sprint D — Pre-HIL completion. RC-11, RC-7, RC-8 (OI-16, OI-17, OI-18). SetpointCoordinator implementation.

**Actions completed:**
1. Code reading session (4972110): vio_mode.py had zero logs; ESKF had no isfinite guards; mark_send confirmed natively integrated at mavlink_bridge.py lines 358-359 (OI-21 stale); LivePipeline.setpoint_queue and MAVLinkBridge.update_setpoint() were unconnected.
2. Specification (2625050): Sprint D spec written first — 4 deliverables, 9 SD gates, RC-11a–d + RC-7 + RC-8 specifications, including Jetson caveats.
3. vio_mode.py logging (308016b): PD-authorised frozen file modification. Three log insertions: VIO_OUTAGE_DETECTED (WARNING), VIO_RESUMPTION_STARTED (INFO), VIO_NOMINAL_RESTORED (INFO). Backup created at vio_mode_FROZEN_BASELINE.py.
4. SetpointCoordinator (7bebc8c): External wiring pattern — drains LivePipeline.setpoint_queue at 50 Hz, keeps most-recent setpoint, calls bridge.update_setpoint(). Does not modify LivePipeline or MAVLinkBridge.
5. test_prehil_rc11.py (7bebc8c): RC-11a (OUTAGE detection), RC-11b (6000-step ESKF NaN stability), RC-11c (setpoint continuity), RC-11d (RESUMPTION→NOMINAL correctness). All 4 pass.
6. test_prehil_rc7.py (7bebc8c): IFM-01 monotonicity injection — violation_count==1 after one bad timestamp, subsequent frames accepted. SD-06 PASS.
7. test_prehil_rc8.py (7bebc8c): FusionLogger 12000 entries at 200 Hz — completeness=1.0, worst_call=0.173 ms. SD-07 PASS.

**Key engineering finding:**
- LivePipeline not importable in SIL (psutil absent, OI-13 pre-existing). All RC-11 tests drive VIONavigationMode and ErrorStateEKF directly. SetpointCoordinator tested with _MockPipeline + _MockBridge.
- FusionLogger is fully synchronous (in-memory list append) — no async queue, no T-LOG thread. RC-8 "drop_count" is computed as submitted − written.
- RC-11a outage_threshold_s=0.2 used in test (not 2.0) — tests the detection mechanism, not the production threshold value.

**Nine SD gates status:**
- SD-01 RC-11a OUTAGE detected within 500 ms, log present: PASS
- SD-02 RC-11b zero NaN across 6000 steps: PASS
- SD-03 RC-11c setpoints forwarded, finite, non-frozen, rate >= 20 Hz: PASS
- SD-04 RC-11d NOMINAL restored within 2 s, no jump > 50 m: PASS
- SD-05 RC-11e 119/119 + 68/68 + 90/90 + 6/6 = 283: PASS
- SD-06 RC-7 IFM-01 violation_count==1: PASS
- SD-07 RC-8 completeness=1.0, worst_call=0.173 ms: PASS
- SD-08 SetpointCoordinator frozen files untouched: PASS
- SD-09 Jetson caveat in RC-11b and RC-8 output: PASS

**OI closures:**
- [HIGH — OI-16 CLOSED] RC-11 all criteria met. SetpointCoordinator wired. vio_mode.py logging present.
- [HIGH — OI-17 CLOSED] RC-7 IFM-01 guard directly tested. violation_count and event_id confirmed.
- [HIGH — OI-18 CLOSED] RC-8 completeness >= 0.99, no blocking call > 5 ms.
- [MEDIUM — OI-21 CLOSED] mark_send confirmed natively integrated at mavlink_bridge.py:358-359. CP-2 asterisk withdrawn.

**Regression baseline:** 283 tests green
  (119 S5 + 68 S8 + 90 BCMP-2 + 6 ADV)

**CP-3 status:** OI-16, OI-17, OI-18 now closed. CP-3 Pre-HIL declaration prerequisites met (pending programme director review).

**Next sprint:** CP-3 declaration review, then SB-5 (BCMP-2 repeatability + closure) per AT-6 acceptance criteria (17 gates, defined at docs/qa/AT6_Acceptance_Criteria.md).

---

## Entry QA-005 — 05 April 2026
**Session Type:** Sprint
**Focus:** Sprint C — OrthophotoMatchingStub, terrain texture cost, featureless terrain test (OI-05, OI-08, OI-11)

**Actions completed:**
1. Architecture specification produced by QA agent, committed at c5ac91a before any code written.
2. Implementation (96bf98a): orthophoto_matching_stub.py (326 lines), hybrid_astar.py texture cost term, test_sprint_c_om_stub.py (8 tests). All 8 SC gates PASS.
3. SC-06 conflict resolved: grep narrowed to implementation artifacts (RadarAltimeterSim, DEMProvider, elevation strip) — header provenance text retained. Claude Code correctly stopped and reported the contradiction rather than deciding.
4. Tests integrated into run_s5_tests.py runner (6af0e4b): 111 → 119 tests. Converted from pytest to unittest.TestCase to match existing pattern.

**Key QA findings:**
- [HIGH — OI-05 CLOSED] OM stub correctly implements measurement-provider-only pattern (AD-03). R matrix confirmed 81.0 m² not old 225 m².
- [HIGH — OI-08 CLOSED] Texture cost default=30.0 preserves all existing test behaviour. Zero existing tests affected.
- [HIGH — OI-11 CLOSED] Featureless terrain failure mode (sigma=5, 14 km, zero corrections) exercised for first time. Was structurally untestable with synthetic DEM.
- [MEDIUM] ADV-07 (corridor violation integration path) still deferred — noted in adversarial test file.

**Regression baseline:** 283 tests green
  (119 S5 + 68 S8 + 90 BCMP-2 + 6 ADV)

**Next sprint:** Sprint D — Pre-HIL completion.
  RC-7 (timestamp monotonicity), RC-8 (logger non-blocking 200 Hz), RC-11 (VIO OUTAGE + setpoint continuity). Closes OI-16, OI-17, OI-18.
  Requires CP-3 before Pre-HIL can be declared.

---

## Entry QA-004 — 04 April 2026
**Session Type:** Sprint
**Focus:** Sprint B — L10s-SE and DMRL adversarial SIL (OI-26)

**Actions completed:**
1. Code reading (88c077e): established that inputs_from_dmrl() always defaulted civilian_confidence=0.0; Gate 3 had never been reached through DMRL integration path in any prior test.
2. Scenario specification (88c077e): 6 adversarial scenarios ADV-01 through ADV-06 defined, reviewed by QA agent, approved before any code written.
3. Test implementation (41238ae): tests/test_s5_l10s_se_adversarial.py — 6/6 pass. Full regression: 111/111 + 68/68 + 90/90 green.

**Deviations from spec (both approved):**
- ADV-04: scene-level architecture required — decoy-only DMRL call structurally cannot acquire lock (cap < 0.84). Fix: process real target for lock + decoy target for is_decoy flag. More realistic than spec.
- ADV-06: thermal_signature raised 0.75 → 0.88. Spec value structurally below 0.85 gate. Prerequisite guard caught this correctly.

**Known follow-on item (not blocking):**
  DMRL lock confidence formula stochastic term makes boundary thermal_signature values unreliable for test design. Document formula before S-NEP-04 integration.

**Findings:**
- [HIGH — OI-26 CLOSED] Gate 3 civilian detection was unreachable through integration path. Now covered by ADV-01, ADV-03, ADV-06.
- [MEDIUM] ADV-07 (corridor violation integration path) deferred — noted in test file as known gap.
- [LOW] DMRL lock confidence formula needs documentation before S-NEP-04.

**Next sprint:** Sprint C — Orthophoto Matching stub + route planner texture cost (OI-05, OI-08, OI-11). Largest sprint. Re-grounds the navigation claim on correct evidence.

---

## Entry QA-003 — 04 April 2026
**Session Type:** Documentation
**Focus:** Sprint 0 — governing document conflict resolution complete

**Actions completed:**
1. Part Two V7.2 produced (b2bae3d + 605a747): 12 amendments applied. Navigation L1/L2/L3 architecture, RADALT scoped to terminal, OM replaces RADALT-NCC, ZPI schema, L10s-SE CNN gate, SHM HIL RF gate, state machine VIO gap closed, authority chain hash failure added, BIM adaptive spoof KPI, §1.15 residual fix.
2. SRS v1.3 produced (2600977): 12 amendments applied. NAV-02 rewritten for orthophoto matching. AVP-01 deferred. AVP fields in §10.2. AVP fallback events in §10.16. GAP-10/11/12 and AMB-06 added. EC-13 added to §17 SB-5 entry criteria.
3. All 10 conflicts from review document closed.
4. Test suites held green throughout: 111/111, 68/68, 90/90.

**Open items status after this session:**
- OI-05: NAV-02 v1.3 rewritten to match AD-01. SIL tests still required (GAP-10).
- OI-09: CLOSED — Mission Envelope Schema AVP fields added (§10.2, Amendment 7).
- OI-10: CLOSED — BCMP-1 traceability table added (Part Two V7.2 §5.3.3, Amendment 11).

**Next session:** Sprint B — L10s-SE and DMRL adversarial SIL (OI-26). Define adversarial synthetic EO scenarios before writing any code.

---

## Entry QA-002 — 04 April 2026
**Session Type:** Documentation
**Focus:** Architecture decisions register completion; context file maintenance

**Actions completed:**
1. AD-03 through AD-21 drafted by QA agent and reviewed by Programme Director before commit. All 19 previously undocumented decisions now in MICROMIND_DECISIONS.md.
2. Three decisions revised before commit based on Programme Director corrections: AD-10 (Ubuntu 24.04 platform rationale confirmed against ROS2 official docs — 24.04 is correct tier-1 Jazzy platform, not a compromise), AD-11 (clock ownership scoped to SITL only; HIL/production requires shared hardware clock source), AD-15 (Vehicle A reframed as illustrative drift model, not precision mechanisation).
3. SB-4 confirmed CLOSED (tag c183b9c, 31 March 2026) — context file corrected.
4. OI-14 and OI-15 closed. OI-19 (AT-6 acceptance criteria) remains open — next session.

**Findings:**
- [HIGH — OI-15 CLOSED] 19 of 21 architecture decisions were undocumented. Now resolved.
- [LOW — OI-14 CLOSED] Context file showed SB-4 pending; git history confirmed closed 31 March.
- [MEDIUM — OI-19 OPEN] AT-6 gate count and acceptance criteria still undefined. Must be specified before SB-5 sprint begins.

**Next Session:** Define AT-6 acceptance criteria (OI-19). Then proceed to Sprint A completion.

---

## Entry QA-001 — 03 April 2026
**Session Type:** Onboarding / Architecture Review  
**Focus:** Programme context establishment, navigation architecture decision, project folder setup

**Findings:**
1. **[CRITICAL — OI-05]** `trn_stub.py` implies RADALT-NCC as production correction mechanism. Architecture decision taken 03 April to replace ingress correction with orthophoto image matching. Stub must be updated before fusion integration (S-NEP-04) to avoid incorrect implementation assumption being frozen into the ESKF interface.
2. **[HIGH — OI-04]** OpenVINS → ESKF interface not documented. Message format, covariance representation, and FM event handling protocol must be specified before S-NEP-04 code is written.
3. **[HIGH — OI-01]** STIM300 ARW 0.15°/√hr exceeds V7 spec floor of 0.1°/√hr. Spec must be updated to ≤ 0.2°/√hr before TASL meeting. Confirmed as S8 finding — not yet actioned in the spec document.
4. **[HIGH — OI-07]** OpenVINS Stage-2 GO verdict is indoor / short-sequence only (≤ 130 m). Km-scale and outdoor validation pending. This must be stated explicitly in any external-facing document referencing VIO performance.
5. **[MEDIUM — OI-06]** DMRL stub is rule-based. All BCMP-1 terminal guidance acceptance results (AT-2 through AT-5) are stub-based. Any report sent to TASL or external audience must include this caveat.

**Architecture Decision Recorded:**
- Navigation ingress correction: RADALT-NCC TRN → Orthophoto image matching (L2 Absolute Reset layer)
- RADALT retained for terminal phase only (0–300 m AGL, final 5 km)
- LWIR camera declared dual-use: orthophoto matching (ingress) + DMRL (terminal)
- Route planner terrain-texture cost term required (OI-08)

**Context File:** Created. Sections 1–10 populated.  
**Session Start Prompt:** Created.  
**Next Session:** Share milestone reports and ALS-250 overnight run results for QA review.

---
*Append new entries above this line.*

---

## Entry QA-009 — 06 April 2026
**Session Type:** Sprint + QA Audit + Documentation
**Focus:** S-NEP-03R remediation, S-NEP-04 through S-NEP-09 gate formalisation, OI-04 closure

**Actions completed:**
1. Falsifiability audit of S-NEP-03 through S-NEP-09. Seven findings documented (F-01 through F-07). Core finding: S-NEP-04 through S-NEP-09 had zero pytest-enforced acceptance criteria — all gate evaluations were print() statements. The end-to-end MetricSet pipeline had never produced a valid result (EXP_VIO_013: ATE=12.17 m, tracking_loss=100%, acceptance_pass=false — accepted on status=complete only).
2. F-05 cleared — error_state_ekf.py unmodified since freeze tag. Docstring bug logged OI-NEW-01.
3. Five surgical fixes to evaluation/metrics_engine.py in nep-vio-sandbox: (a) Umeyama alignment inserted before APE.process_data(), (b) RPE block wrapped in inner try/except — RPE failure no longer discards ATE, (c) _compute_trajectory_errors() returns 6-tuple; _compute_drift() now receives aligned positions, (d) feature_count < 20 tracking loss condition removed (OI-NEW-02), (e) RPE delta unit Unit.seconds → Unit.frames delta=1 (evo 1.34.3 incompatibility, OI-NEW-03).
4. Two tests rewritten in test_metrics_engine.py for aligned ATE semantics. drifting_poses() helper added.
5. Root cause of ATE=12.17 m identified: FilterException('unsupported delta unit: Unit.seconds') in RPE silently discarded every correct ATE result and fell to centroid-only fallback which cannot correct ~180° frame rotation. Fix resolved ATE to 0.087 m matching Stage-2 benchmark.
6. S-NEP-03R gate file tests/test_snep03r_e2e.py committed (0a93567 / ae0d563): 21 pytest assertions across 8 gates. 464/464 PASS.
7. Retroactive pytest gates written for S-NEP-04 through S-NEP-09: test_snep04_gates.py (10 gates), test_snep05_gates.py (5 gates), test_snep06_gates.py (10 gates), test_snep08_gates.py (7 gates), test_snep09_gates.py (10 gates). F-01 closed. Committed 520b52e.
8. OI-04 closed — docs/OpenVINS_ESKF_Interface_Spec.md written and committed a014997 in nep-vio-sandbox. Consolidates frame conventions, ROS2 field mapping, IFM fault modes, ESKF update signature, test gate registry, frozen file registry.
9. OI-NEW-01 closed — update_vio() docstring corrected to 3-tuple return signature. Committed f18c5e9.
10. Context file updated — all sprint rows corrected, OI register updated, S-NEP-10 marked READY. Committed 01de1c3.

**Key engineering findings:**
- Silent ATE discard: FilterException('unsupported delta unit: Unit.seconds') in evo RPE caused entire evo block to fall to unaligned fallback on every real-data run since S-NEP-03.
- Frame rotation gap: centroid alignment in fallback cannot correct ~180° Umeyama rotation between OpenVINS world frame and EuRoC GT frame. Produced 7.34 m after centroid alignment vs 0.087 m after SE3 Umeyama.
- _compute_drift() was receiving raw unaligned positions — produced 136.87 m/km vs 0.912 m/km after aligned positions passed.
- S-NEP-05 BOUNDED classification is self-disclaimed (r2_linear=0.149, below 0.3 floor). Gate G-05-06 pins the weak-fit caveat rather than asserting the classification.
- S-NEP-06 ctrl2 divergences (div=True for ≥10s outages) superseded by ctrl3. Gate G-06-08 explicitly guards against regression.

**Gate summary:**
- S-NEP-03R: 21/21 PASS (tag 0a93567 / ae0d563)
- S-NEP-04 retroactive: 10/10 PASS
- S-NEP-05 retroactive: 5/5 PASS
- S-NEP-06 retroactive: 10/10 PASS
- S-NEP-08 retroactive: 7/7 PASS
- S-NEP-09 retroactive: 10/10 PASS
- Full suite: 531/531 PASS

**OI status changes:**
- OI-04 CLOSED: OpenVINS_ESKF_Interface_Spec.md committed a014997 in nep-vio-sandbox
- OI-NEW-01 CLOSED: update_vio() docstring corrected f18c5e9
- OI-NEW-02 OPEN: MetricsEngine feature_count gate removed — reinstate when parser emits real counts
- OI-NEW-03 OPEN: RPE 1-frame windows (evo 1.34.3 Unit.seconds incompatibility) — fix before external report

**Findings carried forward:**
- F-04 HIGH: NIS EC-02 never passed (mean 0.003 vs floor 0.5) — TD decision required to retire under PF-03 or fix covariance
- F-06 MEDIUM: Stage-2 GO drift proxy formula not equivalent to NAV-03 km-scale criterion — document in any closure report
- F-07 MEDIUM: S-NEP-05 BOUNDED classification self-disclaimed — pinned in G-05-06, not resolved
- OI-07 HIGH: Outdoor km-scale VIO validation pending

**Next milestone:** S-NEP-10 — OpenVINS → ESKF full integration on EuRoC MH_03 and V1_01.
