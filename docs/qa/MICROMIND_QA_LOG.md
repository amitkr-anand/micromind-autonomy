# MicroMind QA Log
**Format:** One entry per session. Append; never delete. Most recent at top.  
**Owner:** QA Agent (Claude) + Programme Director (Amit)

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
