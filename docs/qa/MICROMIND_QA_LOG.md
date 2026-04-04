# MicroMind QA Log
**Format:** One entry per session. Append; never delete. Most recent at top.  
**Owner:** QA Agent (Claude) + Programme Director (Amit)

---

## Entry QA-003 — 04 April 2026
**Session Type:** Documentation
**Focus:** SRS v1.3 production — 12 amendments applied

**Actions completed:**
1. MicroMind_SRS_v1_3.docx produced from MicroMind_SRS_v1_2_1.docx (original MD5 e3272feb preserved unchanged).
2. Amendments 0–11 applied: version history updated; §1.1 AVP-01 deferred; §1.3 taxonomy extended with AVP_ALT_REDUCED and AVP_SPD_REDUCED; §1.4 AVP-01 row removed; §2.1 NAV-01 calibration caveat row added; §2.2 NAV-02 fully rewritten for L2 Absolute Reset / Orthophoto Matching, SIL status note appended; §5.2 MM-02 AVP constraint enforcement row added; §10.2 Mission Envelope Schema extended with 6 AVP fields; §10.16 logging schema extended with 6 AVP/OM events; unit test Entry Criteria updated with AVP scope notes (7 tests); §14 traceability NAV-02 status revised, FR-107 and FR-110a added; §15 GAP-10/11/12 and AMB-06 added; §17 EC-13 added.
3. All 11 string checks PASS. Source MD5 unchanged.
4. Test suites green: 111/111 (S5), 68/68 (S8), 90/90 (BCMP-2).
5. MICROMIND_PROJECT_CONTEXT.md Section 10 updated: SRS reference changed from v1.2.1 to v1.3.
6. Commit: 2600977.

**Findings:**
- [HIGH — OI-05 CONFIRMED] NAV-02 replacement required to document that TRN stub still implements RADALT-NCC; UT-NAV-02-A/B now explicitly marked rewrite-required in SRS.
- [HIGH — GAP-10 NEW] Zero SIL tests for orthophoto matching. Captured in §15 Critical Gaps and §17 EC-13.
- [HIGH — GAP-11 NEW] C-2 drift envelopes not characterised beyond km 120 for AVP-04.
- [MEDIUM — GAP-12 NEW] BIM adaptive spoof resistance test (UT-BIM-03) not yet written.
- [MEDIUM — AMB-06 NEW] FR-108 (satellite avoidance) has no test ID or KPI in SRS.

**Next Session:** Sprint 0 Step 3 of 3 (pending).

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
