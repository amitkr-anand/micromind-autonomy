# Deputy 1 Standing Notes
**Document ID:** TASL-MM-GOV-DSN-1.0  
**Authority:** Agent 1 (Architect Lead, Deputy 1)  
**Purpose:** Persistent cross-session notes for Deputy 1 — active open items,
sprint plan reference, and handoff reminders. Update at each session close.  
**Last updated:** 11 April 2026 (Handoff 1 final closure)

---

## Active Open Items

| ID | Severity | Summary |
|---|---|---|
| EF-01 | 🔴 OPEN | Vehicle A OFFBOARD failsafe — PX4 instance 1 (PX4_GZ_STANDALONE=1) triggers mc_pos_control invalid setpoints → blind land immediately after OFFBOARD engagement. Pre-existing; separate investigation required before external demo. |
| OI-37 | LOW | `MISSION_TIMEOUT_S = 300` hardcoded in `simulation/run_mission.py main()`. Must be moved to config before external demo builds. Documented in `simulation/TECHNICAL_NOTES.md`. |
| OI-40 | MEDIUM | EC-07 non-compliance — Corridor Violation (predicted) has no recovery ownership row in §16. `core/state_machine/state_machine.py` emits `CORRIDOR_VIOLATION` trigger. Fix: add §16 row in SRS v1.4 revision. See `docs/qa/SB5_EC07_OwnershipVerification.md`. |
| OI-41 | LOW | `core/bim/bim.py` structured log debt — uses stdlib `logging` rather than the programme's event_log dict pattern (req_id, severity, module_name, timestamp_ms). Deferred to DD-01 phased migration. Not blocking. |

## Active Process Rules

| Rule | Description | Severity | Status |
|---|---|---|---|
| Process rule — Deputy 2 commit discipline | Deputy 2 must commit ALL test artefacts and the QFR document to the repository BEFORE submitting the QFR to Deputy 1. A QFR that references uncommitted files will be rejected. Deputy 1 must verify every test file cited in a QFR with: `git log -- <file>` before countersigning. Agent 2 must not edit any file in `tests/` authored by Deputy 2. If a production code fix is needed to make a Deputy 2 test pass, Agent 2 fixes production code only — Deputy 2 re-runs their own test. Ref: PF-01, PF-02, PF-03 (11 Apr 2026). | HIGH | Permanent standing rule |

---

## SIL Baseline (Certified 11 April 2026)

**Total: 314/314**

| Suite | Count |
|---|---|
| S5 (`run_s5_tests.py`) | 119 |
| S8 (`run_s8_tests.py`) | 68 |
| BCMP2 (`run_bcmp2_tests.py`) | 90 |
| Pre-HIL RC+ADV (RC-11/RC-7/RC-8 + ADV-01–06) | 13 |
| SB-5 Phase A (SA-01–SA-07) | 7 |
| SB-5 Phase B (SB-01–SB-07 + RS-04) | 9 |
| EC-01 (IT-PX4-01 continuity gate) | 3 |
| Deputy 2 adversarial d2 (test_sb5_adversarial_d2.py) | 5 |
| **Total** | **314** |

---

## Sprint Plan Reference — SB-5 Remaining Prompts

| Prompt | Focus | Status |
|---|---|---|
| Prompt 7 | PLN-02 Retask R-01–R-06 + PLN-03 Dead-End (SB-01–SB-05) | ✅ COMPLETE — `6c405aa` |
| Prompt 8 | MM-04 Queue Latency + SB-06 — Phase B CLOSED | ✅ COMPLETE — `62456c4` |
| Prompt 9 | Housekeeping — OI-29/OI-02/OI-23 CLOSED | ✅ COMPLETE — `ec5d26f` |
| Prompt 11 (this session) | IT-PX4-01 OFFBOARD continuity gate — EC01-01–03 PASS, SIL 308/308 | ✅ COMPLETE — `c9c9e5c` |
| Prompt 10A | Deputy 1 Pre-Handoff Checklist (`DEPUTY1_PREHANDOFF_CHECKLIST.md`) | ✅ COMPLETE |
| Handoff 1 | Pass MRM to Deputy 2 for adversarial stress review | ✅ COMPLETE — QFR CLOSED, 314/314, Phase C authorised |

---

---

## Retired Notes

| Milestone | Summary | Date | Commits |
|---|---|---|---|
| Handoff 1 | MRM signed, QFR integrity failure resolved (PF-01/02/03), 314/314 certified, Phase C authorised | 11 Apr 2026 | Test: `f909a7c` QFR: `99fd55b` |

---

## Version History

| Version | Date | Author | Change |
|---|---|---|---|
| 1.0 | 11 April 2026 | Agent 1 (Deputy 1) | Initial issue — active items EF-01, OI-37, OI-40, OI-41; sprint plan ref; Handoff 1 reminder |
| 1.1 | 11 April 2026 | Agent 2 (Implementer) | Prompt 8 complete — MM-04 SB-06 PASS, Phase B CLOSED, OI-38 CLOSED, SIL 305/305 |
| 1.2 | 11 April 2026 | Agent 2 (Implementer) | Prompt 9 complete — OI-29/OI-02/OI-23 CLOSED, SIL 305/305 |
| 1.3 | 11 April 2026 | Agent 2 (Implementer) | Prompt 11 complete — IT-PX4-01 EC01-01–03 PASS, SIL 308/308 |
| 1.4 | 11 April 2026 | Agent 2 (Implementer) | Handoff 1 final closure — PF-01/02/03 process rules added, 314/314 SIL baseline confirmed, Phase C authorised, Handoff 1 reminder retired |
