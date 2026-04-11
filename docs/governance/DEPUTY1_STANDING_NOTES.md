# Deputy 1 Standing Notes
**Document ID:** TASL-MM-GOV-DSN-1.0  
**Authority:** Agent 1 (Architect Lead, Deputy 1)  
**Purpose:** Persistent cross-session notes for Deputy 1 — active open items,
sprint plan reference, and handoff reminders. Update at each session close.  
**Last updated:** 11 April 2026 (Prompt 11 session close, c9c9e5c)

---

## Active Open Items

| ID | Severity | Summary |
|---|---|---|
| EF-01 | 🔴 OPEN | Vehicle A OFFBOARD failsafe — PX4 instance 1 (PX4_GZ_STANDALONE=1) triggers mc_pos_control invalid setpoints → blind land immediately after OFFBOARD engagement. Pre-existing; separate investigation required before external demo. |
| OI-37 | LOW | `MISSION_TIMEOUT_S = 300` hardcoded in `simulation/run_mission.py main()`. Must be moved to config before external demo builds. Documented in `simulation/TECHNICAL_NOTES.md`. |
| OI-40 | MEDIUM | EC-07 non-compliance — Corridor Violation (predicted) has no recovery ownership row in §16. `core/state_machine/state_machine.py` emits `CORRIDOR_VIOLATION` trigger. Fix: add §16 row in SRS v1.4 revision. See `docs/qa/SB5_EC07_OwnershipVerification.md`. |
| OI-41 | LOW | `core/bim/bim.py` structured log debt — uses stdlib `logging` rather than the programme's event_log dict pattern (req_id, severity, module_name, timestamp_ms). Deferred to DD-01 phased migration. Not blocking. |

---

## Sprint Plan Reference — SB-5 Remaining Prompts

| Prompt | Focus | Status |
|---|---|---|
| Prompt 7 | PLN-02 Retask R-01–R-06 + PLN-03 Dead-End (SB-01–SB-05) | ✅ COMPLETE — `6c405aa` |
| Prompt 8 | MM-04 Queue Latency + SB-06 — Phase B CLOSED | ✅ COMPLETE — `62456c4` |
| Prompt 9 | Housekeeping — OI-29/OI-02/OI-23 CLOSED | ✅ COMPLETE — `ec5d26f` |
| Prompt 11 (this session) | IT-PX4-01 OFFBOARD continuity gate — EC01-01–03 PASS, SIL 308/308 | ✅ COMPLETE — `c9c9e5c` |
| Prompt 10A | Deputy 1 Pre-Handoff Checklist (`DEPUTY1_PREHANDOFF_CHECKLIST.md`) | 🔲 PENDING |
| Handoff 1 | Pass MRM to Deputy 2 for adversarial stress review | 🔲 PENDING |

---

## Handoff 1 Reminder

**Trigger:** After Prompt 10A Pre-Handoff Checklist — all five layers PASS.  
**MRM destination:** Deputy 2 (adversarial stress and logic-bleed detection).  
**Deputy 2's scope:** Adversarial SIL, fault injection, logic-bleed checks —
not hygiene issues. Deputy 1 must resolve all hygiene Hard Fails before passing.  
**MRM location:** `docs/handoffs/` (YAML format per §2.6).  
**Authority gate:** Deputy 1 countersigns MRM only when all five checklist
layers are clean and SIL baseline is confirmed.

### Pre-Handoff hard blockers to clear
- EF-01: Determine whether vehicle A OFFBOARD failsafe is in scope for Handoff 1
  or deferred to a post-HIL investigation sprint. Confirm with Programme Director.
- OI-40: §16 Corridor Violation row — SRS v1.4 fix must be committed or formally
  deferred with Programme Director sign-off before MRM is signed.

---

## Version History

| Version | Date | Author | Change |
|---|---|---|---|
| 1.0 | 11 April 2026 | Agent 1 (Deputy 1) | Initial issue — active items EF-01, OI-37, OI-40, OI-41; sprint plan ref; Handoff 1 reminder |
| 1.1 | 11 April 2026 | Agent 2 (Implementer) | Prompt 8 complete — MM-04 SB-06 PASS, Phase B CLOSED, OI-38 CLOSED, SIL 305/305 |
| 1.2 | 11 April 2026 | Agent 2 (Implementer) | Prompt 9 complete — OI-29/OI-02/OI-23 CLOSED, SIL 305/305 |
| 1.3 | 11 April 2026 | Agent 2 (Implementer) | Prompt 11 complete — IT-PX4-01 EC01-01–03 PASS, SIL 308/308 |
