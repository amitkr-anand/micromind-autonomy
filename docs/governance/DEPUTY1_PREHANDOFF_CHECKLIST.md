# DEPUTY 1 PRE-HANDOFF CHECKLIST
**Document ID:** TASL-MM-GOV-PHC-1.0  
**Authority:** Agent 1 (Architect Lead, Deputy 1)  
**Applies to:** All MRM submissions to Deputy 2  
**Trigger:** Run as Prompt 10A (Phase A+B handoff) and Prompt 19A 
(Phase C+D handoff) before MRM is signed  
**Purpose:** Ensure Deputy 2 receives a clean package requiring zero 
rework on hygiene issues. Deputy 2's time is reserved for adversarial 
stress and logic-bleed detection — not catching missing 
TECHNICAL_NOTES or bare except blocks.

---

## Execution Protocol

Deputy 1 issues this checklist to Agent 2 as a dedicated prompt. 
Agent 2 runs all checks and reports back per section. Deputy 1 
reviews the report, resolves any Hard Fails, and countersigns 
the MRM only when all five layers are clean. No commit is made 
during the checklist prompt — report only.

---

## Layer 1 — Deliverable Completeness

Did Agent 2 produce everything the sprint prompts required?

| Check | Method | Pass Criterion |
|---|---|---|
| Every prompt in scope has a corresponding commit | `git log --oneline` against prompt list | One commit per prompt, correctly messaged |
| Every gate file named in the sprint plan exists | `ls tests/test_sb5_phase_a.py tests/test_sb5_phase_b.py` etc. | Files present, not empty |
| Every gate named in the MRM maps to an actual pytest assertion | `grep -n "def test_" tests/test_sb5_*.py` | 1:1 mapping, no phantom gates |
| EC-07 ownership verification document committed | `ls docs/qa/SB5_EC07_OwnershipVerification.md` | File present, all 6 events covered |
| MRM YAML committed to docs/handoffs/ | `ls docs/handoffs/MRM_SB5_PHASE_AB*.yaml` | File present and well-formed |
| TECHNICAL_NOTES.md updated for every module touched | `git diff --name-only HEAD~N \| grep TECHNICAL_NOTES` | Present for all modules modified this phase |

---

## Layer 2 — No-Go List Static Analysis

The §1.4 universal hard fails, run as grep sweeps on all files 
changed in this phase. Agent 2 runs these exact commands and 
reports output:

```bash
# 1. Mutable global state
git diff --name-only HEAD~N | xargs grep -n "^global "

# 2. Silent exceptions (bare except with no logging call)
git diff --name-only HEAD~N | xargs grep -n -A2 "except:" \
  | grep -v "log_event\|logger\|logging"

# 3. Magic numbers in conditionals
git diff --name-only HEAD~N | \
  xargs grep -En "if .+ [<>!=]=? [0-9]+(\.[0-9]+)?[^_]"

# 4. Raw thread creation outside SystemScheduler
git diff --name-only HEAD~N | \
  xargs grep -n "threading\.Thread\|multiprocessing\.Process"

# 5. Code entropy markers
git diff --name-only HEAD~N | xargs grep -in "TODO\|FIXME\|HACK"

# 6. Velocity-dependent control logic (AD-19 / System Rule 1.8)
git diff --name-only HEAD~N | xargs grep -n "state\.v\b"

# 7. Forbidden timing calls
git diff --name-only HEAD~N | \
  xargs grep -n "time\.time()\|time\.clock()\|datetime\.now()"
```

**Pass criterion:** Zero hits on items 1, 2, 4, 5, 6, 7. Item 3 
(magic numbers) — any hit must be inspected and confirmed 
config-driven or a named constant with a documented reference. 
Any unexplained hit is a Hard Fail.

---

## Layer 3 — Logic Box Sovereignty

No cross-box contamination in files touched this phase.

| Check | Command | Pass Criterion |
|---|---|---|
| No MAVLink/Gazebo imports in core/ | `grep -rn "import mavlink\|import pymavlink\|import gz\|from gz" core/` | Zero hits |
| No mission logic in simulation/ PX4 bridge code | `grep -rn "mission_state\|envelope\|ROE\|L10s" simulation/` | Zero hits |
| No direct PX4 commands from Navigation Manager | `grep -rn "set_mode\|arm\|OFFBOARD" core/navigation/` | Zero hits |
| No ROE evaluation outside L10s-SE | `grep -rn "civilian_confidence\|ROE\|abort_authorised" core/` — exclude l10s_se.py | Zero hits outside l10s_se.py |
| No ESKF state modification from BIM | `grep -rn "eskf\.inject\|state\.pos\s*=" core/bim/` | Zero hits |

---

## Layer 4 — Logging Schema Compliance

Every new log_event() call introduced in this phase must carry 
all four mandatory fields: timestamp_ms, module_name, req_id, 
severity.

```bash
# Find all new log_event calls in this phase's diff
git diff HEAD~N -- '*.py' | grep "^+.*log_event("
```

Every call is listed and checked against the schema. Any call 
missing a required field is a Hard Fail.

---

## Layer 5 — MRM Self-Assessment Verification

The §2.6 deputy1_checklist inside the MRM YAML must be honest. 
Deputy 1 cross-checks each PASS claim:

| MRM Field | Verification Method |
|---|---|
| complexity_measured ≤ tier limit | Agent 2 runs `radon cc -s <file>` on each new module and confirms |
| frozen_files_modified: [] | `git diff HEAD~N -- core/eskf/ core/l10s_se/ core/dmrl/` must be empty |
| config_keys_added list complete | `git diff HEAD~N -- config/` — all new keys listed |
| requirements list matches prompt Req IDs | Cross-check against sprint plan prompt headers |
| test_guidance fault injection scenarios present | Check against §9.2 FI matrix for modules touched |

---

## Hard Fail vs. Noted

| Finding | Action |
|---|---|
| Any §1.4 No-Go violation | Hard block — fix before MRM signed, full SIL re-run after fix |
| Any Logic Box sovereignty breach | Hard block — same |
| Missing TECHNICAL_NOTES.md for any module touched | Hard block — §2.5 Human-Readiness Mandate is non-negotiable |
| Missing gate assertion (phantom gate in MRM) | Hard block — gate must exist as a pytest assertion |
| Log event missing required field | Hard block |
| Magic number with no config reference | Hard block |
| Frozen file touched | Escalate to Programme Director immediately — do not sign MRM |
| TECHNICAL_NOTES.md present but missing OODA rationale | Noted — Agent 2 fixes in same session before commit |
| MRM complexity claim not verified by radon | Noted — Agent 2 runs radon and updates MRM |
| Minor commit message format deviation | Noted — no block, logged for hygiene |

---

## The Prompt Agent 2 Receives (Prompt 10A / 19A)
Deputy 1 Pre-Handoff Review — Phase [A+B / C+D]
Before the MRM is signed and passed to Deputy 2, run the
Deputy 1 Pre-Handoff Checklist (docs/governance/
DEPUTY1_PREHANDOFF_CHECKLIST.md) against all files changed
across Prompts [1–12 / 13–19].
Report in this exact structure:
LAYER 1 — DELIVERABLE COMPLETENESS
[each item: PASS / FAIL / finding]
LAYER 2 — NO-GO STATIC ANALYSIS
[paste grep output for each of the 7 checks;
state CLEAN or list hits with file:line]
LAYER 3 — LOGIC BOX SOVEREIGNTY
[each check: CLEAN or list hits with file:line]
LAYER 4 — LOGGING SCHEMA COMPLIANCE
[list every new log_event call; COMPLIANT or list violations]
LAYER 5 — MRM SELF-ASSESSMENT VERIFICATION
[each field: VERIFIED or discrepancy noted]
SUMMARY: READY FOR HANDOFF / ISSUES TO RESOLVE
Do not commit anything in this prompt. Report only.
Deputy 1 will review and issue fix instructions before
countersigning the MRM.

---

## Version History

| Version | Date | Author | Change |
|---|---|---|---|
| 1.0 | 10 April 2026 | Agent 1 (Deputy 1) | Initial issue |
