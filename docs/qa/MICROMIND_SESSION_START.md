# MicroMind Session Start Prompt
**Use this prompt verbatim at the start of every new Claude session.**  
**Paste it as your first message. Do not add pleasantries or context before it.**

---

## PROMPT (copy everything between the lines)

---

You are the QA agent for the MicroMind / NanoCorteX programme. Read `MICROMIND_PROJECT_CONTEXT.md` first — it is your complete briefing. Do not ask me to re-explain the programme.

**Your role:** Rigorous, independent technical reviewer. You review code, test results, and architecture decisions against the SRS. You flag gaps where tests pass by construction rather than genuine falsification. You maintain the standing QA rules in Section 9 of the context file.

**Session type today:** [REPLACE WITH ONE OF THE BELOW]
- `CODE REVIEW` — I will share code files for review against SRS requirements
- `SPRINT` — We are building a new sprint deliverable together  
- `TEST REVIEW` — I will share test output logs for QA assessment
- `ARCHITECTURE` — We are making or reviewing a design decision
- `DOCUMENTATION` — Updating specs, SRS, or open items

**Today's focus:** [REPLACE WITH 1–2 SENTENCES describing what you want to accomplish]

**Programme state as of today:** [PASTE the current Section 6 table from MICROMIND_PROJECT_CONTEXT.md, updated if anything has changed]

**Open items relevant to today:** [LIST the OI numbers from Section 8 that are relevant, e.g. OI-04, OI-05]

Before starting work, confirm you have read the context file and state in one sentence what you understand today's goal to be. Then ask one question only if something is genuinely ambiguous — do not ask for information already in the context file.

---

## How to Use This System

### File Structure to Maintain in Your Project Folder

```
MicroMind_Project/
├── MICROMIND_PROJECT_CONTEXT.md     ← Master context. Load every session.
├── MICROMIND_SESSION_START.md       ← This file. Copy the prompt from here.
├── MICROMIND_QA_LOG.md              ← Running log of QA findings (append each session)
├── MICROMIND_DECISIONS.md           ← Architecture decisions with rationale
└── Sessions/
    ├── 2026-04-03_navigation_arch.md    ← Today's session notes
    ├── 2026-03-30_bcmp2_sb3.md
    └── [date]_[topic].md
```

### Session Workflow (5 Steps)

**Step 1 — Session Start (~30 seconds)**  
Paste the session start prompt. Claude reads context, confirms understanding, asks at most one question.

**Step 2 — Work**  
Do the actual work. Code review, sprint, test analysis, whatever the session type is.

**Step 3 — Session End Update (before closing)**  
Ask Claude: *"Update Section 6 and Section 8 of the context file for today's changes."*  
Copy the updated sections back into `MICROMIND_PROJECT_CONTEXT.md`.

**Step 4 — QA Log Entry**  
Ask Claude: *"Write a 5-line QA log entry for today's session."*  
Append to `MICROMIND_QA_LOG.md`.

**Step 5 — Session Note**  
Save the conversation summary as `Sessions/[date]_[topic].md` for the record.

### Token Economy Rules

| What | Rule |
|---|---|
| Context file | Load once per session. Never paste it more than once. |
| SRS | Load only when reviewing specific requirements. Reference by section number otherwise. |
| Code files | Paste only the file(s) being reviewed. Not the whole codebase. |
| Test logs | Paste only the relevant test output, not 500 lines of passing tests. |
| Frozen files | Never paste these unless there is a specific QA concern about them. |
| Prior sessions | Do not re-paste old conversation history. The context file captures decisions. |

### What Goes in the Context File vs Session Notes

| Content | Context File | Session Note |
|---|---|---|
| Current sprint state | ✅ Section 6 | — |
| Open items | ✅ Section 8 | — |
| Architecture decisions | ✅ Section 8 or DECISIONS.md | Background reasoning |
| QA findings | ✅ Section 8 (if open) | Full detail in QA log |
| Code review detail | — | ✅ Full in session note |
| Test output analysis | — | ✅ Full in session note |
| What was built this session | Update Section 6 | ✅ How and why |

---

## Reference: Session Type Templates

### CODE REVIEW session
After start prompt:
> *"Here is [filename]. Review it against [requirement ID / module spec]. I want to know: (1) does it correctly implement the requirement, (2) are there any edge cases the test suite does not cover, (3) any frozen file violations."*

### SPRINT session  
After start prompt:
> *"We are building [deliverable] for [sprint ID]. Entry checklist first — confirm gates are green before we write any code. Then let's define the acceptance criteria before implementation."*

### TEST REVIEW session
After start prompt:
> *"Here is the test output from [test suite / run]. Review against SRS requirements [list]. Flag any pass-by-construction results and any missing coverage."*

### ARCHITECTURE session
After start prompt:
> *"I want to make a decision about [topic]. Here is the options analysis: [brief]. I need you to: (1) check against the constraints in the context file, (2) identify which open items this affects, (3) recommend and document the decision in DECISIONS.md format."*

---

## Red Lines — Never Do These

- Do not start a session without loading `MICROMIND_PROJECT_CONTEXT.md`
- Do not present EuRoC Stage-2 results as mission-scale evidence
- Do not modify frozen files without explicit approval + full regression re-run
- Do not write DMRL test results as if the CNN is implemented (it is a stub)
- Do not approve a test result without checking the test environment against operational conditions
- Do not let a session end without updating the context file Section 6 and Section 8
