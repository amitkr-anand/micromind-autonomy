# Deputy 1 — Final Two-Week Prompt

Deputy 1,

Your proposed two-week plan is approved with the following corrections, sequencing rules, and an added governance requirement.

The focus remains core product compliance only. Do not divert effort toward live demo preparation, operator-facing visualisation, dual-vehicle showcase flows, or polished presentation features. The objective of this sprint is to improve engineering correctness, SRS compliance, recovery behaviour, and evidence-backed closure.

# Priority Structure

There are three active threads:

- Thread A — Mission autonomy correctness and recovery logic
- Thread B — Navigation validation and operational envelope maturation
- Thread C — Orin deployment readiness

Thread A has absolute priority. Thread B may proceed only after Thread A gaps are understood. Thread C remains blocked until Thread A is stable and SIL remains green.

# Additional Governance Requirement

Before starting new implementation work, create:

`docs/qa/SRS_COMPLIANCE_MATRIX.md`

This document becomes the authoritative weekly compliance tracker and must be updated every week before programme review.

The matrix must not become a generic checklist. It must remain evidence-based and linked to code, tests, commits, and SRS clauses.

Use the following columns:

| SRS Section | Requirement ID | Requirement Title | Current Status | Implementation Status | Test Coverage Status | Evidence | Commit / Tag | Linked Tests | Closure Confidence | Owner | Open Gap / Risk | Next Action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

Use the following status values only:

- OPEN
- PARTIAL
- CLOSED
- BLOCKED
- NOT APPLICABLE

Use the following implementation status values only:

- Not Started
- Documented Only
- Partially Implemented
- Implemented
- Implemented but Untested
- Tested and Verified

Use the following closure confidence values only:

- Low
- Medium
- High

Example entries:

| SRS Section | Requirement ID | Requirement Title | Current Status | Implementation Status | Test Coverage Status | Evidence | Commit / Tag | Linked Tests | Closure Confidence | Owner | Open Gap / Risk | Next Action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| §2.2 | NAV-02 | Orthophoto Matching Correction Accuracy | PARTIAL | Partially Implemented | Existing tests obsolete | orthophoto_matching_stub.py exists; LightGlue baseline complete | 96bf98a, f74bd82 | UT-NAV-02-A/B rewrite pending | Medium | Deputy 1 | SIL validation not complete | Rewrite UT-NAV-02-A/B and define OM SIL gate |
| §2.3 | NAV-03 | VIO Drift Rate Limit | PARTIAL | Implemented | Partially Tested | OpenVINS SIL evidence exists but no outdoor km-scale validation | a014997 | S-NEP tests | Medium | Deputy 1 | Outdoor validation pending | Define outdoor validation plan |
| §8.3 | EC-01 | Waypoint Upload Sequencing and OFFBOARD Continuity | PARTIAL | Partially Implemented | Partially Tested | SB-5 Phase A/B partial coverage | SB-05 gates | IT-PX4-01 | Medium | Deputy 1 | Longer continuity testing pending | Extend OFFBOARD continuity tests |
| §16 | EC-07 | Recovery Ownership Matrix | OPEN | Documented Only | Not Tested | SRS matrix exists | N/A | N/A | Low | Deputy 1 | Corridor violation ownership gap remains | Review ownership table and identify non-compliant rows |

Each week, push a short summary of:

- newly closed clauses
- newly identified gaps
- incorrectly assumed closures
- blocked items
- high-risk open requirements

into the Project Context file.

# Approved Two-Week Execution Order

## Week 1 — Thread A / Core SRS Compliance

1. SAL-3 sandbox scope definition only
   - Reference: AD-24
   - Scope only, no implementation

2. Synthetic terrain README caveat
   - Add explicit documentation on synthetic terrain limitations and structured-terrain bias

3. Recovery ownership verification
   - Reference: SRS §16 EC-07
   - Identify unresolved ownership rows and corridor-violation ownership gaps

4. Create initial SRS_COMPLIANCE_MATRIX.md baseline
   - Populate at least all Priority 1 clauses before any new coding

5. Read SB-5 Phase B retask cluster before any implementation
   - Reference: PLN-02
   - Confirm what already exists versus what is still open

6. Route invalidation + EW stale map + terrain ordering cluster
   - Reference: PLN-02 R-01 through R-06
   - Define only the delta from current implementation

7. Rollback behaviour gate
   - Reference: PLN-02 R-04 / R-06
   - Implement only after retask cluster is understood

8. Waypoint upload sequencing gate
   - Reference: EC-01 / SRS §8.3
   - Deputy 1 defines exact validation sequence before implementation

9. OFFBOARD continuity hardening
   - Reference: EC-01
   - Extend continuity duration and failure handling tests

10. Checkpoint retention and purge confirmation
    - Reference: EC-02
    - Confirm whether current implementation already satisfies the requirement

11. PX4 reboot recovery gap confirmation
    - Reference: EC-03 / Appendix D D7–D10
    - Identify remaining gaps only after reading current implementation

12. Full GNSS-denied retask integration test
    - Reference: IT-PLN-02
    - Run only after preceding items are complete and green

## Week 2 — Thread B / Validation Maturation

13. OI-07 outdoor VIO validation planning
    - Planning only
    - Define what constitutes valid outdoor evidence

14. Terrain-class dataset matrix
    - Define terrain classes, altitude bands, seasonality, and satellite GSD assumptions

15. High-altitude navigation behaviour scoping
    - Reference: AVP-03 / AVP-04
    - Define acceptable correction interval, confidence, and error budgets

16. Confidence-scored EO correction logic
    - Define behaviour for CAUTION-confidence corrections between ACCEPT and SUPPRESS

17. Long-endurance groundwork
    - Reference: RS-04
    - Memory, logging, checkpoint retention, and stability assumptions

18. Weekly update to SRS_COMPLIANCE_MATRIX.md and Project Context summary
    - Required at end of week

19. Orin deployment readiness review
    - Review only
    - No deployment unless all Week 1 gates are green and SIL remains stable

# Explicit Deferrals

The following remain deferred and are not part of this sprint:

- Live demo flows
- Real-time map underlay UI
- Vehicle-vs-vehicle comparison visualisation
- Demo narration logic
- Public-facing dashboard refinement
- Dual-stream demo architecture
- Polished HMI / operator console behaviour
- Final Orin telemetry dashboards
- Full HIL deployment beyond readiness review

# Mandatory Working Rules

1. Deputy 1 must read the SRS clause and current implementation before issuing any implementation prompt.
2. No item may be declared closed without:
   - code evidence
   - test evidence
   - commit reference
   - pass/fail status
3. If evidence is incomplete, the item must remain PARTIAL.
4. No requirement may be marked CLOSED if it is only documented but not tested.
5. No hardware deployment activity begins until Thread A is stable and SIL remains green.
6. The SRS_COMPLIANCE_MATRIX.md becomes the authoritative closure tracker for all future reviews.

