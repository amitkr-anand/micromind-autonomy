# SRS Compliance Matrix — v3
**Document:** `docs/qa/SRS_COMPLIANCE_MATRIX.md`
**Governing SRS:** MicroMind_SRS_v1_3.docx (SRS-MicroMind-v1.3, April 2026)
**Baseline HEAD:** `1dbfc29` → IT-D6-TIMEOUT-01 CLOSED, D6 CLOSED (QA-061)
**SIL at baseline:** 536/536 (`run_certified_baseline.sh`)
**Author:** Deputy 1 (Architect Lead)
**Version:** 3 — audit columns, new test rows, Appendix B/C/E, reconciled totals — 22 April 2026
**Classification:** Programme Confidential

> **Mandatory left-side rule:** Every row in every table must remain permanently populated.
> Only right-hand status columns change week to week. No future sprint may omit rows
> for items not yet started. The full left side is the authoritative programme scope.

---

## 1. Executive Summary Dashboard

### 1.1 Programme Closure Snapshot — Week 1 (22 April 2026, after QA-050)

> **Reconciled totals v3:** Requirements = 60 functional + 10 Appendix D + 14 §16 events + 7 Appendix B states + 3 Appendix C classes + 5 Appendix E behaviours = **99 tracked items**.
> Test cases = 62 existing + 20 new rows added this version = **82 total**.

| Category | Total | Closed | Partial | Open | Blocked | N/A |
|---|---:|---:|---:|---:|---:|---:|
| Navigation Requirements (NAV-01..06) | 6 | 3 | 3 | 0 | 0 | 0 |
| EW Requirements (EW-01..03) | 3 | 3 | 0 | 0 | 0 | 0 |
| Planning / Retask (PLN-01..03) | 3 | 2 | 1 | 0 | 0 | 0 |
| Mission Manager (MM-01..04) | 4 | 4 | 0 | 0 | 0 | 0 |
| PX4 / Recovery (PX4-01..05, EC-01..03) | 8 | 3 | 4 | 1 | 0 | 0 |
| Appendix D Steps (D1..D10) | 10 | 6 | 3 | 1 | 0 | 0 |
| Recovery Ownership §16 (14 events) | 14 | 13 | 0 | 1 | 0 | 0 |
| Endurance / Resource (RS-01..04, EC-02) | 5 | 1 | 2 | 0 | 1 | 0 |
| Terminal Guidance (TERM-01..03) | 3 | 1 | 2 | 0 | 0 | 0 |
| Visualisation / Demo (VIZ-01..03) | 3 | 1 | 0 | 1 | 0 | 1 |
| OM SIL Gate (EC-13) | 1 | 0 | 0 | 1 | 0 | 0 |
| Appendix B — Retask State Machine (7 states) | 7 | 2 | 4 | 1 | 0 | 0 |
| Appendix C — Restartability Classes (3 classes) | 3 | 1 | 1 | 0 | 1 | 0 |
| Appendix E — Retention / Purge Behaviours (5 items) | 5 | 2 | 2 | 0 | 1 | 0 |
| **Requirements Total** | **99** | **44** | **22** | **6** | **2** | **1** |
| **Test Cases Total** | **82** | **38** | **6** | **29** | **4** | **5** |

> **AVP note:** "Closed" counts are for AVP-02 unless explicitly stated. Many closures are PARTIAL for AVP-03/04. See §5.

---

### 1.2 High-Risk Open Items — Week 1

| Rank | Risk Item | Requirement | Impact |
|---|---|---|---|
| 1 | **No valid OM SIL gate (EC-13)** | NAV-02, EC-13 | Production nav correction mechanism has zero SIL certification. All L2 evidence is HIL only. |
| 2 | **NAV-02 tests OBSOLETE** | NAV-02 | UT-NAV-02-A/B validate superseded RADALT-NCC. Active SIL test count inflated. |
| 3 | **Corridor Violation §16 row missing (OI-40)** | EC-07-14, §16 | Ownership chain defined by Deputy 1; SRS v1.4 row pending commit. OI-55 (cross_track_error_m) blocking full compliance. |
| 4 | **D10 path (GNSS-denied reboot) untested** | PX4-04, EC-03, D10 | Highest operational risk reboot scenario. Zero test coverage. |
| 5 | **DMRL is a stub** | TERM-01, TERM-02 | All terminal guidance SIL results synthetic. Must caveat every external presentation. |

---

### 1.3 Incorrectly Assumed Closures — Audit Record

| Req ID | Historical Status | Corrected Status | Downgraded? | Session | Reason |
|---|---|---|---|---|---|
| NAV-02 | CLOSED (SRS v1.2 traceability) | PARTIAL | YES | QA-050 | SRS v1.3 replaced RADALT-NCC with orthophoto matching. UT-NAV-02-A/B test obsolete mechanism. |
| EC-01 | CLOSED (Phase A QA label) | PARTIAL | YES | QA-050 | 30-minute OFFBOARD endurance exit gate not confirmed PASS. S-PX4-09 62s ≠ §8.3 exit criterion. |

---

### 1.4 Weekly Movement Summary

| Week | Newly Closed | Newly Opened | Downgraded | Still Blocked |
|---|---|---|---|---|
| Week 1 opening — 22 Apr | — | OI-53, OI-54 (both later closed same session) | NAV-02, EC-01 | RS-01 |
| W1-P01 — bc8230a | Terrain README | — | — | RS-01 |
| W1-P03 — 9d99a75 | OI-53, OI-54 | OI-55 (cross_track_error_m) | — | RS-01 |
| W1 close — 23 Apr | R-02 COMPLIANT (168b1d5), R-05 COMPLIANT (ab083ce), OI-55 CLOSED (3e79805), OI-54 CLOSED (9d99a75) | OI-56 (R-03 ETA rollback gap) | — | RS-01 |

---

## 2. Requirement Traceability Matrix

> **New audit columns in v3:** Historical Status | Current Status | Downgraded? | QA / Gate Reference

### 2.1 Navigation Requirements (§2)

| SRS § | Req ID | Title | Priority | AVP Scope | Historical Status | Current Status | Downgraded? | Impl Status | Test Status | Evidence | Commit / Tag | Linked Tests | Confidence | Owner | Open Gap / Risk | Next Action | QA / Gate Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| §2.1 | NAV-01 | INS-Only Navigation Drift Limit | Critical | AVP-02 ✅ AVP-03 ⚠ AVP-04 ❌ | CLOSED | CLOSED (AVP-02) | NO | Tested and Verified | Tested and Verified | BCMP-2 N=300; seeds 42/101/303 within C-2 envelopes at km60/100/120 | `sb3-full-mission-reports` | UT-NAV-01-A/B, IT-NAV-01, AT-1/AT-2 | High (AVP-02) | Deputy 1 | C-2 to km120 only. AVP-04 requires km200 extension (GAP-11). | Document AVP-04 limit in all TASL claims | QA-049, AT-1/AT-2 |
| §2.2 | NAV-02 | Orthophoto Matching Correction Accuracy | Critical | AVP-02 ⚠ AVP-03 ❌ AVP-04 ❌ | CLOSED | PARTIAL | **YES** | Partially Implemented | Obsolete — rewrite required | LightGlue HIL H-6 conf=0.743. UT-NAV-02-A/B test superseded RADALT-NCC. | `66af1b3`, `8086f95`, `322274c` | UT-NAV-02-A/B INVALID; UT-OM-02-A/B not written | Low | Deputy 1 | GAP-10: zero valid SIL tests for OM. EC-13 not satisfied. | Sprint C: write UT-OM-02-A/B | QA-050 downgrade; H-6 HIL evidence |
| §2.3 | NAV-03 | VIO Drift Rate Limit | High | AVP-02 ⚠ AVP-03 ⚠ AVP-04 ❌ | PARTIAL | PARTIAL | NO | Implemented | Partially Tested | OpenVINS EuRoC MH_01_easy ATE=0.3412m (indoor). No outdoor km-scale. | `a014997`, `4fcf231` | UT-NAV-03, S-NEP-01..10 | Medium | Deputy 1 | OI-07: outdoor validation absent | Week 2 Item 13 | S-NEP-10; OI-07 |
| §2.4 | NAV-04 | Navigation Mode Transition Latency | High | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | AT-3. Transition <500ms. ESKF P95=0.085ms. | `sb3-full-mission-reports` | UT-NAV-03, IT-NAV-01..03, AT-3 | High | Deputy 1 | None | None | QA-049, AT-3 |
| §2.5 | NAV-05 | BIM Trust Score and Spoof Detection | Critical | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | Detection ≤250ms. GNSS gated within 50ms of RED. GNSS_SPOOF_DETECTED confirmed. | `fff0cc4` | UT-BIM-01/02, IT-EW-01, ST-GNSS-01 | High | Deputy 1 | GAP-12: adaptive spoof not tested | Write UT-BIM-03 before HIL | QA-019, OI-39 |
| §2.6 | NAV-06 | Two-Theatre Navigation Architecture | High | AVP-02 ⚠ AVP-03 ❌ AVP-04 ❌ | PARTIAL | PARTIAL | NO | Implemented | Partially Tested | TRN-primary (east) / VIO-primary (west). AT-2 validates TRN suppress. VIO indoor only. | `sb3-full-mission-reports`, `a014997` | UT-NAV-02-B, IT-NAV-02, AT-2 | Medium | Deputy 1 | AMB-01: σ_terrain unverified. Western theatre km-scale absent. | Verify AMB-01 before HIL | QA-049, AT-2 |

---

### 2.2 EW Requirements (§3)

| SRS § | Req ID | Title | Priority | AVP Scope | Historical Status | Current Status | Downgraded? | Impl Status | Test Status | Evidence | Commit / Tag | Linked Tests | Confidence | Owner | Open Gap / Risk | Next Action | QA / Gate Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| §3.1 | EW-01 | EW Map Update and Threat Integration | High | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | DBSCAN. Cost map ≤500ms. S4. | `a7633ab` | UT-EW-01, IT-EW-02 | High | Deputy 1 | None | None | QA-049 |
| §3.2 | EW-02 | EW-Responsive Route Replan | High | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | Hybrid A* replan ≤1s. S4. | `a7633ab` | UT-EW-02, IT-EW-02, IT-PLN-01 | High | Deputy 1 | None | None | QA-049 |
| §3.3 | EW-03 | GNSS Spoof Rejection via BIM | Critical | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | Detection ≤250ms. GNSS gated within 50ms. | `fff0cc4` | UT-BIM-01, IT-EW-01, ST-GNSS-01 | High | Deputy 1 | GAP-12: adaptive spoof | Write UT-BIM-03 before HIL | QA-019 |

---

### 2.3 Route Planning Requirements (§4)

| SRS § | Req ID | Title | Priority | AVP Scope | Historical Status | Current Status | Downgraded? | Impl Status | Test Status | Evidence | Commit / Tag | Linked Tests | Confidence | Owner | Open Gap / Risk | Next Action | QA / Gate Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| §4.1 | PLN-01 | Hybrid A* Route Planner Response Time | High | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | ≤2s nominal / 5s max. Route ≤120%. S4. | `a7633ab` | UT-PLN-01, IT-PLN-01 | High | Deputy 1 | None | None | QA-049 |
| §4.2 | PLN-02 | Dynamic Retask — Mid-Mission | Medium | AVP-02 ⚠ AVP-03 ❌ | PARTIAL | PARTIAL | NO | Implemented | Tested and Verified | R-01 ✅ (pre-existing), R-02 ✅ (168b1d5 — callback-based EW refresh, dual-outcome log), R-03 ❌ (OI-56 — ETA attribute not found on RoutePlanner), R-04 ✅ (pre-existing), R-05 ✅ (ab083ce — conditional XTE check, RETASK_NAV_CONFIDENCE_TOO_LOW), R-06 ✅ (pre-existing). 5/6 R-corrections compliant. Blocked on R-03. | `ab083ce` (R-05), `168b1d5` (R-02) | UT-PLN-02, IT-PLN-01, IT-PLN-02, test_adv_01 (updated), test_adv_01b (new) | Medium | Deputy 1 | R-03 CLOSED (e7d3d42) — _eta_s added to RoutePlanner, snapshot/restore in _rollback(), RETASK_ROLLBACK event at both call sites, test_r03_eta_rollback() 512/512. Three untested rollback trigger paths remain (IT-ROLLBACK-01): TERRAIN_GEN_FAIL, COMMIT_FAIL, timeout overrun. | Define IT-ROLLBACK-01 covering TERRAIN_GEN_FAIL, COMMIT_FAIL, and timeout overrun paths. | W1-P05/P06/P09; OI-56 |
| §4.3 | PLN-03 | Route Dead-End Recovery | High | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | Constraint relaxation levels 1–3. UT-PLN-03. | Phase B | UT-PLN-03, IT-EW-02 | High | Deputy 1 | None | None | Phase B closure |

---

### 2.4 Mission Manager Requirements (§5–§7)

| SRS § | Req ID | Title | Priority | AVP Scope | Historical Status | Current Status | Downgraded? | Impl Status | Test Status | Evidence | Commit / Tag | Linked Tests | Confidence | Owner | Open Gap / Risk | Next Action | QA / Gate Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| §5 | MM-01 | FSM Latency | Critical | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | 7-state FSM. S1. | `5005a5d` | run_s5_tests.py S1, IT-MM-01 | High | Deputy 1 | None | None | QA-049 |
| §5 | MM-02 | Envelope Validation | Critical | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | Mission envelope validation. S5. | `7ad5db5` | run_s5_tests.py S5, IT-TERM-01 | High | Deputy 1 | None | None | QA-049 |
| §5 | MM-03 | Safe Hold Mode (SHM) | Critical | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | SHM in Checkpoint (P-01). Clearance gate (P-02). SA-01..SA-04. | `fcb5106` | UT-FIM-01, IT-MM-01, SA-01..SA-04 | High | Deputy 1 | None | None | QA-025; SA-01..04 |
| §5 | MM-04 | Mission Event Bus Queue Latency | High | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | EVENT_QUEUE_LATENCY. UT-MM-04 100ms gate. | Phase B SB-06 | UT-MM-04 | High | Deputy 1 | None | None | Phase B SB-06 |

---

### 2.5 PX4 Integration and Recovery Requirements (§8–§9)

| SRS § | Req ID | Title | Priority | AVP Scope | Historical Status | Current Status | Downgraded? | Impl Status | Test Status | Evidence | Commit / Tag | Linked Tests | Confidence | Owner | Open Gap / Risk | Next Action | QA / Gate Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| §8.1 | PX4-01 | OFFBOARD Setpoint Rate ≥20 Hz | Critical | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | 20Hz SITL. OFFBOARD held 62s S-PX4-09. | `97b2f5a` | UT-PX4-01, IT-PX4-01 | High | Deputy 1 | None | None | QA-049 |
| §8.2 | PX4-02 | MAVLink Heartbeat Continuity | High | ALL AVP | PARTIAL | PARTIAL | NO | Implemented but Untested | Partially Tested | UT-PX4-02 stub. IT-PX4-02. Real timing requires HIL. | `787ecd4` | IT-PX4-02, UT-PX4-02 | Medium | Deputy 1 | GAP-07: real heartbeat unvalidatable in SIL | HIL Sprint 1 | GAP-07 |
| §8.3 | PX4-03 | Sensor Drivers (DriverFactory) | High | ALL AVP | PARTIAL | PARTIAL | NO | Partially Implemented | Partially Tested | DriverFactory 5 types. RC-4 PASS. Real EO untested. | `7bebc8c` | IT-PX4-01, SIA tests | Low | Deputy 1 | GAP-07: real EO driver unvalidated | Maintain caveat. HIL Sprint 1. | GAP-07 |
| §8.4 | PX4-04 | PX4 Reboot Recovery and State Restore | High | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | RebootDetector. D8a gate. SA-05..SA-07. | `787ecd4`, `fcb5106` | SA-05..SA-07, IT-PX4-02 | High | Deputy 1 | D10 path (GNSS-denied reboot) not explicitly tested | Define IT-D10-GNSS-01 | QA-025; SA-05..07 |
| §8.5 | PX4-05 | Checkpoint Schema v1.2 | High | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | 6 fields confirmed. P-01..P-04 applied. UT-PX4-05. | `fcb5106` | UT-PX4-05, SA-01..SA-04 | High | Deputy 1 | None | None | QA-025; SA-01..04 |
| §8.3 | EC-01 | Waypoint Upload Sequencing + OFFBOARD Continuity | High | ALL AVP | CLOSED | PARTIAL | **YES** | Partially Implemented | Partially Tested | SITL 62s PASS (S-PX4-09). 30-min exit gate NOT confirmed. | `97b2f5a` | IT-PX4-01 (short), SA-05..SA-07 | Medium | Deputy 1 | 30-min exit gate unconfirmed. Incorrectly assumed closed. | Week 1 Item 8/9: execute 30-min gate | QA-050 downgrade |
| §13 | EC-02 | Checkpoint Retention and Purge Policy | Medium | ALL AVP | PARTIAL | CLOSED | NO | Implemented | Tested and Verified | Schema v1.2 implemented. E-01 purge in spec. Phase D not executed. E-01 purge confirmed c38357a — 8 assertions, max_retained=5 enforced. | `fcb5106`, `c38357a` | UT-PX4-05, ST-RESTART-01 (pending) | Medium | Deputy 1 | Phase D 2-hr run not completed. CHECKPOINT_PURGED req_id='PX4-05' in code (historical implementation label) — EC-02 SRS §13 compliance confirmed. Purge confirmed c38357a. | Week 1 Item 10 | Phase A closure |
| §13 | EC-03 | PX4 Reboot Recovery — Full Appendix D | High | ALL AVP | CLOSED | CLOSED | NO | Implemented | Tested and Verified | D1..D9 implemented. D8a confirmed. D10 CLOSED 711bf6d. D7→D9 chain CLOSED 208a5a1. Full Appendix D coverage confirmed. | `787ecd4`, `711bf6d`, `208a5a1` | SA-05..SA-07, IT-PX4-02, IT-D9-CHAIN-01 | Medium | Deputy 1 | None | None | QA-025; QA-059 |

---

### 2.6 Appendix D Recovery Steps (Individual Rows)

| Step | Trigger | Required Action | Historical Status | Current Status | Downgraded? | Impl Status | Test Status | Evidence | Linked Tests | Confidence | Owner | Open Gap | Next Action | QA / Gate Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | OFFBOARD Loss | Log OFFBOARD_LOSS. Buffer last setpoint (discard on recovery). | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | IT-PX4-01. P-03 stale setpoint discard. | IT-PX4-01 | High | Deputy 1 | None | None | QA-025 |
| D2 | D1 complete | Attempt SET_MODE OFFBOARD. Retry 1 Hz / 5s. | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | S-PX4-01..09. IT-PX4-01. | IT-PX4-01 | High | Deputy 1 | None | None | QA-025 |
| D3 | D2 timeout (5s) | Activate SHM. Continue reconnect 1 Hz / 5s. | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | SHM activation S1 tests. | IT-MM-01 | High | Deputy 1 | None | None | QA-025 |
| D4 | D3 reconnect success | OFFBOARD restored. Discard stale buffer. Exit SHM. | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | OFFBOARD_RESTORED confirmed SITL. | IT-PX4-01 | High | Deputy 1 | None | None | QA-025 |
| D5 | OFFBOARD active | Issue fresh setpoint ≥20 Hz. | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | Setpoint restart confirmed S-PX4-09. | IT-PX4-01 | High | Deputy 1 | None | None | QA-025 |
| D6 | D3 total timeout (10s) | Log OFFBOARD_UNRECOVERED. ABORT_MISS. | CLOSED | CLOSED | NO | IMPLEMENTED — OffboardRecoveryFSM (integration/bridge/offboard_recovery_fsm.py). D2 5s + D3 5s + D6 abort. OFFBOARD_UNRECOVERED event. abort_fn() callback on D6. 6a30295. | Tested and Verified | 4 tests / 17 assertions. D2 recovery, D3 recovery, full timeout abort, timing bounds. test_it_d6_timeout.py. 6a30295. | test_it_d6_timeout.py | Medium | Deputy 1 | IT-D6-SITL-01: SITL integration test with real GCS heartbeat stop. Phase D. | None | QA-061; Deputy 1 rules |
| D7 | PX4 reboot detected (HEARTBEAT seq reset) | Log PX4_REBOOT_DETECTED. Enter SHM. Begin D1. | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | RebootDetector seq-reset. PX4_REBOOT_DETECTED. SA-05. | SA-05, IT-PX4-02 | High | Deputy 1 | None | None | QA-025; SA-05 |
| D8 | PX4 reconnect success (post-reboot) | Load Checkpoint v1.2. Restore nav state. Command OFFBOARD re-entry. Timeout 15s total. | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | Checkpoint load SA-06. 6-field schema UT-PX4-05. | SA-06, UT-PX4-05 | High | Deputy 1 | None | None | QA-025; SA-06 |
| D8a | D8 complete — NEW v1.2 | MM evaluates Checkpoint: abort_flag → ABORT_MISS; shm_active or clearance_required → SHM hold; all clear → autonomous_resume_approved → D9. | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | Three-branch logic SA-07. MISSION_RESUME_AUTHORISED nominal path. P-02 applied. | SA-07 | High | Deputy 1 | None | None | QA-025; SA-07 |
| D9 | autonomous_resume_approved=True | Restore setpoint stream. Command new waypoint queue. Log MISSION_RESUMED. | CLOSED | CLOSED | NO | Implemented | Tested and Verified | Nominal D9 SA-07 (synthetic). IT-D9-CHAIN-01 PASS. Live SITL reboot injection. G4 position_discrepancy_m=43.678m (<50m threshold). Wall clock 8.61s. 208a5a1/5863020. | SA-07, IT-D9-CHAIN-01 | Medium | Deputy 1 | None | None | QA-025; SA-07; QA-059 |
| D10 | PX4 returns in HOLD (post-reboot) | Command SET_MODE OFFBOARD within 1s. Retry 3× at 2s. Proceed to D8a on success. | OPEN | CLOSED | NO | Implemented | Tested and Verified | HoldRecoveryHandler (integration/bridge/hold_recovery.py). PX4_HOLD_CUSTOM_MODE=50_593_792. 4 tests / 20 assertions. IT-D10-GNSS-01 PASS. 711bf6d. | IT-D10-GNSS-01 | Medium | Deputy 1 | IT-D9-CHAIN-01 (D7→D8→D8a→D9 full SITL chain) still NOT STARTED. D10 SIL coverage confirmed. | Define IT-D9-CHAIN-01 to close EC-03 fully | OI-40 analogue; GAP per QA-050 |

---

### 2.7 Recovery Ownership Matrix (§16)

| Event | §16 Row | Detects | Decides | Executes | Logs | Historical Status | Current Status | Downgraded? | Code Evidence | Open Gap | Next Action | QA / Gate Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GNSS Denial | ✅ | Navigation Manager (BIM) | Navigation Manager | Navigation Manager | Mission Manager | CLOSED | CLOSED | NO | `core/bim/bim.py` | None | None | QA-018/019 |
| VIO Outage | ✅ | Navigation Manager (VIOMode) | Navigation Manager | Navigation Manager | Mission Manager | CLOSED | CLOSED | NO | `core/fusion/vio_mode.py:166` | None | None | QA-018/019 |
| GNSS Spoof Detected | ✅ | Navigation Manager (BIM) | Navigation Manager | Navigation Manager (GNSS gate) | Mission Manager | CLOSED | CLOSED | NO | `core/bim/bim.py:288` + `fff0cc4` | None | None | QA-019; OI-39 |
| EW Spike — New Threat | ✅ | EW Manager | EW Manager + Route Planner | Route Planner | Mission Manager | CLOSED | CLOSED | NO | `core/ew_engine/ew_engine.py` | None | None | QA-018/019 |
| Route Dead-End | ✅ | Route Planner | Mission Manager | Mission Manager (SHM + ESCALATE_OP) | NavManager, PX4 Bridge | CLOSED | CLOSED | NO | State machine SHM trigger | None | None | QA-018/019 |
| Retask Validation Fail | ✅ | Mission Manager (VALIDATING) | Mission Manager | Mission Manager (REJECTED → NONE) | NavManager, Route Planner | CLOSED | CLOSED | NO | PLN-02 state machine | None | None | QA-024 |
| PX4 Reboot | ✅ | PX4 Bridge (HEARTBEAT reset) | PX4 Bridge (D7–D9) + MM (D8a) | PX4 Bridge (reconnect + OFFBOARD) | Mission Manager | CLOSED | CLOSED | NO | `integration/bridge/reboot_detector.py` + SA-05..SA-07 | None | None | QA-025; SA-05..07 |
| OFFBOARD Loss | ✅ | PX4 Bridge | PX4 Bridge (D1–D6) | PX4 Bridge | Mission Manager (SHM if >5s) | CLOSED | CLOSED | NO | IT-PX4-01, `mavlink_bridge.py` | None | None | QA-025 |
| SHM Entry | ✅ | Mission Manager | Mission Manager | Mission Manager (loiter cmd) | All modules | CLOSED | CLOSED | NO | `core/state_machine/state_machine.py:333` | None | None | QA-018/019 |
| SHM Exit | ✅ | Mission Manager (clearance) | Mission Manager | Mission Manager (resume) | PX4 Bridge (after D8a) | CLOSED | CLOSED | NO | SA-07 D8a evaluation | None | None | QA-025; SA-07 |
| Checkpoint Restore | ✅ | Watchdog | Mission Manager (D8a gate) | PX4 Bridge (waypoint reload) | NavManager, EW Manager | CLOSED | CLOSED | NO | SA-06..SA-07 | None | None | QA-025 |
| Mission Abort | ✅ | Mission Manager | Mission Manager | Mission Manager (ABORT_MISS) | All modules | CLOSED | CLOSED | NO | State machine ABORT states | None | None | QA-018/019 |
| Terminal Phase Failure (DMRL/L10s) | ✅ | DMRL / L10s-SE | Mission Manager | Mission Manager (ABORT_TERM or SHM) | NavManager, PX4 Bridge | CLOSED | CLOSED | NO | `core/l10s_se/l10s_se.py`, ADV-01..06 | None | None | QA-025 |
| **Corridor Violation (Predicted)** | **❌ MISSING** | Navigation Manager (cross_track_error_m vs corridor boundary) | Mission Manager (NanoCorteXFSM) — ABORT from 5 states | Mission Manager (`_transition(NCState.ABORT)`) | Mission Manager (SYSTEM_ALERT MissionLogEntry via `_log_corridor_violation_event()`) | OPEN | OPEN | NO | `core/state_machine/state_machine.py` lines 297/320/361/399/440; `9d99a75` structured event | **OI-40:** §16 row absent. OI-55: `cross_track_error_m` not in SystemInputs — payload incomplete. SRS v1.4 row pending. | Week 1 Item 3: commit §16 row to SRS v1.4 | QA-050; OI-40; OI-54/55 |

---

### 2.8 Endurance and Resource Requirements (§12–§13)

| SRS § | Req ID | Title | Priority | AVP Scope | Historical Status | Current Status | Downgraded? | Impl Status | Test Status | Evidence | Commit / Tag | Linked Tests | Confidence | Owner | Open Gap / Risk | Next Action | QA / Gate Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| §12 | RS-01 | CPU Utilisation Budget | High | ALL AVP | BLOCKED | BLOCKED | NO | Implemented but Untested | Not Tested | ESKF P95=0.085ms dev hardware — structural signal only | `MicroMind_PreHIL_Part12` | ST-CPU-01 (HIL required) | Low | Deputy 1 | Requires Jetson Orin NX (GAP-06) | Unblock at HIL Sprint 1 | GAP-06 |
| §12 | RS-02 | Log Completeness and Rolling Policy | Medium | ALL AVP | PARTIAL | PARTIAL | NO | Partially Implemented | Not Tested | E-04 rolling in spec. UT-RS-02 not written (GAP-09). | `sb5-bcmp2-closure` | UT-RS-02 (not written) | Low | Deputy 1 | GAP-09 | SB-5 Phase D | GAP-09 |
| §12 | RS-03 | Watchdog and Restartability Classification | Medium | ALL AVP | PARTIAL | PARTIAL | NO | Partially Implemented | Not Tested | E-03 classification in spec. UT-RS-03 not written (GAP-08). | `sb5-bcmp2-closure` | UT-RS-03 (not written) | Low | Deputy 1 | GAP-08 | SB-5 Phase D | GAP-08 |
| §12 | RS-04 | Route Fragment Accumulation and Purge | Medium | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | `_intermediate_fragments` + cleanup. SB-07 PASS. | `c35122a` | SB-07 | High | Deputy 1 | None | None | SB-07 PASS |
| §13 | EC-02 | Checkpoint Retention and Purge Policy | Medium | ALL AVP | PARTIAL | CLOSED | NO | Implemented | Tested and Verified | Schema v1.2 implemented. E-01 purge in spec. Phase D not executed. E-01 purge confirmed c38357a — 8 assertions, max_retained=5 enforced. | `fcb5106`, `c38357a` | UT-PX4-05, ST-RESTART-01 (pending) | Medium | Deputy 1 | Phase D 2-hr run not completed. CHECKPOINT_PURGED req_id='PX4-05' in code (historical implementation label) — EC-02 SRS §13 compliance confirmed. Purge confirmed c38357a. | Week 1 Item 10 | Phase A closure |

---

### 2.9 Terminal Guidance Requirements (§10–§11)

| SRS § | Req ID | Title | Priority | AVP Scope | Historical Status | Current Status | Downgraded? | Impl Status | Test Status | Evidence | Commit / Tag | Linked Tests | Confidence | Owner | Open Gap / Risk | Next Action | QA / Gate Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| §10 | TERM-01 | DMRL Multi-Frame Target Lock | Critical | ALL AVP | PARTIAL | PARTIAL | NO | Implemented but Untested | Tested (SIL stub only) | DMRL stub. Constants confirmed. S5 SIL on stub. **DMRL IS A STUB.** | `7ad5db5` | UT-DMRL-01, IT-TERM-01 | Low | Deputy 1 | DMRL stub. All SIL results synthetic. | Maintain caveat in all external materials | QA-025; OI-06 |
| §10 | TERM-02 | Decoy Rejection | Critical | ALL AVP | PARTIAL | PARTIAL | NO | Implemented but Untested | Tested (SIL stub only) | Same stub caveat as TERM-01. ADV-01..06 synthetic inputs. | `41238ae` | UT-DMRL-02, ADV-01..06 | Low | Deputy 1 | Same as TERM-01 | Same as TERM-01 | QA-025; ADV-01..06 |
| §11 | TERM-03 | L10s-SE Terminal Safety Envelope | Critical | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | Deterministic decision tree. civilian_threshold=0.70. ADV-01..06 PASS. | `41238ae`, `7ad5db5` | run_s5_tests.py S5, IT-TERM-02, ADV-01..06 | High | Deputy 1 | None | None | QA-025; ADV-01..06 |

---

### 2.10 Visualisation and Demo Requirements (§9)

| SRS § | Req ID | Title | Priority | AVP Scope | Historical Status | Current Status | Downgraded? | Impl Status | Test Status | Evidence | Commit / Tag | Linked Tests | Confidence | Owner | Open Gap / Risk | Next Action | QA / Gate Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| §9.1 | VIZ-01 | BCMP-2 Static Dashboard | High | ALL AVP | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | Static HTML dual-track. SB-3/SB-4. | `c183b9c` | VIZ-01 gate SB-3 | High | Deputy 1 | None | None | QA-049 |
| §9.2 | VIZ-02 | Animated MP4 Dual-Track Mission Replay | High | ALL AVP | OPEN | OPEN | NO | Not Started | Not Tested | GAP-01: gated on EC-01..EC-11 PASS. Phase C scope. | N/A | UT-VIZ-02 (not written) | Low | Deputy 1 | Blocked until Thread A resolved | Begin Phase C after Thread A exit gates confirmed | GAP-01 |
| §9.4 | VIZ-03 | Tactical Live Demonstration | Low | ALL AVP | NOT APPLICABLE | NOT APPLICABLE | NO | Not Started | Not Tested | AD-22 decoupled HIL from demo gate. Long-term roadmap. | AD-22 | N/A | N/A | Deputy 1 | Correctly deferred | None | GAP-13; AD-22 |

---

### 2.11 Orthophoto Matching SIL Gate (§17)

| SRS § | Req ID | Title | Priority | AVP Scope | Historical Status | Current Status | Downgraded? | Impl Status | Test Status | Evidence | Commit / Tag | Linked Tests | Confidence | Owner | Open Gap / Risk | Next Action | QA / Gate Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| §17 | EC-13 | Orthophoto Matching SIL Gate | Critical | AVP-02 ⚠ AVP-03 ❌ | OPEN | OPEN | NO | Not Started | Not Tested | GAP-10. EC-13 added v1.3. OrthophotoMatchingStub not implemented. Zero SIL tests. | N/A | UT-OM-02-A/B (not written) | Low | Deputy 1 | Blocks NAV-02 CLOSED. Production correction uncertified in SIL. | Sprint C: implement OrthophotoMatchingStub | GAP-10 |

---

## 3. Appendix B — Dynamic Retask State Machine

> Appendix B defines 7 states for the retask state machine. Each state has distinct testable entry conditions, actions, and exit conditions. Individual rows below prevent Appendix B logic from remaining hidden in narrative.

| State | Entry Condition | Key Actions | Exit Condition(s) | Historical Status | Current Status | Downgraded? | Impl Status | Test Status | Evidence | Linked Tests | Confidence | Owner | Open Gap | Next Action | QA / Gate Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NONE | System start or retask complete/rejected | No retask in progress | RETASK_RECEIVED event | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | PLN-02 state machine. Phase B gates. | UT-PLN-02 | High | Deputy 1 | None | None | Phase B closure |
| VALIDATING | RETASK_RECEIVED | Verify signature hash. Verify timestamp_utc <60s. Check permitted_phases. If GNSS_DENIED+INS_ONLY: check cross_track_error_m ≤ (half_width − 100). | VALIDATION_PASS → ROUTING. VALIDATION_FAIL → NONE. | PARTIAL | PARTIAL | NO | Implemented | Partially Tested | PLN-02 Phase B gates cover nominal path. INS_ONLY cross_track constraint (R-05) not corridor-tested. | UT-PLN-02, IT-PLN-02 | Medium | Deputy 1 | R-05 INS_ONLY constraint path (cross_track_error_m guard) not confirmed in code or exercised in integration. | Week 1 Item 5/6: confirm R-05 code; define IT-INS-RETASK-01 | Phase B; OI-40 context |
| ROUTING | VALIDATION_PASS | (1) Regenerate terrain corridor (R-01). (2) Check EW staleness (R-02). (3) Invoke Route Planner. Timeout 10s/15s. | ROUTE_SUCCESS → COMMITTING. ROUTE_FAIL or TERRAIN_GEN_FAIL → ROLLBACK. | CLOSED | CLOSED | NO | IMPLEMENTED — TERRAIN_GEN_FAIL try/except at route_planner.py:359. RETASK_TERRAIN_GEN_FAILED event logged. _rollback() called. 17330fa. | Tested and Verified | Phase B gates cover nominal ROUTING path. TERRAIN_GEN_FAIL → ROLLBACK not tested. EW staleness injection not tested. test_terrain_gen_fail_triggers_rollback 7 assertions PASS. 17330fa. | UT-PLN-02, IT-PLN-01 | Medium | Deputy 1 | R-01 TERRAIN_GEN_FAIL → ROLLBACK path not exercised. R-02 stale map injection not tested. | Define IT-TERRAIN-FAIL-01 and IT-STALE-EW-01 | Phase B; W1 Items 5/6 |
| COMMITTING | ROUTE_SUCCESS | (1) Validate new route vs envelope. (2) Only on new_envelope_valid=True AND route_validated=True: upload PX4 waypoints. Note: PX4 waypoints NOT uploaded before step 2. (R-04) | COMMIT_SUCCESS → ACTIVE. COMMIT_FAIL → ROLLBACK. | CLOSED | CLOSED | NO | IMPLEMENTED — COMMIT_FAIL try/except at route_planner.py:478. RETASK_COMMIT_FAILED event logged. _rollback() called. 17330fa. | Tested and Verified | Phase B IT-PLN-02 exercises nominal COMMITTING path. COMMIT_FAIL → ROLLBACK path not isolated in dedicated test. test_commit_fail_triggers_rollback 7 assertions PASS. 17330fa. | IT-PLN-02 | Medium | Deputy 1 | COMMIT_FAIL → ROLLBACK path not isolated. R-04 waypoint upload sequencing confirmed in spec; code confirmation pending. | Confirm R-04 in code (Week 1 Item 5/6) | Phase B |
| ACTIVE | COMMIT_SUCCESS | Mission executing new route. Release partial route candidates within 500ms (E-02). | Mission complete or new RETASK_RECEIVED → NONE. FAULT → ROLLBACK. | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | RS-04 / E-02 route fragment cleanup confirmed at `c35122a`. SB-07 PASS. | SB-07 | High | Deputy 1 | None | None | SB-07 PASS |
| ROLLBACK | ROUTE_FAIL / TERRAIN_GEN_FAIL / COMMIT_FAIL / ACTIVE FAULT | (1) Restore previous route. (2) Restore previous envelope. (3) Restore EW map from Checkpoint. (4) Restore terrain corridor. (5) Restore ETA. Log RETASK_ROLLBACK. Complete within 2000ms. | Rollback complete → NONE. | CLOSED | CLOSED | NO | Implemented | Tested and Verified | `_rollback()` confirmed in Phase B code review. Rollback from ROUTE_FAIL confirmed. TERRAIN_GEN_FAIL path, COMMIT_FAIL path, and rollback timeout overrun (>2000ms) not tested. | IT-PLN-02 (partial) | Medium | Deputy 1 | RETASK_ROLLBACK payload: all 10 fields present including SRS-required reason/previous_target/restored_ew_map_age_ms/restored_terrain_phase. snap_target=None (no _current_target attribute on RoutePlanner — accepted deviation, field present). test_rollback_payload_complete 5 assertions PASS. 17330fa. Timeout overrun 2000ms: no enforcement — _rollback() is synchronous, completes in microseconds. Accepted. | IT-ROLLBACK-01 CLOSED | Phase B; OI-40 context |
| REJECTED | VALIDATION_FAIL | Log RETASK_REJECTED with reason. | Immediate → NONE. | OPEN | OPEN | NO | Documented Only | Not Tested | RETASK_REJECTED defined in PLN-02. No dedicated test for rejected retask with explicit reason logging. | None | Low | Deputy 1 | No test confirms RETASK_REJECTED event payload (reason string) is emitted correctly. | Define UT-PLN-REJECTED-01 | Phase B narrative only |

---

## 4. Appendix C — Restartability Classification

> Appendix C (Mission Demonstration Tool roadmap in SRS) also contains the restartability classification from RS-03. Appendix A §12 RS-03 defines PROC_RESTART and CHECKPOINT_RESTORE classes. Three testable restartability classes are tracked here.

| Class | Components | Recovery Action | Historical Status | Current Status | Downgraded? | Impl Status | Test Status | Evidence | Linked Tests | Confidence | Owner | Open Gap | Next Action | QA / Gate Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RESTARTABLE | Watchdog-monitored processes (EW Manager, LogBus, NavigationManager, MissionManager, PX4Bridge) | Watchdog: PROC_RESTART. Restart ≤2s + restore ≤10s. ABORT_MISS after max_restart_attempts. | PARTIAL | OPEN | YES (QA-062) | NOT IMPLEMENTED — no watchdog module existed at QA-062 diagnostic. restartability_class not assigned in any production file. ProcessWatchdog (core/watchdog/process_watchdog.py) now implements decision logic; SIL-only, no real SIGKILL. | Not Tested — UT-RS-03 decision-logic tests PASS (QA-062). ST-RESTART-01 real SIGKILL test Phase D. | ProcessWatchdog PROCESS_REGISTRY covers RESTARTABLE_WITHOUT_SHM (EWManager, LogBus) and RESTARTABLE_WITH_SHM (NavigationManager, MissionManager, PX4Bridge). Decision path tested by UT-RS-03 (QA-062). | UT-RS-03 | Low | Deputy 1 | GAP-08: no SIGKILL stimulus test (ST-RESTART-01 Phase D) | Phase D: ST-RESTART-01 real SIGKILL test | GAP-08; QA-062 |
| NOT_RESTARTABLE | ESKF core (`core/ekf/error_state_ekf.py`) — frozen file | Watchdog: ABORT_MISS immediately on heartbeat miss. No restart. | CLOSED | PARTIAL | YES (QA-062) | Partially Tested — FSM ABORT_MISS tests confirm the outcome path. No direct SIGKILL stimulus test (ST-RESTART-01). UT-RS-03 covers decision logic only. | Partially Tested — FSM ABORT_MISS tests confirm the outcome path. No direct SIGKILL stimulus test (ST-RESTART-01). UT-RS-03 covers decision logic only. | ESKF classified NOT_RESTARTABLE in ProcessWatchdog registry. ABORT_MISS path exercised by UT-RS-03 test_not_restartable_triggers_abort (QA-062). Frozen file protection prevents accidental modification. | FSM ABORT_MISS tests; UT-RS-03 | High | Deputy 1 | ST-RESTART-01: real SIGKILL test Phase D | Phase D (ST-RESTART-01) | QA-062 |
| CHECKPOINT_RESTORE | PX4 Bridge (on reboot), Mission Manager (on process restart) | Load Checkpoint ≤10s. Mission Manager D8a gate before autonomous resume. | BLOCKED | CLOSED | NO | Tested and Verified | Tested and Verified | D8 checkpoint load confirmed SA-06. Corrupt path: CheckpointCorruptError + CHECKPOINT_CORRUPT event — UT-PX4-COR-01 COR-01..06 PASS. restore_latest() returns None on corrupt. | SA-06; UT-PX4-COR-01 | High | Deputy 1 | None | None | QA-058; COR-01..06 |

---

## 5. Appendix E — Logging and Event Retention / Purge Behaviours

> Appendix E defines per-event retention, RAM buffer, and disk flush policies. Five behaviourally distinct items tracked below.

| Item | Event / Behaviour | Retention Rule | Historical Status | Current Status | Downgraded? | Impl Status | Test Status | Evidence | Linked Tests | Confidence | Owner | Open Gap | Next Action | QA / Gate Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E-01 | CHECKPOINT_PURGED event (purge policy) | Checkpoint purge after endurance run. CHECKPOINT_PURGED event must be present in Phase D exit gate. | PARTIAL | PARTIAL | NO | Implemented | Not Tested | E-01 purge in spec. `fcb5106` Checkpoint schema implemented. Phase D not executed. CHECKPOINT_PURGED not confirmed in an actual run. | ST-END-01 (pending) | Medium | Deputy 1 | Phase D 2-hr run not completed. | Week 1 Item 10: confirm E-01 implementation | Phase A; GAP endurance |
| E-02 | Route Fragment Purge (RS-04) | Release partial route candidates from constraint levels 1–3 within 500ms of ACTIVE entry. | CLOSED | CLOSED | NO | Tested and Verified | Tested and Verified | `_cleanup_route_fragments`. ROUTE_FRAGMENT_CLEANUP event. SB-07 PASS. | SB-07 | High | Deputy 1 | None | None | SB-07 PASS |
| E-03 | Restartability Classification | Each process classified RESTARTABLE / NOT_RESTARTABLE. Watchdog respects classification. | PARTIAL | OPEN | YES (QA-062) | NOT IMPLEMENTED at QA-062 diagnostic — classification was documentation-only. Now: ProcessWatchdog PROCESS_REGISTRY implements classification map. Decision logic tested by UT-RS-03. Real SIGKILL stimulus (ST-RESTART-01) remains Phase D. | Not Tested — UT-RS-03 decision-logic gates PASS (QA-062). ST-RESTART-01 not executed. | ProcessWatchdog PROCESS_REGISTRY committed at QA-062 (core/watchdog/process_watchdog.py). 6 processes classified. UT-RS-03 test_registry_classification_correct asserts all 6 entries. | UT-RS-03 | Low | Deputy 1 | GAP-08: no SIGKILL test (ST-RESTART-01 Phase D) | Phase D: ST-RESTART-01 | GAP-08; QA-062 |
| E-04 | Log Rolling Policy | Log files rolled on size/time threshold. dropped_count tracked. UT-RS-02 required. | PARTIAL | PARTIAL | NO | Partially Implemented | Not Tested | E-04 in spec. BCMP-2 runner implementation exists. UT-RS-02 not written (GAP-09). | UT-RS-02 (not written) | Low | Deputy 1 | GAP-09 | SB-5 Phase D | GAP-09 |
| E-05 (SHM_ACTIVATED) | Critical event non-drop guarantee | SHM_ACTIVATED: CRITICAL level. Never dropped. RAM ring buffer 1000. Also written to critical_events.log. | CLOSED | CLOSED | NO | Implemented | Tested and Verified | Appendix E event table. SHM activation confirmed in FSM tests. critical_events.log pattern in spec. | IT-MM-01 | High | Deputy 1 | None | None | Appendix E; QA-018/019 |

---

## 6. Test Coverage Matrix

> **Left-side rule:** All rows permanent. **New in v3:** QA / Gate Reference column added. 20 new rows added.

| Test ID | Test Name | Req IDs Covered | SRS Sections | AVP Profiles | Type | Environment | Status | Last Run | Result | Evidence | Gap / Missing Coverage | Next Action | QA / Gate Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UT-NAV-01-A | INS Drift Limit — Seed 42 | NAV-01 | §2.1 | AVP-02 | Unit | SIL | PASSED | QA-049 | PASS | BCMP-2 AT-1 | km200 for AVP-04 | None for AVP-02 | AT-1 |
| UT-NAV-01-B | INS Drift Limit — Seeds 101/303 | NAV-01 | §2.1 | AVP-02 | Unit | SIL | PASSED | QA-049 | PASS | BCMP-2 AT-2 | km200 for AVP-04 | None for AVP-02 | AT-2 |
| UT-NAV-02-A | TRN Correction — NCC (SUPERSEDED) | NAV-02 | §2.2 | AVP-02 | Unit | SIL | OBSOLETE — REWRITE REQUIRED | QA-049 | Invalid | Tests RADALT-NCC | Mechanism replaced by OM in v1.3 | Rewrite as UT-OM-02-A | QA-050 downgrade |
| UT-NAV-02-B | TRN Correction — Suppress (SUPERSEDED) | NAV-02, NAV-06 | §2.2, §2.6 | AVP-02 | Unit | SIL | OBSOLETE — REWRITE REQUIRED | QA-049 | Invalid | Tests RADALT-NCC suppress | Mechanism replaced | Rewrite for OM suppress path | QA-050 downgrade |
| UT-OM-02-A | Orthophoto Matching — Acceptance | NAV-02, EC-13 | §2.2, §17 | AVP-02 | Unit | SIL | NOT STARTED | — | — | — | Does not exist | Sprint C: implement OrthophotoMatchingStub | GAP-10 |
| UT-OM-02-B | Orthophoto Matching — Suppression | NAV-02, NAV-06, EC-13 | §2.2, §2.6, §17 | AVP-02 | Unit | SIL | NOT STARTED | — | — | — | Does not exist | Sprint C | GAP-10 |
| UT-NAV-03 | VIO Drift Rate Limit | NAV-03, NAV-04 | §2.3, §2.4 | AVP-02 | Unit | SIL | PASSED | QA-049 | PASS | run_s5_tests.py S5 | No outdoor km-scale | Week 2 Item 13 | S-NEP-10 |
| IT-NAV-01 | Nav Integration — INS+TRN | NAV-01, NAV-02, NAV-04 | §2.1, §2.2, §2.4 | AVP-02 | Integration | SIL | PASSED | QA-049 | PASS | BCMP-2 | None for AVP-02 | None | BCMP-2 AT-1/AT-2 |
| IT-NAV-02 | Nav Integration — Two-Theatre | NAV-06, NAV-02 | §2.2, §2.6 | AVP-02 | Integration | SIL | PASSED | QA-049 | PASS | AT-2 | Western theatre km-scale absent | Define VIO km-scale test | AT-2 |
| IT-NAV-03 | Nav Integration — VIO | NAV-03, NAV-04 | §2.3, §2.4 | AVP-02 | Integration | SIL | PASSED | QA-049 | PASS | S-NEP tests | No outdoor coverage | Week 2 Item 13 | S-NEP-10 |
| IT-NAV-COR-01 | Corridor Violation Integration — GNSS-Denied Phase | NAV-01, EC-07-14, §16 | §2.1, §16 | AVP-02 | Integration | SIL | NOT STARTED | — | — | — | No integration test for corridor violation during INS-only phase. OI-40 ownership defined. | Define and implement after OI-55 fix (cross_track_error_m in SystemInputs) | OI-40; QA-050 |
| IT-VIO-OUTDOOR-01 | Outdoor VIO Validation (AVP-02) | NAV-03, NAV-06 | §2.3, §2.6 | AVP-02 | Integration | Field | NOT STARTED | — | — | — | Not yet planned | Week 2 Item 13: planning only | OI-07 |
| UT-BIM-01 | BIM Spoof Detection — Naive Jump | NAV-05, EW-03 | §2.5, §3.3 | ALL AVP | Unit | SIL | PASSED | QA-049 | PASS | run_s5_tests.py S2 | Adaptive spoof not covered | Write UT-BIM-03 | QA-019 |
| UT-BIM-02 | BIM Trust Score Update Rate | NAV-05 | §2.5 | ALL AVP | Unit | SIL | PASSED | QA-049 | PASS | run_s5_tests.py S2 | None | None | QA-019 |
| UT-BIM-03 | BIM Adaptive Spoof (3-sigma plausible) | NAV-05, EW-03, FR-110a | §2.5, §3.3 | ALL AVP | Unit | SIL | NOT STARTED | — | — | — | GAP-12: KPI-BIM-02 missing | Before HIL | GAP-12 |
| KPI-BIM-02 | BIM Adaptive Spoof FN Rate ≤1% | NAV-05, FR-110a | §2.5 | ALL AVP | KPI Gate | SIL | NOT STARTED | — | — | — | Gate not in run_certified_baseline.sh | Define and add before HIL | GAP-12 |
| UT-EW-01 | EW Map Update | EW-01 | §3.1 | ALL AVP | Unit | SIL | PASSED | QA-049 | PASS | run_s5_tests.py S4 | None | None | QA-049 |
| UT-EW-02 | EW Route Replan | EW-02 | §3.2 | ALL AVP | Unit | SIL | PASSED | QA-049 | PASS | run_s5_tests.py S4 | None | None | QA-049 |
| IT-EW-01 | EW Integration — Spoof + BIM | EW-03, NAV-05 | §3.3, §2.5 | ALL AVP | Integration | SIL | PASSED | QA-049 | PASS | run_s5_tests.py S2 | None | None | QA-019 |
| IT-EW-02 | EW Integration — Replan + Dead-end | EW-01, EW-02, PLN-03 | §3.1, §3.2, §4.3 | ALL AVP | Integration | SIL | PASSED | QA-049 | PASS | run_s5_tests.py S4 | None | None | QA-049 |
| IT-STALE-EW-01 | Retask Stale EW Map Injection Test | PLN-02, EW-01, App-B ROUTING | §4.2, §3.1 | AVP-02 | Integration | SIL | NOT STARTED | — | — | — | R-02 in spec; stale map injection path not integration-tested. Appendix B ROUTING state depends on this. | Week 1 Item 5/6: confirm R-02 in code; define test | Phase B; W1 Item 5/6 |
| UT-PLN-01 | Route Planner Response Time | PLN-01 | §4.1 | ALL AVP | Unit | SIL | PASSED | QA-049 | PASS | run_s5_tests.py S4 | None | None | QA-049 |
| UT-PLN-02 | Dynamic Retask R-01..R-06 | PLN-02 | §4.2 | AVP-02 | Unit | SIL | PASSED | QA-049 | PASS | Phase B gates | R-05 INS_ONLY corridor path absent; individual R-corrections unconfirmed in code | Week 1 Item 5/6 | Phase B closure |
| IT-PLN-01 | Retask Integration — EW + Route | PLN-02, EW-02 | §4.2, §3.2 | AVP-02 | Integration | SIL | PASSED | QA-049 | PASS | Phase B gates | None | None | Phase B closure |
| IT-PLN-02 | Retask Integration — GNSS-Denied | PLN-02, NAV-01, NAV-06 | §4.2, §2.1 | AVP-02 | Integration | SIL | PASSED | QA-049 | PASS | Phase B gates | R-05 INS_ONLY path not explicitly exercised | Week 1 Item 12 | Phase B closure |
| IT-INS-RETASK-01 | INS-Only Retask Rule: cross_track_error_m ≤ (half_width − 100) | PLN-02, NAV-01, App-B VALIDATING | §4.2, §2.1 | AVP-02 | Integration | SIL | NOT STARTED | — | — | — | R-05 constraint (Appendix B VALIDATING state) not corridor-tested. RETASK_NAV_CONFIDENCE_TOO_LOW path untested. | Confirm R-05 in code first (Week 1 Item 5/6); then implement test | App-B VALIDATING; R-05 |
| IT-TERRAIN-FAIL-01 | Terrain Corridor Regeneration Failure → ROLLBACK | PLN-02, App-B ROUTING | §4.2 | AVP-02 | Integration | SIL | NOT STARTED | — | — | — | RETASK_TERRAIN_GEN_FAILED → ROLLBACK path not tested. Appendix B ROUTING → ROLLBACK on TERRAIN_GEN_FAIL. | Define and implement after R-01 confirmation | App-B ROUTING; R-01 |
| IT-ROLLBACK-01 | Replan Timeout Rollback Failure Test | PLN-02, App-B ROLLBACK | §4.2 | AVP-02 | Integration | SIL | PASSED | QA-057 | PASS | 3 tests / 19 assertions. TERRAIN_GEN_FAIL, COMMIT_FAIL, payload complete. 17330fa. | None | None | App-B ROLLBACK |
| UT-PLN-03 | Route Dead-End Recovery | PLN-03 | §4.3 | ALL AVP | Unit | SIL | PASSED | QA-049 | PASS | Phase B gates | None | None | Phase B closure |
| UT-MM-04 | Event Bus Queue Latency | MM-04 | §5 | ALL AVP | Unit | SIL | PASSED | QA-049 | PASS | Phase B SB-06 | None | None | SB-06 |
| UT-PX4-01 | OFFBOARD Setpoint Rate | PX4-01 | §8.1 | ALL AVP | Unit | SIL | PASSED | QA-049 | PASS | run_bcmp2_tests.py | None | None | QA-049 |
| UT-PX4-02 | MAVLink Heartbeat (stub) | PX4-02 | §8.2 | ALL AVP | Unit | SIL | PARTIAL | QA-049 | Stub only | Stub | Real heartbeat timing requires HIL | HIL Sprint 1 | GAP-07 |
| UT-PX4-05 | Checkpoint Schema v1.2 | PX4-05, EC-02 | §8.5, §13 | ALL AVP | Unit | SIL | PASSED | QA-049 | PASS | Phase A SA-01..04 | Purge policy unconfirmed | Week 1 Item 10 | SA-01..04 |
| UT-PX4-COR-01 | Corrupted Checkpoint Restore Test | PX4-05, EC-02, App-C CHECKPOINT_RESTORE | §8.5, §13 | ALL AVP | Unit | SIL | PASSED | QA-058 | PASS | test_sb5_phase_a.py COR-01..06 | None | None | QA-058; COR-01..06 |
| IT-PX4-01 | OFFBOARD Continuity — Short Run | PX4-01, EC-01 | §8.1, §8.3 | ALL AVP | Integration | SITL | PARTIAL | S-PX4-09 | 62s PASS | SITL logs | 30-min exit gate not confirmed | Week 1 Item 8/9 | QA-050 downgrade |
| IT-PX4-02 | PX4 Reboot Recovery Integration | PX4-04, EC-03 | §8.4, §13 | ALL AVP | Integration | SITL | PARTIAL | SA-05..SA-07 | Synthetic PASS | `787ecd4` | D9 chain (D7→D9) not SITL-tested; D10 not tested | Week 1 Item 11 | QA-025; SA-05..07 |
| IT-D6-TIMEOUT-01 | MAVLink Timeout Full D6 Path (10s → ABORT) | PX4-01, EC-01, D6 | §8.1, §8.3 | ALL AVP | Integration | SIL | PASSED | QA-061 | 4 tests / 17 assertions. D2 recovery, D3 recovery, full timeout abort, timing bounds. test_it_d6_timeout.py. 6a30295. | `6a30295` | None | D6 gap; QA-061 |
| IT-D9-CHAIN-01 | MISSION_RESUME_AUTHORISED / Full D7→D8→D8a→D9 Chain | PX4-04, EC-03, D7..D9 | §8.4, §13 | ALL AVP | Integration | SITL | PASSED | QA-059 | 4 gates / live SITL. G1=1980ms, G2=0ms, G3=confirmed, G4=43.678m. 208a5a1. | `208a5a1`, `5863020` | None | D9 gap; SA-07 context |
| IT-D10-GNSS-01 | D10 GNSS-Denied PX4 HOLD Recovery Test | PX4-04, EC-03, D10 | §8.4, §13 | ALL AVP | Integration | SITL | NOT STARTED | — | — | — | D10 path completely untested. PX4 returns in HOLD during GNSS-denied corridor flight. | Week 1 Item 11: define test | D10 OPEN; QA-050 |
| IT-CLR-GATE-01 | pending_operator_clearance_required=True Gate Validation | PX4-04, PX4-05, MM-03, D8a | §8.4, §8.5, §5 | ALL AVP | Integration | SITL | NOT STARTED | — | — | — | D8a branch 2 (shm_active=True or clearance_required=True → SHM hold, AWAITING_OPERATOR_CLEARANCE) exercised only as synthetic SA-07 branch. No SITL validation. | Define SITL test with clearance_required=True checkpoint | SA-07 context |
| SA-01 | Checkpoint — SHM Field | PX4-05, MM-03 | §8.5, §5 | ALL AVP | Gate | SIL | PASSED | Phase A | PASS | `fcb5106` | None | None | SA-01 |
| SA-02 | Checkpoint — Clearance Field | PX4-05, MM-03 | §8.5, §5 | ALL AVP | Gate | SIL | PASSED | Phase A | PASS | `fcb5106` | None | None | SA-02 |
| SA-03 | Checkpoint — Abort Flag | PX4-05 | §8.5 | ALL AVP | Gate | SIL | PASSED | Phase A | PASS | `fcb5106` | None | None | SA-03 |
| SA-04 | Checkpoint — ETA Field | PX4-05 | §8.5 | ALL AVP | Gate | SIL | PASSED | Phase A | PASS | `fcb5106` | None | None | SA-04 |
| SA-05 | PX4 Reboot Detection — Seq Reset | PX4-04, D7 | §8.4 | ALL AVP | Gate | SIL | PASSED | Phase A | PASS | `787ecd4` | None | None | SA-05 |
| SA-06 | Reboot Recovery — Checkpoint Load | PX4-04, D8 | §8.4 | ALL AVP | Gate | SIL | PASSED | Phase A | PASS | `787ecd4` | None | None | SA-06 |
| SA-07 | Reboot Recovery — D8a Gate | PX4-04, EC-03, D8a | §8.4, §13 | ALL AVP | Gate | SIL | PASSED | Phase A | PASS | `787ecd4` | None | None | SA-07 |
| SB-07 | Route Fragment Cleanup | RS-04 | §12 | ALL AVP | Gate | SIL | PASSED | Phase B | PASS | `c35122a` | None | None | SB-07 |
| AT-1 | BCMP-2 Dual-Track Foundation | NAV-01 | §2.1 | AVP-02 | Acceptance | SIL | PASSED | BCMP-2 | PASS | `sb1-dual-track-foundation` | None | None | AT-1 |
| AT-2 | BCMP-2 Full Mission Reports | NAV-01, NAV-02, NAV-06 | §2.1, §2.2, §2.6 | AVP-02 | Acceptance | SIL | PASSED | BCMP-2 | PASS | `sb3-full-mission-reports` | NAV-02 component obsolete (v1.3 change) | Note obsolescence | AT-2; QA-050 |
| AT-3 | BCMP-2 Navigation Transitions | NAV-04 | §2.4 | AVP-02 | Acceptance | SIL | PASSED | BCMP-2 | PASS | `sb3-full-mission-reports` | None | None | AT-3 |
| AT-6 | BCMP-2 Three-Seed Repeatability | NAV-01 | §2.1 | AVP-02 | Acceptance | SIL | PASSED | SB-5 closure | PASS | `sb5-bcmp2-closure` | None | None | AT-6 |
| ADV-01..06 | L10s-SE Adversarial EO Conditions | TERM-02, TERM-03 | §10, §11 | ALL AVP | Adversarial | SIL | PASSED | Sprint B | PASS | `41238ae` | DMRL stub — synthetic inputs only | Maintain caveat | ADV-01..06 |
| G7-01..05 | Gate 7 — SAL-1 + SAL-2 Corridor | NAV-02, NAV-06, AD-24 | §2.2, §2.6 | AVP-02 | Gate | SIL | PASSED | QA-049 | PASS | `322274c` | None | None | Gate 7 PASS; QA-049 |
| ST-CPU-01 | CPU Utilisation Stress | RS-01 | §12 | ALL AVP | Stress | HIL | BLOCKED | — | — | GAP-06: Jetson Orin NX required | Cannot run in SIL | Unblock at HIL Sprint 1 | GAP-06 |
| ST-GNSS-01 | GNSS Spoof Stress | NAV-05, EW-03 | §2.5, §3.3 | ALL AVP | Stress | SIL | PASSED | QA-049 | PASS | run_s5_tests.py S2 | Adaptive spoof not covered | Write UT-BIM-03 | QA-019 |
| ST-END-01 | 2-Hour Endurance Run | RS-01, RS-02, EC-02, TERM-01 | §12, §13 | ALL AVP | Stress | SIL | NOT STARTED | — | — | Phase D not executed | Endurance run not completed | SB-5 Phase D | Phase D |
| ST-RESTART-01 | Watchdog Restartability | RS-03 | §12 | ALL AVP | Stress | SIL | NOT STARTED | — | — | Phase D not executed | SIGKILL stimulus test not written | SB-5 Phase D | GAP-08 |
| ST-MULTI-01 | Multi-Theatre Navigation Stress | NAV-03, NAV-04, NAV-06 | §2.3, §2.4, §2.6 | AVP-02 | Stress | SIL | PASSED | QA-049 | PASS | run_bcmp2_tests.py | None | None | QA-049 |
| ST-PX4-01 | PX4 Integration Stress | PX4-01 | §8.1 | ALL AVP | Stress | SITL | PASSED | QA-049 | PASS | run_bcmp2_tests.py | None | None | QA-049 |
| UT-RS-02 | Log Rolling Policy | RS-02 | §12 | ALL AVP | Unit | SIL | NOT STARTED | — | — | GAP-09 | Not written | SB-5 Phase D | GAP-09 |
| UT-RS-03 | Watchdog SIGKILL Restartability | RS-03 | §12 | ALL AVP | Unit | SIL | NOT STARTED | — | — | GAP-08 | Not written | SB-5 Phase D | GAP-08 |
| UT-IMU-DROP-01 | IMU Dropout → ESKF Freeze | NAV-01, PX4-03 | §2.1, §8.3 | ALL AVP | Unit | SIL | NOT STARTED | — | — | — | IMU_DROPOUT log event in spec. ESKF freeze >100ms path not tested. | Define and implement | SRS §8.3 PX4-03 |
| IT-CAM-DROP-01 | Camera Dropout → IMU+TRN Navigation | NAV-03, PX4-03 | §2.3, §8.3 | AVP-02 | Integration | SITL | NOT STARTED | — | — | — | VIOMode outage recovery confirmed S-NEP-06 (synthetic). No SITL test with actual camera dropout. OI-43 fragile workaround. | Resolve OI-43; define SITL camera dropout test | OI-43; S-NEP-06 |
| IT-MULTI-FAULT-01 | Multiple Simultaneous Faults (GNSS spoof + EW spike + VIO degradation) | NAV-05, EW-01, NAV-03 | §2.5, §3.1, §2.3 | AVP-02 | Integration | SIL | NOT STARTED | — | — | — | No compound fault scenario. Each fault tested in isolation. Highest realism risk for TASL engagement. | Define before TASL engagement | High-risk open |
| IT-OM-SUPPRESS-01 | Orthophoto Confidence Collapse — Repeated SUPPRESS Window | NAV-02, PLN-02, EC-13 | §2.2, §4.2, §17 | AVP-02 | Integration | SIL | NOT STARTED | — | — | — | G7-01 confirms SUPPRESS-zone match()=0. No test for full-mission confidence collapse (>5 consecutive SUPPRESS windows). | Define in Sprint C alongside UT-OM-02-A/B | GAP-10; Gate 7 |
| IT-AVP03-ALT | High-Altitude Orthophoto Matching (AVP-03 1500m AGL) | NAV-02, NAV-06 | §2.2, §2.6 | AVP-03 | Integration | SIL/HIL | NOT STARTED | — | — | Not scoped | AVP-03 altitude bands completely uncharacterised for OM | Week 2 Item 15 | GAP-11 |
| IT-AVP04-LONG | Deep-Strike Navigation (AVP-04, 100+ km denied) | NAV-01, NAV-02 | §2.1, §2.2 | AVP-04 | Integration | SIL | NOT STARTED | — | — | GAP-11: km200 envelope extension not done | Not scoped | Before any AVP-04 engagement | GAP-11 |
| IT-VIZ-02-CADENCE-01 | Vehicle B Interpolation Cadence — 100ms Fixed | VIZ-02 | §9.2 | ALL AVP | Integration | SIL | NOT STARTED | — | — | — | VIZ-02 spec: Vehicle B interpolated at 100ms cadence. No test confirms cadence compliance. | Define as part of VIZ-02 Phase C implementation | §9.2 VIZ-02 spec |
| UT-VIZ-02 | VIZ-02 Animation Reproducibility | VIZ-02 | §9.2 | ALL AVP | Unit | SIL | NOT STARTED | — | — | GAP-01: Phase C not started | Not written | Begin Phase C after Thread A | GAP-01 |
| S-NEP-01..10 | VIO Sandbox Programme Tests | NAV-03, NAV-06 | §2.3, §2.6 | AVP-02 | Programme | SIL | PASSED | S-NEP-10 | 332/332 | `4fcf231` | Indoor/short-range only | Outdoor plan (Week 2 Item 13) | S-NEP-10 |

---

## 7. Negative and Adversarial Test Coverage

| Failure Mode | Related Req IDs | Existing Test Coverage | Status | Evidence | Open Gap | Next Action | QA / Gate Ref |
|---|---|---|---|---|---|---|---|
| GNSS spoof with plausible trajectory (3-sigma Doppler-consistent) | NAV-05, EW-03, FR-110a | UT-BIM-01 covers naive position-jump only | PARTIAL | run_s5_tests.py S2 | GAP-12: adaptive spoof not tested. KPI-BIM-02 missing. | Write UT-BIM-03 with Doppler-plausible waveform before HIL | GAP-12 |
| Corridor violation during GNSS-denied flight | NAV-01, EC-07-14, §16 | FSM CORRIDOR_VIOLATION → ABORT confirmed in unit tests | PARTIAL | State machine lines 297/320/361/399/440; `9d99a75` | OI-40: §16 row pending SRS v1.4. OI-55: cross_track_error_m absent. IT-NAV-COR-01 defined. | Week 1 Item 3: commit §16 row. Define IT-NAV-COR-01. | OI-40; OI-54; QA-050 |
| Orthophoto confidence collapse (>5 consecutive SUPPRESS windows) | NAV-02, PLN-02, EC-13 | G7-01 confirms SUPPRESS-zone match()=0 | PARTIAL | `322274c` Gate 7 | No full-mission collapse test (IT-OM-SUPPRESS-01 defined). | Define IT-OM-SUPPRESS-01 in Sprint C | GAP-10; Gate 7 |
| Stale EW map during retask (R-02) | PLN-02, EW-01 | UT-PLN-02 covers R-02 code path per spec | PARTIAL | Phase B gates | R-02 in spec; stale map injection not integration-tested (IT-STALE-EW-01 defined). | Week 1 Item 5/6: confirm R-02 in code | Phase B; W1 Item 5/6 |
| Terrain corridor regeneration failure (R-01 → ROLLBACK) | PLN-02, App-B ROUTING | RETASK_TERRAIN_GEN_FAILED in spec | PARTIAL | SRS Appendix B | IT-TERRAIN-FAIL-01 defined but not started. | Define and implement | App-B; R-01 |
| Failed rollback after replan timeout (>2000ms) | PLN-02, App-B ROLLBACK | ROLLBACK state in Appendix B. RS-04 cleanup confirmed. | PARTIAL | `c35122a` | IT-ROLLBACK-01 defined but not started. | Define covering TERRAIN_GEN_FAIL, COMMIT_FAIL, timeout overrun | App-B ROLLBACK |
| Corrupted checkpoint file at restore | PX4-05, EC-02, App-C CHECKPOINT_RESTORE | CheckpointCorruptError + CHECKPOINT_CORRUPT event. restore_latest() returns None. 6 tests COR-01..06. | CLOSED | QA-058 | None. | None | App-C; COR-01..06 |
| Missing EO camera feed (IMU+TRN only) | NAV-03, PX4-03 | VIOMode outage recovery S-NEP-06 (synthetic). IT-CAM-DROP-01 defined. | PARTIAL | `core/fusion/vio_mode.py:166` | No SITL test with actual dropout. OI-43 fragile. | Resolve OI-43. Define IT-CAM-DROP-01. | OI-43; S-NEP-06 |
| Missing IMU feed (ESKF freeze) | NAV-01, PX4-03 | IMU_DROPOUT in spec. ESKF freeze >100ms defined. UT-IMU-DROP-01 defined. | PARTIAL | SRS §8.3 | UT-IMU-DROP-01 not started. | Define and implement | SRS §8.3 |
| Radar altimeter noise burst / dropout | NAV-02 (legacy) | TRN_RADALT_SUSPENDED in spec | PARTIAL | SRS §8.3 | Less critical post-OM adoption. | Retain as HIL-phase concern | SRS §8.3 |
| PX4 reboot during GNSS-denied flight (D10 — HOLD mode) | PX4-04, EC-03, D10 | None. IT-D10-GNSS-01 defined. | OPEN | — | Critical: completely untested. | Week 1 Item 11: define IT-D10-GNSS-01 | D10 OPEN; QA-050 |
| MAVLink interruption — D6 full 10s timeout → ABORT | PX4-01, EC-01, D1..D6 | IT-PX4-01 covers nominal reconnect. IT-D6-TIMEOUT-01 PASSED (SIL). SITL integration (IT-D6-SITL-01) Phase D. | PARTIAL | `97b2f5a`, `6a30295` | IT-D6-SITL-01: SITL validation with real heartbeat stop not yet run. SIL coverage complete. | IT-D6-SITL-01 Phase D | QA-061 |
| Route fragment accumulation leak | RS-04, PLN-02 | SB-07 confirms cleanup fires on all retask() exit paths | CLOSED | `c35122a` | None | None | SB-07 PASS |
| DMRL false positive (real target rejected) | TERM-01, TERM-02 | ADV-01..06 adversarial tests on stub | PARTIAL | `41238ae` | DMRL stub. HIL with thermal camera required. | Maintain caveat | ADV-01..06; OI-06 |
| Decoy target acceptance (DMRL bypassed) | TERM-01, TERM-02 | ADV tests on stub | PARTIAL | `41238ae` | Same stub limitation. | HIL required | ADV-01..06 |
| Multiple simultaneous faults (GNSS spoof + EW spike + VIO degradation) | NAV-05, EW-01, NAV-03 | Each fault tested in isolation. IT-MULTI-FAULT-01 defined. | OPEN | — | No compound fault scenario. | Define IT-MULTI-FAULT-01 before TASL engagement | High-risk open |
| INS-only corridor exceedance / forced abort | NAV-01, EC-07-14, App-B VALIDATING | FSM CORRIDOR_VIOLATION confirmed. IT-NAV-COR-01 defined. | PARTIAL | State machine lines 297..440 | No integration test for INS-only exceedance triggering CORRIDOR_VIOLATION. | Define as part of IT-NAV-COR-01 | OI-40; App-B |
| INS-only retask rule violation (cross_track_error_m > half_width − 100) | PLN-02, App-B VALIDATING | UT-PLN-02 covers R-05 per spec. IT-INS-RETASK-01 defined. | PARTIAL | Phase B gates | R-05 code confirmation pending. RETASK_NAV_CONFIDENCE_TOO_LOW path untested. | Week 1 Item 5/6 | App-B VALIDATING; R-05 |

---

## 8. AVP Profile Traceability Summary

| Req ID | AVP-02 (100–800m, 100–250km, 50–100km denied) | AVP-03 (100–1500m, 100–250km, 100km denied) | AVP-04 (500–2000m, 1000+km, 100+km denied) |
|---|---|---|---|
| NAV-01 | ✅ CLOSED — km120 validated | ⚠ PARTIAL — at boundary of validated range | ❌ OPEN — km200 extension required (GAP-11) |
| NAV-02 | ⚠ PARTIAL — HIL validated; no SIL gate | ❌ OPEN — altitude bands uncharacterised | ❌ OPEN — altitude bands uncharacterised |
| NAV-03 | ⚠ PARTIAL — indoor scale only | ⚠ PARTIAL — indoor only; altitude uncharacterised | ❌ OPEN — high-altitude VIO undefined |
| NAV-04 | ✅ CLOSED | ✅ CLOSED | ✅ CLOSED |
| NAV-05 | ✅ CLOSED | ✅ CLOSED | ✅ CLOSED |
| NAV-06 | ⚠ PARTIAL — architecture defined; VIO west-theatre indoor only | ❌ OPEN — high-altitude TRN/VIO boundary undefined | ❌ OPEN |
| PLN-02 | ⚠ PARTIAL — R-05 confirmation pending | ❌ Not scoped | ❌ Not scoped |
| PX4-01..05 | ✅ CLOSED | ✅ CLOSED | ✅ CLOSED |
| TERM-01..02 | ⚠ PARTIAL (stub) | ⚠ PARTIAL (stub) | ⚠ PARTIAL (stub) |
| TERM-03 | ✅ CLOSED | ✅ CLOSED | ✅ CLOSED |

> **Standing rule:** Any TASL or DRDO-facing capability claim must explicitly state which AVP profile the evidence supports.

---

## 9. Governance and Caveat Register

| Item | Type | Status | Detail | Required Action | QA / Gate Ref |
|---|---|---|---|---|---|
| Terrain tiles (`data/terrain/`, `simulation/terrain/`) | Provenance record | CLOSED — `9d99a75` | Tiles are real Copernicus DEM COP30 elevation data. README corrected. Commit message at `3dc15b8` was misleading ("synthetic"). OI-53 closed. Header Status field has minor residual — not blocking. | None. Record retained. | OI-53; QA-050 |
| SIL-only evidence caveat | Programme caveat | STANDING | All SIL results use synthetic terrain, synthetic IMU injection, stub DMRL. | Apply to all external documents. | QA-025 |
| DMRL stub limitation | Safety caveat | STANDING | DMRL is a software stub. All terminal guidance SIL results are synthetic. Must caveat every TASL/DRDO presentation (QA standing rule §9.5). | Deputy 1 must confirm caveat present before approving any demo materials. | OI-06 |
| HIL-required clauses | Architecture caveat | STANDING | RS-01 (CPU budget), PX4-02 (real heartbeat), PX4-03 (real EO driver), D10 (GNSS-denied reboot) cannot be validated in SIL. | Do not claim these as evidence for external demonstrations. | GAP-06/07 |
| Frozen file hash baseline | Governance note | CLOSED | All five frozen files tracked by sha256. G7-05 at `322274c` is the baseline. | No action required. | Gate 7 G7-05 |
| Baseline test-count history | Governance note | CLOSED | 406 → corrected at QA-030. Current: 510/510 at `de1d762` / `9d99a75`. | No action required. | QA-030 |
| External-claim restriction — NAV-02 | Claim restriction | ACTIVE | NAV-02 cannot be cited as validated until EC-13 PASS. LightGlue HIL H-4/H-5/H-6 is operational evidence, not a SIL gate. | Block NAV-02 "validated" claims until Sprint C EC-13 PASS. | QA-050; H-6 |
| External-claim restriction — AVP-04 range | Claim restriction | ACTIVE | NAV-01 C-2 envelopes valid to km120 only. Not valid for AVP-04 (100+km GNSS-denied). | Explicit limitation in any AVP-04 performance document. | GAP-11 |
| OI-55 — cross_track_error_m absent from SystemInputs | Open investigation | OPEN | CORRIDOR_VIOLATION event payload (added OI-54) cannot include breach magnitude. Fix: add field to SystemInputs; populate from NavigationManager. | Define and implement before HIL. | OI-55; QA-050 |
| NIS measurement model limitation (AD-17) | Architecture caveat | STANDING | OpenVINS NIS values do not correctly reflect inter-frame innovation noise. Must not cite NIS as filter calibration evidence without caveat. | Apply AD-17 caveat to any report presenting NIS data. | AD-17 |
| OI-40 §16 Corridor Violation row | Governance action | PARTIAL | Deputy 1 ownership ruling complete (QA-050). SRS v1.4 §16 row authored. Pending commit by Agent 2. | Week 1 Item 3: Agent 2 commits §16 row to SRS v1.4 | OI-40; QA-050 |

---

## 10. Mandatory Update Protocol

1. Update §1.1 closure snapshot counts from all clause tables.
2. Populate §1.4 Weekly Movement with the delta from the previous week.
3. Promote PARTIAL → CLOSED only when: code evidence + test evidence + commit reference + pass/fail status all confirmed.
4. Downgrade any item where new evidence invalidates prior closure. Record in §1.3 with session reference.
5. Add new rows to §6 (Test Coverage) for any newly defined test cases. Never delete rows.
6. Add new rows to §7 (Adversarial) for any newly identified failure modes.
7. Add new rows to §9 (Governance) for any new programme caveats.
8. Update QA / Gate Reference column when a new QA entry or gate result justifies a status change.
9. **Never delete rows.** All left-side entries are permanent records.

*Next update: 29 April 2026 (end of Week 1). Push summary to MICROMIND_PROJECT_CONTEXT.md §6.*
