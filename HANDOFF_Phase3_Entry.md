# HANDOFF — Phase 3 Entry
**Date:** 29 March 2026
**Basis:** MicroMind_PreHIL_v1_2.docx (Final) · micromind-autonomy @ e5d8530
**Previous phase:** Phase 2 complete · CP-2 tagged · 332/332 SIL · 206/206 integration

---

## Programme Gate Status

| Gate | Status | Tag |
|---|---|---|
| CP-0 — Architecture lock | ✅ PASS | — |
| CP-1 — Phase 1 driver layer | ✅ PASS | `cp1-phase1-pass` |
| CP-1.5 — PX4 OFFBOARD | ✅ PASS | `cp1_5-offboard-pass` |
| CP-2 — Latency + timing | ✅ PASS | `cp2-latency-pass` |
| CP-3 — Pre-HIL readiness | ⏳ PENDING | — |

---

## CP-3 Exit Conditions (Part 12.5)

CP-3 is NOT closed until all six conditions are met. No partial credit.

| # | Condition | Status |
|---|---|---|
| 1 | RC-7 pass — IFM-01 fires on injected timestamp, ESKF continues, event logged | PENDING |
| 2 | RC-8 pass — logger drop rate <0.5% at 200Hz for 60s | PENDING |
| 2a | RC-8 Jetson note — repeated under renice +10 or 2-core taskset | PENDING |
| 3 | RC-11a–e all pass — see decomposition below | PENDING |
| 4 | Gazebo rendering functional — vehicle visible in GUI | PENDING |
| 5 | mark_send integrated natively into mavlink_bridge._setpoint_loop | PENDING |
| 6 | CP-3 readiness report committed to dashboard/ | PENDING |

---

## RC-11 Decomposition (Part 12.4)

RC-11 is decomposed into five sub-criteria. All five must pass. Test in order.

| Sub-RC | Criterion | Risk |
|---|---|---|
| RC-11a | Setpoint continuity at 20Hz during 10s VIO blackout | Medium |
| RC-11b | No NaN / non-finite setpoints during OUTAGE — ESKF does not diverge | **HIGH — may expose new failure mode** |
| RC-11c | OFFBOARD retained throughout VIO outage (custom_mode=393216) | Medium |
| RC-11d | OFFBOARD recovered after 5s MAVLink disconnect | Medium |
| RC-11e | OUTAGE→RESUMPTION→NOMINAL chain logged and auditable | Low |

**RC-11b is the highest-risk step in Phase 3.** If the ESKF produces NaN during OUTAGE,
the correct response is to log it, escalate to TD, and not patch the ESKF.
The ESKF is frozen. Any change requires a full S-NEP re-run.

---

## Demo Readiness Ladder (Part 12.8)

| Rung | Label | Status |
|---|---|---|
| DR-1 | CP-3 passed | PENDING |
| DR-2 | Gazebo rendering working | PENDING — parallel to Phase 3 |
| DR-3 | run_demo.sh reproducible 3× | PENDING — Phase 4 |
| DR-4 | Overlays working (6 mandatory, v1.2 §9.11) | PENDING — Phase 4 |
| DR-5 | Screen recording captured | PENDING — Phase 4 |
| DR-6 | OEM-ready demonstration | PENDING — pre-TASL |

---

## Phase 3 Scope — Bounded

**In scope:**
- integration/drivers/vio_driver.py — OfflineVIODriver wrapping EuRoC .npy + ENU→NED + IFM-01
- RC-7 formal timestamp injection test
- RC-8 formal logger drop-rate test (+ Jetson note run)
- RC-11a–e OUTAGE injection test in live SITL
- mark_send native integration (one line, pre-Phase 3)

**Explicitly out of scope before CP-3:**
- ROS2 bridge
- run_demo.sh
- Terminal overlay / curses panel
- HTML report wiring to live log
- Any modification to ESKF, BIM, vio_mode, or enforcement blocks
- Jetson Orin profiling
- EO/IR or sensor pipeline work

---

## Governing Documents

| Document | Location | Role |
|---|---|---|
| MicroMind_PreHIL_v1_2.docx | Project knowledge | Governing specification |
| MicroMind_SIA_v1_0.docx | Project knowledge | Sensor integration addendum |
| MicroMind_PreHIL_TD_Update_20260329.docx | Project knowledge | Session closure — Phase 0–2 evidence |
| MicroMind_PreHIL_Part12_ProgrammeAssessment_20260329.docx | Project knowledge | CP-3 exit definition · RC-11 decomposition · Demo Readiness Ladder |
| dashboard/micromind_prehil_cp2_latency_20260329.json | Repo @ e5d8530 | CP-2 latency evidence |
| dashboard/micromind_prehil_cp2_report_20260329.html | Repo @ e5d8530 | CP-2 HTML demo report |

---

## Next Session Entry
```bash
cd ~/micromind/repos/micromind-autonomy
git log --oneline | head -3
python3 -m pytest tests/ -q 2>&1 | tail -3        # 332 passed
python3 -m pytest integration/ -q 2>&1 | tail -3  # 206 passed
```

**First deliverable:** native mark_send integration (one line in
integration/bridge/mavlink_bridge._setpoint_loop). Commit before
any Phase 3 work begins.

**Second deliverable:** integration/drivers/vio_driver.py —
OfflineVIODriver wrapping EuRoC .npy loader with ENU→NED
(frame_utils) + IFM-01 monotonicity gate.
