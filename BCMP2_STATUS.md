# BCMP-2 Sprint Status

**Programme:** MicroMind / NanoCorteX  
**Phase:** BCMP-2 — Extended Mission Acceptance Regime  
**Repository:** amitkr-anand/micromind-autonomy (branch: main)  
**Document:** BCMP2_STATUS.md  
**Last updated:** 29 March 2026 2000 IST

---

## Governing Document

**MicroMind_BCMP2_Implementation_Architecture_v1_1.docx** (Project Knowledge)

Central question every scenario answers:
> *"What mission succeeds with MicroMind that fails without it?"*

---

## Current State

**SB-1 CLOSED** — Dual-track foundation complete.  
**Tag:** `sb1-dual-track-foundation`  
**Validated on:** micromind-node01 (Python 3.12.3, numpy 1.26.4, system python — no conda)

---

## Fixed Constraints (non-negotiable)

| ID | Constraint | Summary |
|---|---|---|
| C-1 | IMU Parity | Both vehicles use identical IMU class, noise profile, seed. Only correction stack differs. |
| C-2 | Drift Envelopes | Vehicle A drift pre-calculated from STIM300 TS1524 rev.31. Monte Carlo calibrated gates (N=300). |
| C-3 | Synthetic Terrain | Committed, offline, reproducible. 7 documented parameters per phase. |
| C-4 | Disturbance Parity | One schedule per run, generated from shared seed, serialised at top-level of JSON output. |

**C-2 Phase-Boundary Envelopes (Monte Carlo calibrated, N=300):**

| Phase Boundary | Floor | Nominal | Ceiling | Basis |
|---|---|---|---|---|
| km 60 — end P2 | 5 m | 19 m | 80 m | P5 / P50 / P99 of normal gyro bias draw |
| km 100 — end P3 | 12 m | 96 m | 350 m | P5 / P50 / P99 |
| km 120 — end P4 | 15 m | 155 m | 650 m | P5 / P50 / P99 |

Analytical derivation (STIM300 TS1524 rev.31): nominal 1σ = 44 / 211 / 338 m.  
Monte Carlo floors lower because individual seeds draw gyro bias from normal(0, 0.5 deg/h σ).

---

## Frozen Baseline — Zero Modifications Permitted

- `core/ekf/error_state_ekf.py`
- `core/fusion/vio_mode.py`
- `core/fusion/frame_utils.py`
- `core/bim/bim.py`
- `scenarios/bcmp1/bcmp1_runner.py` — enforcement blocks E-1..E-5
- All 332 SIL gates — must remain green after every addition

---

## Sprint History

### SB-1 — Dual-Track Foundation ✅ CLOSED

**Closed:** 29 March 2026  
**Tag:** `sb1-dual-track-foundation`

| Step | Commit | File | Gate |
|---|---|---|---|
| 1 | `7e29aad` | `scenarios/bcmp2/bcmp2_drift_envelopes.py` | Derived from STIM300 TS1524 rev.31 + S9 ESKF constants. Self-prints clean. |
| 2 | `0ad99c5` | `scenarios/bcmp2/bcmp2_terrain_gen.py` | 5 phases all OK. Determinism PASS. DEM patch (64×64) float64. |
| 3 | `c9012fa` | `scenarios/bcmp2/bcmp2_scenario.py` | Route, phase attribution, disturbance schedule, C-4 determinism PASS. |
| 4 | `62a386b` | `scenarios/bcmp2/baseline_nav_sim.py` + envelope update | C-2 gates: seeds 42/101/303 all PASS. 2.6 s per full 150 km run. |
| 5+6 | `6c37697` | `scenarios/bcmp2/bcmp2_runner.py` + `tests/test_bcmp2_at1.py` + `run_bcmp2_tests.py` | 17/17 AT-1 PASS. 111/111 S5 PASS. 68/68 S8 PASS. |

**AT-1 Gate Results (on micromind-node01):**

```
17/17 AT-1 PASS  (3.98s)
111/111 S5 PASS
68/68 S8 PASS
0 failures
```

**C-2 Gate Results (SB-1 validation runs):**

| Seed | km 60 | km 100 | km 120 | Corridor breach |
|---|---|---|---|---|
| 42 | 17 m ✅ | 91 m ✅ | 126 m ✅ | None |
| 101 | 57 m ✅ | 299 m ✅ | 471 m ✅ | km 123.4 |
| 303 | 24 m ✅ | 145 m ✅ | 249 m ✅ | km 149.2 |

**Key engineering decision made in SB-1:**  
Full 3D INS mechanisation feeds IMU noise terms only — without true specific force (gravity + vehicle acceleration) this causes unphysical vertical channel contamination that corrupts heading. Resolution: cross-track error propagation model, which directly matches the C-2 analytical derivation. Documented in JOURNAL.md.

---

### SB-2 — Fault Injection Infrastructure ⏳ PENDING

**Entry gate:** SB-1 closed ✅

**Deliverables:**

| File | Responsibility |
|---|---|
| `fault_injection/fault_manager.py` | Thread-safe fault state registry (threading.Lock on all reads/writes) |
| `fault_injection/sensor_fault_proxy.py` | Wraps GNSS, VIO, RADALT, EO outputs. Transparent when no fault active. |
| `fault_injection/nav_source_proxy.py` | TRN suppression, VIO suppression, IMU jitter |
| `tests/test_bcmp2_sb2.py` | Three scripted fault injection runs: FI-01, FI-02, FI-05 |

**Exit condition:** Three scripted fault runs produce correct Vehicle B mode transitions. Fault event log populated. No proxy call modifies any frozen core file.

**Thread safety note:** Follow Pre-HIL B-2 and B-3 threading fix pattern (threading.Event / threading.Lock). Design lock pattern before writing any proxy code.

---

### SB-3 — Full Mission and Reports ⏳ PENDING

**Deliverables:** Full five-phase terrain, FI-01..FI-15 scripted, `test_bcmp2_at2.py` + `test_bcmp2_at3_5.py`, `bcmp2_report.py` (JSON + HTML, business comparison block first).

**Exit condition:** AT-2 nominal run produces expected comparative outcome. Vehicle A exceeds corridor by P4. Vehicle B corridor maintained. HTML report business comparison block correct.

---

### SB-4 — Dash GUI and Replay ⏳ PENDING

**Deliverables:** `dashboard/bcmp2_dashboard.py` (7-panel Plotly Dash), `dashboard/bcmp2_replay.py` (4 replay modes: executive 2–3 min / technical 8–10 min / high-fidelity / overnight).

**Exit condition:** Executive replay runs 2–3 min correctly. Panel 7 outcome summary updates live. Operator-triggered fault produces identical Vehicle B behaviour to scripted fault.

---

### SB-5 — Repeatability and Closure ⏳ PENDING

**Deliverables:** `tests/test_bcmp2_at6.py`, overnight stress run, final HTML report, BCMP-2 Closure Report.

**Exit condition:** Three consecutive seed runs (42/101/303) produce identical phase transition chains. No memory leak over 4-hour stress run.

---

## File Ownership Map

```
scenarios/bcmp2/
    bcmp2_drift_envelopes.py     ✅ SB-1  C-2 envelope constants (STIM300 derived + Monte Carlo)
    bcmp2_terrain_gen.py         ✅ SB-1  C-3 terrain generator (7 documented parameters)
    bcmp2_scenario.py            ✅ SB-1  5-phase mission + C-4 disturbance schedule
    baseline_nav_sim.py          ✅ SB-1  Vehicle A cross-track dead-reckoning
    bcmp2_runner.py              ✅ SB-1  Dual-track orchestrator
    bcmp2_report.py              ⏳ SB-3  JSON + HTML comparative report
    __init__.py                  ✅ SB-1

fault_injection/
    __init__.py                  ✅ SB-1  (stub)
    fault_manager.py             ⏳ SB-2  Thread-safe fault state registry
    sensor_fault_proxy.py        ⏳ SB-2  GNSS / VIO / RADALT / EO intercept
    nav_source_proxy.py          ⏳ SB-2  TRN / VIO / IMU jitter suppression

dashboard/
    bcmp2_dashboard.py           ⏳ SB-4  Plotly Dash 7-panel GUI
    bcmp2_replay.py              ⏳ SB-4  Replay driver (4 modes)

tests/
    test_bcmp2_at1.py            ✅ SB-1  17 AT-1 gates (boot + regression)
    test_bcmp2_sb2.py            ⏳ SB-2  Fault injection scripted gates
    test_bcmp2_at2.py            ⏳ SB-3  AT-2: 150 km nominal dual-track
    test_bcmp2_at3_5.py          ⏳ SB-3  AT-3 through AT-5: failure missions
    test_bcmp2_at6.py            ⏳ SB-5  AT-6: 3× repeatability

run_bcmp2_tests.py               ✅ SB-1  Root runner (mirrors run_s5_tests.py)
```

---

## Acceptance Test Summary

| AT | Purpose | Mode | Status |
|---|---|---|---|
| AT-1 | Boot + regression, 5 km | SITL | ✅ 17/17 PASS |
| AT-2 | Nominal 150 km dual-track | SITL | ⏳ SB-3 |
| AT-3 | Single-failure mission | SITL | ⏳ SB-3 |
| AT-4 | Multi-failure mission | SITL | ⏳ SB-3 |
| AT-5 | Terminal integrity | SITL | ⏳ SB-3 |
| AT-6 | 3× repeatability / endurance | SIL | ⏳ SB-5 |

---

## Regression Baseline

Must remain green after every sprint:

| Suite | Runner | Gates | Last checked |
|---|---|---|---|
| S5 acceptance | `run_s5_tests.py` | 111/111 | 29 March 2026 — SB-1 close |
| S8 IMU / ALS-250 | `run_s8_tests.py` | 68/68 | 29 March 2026 — SB-1 close |
| BCMP-2 AT-1 | `run_bcmp2_tests.py` | 17/17 | 29 March 2026 — validated on micromind-node01 |

---

## Do Not Start SB-2 Without

- [ ] BCMP2_STATUS.md and JOURNAL.md committed to repo
- [ ] Session goal confirmed with Amit
- [ ] `run_bcmp2_tests.py` green on micromind-node01 (confirmed ✅)
- [ ] `run_s5_tests.py` green on micromind-node01 (confirmed ✅)
