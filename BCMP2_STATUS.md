# BCMP-2 Sprint Status

**Programme:** MicroMind / NanoCorteX  
**Phase:** BCMP-2 — Extended Mission Acceptance Regime  
**Repository:** amitkr-anand/micromind-autonomy (branch: main)  
**Document:** BCMP2_STATUS.md  
**Last updated:** 30 March 2026 2010 IST

---

## Governing Document

**MicroMind_BCMP2_Implementation_Architecture_v1_1.docx** (Project Knowledge)

Central question every scenario answers:
> *"What mission succeeds with MicroMind that fails without it?"*

---

## Current State

**SB-2 CLOSED** — Fault injection infrastructure complete.  
**Tag:** `sb2-fault-injection-foundation`  
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
**Commit:** `6c37697` → `c23696e` (documentation) → `36dc37c` (gitignore)

| Step | Commit | File |
|---|---|---|
| 1 | `7e29aad` | `scenarios/bcmp2/bcmp2_drift_envelopes.py` |
| 2 | `0ad99c5` | `scenarios/bcmp2/bcmp2_terrain_gen.py` |
| 3 | `c9012fa` | `scenarios/bcmp2/bcmp2_scenario.py` |
| 4 | `62a386b` | `scenarios/bcmp2/baseline_nav_sim.py` + envelope update |
| 5+6 | `6c37697` | `scenarios/bcmp2/bcmp2_runner.py` + `tests/test_bcmp2_at1.py` + `run_bcmp2_tests.py` |
| docs | `c23696e` | `BCMP2_STATUS.md` + `BCMP2_JOURNAL.md` |
| gitignore | `36dc37c` | `.gitignore` — dashboard runtime outputs excluded |

**Gate results (micromind-node01):** 17/17 AT-1 PASS · 111/111 S5 PASS · 68/68 S8 PASS

**C-2 canonical seed validation:**

| Seed | km 60 | km 100 | km 120 | Breach |
|---|---|---|---|---|
| 42 | 17 m ✅ | 91 m ✅ | 126 m ✅ | None |
| 101 | 57 m ✅ | 299 m ✅ | 471 m ✅ | km 123.4 |
| 303 | 24 m ✅ | 145 m ✅ | 249 m ✅ | km 149.2 |

---

### SB-2 — Fault Injection Infrastructure ✅ CLOSED

**Closed:** 30 March 2026 2010 IST  
**Tag:** `sb2-fault-injection-foundation`  
**Commit:** `45e3d79`

| Step | Commit | File | Self-checks |
|---|---|---|---|
| 1 | `8e05a59` | `fault_injection/fault_manager.py` | 7/7 PASS |
| 2 | `64fc1b7` | `fault_injection/sensor_fault_proxy.py` | 8/8 PASS |
| 3 | `615a2c8` | `fault_injection/nav_source_proxy.py` | 7/7 PASS |
| 4 | `45e3d79` | `tests/test_bcmp2_sb2.py` + `run_bcmp2_tests.py` | 25/25 PASS |

**Gate results (micromind-node01):**

```
run_bcmp2_tests.py:  42/42 PASS  (4.7s)
  AT-1 Boot & Regression:       17/17 PASS
  SB-2 Fault Injection Proxies: 25/25 PASS
run_s5_tests.py:    111/111 PASS
```

**SB-2 exit conditions confirmed:**
- FI-01: BIM moves toward RED on denied GNSS measurement ✅
- FI-02: VIOMode transitions NOMINAL → OUTAGE after 3s suppression ✅
- FI-05: EO stale frame returned, resumes correctly after clear ✅
- Multi-fault: PRESET_VIO_GNSS activates both simultaneously ✅
- Frozen core: ESKF `_ACC_BIAS_RW=9.81e-7`, `_GYRO_BIAS_RW=4.04e-8` unchanged ✅

**Proxy architecture:**

| Proxy | Intercepts | Transparent when |
|---|---|---|
| `SensorFaultProxy.gnss()` | Returns denied GNSSMeasurement | FI_GNSS_LOSS not active |
| `SensorFaultProxy.vio_update()` | Returns (False, 0.0) | FI_VIO_LOSS not active |
| `SensorFaultProxy.trn_correction()` | Returns None | FI_RADALT_LOSS not active |
| `SensorFaultProxy.eo_frame()` | Returns stale cached frame | FI_EO_FREEZE not active |
| `NavSourceProxy.trn_update()` | Returns None (skips TRNStub call) | FI_TERRAIN_CONF_DROP not active |
| `NavSourceProxy.vio_source_available()` | Returns False | FI_VIO_LOSS not active |
| `NavSourceProxy.dt_ticked()` | Adds ±2ms jitter to dt | FI_IMU_JITTER not active |

---

### SB-3 — Full Mission and Reports ⏳ PENDING

**Entry gate:** SB-2 closed ✅

**Deliverables:**

| File | Responsibility |
|---|---|
| Full five-phase terrain scripted | FI-01..FI-15 wired into bcmp2_runner |
| `tests/test_bcmp2_at2.py` | AT-2: 150 km nominal dual-track run |
| `tests/test_bcmp2_at3_5.py` | AT-3 through AT-5: single/multi/terminal failure |
| `scenarios/bcmp2/bcmp2_report.py` | JSON + HTML comparative report (business block first) |

**Exit condition:** AT-2 nominal run produces expected comparative outcome. Vehicle A exceeds corridor by P4. Vehicle B corridor maintained. HTML report business comparison block correct.

---

### SB-4 — Dashboard and Replay ⏳ PENDING

**Deliverables:** `dashboard/bcmp2_dashboard.py` (7-panel Plotly Dash), `dashboard/bcmp2_replay.py` (4 replay modes).

**Exit condition:** Executive replay runs 2–3 min correctly. Operator-triggered fault produces identical Vehicle B behaviour to scripted fault.

---

### SB-5 — Repeatability and Closure ⏳ PENDING

**Deliverables:** `tests/test_bcmp2_at6.py`, overnight stress run, final HTML report, BCMP-2 Closure Report.

**Exit condition:** Three consecutive seed runs (42/101/303) produce identical phase transition chains. No memory leak over 4-hour stress run.

---

## File Ownership Map

```
scenarios/bcmp2/
    bcmp2_drift_envelopes.py     ✅ SB-1
    bcmp2_terrain_gen.py         ✅ SB-1
    bcmp2_scenario.py            ✅ SB-1
    baseline_nav_sim.py          ✅ SB-1
    bcmp2_runner.py              ✅ SB-1
    bcmp2_report.py              ⏳ SB-3

fault_injection/
    __init__.py                  ✅ SB-1 (stub)
    fault_manager.py             ✅ SB-2  Thread-safe singleton, FI-01..FI-13
    sensor_fault_proxy.py        ✅ SB-2  GNSS / VIO / RADALT / EO intercept
    nav_source_proxy.py          ✅ SB-2  TRN / VIO nav source / IMU jitter

dashboard/
    bcmp2_dashboard.py           ⏳ SB-4
    bcmp2_replay.py              ⏳ SB-4

tests/
    test_bcmp2_at1.py            ✅ SB-1  17 AT-1 gates
    test_bcmp2_sb2.py            ✅ SB-2  25 fault injection gates
    test_bcmp2_at2.py            ⏳ SB-3
    test_bcmp2_at3_5.py          ⏳ SB-3
    test_bcmp2_at6.py            ⏳ SB-5

BCMP2_STATUS.md                  ✅ (this file)
BCMP2_JOURNAL.md                 ✅
run_bcmp2_tests.py               ✅ SB-1+SB-2  42 gates total
```

---

## Acceptance Test Summary

| AT | Purpose | Gates | Status |
|---|---|---|---|
| AT-1 | Boot + regression, 5 km | 17 | ✅ PASS |
| SB-2 | Fault injection proxies | 25 | ✅ PASS |
| AT-2 | Nominal 150 km dual-track | TBD | ⏳ SB-3 |
| AT-3 | Single-failure mission | TBD | ⏳ SB-3 |
| AT-4 | Multi-failure mission | TBD | ⏳ SB-3 |
| AT-5 | Terminal integrity | TBD | ⏳ SB-3 |
| AT-6 | 3× repeatability / endurance | TBD | ⏳ SB-5 |

**Total gates active: 42/42 ✅**

---

## Regression Baseline

| Suite | Runner | Gates | Last checked |
|---|---|---|---|
| S5 acceptance | `run_s5_tests.py` | 111/111 | 30 March 2026 — SB-2 close |
| S8 IMU / ALS-250 | `run_s8_tests.py` | 68/68 | 29 March 2026 — SB-1 close |
| BCMP-2 combined | `run_bcmp2_tests.py` | 42/42 | 30 March 2026 — SB-2 close |

---

## Do Not Start SB-3 Without

- [ ] BCMP2_STATUS.md and BCMP2_JOURNAL.md updated and committed
- [ ] Session goal confirmed with Amit
- [ ] `run_bcmp2_tests.py` green on micromind-node01 (42/42 ✅)
- [ ] `run_s5_tests.py` green on micromind-node01 (111/111 ✅)
