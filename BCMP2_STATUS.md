# BCMP-2 Sprint Status

**Programme:** MicroMind / NanoCorteX  
**Phase:** BCMP-2 — Extended Mission Acceptance Regime  
**Repository:** amitkr-anand/micromind-autonomy (branch: main)  
**Document:** BCMP2_STATUS.md  
**Last updated:** 30 March 2026 2300 IST

---

## Governing Document

**MicroMind_BCMP2_Implementation_Architecture_v1_1.docx** (Project Knowledge)

Central question:
> *"What mission succeeds with MicroMind that fails without it?"*

---

## Current State

**SB-3 CLOSED** — Full mission and reports complete.  
**Tag:** `sb3-full-mission-reports` · **Commit:** `2352382`  
**Total gates active: 90/90 ✅ — validated on micromind-node01**

---

## Fixed Constraints

| ID | Constraint | Summary |
|---|---|---|
| C-1 | IMU Parity | Both vehicles use identical IMU class, noise profile, seed. |
| C-2 | Drift Envelopes | Monte Carlo calibrated (N=300). Floors P5, ceilings P99. |
| C-3 | Synthetic Terrain | Committed, offline, reproducible. 7 documented parameters. |
| C-4 | Disturbance Parity | One schedule per run from shared seed, serialised at top level. |

**C-2 Phase-Boundary Envelopes:**

| Boundary | Floor | Nominal | Ceiling |
|---|---|---|---|
| km 60 | 5 m | 19 m | 80 m |
| km 100 | 12 m | 96 m | 350 m |
| km 120 | 15 m | 155 m | 650 m |

---

## Frozen Baseline — Zero Modifications Permitted

- `core/ekf/error_state_ekf.py`
- `core/fusion/vio_mode.py`
- `core/fusion/frame_utils.py`
- `core/bim/bim.py`
- `scenarios/bcmp1/bcmp1_runner.py`
- All 332 SIL gates

---

## Sprint History

### SB-1 ✅ CLOSED — 29 March 2026
**Tag:** `sb1-dual-track-foundation` · **Final commit:** `36dc37c`  
Gates: 17/17 AT-1 · 111/111 S5 · 68/68 S8  
Key files: `bcmp2_drift_envelopes.py`, `bcmp2_terrain_gen.py`, `bcmp2_scenario.py`, `baseline_nav_sim.py`, `bcmp2_runner.py`, `test_bcmp2_at1.py`, `run_bcmp2_tests.py`

### SB-2 ✅ CLOSED — 30 March 2026 2010 IST
**Tag:** `sb2-fault-injection-foundation` · **Commit:** `45e3d79`  
Gates: 25/25 · Cumulative: 42/42  
Key files: `fault_manager.py`, `sensor_fault_proxy.py`, `nav_source_proxy.py`, `test_bcmp2_sb2.py`

### SB-3 ✅ CLOSED — 30 March 2026
**Tag:** `sb3-full-mission-reports` · **Commit:** `2352382`  
Gates: 29/29 AT-2 + 19/19 AT-3/4/5 · Cumulative: **90/90**

| Step | Commit | File |
|---|---|---|
| 1 | `23b665f` | `scenarios/bcmp2/bcmp2_report.py` |
| 2+3 | `2352382` | `tests/test_bcmp2_at2.py` + `tests/test_bcmp2_at3_5.py` + `run_bcmp2_tests.py` |

**micromind-node01 validation (30 March 2026):**

```
run_bcmp2_tests.py:  90/90 PASS  (31.3s)
  AT-1 Boot & Regression:       17/17 PASS  (3.95s)
  SB-2 Fault Injection Proxies: 25/25 PASS  (0.16s)
  AT-2 Nominal 150 km:          29/29 PASS  (24.28s)
  AT-3/4/5 Failure Missions:    19/19 PASS  (2.31s)
run_s5_tests.py:    111/111 PASS
```

**AT-2 canonical seed results (micromind-node01):**

| Seed | km 60 | km 100 | km 120 | C-2 | Breach |
|---|---|---|---|---|---|
| 42 | 17 m ✅ | 91 m ✅ | 126 m ✅ | PASS | None |
| 101 | 57 m ✅ | 299 m ✅ | 471 m ✅ | PASS | km 123.4 ✅ |

**Open cosmetic item:** `datetime.utcnow()` deprecation warning in `bcmp2_report.py` line 420 (Python 3.12). Non-functional. Fix in next session touching the file.

**SB-3 exit conditions confirmed:**
- AT-2: Vehicle A C-2 gates PASS for seeds 42 and 101 ✅
- AT-2: Seed 101 corridor breach demonstrated by km 150 ✅
- AT-2: Report business comparison block before technical tables ✅
- AT-3: Single fault — Vehicle A unaffected by proxy ✅
- AT-4: Multi-fault — C-2 gates unaffected by proxy chain ✅
- AT-5: Frozen core unchanged after full proxy chain ✅

---

### SB-4 — Dashboard and Replay ⏳ PENDING

**Entry gate:** SB-3 closed ✅  

**Architecture note:** Plotly/Dash not installed on micromind-node01. Existing programme pattern uses matplotlib (see `dashboard/bcmp1_dashboard.py`). Implement SB-4 in matplotlib — self-contained HTML output, no external dependencies, air-gap safe. Matches existing programme convention.

**Deliverables:**

| File | Responsibility |
|---|---|
| `dashboard/bcmp2_dashboard.py` | 7-panel matplotlib dashboard. Panel 7 outcome always visible. Static PNG + self-contained HTML. |
| `dashboard/bcmp2_replay.py` | 4 replay modes: executive (2–3 min) / technical / high-fidelity / overnight. CLI `--mode` argument. |

**Exit condition:** Executive replay generates correctly. Panel 7 outcome summary correct. Operator fault injection CLI produces same Vehicle B behaviour as scripted fault.

---

### SB-5 — Repeatability and Closure ⏳ PENDING

**Deliverables:** `tests/test_bcmp2_at6.py` (seeds 42/101/303), overnight stress, final HTML report, BCMP-2 Closure Report.

---

## File Ownership Map

```
scenarios/bcmp2/
    bcmp2_drift_envelopes.py     ✅ SB-1
    bcmp2_terrain_gen.py         ✅ SB-1
    bcmp2_scenario.py            ✅ SB-1
    baseline_nav_sim.py          ✅ SB-1
    bcmp2_runner.py              ✅ SB-1
    bcmp2_report.py              ✅ SB-3  ⚠ utcnow() cosmetic fix pending

fault_injection/
    __init__.py                  ✅ SB-1
    fault_manager.py             ✅ SB-2
    sensor_fault_proxy.py        ✅ SB-2
    nav_source_proxy.py          ✅ SB-2

dashboard/
    bcmp2_dashboard.py           ⏳ SB-4  matplotlib, 7 panels
    bcmp2_replay.py              ⏳ SB-4  4 replay modes, CLI

tests/
    test_bcmp2_at1.py            ✅ SB-1  17 gates
    test_bcmp2_sb2.py            ✅ SB-2  25 gates
    test_bcmp2_at2.py            ✅ SB-3  29 gates
    test_bcmp2_at3_5.py          ✅ SB-3  19 gates
    test_bcmp2_at6.py            ⏳ SB-5

BCMP2_STATUS.md                  ✅
BCMP2_JOURNAL.md                 ✅
run_bcmp2_tests.py               ✅  4 suites, 90 gates
```

---

## Acceptance Test Summary

| AT | Purpose | Gates | Status |
|---|---|---|---|
| AT-1 | Boot + regression, 5 km | 17 | ✅ PASS |
| SB-2 | Fault injection proxies | 25 | ✅ PASS |
| AT-2 | Nominal 150 km dual-track | 29 | ✅ PASS |
| AT-3 | Single-failure mission | 7 | ✅ PASS |
| AT-4 | Multi-failure mission | 6 | ✅ PASS |
| AT-5 | Terminal integrity | 6 | ✅ PASS |
| AT-6 | 3× repeatability / endurance | TBD | ⏳ SB-5 |
| **Total** | | **90** | ✅ |

---

## Regression Baseline

| Suite | Gates | Last verified on hardware |
|---|---|---|
| S5 acceptance | 111/111 | 30 March 2026 — SB-3 close |
| S8 IMU / ALS-250 | 68/68 | 29 March 2026 — SB-1 close |
| BCMP-2 combined | 90/90 | 30 March 2026 — SB-3 close |

---

## SB-4 Entry Checklist

- [x] SB-3 committed and tagged on micromind-node01 (`2352382`, `sb3-full-mission-reports`)
- [x] BCMP2_STATUS.md + JOURNAL committed
- [x] `run_bcmp2_tests.py` green on micromind-node01 (90/90)
- [x] `run_s5_tests.py` green on micromind-node01 (111/111)
- [x] Dashboard implementation approach confirmed: matplotlib (matches bcmp1_dashboard.py)
- [ ] Session goal confirmed with Amit
