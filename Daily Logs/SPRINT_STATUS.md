# MicroMind / NanoCorteX — Sprint Status
**Last Updated:** 27 February 2026
**Active Sprint:** S9 (scope TBD — pending TASL meeting outcome)
**GitHub:** amitkr-anand/micromind-autonomy
**Branch:** main (all sprints merged)
**Latest commit:** f91180d
**Environment:** conda `micromind-autonomy` / Python 3.10 / macOS Ventura

---

## Project Knowledge Files (Claude Project Folder)

| File | Role |
|---|---|
| `MicroMind_V6__PART_ONE.pdf` | Operational requirement — the "why". User needs, threat scenarios, BCMP-1 origin. |
| `MicroMind_Context_Brief.txt` | Programme overview, BCMP-1 scenario definition, operational context summary |
| `MicroMind_PartTwo_V7_Live.docx` | **Live engineering spec** — FRs, boundary conditions, SRS, design decisions DD-01/DD-02 |
| `MicroMind_PartTwo_TechReview_v1_1.docx` | Technical review — alignment analysis, gap list, architectural recommendations |
| `MicroMind_Demand_Analysis.docx` | Demand and market analysis |
| `SPRINT_STATUS.md` | This file — current sprint state, acceptance gates, decisions |
| `HANDOFF_S8_to_S9.md` | Latest handoff — S8 deliverables, S9 scope options |

**Reading priority for any new session:**
1. SPRINT_STATUS.md — understand where we are right now
2. Latest HANDOFF file in Daily Logs — sprint-to-sprint context
3. Part Two V7 — boundary conditions and FRs for whatever module is being built

---

## Sprint S0 — Foundation ✅ COMPLETE
**Commit:** 6e1c70a

### Delivered
- `core/math/quaternion.py` — quat_multiply, quat_rotate, quat_from_gyro, quat_normalize
- `core/constants.py` — GRAVITY constant (ENU frame, 3-vector [0, 0, -9.80665])
- `core/ins/state.py` — INSState dataclass
- `core/ins/mechanisation.py` — ins_propagate
- `core/ekf/error_state_ekf.py` — 15-state ESKF V2
- `sim/eskf_simulation.py` — 3-scenario simulation

### Acceptance gate: PASSED ✅

---

## Sprint S1 — Architecture Shell ✅ COMPLETE
**Commit:** 5005a5d

### Delivered
- `core/state_machine/state_machine.py` — 7-state FSM
- `core/clock/sim_clock.py` — simulation timestep manager
- `logs/mission_log_schema.py` — learning-field-aware schema (DD-02 Phase 1)
- `scenarios/bcmp1/bcmp1_scenario.py` — 100 km corridor, 2 jammer events, satellite overpass

### Acceptance gate: PASSED — 9/9 ✅

---

## Sprint S2 — BIM ✅ COMPLETE
**Commit:** e86140f

### Delivered
- `core/bim/bim.py` — GNSS trust scorer; G/A/R state; 3-sample hysteresis (FR-101)
- `sim/gnss_spoof_injector.py` — simulated GNSS position offset injection

### Acceptance gate: PASSED — 9/9 ✅

---

## Sprint S3 — Navigation + Dashboard ✅ COMPLETE
**Commit:** 284acb4

### Delivered
- `core/ins/trn_stub.py` — TRN Kalman correction stub (NCC terrain matching)
- `sim/nav_scenario.py` — 50 km corridor with GNSS loss event
- `dashboard/mission_dashboard.py` — Plotly Dash live display (S3 artefact — do not modify)

### Acceptance gate: PASSED — 8/8 ✅

---

## Sprint S4 — EW Engine + Route Planner ✅ COMPLETE
**Commit:** 366f963

### Delivered
- `core/ew_engine/ew_engine.py` — jammer hypothesis, DBSCAN clustering, EW cost map
- `core/route_planner/hybrid_astar.py` — Hybrid A* with EW cost overlay
- `sim/bcmp1_ew_sim.py` — 2 jammer nodes, 2 mandatory replans

### Acceptance gate: PASSED — 8/8 ✅

---

## Sprint S5 — Terminal Guidance + BCMP-1 Demo ✅ COMPLETE
**Commit:** 7ad5db5

### Delivered
- `core/dmrl/dmrl_stub.py` — EO lock confidence, rule-based decoy rejection (FR-103)
- `core/l10s_se/l10s_se.py` — deterministic abort/continue decision tree (FR-105)
- `scenarios/bcmp1/bcmp1_runner.py` — full end-to-end BCMP-1 runner (all 11 criteria)
- `tests/test_s5_dmrl.py` / `test_s5_l10s_se.py` / `test_s5_bcmp1_runner.py`
- `run_s5_tests.py` — master test runner (repo root)

### Acceptance gate: PASSED — 111/111 ✅

---

## Sprint S6 — CEMS + ZPI Multi-UAV ✅ COMPLETE
**Commit:** a7633ab
**Date completed:** 22 February 2026

### Delivered
- `core/zpi/zpi.py` — ZPI Burst Scheduler: HKDF-SHA256 hop plan, DF adaptation, SHM suppression (FR-104)
- `core/cems/cems.py` — CEMS Engine: spatial-temporal merge, auth validator, replay protection (FR-102)
- `sim/bcmp1_cems_sim.py` — multi-UAV BCMP-1 sim: 2 UAVs, shared EW picture, route replans
- `tests/test_s6_zpi_cems.py` — 36 tests (16 ZPI + 20 CEMS)

### Acceptance gate: PASSED — 36/36 + 7/7 CEMS criteria ✅

---

## Sprint S7 — Dashboard + Mission Debrief Report ✅ COMPLETE
**Commit:** aa3302a
**Date completed:** 22 February 2026

### Delivered
- `dashboard/bcmp1_dashboard.py` — 9-panel full-stack mission dashboard (S0–S6 in one view)
- `dashboard/bcmp1_report.py` — self-contained HTML mission debrief report generator

### Acceptance gate: PASSED ✅
- 111/111 (S5) + 36/36 (S6) regression clean
- Dashboard renders all 9 panels without error or warnings
- HTML report self-contained, no external dependencies

---

## Sprint S8 — IMU Sensor Characterisation + Noise Propagation ✅ COMPLETE
**Commit:** f91180d
**Date completed:** 27 February 2026

### What S8 built
Sprint S8 was an unplanned fork — not one of the three candidate forks in the handoff.
It characterises real IMU sensors from published datasheets and propagates their noise
through the full SIL stack, enabling sensor-grade-specific BCMP-1 performance claims.

### Modules delivered

| File | Description | Gate |
|---|---|---|
| `core/ins/imu_model.py` | IMUModel dataclass + 3 sensor profiles (STIM300, ADIS16505-3, BASELINE) | S8-A ✅ |
| `core/ins/mechanisation.py` | `ins_propagate()` extended with optional IMU noise injection | S8-B ✅ |
| `sim/als250_nav_sim.py` | 250 km GNSS-denied corridor sim with selectable IMU model | S8-C ✅ |
| `scenarios/bcmp1/bcmp1_runner.py` | BCMP-1 runner extended with `imu_model` parameter; S5 API preserved | S8-E ✅ |

### Sensor profiles (core/ins/imu_model.py)

| Key | Sensor | Grade | ARW (°/√hr) | Gyro Bias (°/hr) |
|---|---|---|---|---|
| `STIM300` | Safran STIM300 | Tactical | 0.15 (typical) | 0.5 |
| `ADIS16505_3` | Analog Devices ADIS16505-3 | MEMS | 0.22 | 8.0 |
| `BASELINE` | Simplified (S0–S7 equivalent) | — | 0.05 | 0.3 |

**⚠ Spec finding:** STIM300 typical ARW of 0.15°/√hr exceeds V7 spec floor of 0.1°/√hr.
Action required: update Part Two V7 spec to ≤ 0.2°/√hr before TASL meeting.

### Test suite

| Suite | File | Tests | Result |
|---|---|---|---|
| IMU noise model | `tests/test_s8a_imu_model.py` | 16 | ✅ PASS |
| INS mechanisation | `tests/test_s8b_mechanisation.py` | 21 | ✅ PASS |
| ALS-250 nav sim | `tests/test_s8c_als250_nav_sim.py` | 17 | ✅ PASS |
| BCMP-1 IMU ext. | `tests/test_s8e_bcmp1_runner_imu.py` | 14 | ✅ PASS |
| **S8 total** | | **68** | **✅ 68/68** |

Master runner: `PYTHONPATH=. python run_s8_tests.py`

### Key interfaces added (S8)

| Interface | Signature | Notes |
|---|---|---|
| `get_imu_model(name)` | `→ IMUModel` | Keys: `"STIM300"`, `"ADIS16505_3"`, `"BASELINE"` |
| `generate_imu_noise(model, n, dt, seed)` | `→ IMUNoiseSample` | Module-level function |
| `ins_propagate(..., imu_model, imu_noise, step)` | Extended | Backward compatible — all S8-B args optional |
| `run_als250_sim(imu_name, duration_s, seed)` | `→ dict` | `duration_s` defaults to full 250 km |
| `run_bcmp1(seed, kpi_log_path, imu_model, corridor_km, imu_name)` | `→ BCMPResult` | S5 `run_bcmp1(n_runs, ...)` still works via `_run_bcmp1_s5()` |
| `run_bcmp1_s8(...)` | `→ BCMPResult` | Direct S8-E entry point |

### Backward compatibility
- `ins_propagate()` — 147/147 pre-S8 tests pass unchanged
- `run_bcmp1(n_runs=5, seed=42, ...)` — S5 callers still work via dispatcher
- `BCMP1Runner`, `BCMP1KPI`, `BCMP1RunResult` — all S5 symbols preserved

### Deferred from S8 (S8-D)
- `dashboard/als250_drift_chart.py` — three-curve TASL chart (STIM300 vs ADIS vs BASELINE)
- Full 250 km run for all three models is running overnight (Option B)
- Chart will be generated from cached `.npy` files once overnight run completes
- See `sim/run_als250_overnight.sh` for run commands

### Acceptance gate: PASSED — 68/68 ✅
Full regression clean: 111/111 (S5) + 36/36 (S6) + 68/68 (S8) = **215/215 tests**

---

## Full Regression State (27 Feb 2026 — post S8)

```
python run_s5_tests.py                    → 111/111  PASS ✅
python tests/test_s6_zpi_cems.py          → 36/36    PASS ✅
python run_s8_tests.py                    → 68/68    PASS ✅
```

Total tests on main: **215/215** passing
BCMP-1 acceptance: **5/5 runs × 11/11 criteria** every run
CEMS acceptance: **7/7 criteria** passing
S8 IMU gate: **4/4 gates** passing

---

## Repository State (main branch, 27 Feb 2026)

```
core/
  math/quaternion.py              ✅ S0
  constants.py                    ✅ S0  (GRAVITY = np.array([0,0,-9.80665]) — ENU 3-vector)
  ins/state.py                    ✅ S0
  ins/mechanisation.py            ✅ S0 + S8-B (IMU noise injection, backward compat)
  ins/imu_model.py                ✅ S8-A (IMUModel, STIM300, ADIS16505_3, BASELINE)
  ins/trn_stub.py                 ✅ S3
  ekf/error_state_ekf.py          ✅ S0 V2
  bim/bim.py                      ✅ S2
  clock/sim_clock.py              ✅ S1
  state_machine/state_machine.py  ✅ S1
  ew_engine/ew_engine.py          ✅ S4
  route_planner/hybrid_astar.py   ✅ S4
  dmrl/dmrl_stub.py               ✅ S5
  l10s_se/l10s_se.py              ✅ S5
  zpi/zpi.py                      ✅ S6
  cems/cems.py                    ✅ S6

sim/
  eskf_simulation.py              ✅ S0
  gnss_spoof_injector.py          ✅ S2
  nav_scenario.py                 ✅ S3
  bcmp1_ew_sim.py                 ✅ S4
  bcmp1_cems_sim.py               ✅ S6
  als250_nav_sim.py               ✅ S8-C (250 km corridor, selectable IMU)

scenarios/bcmp1/
  bcmp1_scenario.py               ✅ S1
  bcmp1_runner.py                 ✅ S5 + S8-E (IMU extension, S5 API preserved)

dashboard/
  mission_dashboard.py            ✅ S3 (nav scenario — do not modify)
  bcmp1_dashboard.py              ✅ S7
  bcmp1_report.py                 ✅ S7
  als250_drift_chart.py           🔲 S8-D (pending overnight run completion)

logs/mission_log_schema.py        ✅ S1

tests/
  test_sprint_s1_acceptance.py    ✅ S1
  test_sprint_s2_acceptance.py    ✅ S2
  test_sprint_s3_acceptance.py    ✅ S3
  test_sprint_s4_acceptance.py    ✅ S4
  test_s5_dmrl.py                 ✅ S5
  test_s5_l10s_se.py              ✅ S5
  test_s5_bcmp1_runner.py         ✅ S5
  test_s6_zpi_cems.py             ✅ S6
  test_s8a_imu_model.py           ✅ S8-A
  test_s8b_mechanisation.py       ✅ S8-B
  test_s8c_als250_nav_sim.py      ✅ S8-C
  test_s8e_bcmp1_runner_imu.py    ✅ S8-E

run_s5_tests.py                   ✅ S5 (repo root)
run_s8_tests.py                   ✅ S8 (repo root)

Daily Logs/
  HANDOFF_S5_to_S6.md             ✅
  HANDOFF_S6_to_S7.md             ✅
  HANDOFF_S7_to_S8.md             ✅
  HANDOFF_S8_to_S9.md             ✅ (generated this session)
  README_2026-02-21_S5_Complete.md ✅
  README_2026-02-22_S7_Complete.md ✅
  README_2026-02-27_S8_Complete.md ✅ (generated this session)
```

---

## Sprint S9 — Scope TBD 🔲 NOT STARTED
**Trigger:** TASL meeting outcome + overnight ALS-250 run completion

### Immediate next action (morning after overnight run)
```bash
# Check overnight run completed
ls -la sim/als250_results/
# Should contain: als250_nav_*_position.npy, *_drift.npy, *_meta.json for all 3 models

# Generate drift chart
PYTHONPATH=. python dashboard/als250_drift_chart.py --results-dir sim/als250_results/

# Commit chart
git add dashboard/als250_drift_chart.py dashboard/als250_drift_*.png
git commit -m "S8-D: ALS-250 three-curve drift chart from overnight run"
```

### Candidate forks for S9

| Fork | Modules | FRs | Readiness |
|---|---|---|---|
| A — Cybersecurity hardening | `core/cybersec/` — key loading, envelope verification, PQC-ready stack | FR-109–112 | Architecture ready, no blockers |
| B — DMRL CNN upgrade | Replace rule-based stub with trained CNN — Hailo-8 target | FR-103 | Blocked: GPU + training data + Indigenous Threat Library clearance |
| C — HIL integration prep | ROS2 node wrappers, PX4 SITL skeleton | — | Blocked: hardware platform decision from TASL |
| D — CEMS S6 clean sweep | Resolve deferred 5× clean sweep validation | — | Pending diagnostic |

### Session start checklist for S9
```bash
git checkout main && git pull origin main
git log --oneline main | head -5

python run_s5_tests.py               # must be 111/111
python tests/test_s6_zpi_cems.py     # must be 36/36
python run_s8_tests.py               # must be 68/68

# Check overnight ALS-250 results exist
ls sim/als250_results/
```

---

## Deferred (Post-TASL / HIL Phase)
- Full CNN for DMRL (requires GPU + training data + Indigenous Threat Library)
- PQC cryptography stack (FR-109–112) — S9 candidate
- ROS2 node wrapping — HIL phase
- Real RADALT hardware — sensor procurement after TASL partnership
- Cross-mission learning pipeline (DD-02 Phase 2) — post-HIL
- **V7 spec update:** STIM300 ARW floor must be updated to ≤ 0.2°/√hr (currently 0.1°/√hr — exceeded by real sensor)
