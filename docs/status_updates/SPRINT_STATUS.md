# MicroMind / NanoCorteX — Sprint Status
**Last Updated:** 12 March 2026 (S10 COMPLETE)
**Active Sprint:** None — S10 closed. Awaiting TASL decision before S11 scope.
**GitHub:** amitkr-anand/micromind-autonomy
**Branch:** main (all sprints merged)
**Latest commit:** 2de6089 (tag: s10-m2-m5-closed)
**Environment:** conda `micromind-autonomy` / Python 3.10 / Azure VM (Standard_D8s_v5, 8 vCPU, 32 GB RAM)

---

## Project Knowledge Files (Claude Project Folder)

| File | Role |
|---|---|
| `MicroMind_V6__PART_ONE.pdf` | Operational requirement — the "why". Threat scenarios, BCMP-1 origin. |
| `MicroMind_Context_Brief.txt` | Programme overview, BCMP-1 scenario definition |
| `MicroMind_PartTwo_V7_1_Live.docx` | **Live engineering spec** — FRs, boundary conditions, DD-01/DD-02 (V7.1) |
| `MicroMind_PartTwo_TechReview_v1_1.docx` | Technical review — gap list, architectural recommendations |
| `MicroMind_Demand_Analysis.docx` | Demand and market analysis |
| `MicroMind_PartTwo_DemoEdition_v1_0.docx` | Demo Fork scope — Phase-1 bounded autonomy core |
| `SPRINT_STATUS.md` | This file |
| `HANDOFF_S10_to_S11.md` | Latest handoff — S10 closure, performance fixes, S11 readiness |

**Session reading order:** SPRINT_STATUS.md → HANDOFF_S10_to_S11.md → Part Two V7.1 (if building new modules)

---

## Sprint History

| Sprint | Commit | Gate | Delivered |
|---|---|---|---|
| S0 | 6e1c70a | ✅ | ESKF V2, quaternion math, INS mechanisation |
| S1 | 5005a5d | ✅ 9/9 | 7-state FSM, SimClock, MissionLogSchema, BCMP-1 scenario |
| S2 | e86140f | ✅ 9/9 | BIM trust scorer, GNSS spoof injector |
| S3 | 284acb4 | ✅ 8/8 | TRN stub, nav scenario, Plotly dashboard |
| S4 | 366f963 | ✅ 8/8 | EW engine, Hybrid A*, BCMP-1 EW sim |
| S5 | 7ad5db5 | ✅ 111/111 | DMRL, L10s-SE, BCMP-1 full runner |
| S6 | a7633ab | ✅ 36/36 | ZPI, CEMS, multi-UAV sim (Phase-2 frozen) |
| S7 | aa3302a | ✅ | 9-panel dashboard, HTML debrief report |
| S8 | f91180d | ✅ 68/68 | IMU models (STIM300/ADIS/BASELINE), ALS-250 corridor sim |
| S9 | 7fba53c | ✅ 215/215 | TRN+ESKF architectural fix, NAV-01 closed at 150 km |
| S10 | 2de6089 | ✅ 222/222 | NCC vectorisation, S9 regression gates, parallel runner, S8-D drift chart, 134× perf fix |

---

## Sprint S10 — Simulation Infrastructure + Demo Artefact ✅ COMPLETE
**Commit:** 2de6089 | **Tag:** s10-m2-m5-closed | **Date:** 12 March 2026

### S10 Deliverables

| ID | Deliverable | File | Gate | Status |
|---|---|---|---|---|
| S10-1 | NCC Vectorisation | `core/ins/trn_stub.py` | 68/68 S8 regression | ✅ COMPLETE |
| S10-2 | S9 Regression Test | `tests/test_s9_nav01_pass.py` | 10 pass, 2 skip | ✅ COMPLETE |
| S10-3 | S8-D Drift Chart | `dashboard/als250_drift_chart.py` | PNG + HTML, all 3 PASS | ✅ COMPLETE |
| S10-4 | Parallel IMU Runner | `run_als250_parallel.py` | 50 km smoke 3/3 PASS | ✅ COMPLETE |

### NAV-01 Results at 250 km (S10-3 — primary TASL artefact)

| IMU | Max 5km drift | Final drift | TRN corrections | NAV-01 |
|---|---|---|---|---|
| Safran STIM300 | 13.9 m | 6.3 m | 166 | ✅ PASS |
| ADI ADIS16505-3 | 16.0 m | 5.4 m | 166 | ✅ PASS |
| BASELINE (ideal) | 9.6 m | 3.4 m | 166 | ✅ PASS |

Limit: < 100 m per 5 km segment. All three models achieve >6× margin.

### S10 Performance Fix (critical — carry forward)

**Root cause:** `imu_noise.total_gyro()[step]` and `imu_noise.total_accel()[step]` were called inside the main propagation loop at 200 Hz. Each call reconstructed the full `(n_steps, 3)` noise array from scratch, then discarded all but one row — O(n²) complexity. At 100 km (363k steps) this caused 93 min wall time vs expected 9 min.

**Fix:** Cache arrays on first call in `core/ins/mechanisation.py`:
```python
# S10-perf: cache pre-computed noise arrays to avoid O(n²) recomputation
if not hasattr(imu_noise, '_gyro_cache'):
    imu_noise._gyro_cache  = imu_noise.total_gyro()
    imu_noise._accel_cache = imu_noise.total_accel()
gyro_effective  = gyro_b * (1.0 + sf) + imu_noise._gyro_cache[step]
accel_effective = accel_b + imu_noise._accel_cache[step]
```

**Result:** 134× speedup. 250 km now completes in ~105 seconds. Results are bit-identical — pure performance optimisation, zero fidelity impact.

**Secondary fix:** ESKF pre-allocated buffers in `core/ekf/error_state_ekf.py` (self._F, self._Q). Minor contribution but correct for long runs.

### Regression at S10 close
```
python run_s5_tests.py              → 111/111  PASS ✅
python tests/test_s6_zpi_cems.py   → 36/36    PASS ✅
python run_s8_tests.py             → 68/68    PASS ✅  (S8-A 16/16, S8-B 21/21 confirmed post-patch)
pytest tests/test_s9_nav01_pass.py → 10/12    PASS ✅  (2 skip: ESKF Q matrix class-private)
Total: 222/222 (excluding 2 expected skips)
```

### Milestone state after S10

| Milestone | Status |
|---|---|
| M1 — Autonomy Core | ✅ CLOSED |
| M2 — GNSS-Denied Navigation | ✅ CLOSED (250 km NAV-01 PASS, all 3 IMUs) |
| M3 — EW Survivability | ✅ CLOSED |
| M4 — Terminal Autonomy | ✅ CLOSED |
| M5 — Demo Presentation | ✅ CLOSED (S8-D three-curve drift chart) |

**All five Phase-1 milestones closed. Phase-1 demonstration package complete.**

### Files removed at S10 close
- `run_als250_parallel_v2.py` — MKL fork deadlock, permanently broken
- `run_als250_parallel_old.py` — superseded by S10-4

---

## Pre-Flight Check Protocol (mandatory for any run > 30 minutes)

Before launching any long simulation, answer all four questions:

**1. Step count:**
```bash
python3 -c "
corridor_km = 250
speed_ms = 55.0; hz = 200
n_steps = int((corridor_km * 1000 / speed_ms) * hz)
print(f'Steps: {n_steps:,}  Duration: {corridor_km*1000/speed_ms:.0f}s')
"
```

**2. Per-step cost audit — check for any O(n²) search operations:**
```bash
grep -n "search_pad_px\|\[step\]\|total_gyro\|total_accel" \
  sim/als250_nav_sim.py core/ins/mechanisation.py core/ins/trn_stub.py
```
Must confirm: `search_pad_px=25`, no unbuffered `total_gyro()[step]` calls.

**3. Throughput estimate from prior meta:**
```bash
python3 -c "
import json
d = json.load(open('sim/als250_results/als250_nav_STIM300_42_meta.json'))
sps = d['n_steps'] / d['sim_wall_s']
n = int((250 * 1000 / 55.0) * 200)
print(f'Steps/sec: {sps:.0f}  Est 250km wall: {n/sps/60:.0f} min')
"
```

**4. Correct invocation — `als250_nav_sim.py` uses `--duration` (seconds), NOT `--corridor-km`:**
```bash
# CORRECT:
python sim/als250_nav_sim.py --imu STIM300 --seed 42 --duration 4545 --out sim/als250_results/
# WRONG (silent fail — argument does not exist):
python sim/als250_nav_sim.py --corridor-km 250  # ← DO NOT USE
```

---

## Performance Notes (Updated S10)

| Configuration | Steps/sec | Wall time 250 km | Notes |
|---|---|---|---|
| After S10 perf fix (measured) | ~8,600 | ~105 sec | **Current baseline** |
| Before S10 perf fix | ~65 | ~14,000 sec | O(n²) bug |
| search_pad_px=80 (wrong) | ~15 | ~60,000 sec | 7× NCC expansion |
| verbose=True + tee over SSH | ~45 | N/A | Never use |

**Timing estimates are lower bounds, not predictions.** Always run a 60-second smoke test before committing to a full 250 km run to confirm throughput.

---

## Repository State (main branch, 12 March 2026)

```
core/
  math/quaternion.py              ✅ S0
  constants.py                    ✅ S0
  ins/state.py                    ✅ S0
  ins/mechanisation.py            ✅ S0 + S8-B + S10-perf (noise cache fix)
  ins/imu_model.py                ✅ S8-A
  ins/trn_stub.py                 ✅ S3 + S9 + S10-1 (NCC vectorised)
  ekf/error_state_ekf.py          ✅ S0 V2 + S9 (Q-matrix) + S10-perf (F/Q buffers)
  bim/bim.py                      ✅ S2
  clock/sim_clock.py              ✅ S1
  state_machine/state_machine.py  ✅ S1
  ew_engine/ew_engine.py          ✅ S4
  route_planner/hybrid_astar.py   ✅ S4
  dmrl/dmrl_stub.py               ✅ S5
  l10s_se/l10s_se.py              ✅ S5
  zpi/zpi.py                      ✅ S6
  cems/cems.py                    ✅ S6 (Phase-2, frozen)

sim/
  eskf_simulation.py              ✅ S0
  gnss_spoof_injector.py          ✅ S2
  nav_scenario.py                 ✅ S3
  bcmp1_ew_sim.py                 ✅ S4
  bcmp1_cems_sim.py               ✅ S6
  als250_nav_sim.py               ✅ S8-C + S9 (search_pad_px=25 confirmed)

scenarios/bcmp1/
  bcmp1_scenario.py               ✅ S1
  bcmp1_runner.py                 ✅ S5 + S8-E + S9-0

dashboard/
  mission_dashboard.py            ✅ S3
  bcmp1_dashboard.py              ✅ S7
  bcmp1_report.py                 ✅ S7
  als250_drift_chart.py           ✅ S10-3 (attribute fixes applied)
  als250_drift_chart_20260312_1416.png  ✅ S10-3 TASL artefact
  als250_drift_chart_20260312_1416.html ✅ S10-3 TASL artefact

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
  test_s9_nav01_pass.py           ✅ S10-2

run_s5_tests.py                   ✅ S5 (repo root)
run_s8_tests.py                   ✅ S8 (repo root)
run_als250_parallel.py            ✅ S10-4 (subprocess.Popen, no MKL deadlock)
pytest.ini                        ✅ S10 (registers 'slow' mark)
trn_stub_ncc_patch.py             ✅ S10-1 tool (applied, keep for record)
```

---

## What Is Next (S11)
**Gate:** TASL partnership decision. S11 scope TBD pending outcome.

**If TASL proceeds:** Infrastructure investment decision — dedicated workstation vs continued Azure VM. S11 likely focuses on HIL preparation, BCMP-1 full scenario hardening, and DMRL upgrade.

**If TASL deferred:** Whitepaper drafting (`TRN_Whitepaper_Outline.docx`) — 8 sections, aerospace engineering audience, ready for drafting.

**Do not start S11 without confirming scope with Amit.**

---

## Deferred (Post-TASL / HIL)
- Full CNN for DMRL (GPU + training data + ITL)
- PQC cryptography stack (FR-109–112)
- ROS2 / PX4 SITL integration
- Real RADALT hardware (post-TASL procurement)
- Cross-mission learning pipeline (DD-02 Phase 2)
- GPU/CUDA NCC (Phase-2, if Monte Carlo sweeps required)
- CEMS active use, satellite masking, predictive EW (Phase-2)
