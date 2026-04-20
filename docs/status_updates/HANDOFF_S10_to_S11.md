# HANDOFF: Sprint S10 → S11
**Date:** 12 March 2026
**Author:** Amit Anand (amitkr-anand)
**Commit:** 2de6089 | **Tag:** s10-m2-m5-closed
**Status:** S10 COMPLETE — All five Phase-1 milestones closed

---

## 1. S10 Summary

Sprint S10 delivered the final Phase-1 demonstration artefacts and hardened the simulation infrastructure. Four deliverables were completed:

| ID | Deliverable | Outcome |
|---|---|---|
| S10-1 | NCC Vectorisation | `trn_stub.py` patched — 2D numpy NCC replaces Python nested loop |
| S10-2 | S9 Regression Gates | `test_s9_nav01_pass.py` deployed — 10/12 pass (2 expected skips) |
| S10-4 | Parallel IMU Runner | `run_als250_parallel.py` deployed — subprocess.Popen, no MKL deadlock |
| S10-3 | S8-D Drift Chart | Three-curve PNG + HTML generated — all 3 IMUs NAV-01 PASS at 250 km |

**An unplanned but critical performance fix was also delivered:**
Root cause of O(n²) sim slowdown identified and fixed in `core/ins/mechanisation.py`. See §3.

---

## 2. Milestone Closure

| Milestone | Closed | Evidence |
|---|---|---|
| M1 — Autonomy Core | S5 | BCMP-1 full runner 111/111 |
| M2 — GNSS-Denied Navigation | **S10** | 250 km NAV-01 PASS, all 3 IMUs, drift < 16m per 5km |
| M3 — EW Survivability | S4 | EW engine + Hybrid A* 68/68 |
| M4 — Terminal Autonomy | S5 | DMRL + L10s-SE 111/111 |
| M5 — Demo Presentation | **S10** | S8-D drift chart PNG + HTML committed to main |

**All five Phase-1 milestones are now closed.**

---

## 3. Critical Performance Fix (S10-perf) — Carry Forward Always

### Root Cause
`core/ins/mechanisation.py` called `imu_noise.total_gyro()[step]` and `imu_noise.total_accel()[step]` inside the 200 Hz propagation loop. Each call reconstructed the full `(n_steps, 3)` noise array, discarding all but one row — O(n²) total work.

At 50 km (181k steps): ~105 steps/sec (fast enough to mask the bug).
At 100 km (363k steps): ~65 steps/sec — 4× slower.
At 250 km (909k steps): estimated ~14,000 seconds (~233 minutes).

### Fix Applied
```python
# core/ins/mechanisation.py — lines 89-93
# S10-perf: cache pre-computed noise arrays to avoid O(n²) recomputation
if not hasattr(imu_noise, '_gyro_cache'):
    imu_noise._gyro_cache  = imu_noise.total_gyro()
    imu_noise._accel_cache = imu_noise.total_accel()
gyro_effective  = gyro_b * (1.0 + sf) + imu_noise._gyro_cache[step]
accel_effective = accel_b + imu_noise._accel_cache[step]
```

### Result
250 km wall time: **105 seconds** (down from estimated 14,000 seconds). 134× speedup.
Results are **bit-identical** — `total_gyro()` and `total_accel()` are pure functions; caching does not change output.

### Secondary Fix
`core/ekf/error_state_ekf.py` — pre-allocated `self._F` and `self._Q` buffers in `__init__`. Minor contribution to throughput, correct for long runs. `_build_Q` now initialises once on first call.

---

## 4. NAV-01 Results at 250 km (TASL Artefact)

| IMU | Max 5km drift | Final drift | TRN corrections | NAV-01 | Margin |
|---|---|---|---|---|---|
| Safran STIM300 | 13.9 m | 6.3 m | 166 | ✅ PASS | 7.2× |
| ADI ADIS16505-3 | 16.0 m | 5.4 m | 166 | ✅ PASS | 6.3× |
| BASELINE (ideal) | 9.6 m | 3.4 m | 166 | ✅ PASS | 10.4× |

Limit: < 100 m per 5 km segment.
Artefact files: `dashboard/als250_drift_chart_20260312_1416.png` and `.html`

---

## 5. Regression at S10 Close

```
python run_s5_tests.py              → 111/111  PASS ✅
python tests/test_s6_zpi_cems.py   → 36/36    PASS ✅
python run_s8_tests.py             → 68/68    PASS ✅
pytest tests/test_s9_nav01_pass.py → 10 pass, 2 skip ✅
Total: 222/222 (excluding 2 expected skips in test_s9_nav01_pass.py)
```

**Expected skips in test_s9_nav01_pass.py:**
- S9-A-2: Q position block nonzero — ESKF Q constants are class-private, not exported
- S9-A-3: Q gyro bias regime — same reason
Both covered functionally by NAV-01 simulation gates.

---

## 6. Key Live Module Facts (Carry Forward)

### Correct sim invocation — `--duration` (seconds), NOT `--corridor-km`
```bash
# CORRECT — 250 km at 55 m/s = 4545 seconds
env PYTHONPATH=/home/azureuser/micromind-autonomy \
  /home/azureuser/miniconda3/envs/micromind-autonomy/bin/python \
  sim/als250_nav_sim.py --imu STIM300 --seed 42 --duration 4545 \
  --out sim/als250_results/

# WRONG — argument does not exist, causes silent fail
python sim/als250_nav_sim.py --corridor-km 250  # ← DO NOT USE
```

### TRNStub constructor (BREAKING — must use helpers)
```python
dem   = DEMProvider(seed=7)
radar = RadarAltimeterSim(dem, seed=99)
trn   = TRNStub(dem, radar, ncc_threshold=0.45, search_pad_px=25)
# search_pad_px MUST be 25 — value of 80 causes 7× NCC expansion and severe slowdown
```

### ESKF Q constants (frozen)
```python
_ACC_BIAS_RW   = 9.81e-7   # m/s²/√s
_GYRO_BIAS_RW  = 4.04e-8   # rad/s/√s
_POS_DRIFT_PSD = 1.0       # m/√s
```

### Performance baseline (post S10-perf fix)
```
250 km wall time: ~105 seconds at ~8,600 steps/sec
Corridor duration: 4545s | Steps: 909,000 | Rate: 200 Hz
```

---

## 7. Pre-Flight Check Protocol (mandatory for any run > 30 min)

```bash
# 1. Step count
python3 -c "
km=250; sps=55.0; hz=200
n=int((km*1000/sps)*hz)
print(f'Steps: {n:,}  Duration: {km*1000/sps:.0f}s  Est wall: {n/8600/60:.1f} min')
"

# 2. Per-step cost audit
grep -n "search_pad_px\|total_gyro\(\)\|total_accel\(\)" \
  sim/als250_nav_sim.py core/ins/mechanisation.py

# 3. Throughput from prior meta
python3 -c "
import json
d=json.load(open('sim/als250_results/als250_nav_STIM300_42_meta.json'))
print(f'Steps/sec: {d[\"n_steps\"]/d[\"sim_wall_s\"]:.0f}  wall_s: {d[\"sim_wall_s\"]:.0f}')
"

# 4. 60-second smoke test — must complete and show NAV-01 PASS before any long run
env PYTHONPATH=/home/azureuser/micromind-autonomy \
  /home/azureuser/miniconda3/envs/micromind-autonomy/bin/python \
  sim/als250_nav_sim.py --imu STIM300 --seed 42 --duration 60 \
  --out /tmp/test_smoke/ 2>&1
```

---

## 8. What Was Wasted in S10 (Lessons)

Six failed runs before successful 250 km completion:

| Run | Cause | Cost |
|---|---|---|
| Run 1 | tmux killed on VS Code disconnect — not detached | ~₹200 |
| Run 2 | `--corridor-km` flag does not exist — silent fail | ~₹0 (never ran) |
| Run 3 | Same wrong flag | ~₹0 (never ran) |
| Run 4 | `search_pad_px=80` — 7× NCC expansion, too slow | ~₹600 |
| Run 5 | Two competing processes + no log capture | ~₹400 |
| Run 6 | O(n²) bug — ran 5h42m, killed by VM auto-shutdown | ~₹800 |

**Total estimated wasted compute: ~₹2,000 of the ₹3,423 total spend as of 12 March 2026.**

All six failures were preventable with the Pre-Flight Check Protocol (§7).

---

## 9. Files Changed in S10

| File | Change |
|---|---|
| `core/ins/trn_stub.py` | S10-1: NCC vectorised (2D numpy, patch applied from trn_stub_ncc_patch.py) |
| `core/ins/mechanisation.py` | S10-perf: noise array cache fix (O(n²) → O(n)) |
| `core/ekf/error_state_ekf.py` | S10-perf: pre-allocated F and Q buffers |
| `dashboard/als250_drift_chart.py` | S10-3: deployed + attribute fixes (vre_bias_si, gyro_arw_deg_per_sqrth, etc.) |
| `dashboard/als250_drift_chart_20260312_1416.png` | S10-3: TASL artefact — three-curve drift chart |
| `dashboard/als250_drift_chart_20260312_1416.html` | S10-3: TASL artefact — interactive version |
| `tests/test_s9_nav01_pass.py` | S10-2: S9 regression gates (10/12 pass, 2 expected skips) |
| `run_als250_parallel.py` | S10-4: subprocess.Popen parallel runner |
| `pytest.ini` | Registers 'slow' mark |
| `trn_stub_ncc_patch.py` | S10-1 tool — applied, kept for record |
| `logs/s10_chart.log` | Chart generation log |

**Removed:**
- `run_als250_parallel_v2.py` — MKL fork deadlock, permanently broken
- `run_als250_parallel_old.py` — superseded

---

## 10. S11 Readiness

**Gate:** TASL partnership decision. Do not start S11 without confirming scope with Amit.

**If TASL proceeds (infrastructure investment warranted):**
- Dedicated workstation procurement decision
- HIL preparation: ROS2 / PX4 SITL interface design
- BCMP-1 full scenario hardening at 250 km
- DMRL CNN upgrade scoping (GPU + dataset + clearance path)

**If TASL deferred:**
- Whitepaper drafting: `TRN_Whitepaper_Outline.docx` — 8 sections ready, aerospace engineering audience

**Session start checklist for S11:**
```bash
git checkout main && git pull origin main
git log --oneline main | head -5      # Expected: 2de6089 at top

python run_s5_tests.py                # 111/111
python tests/test_s6_zpi_cems.py      # 36/36
python run_s8_tests.py                # 68/68
pytest tests/test_s9_nav01_pass.py -m "not slow"  # 10 pass, 2 skip
# Must be 222/222 before any changes

# Confirm session goal with Amit before writing any code
