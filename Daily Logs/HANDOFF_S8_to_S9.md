# MicroMind — Sprint Handoff: S8 → S9
**Generated:** 27 February 2026
**Outgoing sprint:** S8 — IMU Sensor Characterisation + Noise Propagation
**Incoming sprint:** S9 — TBD (scope pending TASL meeting outcome)
**Author:** Amit (amitkr-anand)

---

## 1. Outgoing Sprint: What Was Completed

Sprint S8 was an unplanned fork — not one of the three candidate forks listed in the
S7→S8 handoff. The sprint characterises real IMU sensors from published datasheets and
propagates their noise through the full SIL stack.

### Modules delivered

| File | FR | Gate | Description |
|---|---|---|---|
| `core/ins/imu_model.py` | — | S8-A ✅ | IMUModel dataclass + Allan Variance parameterisation for 3 sensors |
| `core/ins/mechanisation.py` | — | S8-B ✅ | `ins_propagate()` extended with optional IMU noise injection |
| `sim/als250_nav_sim.py` | — | S8-C ✅ | 250 km GNSS-denied corridor simulation, selectable IMU |
| `scenarios/bcmp1/bcmp1_runner.py` | — | S8-E ✅ | BCMP-1 runner extended with `imu_model` parameter |
| `tests/test_s8a_imu_model.py` | — | S8-A ✅ | 16 tests |
| `tests/test_s8b_mechanisation.py` | — | S8-B ✅ | 21 tests |
| `tests/test_s8c_als250_nav_sim.py` | — | S8-C ✅ | 17 tests |
| `tests/test_s8e_bcmp1_runner_imu.py` | — | S8-E ✅ | 14 tests |
| `run_s8_tests.py` | — | — | Master S8 runner (repo root) |

### Deferred from S8

| Item | Status | Action |
|---|---|---|
| `dashboard/als250_drift_chart.py` | 🔲 S8-D pending | Running overnight — chart once `.npy` files exist |
| V7 spec update (STIM300 ARW) | ⚠ Pending | Must update before TASL meeting |

### Test suite

```
python run_s8_tests.py      →  68/68  PASS ✅  (~50 min — S8-B is the long suite)
```

Full regression at sprint close:
```
python run_s5_tests.py              → 111/111  PASS ✅
python tests/test_s6_zpi_cems.py   → 36/36    PASS ✅
python run_s8_tests.py             → 68/68    PASS ✅
Total: 215/215
```

### Commit
```
f91180d  S8 COMPLETE: 4/4 gates PASS — IMU noise model, INS mechanisation 
             extension, ALS-250 corridor sim, BCMP-1 IMU integration [68/68 S8 
             + 111/111 S5 regression clean]
```
Branch: merged to `main`

---

## 2. Live Interfaces Added in S8

### IMU Model (`core/ins/imu_model.py`)

```python
from core.ins.imu_model import get_imu_model, generate_imu_noise, IMUModel
from core.ins.imu_model import STIM300, ADIS16505_3, BASELINE   # module-level instances

model = get_imu_model("STIM300")          # → IMUModel
noise = generate_imu_noise(model, n_steps=1000, dt=0.005, seed=42)  # → IMUNoiseSample
```

**Registry keys:** `"STIM300"`, `"ADIS16505_3"`, `"BASELINE"`
Note: `imu_model.name` returns display name (`"Safran STIM300"`), not registry key.
Use identity comparison or the known keys for logging.

### INS Propagate (`core/ins/mechanisation.py`)

```python
# S8-B extension — all new args are optional, fully backward compatible
state = ins_propagate(
    state, accel_b, gyro_b, dt,
    imu_model=model,    # optional IMUModel — if None, clean propagation
    imu_noise=noise,    # optional pre-generated IMUNoiseSample
    step=k,             # step index into noise arrays
)
```

**CRITICAL:** `GRAVITY` in `constants.py` is `np.array([0, 0, -9.80665])` — a 3-vector in ENU.
Do NOT wrap it: `np.array([0, 0, -GRAVITY])` will fail. Use `accel_nav = f_nav + GRAVITY` directly.

### ALS-250 Sim (`sim/als250_nav_sim.py`)

```python
from sim.als250_nav_sim import run_als250_sim, CORRIDOR_DURATION_S

result = run_als250_sim(
    imu_name="STIM300",           # registry key or None for clean run
    duration_s=CORRIDOR_DURATION_S,  # ~4545s for full 250 km
    seed=42,
    verbose=True,
    save_outputs=True,            # writes .npy + .json to out_dir
    out_dir="sim/als250_results/",
)
# result keys: position, drift_m, meta, NAV01_pass, n_steps, duration_s, ...
```

### BCMP-1 Runner (`scenarios/bcmp1/bcmp1_runner.py`)

```python
# S8-E entry point
from scenarios.bcmp1.bcmp1_runner import run_bcmp1, run_bcmp1_s8, BCMPResult

result = run_bcmp1(seed=42, imu_model=model, corridor_km=100.0)  # → BCMPResult

# S5 entry point — still works
from scenarios.bcmp1.bcmp1_runner import BCMP1Runner, BCMP1KPI, BCMP1RunResult
runner = BCMP1Runner(verbose=False, seed=42)
```

**BCMPResult fields:** `passed`, `criteria` (dict, 11 keys C-01 to C-11), `event_log`,
`fsm_history`, `kpi` (dict), `imu_model_name` (registry key string, "NONE" if clean)

**corridor_km parameter:** Added for fast testing (5 km in tests). Production runs use
default 100.0. Event times scale proportionally with corridor length.

---

## 3. Decisions Made in S8 (Carry Forward)

| Decision | Detail |
|---|---|
| S5 API preserved | `run_bcmp1(n_runs=5, ...)` still works via `_run_bcmp1_s5()` dispatcher. Do not remove. |
| IMU key vs display name | `imu_model.name` = display name. Use identity map or known keys for logging. |
| GRAVITY is a 3-vector | `constants.GRAVITY = np.array([0,0,-9.80665])`. Use directly, never wrap. |
| Test corridor = 5 km | All S8-E tests use `corridor_km=5.0`. Event times are scaled. Regression passes at 5 km. |
| S8-D deferred | Three-curve drift chart depends on overnight run. Chart script to be built next session. |
| STIM300 ARW finding | Typical ARW 0.15°/√hr > V7 spec floor 0.1°/√hr. Spec must be updated before TASL. |

---

## 4. Technical Finding: STIM300 ARW vs V7 Spec

**Finding:** Part Two V7 specifies IMU ARW floor as ≤ 0.1°/√hr. Safran STIM300 (the
targeted tactical-grade sensor) has a **typical ARW of 0.15°/√hr** per published datasheet.

**Impact:** STIM300 is the strongest IMU we modelled. It exceeds the spec floor on ARW,
though bias instability (0.5°/hr) remains the dominant error source over 150 km.

**Action before TASL:** Update V7 spec floor to ≤ 0.2°/√hr to accurately reflect what
tactical-grade MEMS sensors achieve in practice. This is a spec correction, not a design
regression — the system still meets navigation accuracy targets with STIM300.

---

## 5. Overnight Run — ALS-250 Full 250 km

A full 250 km simulation for all three IMU models is running overnight. When complete:

```
sim/als250_results/
  als250_nav_CLEAN_42_position.npy
  als250_nav_CLEAN_42_drift.npy
  als250_nav_CLEAN_42_meta.json
  als250_nav_STIM300_42_position.npy
  als250_nav_STIM300_42_drift.npy
  als250_nav_STIM300_42_meta.json
  als250_nav_ADIS16505_3_42_position.npy
  als250_nav_ADIS16505_3_42_drift.npy
  als250_nav_ADIS16505_3_42_meta.json
  als250_nav_BASELINE_42_position.npy
  als250_nav_BASELINE_42_drift.npy
  als250_nav_BASELINE_42_meta.json
```

Next session: build `dashboard/als250_drift_chart.py` to render the three-curve chart
from these `.npy` files.

---

## 6. Incoming Sprint: S9 Scope Options

**Status:** NOT STARTED — scope pending TASL meeting outcome

| Fork | Modules | FRs | Readiness |
|---|---|---|---|
| A — Cybersecurity hardening | `core/cybersec/` — key loading, envelope verification, PQC-ready | FR-109–112 | No blockers |
| B — DMRL CNN upgrade | Replace rule-based stub with trained CNN | FR-103 | Blocked: GPU + dataset |
| C — HIL integration prep | ROS2 node wrappers, PX4 SITL skeleton | — | Blocked: TASL platform decision |
| D — S8-D chart + S6 5x sweep | ALS-250 drift chart + CEMS clean sweep diagnostic | — | Ready once overnight run done |

---

## 7. Session Start Checklist for S9

```bash
# 1. Sync repo
git checkout main && git pull origin main
git log --oneline main | head -5

# 2. Full regression — must be clean
python run_s5_tests.py               # 111/111
python tests/test_s6_zpi_cems.py     # 36/36
python run_s8_tests.py               # 68/68

# 3. Check overnight run results
ls -lh sim/als250_results/           # 12 files expected (3 models × 3 files + clean × 3)

# 4. Confirm session goal with Amit before writing any code
```

---

## 8. End of Sprint Reminder (for whoever closes S9)

At the end of Sprint S9:
1. Save handoff as `Daily Logs/HANDOFF_S9_to_S10.md`
2. Commit and push to `main`
3. Upload to Claude Project knowledge
4. Update `SPRINT_STATUS.md` and re-upload
5. Update Project Instructions if any new interfaces or file paths were added
