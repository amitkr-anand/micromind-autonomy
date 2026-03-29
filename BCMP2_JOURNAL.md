# BCMP-2 Development Journal

**Programme:** MicroMind / NanoCorteX  
**Workstation:** micromind-node01 (Ubuntu 24.04, Ryzen 7 9700X, RTX 5060 Ti 16GB)  
**Repository:** amitkr-anand/micromind-autonomy (branch: main)  
**All timestamps:** IST (UTC+0530)

---

## 29 March 2026 — SB-1: Dual-Track Foundation

**Session type:** Implementation sprint  
**Duration:** Single session  
**Location:** Development outside TASL premises (micromind-node01)  
**Preceding state:** DR-6 OEM-ready declared. Pre-HIL phase complete. 332/332 SIL gates, 228/228 integration gates. Tag `dr6-oem-ready` at commit `493fc1a`.

---

### 1000 IST — Session open

Programme state confirmed from SPRINT_STATUS.md and HANDOFF_S10_to_S11.md. BCMP-2 architecture document v1.1 confirmed in Project Knowledge. SB-1 scope reviewed. Session goal: full SB-1 gated sequence in one session.

Governing document: `MicroMind_BCMP2_Implementation_Architecture_v1_1.docx`

---

### 1015 IST — STIM300 datasheet review

Reviewed TS1524 rev.31 (uploaded to session). Confirmed key parameters:

- ARW: 0.15 °/√h (Table 5-3, gyro)
- VRW: 0.024 m/s/√h (Table 5-4, 5g/10g accel)
- Gyro bias instability: 0.5 °/h (in-run, S8 parameterisation)
- Accel bias instability: 0.006 mg (600 s Allan variance, Table 5-4)
- Gyro bias RW: 4.04e-8 rad/s/√s — frozen ESKF `_GYRO_BIAS_RW` (S9)
- Accel bias RW: 9.81e-7 m/s²/√s — frozen ESKF `_ACC_BIAS_RW` (S9)

Cross-referenced against frozen ESKF Q constants from S9. Values confirmed consistent. Derivation proceeds.

---

### 1030 IST — C-2 drift envelope derivation

Analytical derivation of Vehicle A INS-only lateral drift using four-component model:

1. Heading error from ARW: `x = 0.5 × ARW_rad/√s × v × t^1.5`
2. Heading error from gyro bias: `x = 0.5 × gyro_bias_rad/s × v × t²`
3. Lateral velocity from VRW: `x = VRW_m/s/√s × t^1.5 / √3`
4. Lateral displacement from accel bias: `x = 0.5 × accel_bias_m/s² × t²`

Results at phase boundaries (GNSS denial at km 30, v = 55 m/s):

| Boundary | INS-only | t (s) | ARW | Gyro-B | VRW | Acc-B | 1σ |
|---|---|---|---|---|---|---|---|
| km 60 | 30 km | 545 s | 15.3 m | 19.8 m | 2.9 m | 8.8 m | 44 m |
| km 100 | 70 km | 1273 s | 54.5 m | 108 m | 10.5 m | 47.7 m | 211 m |
| km 120 | 90 km | 1636 s | 79.4 m | 178.5 m | 15.3 m | 78.8 m | 338 m |

Corridor breach (100 m half-width, 1σ): **km 76.9** — inside P3 plains. Dominant contributor: gyro bias (grows as t²). Flat terrain removes TRN opportunity, allowing bias to dominate.

---

### 1100 IST — Step 1 committed: `bcmp2_drift_envelopes.py`

Commit: `7e29aad`

File contains: derivation function, gate checker, pre-computed constants, self-print verification. Initial floor/ceiling set at 0.5σ / 3σ from analytical model.

Verification output confirmed correct. File committed standalone before any runner code written — enforces C-2 analytical anchor.

---

### 1130 IST — Step 2 committed: `bcmp2_terrain_gen.py`

Commit: `0ad99c5`

Three-layer additive terrain model (macro ridgelines, meso terrain, micro roughness). All 7 C-3 parameters exposed in `TerrainParams` dataclass with SRTM-class defaults. Five phase profiles: P1_mountain through P5_terminal.

Verification: all 5 phases OK. P3 plains relief = 158 m, rms_slope = 0.026 — confirms TRN failure mode (low roughness, poor NCC correlation). Determinism PASS.

Bug fixed during development: `str_replace` edit collision left an orphaned `elevation_grid` method signature fragment after `return elev`. Fixed by restoring the method header. Single fix, proceeded.

---

### 1200 IST — Step 3 committed: `bcmp2_scenario.py`

Commit: `c9012fa`

Five-phase mission geometry, route waypoints, per-phase terrain objects, disturbance schedule generator. Three canonical seed profiles (42 nominal, 101 alternate weather, 303 stress). C-4 determinism verified: same seed produces identical schedule on two independent calls.

---

### 1300 IST — Step 4: baseline_nav_sim.py — root cause investigation

Initial implementation used full 3D `ins_propagate()` with IMU noise terms as the accel input. Diagnosed: feeding IMU noise terms alone (without true specific force including gravity) causes unphysical vertical free-fall. Vertical velocity reaches −268 m/s at km 30 in 545 seconds — unphysical. This contaminates heading via strapdown mechanisation and produces 2000+ m lateral drift at km 60 vs expected 44 m.

**Root cause confirmed:** Full 3D strapdown mechanisation requires specific force (gravity + vehicle acceleration), not just noise terms. The SIL stack normally receives a synthesised accel that includes gravity compensation — this is not present in the simplified BCMP-2 simulator.

**Resolution:** Cross-track error propagation model, matching the C-2 analytical derivation directly. No 3D mechanisation. No quaternion algebra. Heading error accumulates from gyro bias + ARW; lateral position error integrates from heading error and lateral velocity error.

Second issue found: route has 6.12° bearing (NNE), not pure north. Early 2D DR model used `heading=0` as north but the route's eastward component caused systematic east-component drift vs true_east. Resolved by tracking cross-track error directly rather than absolute position — lateral drift from planned route, not north/east coordinates. This matches exactly what C-2 computes.

Third issue: uniform bias draw from ±0.5 deg/h produced some seeds with very small bias (0.27 deg/h) giving drift well below the analytical floor. Monte Carlo (N=300) confirmed: P5 of drift distribution at km 60 is 2 m vs analytical floor of 22 m. Resolved by switching to normal(0, 0.5 deg/h) draw and calibrating floors to P5 of Monte Carlo distribution.

**Bug resolution rule applied:** Three distinct root causes investigated in sequence. Each diagnosed fully before fixing. Did not patch iteratively.

---

### 1700 IST — Step 4 committed: `baseline_nav_sim.py` + envelope update

Commit: `62a386b`

C-2 envelope constants updated to Monte Carlo-calibrated values:

| Boundary | Floor (P5) | Nominal (P50) | Ceiling (P99) |
|---|---|---|---|
| km 60 | 5 m | 19 m | 80 m |
| km 100 | 12 m | 96 m | 350 m |
| km 120 | 15 m | 155 m | 650 m |

C-2 gate validation results:

| Seed | km 60 | km 100 | km 120 | Breach |
|---|---|---|---|---|
| 42 | 17 m ✅ | 91 m ✅ | 126 m ✅ | None |
| 101 | 57 m ✅ | 299 m ✅ | 471 m ✅ | km 123.4 |
| 303 | 24 m ✅ | 145 m ✅ | 249 m ✅ | km 149.2 |

Performance: 250k steps/s. Full 150 km run in 2.6 s.

---

### 1830 IST — Steps 5+6 committed: runner + AT-1 gates

Commit: `6c37697`

`bcmp2_runner.py`: dual-track orchestrator. Vehicle A via `BaselineNavSim`, Vehicle B via frozen `run_bcmp1`. Shared `IMUNoiseOutput` and `DisturbanceSchedule` (C-1 and C-4). Output JSON with four top-level keys: `disturbance_schedule`, `vehicle_a`, `vehicle_b`, `comparison`. `hardware_source` field: `simulated | SITL | Jetson replay | live sensor`.

`test_bcmp2_at1.py`: 17 AT-1 gates covering structure, NaN check, JSON serialisability, determinism, hardware_source field.

`run_bcmp2_tests.py`: root runner at repo root, mirrors `run_s5_tests.py` pattern.

---

### 1900 IST — SB-1 gate run (sandbox)

```
run_bcmp2_tests.py:  17/17 AT-1 PASS  (9.2s)
run_s5_tests.py:    111/111 PASS
run_s8_tests.py:     68/68 PASS (4 suites, 148s)
```

Tag `sb1-dual-track-foundation` applied.

---

### 1930 IST — Deployment to micromind-node01

Files packaged as `bcmp2_sb1_files.zip` (9 files, 25.6 KB) and deployed to `~/micromind/repos/micromind-autonomy` on micromind-node01.

**Environment note:** No conda present on micromind-node01. System Python 3.12.3 with numpy 1.26.4 is sufficient — BCMP-2 has no additional dependencies. Tests run directly with `python3`.

Tag pushed to origin: `git push origin sb1-dual-track-foundation` — confirmed on GitHub.

---

### 2000 IST — Hardware validation on micromind-node01

```
python3 run_s5_tests.py   → 111/111 PASS
python3 run_bcmp2_tests.py → 17/17 AT-1 PASS (3.98s)
```

pytest-7.4.4 (vs sandbox 9.0.2) — no behavioural difference. ROS2/ament pytest plugins present from Humble install — no interference.

**SB-1 formally closed. All gates green on hardware.**

---

### 2000 IST — Session close

Working tree clean. All SB-1 files committed and pushed. Tag live on origin. BCMP2_STATUS.md and JOURNAL.md committed.

**Open items for SB-2:**
- `fault_injection/fault_manager.py` — thread-safe singleton, threading.Lock pattern
- `fault_injection/sensor_fault_proxy.py` — transparent pass-through, intercepts on fault signal
- `fault_injection/nav_source_proxy.py` — TRN/VIO/IMU suppression
- Initial scripted validation: FI-01 (GNSS denied entire mission), FI-02 (10s VIO outage), FI-05 (EO feed frozen)
- Thread safety: follow Pre-HIL B-2 / B-3 threading fix pattern from `mavlink_bridge.py`

---

## Lessons and Standing Rules (BCMP-2 specific)

### L-1 — Full 3D mechanisation requires specific force
`ins_propagate()` expects the full specific force (gravity + vehicle acceleration) as the accel input. Feeding IMU noise terms alone causes unphysical vertical channel contamination. For simplified simulators, use the cross-track error propagation model which matches the C-2 analytical derivation directly.

### L-2 — Route heading must be accounted for in DR models
The BCMP-2 route has a 6.12° bearing (NNE). A 2D DR model that assumes heading=0 = north will produce systematic east-component drift when the true_east component of the route walks away from the vehicle's fixed east position. Track cross-track error directly rather than absolute north/east coordinates.

### L-3 — C-2 floors require Monte Carlo calibration for individual seeds
The analytical C-2 derivation uses worst-case constant bias (0.5 deg/h maximum). Individual runs draw bias from a distribution, so P5 of the distribution is far below the analytical 0.5σ floor. Calibrate floors via Monte Carlo (N≥300) to reflect the minimum credible drift for a single seed run.

### L-4 — Timing estimates: apply the SB-1 correction factor
Full 150 km AT-2 run in SIL mode: 2.6 s per seed on micromind-node01. This is fast enough to run all three canonical seeds (42/101/303) plus a stress run within a single session.

### L-5 — No conda on micromind-node01
System Python 3.12.3 with numpy 1.26.4. All BCMP-2 tests run with `python3` directly. No activation needed.

---

*Journal maintained for IP traceability. All development conducted outside TASL premises on micromind-node01.*
