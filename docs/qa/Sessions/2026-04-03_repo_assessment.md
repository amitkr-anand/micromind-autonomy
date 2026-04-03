# Independent Repository Assessment
**Date:** 03 April 2026  
**Session type:** QA Independent Assessment (OI-04, OI-05)  
**Assessor:** Claude Code QA Agent  
**Scope:** Frozen-file integrity, TRN stub architecture audit, OpenVINS→ESKF interface documentation survey, test suite baseline confirmation

---

## 1. Frozen File Integrity

**Method:** `git log --oneline <file>` for each file; then `git log --oneline sb1-dual-track-foundation..HEAD -- <files>` to check for post-freeze modifications.

**Freeze tag:** `sb1-dual-track-foundation` → commit `493fc1a` (DR-6: OEM-ready declared, 29 March 2026 13:20 IST)

| File | Commits in log | Post-tag commits |
|---|---|---|
| `core/ekf/error_state_ekf.py` | `542867e` S-NEP-08, `ea1fe3e` S-NEP-04, `a6c6eb3` S10, `7fba53c` S9, `6e1c70a` S0 | **0 — CLEAN** |
| `core/fusion/vio_mode.py` | `542867e` S-NEP-08 | **0 — CLEAN** |
| `core/fusion/frame_utils.py` | `fe6e359` S-NEP-04 | **0 — CLEAN** |
| `core/bim/bim.py` | `e86140f` S2 | **0 — CLEAN** |
| `scenarios/bcmp1/bcmp1_runner.py` | `d3afdd1` S-NEP-10, `76f91a3` fix, `fd93299` S8, `7ad5db5` S5 | **0 — CLEAN** |

**Verdict:** All frozen files are unmodified since the `sb1-dual-track-foundation` tag. The pre-tag commits visible in `git log` (S-NEP-04, S-NEP-08, S-NEP-10, etc.) all predate the freeze declaration and are part of the frozen baseline, not violations of it.

---

## 2. TRN Stub Architecture Audit (OI-05)

**File:** `core/ins/trn_stub.py`

### 2.1 Correction Mechanism Assumptions Baked In

The stub was authored in Sprint S3 and updated to S9/S10 for ESKF integration. Its correction mechanism rests on the following assumptions, all of which reflect the **pre-AD-01 RADALT-NCC architecture**:

| Assumption | Code evidence | Value |
|---|---|---|
| Sensor is a **radar altimeter**, not a camera | `RadarAltimeterSim` class; class docstring; architecture diagram in module header | Explicit |
| Reference map is a **DEM** (terrain elevation) | `DEMProvider` class; `DEM_PIXEL_SIZE = 5.0 m` | Explicit |
| Correlation template is an **elevation strip** | `STRIP_WIDTH_PX = 20`, `STRIP_LEN_PX = 40` — radar swath geometry | Explicit |
| Fix accuracy bounded at **±1 pixel ≈ ±5 m** | NCC docstring comment | 5 m pixel |
| Measurement noise **R = 15² m²** per axis | `R_TRN_NORTH = R_TRN_EAST = 15.0**2` | 225 m² |
| Correction interval **≤ 1500 m** (NAV-01) | `CORRECTION_INTERVAL = 1500.0` | 1.5 km |
| Search radius **125 m** (25 px × 5 m/px) | `SEARCH_PAD_PX = 25` | 125 m |
| Design references: **Part Two V7 §1.7.2, FR-107, NAV-01** | Module header docstring | Pre-revision |

The module header diagram is unambiguous:

```
┌──────────────┐   radar strip   ┌─────────────┐
│ RadarAltimSim│ ──────────────> │  TRNStub    │
└──────────────┘                 │  NCC match  │
┌──────────────┐   DEM patch     │  Kalman fix │
│  DEMProvider │ ──────────────> │             │
└──────────────┘                 └──────┬──────┘
                                        │ position correction Δ(N,E)
                                        ▼
                                  INS state updated
```

There is **no camera model, no optical image model, and no satellite tile reference** anywhere in the file.

### 2.2 Conflicts with AD-01 (Navigation Architecture Decision, §4 of context)

AD-01 (settled 03 April 2026) defines the three-layer navigation stack:

| Layer | AD-01 spec | TRN stub assumption | Conflict? |
|---|---|---|---|
| L2 sensor | Nadir EO/LWIR **camera** | **Radar altimeter** | **YES — sensor type wrong** |
| L2 reference map | Preloaded **satellite imagery tiles** (orthophoto) | **DEM elevation raster** | **YES — map type wrong** |
| L2 matching domain | Optical intensity correlation | NCC on **elevation strips** | **YES — domain wrong** |
| L2 fix accuracy | MAE < **7 m** (context §4) | ±5 m per-pixel (15 m σ noise) | Marginal mismatch |
| L2 coverage | Textured terrain; route planner avoids featureless terrain | DEM texture ubiquitous (synthetic sinusoids always textured) | Masking a real gap |
| RADALT role | Terminal phase only, 0–300 m AGL, final 5 km | **Primary ingress correction mechanism** | **YES — role wrong** |
| Pixel / resolution | Orthophoto: ~0.3–0.5 m; sub-metre matching | DEM_PIXEL_SIZE = **5 m** (radar swath) | Different resolution regime |

**Summary:** The TRN stub represents the superseded RADALT-NCC architecture in its entirety. It will not serve as a valid L2 orthophoto matching stub without substantial reworking. The S9 refactor correctly moved the Kalman update out of the stub into the ESKF (measurement provider only), but the sensor model, reference map model, and feature domain are all wrong relative to AD-01.

### 2.3 What the Stub Does Correctly (Preserve on Rework)

- **Measurement-provider-only pattern** (S9 refactor): `update()` returns a `TRNCorrection` record; the caller applies the correction via `eskf.update_vio()`. This interface pattern is correct for the new architecture — only the measurement model changes.
- **CORRECTION_INTERVAL gate** at 1500 m: AD-01 specifies a hard reset every 2–5 km, so a minimum interval gate remains appropriate.
- **Threshold-gated acceptance** (`ncc_score ≥ NCC_THRESHOLD`): analogous confidence gate will be needed for orthophoto matching (match confidence score replaces NCC peak).
- **`TRNCorrection` dataclass**: captures timestamp, ground track, score, Δ(N, E), accepted flag — this schema is reusable.

### 2.4 OI-05 Assessment

OI-05 is correctly flagged HIGH and must be resolved before fusion integration. Required changes:

1. Replace `RadarAltimeterSim` with a camera/optical image acquisition model.
2. Replace `DEMProvider` (elevation sinusoids) with a satellite tile provider (e.g., GeoTIFF loader + tile cache).
3. Replace elevation-strip NCC with optical-image NCC (same algorithm, different input domain and pixel scale).
4. Update `DEM_PIXEL_SIZE` to reflect orthophoto resolution (likely 0.3–0.5 m).
5. Update `R_TRN_NORTH/EAST` to reflect orthophoto matching accuracy (MAE < 7 m → σ ≈ 3–4 m → R ≈ 9–16 m²).
6. Update design references (Part Two V7 §1.7.2 → revised AD-01 section).
7. Rename class/file to reflect orthophoto matching (e.g., `OrthophotoMatchingStub`).

---

## 3. OpenVINS → ESKF Interface Documentation Survey (OI-04)

### 3.1 What Exists

The interface is **code-complete and tested** but its specification is scattered across multiple files:

| Component | File | Description |
|---|---|---|
| **VIOReading dataclass** (output contract) | `integration/drivers/vio_driver.py` L42–56 | `pos_ned_m` (3,) NED; `cov_ned` (3×3); `t` timestamp; `frame_index`; `valid` |
| **LiveVIODriver stub notes** | `integration/drivers/vio_driver.py` L274–282 | States real impl must: subscribe to OpenVINS, apply ENU→NED, enforce IFM-01 monotonicity |
| **ESKF.update_vio()** | `core/ekf/error_state_ekf.py` L205–235 | H=[I₃|0₃ₓ₁₂], R=cov_pos_ned, Kalman update; signature: `(state, pos_ned, cov_ned) → (nis, rejected, innov_mag)` |
| **Frame rotation (frozen)** | `core/fusion/frame_utils.py` L23–63 | `R_ENU_TO_NED = [[0,1,0],[1,0,0],[0,0,-1]]`; `rotate_pos_enu_to_ned()`, `rotate_cov_enu_to_ned()`, `extract_vio_position_cov()` |
| **Interface gates T-01, T-02** | `tests/test_s_nep_04a_interface.py` | ENU→NED rotation correctness; covariance extraction from 6×6 ROS2 pose block |
| **Driver conformance G-VIO-01 to G-VIO-22** | `integration/tests/test_prehil_vio_driver.py` | 22 gates incl. G-VIO-22: "S-NEP-04 interface: update_vio accepts OfflineVIODriver output" |
| **PoseEstimate (nep-vio-sandbox ICD-05/06)** | `nep-vio-sandbox/interfaces/pose_estimate.py` | ENU position, Hamilton quaternion orientation, 6×6 row-major covariance |
| **ROS2 fusion_node** (future, S-NEP-04 Step 04-B) | `nep-vio-sandbox/fusion/fusion_node.py` L68–223 | Subscribes to `/ov_msckf/odomimu`, extracts ENU position, rotates to NED, calls `eskf.update_vio()` |
| **Offline replay harness** | `nep-vio-sandbox/fusion/run_04b_offline.py` | Same code path as fusion_node, EuRoC CSV transport |

**Synthesised data-flow:**

```
OpenVINS /ov_msckf/odomimu (nav_msgs/Odometry, ENU)
  ↓  msg.pose.pose.position → (x_enu, y_enu, z_enu)
  ↓  msg.pose.covariance[0:36] → 6×6 ROS covariance
extract_vio_position_cov()  → cov_pos_enu (3×3)  [IFM-04: reject zero diagonal]
rotate_pos_enu_to_ned()     → pos_ned
rotate_cov_enu_to_ned()     → cov_ned
IFM-01 monotonicity check   → reject if t_s ≤ t_last
eskf.update_vio(state, pos_ned, cov_ned)  → (nis, rejected, innov_mag)
if not rejected: eskf.inject(state)
```

### 3.2 What Is Missing

| Gap | Severity |
|---|---|
| **No single authoritative interface spec document** — interface defined piecewise across code comments, test assertions, docstrings, no central ICD | HIGH — OI-04 |
| **fusion_node.py not in micromind-autonomy** — lives in nep-vio-sandbox, not yet migrated to main autonomy tree | HIGH — needed for S-NEP-04 |
| **No ROS2 launch file** for fusion node in either repo | MEDIUM |
| **No real OpenVINS subscription code** in main repo — `LiveVIODriver` is a stub placeholder | HIGH — hardware integration |
| **IFM-01/IFM-02/IFM-04 fault codes not defined in a central spec** — scattered across vio_driver.py, frame_utils.py, fusion_node.py | MEDIUM |
| **No ENU→NED frame diagram** — rotation matrix is in frozen code but no visual documentation | LOW |
| **No example bag file** showing real `/ov_msckf/odomimu` output | LOW |

### 3.3 OI-04 Assessment

OI-04 is correctly flagged HIGH and must be resolved before S-NEP-04 starts. The recommended deliverable is a single document at `docs/OpenVINS_ESKF_Interface_Spec.md` consolidating:
- Coordinate frame conventions (ENU→NED diagram)
- ROS2 message structure and field mapping
- Fault modes (IFM-01, IFM-02, IFM-04) with definitions
- ESKF update signature and expected invariants
- Test gates (T-01, T-02, G-VIO-22) cited by ID
- Reference to frozen files that implement the interface

This document does not exist today.

---

## 4. Test Suite Baseline

**Environment note:** The default shell Python on this machine is `miniconda3/bin/python` (Python 3.13.12) with no scientific packages. Tests must be run under `conda activate micromind-autonomy` (Python 3.11). Two packages (`pyyaml`, `lark`) were absent from the conda environment at session start and were installed to resolve BCMP-2 test runner failures. This represents **environment drift** since the last test run — should be documented in `requirements.txt` or `environment.yml`.

| Suite | Command | Result | Count |
|---|---|---|---|
| S5 regression | `python run_s5_tests.py` | **PASS** | 111 / 111 |
| S8 IMU characterisation | `python run_s8_tests.py` | **PASS** | 4/4 suites (68/68 gates) |
| BCMP-2 full mission | `python run_bcmp2_tests.py` | **PASS** | 4/4 suites (90/90 gates) |

All three baselines are green. The programme is in a stable regression state.

---

## 5. Summary of Findings

| Finding | Severity | OI reference |
|---|---|---|
| Frozen files: all CLEAN since sb1-dual-track-foundation tag | — (no issue) | — |
| `trn_stub.py` assumes RADALT + DEM-NCC, conflicts with AD-01 orthophoto-matching decision on sensor type, map type, and role | **HIGH** | OI-05 |
| Measurement noise R=15²m² may overstate error (AD-01 targets MAE <7m) | MEDIUM | OI-05 |
| Synthetic DEM always textured — masks the featureless-terrain failure mode that route planner must handle | MEDIUM | OI-05, OI-08 |
| No central OpenVINS→ESKF interface spec document | **HIGH** | OI-04 |
| fusion_node.py (ROS2 integration) not yet migrated to main autonomy repo | **HIGH** | OI-04 |
| Environment drift: pyyaml and lark absent from conda env | LOW | — |
| Test suites: 111/111 + 68/68 + 90/90 — all green | — (no issue) | — |

---

*Assessment complete. No code modified during this session.*
