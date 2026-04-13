# MicroMind Pre-HIL Navigation Performance Specification

**Date:** 13 April 2026  
**Terrain:** Copernicus GLO-30 (30 m resolution, EPSG:4326)  
**IMU model:** STIM300 ARW 0.15°/√hr  
**Method:** Monte Carlo N=300, AD-16 methodology  
**Corridor:** Shimla–Manali, 180 km, three terrain zones  
**Classification:** Programme Confidential  

---

## Purpose

This document states what MicroMind has proven in software (SIL) before
hardware integration begins. It is the entry specification for HIL
commissioning. The remaining question at HIL is not whether MicroMind
works — it is whether the real hardware produces data in the format,
quality, timing, and rate that MicroMind already expects.

---

## Navigation Performance

Monte Carlo N=300 seeds, STIM300 ARW 0.15°/√hr model, DRIFT_PSD=1.5 m/√s
per axis, σ_TRN=25 m residual accuracy, TRN interval=5 km.

GNSS denial starts at km 10. Corridor: full 180 km Shimla–Manali.

| km  | P99 INS-only | P99 TRN   | P99 VIO+TRN | TRN reduction |
|-----|-------------|-----------|-------------|---------------|
|  30 |    123.5 m  |  104.4 m  |    92.3 m   |      15.5%    |
|  60 |    187.3 m  |   77.3 m  |    77.3 m   |      58.7%    |
|  90 |    240.3 m  |   75.4 m  |    75.4 m   |      68.6%    |
| 120 |    265.9 m  |   75.2 m  |    75.2 m   |      71.7%    |
| 150 |    322.6 m  |  115.4 m  |    96.4 m   |      64.2%    |
| 180 |    372.6 m  |   76.2 m  |    76.2 m   |      79.5%    |

**At 180 km:** P99 TRN = 76.2 m, P99 VIO+TRN = 76.2 m, INS-only = 372.6 m.  
TRN achieves 79.5% P99 drift reduction at mission end.

---

## Terrain Observability Profile

Sampled at 10 km intervals using Copernicus GLO-30, hillshade Laplacian
texture scoring, HillshadeGenerator.generate() method.

| km  | Score | Recommendation | Relief (m) | Texture var |
|-----|-------|---------------|-----------|-------------|
|   0 | 0.572 | CAUTION       |  222 m    |  68.6       |
|  10 | 0.594 | CAUTION       |  258 m    | 117.8       |
|  20 | 0.574 | CAUTION       |  200 m    |  72.0       |
|  30 | 0.000 | SUPPRESS      |  660 m    |   5.8       |
|  40 | 0.613 | ACCEPT        |  239 m    | 160.5       |
|  50 | 0.571 | CAUTION       |  284 m    |  65.5       |
|  60 | 0.598 | CAUTION       |  256 m    | 126.4       |
|  70 | 0.580 | CAUTION       |  241 m    |  85.8       |
|  80 | 0.482 | CAUTION       |  152 m    | 104.2       |
|  90 | 0.583 | CAUTION       |  294 m    |  92.1       |
| 100 | 0.593 | CAUTION       |  196 m    | 131.2       |
| 110 | 0.626 | ACCEPT        |  291 m    | 187.9       |
| 120 | 0.646 | ACCEPT        |  252 m    | 233.1       |
| 130 | 0.578 | CAUTION       |  294 m    |  82.5       |
| 140 | 0.424 | CAUTION       |  136 m    |  57.6       |
| 150 | 0.000 | SUPPRESS      |  379 m    |  49.0       |
| 160 | 0.515 | CAUTION       |  173 m    |  74.5       |
| 170 | 0.000 | SUPPRESS      |  322 m    |  38.9       |
| 180 | 0.318 | CAUTION       |   77 m    | 114.9       |

**Zone means (Copernicus GLO-30 evidence):**

| Zone | km range | Character       | Mean score | TRN eligible |
|------|----------|-----------------|-----------|--------------|
| 1    | 0–60 km  | Forested ridge  | 0.503     | Yes (CAUTION+) |
| 2    | 60–120 km | River gorge    | 0.585     | Yes (CAUTION+) |
| 3    | 120–180 km | High alpine   | 0.306     | Partial (SUPPRESS at km 150, 170) |

Note: SUPPRESS at km 30, 150, 170 reflects localised valley floors where
Laplacian texture variance falls below the 50.0 threshold. These gaps are
bridged by accepted TRN fixes at adjacent km intervals. Monte Carlo results
demonstrate that TRN achieves 79.5% P99 reduction despite these gaps.

---

## Sensor Substitution Readiness

| Component          | SIL source                               | HIL replacement                    | Interface change |
|--------------------|------------------------------------------|------------------------------------|-----------------|
| DEM (terrain)      | Copernicus GLO-30 GeoTIFF (offline)      | Onboard terrain data card (GeoTIFF)| Path only        |
| IMU                | STIM300 model (parametric noise model)   | STIM300 physical unit              | Format: `imu_contract.yaml` |
| EO camera          | Synthetic tiles + Shimla heightmap       | Real nadir EO camera, 5 Hz         | Format: `eo_day_contract.yaml` |
| VIO                | VIOFrameProcessor (OpenCV feature track) | OpenVINS pose estimator            | VIOEstimate interface |
| GNSS               | GNSSMeasurement (BIM evaluation)         | GPS/GLONASS receiver               | GNSSMeasurement dataclass |
| PX4 control        | Gazebo SITL OFFBOARD                     | Physical PX4 OFFBOARD              | MAVLink OFFBOARD (proven in Gazebo) |

All interface contracts committed to `docs/interfaces/`. At HIL, only
file paths and hardware connection strings change — no software interface
modifications required.

---

## Compound Fault Validation (Gate 5 — 13 April 2026)

Scenario: 180 km mission, GNSS denied km 10+, VIO confidence degraded
km 60–75 (atmospheric obscuration per Addendum v2 §10.2), TRN suppressed
km 120–135 (snow-covered alpine terrain per Addendum v2 §10.2).

| Assertion                                  | Result |
|--------------------------------------------|--------|
| SHM not triggered                          | PASS   |
| NAV_TRN_ONLY entered at VIO degradation    | PASS   |
| NAV_MODE_TRANSITION events logged          | PASS   |
| System completes km 180 without ABORT      | PASS   |
| NOMINAL during GNSS-available phase        | PASS   |
| No ABORT events in event log               | PASS   |

Architecture validated: compound environmental degradation (VIO + TRN
simultaneously stressed) does not collapse positional confidence to
the SHM threshold (0.20) when at least one correction source remains
active.

---

## Open Items Before HIL

| ID    | Item                                                        | Severity |
|-------|-------------------------------------------------------------|----------|
| EF-01 | Vehicle A OFFBOARD failsafe — PX4 instance 1 mc_pos_control invalid setpoints → failsafe blind land. Must resolve before dual-vehicle HIL. | HIGH     |
| OI-37 | `MISSION_TIMEOUT_S=300` hardcoded in run_mission.py. Must be moved to config before external demo builds. | LOW      |
| OI-40 | EC-07 Corridor Violation has no §16 recovery ownership row. SRS v1.4 fix required. | MEDIUM   |
| OI-41 | bim.py uses stdlib logging, not programme structured event log. Migrate before SRS external review. | LOW      |

---

## What HIL Must Prove

The remaining question at HIL is not whether MicroMind works. It is
whether the real hardware produces data in the format, quality, timing,
and rate that MicroMind already expects.

Specific HIL acceptance criteria:

1. **STIM300 HIGHRES_IMU at 200 Hz** — format per
   `docs/interfaces/imu_contract.yaml`

2. **EO camera nadir frames at 5 Hz** — format per
   `docs/interfaces/eo_day_contract.yaml`

3. **OpenVINS pose estimates** — VIOEstimate interface
   (delta_north_m, delta_east_m, confidence, feature_count)

4. **PX4 OFFBOARD control** — already proven in Gazebo (OI-30 closed
   10 April 2026). HIL must confirm same MAVLink OFFBOARD behaviour
   on physical hardware.

5. **EF-01 resolution** — OFFBOARD failsafe on PX4 instance 1 must be
   diagnosed and resolved before dual-vehicle HIL integration begins.

---

## SIL Baseline at Gate 5

| Suite                | Count |
|----------------------|-------|
| 406 certified tests  |  406  |
| Gate 4 extended      |   19  |
| Gate 5 corridor      |   17  |
| **TOTAL**            | **442** |

All 442 tests PASS. Zero regressions from Gate 4 baseline.

---

*Document generated at Gate 5 close. Authorised for HIL planning.*  
*Next gate: Gate 6 — Jammu–Leh tactical corridor (pending DEM tiles).*
