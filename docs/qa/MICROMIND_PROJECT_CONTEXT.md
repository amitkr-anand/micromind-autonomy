# MicroMind / NanoCorteX — Project Context
**Classification:** Programme Confidential  
**Last Updated:** 15 April 2026 (QA-037)  
**Role of this file:** Loaded ONCE at session start. Replaces all verbal re-briefing.

---

## 1. What This Programme Is

MicroMind is an onboard autonomy and GNSS-denied navigation stack for tactical UAVs and loitering munitions. It is NOT a flight controller. It operates above PX4, issuing high-level mission intent while PX4 handles vehicle stabilisation.

**The user's metric:** Precision delivery at 150+ km under full GNSS denial and RF severance from km 50. Everything else is implementation detail.

**Core operational assumption (BCMP-1):** GNSS denied or spoofed from mission start. RF link severed by km 50. No operator in the loop from that point. Terminal engagement fully autonomous under L10s-SE envelope.

---

## 2. System Architecture (One Paragraph)

**MicroMind-OS** (ground) authors and cryptographically signs the Mission Envelope before launch. **MicroMind-X** (onboard) runs sensor fusion (IMU + EO/IR + GNSS-integrity + SDR) and produces the Unified State Vector. **NanoCorteX** consumes the State Vector and drives a 7-state deterministic FSM (ST-01 NOMINAL → ST-07 MISSION_FREEZE), issuing navigation commands to PX4. **L10s-SE** is the terminal safety gate — deterministic decision tree, no ML, enforces ROE and civilian abort. **BIM** scores GNSS trust continuously (Green/Amber/Red). **DMRL** does multi-frame thermal decoy rejection. **ZPI** schedules LPI micro-bursts. **CEMS** shares EW intelligence between UAVs.

---

## 3. Air Vehicle Profiles (AVP) — The Design Truth

All test scenarios must be designed against these profiles. No other baseline is valid.

| Profile | Platform | Cruise | Top Speed | Altitude AGL | Range | GNSS-Denied Segment |
|---|---|---|---|---|---|---|
| AVP-02 | Loitering Munition (ALS-25 equiv) | 100 km/h | 140 km/h | 100–800 m | 100–250 km | 50–100 km |
| AVP-03 | Hybrid Platform | 100 km/h | 200 km/h | 100–1,500 m | 100–250 km | 100 km |
| AVP-04 | Deep Strike Expendable | 150 km/h | 200 km/h | 500–2,000 m | 1,000+ km | 100+ km |

**AVP-01 (FPV) deferred** — low-altitude regime requires separate sensor assumptions; not in current validation scope.

---

## 4. Navigation Architecture Decision (Settled 03 April 2026)

**Previous assumption in SRS/Part Two:** TRN via RADALT + NCC altimetry as primary ingress correction.  
**Revised architecture (adopted):** Three-layer stack:

| Layer | Mechanism | Function | Cost |
|---|---|---|---|
| L1 — Relative | IMU + VIO (OpenVINS) | High-rate pose, drifts ~1 m/km | Near-zero — camera + IMU already onboard |
| L2 — Absolute Reset | Orthophoto image matching vs preloaded satellite tiles | Hard position reset every 2–5 km over textured terrain, MAE < 7 m | Zero additional hardware — uses existing nadir EO/LWIR camera |
| L3 — Vertical Stability | Baro-INS fusion | Damps vertical channel divergence, provides MSL reference | Already in IMU package |

**RADALT retained for terminal phase only** (0–300 m AGL, final 5 km). Cheap proximity altimeter class, not tactical missile altimeter.  
**LWIR camera is dual-use:** orthophoto matching during ingress, DMRL decoy rejection during terminal.  
**Route planner** must cost-penalise featureless terrain (desert flat, snowfield) to avoid long gaps between image matches.

---

## 5. Repository Structure

| Repo | Purpose | State |
|---|---|---|
| `amitkr-anand/micromind-autonomy` | Main autonomy stack | S0–S8 complete, 215/215 tests, BCMP-2 CLOSED (107/107 gates, tag sb5-bcmp2-closure) |
| `amitkr-anand/nep-vio-sandbox` | VIO selection + OpenVINS integration | S-NEP-01/02 complete (424/424 tests), S-NEP-03 ready to start |

**Environment:** Python 3.12.3 / Ubuntu 24.04.4 / micromind-node01  
**Test runners:** `run_s5_tests.py` (119), `run_s8_tests.py` (68), `run_bcmp2_tests.py` (90)  
**Certified baseline runner:** `run_certified_baseline.sh` (406) + `test_gate4_extended.py` (19) + `test_gate5_corridor.py` (17) + `test_gate6_cross_modal.py` (15) — use before every gate commit and handoff  
**Total regression baseline:** 457 tests (406 certified + 19 Gate 4 + 17 Gate 5 + 15 Gate 6)

---

## 6. Current Programme State (15 April 2026 — updated QA-037)

### micromind-autonomy
| Sprint | Status | Gates | Tag |
|---|---|---|---|
| S0–S7 | ✅ CLOSED | 215/215 | Various |
| S8 IMU Characterisation | ✅ CLOSED | 68/68 | `f91180d` |
| BCMP-2 SB-1 | ✅ CLOSED | 17/17 AT-1 | `sb1-dual-track-foundation` |
| BCMP-2 SB-2 | ✅ CLOSED | 25/25 | `sb2-fault-injection-foundation` |
| BCMP-2 SB-3 | ✅ CLOSED | 29/29 AT-2 + 19/19 AT-3/4/5 | `sb3-full-mission-reports` |
| BCMP-2 SB-4 | ✅ CLOSED | Dashboard + Replay | `c183b9c` |
| BCMP-2 SB-5 | ✅ CLOSED | 17/17 AT-6 gates PASS — 1483 missions, 0 crashes, slope 1.135 MB/hr | `sb5-bcmp2-closure` |
| Sprint 0 Documentation | ✅ CLOSED | Part Two V7.2 + SRS v1.3 | `b2bae3d`, `605a747`, `2600977` |
| Sprint B Adversarial SIL | ✅ CLOSED | 6/6 ADV tests | `41238ae` |
| Sprint C OM Stub + Route Planner | ✅ CLOSED | 8/8 SC gates | `96bf98a`, `6af0e4b` |
| Sprint D Pre-HIL RC-11 / RC-7 / RC-8 | ✅ CLOSED | 9/9 SD gates | `7bebc8c` |
| OI-20 Gazebo Two-Vehicle SITL | ✅ CLOSED | x500_0 + x500_1 in scene, RTF ~1.0, 35+ s stable | `eb33572` |
| OI-35 Vehicle A OFFBOARD fix | ✅ CLOSED | ARM ✅ OFFBOARD ✅ climb 95 m ✅ lap 1 ✅ — setpoint stream thread fix verified in live SITL (08 Apr 2026) | `cd8b4f0` |
| OI-30 run_demo.sh full integration | ✅ CLOSED | Phase A+B+C verified on micromind-node01 10 Apr 2026: GAZEBO_READY ✅ EKF2 x2 ✅ VEH A ARM ✅ OFFBOARD ✅ alt 95.1 m ✅ lap 1 T+106.0s ✅ MISSION PASS ✅. SIL 290/290 ✅ | `97b2f5a` |
| EF-02 run_demo.sh clean exit | ✅ CLOSED | exec→foreground (python3.12 -u); explicit post-mission cleanup (pkill -9 gz sim, pkill -9 bin/px4, || true guards); EXIT trap fixed (|| true + pkill -9 gz sim added, pkill -9 bin/px4); os._exit() isolation test PASS (daemon thread terminated immediately, exit code 0). SIL 290/290 ✅ | `7ed5a8e` (run_mission.py) + `4ecff95` (run_demo.sh) |
| EF-01 Vehicle A OFFBOARD failsafe | 🔴 OPEN | PX4 instance 1 (PX4_GZ_STANDALONE=1) triggers mc_pos_control invalid setpoints → Failsafe: blind land immediately after OFFBOARD engagement. Vehicle A never completes lap; mission exits via OI-36 timeout guard (ABORT, exit code 2). Pre-existing; not introduced by EF-02. Separate investigation required. | — |
| SB-5 Phase A — Checkpoint v1.2 | 🟡 IN PROGRESS | SA-01–SA-04 ✅ — 6 new fields, P-01 SHM persistence, P-02 operator clearance gate, TECHNICAL_NOTES.md CREATED. SIL 294/294. | `fcb5106` |
| SB-5 Phase A — PX4-04 Reboot Detection + D8a Gate | 🟡 IN PROGRESS | SA-05–SA-07 ✅ — RebootDetector (seq-reset, rollover guard), MAVLinkBridge wired, on_reboot_detected() D8a gate, MISSION_RESUME_AUTHORISED nominal path. SIL 297/297. | `787ecd4` |
| SB-5 Phase A — EC-07 §16 Recovery Ownership Verification | ✅ CLOSED | EC-07 ownership verification complete. SB5_EC07_OwnershipVerification.md committed. Phase A exit gate: SA-01–SA-07 ✅, EC-07 doc ✅. 5/6 events compliant. OI-39 CLOSED (GNSS_SPOOF_DETECTED log event added to bim.py — Deputy 1 unfreeze authorised). OI-40 OPEN (Corridor Violation — §16 doc gap, SRS v1.4 fix). SIL 297/297. | `fff0cc4` (doc) + see OI-39 code commit |
| SB-5 Phase A — IT-PX4-01 Formal OFFBOARD Continuity Gate | ✅ CLOSED | EC-01 addressed — PX4ContinuityMonitor (integration/bridge/offboard_monitor.py) implements OFFBOARD continuity tracking, setpoint rate measurement, stale setpoint discard logging. EC01-01 ✅ continuity ≥ 99.5 %, EC01-02 ✅ loss_count ≤ 1, EC01-03 ✅ setpoint_rate ≥ 20 Hz. SIL 308/308. | this commit |
| SB-5 Phase B — PLN-02 Retask + PLN-03 Dead-End + MM-04 Queue Latency | ✅ CLOSED | SB-01–SB-07 ✅ — R-01–R-06 + PLN-03 + MM-04 event bus (MissionEventBus), EVENT_QUEUE_LATENCY, QUEUE_HIGH, QUEUE_CRITICAL_OVERFLOW, SB-06 PASS, RS-04 route fragment cleanup (_intermediate_fragments + _cleanup_route_fragments()), SB-07 PASS, TECHNICAL_NOTES.md UPDATED. SIL 309/309. | `6c405aa` (SB-01–05) + `d0e4c5d` (SB-06) + `c35122a` (SB-07) |
| Handoff 1 — QFR CLOSED | ✅ | 314/314 certified. Phase C authorised. QFR integrity findings PF-01/02/03 documented and process rules committed. Test: `f909a7c` QFR: `99fd55b` | 11 Apr 2026 |
| SB-5 Phase C — VIZ-02 Start | 🟡 IN PROGRESS | Run 1 overlay + Run 2 data pipeline — demo_overlay.py, baylands_demo_camera.py, demo_data_pipeline.py, KPI pre-computed seeds 42/101/303. SIL 314/314. | `d7dd64f` |
| SB-5 Gate 1 — Real DEM Ingest + TRN Foundation | ✅ COMPLETE | DEMLoader (rasterio GLO-30), HillshadeGenerator (Lambertian + multi-dir, CAS Eq.3), TerrainSuitabilityScorer (texture/relief/GSD → ACCEPT/CAUTION/SUPPRESS), PhaseCorrelationTRN (10-step pipeline, 4 statuses, structured event log). SHIMLA-1 DEM loaded: 1960 m elevation at Shimla, score=0.643 ACCEPT. Self-match confidence=1.0000. 5 interface contracts committed (dem/trn/imu/eo_day/eo_thermal). SIL 314/314 — zero regressions. rasterio 1.4.4 installed. | prior session |
| SB-5 Gate 2 — Gazebo Heightmap + Camera Pipeline + VIO + TRN Drift Reduction | ✅ COMPLETE | NAV-01..04 PASS (9/9 gate tests). Critical bug fix: PhaseCorrelationTRN Step 8 north correction sign inverted (+row_offset convention). TRN drift reduction: 37.17 m → 23.63 m over 35.5 km Shimla corridor. New modules: shimla_heightmap_generator.py, shimla_terrain.sdf, nadir_camera_bridge.py, vio_frame_processor.py. Interface contracts updated (trn_contract.yaml Gate 2 COMPLETE, eo_day_contract.yaml Gate 2 COMPLETE). SIL 323/323 — zero regressions. opencv-python-headless 4.13.0 installed. | `66c2643` |
| SB-5 Gate 3 — Confidence-Aware Fusion + Degraded State Handling | ✅ COMPLETE | NAV-05..08 PASS (18/18 gate tests). update_trn() added to ESKF (Deputy 1 unfreeze authorised; re-frozen post-commit). NavigationManager fusion coordinator: GNSS/VIO/TRN confidence-weighted ESKF injection, nav_confidence scoring (weights: GNSS 1.0, VIO 0.7, TRN 0.5). NAV_TRN_ONLY FSM state (ST-03B) added. Confidence-aware SHM trigger: SHM_ENTRY_LOW_NAV_CONFIDENCE fires at nav_confidence < 0.20. Camera→VIO pipeline wired (Gate 2 open finding closed). Gate 3 drift table: 50 km Shimla, seed=42, 47–70% TRN reduction at 9 fixes. config/tunable_mission.yaml created. SIL 341/341 (119 S5 + 68 S8 + 90 BCMP2 + 64 integration = 341). | `772cbfe` |
| SB-5 Gate 4 — Extended Corridor + Monte Carlo Drift Envelopes | ✅ COMPLETE | NAV-09..12 PASS (19/19 gate tests). New infrastructure: DEMLoader.from_directory() multi-tile rasterio.merge stitching (HIL production path); MissionCorridor dataclass with position_at_km() Haversine interpolation; SHIMLA_MANALI (8 WPs, 180 km, gnss_denial_start=10 km) and SHIMLA_LOCAL (55 km). MonteCarloNavEvaluator AD-16 methodology: N=300 seeds, DRIFT_PSD=1.5 m/√s per axis, σ_TRN=25 m. Monte Carlo N=300 result: P99 at 55 km = 77.1 m (TRN) vs 182.5 m (no correction) — 57.7% P99 reduction. Terrain: Zone 1 CAUTION (score 0.57–0.58), Zones 2–3 SUPPRESS (out-of-tile). OI pending Manali COP30 tile admission. Live SITL VIO: SKIP (Gazebo not available). SIL 406/406 certified + 19 Gate 4 = 425 total. | `968247f` |
| SB-5 Gate 5 — Full 180km Corridor + Monte Carlo N=300 + Compound Fault + Pre-HIL Spec | ✅ COMPLETE | NAV-13..16 PASS (17/17 gate tests). Both DEM tiles admitted (shimla_tile.tif + manali_tile.tif → merged north=32.50°N, all 8 WPs valid). Monte Carlo N=300: P99 TRN at 180km = 76.2m vs INS-only 372.6m (79.5% reduction). Terrain suitability profile: Zone 1 mean 0.503, Zone 2 mean 0.585, Zone 3 mean 0.306 (SUPPRESS at km 30, 150, 170 — valley floors). terrain_zones annotations added to SHIMLA_MANALI. Compound fault: VIO degraded km 60–75, TRN suppressed km 120–135 — SHM NOT triggered, NAV_TRN_ONLY correctly entered, km 180 reached. PREHIL_NAV_SPECIFICATION.md committed. SIL 406 certified + 19 Gate 4 + 17 Gate 5 = 442/442. | `d332a79` |
| Live SITL VIO Confidence Verification | 🔴 INCONCLUSIVE — see QA-033 | Wiring confirmed: gz.transport13 → NadirCameraFrameBridge → VIOFrameProcessor pipeline functional on Python 3.12, 1196 frames at 20.4 Hz received. Camera topic `/nadir_camera/image` publishing. VIO confidence UNMEASURABLE: heightmap terrain visual does not render in headless OGRE1 without `dirt_diffusespecular.png` (Gazebo Classic texture absent from Gazebo Harmonic). Zero Shi-Tomasi features. Two world file fixes applied (ogre2 tag removed, texture block removed). OI-42 raised: provide terrain diffuse texture. OI-43 raised: gz.transport13 in conda env. SIL 442/442 unchanged. | QA-033 |
| SB-5 Gate 6 — Cross-Modal TRN Pre-Work (Blender Frame Ingestor + Evaluator) | ✅ COMPLETE | CM-01..04 PASS (15/15 gate tests). BlenderFrameIngestor: load/validate 640×640 PNG frames, GSD computation (0.270 m/px at 150m/60°FOV). CrossModalEvaluator: clamp TRN GSD to max(camera_gsd, dem_res×0.5) — prevents DEM over-upsampling SUPPRESS. Real Blender frames (12 frames, 5km intervals): all GOOD quality (lap_var 225–340). Cross-modal peaks: 0.09–0.11 (vs current threshold 0.15) — REJECTED. Calibrated threshold: 0.091 (P10 of distribution). Operational finding: RGB/hillshade cross-modal peaks are lower than CAS paper IR/hillshade prediction (0.3–0.7). OI-42 RESOLVED: shimla_texture.png committed, Laplacian variance=3642, corners=1000. SIL 442 certified + 19 Gate 4 + 17 Gate 5 + 15 Gate 6 = 457/457. | QA-034 |
| OI-45 Same-Modal TRN Validation | ✅ COMPLETE | validate_same_modal_trn.py committed (`afb837a`). BlenderFrameRefLoader (DEMLoader-compatible) + PassthroughHillshadeGen deliver unshifted Blender frame as Sentinel-2 same-modal reference. Sentinel-2 source texture: simulation/terrain/shimla/shimla_texture.png, 512×512, 19.53 m/px — scale finding: 19.53 m/px too coarse for direct texture matching at 150 m AGL (173 m footprint ≈ 8.9 px). Same-modal proof via self-offset method: query = frame shifted by (row=20, col=25) = (+5.41 m N, −6.77 m E); reference = original frame. Results: 12/12 ACCEPTED, peaks 0.9874–0.9932 (mean 0.9903), offset recovery error 0.00 m. Cross-modal baseline: 0.09–0.11 (0/12 accepted). OI-44 confirmed ARCHITECTURAL (cross-modal NCC ceiling is expected for RGB vs DEM hillshade). OI-45 CLOSED: AD-01 validated, same-modality >> cross-modal. SIL 457/457. | QA-036 |
| OI-46 Real Sentinel-2 TRN Validation | 🔴 OPEN — action required | validate_real_sentinel_trn.py committed. SentinelTCILoader (DEMLoader-compatible) wraps T43RGQ_20251017T053241_TCI_10m.jp2 (EPSG:32643, 10 m/px, UTM 43N). TRN GSD: 5.0 m/px, ref tile 34×34 px (173 m footprint). Root finding: shimla_texture.png is viz.hh_hillshade-color.png (DEM hillshade-colour, OpenTopography) — NOT Sentinel-2 imagery. Result: 4/12 ACCEPTED, 7/12 REJECTED, 1/12 SUPPRESSED. Peak range 0.0000–0.1142 (mean 0.0911) = cross-modal OI-44 baseline. Peaks do not rise above 0.15 threshold because Blender frames are rendered from a hillshade texture, not from TCI. OI-46 OPEN: action required — replace shimla_texture.png with TCI crop, re-render Blender frames, re-run. SIL 457/457. | QA-037 |

### nep-vio-sandbox
| Sprint | Status | Gates |
|---|---|---|
| S-NEP-01 | ✅ CLOSED | 413/413 |
| S-NEP-02 | ✅ CLOSED | 424/424 |
| S-NEP-03R | ✅ CLOSED | 21/21 — metrics_engine remediation, pipeline end-to-end PASS | `0a93567` |
| S-NEP-04 | ✅ CLOSED | 10/10 pytest gates | `c875356` |
| S-NEP-05 | ✅ CLOSED | 5/5 pytest gates | `4dd3a76` |
| S-NEP-06 | ✅ CLOSED | 10/10 pytest gates | `d090851` |
| S-NEP-08 | ✅ CLOSED | 7/7 pytest gates | `30c2d56` |
| S-NEP-09 | ✅ CLOSED | 10/10 pytest gates | `4fcf231` |
| S-NEP-10 | ✅ CLOSED | 13 pytest gates (15 test methods) — MH_03 ATE 0.2729 m, V1_01 ATE 0.3424 m, acceptance_pass=true both sequences | `4bc22b4` |

### OpenVINS Validation
Stage-2 GO verdict issued 21 March 2026. Drift 0.94–1.01 m/km (3.6% variance) across EuRoC MH_03 + V1_01. Zero FM events. **Outdoor and km-scale validation PENDING (L1, L3 limitations).**

---

### SIL Baseline Definition (Updated 13 April 2026 — QA-032)

| Suite | Files | Count |
|---|---|---|
| S5 runner | test_s5_dmrl, test_s5_l10s_se, test_s5_bcmp1_runner, test_sprint_c_om_stub | 119 |
| S8 runner | test_s8a, test_s8b, test_s8c, test_s8e | 68 |
| BCMP2 runner | test_bcmp2_at1, test_bcmp2_sb2, test_bcmp2_at2, test_bcmp2_at3_5 | 90 |
| AT-6 | test_bcmp2_at6 (excl. G-14) | 16 |
| S6 | test_s6_zpi_cems | 36 |
| Pre-HIL RC | test_prehil_rc11, test_prehil_rc7, test_prehil_rc8 | 7 |
| S5 adversarial | test_s5_l10s_se_adversarial | 6 |
| SB-5 Phase A | test_sb5_phase_a | 7 |
| SB-5 Phase B | test_sb5_phase_b | 9 |
| SB-5 EC01 | test_sb5_ec01 | 3 |
| Deputy 2 adversarial | test_sb5_adversarial_d2 | 5 |
| S9 arch regression | test_s9_nav01_pass | 13 |
| Gate 2 navigation | test_gate2_navigation | 9 |
| Gate 3 fusion | test_gate3_fusion | 18 |
| Gate 4 extended corridor | test_gate4_extended | 19 |
| Gate 5 full corridor | test_gate5_corridor | 17 |
| Gate 6 cross-modal TRN | test_gate6_cross_modal | 15 |
| **TOTAL** | | **457** |

**Excluded from baseline (scope/CI reasons):**

| File | Count | Reason |
|---|---|---|
| test_s_nep_04a_interface | 19 | nep-vio-sandbox scope |
| test_s_nep_08 | 30 | nep-vio-sandbox scope |
| test_s_nep_09 | 21 | nep-vio-sandbox scope |
| test_sprint_s1–s4_acceptance | 34 | Superseded by later sprint gates |

**G-14 exclusion note:** `test_G14_memory_growth_slope` requires `AT6_ENDURANCE_HOURS >= 1.0` to produce valid regression evidence. Excluded from CI baseline. Run manually as overnight dedicated endurance test. Last confirmed pass: SB-5 closure (1483 missions, 1.135 MB/hr slope).

---

## 7. Frozen Baselines — Do Not Modify

| File | Frozen Since |
|---|---|
| `core/ekf/error_state_ekf.py` | SB-1 |
| `core/fusion/vio_mode.py` | SB-1 |
| `core/fusion/frame_utils.py` | SB-1 |
| `core/bim/bim.py` | SB-1 |
| `scenarios/bcmp1/bcmp1_runner.py` | SB-1 |
| All 332 SIL gates | SB-1 |

**C-2 Drift Envelopes (Monte Carlo N=300):**

| Boundary | Floor (P5) | Nominal | Ceiling (P99) |
|---|---|---|---|
| km 60 | 5 m | 19 m | 80 m |
| km 100 | 12 m | 96 m | 350 m |
| km 120 | 15 m | 155 m | 650 m |

**Gate 4 Monte Carlo N=300 — SHIMLA_LOCAL 55km (seed=42, QA-031):**

| km | P5 no-correction | P99 no-correction | P5 TRN | P99 TRN | P99 reduction |
|---|---|---|---|---|---|
| 10 | 7.0 m | 58.9 m | 8.2 m | 75.4 m | — (TRN noise floor > accumulated drift at 5 km denial) |
| 30 | 11.8 m | 131.2 m | 8.8 m | 70.5 m | 46.3% |
| 55 | 21.2 m | 182.5 m | 9.6 m | 77.1 m | 57.7% |

**Product claim:** MicroMind maintains position error below 77 m in 99% of missions at 55 km GNSS-denied (TRN active). DRIFT_PSD=1.5 m/√s, σ_TRN=25 m, trn_interval=5 km, speed=100 km/h.

---

## 8. Known Open Items (Must Not Be Lost)

| ID | Item | Owner | Priority |
|---|---|---|---|
| OI-01 | V7 spec: update IMU ARW floor from ≤ 0.1 to ≤ 0.2 °/√hr (STIM300 finding, S8) | Spec | HIGH — before TASL |
| ~~OI-02~~ **CLOSED** — All 3 `datetime.utcnow()` calls in `bcmp2_report.py` replaced with `datetime.now(timezone.utc)`. `from datetime import timezone` added. | Code | CLOSED |
| OI-03 | ALS-250 overnight run results → `als250_drift_chart.py` (S8-D deferred) | Code | HIGH — TASL chart |
| ~~OI-04~~ **CLOSED** a014997 — OpenVINS_ESKF_Interface_Spec.md committed in nep-vio-sandbox/docs/ | Architecture | HIGH — before S-NEP-04 |
| OI-05 | ~~TRN stub (`trn_stub.py`) still implies RADALT-NCC; must be updated to reflect orthophoto matching decision.~~ **CLOSED: orthophoto_matching_stub.py committed at 96bf98a. Measurement-provider-only pattern (AD-03). OM_R_NORTH = OM_R_EAST = 81.0 m². trn_stub.py preserved as frozen historical artefact.** | Architecture | HIGH — before fusion integration |
| OI-06 | DMRL stub is rule-based; all BCMP-1 terminal guidance results are stub-based, not CNN-based | QA Caveat | MEDIUM — document in all external reports |
| OI-07 | Outdoor / km-scale OpenVINS validation pending (L1, L3 from Stage-2 report) | Testing | HIGH — before operating envelope declared |
| OI-08 | ~~Route planner terrain-texture cost term not yet implemented~~ **CLOSED: terrain_texture_cost() added to hybrid_astar.py at 96bf98a. Default sigma_terrain=30.0 preserves all existing test behaviour.** | Code | MEDIUM — needed for featureless terrain robustness |
| OI-09 | ~~SRS §10.2 Mission Envelope Schema missing AVP speed/altitude fields~~ **CLOSED: AVP fields added in SRS v1.3 §10.2, Amendment 7 (2600977).** | Spec | MEDIUM — before SRS next revision |
| OI-10 | ~~BCMP-1 pass criteria ↔ SRS test ID traceability table missing~~ **CLOSED: Traceability table added in Part Two V7.2 §5.3.3, Amendment 11 (b2bae3d).** | Documentation | MEDIUM — before TASL |
| OI-11 | ~~Synthetic DEM always textured — featureless terrain failure mode never exercised in any test~~ **CLOSED: OM-08 test in test_sprint_c_om_stub.py committed at 96bf98a. First test to exercise featureless terrain failure mode through full OM pipeline. 14 km featureless zone, zero corrections applied.** | Testing | HIGH |
| OI-12 | fusion_node.py not in main autonomy repo — must migrate before S-NEP-04 | Architecture | HIGH | 
| OI-13 | Environment drift: pyyaml, lark absent from conda env — add to requirements.txt | Code | LOW |

| OI-14 | ~~SB-4 CLOSED — context file shows PENDING; update Section 6~~ **CLOSED: Resolved this session — Section 6 updated.** | Documentation | LOW — immediate |
| OI-15 | ~~19 of 21 architecture decisions undocumented in DECISIONS.md~~ **CLOSED: Resolved this session — AD-03 through AD-21 committed.** | Documentation | HIGH — before TASL |
| OI-16 | ~~RC-11 (VIO OUTAGE + setpoint continuity) — 5 sub-criteria all pending; RC-11b highest risk~~ **CLOSED: RC-11a–d tests committed at 7bebc8c. SetpointCoordinator committed at 7bebc8c. All 9 SD gates PASS. vio_mode.py logging added at 308016b (PD authorised). Zero NaN across 6000 ESKF steps.** | Architecture | HIGH — before CP-3 |
| OI-17 | ~~RC-7 (timestamp monotonicity injection under live timing) — pending Phase 3~~ **CLOSED: test_prehil_rc7.py committed at 7bebc8c. IFM-01 rejects non-monotonic, violation_count==1, subsequent frames accepted. SD-06 PASS.** | Architecture | HIGH — before CP-3 |
| OI-18 | ~~RC-8 (logger non-blocking 200 Hz, 60 s formal test) — pending Phase 3~~ **CLOSED: test_prehil_rc8.py committed at 7bebc8c. 12000 entries, completeness=1.0, worst_call=0.173 ms. SD-07 PASS.** | Code | HIGH — before CP-3 |
| OI-19 | ~~AT-6 gate count and exact acceptance criteria undefined — must be specified before SB-5~~ **CLOSED: Resolved this session — AT6_Acceptance_Criteria.md committed. 17 gates defined across 4 groups.** | Testing | MEDIUM — before SB-5 |
| ~~OI-20~~ **CLOSED** eb33572 — Two-vehicle simultaneous Gazebo rendering verified on micromind-node01 (07 Apr 2026). Root cause: RTX 5060 Ti requires NVIDIA EGL (`__EGL_VENDOR_LIBRARY_FILENAMES=10_nvidia.json`) + OGRE1 (`GZ_ENGINE_NAME=ogre`). Fix confirmed in px4-rc.gzsim (single vehicle) and now extended to two-vehicle world. New files: `simulation/worlds/two_vehicle_sitl.sdf`, `simulation/launch_two_vehicle_sitl.sh`. Scene check: x500_0 ✅ x500_1 ✅ — RTF ~1.0, stable 35+ s, zero render errors. SIL: 460/460. OI-30 (run_demo.sh integration) remains open. | Code | CLOSED |
| AD-22 | Demo environment (micromind-node01) and embedded compute (Jetson Orin NX) are architecturally decoupled per AD-22 (06 April 2026). SRS §9.4 VIZ-03 HIL prerequisite clause removed. Demo design session scheduled before SB-5 Phase C. | Architecture | CLOSED — decision recorded. |
| ~~OI-30~~ **CLOSED** 97b2f5a (10 Apr 2026) — `run_demo.sh` v3.0 full integration: Phase A (Gazebo server + GUI, baylands, OGRE1), Phase B (PX4 inst 0 + inst 1, MAVLink EKF2 check), Phase C (`exec python3.12 simulation/run_mission.py "$@"`). Infrastructure fixes this session: EKF2 check switched from `gz topic` (DDS realm, not visible via Gazebo transport) to MAVLink `LOCAL_POSITION_NED`; `PX4_GZ_WORLD=baylands` added to inst 1 (`PX4_GZ_STANDALONE=1` requires it for scene/info service path). Live SITL verified on micromind-node01: GAZEBO_READY ✅ EKF2 x2 ✅ VEH A ARM ✅ OFFBOARD ✅ alt 95.1 m ✅ lap 1 T+106.0s ✅ MISSION PASS ✅. SIL 290/290. | Code | CLOSED |
| ~~OI-31~~ **CLOSED** 12 Apr 2026 — Demo design session complete. Decisions: Run 1 = Gazebo SITL + matplotlib overlay (demo_overlay.py, baylands_demo_camera.py); Run 2 = three modes: Layout A Replay 150km, Layout B Live 150km, Layout C Comparative 50km. Phase C builds operational infrastructure. Polished UI controls deferred Phase D. Data pipeline: demo_data_pipeline.py (load_kpi_json, get_vehicle_tracks, get_mission_events, get_comparative_metrics, LiveMissionFeed). KPI seeds 42/101/303 pre-computed. V-02/V-03 compliance: event positions from KPI log data only; Vehicle B interpolated to 100ms display cadence. design decisions in simulation/demo/TECHNICAL_NOTES.md. | Architecture | CLOSED |
| OI-21 | ~~mark_send not natively integrated into mavlink_bridge setpoint loop — CP-2 latency result has asterisk~~ **CLOSED: Sprint D code review (4972110) confirmed mark_send IS natively integrated at mavlink_bridge.py lines 358-359. CP-2 asterisk withdrawn.** | Code | MEDIUM — before CP-3 |
| OI-22 | ESKF position PSD (1.0 m/√s) empirically set; needs derivation from STIM300 data before HIL | Architecture | MEDIUM — before HIL |
| ~~OI-23~~ **CLOSED** — AD-19 velocity check run across `bcmp1_runner.py` and all `scenarios/bcmp2/*.py`. Result: CLEAN — zero hits for `state\.v\b` or `\.velocity`. `scenarios/bcmp2/TECHNICAL_NOTES.md` created with findings. | Code | CLOSED |
| OI-24 | Drift envelope metric over-conserves 3.3–9.8× on diverging trajectories; must be documented in external reports | Documentation | MEDIUM |
| OI-25 | Jetson Orin latency margins unknown — all timing evidence from Ryzen 7 9700X | Testing | MEDIUM — before HIL |
| OI-26 | ~~L10s-SE adversarial EO condition tests absent — QA standing rule #2 currently violated by all test results~~ **CLOSED: 6 adversarial integration tests ADV-01 through ADV-06 committed at 41238ae. Gate 3 civilian detection now exercised through full DMRL pipeline for first time. QA standing rule #2 satisfied for terminal guidance.** | Testing | HIGH — SIL completeness |
| OI-27 | ZPI and CEMS not integrated into any mission runner — must be caveated in capability claims | QA Caveat | MEDIUM |
| OI-28 | NIS is diagnostic only (PF-03) — must not be tuned without TD approval; not documented externally | Documentation | MEDIUM — before HIL |
| ~~OI-29~~ **CLOSED** — `endurance` marker added to `pytest.ini` `markers` list. Warning eliminated. | Code | CLOSED |
| ~~OI-NEW-01~~ **CLOSED** fix committed — docstring corrected | QA Caveat | HIGH — before any sprint is declared formally closed |
| OI-NEW-02 | nep-vio-sandbox sprint table header is missing a `Commit` column — added ad hoc for S-NEP-03R onwards; table schema should be formalised | Documentation | LOW — cosmetic |
| OI-NEW-03 | G-03R-08 SIL regression must exclude test_snep03r_e2e.py itself (--ignore flag) to avoid recursive subprocess timeout — pattern must be applied to any future e2e gate files that include a SIL regression test | Architecture | MEDIUM — apply to future gate files |
| ~~S-NEP-10-PRE~~ | **CLOSED** — S-NEP-10 complete, 552/552 gates green, tag 4bc22b4 | QA | CLOSED |
| ~~OI-32~~ | **CLOSED** e70b981 — MH_01_easy added to S-NEP-10 gate file (G-10-18 to G-10-23). Reproducible Option B IMU+VIO baseline: ATE 0.3412 m. The mh01_run1.json figure of 0.0865 m is superseded (produced by an unrestorable pipeline version without IMU propagation). External reports must cite 0.3412 m for MH_01_easy. | Code/QA | CLOSED |
| ~~OI-33~~ | **CLOSED** 07 Apr 2026 — Demo world: Baylands selected (option a). Terrain 899 m × 587 m, non-repeating, Fuel assets fully cached locally (407 MB, no network dependency). Two-vehicle spawn verified at [0,0,0.5] and [0,5,0.5] — both land on solid terrain. 900 m X-axis flight corridor available. Flight altitude must be ≥ 50 m AGL (tree collision mesh present below). two_vehicle_sitl.sdf retained as rendering verification tool only — not demo world. OI-30 unblocked. | Architecture | CLOSED |
| ~~OI-36~~ **CLOSED** 4fbe1d4 — `t_a.join(timeout=300)` + `t_b.join(timeout=300)` with `is_alive()` guards added to `main()`. ABORT fires `os._exit(2)` if either thread does not complete within 300 s. Live SITL verification 10 Apr 2026: ABORT fired correctly, exit to prompt within 5 s, gz sim gone ✅, px4 gone ✅. OI-37 raised for `MISSION_TIMEOUT_S` config governance. | Code | CLOSED |
| OI-37 | `MISSION_TIMEOUT_S = 300` is hardcoded in `simulation/run_mission.py main()`. No `mission_timeout` key in any config. Must be moved to config before external demo builds. Documented in `simulation/TECHNICAL_NOTES.md`. | Code | LOW — config governance, not blocking |
| ~~OI-38~~ **CLOSED** — SB-5 Phase A closed (7 gates, 297/297 SIL) at 787ecd4. Phase B closed (SB-01–SB-06, 305/305 SIL) this session. EC-08 addressed — MM-04 queue latency instrumented. MissionEventBus committed with EVENT_QUEUE_LATENCY, QUEUE_HIGH, QUEUE_CRITICAL_OVERFLOW. Phase B exit gates: UT-PLN-02 ✅, IT-PLN-01 ✅, IT-PLN-02 ✅, UT-MM-04 ✅. | Code | CLOSED |
| ~~OI-39~~ | **CLOSED** — `_log.warning("GNSS_SPOOF_DETECTED: bim_score=%.4f", raw)` added to `core/bim/bim.py:252`. `import logging as _logging` + `_log = _logging.getLogger(__name__)` added (pattern from vio_mode.py). No logic changes. Deputy 1 unfreeze authorised for this change only. SIL 297/297. EC-07 §16 GNSS Spoof row now Compliant=Y. | Code | CLOSED |
| OI-40 | EC-07 non-compliance — Corridor Violation (predicted) has no recovery ownership row in §16. `core/state_machine/state_machine.py` emits `CORRIDOR_VIOLATION` trigger → ABORT from NOMINAL (line 240), EW_AWARE (line 263), GNSS_DENIED (line 304), SILENT_INGRESS (line 325). `core/l10s_se/l10s_se.py:48` defines `AbortReason.CORRIDOR_VIOLATION`. §16 has no authoritative definition of Detects/Decides/Executes/Logs roles for this event. Current implementation (NanoCorteXFSM → ABORT directly) has not been reviewed against §16 ownership rules. Fix: add §16 row for Corridor Violation (predicted) in SRS v1.4 revision. See `docs/qa/SB5_EC07_OwnershipVerification.md`. | Architecture/Docs | MEDIUM — §16 documentation gap; fix required before SRS v1.4 |
| ~~OI-35~~ **CLOSED** cd8b4f0 — `_start_setpoint_stream()` added to `simulation/run_mission.py`. Two call-sites in `mission_vehicle_a()`: thread starts before `_arm_and_offboard()`, stops (with `.join(timeout=1.0)`) after ACK. Live SITL verification 08 Apr 2026: VEH A ARM ✅ OFFBOARD ✅ altitude 95 m ✅ lap 1 complete T+107.7s ✅. Two infrastructure findings fixed this session: (1) `~/.gz/sim/8/server.config` updated to include PX4 sensor system plugins (Imu, NavSat, AirPressure, Magnetometer, Contact) — root cause of EKF2 alignment failures in all previous headless SITL attempts on this machine; (2) two Gazebo instance accumulation risk documented for OI-30 cleanup phase. | Code | CLOSED |
| OI-41 | `core/bim/bim.py` structured log debt — bim.py uses stdlib `logging` (pattern from vio_mode.py, introduced at OI-39 fix) but does not use the programme's structured event log dict pattern (req_id, severity, module_name, timestamp_ms). All other modules introduced in SB-5 Phase A/B use the event_log dict pattern. bim.py should be migrated to structured logging before SRS external review. Not blocking — stdlib log is auditable but not machine-parseable via the programme log schema. | Code | LOW — tech debt, not blocking |
| ~~OI-42~~ **CLOSED** QA-034 — `simulation/terrain/shimla/shimla_texture.png` committed (viz.hh_hillshade-color.png from OpenTopography download). World file updated: OGRE2 PBR plane at Z=1800m carries texture via `albedo_map`. Laplacian variance=3642, Shi-Tomasi corners=1000. VIO confidence measurement unblocked for next SITL session. | Code | CLOSED |
| ~~OI-44~~ **CLOSED as ARCHITECTURAL** QA-036 — Cross-modal (RGB vs DEM hillshade) NCC ceiling 0.09–0.11 is the EXPECTED result for different modalities. AD-01 requires same-modality (Sentinel-2 vs Sentinel-2), not cross-modal. OI-45 (same-modal validation) resolves this. No threshold change needed for correct operational mode. | Architecture | CLOSED |
| ~~OI-45~~ **CLOSED** `afb837a` (15 April 2026) — Same-modality TRN validation complete. validate_same_modal_trn.py: BlenderFrameRefLoader + PassthroughHillshadeGen. Results: 12/12 ACCEPTED, peaks 0.9874–0.9932, offset recovery 0.00 m. Sentinel-2 texture scale finding: shimla_texture.png at 19.53 m/px is too coarse for direct texture matching at 150 m AGL; production requires ≤3 m/px Sentinel-2 GeoTIFF or ≥500 m AGL operation. AD-01 validated. | Architecture | CLOSED |
| OI-46 | Real Sentinel-2 TCI validation (validate_real_sentinel_trn.py): peaks 0.09–0.11 = cross-modal baseline. Root cause: shimla_texture.png is DEM hillshade-colour (viz.hh_hillshade-color.png, OpenTopography), not Sentinel-2 imagery. Action: (1) crop T43RGQ TCI to corridor extent → replace shimla_texture.png; (2) re-render 12 Blender frames with TCI texture; (3) re-run validate_real_sentinel_trn.py to confirm same-modality peaks ≥ 0.30. km=55 SUPPRESSED anomaly also requires investigation (pre-TRN diagnostic passes scorer thresholds but TRN returns SUPPRESSED — likely edge case in JP2 windowed read or UTM→pixel conversion). | TRN/Blender | HIGH — genuine same-modality corridor validation incomplete |
| OI-43 | `gz.transport13` Python bindings not installed in micromind-autonomy conda env. `NadirCameraFrameBridge` silently falls back to inject-only mode. Current workaround: use system Python 3.12 + `PYTHONPATH=/usr/lib/python3/dist-packages`. Fix: install gz-transport13 Python bindings in conda env, or add Python 3.12 dispatch wrapper to run_sitl_vio.py. Not blocking (workaround exists) but fragile. | Code | MEDIUM — fragile workaround |
| ~~E-02~~ **CLOSED** c35122a — RS-04 v1.2 route fragment cleanup implemented. `_intermediate_fragments` list tracks non-adopted replan attempts per retask. `_cleanup_route_fragments(ts_ms)` clears fragments and logs `ROUTE_FRAGMENT_CLEANUP` (DEBUG, req_id='RS-04', payload: fragments_cleared, bytes_freed_estimate) on all retask() exit paths. No time.time() calls; ts_ms passed from caller. SB-07 PASS. SIL 309/309. | Code | CLOSED |
---

## 9. QA Agent Standing Instructions

Claude is acting as QA agent on this programme. Standing rules:

1. **Never approve a test result without asking:** does the test environment represent operational conditions?
2. **The L10s-SE is a legal/ethical gate, not just a functional threshold.** Any test touching terminal guidance must verify the decision tree executes correctly under adversarial EO conditions, not just clean synthetic inputs.
3. **The SRS is the authority for pass/fail criteria.** When reviewing code, always trace to a requirement ID.
4. **Outdoor and km-scale validation is not complete.** Do not allow EuRoC indoor results to be presented as mission-scale evidence.
5. **DMRL is a stub.** Flag this clearly in any report that references terminal guidance test results.
6. **Frozen files are frozen.** Any proposed modification to frozen files requires explicit programme director approval and re-running the full regression suite.

---

## 10. Governing Documents (Load on Demand)

| Document | When to Load |
|---|---|
| `MicroMind_PartTwo_V7_2.docx` | Reviewing FR boundary conditions, algorithm parameters, NFRs. (V7.1 preserved as baseline in docs/qa/) |
| `MicroMind_SRS_v1_3.docx` | Reviewing any test case, requirement traceability, or acceptance criteria. (v1.2.1 preserved as baseline in docs/qa/) |
| `MicroMind_V6_PART_ONE.pdf` | Checking whether implementation matches operational intent |
| `BCMP2_STATUS.md` | BCMP-2 sprint work |
| `NEP_SPRINT_STATUS.md` | VIO integration work |
| Stage-2 OpenVINS Report | VIO performance claims |

---
*This file is the session entry point. Update Section 6 (Programme State) and Section 8 (Open Items) at the end of every session.*
