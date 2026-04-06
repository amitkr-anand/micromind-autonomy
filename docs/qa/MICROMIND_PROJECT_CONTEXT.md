# MicroMind / NanoCorteX — Project Context
**Classification:** Programme Confidential  
**Last Updated:** 06 April 2026  
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
**Integration RC tests:** 7 additional (4 RC-11 + 2 RC-7 + 1 RC-8)  
**Total regression baseline:** 290 tests

---

## 6. Current Programme State (06 April 2026)

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

### nep-vio-sandbox
| Sprint | Status | Gates |
|---|---|---|
| S-NEP-01 | ✅ CLOSED | 413/413 |
| S-NEP-02 | ✅ CLOSED | 424/424 |
| S-NEP-03 | 🔲 READY | EuRoC end-to-end, real MetricSet |
| S-NEP-04 | 🔲 PLANNED | OpenVINS → ESKF integration |

### OpenVINS Validation
Stage-2 GO verdict issued 21 March 2026. Drift 0.94–1.01 m/km (3.6% variance) across EuRoC MH_03 + V1_01. Zero FM events. **Outdoor and km-scale validation PENDING (L1, L3 limitations).**

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

---

## 8. Known Open Items (Must Not Be Lost)

| ID | Item | Owner | Priority |
|---|---|---|---|
| OI-01 | V7 spec: update IMU ARW floor from ≤ 0.1 to ≤ 0.2 °/√hr (STIM300 finding, S8) | Spec | HIGH — before TASL |
| OI-02 | `bcmp2_report.py` line 420: `datetime.utcnow()` → `datetime.now(UTC)` (Python 3.12) | Code | LOW — cosmetic |
| OI-03 | ALS-250 overnight run results → `als250_drift_chart.py` (S8-D deferred) | Code | HIGH — TASL chart |
| OI-04 | OpenVINS → ESKF interface spec not documented | Architecture | HIGH — before S-NEP-04 |
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
| OI-20 | Gazebo GUI blank on micromind-node01 (X11/OGRE2) — blocks OEM demo and run_demo.sh | Code | HIGH — before any OEM meeting |
| OI-21 | ~~mark_send not natively integrated into mavlink_bridge setpoint loop — CP-2 latency result has asterisk~~ **CLOSED: Sprint D code review (4972110) confirmed mark_send IS natively integrated at mavlink_bridge.py lines 358-359. CP-2 asterisk withdrawn.** | Code | MEDIUM — before CP-3 |
| OI-22 | ESKF position PSD (1.0 m/√s) empirically set; needs derivation from STIM300 data before HIL | Architecture | MEDIUM — before HIL |
| OI-23 | System rule 1.8 (no velocity-dependent control logic) not enforced in BCMP runners | Code | MEDIUM |
| OI-24 | Drift envelope metric over-conserves 3.3–9.8× on diverging trajectories; must be documented in external reports | Documentation | MEDIUM |
| OI-25 | Jetson Orin latency margins unknown — all timing evidence from Ryzen 7 9700X | Testing | MEDIUM — before HIL |
| OI-26 | ~~L10s-SE adversarial EO condition tests absent — QA standing rule #2 currently violated by all test results~~ **CLOSED: 6 adversarial integration tests ADV-01 through ADV-06 committed at 41238ae. Gate 3 civilian detection now exercised through full DMRL pipeline for first time. QA standing rule #2 satisfied for terminal guidance.** | Testing | HIGH — SIL completeness |
| OI-27 | ZPI and CEMS not integrated into any mission runner — must be caveated in capability claims | QA Caveat | MEDIUM |
| OI-28 | NIS is diagnostic only (PF-03) — must not be tuned without TD approval; not documented externally | Documentation | MEDIUM — before HIL |
| OI-29 | `pytest.ini` missing `endurance` marker registration — pytest warns "Unknown pytest.mark.endurance" on every AT-6 endurance run | Code | LOW — cosmetic; add `endurance` to `markers` in pytest.ini |
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
