# MicroMind / NanoCorteX — Project Context
**Classification:** Programme Confidential  
**Last Updated:** 03 April 2026  
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
| `amitkr-anand/micromind-autonomy` | Main autonomy stack | S0–S8 complete, 215/215 tests, BCMP-2 SB-3 closed (90/90 gates) |
| `amitkr-anand/nep-vio-sandbox` | VIO selection + OpenVINS integration | S-NEP-01/02 complete (424/424 tests), S-NEP-03 ready to start |

**Environment:** Python 3.12.3 / Ubuntu 24.04.4 / micromind-node01  
**Test runners:** `run_s5_tests.py` (111), `run_s8_tests.py` (68), `run_bcmp2_tests.py` (90)

---

## 6. Current Programme State (03 April 2026)

### micromind-autonomy
| Sprint | Status | Gates | Tag |
|---|---|---|---|
| S0–S7 | ✅ CLOSED | 215/215 | Various |
| S8 IMU Characterisation | ✅ CLOSED | 68/68 | `f91180d` |
| BCMP-2 SB-1 | ✅ CLOSED | 17/17 AT-1 | `sb1-dual-track-foundation` |
| BCMP-2 SB-2 | ✅ CLOSED | 25/25 | `sb2-fault-injection-foundation` |
| BCMP-2 SB-3 | ✅ CLOSED | 29/29 AT-2 + 19/19 AT-3/4/5 | `sb3-full-mission-reports` |
| BCMP-2 SB-4 | ⏳ PENDING | Dashboard + Replay | Entry gate: SB-3 ✅ |
| BCMP-2 SB-5 | ⏳ PENDING | Repeatability + Closure | After SB-4 |

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
| OI-05 | TRN stub (`trn_stub.py`) still implies RADALT-NCC; must be updated to reflect orthophoto matching decision | Architecture | HIGH — before fusion integration |
| OI-06 | DMRL stub is rule-based; all BCMP-1 terminal guidance results are stub-based, not CNN-based | QA Caveat | MEDIUM — document in all external reports |
| OI-07 | Outdoor / km-scale OpenVINS validation pending (L1, L3 from Stage-2 report) | Testing | HIGH — before operating envelope declared |
| OI-08 | Route planner terrain-texture cost term not yet implemented | Code | MEDIUM — needed for featureless terrain robustness |
| OI-09 | SRS §10.2 Mission Envelope Schema missing AVP speed/altitude fields | Spec | MEDIUM — before SRS next revision |
| OI-10 | BCMP-1 pass criteria ↔ SRS test ID traceability table missing | Documentation | MEDIUM — before TASL |

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
| `MicroMind_SRS_v1.2.1` | Reviewing any test case, requirement traceability, or acceptance criteria |
| `MicroMind_PartTwo_V7_Live.docx` | Reviewing FR boundary conditions, algorithm parameters, NFRs |
| `MicroMind_V6_PART_ONE.pdf` | Checking whether implementation matches operational intent |
| `BCMP2_STATUS.md` | BCMP-2 sprint work |
| `NEP_SPRINT_STATUS.md` | VIO integration work |
| Stage-2 OpenVINS Report | VIO performance claims |

---
*This file is the session entry point. Update Section 6 (Programme State) and Section 8 (Open Items) at the end of every session.*
