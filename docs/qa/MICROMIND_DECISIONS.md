# MicroMind Architecture Decisions Register
**Format:** One entry per decision. Immutable once recorded — amendments create new entries.  
**Reference:** Complements DD-01 and DD-02 in Part Two V7.

---

## AD-01 — Navigation Ingress Correction Mechanism
**Date:** 03 April 2026  
**Status:** ADOPTED  
**Owner:** Programme Director

### Decision
Replace RADALT-NCC TRN as the primary ingress correction mechanism with Orthophoto Image Matching against preloaded satellite imagery tiles.

### Context
Part Two V7 specified TRN via radar altimeter (0.5–50 m AGL range) + Normalised Cross-Correlation against DEM elevation profile. This specification was found to be inconsistent with AVP-02/03/04 cruise altitudes (100–2,000 m AGL), where a conventional RADALT beam footprint spans 9–13 DEM grid cells, averaging out terrain relief and degrading correction accuracy.

Research review (03 April 2026) established that orthophoto image matching achieves MAE < 7 m across 20 experimental scenarios including night-time LWIR operation, with no additional hardware required. The system uses the existing downward-facing EO/LWIR camera already specified for DMRL.

### Architecture
**Three-layer navigation stack:**
- L1 (Relative): IMU + VIO (OpenVINS) — high-rate pose, ~1 m/km drift
- L2 (Absolute Reset): Orthophoto matching vs preloaded satellite tiles — hard position reset every 2–5 km over textured terrain, MAE < 7 m, no accumulated error
- L3 (Vertical Stability): Baro-INS — damps vertical channel divergence, no terrain sensing function

**RADALT retained** for terminal phase only: 0–300 m AGL, final 5 km, altitude input for aimpoint offset computation.

**LWIR camera dual-use:** orthophoto matching during ingress (L2); DMRL decoy rejection during terminal.

### Consequences
- `trn_stub.py` must be updated to reflect orthophoto matching as the L2 mechanism (OI-05)
- SRS NAV-02 test cases must be rewritten: correction mechanism is image matching, not NCC altimetry. Pass criterion (< 50 m CEP-95) unchanged.
- Route planner (hybrid_astar.py) requires terrain-texture cost term to penalise featureless zones (OI-08)
- V7 RADALT spec must be scoped to terminal phase only (remove from ingress navigation section)
- Storage: ~10–15 GB satellite tiles for 150 km radius at 1 m resolution — within 32 GB eMMC spec

### Risks
- Image matching fails over featureless terrain (Thar Desert flat, Himalayan snowfield). Mitigated by route planner texture cost and VIO bridging.
- Night operation requires LWIR image matching against visible-light orthophotos — cross-spectral matching tested and demonstrated in literature but not yet validated in MicroMind SIL.

### References
- Yao et al. (2024), GNSS-Denied Geolocalization using Terrain-Weighted Constraint Optimization
- OKSI OMNInav system architecture
- ICRA 2001 Sinopoli et al. (hierarchical DEM + probabilistic navigation)

---

## AD-02 — IMU Specification Floor Correction
**Date:** 03 April 2026 (flagged in S8, 27 February 2026)  
**Status:** PENDING SPEC UPDATE  
**Owner:** Spec author

### Decision
Update Part Two V7 IMU ARW floor from ≤ 0.1°/√hr to ≤ 0.2°/√hr.

### Context
S8 sensor characterisation established that the STIM300 (primary tactical IMU candidate) has a typical ARW of 0.15°/√hr, which exceeds the V7 spec floor of 0.1°/√hr. The ADIS16505-3 (MEMS candidate) is at 0.22°/√hr. The BASELINE model used for BCMP-2 C-2 envelope calibration is 0.05°/√hr — a value no real candidate sensor achieves.

### Consequences
- V7 spec update required before TASL meeting
- C-2 envelopes must be re-validated with STIM300 noise profile once ALS-250 overnight run data is available (OI-03)
- Any BCMP-2 results presented externally must note that C-2 envelopes were calibrated on BASELINE IMU, not STIM300

---

## AD-03 — Single ESKF Architecture (TRN as Measurement Provider Only)
**Date:** 04 March 2026 (Sprint S9)
**Status:** ADOPTED
**Owner:** Software Architect

### Decision
Remove the internal Kalman filter from the TRN module. TRN operates as a measurement provider only — it computes a position correction and returns it to the caller. The ESKF applies the correction via its standard measurement update path. No second estimator runs independently.

### Context
Prior to S9, the TRN module contained its own Kalman filter that ran in parallel with the ESKF. Both estimators received measurements derived from the same IMU and DEM sources. Analysis during S9 root-cause investigation established that two independent estimators receiving overlapping measurements will corrupt each other's covariance — the shared information is counted twice, producing overconfident state estimates that do not reflect true uncertainty. This was one of five root causes corrected in S9.

### Consequences
- `trn_stub.py` `update()` method returns a `TRNCorrection` record; caller applies via `eskf.update_vio()` — this interface pattern is correct and must be preserved in the orthophoto matching rework (OI-05)
- Any future correction source (orthophoto matching, terrain feature matching) must follow the same measurement-provider-only pattern
- No module below the ESKF layer may maintain an independent position estimator

### References
- S9 architectural correction (4 March 2026) — five-root-cause fix session
- `MicroMind_S9_TD_Update.md`

---

## AD-04 — 2D Patch NCC over 1D Profile NCC for Terrain Correlation
**Date:** March 2026 (TRN Sandbox, pre-S10)
**Status:** ADOPTED
**Owner:** Software Architect

### Decision
Use 2D patch Normalised Cross-Correlation (NCC) against DEM tiles for terrain correlation. 1D profile NCC (single elevation strip) is rejected.

### Context
A forced pre-S10 evaluation ran both approaches on a 150 km Himalayan manoeuvring corridor with real SRTM DEM data. Results were definitive:

| Method | NAV-01 result | Status |
|---|---|---|
| 1D profile NCC | 421.9 m drift | FAIL — 8× over limit |
| 2D patch NCC | 33.3 m drift | PASS — 3× margin |

The performance difference is architectural, not parametric. 1D NCC requires approximately 33,640 Python DEM lookups per correction cycle. 2D numpy NCC requires 2 DEM reads and one matrix operation — vectorised by numpy. The 1D approach failed not from correlation quality but from accumulated numerical error in the sequential lookup path.

### Consequences
- `trn_stub.py` uses 2D NCC; `STRIP_WIDTH_PX` and `STRIP_LEN_PX` define the patch dimensions
- The orthophoto matching rework (OI-05, AD-01) uses the same 2D NCC algorithm — only the input domain changes (elevation patches → optical image patches)
- GPU NCC optimisation reclassified Phase-2; 2D numpy NCC completes 150 km in ~1 s on micromind-node01
- `search_pad_px=25` (125 m search radius at 5 m/px) is a critical parameter — reducing it causes NAV-01 failure

### References
- `TRN_Sandbox_ClosureReport_S10Handoff.md`
- Pre-S10 sandbox session (March 2026, Azure VM)

---

## AD-05 — subprocess.Popen for Parallelism, not multiprocessing
**Date:** March 2026 (Sprint S10)
**Status:** ADOPTED
**Owner:** Software Architect

### Decision
Use `subprocess.Popen` to run parallel IMU simulation instances. Python's `multiprocessing` module is not used for this workload.

### Context
S10 implemented parallel IMU runner execution to accelerate the three-IMU ALS-250 250 km corridor simulation. Initial implementation using `multiprocessing.Pool` produced a deadlock on Linux: numpy and BLAS use Intel MKL, and MKL's thread pool does not survive a `fork()` call cleanly. `subprocess.Popen` spawns isolated Python interpreter processes with no shared memory state — each child initialises its own MKL context cleanly.

### Consequences
- `run_als250_parallel.py` uses `subprocess.Popen`; this pattern should be used for any future multi-process parallelism in the stack
- `multiprocessing` is safe for CPU-bound pure-Python workloads with no numpy/BLAS dependencies
- Azure cost reduction from parallel execution: forecast ₹9,405 → actual ₹3,423 for S8–S10

### References
- `HANDOFF_S10_to_S11.md`
- Sprint S10 (12 March 2026, Azure VM)

---

## AD-06 — IMU Noise Array Caching in Mechanisation Loop
**Date:** March 2026 (Sprint S10)
**Status:** ADOPTED
**Owner:** Software Architect

### Decision
Cache the IMU noise arrays on first call within the mechanisation loop. Do not recompute `total_gyro()` and `total_accel()` at every timestep.

### Context
Performance profiling during S10 revealed a critical O(n²) bug: `total_gyro()` and `total_accel()` were pure functions called inside the per-timestep loop. For a 250 km corridor at 200 Hz, this produced approximately 180,000,000 redundant function calls. Caching on first call produced a 134× speedup (14,000 s → 105 s). The caching is safe because the functions are side-effect-free and the IMU model does not change during a run.

### Consequences
- `core/ins/mechanisation.py` caches noise arrays; any modification to the mechanisation loop must preserve this caching behaviour
- Developers adding new IMU noise sources must ensure their generator functions are pure (no side effects, deterministic for a given seed) before the caching pattern applies

### References
- `HANDOFF_S10_to_S11.md`
- Sprint S10 critical performance fix (12 March 2026)

---

## AD-07 — Bounded Deterministic State Machine, No Learned Policy
**Date:** Sprint S1 / Demo Fork (28 February 2026)
**Status:** ADOPTED
**Owner:** Programme Director

### Decision
NanoCorteX executes a formally specified deterministic finite automaton. No reinforcement learning, no neural policy, no probabilistic transition logic. All state transitions are defined by explicit guard conditions evaluated against the Unified State Vector.

### Context
Two constraints drove this decision jointly. First, tactical systems operating under ROE require deterministic, auditable behaviour — every decision must be traceable to a specific input condition and a specific rule. A learned policy cannot be audited in this sense. Second, the Demo Fork scope constraint (28 February 2026) formally bounded Phase-1 to capabilities validatable in SIL without hardware-dependent training data.

### Consequences
- The 7-state FSM (ST-01 through ST-07) in `core/state_machine/state_machine.py` is the sole authority for mode transitions
- L10s-SE decision tree is explicitly ML-free (FR-105 requirement)
- Any future capability requiring probabilistic decision logic must be implemented as a separate advisory layer — it cannot replace the FSM
- Consistent with IHL requirements for meaningful human control in lethal autonomous systems

### References
- `MicroMind_SoftwareArchitecture_v1_0.md`
- Demo Fork scope decision (28 February 2026)
- Part Two V7 §1.2

---

## AD-08 — Autonomy as Payload, No Autopilot Modification
**Date:** Sprint S0
**Status:** ADOPTED
**Owner:** Programme Director

### Decision
MicroMind integrates with host UAV platforms via standard MAVLink interfaces. No modification of the PX4 flight controller core is required or permitted. MicroMind operates strictly at the mission and trajectory layer.

### Context
The programme targets multiple airframe classes (AVP-02, AVP-03, AVP-04) with a single software stack. Hard-coding to any specific autopilot would create platform lock-in and require re-certification for each new airframe. The payload architecture enables retrofit integration without touching flight controller firmware, and preserves PX4's safety envelope independently of MicroMind's mission logic.

### Consequences
- All autopilot interaction is via MAVLink OFFBOARD mode setpoints
- `integration/mavlink_bridge.py` is the sole interface point between MicroMind and PX4
- Driver abstraction (AD-09) is a direct consequence of this decision
- TASL hardware decision on airframe is deferred post-SIL; this decision enables that deferral

### References
- `MicroMind_SoftwareArchitecture_v1_0.md`
- Sprint S0
- Part Two V7 §4.5

---

## AD-09 — Python ABCs for Driver Types; DriverFactory at Startup
**Date:** 28 March 2026 (ADR-0)
**Status:** ADOPTED
**Owner:** Software Architect

### Decision
All sensor driver types are defined as Python Abstract Base Classes. A `DriverFactory` module selects and instantiates the appropriate concrete driver at startup based on configuration. Core navigation code depends only on the ABC interface, never on a concrete driver.

### Context
The Pre-HIL integration sprint required demonstrating that the simulation → real hardware swap could be performed without modifying any core navigation code (RC-4 pass criterion). The ABC pattern enforces this at the type level — if a concrete driver does not implement the full ABC interface, the programme fails at import time rather than at runtime.

### Consequences
- RC-4 confirmed: `OfflineVIODriver` → `LiveVIODriver` swap produces no changes in `core/` or `scenarios/`
- Adding a new sensor type requires: (1) defining an ABC in `integration/drivers/`, (2) implementing a concrete class, (3) registering in `DriverFactory` — no other changes
- `LiveVIODriver` is currently a stub placeholder; implementing real OpenVINS subscription is a future sprint task (OI-04, OI-12)

### References
- `MicroMind_PreHIL_ADR0_v1_1.md` (D-3)
- Pre-HIL Integration Sprint (29 March 2026)

---

## AD-10 — Platform: Ubuntu 24.04 LTS (RTX 5060 Ti Driver Incompatibility with 22.04)
**Date:** March 2026 (platform transition to micromind-node01)
**Status:** ADOPTED
**Owner:** Programme Director

### Decision
micromind-node01 runs Ubuntu 24.04 LTS (Noble Numbat). Ubuntu 22.04 LTS was attempted first and abandoned.

### Context
Initial Ubuntu 22.04 LTS installation on micromind-node01 (Ryzen 7 9700X, RTX 5060 Ti) failed to produce a stable GPU driver environment. The RTX 5060 Ti is a recent-generation GPU; NVIDIA driver libraries at the time of installation were not fully compatible with the Ubuntu 22.04 userspace stack. The decision to move to Ubuntu 24.04 LTS resolved this.

The concern about ROS2 support on Ubuntu 24.04 was independently verified and found to be unfounded in the opposite direction from what was suspected: ROS2 Jazzy Jalisco uses Ubuntu 24.04 Noble as its primary tier-1 supported platform. Ubuntu 22.04 is only tier-3 supported for Jazzy (source-build only, not recommended). The platform choice of Ubuntu 24.04 is therefore correct for both the GPU stack and the ROS2 stack simultaneously.

### Consequences
- micromind-node01 runs Ubuntu 24.04.4 LTS, Kernel 6.17, NVIDIA driver 580.126.09, CUDA 13.2
- ROS2 Jazzy installs from standard deb packages on this platform — no source build required
- Ubuntu 22.04 is not a supported development or test environment for this programme
- Any future hardware additions must be verified for Ubuntu 24.04 driver availability before procurement

### References
- `SYSTEM_BASELINE_2026-03-19.md`
- ROS2 Jazzy official documentation — Ubuntu 24.04 listed as primary platform
- Pre-HIL transition (March 2026)

---

## AD-11 — Mission Clock Ownership: MicroMind in SITL; Shared Hardware Source in HIL/Production
**Date:** 28 March 2026 (ADR-0) / Extended April 2026
**Status:** ADOPTED (SITL scope); DESIGN INTENT (HIL/Production)
**Owner:** Software Architect (SITL) / Hardware Architect (HIL/Production)

### Decision
**SITL:** MicroMind maintains the authoritative monotonic mission clock. PX4 receives `SYSTEM_TIME` messages as a courtesy reference only.

**HIL and Production:** Both MicroMind and PX4 derive timing from a single shared hardware clock source. Neither system owns the clock independently — both are disciplined to the same reference.

### Context
In SITL, there is no shared hardware clock. MicroMind owning the clock prevents IFM-01 (timestamp misalignment) — the highest-consequence silent failure mode identified in Pre-HIL architecture review. In HIL and production, the correct architecture is a single hardware clock source (GPS-disciplined oscillator, GNSS timing receiver, or dedicated timing module) from which both MicroMind-X and PX4 derive their time references, eliminating the clock ownership question entirely.

### Consequences
- The SITL clock ownership pattern in `integration/mavlink_bridge.py` is correct for SITL only and must not be carried forward unchanged into HIL
- HIL integration requires defining the shared clock source hardware and synchronisation protocol before CP-3
- The IFM-01 monotonicity guard remains valid in all configurations
- RC-7 (formal timestamp monotonicity injection test under live timing) is pending Phase 3 (OI-17)

### References
- `MicroMind_PreHIL_ADR0_v1_1.md` (D-2)
- Pre-HIL Integration Sprint (28 March 2026)

---

## AD-12 — Seven PX4 Failure Modes as Governing Integration Constraints
**Date:** 28 March 2026 (ADR-0)
**Status:** ADOPTED
**Owner:** Software Architect

### Decision
The Pre-HIL integration layer is designed against seven formally enumerated PX4 failure modes (FM-1 through FM-7). These failure modes are the governing constraints for all integration decisions.

### Context
A December 2025 integration attempt produced a root-cause analysis identifying seven distinct failure modes in the PX4 OFFBOARD interface under realistic conditions: OFFBOARD timeout, heartbeat loss, sequence number reset (reboot detection), setpoint type mismatch, arming state conflict, mode rejection, and coordinate frame error. Designing against documented failure modes rather than nominal behaviour ensures the integration handles real operational conditions.

### Consequences
- Each FM has a specific detection method, state transition, and recovery behaviour in `integration/mavlink_bridge.py`
- S-PX4-01 through S-PX4-09 test suite validates each FM path
- FM-7 (coordinate frame error) drove the ENU→NED frame rotation requirement now frozen in `core/fusion/frame_utils.py`
- Any future PX4 interface change must be assessed against all seven FM definitions before integration

### References
- `MicroMind_PreHIL_ADR0_v1_1.md` (D-4)
- Pre-HIL Integration Sprint (28 March 2026)

---

## AD-13 — Listener-First Sensor Integration Architecture
**Date:** 28 March 2026 (SIA v1.0)
**Status:** ADOPTED
**Owner:** Software Architect

### Decision
Sensors are integrated using a listener-first architecture. MicroMind subscribes to sensor data streams wherever performance is adequate. Direct sensor ownership is used only where listener performance is unacceptable.

### Decision rule applied:
- IMU, GNSS, RADALT (terminal): listener via MAVLink (`HIGHRES_IMU`, `GPS_RAW_INT`, `DISTANCE_SENSOR`)
- EO camera: direct MIPI/CSI-2 or USB3 — no MAVLink path for video
- SDR front-end: direct USB3 or PCIe — no autopilot interface for raw IQ data

### Consequences
- Sensor stream rate verification (V-9 from ADR-0 v1.1: HIGHRES_IMU 200 Hz, GPS_RAW_INT 5 Hz, DISTANCE_SENSOR 10 Hz) pending confirmation
- Camera and SDR driver stubs exist in `integration/drivers/`; real implementations require hardware procurement
- BIM spoof detection via GPS_RAW_INT listener cannot intercept pre-filter spoofing — noted limitation for operational clearance

### References
- `MicroMind_PreHIL_ADR0_v1_1.md` SIA section
- SIA v1.0 (28 March 2026)

---

## AD-14 — Demo Fork: Phase-1 Scope Boundary
**Date:** 28 February 2026
**Status:** ADOPTED
**Owner:** Programme Director

### Decision
Phase-1 scope is formally bounded. The following capabilities are reclassified to Phase-2: CEMS active use in mission, predictive EW, FR-108 (satellite masking), FR-109–112 (PQC cryptography stack). All completed sprint work (S0–S8) is retained as valid Phase-1 evidence.

### Context
The programme is founder-led and resource-constrained. Attempting 100% of the full feature set risks delivering 60% of a broader stack rather than 100% of a bounded core. The Demo Fork was a scope isolation exercise, not a redesign.

### Consequences
- CEMS and ZPI are code-complete and unit-tested (36/36 S6 gates) but not integrated into any mission scenario runner — capability claims must note this (OI-27)
- Predictive EW branch exists in `core/ew_engine/ew_engine.py` but is not exercised by any test
- The same discipline applied at Demo Fork should be applied at each subsequent phase boundary

### References
- `MicroMind_DemoEdition_RealignmentAnalysis.md`
- MicroMind_Phase1_ClosureReport.md (Lesson 8)

---

## AD-15 — Vehicle A: Illustrative INS Drift Model, Not Full 3D Mechanisation
**Date:** 29 March 2026 (BCMP-2 SB-1)
**Status:** ADOPTED
**Owner:** Software Architect

### Decision
Vehicle A (INS-only baseline) in the BCMP-2 dual-track comparison uses a simplified cross-track error propagation model. Its purpose is to illustrate how INS drift accumulates in the absence of any correction — not to be a high-fidelity replica of a specific inertial mechanisation.

### Context
Vehicle A exists to answer one question visually: what does uncorrected INS drift look like over 150 km? Full 3D mechanisation would add implementation complexity without changing this visual argument. Vehicle A and Vehicle B start from identical IMU noise conditions (same model, same seed per C-1 constraint) — Vehicle B receives navigation corrections and Vehicle A does not. The comparison is therefore valid for its stated purpose: demonstrating the correction value MicroMind provides.

### Consequences
- Vehicle A is not a valid reference for precise quantification of INS-only position error in a specific airframe
- Any external claim about Vehicle A behaviour must describe it as "illustrative INS-only drift" rather than "simulated [specific platform] inertial performance"
- The C-2 envelopes characterise Vehicle A drift statistically and are calibrated against this simplified model — they are not portable to a different propagation model

### References
- `MicroMind_BCMP2_Implementation_Architecture_v1_1.md`
- BCMP-2 SB-1 (29 March 2026)
- `scenarios/bcmp2/baseline_nav_sim.py`

---

## AD-16 — Monte Carlo (N=300) Calibration for C-2 Drift Envelopes
**Date:** 29 March 2026 (BCMP-2 SB-1)
**Status:** ADOPTED
**Owner:** Software Architect

### Decision
C-2 drift envelopes are calibrated using Monte Carlo simulation with N=300 seeds. Floors at P5; ceilings at P99. Analytical worst-case derivation is not used as the envelope definition.

### Context
Analytical derivation uses worst-case constant bias — a single deterministic ceiling. Monte Carlo with N=300 seeds draws bias values from the IMU noise distribution, producing a statistically honest envelope: 99% of missions will stay within the ceiling, and the 1% that breach it are the legitimate statistical tail.

### Consequences
- **Critical caveat:** The N=300 calibration used the BASELINE IMU model (ARW 0.05 °/√hr), not the STIM300 (0.15 °/√hr). Envelopes must be re-run with STIM300 noise profile (OI-03). Any external report citing C-2 envelopes must note the calibration baseline.
- Seed 303 (stress) is reserved for SB-5 AT-6
- Envelope values (km 60: [5,80]m, km 100: [12,350]m, km 120: [15,650]m) are frozen; changing them requires a new Monte Carlo run and re-tagging

### References
- `MicroMind_BCMP2_Implementation_Architecture_v1_1.md`
- BCMP-2 SB-1 (29 March 2026)
- `bcmp2_drift_envelopes.py`

---

## AD-17 — OpenVINS Selected as VIO System
**Date:** 21 March 2026 (S-NEP Stage-2 GO)
**Status:** ADOPTED
**Owner:** Programme Director

### Decision
OpenVINS is selected as the VIO system for MicroMind L1 relative navigation.

| Candidate | Outcome | Reason |
|---|---|---|
| OpenVINS | GO — Selected | Drift 0.94–1.01 m/km, zero FM events, honest covariance |
| RTABMap | NO-GO (Stage-1) | RPE failure + covariance type incompatibility (PF-01) |
| ORB-SLAM3 | NOT READY | No global covariance output by design |
| VINS-Fusion | DEFERRED | No stable ROS2 Humble port at evaluation time |
| Kimera-VIO | DEFERRED | Programme pacing rule |

### Consequences
- Stage-2 GO verdict is based on indoor EuRoC sequences (≤130 m). Km-scale and outdoor validation pending (OI-07) — must be stated in any external VIO performance claim
- NIS is diagnostic only (AD-18); must not be used as a tuning signal
- System rule 1.8 (no control logic depending on `state.v`) applies (AD-19)
- `fusion_node.py` migration to main repo pending (OI-12)

### References
- OpenVINS Stage-2 Endurance Validation Report v4.1 (21 March 2026)
- VIO Selection Standard v1.2

---

## AD-18 — NIS is Diagnostic Only; Must Not Be Tuned Without TD Approval
**Date:** 22 March 2026 (S-NEP-04)
**Status:** ADOPTED
**Owner:** Technical Director

### Decision
The Normalised Innovation Squared (NIS) metric from the OpenVINS integration is a diagnostic signal only. It must not be used to tune filter parameters, adjust measurement weights, or gate VIO updates without Technical Director approval.

### Context
OpenVINS produces position updates at high rate (~122 Hz) from consecutive frame estimates — these are not independent measurements in the statistical sense the NIS model assumes. The absolute covariance R used in the ESKF update is not a correct noise model for inter-frame innovations. NIS values outside the [0.3, 3.0] consistent band do not unambiguously indicate filter mistuning.

### Consequences
- NIS is logged and monitored for gross anomalies (FM-03: NIS p95 > 5.0 triggers an event)
- NIS values must not be cited as evidence of filter calibration quality without the measurement model caveat
- Any proposal to change ESKF R based on NIS observations requires TD approval and documented rationale
- This rule must be referenced in any external report presenting NIS data (OI-28)

### References
- `NEP_SPRINT_STATUS.md` (PF-03)
- `HANDOFF_S-NEP-04_CLOSURE.md`
- S-NEP-04 (22 March 2026)

---

## AD-19 — Velocity State Must Not Be Used as Primary Control Input (System Rule 1.8)
**Date:** 22 March 2026 (S-NEP-07 Rev 3)
**Status:** ADOPTED
**Owner:** Technical Director

### Decision
No control, planning, or decision logic shall rely on `state.v` (velocity state vector) as a primary input. Velocity may be used as a secondary or advisory signal only.

### Context
OpenVINS provides position-only measurement updates. The velocity state is propagated by IMU integration between updates and is not directly observed by VIO. Velocity takes more than 60 seconds from zero-initialisation to converge to a reliable estimate. Any logic treating `state.v` as ground truth during this window will make incorrect decisions silently.

### Consequences
- `state.v` must not be used as: a navigation health signal, route planner input, EW threat parameter, or BCMP acceptance criterion
- Velocity convergence time (>60 s) must be noted in any operational procedure requiring rapid engagement after launch
- The BCMP runners do not currently enforce this rule explicitly (OI-23) — enforcement should be added before any velocity-dependent feature is introduced

### References
- `MicroMind_SNEP07_EngineeringDecisions_Rev3.md`
- S-NEP-07 Rev 3 (22 March 2026)

---

## AD-20 — Dashboard in Matplotlib, not Plotly/Dash
**Date:** 30 March 2026 (SB-3/SB-4 boundary)
**Status:** ADOPTED
**Owner:** Software Architect

### Decision
The BCMP-2 dashboard and replay tools are implemented in matplotlib, producing static PNG and self-contained HTML output. Plotly/Dash is not used.

### Context
Plotly and Dash are not installed on micromind-node01 and are not in the programme's conda environment. The existing BCMP-1 dashboard uses matplotlib — consistency avoids introducing a second visualisation dependency. Matplotlib's self-contained HTML export opens in any browser without a running server process, satisfying the air-gap safety requirement.

### Consequences
- `dashboard/bcmp2_dashboard.py` and `dashboard/bcmp2_replay.py` are matplotlib-only
- Any future dashboard work follows the matplotlib pattern unless the environment policy changes
- The S3 Plotly dashboard (`dashboard/mission_dashboard.py`) is a legacy artefact — it is not modified or extended

### References
- `BCMP2_STATUS.md` SB-4 architecture note
- SB-3/SB-4 boundary decision (30 March 2026)

---

## AD-21 — Drift Envelope Metric is a Confidence Signal, Not a Hard Error Bound
**Date:** March 2026 (S-NEP-09)
**Status:** ADOPTED
**Owner:** Technical Director

### Decision
The `drift_envelope_m` metric is a confidence degradation signal. It must not be interpreted as a guaranteed upper bound on position error.

### Context
S-NEP-09 analysis established that the drift envelope metric over-conserves by 3.3–9.8× on diverging trajectories and produces infinite over-estimation on loopback trajectories. The metric is computed from accumulated displacement rather than actual position error — these diverge significantly when the trajectory contains turns, loops, or reversals.

### Consequences
- `drift_envelope_m` may be used to trigger confidence-degradation responses in the state machine
- External reports must describe it as a conservative confidence indicator, not a CEP or position error bound
- Actual position error is only known relative to ground truth — in SIL this is the Gazebo reference; in field operation it is not directly observable without GNSS
- This limitation is not yet in any external-facing document (OI-24)

### References
- `MicroMind_SNEP09_AnalysisReport_Rev2.md`
- S-NEP-09 analysis (March 2026)

---
*Append new decisions above the final line.*
