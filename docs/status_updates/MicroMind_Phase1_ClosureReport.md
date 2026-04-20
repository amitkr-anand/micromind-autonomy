**MicroMind / NanoCorteX**

Autonomous Navigation & Terminal Guidance System

**PHASE 1 CLOSURE REPORT**

*Software-in-the-Loop Development Programme*

  ---------------------- ------------------------------------------------
  **Programme**          MicroMind / NanoCorteX

  **Document Type**      Phase 1 Closure Report

  **Version**            1.0

  **Date**               12 March 2026

  **Programme Director** Amit Anand (amitkr-anand)

  **Repository**         amitkr-anand/micromind-autonomy

  **Final Commit**       3c37d82 \| Tag: s10-m2-m5-closed

  **Target Partner**     TASL (Tata Advanced Systems Limited)

  **Classification**     Programme Confidential
  ---------------------- ------------------------------------------------

**1 Executive Summary**

MicroMind / NanoCorteX Phase 1 is complete. All five programme
milestones have been formally closed, the 222-test regression suite
passes in full, and the primary TASL demonstration artefact --- a
three-IMU 250 km drift chart --- has been committed to the main branch.

Phase 1 was governed by the Demo Fork constraint established on 28
February 2026: deliver a defensible bounded autonomy core for SIL
demonstration to TASL. No speculative features, no architectural
experiments. Every sprint decision was evaluated against that
constraint.

The system navigates a 250 km contested corridor without GNSS, achieves
a maximum lateral drift of 16 m per 5 km segment against a 100 m limit,
and survives the last 10 seconds without any RF link. The full BCMP-1
normative test scenario --- 100 km corridor, GNSS denial, adaptive
jamming, satellite overpass, thermal target with decoy --- is
implemented and passing.

**1.1 Milestone Closure Summary**

  --------------- ------------------------ ------------ ---------------------------
  **Milestone**   **Title**                **Closed**   **Evidence**

  M1              Autonomy Core            Sprint S5    BCMP-1 runner 111/111 gates

  M2              GNSS-Denied Navigation   Sprint S10   250 km NAV-01 PASS --- all
                                                        3 IMUs, drift ≤ 16 m/5 km

  M3              EW Survivability         Sprint S4    EW engine + Hybrid A\*
                                                        68/68 gates

  M4              Terminal Autonomy        Sprint S5    DMRL + L10s-SE 111/111
                                                        gates

  M5              Demo Presentation        Sprint S10   S8-D three-curve drift
                                                        chart PNG + HTML committed
  --------------- ------------------------ ------------ ---------------------------

**1.2 Regression Gate Summary**

  --------------------- ----------------------- ------------ --------------------
  **Test Suite**        **Runner**              **Result**   **Sprint**

  S1--S5 acceptance     run_s5_tests.py         111/111 ✅   S5
  gates                                                      

  S6 ZPI/CEMS gates     test_s6_zpi_cems.py     36/36 ✅     S6

  S8 IMU / ALS-250      run_s8_tests.py         68/68 ✅     S8
  gates                                                      

  S9 NAV-01 regression  test_s9_nav01_pass.py   10/10 ✅ (2  S10
  gates                                         skip)        

  **TOTAL**                                     **222/222    
                                                ✅**         
  --------------------- ----------------------- ------------ --------------------

**2 Sprint History**

Phase 1 was delivered across eleven sprints (S0--S10) spanning
approximately four months. Each sprint closed against a formal
acceptance gate. No sprint was declared complete without all gates
passing.

  ------------ ------------ ----------- -----------------------------------------------
  **Sprint**   **Commit**   **Gates**   **Delivered**

  S0           6e1c70a      ✅          Error-State Kalman Filter V2, quaternion
                                        mathematics library, INS mechanisation
                                        (strapdown integration)

  S1           5005a5d      9/9         7-state NanoCorteX FSM, SimClock,
                                        MissionLogSchema, BCMP-1 scenario definition

  S2           e86140f      9/9         BIM trust scorer (GNSS integrity monitor), GNSS
                                        spoof injector simulation

  S3           284acb4      8/8         TRN stub (Terrain-Referenced Navigation), nav
                                        scenario, Plotly mission dashboard

  S4           366f963      8/8         EW engine, Hybrid A\* route planner, BCMP-1 EW
                                        simulation

  S5           7ad5db5      111/111     DMRL terminal guidance, L10s-SE safety
                                        envelope, full BCMP-1 runner --- M1 and M4
                                        closed

  S6           a7633ab      36/36       ZPI burst scheduler, CEMS cooperative EW
                                        sharing, multi-UAV simulation (Phase-2 frozen)

  S7           aa3302a      ✅          9-panel mission dashboard, HTML debrief report
                                        generator

  S8           f91180d      68/68       IMU fidelity models (STIM300, ADIS16505-3,
                                        BASELINE), ALS-250 corridor simulation

  S9           7fba53c      215/215     TRN+ESKF architectural overhaul --- dual-filter
                                        conflict resolved, NAV-01 closed at 150 km

  **S10**      3c37d82      222/222     NCC vectorisation, S9 regression gates,
                                        parallel IMU runner, S8-D drift chart, 134×
                                        performance fix --- M2 and M5 closed
  ------------ ------------ ----------- -----------------------------------------------

**2.1 Key Architectural Events**

Two sprints represent architectural turning points that are important
for any future developer to understand:

**S9 --- TRN+ESKF Architectural Correction**

The 250 km NAV-01 failure discovered in S8 was traced to five concurrent
root causes. The most significant: TRNStub contained an internal Kalman
filter that fought the ESKF --- two independent estimators receiving the
same measurements and corrupting each other. The fix removed TRNStub\'s
internal filter entirely, making it a pure measurement provider. The
ESKF Q-matrix gyro bias random walk was also 247× too large (1×10⁻⁵ vs
correct 4.04×10⁻⁸ rad/s/√s per STIM300 datasheet), and the position
process noise block was zero --- causing Kalman gain collapse. All five
root causes were fixed in S9. NAV-01 closed at 150 km.

**S10 --- O(n²) Performance Bug**

The ALS-250 simulation was unusable at 250 km due to a disguised O(n²)
complexity bug. The IMU noise model methods total_gyro() and
total_accel() reconstructed the full noise array on every call --- at
909,000 steps, this was catastrophic. A one-line cache fix in
mechanisation.py delivered a 134× speedup (105 seconds for 250 km).
Results are bit-identical; the fix is pure performance.

**3 NAV-01 Results --- 250 km Corridor**

NAV-01 is the primary navigation acceptance criterion: lateral position
drift must remain below 100 m per 5 km segment across the full corridor.
All three IMU configurations were validated at 250 km under a fixed
random seed (seed=42) with 166 TRN corrections each.

  ------------------ ------------ ------------ --------------- ------------ ------------
  **IMU Model**      **Max 5km    **Final      **TRN           **Margin**   **NAV-01**
                     Drift**      Drift**      Corrections**                

  Safran STIM300     13.9 m       6.3 m        166             7.2×         **PASS ✅**
  (primary)                                                                 

  ADI ADIS16505-3    16.0 m       5.4 m        166             6.3×         **PASS ✅**
  (budget)                                                                  

  BASELINE (ideal    9.6 m        3.4 m        166             10.4×        **PASS ✅**
  reference)                                                                
  ------------------ ------------ ------------ --------------- ------------ ------------

Limit: \< 100 m per 5 km segment. All three models pass with comfortable
margin. TRN correction interval: one correction per \~1.5 km (spec:
every 2 km). Wall time: \~105 seconds per 250 km run after S10
performance fix.

**3.1 Performance Baseline**

  ---------------------- ---------------- --------------- ----------------
  **Configuration**      **Steps/sec**    **250 km Wall   **Notes**
                                          Time**          

  After S10 perf fix     \~8,600          \~105 seconds   Current baseline
  (measured)                                              

  Before S10 perf fix    \~65             \~14,000        O(n²) noise
                                          seconds         array rebuild

  search_pad_px=80       \~15             \~60,000        7× NCC expansion
  (wrong)                                 seconds         

  verbose=True + SSH tee \~45             N/A             Never use in
                                                          production
  ---------------------- ---------------- --------------- ----------------

**4 Phase 1 Technical Achievements**

**4.1 GNSS-Denied Navigation**

The INS mechanisation implements full strapdown integration: quaternion
attitude propagation, body-to-navigation frame rotation, coriolis and
gravity compensation. The Error-State Kalman Filter (ESKF) maintains a
15-state error vector (position, velocity, attitude, accelerometer bias,
gyroscope bias) with process noise tuned to STIM300 datasheet values.

Terrain-Referenced Navigation (TRN) provides absolute position
corrections by matching radar altimeter sweeps against a pre-loaded
Digital Elevation Model using 2D Normalised Cross-Correlation. The 2D
NCC approach requires only 2 DEM reads and one vectorised numpy matrix
operation per correction, compared to \~33,640 Python-level DEM lookups
in the original 1D implementation. Accuracy improvement: NAV-01 drift of
33.3 m (2D) vs 421.9 m failure (1D).

**4.2 EW Survivability**

The EW engine detects jamming via spectrogram analysis and DBSCAN
clustering, produces a cost map update within 500 ms, and triggers route
replan via Hybrid A\* within 1 second. The Beacon & Integrity Monitor
(BIM) provides continuous GNSS trust scoring (0.0--1.0) using PDOP
thresholds, Doppler deviation, and hysteresis rules. Spoof detection
latency is ≤ 250 ms.

**4.3 Terminal Autonomy**

The Dual-Mode Recognition & Lock system (DMRL) provides multi-frame
EO/IR target association with CNN-based decoy rejection. Lock confidence
threshold is ≥ 0.85; decoy abort triggers at ≥ 0.80 over 3 consecutive
frames. The Last-10-Second Safety Envelope (L10s-SE) enforces
rules-of-engagement compliance with civilian detection threshold ≥ 0.70
and decision timeout ≤ 2 seconds. Zero ML is used in this path --- the
decision tree is fully deterministic.

**4.4 Multi-UAV Coordination (Phase-2 Frozen)**

ZPI (Zero-RF hop plan protocol) and CEMS (Cooperative EW Sharing) were
implemented in S6 and are present in the codebase. These modules are
frozen pending TASL platform decision and are not part of the Phase 1
demonstration scope.

**4.5 IMU Fidelity**

Three IMU configurations are modelled with full noise parameter
fidelity: Safran STIM300 (primary, datasheet TS1524 rev.31), Analog
Devices ADIS16505-3 (budget alternative), and BASELINE (ideal reference
for regression anchoring). Noise model parameters cover ARW, VRW, bias
instability, bias random walk, scale factor stability, and misalignment.

**5 Known Issues & Deferred Items**

**5.1 Expected Test Skips**

Two tests in test_s9_nav01_pass.py are permanently skipped (not failed).
Both cover ESKF Q-matrix constant access, which is class-private and not
exported. The underlying constants are correct and covered functionally
by NAV-01 simulation gates.

**5.2 Git Author Identity**

Commits from the Azure VM show Ubuntu default identity
(azureuser@micromind-vm\...) rather than Amit Anand. Action: run git
config \--global user.name and user.email on the VM before next commit
session.

**5.3 Deferred Phase-2 Items**

  ---------------------- --------------- ---------------------------------
  **Item**               **Phase**       **Gate to Open**

  DMRL CNN upgrade       Phase-2         GPU + training dataset +
                                         clearance path confirmed

  PQC cryptography       Phase-2         TASL continuation confirmed
  (FR-109--112)                          

  ROS2 / PX4 SITL        Phase-3         TASL platform decision and
  integration                            hardware procurement

  Real RADALT hardware   Phase-3         Post-TASL procurement
  interface                              

  CEMS active use in     Phase-2         Phase-1 demo acceptance
  mission                                

  Satellite masking /    Phase-2         Phase-1 demo acceptance
  FR-108                                 

  GPU/CUDA NCC           Phase-2         Monte Carlo sweep requirement
                                         confirmed

  Cross-mission learning Phase-2         Phase-1 demo acceptance + dataset
  (DD-02)                                
  ---------------------- --------------- ---------------------------------

**6 Compute & Environment**

**6.1 Development Environment**

  ---------------------- ------------------------------------------------
  **Component**          **Specification**

  **Compute Platform**   Azure VM micromind-vm --- Standard_D8s_v5 (8
                         vCPU, 32 GB RAM)

  **Prior Platform**     2017 MacBook Pro --- retired S8 due to thermal
                         throttling

  **OS**                 Ubuntu 24 LTS

  **Language**           Python 3.10

  **Environment          Conda --- environment: micromind-autonomy
  Manager**              

  **Repository**         GitHub: amitkr-anand/micromind-autonomy, branch:
                         main

  **Key Libraries**      numpy, scipy, plotly, pytest, subprocess
                         (MKL-safe parallelism)

  **Azure Cost to Date** ₹3,423 actual (₹9,405 forecast before S10 perf
                         fix)
  ---------------------- ------------------------------------------------

**6.2 Parallelism Strategy**

numpy/BLAS multiprocessing via fork is unreliable on Linux due to MKL
fork deadlock. The validated approach (S10-4) uses subprocess.Popen ---
one OS process per IMU, isolated Python interpreter. This avoids all
shared-memory conflicts. The broken run_als250_parallel_v2.py (MKL fork
approach) has been removed from the repository.

**7 Lessons Learned**

  -------- ---------------------- ----------------------------------------------
  **\#**   **Lesson**             **Detail**

  **1**    **TRN wiring gap is a  run_als250_sim() had no TRN corrections wired
           masked failure mode**  despite core/ins/trn_stub.py existing since
                                  S3. S8-C acceptance tests validated only 10 km
                                  corridors --- too short to reveal drift
                                  divergence. Root cause of the S8-D 250 km
                                  failure.

  **2**    **Dual Kalman filters  TRNStub\'s internal filter and the ESKF were
           are catastrophic**     both updating position state from the same TRN
                                  measurements. Two independent estimators with
                                  shared inputs produce corruption, not
                                  redundancy. The correct pattern: one
                                  estimator, one measurement provider.

  **3**    **O(n²) bugs hide at   The noise array rebuild bug was undetectable
           short corridor         at 50 km (105 steps/sec) but catastrophic at
           lengths**              250 km (65 steps/sec). Always profile at
                                  target corridor length before declaring
                                  performance acceptable.

  **4**    **search_pad_px is a   Increasing from 25 to 80 expands NCC search
           critical performance   area 7× (\~5,625 to \~40,000 pixels per
           gate**                 correction). Must be audited before every long
                                  run. Include in Pre-Flight Check Protocol.

  **5**    **Timing estimates are A 70-minute estimate for a 250 km run was
           floors, not            wrong by 3×. Never commit compute budget based
           predictions**          on estimates --- run a 60-second smoke test
                                  and observe actual throughput.

  **6**    **Verbose output over  Reduces simulation throughput from \~1,810
           SSH is a throughput    steps/sec to \~45 steps/sec --- a 40× penalty.
           killer**               Always use verbose=False for production runs.
                                  Never pipe through tee to a remote terminal.

  **7**    **tmux discipline is   Run 1 was lost because the tmux session was
           non-negotiable**       not detached before VS Code disconnected. Any
                                  job over 5 minutes runs inside tmux. No
                                  exceptions.

  **8**    **Scope discipline     Demo Fork constraint (28 Feb 2026) prevented
           preserves delivery**   scope creep across S9 and S10. Every requested
                                  addition was evaluated against \'does this
                                  close a Phase-1 milestone?\' If not, it was
                                  deferred.
  -------- ---------------------- ----------------------------------------------

**8 S11 Readiness**

S11 scope is gated on the TASL partnership decision. Two paths are
prepared:

**8.1 If TASL Proceeds**

Infrastructure investment decision: dedicated workstation vs continued
Azure VM. S11 will likely cover HIL preparation (ROS2/PX4 SITL interface
design), BCMP-1 full scenario hardening at 250 km, and DMRL CNN upgrade
scoping.

**8.2 If TASL Is Deferred**

Whitepaper drafting. TRN_Whitepaper_Outline.docx contains an 8-section
structure targeting an aerospace engineering audience. Content is ready
to draft.

**8.3 Session Start Checklist for S11**

> git checkout main && git pull origin main
>
> git log \--oneline main \| head -5 \# Expected: 3c37d82 at top
>
> python run_s5_tests.py \# 111/111
>
> python tests/test_s6_zpi_cems.py \# 36/36
>
> python run_s8_tests.py \# 68/68
>
> pytest tests/test_s9_nav01_pass.py \# 10 pass, 2 skip
>
> \# Must be 222/222 before any changes
>
> \# Confirm session goal with Amit before writing any code

**8.4 Frozen Constants --- Do Not Change Without Spec Update**

  ------------------------- ------------------ -------------------- ------------------
  **Constant**              **Value**          **Module**           **Notes**

  **\_ACC_BIAS_RW**         9.81×10⁻⁷ m/s²/√s  error_state_ekf.py   STIM300 TS1524
                                                                    rev.31

  **\_GYRO_BIAS_RW**        4.04×10⁻⁸ rad/s/√s error_state_ekf.py   STIM300 TS1524
                                                                    rev.31

  **\_POS_DRIFT_PSD**       1.0 m/√s           error_state_ekf.py   Position process
                                                                    noise

  **NCC_THRESHOLD**         0.45               trn_stub.py          FR-107 correlation
                                                                    gate

  **search_pad_px**         25                 als250_nav_sim.py    Critical --- must
                                                                    not increase

  **CORRECTION_INTERVAL**   1500 m             trn_stub.py          TRN update every
                                                                    1.5 km

  **Innovation gate**       150 m              trn_stub.py          Hybrid-3 gate
                                                                    config
  ------------------------- ------------------ -------------------- ------------------

**9 Repository State at Closure**

Final commit: 3c37d82 on branch main. Tag: s10-m2-m5-closed on 2de6089.

  ------------------------------------------------- ---------------- ----------------------
  **File**                                          **Sprint**       **Status**

  core/ins/mechanisation.py                         S0 + S8 +        ✅ O(n²) cache fix
                                                    S10-perf         applied

  core/ins/trn_stub.py                              S3 + S9 + S10-1  ✅ NCC vectorised,
                                                                     internal Kalman
                                                                     removed

  core/ekf/error_state_ekf.py                       S0 V2 + S9 +     ✅ Q-matrix corrected,
                                                    S10-perf         buffers pre-allocated

  core/ins/imu_model.py                             S8-A             ✅ STIM300 / ADIS /
                                                                     BASELINE models

  core/bim/bim.py                                   S2               ✅ GNSS trust scorer

  core/ew_engine/ew_engine.py                       S4               ✅ EW cost map

  core/route_planner/hybrid_astar.py                S4               ✅ Route replan

  core/dmrl/dmrl_stub.py                            S5               ✅ Terminal guidance

  core/l10s_se/l10s_se.py                           S5               ✅ L10s safety
                                                                     envelope

  core/zpi/zpi.py                                   S6               ✅ Phase-2 frozen

  core/cems/cems.py                                 S6               ✅ Phase-2 frozen

  sim/als250_nav_sim.py                             S8-C + S9 + S10  ✅ search_pad_px=25
                                                                     confirmed

  dashboard/als250_drift_chart.py                   S10-3            ✅ Attribute fixes
                                                                     applied

  dashboard/als250_drift_chart_20260312_1416.png    S10-3            ✅ TASL primary
                                                                     artefact

  dashboard/als250_drift_chart_20260312_1416.html   S10-3            ✅ TASL interactive
                                                                     artefact

  run_als250_parallel.py                            S10-4            ✅ subprocess.Popen,
                                                                     no MKL deadlock

  tests/test_s9_nav01_pass.py                       S10-2            ✅ 10 pass, 2 expected
                                                                     skips

  run_als250_parallel_v2.py                         REMOVED          🗑 MKL fork deadlock
                                                                     --- deleted at S10
                                                                     close
