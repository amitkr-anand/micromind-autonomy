**MicroMind / NanoCorteX**

*Autonomous Navigation & Terminal Guidance System for Tactical UAVs*

**SOFTWARE ARCHITECTURE DOCUMENT**

*Phase 1 --- Software-in-the-Loop Implementation*

  --------------------- -------------------------------------------------
  **Programme**         MicroMind / NanoCorteX

  **Document Type**     Software Architecture Document (SAD)

  **Version**           1.0 --- Phase 1 Closure

  **Date**              12 March 2026

  **Programme           Amit Anand (amitkr-anand)
  Director**            

  **Repository**        amitkr-anand/micromind-autonomy (branch: main)

  **Commit**            3c37d82

  **Spec Reference**    MicroMind Part Two V7.1 Live

  **SIL Target**        TASL (Tata Advanced Systems Limited)
  --------------------- -------------------------------------------------

**1 System Overview**

MicroMind / NanoCorteX is a bounded deterministic autonomy payload for
tactical UAVs operating in GNSS-denied, RF-contested environments. It is
designed as an autonomy payload --- a compute board that sits alongside
an existing autopilot (PX4 or equivalent) and provides mission-level
intelligence without modifying the flight controller.

The system is structured around three segments:

  ---------------- ---------------- -------------------------------------
  **Segment**      **Component**    **Role**

  **Commander      MicroMind-OS     Mission planning, signed mission
  Interface**                       envelope generation, pre-flight
                                    authority configuration

  **Onboard        MicroMind-X      Sensor fusion, EW observation,
  Mission                           unified state vector, hardware
  Computer**                        abstraction

  **Autonomy       NanoCorteX       Navigation, routing, terminal
  Engine**                          guidance, state machine, L10s-SE
                                    enforcement
  ---------------- ---------------- -------------------------------------

The host autopilot handles only vehicle stabilisation. NanoCorteX
provides guidance commands via MAVLink. No flight controller
modification is required --- the system is airframe-agnostic.

**1.1 Normative Test Scenario --- BCMP-1**

BCMP-1 defines the minimum operational corridor against which all Phase
1 acceptance criteria are evaluated:

  ---------------------- ------------------------------------------------
  **Parameter**          **Value**

  **Corridor length**    100 km (250 km validated in SIL)

  **GNSS environment**   Full denial from T+0

  **EW environment**     Adaptive jamming, site relocation during mission

  **Satellite overpass** One scheduled overpass --- terrain masking
                         required

  **Target**             Thermal signature with thermal decoy present

  **Terminal phase**     Last 10 seconds zero RF, L10s-SE enforced

  **Navigation limit     Lateral drift \< 100 m per 5 km segment
  (NAV-01)**             
  ---------------------- ------------------------------------------------

**2 Module Catalogue**

All production modules reside under core/. Simulation harnesses reside
under sim/. Scenarios under scenarios/. Tests under tests/. Dashboard
outputs under dashboard/.

**2.1 core/ins --- Inertial Navigation System**

  ---------------------- ------------ ----------------------------------------
  **File**               **Sprint**   **Responsibility**

  **mechanisation.py**   S0 + S10     Strapdown INS integration. Accepts IMU
                                      measurements (gyro + accel), propagates
                                      quaternion attitude, rotates body-frame
                                      acceleration to navigation frame,
                                      integrates velocity and position.
                                      Applies coriolis and gravity
                                      compensation. Contains S10-perf noise
                                      cache fix.

  **state.py**           S0           INSState dataclass: position (NED, m),
                                      velocity (m/s), attitude quaternion,
                                      accelerometer bias, gyroscope bias.
                                      Single state vector shared between
                                      mechanisation and ESKF.

  **trn_stub.py**        S3 + S9 +    Terrain-Referenced Navigation. Provides
                         S10          horizontal position corrections by
                                      matching radar altimeter readings
                                      against a pre-loaded DEM using 2D
                                      Normalised Cross-Correlation (numpy
                                      sliding_window_view). Returns
                                      TRNCorrection (north_offset_m,
                                      east_offset_m, confidence) or None.
                                      Internal Kalman removed in S9 --- pure
                                      measurement provider only.

  **imu_model.py**       S8-A         Three IMU noise models: STIM300 (Safran,
                                      primary), ADIS16505-3 (Analog Devices,
                                      budget), BASELINE (ideal). Parameters:
                                      ARW, VRW, bias instability, bias RW,
                                      scale factor ppm, misalignment. Exposes
                                      total_gyro() and total_accel() --- full
                                      noise timeseries as numpy arrays.
  ---------------------- ------------ ----------------------------------------

**2.2 core/ekf --- Error-State Kalman Filter**

  ------------------------ ------------ ----------------------------------------
  **File**                 **Sprint**   **Responsibility**

  **error_state_ekf.py**   S0 V2 + S9 + 15-state Error-State Kalman Filter.
                           S10          States: position error (3), velocity
                                        error (3), attitude error (3),
                                        accelerometer bias (3), gyroscope bias
                                        (3). Key methods: propagate(state, imu,
                                        dt) advances the error state;
                                        update_gnss(state, pos, trust_score)
                                        applies a GNSS position correction
                                        weighted by BIM trust score;
                                        update_trn(state, correction) applies a
                                        TRN correction. Q-matrix frozen to
                                        STIM300 datasheet values in S9. F and Q
                                        buffers pre-allocated in S10.
  ------------------------ ------------ ----------------------------------------

**2.3 core/bim --- Beacon & Integrity Monitor**

  ------------------- ------------ ----------------------------------------
  **File**            **Sprint**   **Responsibility**

  **bim.py**          S2           GNSS integrity monitor. Evaluates PDOP,
                                   Doppler deviation, signal-to-noise
                                   ratio, and cross-checks satellite
                                   constellation consistency. Outputs a
                                   continuous trust score (0.0--1.0) and a
                                   discrete trust state (Green / Amber /
                                   Red). Hysteresis prevents rapid state
                                   oscillation. Spoof detection latency ≤
                                   250 ms. Trust = Red (\< 0.1) triggers
                                   FSM transition to GNSS_DENIED.
  ------------------- ------------ ----------------------------------------

**2.4 core/ew_engine --- Electronic Warfare Engine**

  ------------------- ------------ ----------------------------------------
  **File**            **Sprint**   **Responsibility**

  **ew_engine.py**    S4           Produces and maintains a 2D EW cost map
                                   over the mission corridor. Ingests SDR
                                   observations (simulated in SIL), runs
                                   spectrogram analysis and DBSCAN
                                   clustering to identify jammer sites and
                                   emission patterns. Cost map update
                                   latency ≤ 500 ms. Cost map feeds the
                                   Hybrid A\* route planner. Also tracks
                                   jammer site relocation across time.
  ------------------- ------------ ----------------------------------------

**2.5 core/route_planner --- Hybrid A\* Route Planner**

  --------------------- ------------ ----------------------------------------
  **File**              **Sprint**   **Responsibility**

  **hybrid_astar.py**   S4           Continuous-state Hybrid A\* path planner
                                     over the EW cost map. Replans route when
                                     cost map changes. Replan latency ≤ 1
                                     second. Respects mission envelope
                                     boundary polygon. Outputs a waypoint
                                     sequence passed to the autopilot
                                     interface.
  --------------------- ------------ ----------------------------------------

**2.6 core/state_machine --- NanoCorteX FSM**

  ---------------------- ------------ ----------------------------------------
  **File**               **Sprint**   **Responsibility**

  **state_machine.py**   S1           7-state deterministic FSM. States: ST-01
                                      NOMINAL, ST-02 EW_AWARE, ST-03
                                      GNSS_DENIED, ST-04 SILENT_INGRESS, ST-05
                                      SHM_ACTIVE, ST-06 ABORT, ST-07
                                      MISSION_FREEZE. Each transition has a
                                      defined guard condition and maximum
                                      latency (≤ 2 seconds, NFR-002). No
                                      transitions without guard satisfaction.
                                      No free-form AI decisions.
  ---------------------- ------------ ----------------------------------------

**2.7 core/dmrl --- Dual-Mode Recognition & Lock**

  ------------------- ------------ ----------------------------------------
  **File**            **Sprint**   **Responsibility**

  **dmrl_stub.py**    S5           Terminal guidance recognition and lock.
                                   Multi-frame EO/IR target association
                                   using thermal dissipation model and
                                   CNN-based decoy signature rejection.
                                   Outputs DMRLResult with lock_confidence,
                                   decoy_flag, aimpoint_correction_deg.
                                   Lock threshold ≥ 0.85; decoy abort ≥
                                   0.80 over 3 frames; min dwell 5 frames @
                                   25 FPS; aimpoint correction limit ±15°;
                                   reacquisition timeout 1.5 s.
  ------------------- ------------ ----------------------------------------

**2.8 core/l10s_se --- Last-10-Second Safety Envelope**

  ------------------- ------------ ----------------------------------------
  **File**            **Sprint**   **Responsibility**

  **l10s_se.py**      S5           Final-phase rules-of-engagement
                                   enforcer. Operates in ST-05 (SHM_ACTIVE)
                                   and ST-06 (ABORT). Deterministic
                                   decision tree --- no ML. Civilian
                                   detection threshold ≥ 0.70 triggers
                                   abort. Decision timeout ≤ 2 seconds.
                                   Outputs L10sOutput with proceed/abort
                                   decision and rationale string.
  ------------------- ------------ ----------------------------------------

**2.9 core/zpi + core/cems --- Phase-2 Modules (Frozen)**

  ------------------- ------------ ----------------------------------------
  **File**            **Sprint**   **Responsibility**

  **zpi/zpi.py**      S6           Zero-RF hop plan protocol.
                                   Low-probability-of-intercept burst
                                   scheduler for telemetry and EW flash
                                   updates. Hop plan derived via
                                   HKDF-SHA256 from mission key. Packet ≤
                                   256 bytes. Phase-2 frozen --- not used
                                   in Phase 1 demo.

  **cems/cems.py**    S6           Cooperative EW Sharing. Peer-to-peer EW
                                   intelligence merge between UAVs.
                                   Confidence-weighted spatial-temporal
                                   merge; temporal decay; replay window 30
                                   s. Phase-2 frozen --- not used in Phase
                                   1 demo.
  ------------------- ------------ ----------------------------------------

**2.10 core/math + core/constants --- Foundations**

  ------------------------ ------------ ----------------------------------------
  **File**                 **Sprint**   **Responsibility**

  **math/quaternion.py**   S0           Quaternion algebra library:
                                        multiplication, conjugate,
                                        normalisation, rotation matrix
                                        conversion, slerp. Used throughout
                                        mechanisation and ESKF for attitude
                                        representation.

  **constants.py**         S0           Physical constants: WGS-84 Earth
                                        parameters, standard gravity, speed of
                                        light, atmospheric model parameters.
                                        Single source of truth --- never
                                        duplicated in other modules.
  ------------------------ ------------ ----------------------------------------

**3 Simulation Modules**

Simulation modules are not deployed to hardware. They provide stimulus
inputs and scenario orchestration for SIL validation. All SIL results
are deterministic given a fixed random seed.

  -------------------------------- ------------ ---------------------------------------
  **File**                         **Sprint**   **Role**

  **sim/als250_nav_sim.py**        S8-C + S9    Primary 250 km corridor navigation
                                                simulation. Accepts \--imu (STIM300 /
                                                ADIS16505_3 / BASELINE), \--seed,
                                                \--duration (seconds), \--out (results
                                                directory). Runs at 200 Hz. Calls
                                                mechanisation → ESKF propagate → TRN
                                                (every 1.5 km) → ESKF update. Outputs
                                                .npy position arrays and meta JSON.
                                                search_pad_px=25 (critical --- must not
                                                be changed).

  **sim/nav_scenario.py**          S3           Short-corridor navigation scenario
                                                (10--50 km). Used in S3/S8 unit-level
                                                acceptance tests. Simpler harness than
                                                als250 --- no DEM, no parallel
                                                execution.

  **sim/eskf_simulation.py**       S0           ESKF standalone simulation. Drives ESKF
                                                with synthetic IMU and GNSS
                                                measurements. Used for filter tuning
                                                and unit verification.

  **sim/gnss_spoof_injector.py**   S2           Injects spoofed GNSS measurements at
                                                configurable latency and offset. Used
                                                with BIM tests to verify spoof
                                                detection ≤ 250 ms.

  **sim/bcmp1_ew_sim.py**          S4           BCMP-1 EW scenario simulation.
                                                Generates jammer site positions,
                                                emission schedules, and SDR
                                                observations for EW engine and Hybrid
                                                A\* testing.

  **sim/bcmp1_cems_sim.py**        S6           Multi-UAV CEMS simulation. Generates
                                                peer EW packets and tests cooperative
                                                merge. Phase-2 frozen.
  -------------------------------- ------------ ---------------------------------------

**3.1 Scenario Runners**

  --------------------------------------- ------------ ---------------------------------------
  **File**                                **Sprint**   **Role**

  **scenarios/bcmp1/bcmp1_runner.py**     S5 + S8 + S9 Full BCMP-1 scenario orchestrator.
                                                       Entry point: run_bcmp1(seed,
                                                       kpi_log_path). Wires all core modules:
                                                       FSM → BIM → ESKF → mechanisation → TRN
                                                       → EW engine → Hybrid A\* → DMRL →
                                                       L10s-SE. Outputs KPI log JSON.

  **scenarios/bcmp1/bcmp1_scenario.py**   S1           BCMP-1 scenario parameter definition.
                                                       Corridor geometry, GNSS denial zones,
                                                       jammer schedules, satellite overpass
                                                       time, target position, decoy position.

  **run_als250_parallel.py**              S10-4        Parallel runner for three IMU
                                                       configurations. Launches one
                                                       subprocess.Popen per IMU --- avoids MKL
                                                       fork deadlock. Used to generate the
                                                       S8-D TASL drift chart.
  --------------------------------------- ------------ ---------------------------------------

**4 Data Flow & Call Sequences**

**4.1 Navigation Loop (200 Hz --- Core Call Sequence)**

The navigation loop runs at 200 Hz inside als250_nav_sim.py. The correct
propagation order is critical --- violating it (as in pre-S9 code)
causes filter divergence:

> for step in range(n_steps):
>
> \# 1. Build IMU measurement (cached noise arrays --- O(1) after S10)
>
> gyro = gyro_bias \* (1 + sf) + imu_noise.\_gyro_cache\[step\]
>
> accel = accel_bias + imu_noise.\_accel_cache\[step\]
>
> \# 2. ESKF error propagation (predict step)
>
> eskf.propagate(state, imu_measurement, dt)
>
> \# 3. INS mechanisation (state propagation)
>
> state = mechanisation.propagate(state, gyro, accel, dt)
>
> \# 4. TRN correction (every CORRECTION_INTERVAL metres)
>
> if ground_track_m \>= next_trn_m:
>
> correction = trn.update(ins_north, ins_east,
>
> true_north, true_east,
>
> ground_track_m)
>
> if correction and correction.confidence \>= NCC_THRESHOLD:
>
> eskf.update_trn(state, correction)
>
> next_trn_m += CORRECTION_INTERVAL

**4.2 BCMP-1 Full Scenario Call Graph**

The full BCMP-1 runner wires all subsystems. Calls are sequential within
each 200 Hz tick; EW and route planning are event-driven on cost map
change:

> bcmp1_runner.run_bcmp1(seed, kpi_log_path)
>
> ├─ FSM.tick() \# Evaluate state transitions
>
> ├─ BIM.evaluate(gnss_obs) \# → trust_score, trust_state
>
> │ └─ If trust_state == RED:
>
> │ FSM.transition(GNSS_DENIED)
>
> ├─ ESKF.propagate(state, imu, dt) \# Error prediction
>
> ├─ mechanisation.propagate(\...) \# INS state update
>
> ├─ TRN.update(\...) \# Every 1.5 km
>
> │ └─ ESKF.update_trn(correction) \# Position correction
>
> ├─ EWEngine.update(sdr_obs) \# → cost_map (≤ 500 ms)
>
> │ └─ If cost_map_changed:
>
> │ HybridAstar.replan(cost_map) \# → waypoints (≤ 1 s)
>
> ├─ \[Terminal phase only\]
>
> │ ├─ DMRL.process(frame) \# → DMRLResult
>
> │ └─ L10sSE.decide(dmrl, context) \# → L10sOutput
>
> └─ MissionLog.record(kpi_snapshot) \# ≥ 99% completeness

**4.3 TRNStub Internal Design**

TRNStub is a pure measurement provider since the S9 architectural fix.
It does not maintain any internal state estimate or Kalman filter:

> TRNStub.\_\_init\_\_(dem, radar, ncc_threshold=0.45, search_pad_px=25)
>
> ├─ DEMProvider(seed=7) \# Pre-loaded elevation grid
>
> └─ RadarAltimeterSim(dem, seed=99) \# Simulated radar altimeter
>
> TRNStub.update(ins_north, ins_east, true_north, true_east,
>
> ground_track_m, timestamp_s) → TRNCorrection \| None
>
> ├─ radar.measure(true_north, true_east) \# Simulated altimeter reading
>
> ├─ dem.get_patch(ins_north, ins_east, \# DEM tile at INS estimate
>
> │ search_pad_px)
>
> ├─ ncc_2d(radar_patch, dem_patch) \# 2D NCC (numpy vectorised)
>
> │ sliding_window_view → single matrix op, no Python loops
>
> ├─ If peak_ncc \>= ncc_threshold (0.45):
>
> │ return TRNCorrection(north_m, east_m, confidence=peak_ncc)
>
> └─ Else: return None (suppressed --- Hybrid-3 gate counts consecutive)

**4.4 BIM Trust Scoring**

BIM outputs a scalar trust score and a discrete state used by both the
ESKF (as a weight on GNSS update) and the FSM (as a state transition
trigger):

> BIM.evaluate(gnss_obs) → (trust_score: float, trust_state: str)
>
> ├─ pdop_score = f(obs.pdop) \# PDOP threshold test
>
> ├─ doppler_score = f(obs.doppler_dev) \# Doppler deviation test
>
> ├─ snr_score = f(obs.snr) \# Signal quality test
>
> ├─ cross_check = constellation_consistency(obs)
>
> ├─ raw_score = weighted_mean(pdop, doppler, snr, cross_check)
>
> ├─ trust_score = hysteresis_filter(raw_score, prev_score)
>
> └─ trust_state: GREEN (≥0.7) \| AMBER (0.1--0.7) \| RED (\<0.1)
>
> ESKF.update_gnss(state, pos, trust_score) \# trust_score weights R
> matrix

**5 Key Module Interfaces**

All public interfaces are stable as of Phase 1 closure. Breaking changes
require a spec update and regression re-run.

  ------------------------ -------------------- ----------------------- ---------
  **Interface**            **Module**           **Signature / Output**  **FR**

  **ESKF propagate**       error_state_ekf.py   propagate(state, imu,   ---
                                                dt) → INSState          

  **ESKF GNSS update**     error_state_ekf.py   update_gnss(state, pos, FR-101
                                                trust_score) → None     

  **ESKF TRN update**      error_state_ekf.py   update_trn(state,       FR-107
                                                correction) → None      

  **BIM trust score**      bim.py               evaluate(gnss_obs) →    FR-101
                                                (float, str)            

  **TRN correction**       trn_stub.py          update(N, E, tN, tE,    FR-107
                                                track, t) →             
                                                TRNCorrection \| None   

  **FSM transition**       state_machine.py     transition(new_state)   NFR-002
                                                --- guard enforced      

  **EW cost map**          ew_engine.py         update(sdr_obs) →       EW-01
                                                CostMap2D               

  **Route replan**         hybrid_astar.py      replan(cost_map) →      EW-02
                                                List\[Waypoint\]        

  **DMRL output**          dmrl_stub.py         process(frame) →        FR-103
                                                DMRLResult              

  **L10s-SE decision**     l10s_se.py           decide(dmrl, context) → FR-105
                                                L10sOutput              

  **BCMP-1 runner**        bcmp1_runner.py      run_bcmp1(seed,         All
                                                kpi_log_path) → KPILog  
  ------------------------ -------------------- ----------------------- ---------

**6 Acceptance Boundary Conditions**

These constants are frozen. Any change requires a formal spec update to
MicroMind Part Two and a full regression re-run.

  --------------------- ------------- ----------------------- --------------------
  **Constant**          **Value**     **Module**              **FR / NFR**

  **BIM trust → Red**   \< 0.1        bim.py                  FR-101

  **Spoof detection     ≤ 250 ms      bim.py                  FR-101
  latency**                                                   

  **FSM transition      ≤ 2 s         state_machine.py        NFR-002
  latency**                                                   

  **TRN correction      every 1,500 m trn_stub.py             FR-107
  interval**                                                  

  **NCC correlation     ≥ 0.45        trn_stub.py             FR-107
  threshold**                                                 

  **search_pad_px**     25 (critical) als250_nav_sim.py       FR-107

  **Navigation drift    \< 100 m / 5  nav_scenario.py         NAV-01
  limit**               km                                    

  **EW cost map         ≤ 500 ms      ew_engine.py            EW-01
  update**                                                    

  **Route replan        ≤ 1 s         hybrid_astar.py         EW-02
  latency**                                                   

  **DMRL lock           ≥ 0.85        dmrl_stub.py            FR-103
  confidence**                                                

  **Decoy abort         ≥ 0.80 / 3    dmrl_stub.py            FR-103
  threshold**           frames                                

  **Min dwell frames**  5 @ 25 FPS    dmrl_stub.py            FR-103

  **Aimpoint correction ±15°          dmrl_stub.py            FR-103
  limit**                                                     

  **Reacquisition       1.5 s         dmrl_stub.py            FR-103
  timeout**                                                   

  **L10s decision       ≤ 2 s         l10s_se.py              FR-105
  timeout**                                                   

  **Civilian detect     ≥ 0.70        l10s_se.py              FR-105
  threshold**                                                 

  **Log completeness**  ≥ 99%         mission_log_schema.py   NFR-013

  **\_ACC_BIAS_RW**     9.81×10⁻⁷     error_state_ekf.py      STIM300 TS1524
                        m/s²/√s                               

  **\_GYRO_BIAS_RW**    4.04×10⁻⁸     error_state_ekf.py      STIM300 TS1524
                        rad/s/√s                              

  **\_POS_DRIFT_PSD**   1.0 m/√s      error_state_ekf.py      Process noise
  --------------------- ------------- ----------------------- --------------------

**7 Test Architecture**

The regression suite contains 222 gates across 4 test runners. All gates
must pass before any sprint can be declared closed.

  ------------------------------ ------------------ ----------- --------------------
  **Test File**                  **Runner**         **Gates**   **Coverage**

  test_sprint_s1_acceptance.py   run_s5_tests.py    S1 subset   FSM, SimClock,
                                                                MissionLog, BCMP-1
                                                                scenario

  test_sprint_s2_acceptance.py   run_s5_tests.py    S2 subset   BIM trust scoring,
                                                                spoof injection

  test_sprint_s3_acceptance.py   run_s5_tests.py    S3 subset   TRN stub, nav
                                                                scenario

  test_sprint_s4_acceptance.py   run_s5_tests.py    S4 subset   EW engine, Hybrid
                                                                A\*

  test_s5_dmrl.py                run_s5_tests.py    S5 subset   DMRL lock, decoy
                                                                abort, dwell

  test_s5_l10s_se.py             run_s5_tests.py    S5 subset   L10s-SE civilian
                                                                gate, timeout

  test_s5_bcmp1_runner.py        run_s5_tests.py    S5 subset   Full BCMP-1 KPI
                                                                gates

  test_s6_zpi_cems.py            direct pytest      36/36       ZPI hop plan, CEMS
                                                                merge

  test_s8a_imu_model.py          run_s8_tests.py    16/16       IMU noise model
                                                                parameters

  test_s8b_mechanisation.py      run_s8_tests.py    21/21       Strapdown
                                                                integration accuracy

  test_s8c_als250_nav_sim.py     run_s8_tests.py    S8-C subset 50 km corridor
                                                                NAV-01

  test_s8e_bcmp1_runner_imu.py   run_s8_tests.py    S8-E subset BCMP-1 with IMU
                                                                fidelity models

  test_s9_nav01_pass.py          direct pytest      10/10 + 2   TRN+ESKF
                                                    skip        architecture, 150 km
                                                                NAV-01
  ------------------------------ ------------------ ----------- --------------------

**7.1 Running the Full Suite**

> conda activate micromind-autonomy
>
> cd \~/micromind-autonomy
>
> python run_s5_tests.py \# 111/111
>
> python tests/test_s6_zpi_cems.py \# 36/36
>
> python run_s8_tests.py \# 68/68
>
> pytest tests/test_s9_nav01_pass.py \# 10 pass, 2 skip

**8 Architectural Decisions & Rationale**

  ----------- -------------------- ----------------------- -----------------------
  **ID**      **Decision**         **Rationale**           **Consequence**

  **AD-01**   **Single ESKF --- no Two independent         TRNStub.update()
              TRN internal         estimators receiving    returns raw offsets
              filter**             the same measurements   only. ESKF applies
                                   corrupt each other. One correction via
                                   estimator, one          update_trn().
                                   measurement provider is Implemented S9.
                                   the correct pattern.    

  **AD-02**   **2D NCC over 1D NCC 2D approach: 2 DEM      search_pad_px=25 is a
              for TRN**            reads + one numpy       critical parameter ---
                                   matrix op per           larger values expand
                                   correction. 1D          search area
                                   approach: \~33,640      quadratically.
                                   Python-level DEM        
                                   lookups. NAV-01: 33.3 m 
                                   (2D) vs 421.9 m failure 
                                   (1D).                   

  **AD-03**   **subprocess.Popen   numpy/BLAS MKL fork     Each IMU run is a fully
              for parallelism (not deadlock on Linux makes independent OS process.
              multiprocessing)**   multiprocessing         Output files are
                                   unreliable.             independent .npy
                                   subprocess.Popen gives  arrays.
                                   isolated Python         
                                   interpreters --- no     
                                   shared memory, no       
                                   deadlock.               

  **AD-04**   **Noise array        total_gyro() and        Cache lives on the
              caching in           total_accel() are pure  imu_noise object.
              mechanisation loop** functions --- same      Multi-run re-use
                                   output given same seed. requires cache
                                   Calling them per-step   invalidation ---
                                   reconstructs (n_steps,  handled by subprocess
                                   3) arrays at O(n²)      isolation.
                                   cost. Cache on first    
                                   call is safe and        
                                   delivers 134× speedup.  

  **AD-05**   **Bounded            Tactical systems        L10s-SE uses a decision
              deterministic state  require deterministic,  tree, not a neural
              machine (no learned  auditable behaviour. No network. DMRL uses CNN
              policy)**            ML in the navigation or only for decoy
                                   engagement path. Demo   rejection ---
                                   Fork constraint (28 Feb lock/abort thresholds
                                   2026) enforced this.    are hardcoded.

  **AD-06**   **Autonomy as a      Airframe-agnostic       NanoCorteX outputs
              payload --- no       design allows           MAVLink guidance
              autopilot            integration with any    commands. The autopilot
              modification**       PX4-compatible          handles stabilisation
                                   platform. TASL hardware only.
                                   decision deferred       
                                   post-SIL.  
