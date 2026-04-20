**MicroMind**

**Pre-HIL Software Completion Plan**

*v1.2 — Technical Director Review — Accelerated Integration Phase*

| **Parameter** | **Value** |
| --- | --- |
| Document version | v1.2 — Final Pre-HIL Integration Plan. Merged from v1.0 + Amendment Record ADD-01..ADD-11 + v1.2 targeted insertions |
| Date | 26 March 2026 |
| Basis | SIL Phase 1 complete — S-NEP-10, BCMP-1 two-theatre, commit 405ec1d, 332/332 tests |
| Scope | Integration-layer software only. No hardware. No ML. No core modifications. |
| Critical risk | PX4 OFFBOARD integration — previous attempt (Dec) failed. Primary risk area. |
| Purpose | Full execution plan before any code is written. Go/No-Go gate. |
| v1.2 | All v1.1 content preserved. Six targeted insertions: control authority clarification, estimator ownership, EKF2 validation check, non-blocking queue constraint, determinism verification link, PX4 mode overlay. Status: FINAL. |

**This document is produced BEFORE any code is written. Its purpose is to prevent a second PX4 integration failure by defining the exact architecture, sequence, and validation criteria before implementation begins.**

# **Part 1 — Assessment: OEM Integration Readiness Lens**

Evaluation question: Is MicroMind ready to be perceived as an integration-ready autonomy payload that can be tested in an OEM lab immediately?

## **1.1  Validation Points**

| **#** | **Statement** | **Status** | **Basis** |
| --- | --- | --- | --- |
| V-1 | Navigation core is integration-ready. ESKF, TRN, VIO, BIM interfaces are clean module boundaries. No OEM needs to understand their internals to integrate. | **Green** | 332/332 tests, S-NEP-04 interface contracts, S-NEP-10 enforcement. All interfaces documented. |
| V-2 | The enforcement layer (S-NEP-10 E-1..E-5) demonstrates that the system actively prevents unsafe decisions. This is a credibility differentiator, not just a software feature. | **Green** | Two-theatre validation. TERM KPIs only recorded from NOMINAL position. FI-W2 terminal deferral confirmed. |
| V-3 | Driver abstraction architecture is the correct approach for replaceability. Abstract base classes per driver type, consistent with S-NEP-04 modular design. | **Green** | Pre-HIL assessment document approved. Module boundary pattern matches V6 §5.1 retrofit model. |
| V-4 | The problem being solved is correct: integration layer only. No new algorithm development. No scope expansion. PX4 wiring + driver abstraction + config switching is the entire scope. | **Green** | SPRINT_STATUS deferred list. Two consecutive TD reviews confirm scope. |
| V-5 | The IFM failure mode library from S-NEP-04 is directly applicable to the live pipeline. IFM-01, IFM-02, IFM-04 are all live-pipeline risks with documented detection signals. | **Green** | S-NEP-04 §3 Integration Failure Modes. These are system-level failures invisible in pure SIL. |

## **1.2  What Is Strong and Must Not Be Touched**

| **Module** | **Why it must not change** |
| --- | --- |
| core/ekf/error_state_ekf.py | Frozen constants and validated interface. 332 gates calibrated to this baseline. S-NEP-09 anchored here. |
| core/fusion/vio_mode.py | Mode Integrity Invariant. VIONavigationMode transition latency confirmed 1-step. S-NEP-09 re-run required if changed. |
| core/fusion/frame_utils.py | ENU→NED transform matrices validated in S-NEP-04. Do not rewrite. Call existing functions only. |
| scenarios/bcmp1/bcmp1_runner.py E-1..E-5 | Committed enforcement. Validated two-theatre. TERM KPIs depend on this layer being active. |
| Frozen estimator constants | _ACC_BIAS_RW, _GYRO_BIAS_RW, _POS_DRIFT_PSD, _GNSS_R_NOMINAL. TD approval required to change. |
| tests/ (332 gates) | No gate modified to pass. All 332 pass unchanged after every integration addition. |

## **1.3  What Is Missing for OEM Confidence**

| **Gap** | **Why OEM notices it** | **What it blocks** |
| --- | --- | --- |
| PX4 / MAVLink not wired | First question in any lab: 'show me the vehicle responding.' Currently unanswerable. | Demo. HIL. TASL partnership credibility. |
| No real-time loop | System runs in batch simulation steps, not a continuous 200Hz navigation loop feeding live outputs. | Performance claims. Compute feasibility proof. |
| No hardware timestamp binding | All timestamps from dataset or simulation clock. Live sensors produce hardware timestamps that will desync silently. | IFM-01 (timestamp misalignment) — highest-consequence silent failure per S-NEP-04. |
| No compute profiling | No measurement of CPU/memory at 200Hz IMU rate on any hardware. | OEM cannot assess whether their compute platform is sufficient. |
| No Gazebo visual validation | No simulated 3D environment. Behaviour under VIO outage is log-only, not visually observable. | Demo impact. Reviewer trust. |

## **1.4  Where PX4 Integration Will Fail Again (Root Cause Analysis)**

The December attempt failed. The following are the most probable root causes based on standard PX4 OFFBOARD integration failure patterns, the S-NEP-04 IFM library, and the current codebase state.

| **Failure mode** | **Root cause** | **Why it is hard to see** | **Definitive fix** |
| --- | --- | --- | --- |
| OFFBOARD mode exits silently | Heartbeat not sent at ≥2 Hz. PX4 reverts to MANUAL or STABILIZE without error if any gap >500ms. | PX4 logs show mode change but not why. From MicroMind side: setpoints sent, no error returned. Looks like 'commands ignored.' | Dedicated heartbeat thread at 2Hz, completely independent of navigation loop. Never blocked by compute. Validated: OFFBOARD held for 60s. |
| Setpoints sent before OFFBOARD engaged | arming and mode_set sent then setpoints start immediately. PX4 requires setpoints already flowing BEFORE mode switch. | Mode switch appears to succeed (MAV_RESULT_ACCEPTED), but vehicle immediately exits OFFBOARD because setpoint stream had a gap at the critical moment. | Pre-stream: send at least 2s of setpoints at ≥20Hz before requesting mode change. Explicit wait for LOCAL_POSITION_NED valid. Only then send mode change. |
| Frame mismatch | Setpoints sent in ENU or body frame. POSITION_TARGET_LOCAL_NED expects NED. Z-sign flip produces inverted altitude commands. | Vehicle climbs when it should descend. Developer assumes flight controller bug. The frame in the MAVLink message is specified by coordinate_frame field that must be explicitly set to MAV_FRAME_LOCAL_NED (1). | Explicitly set coordinate_frame=1 (MAV_FRAME_LOCAL_NED). Unit test: send known NED setpoint, verify PX4 moves in correct direction in Gazebo. |
| Setpoint rate too low | Setpoints sent at <5Hz (only on decision events). PX4 requires continuous stream at ≥20Hz. | Vehicle enters OFFBOARD briefly then exits. Appears as heartbeat issue. Rate logging is required to distinguish. | Dedicated setpoint loop at 20Hz, interpolating between navigation waypoints. Independent of decision event rate. |
| Timestamp mismatch | Setpoints carry MicroMind internal timestamps; PX4 expects its own boot-time microseconds. Some implementations set time_boot_ms incorrectly, causing PX4 to ignore or rate-limit messages. | No visible error. PX4 receives messages but does not act. Looks like command rejection. | Use PX4-synced time_boot_ms. Call SYSTEM_TIME to get PX4 time on connection. |

## **1.5  Where Silent Failures Occur**

| **Location** | **Failure** | **Why silent** | **Detection** |
| --- | --- | --- | --- |
| VIO driver → ESKF | ENU input without NED rotation (IFM-02) | ESKF accepts any 3-vector. Heading corrupt. No flag. | C-3 frame assertion at ESKF input gate. |
| Sensor timestamp → ESKF | Non-monotonic or hardware-software timestamp gap (IFM-01) | ESKF propagates correctly but fuses stale measurement. | Monotonicity guard + IFM-01 log (S-NEP-04 §1.5). |
| MAVLink bridge | PX4 OFFBOARD exits due to heartbeat gap | No error from pymavlink. OFFBOARD mode changes in PX4 log. | Monitor PX4 mode state in receive loop. Alert immediately on mode change. |
| MAVLink bridge | Wrong coordinate frame in POSITION_TARGET_LOCAL_NED | MAV_RESULT_ACCEPTED returned. Vehicle moves in wrong direction. | Frame field explicitly set. Gazebo visual validation as primary check. |
| VIO estimator reset → ESKF | Large pose jump fused as valid (IFM-04) | Innovation gate may not trigger if jump is within gate. State corrupted. | Monitor consecutive pose delta. If >1.0m in one step: classify reset, reject. |
| LiveLogger | Queue overflow drops events | No error raised. Events silently lost. | Drop counter. RC-7 <0.5% drop test. |

## **1.6  Go / No-Go**

**GO — with four mandatory pre-code conditions.  (1) PX4 OFFBOARD failure mode analysis reviewed and signed off by TD before Phase 1.5 begins. (2) ROS2 vs direct-call middleware decision documented. (3) Time reference strategy documented and agreed. (4) Module boundary diagram: which existing files are touched (none) vs new files only.  The December PX4 failure must be treated as a lesson, not a setback. The root causes are known and fixable. The approach is correct. The risks are now explicitly enumerated. Execution may proceed.**

## **1.7  Control Authority and System Positioning**

This section defines who is flying the vehicle. It must be unambiguous before any integration work begins.

**MicroMind is an external guidance, navigation, and enforcement module. It is not a flight controller and does not replace one.  PX4 (or equivalent autopilot) remains fully responsible for: •  Vehicle stabilisation (attitude control, rate control) •  Actuation (motors, servos, ESC management) •  Safety enforcement (geofence, failsafe, arming checks)  MicroMind****'****s role in the control authority chain: •  Computes position and navigation state (ESKF, TRN, VIO) •  Applies mission enforcement (S-NEP-10 E-1..E-5) •  Generates trajectory setpoints (NED position + heading) •  Transmits setpoints via MAVLink OFFBOARD interface  PX4 closes the guidance loop: it receives MicroMind setpoints and executes them using its own inner-loop controllers.  Answer: PX4 is flying the vehicle. MicroMind is deciding where it goes and enforcing mission rules. The two systems have a clean authority boundary at the MAVLink interface. No modification of PX4 internals is required or permitted.**

| **Layer** | **System** | **Responsibility** | **Interface** |
| --- | --- | --- | --- |
| Guidance (mission) | MicroMind | Navigation state, enforcement, setpoint generation | MAVLink OFFBOARD setpoints (output) |
| Control (flight) | PX4 | Stabilisation, actuation, inner-loop control | MAVLink OFFBOARD setpoints (input) |
| Sensing (state) | MicroMind-X drivers | IMU, VIO, GNSS, TRN, EO/IR, SDR ingestion | Abstract driver interfaces |
| Vehicle | Airframe + motors | Physical execution | PWM / CAN from PX4 |

**No modification of PX4 is required for MicroMind integration. MicroMind operates entirely outside PX4 and communicates only via the standard MAVLink OFFBOARD protocol.**

*Clarification: MicroMind does not generate actuator-level commands (attitude, thrust, or motor control). It produces only position/trajectory setpoints. All low-level control authority remains fully within PX4.*

# **Part 2 — OEM Challenges → Proof Artefacts**

Each expected OEM challenge is mapped to a specific demonstrable artefact. This is the demo backbone.

| **OEM Challenge** | **Proof Artefact** | **How Demonstrated** |
| --- | --- | --- |
| 'Show me the vehicle responding to MicroMind.' | PX4 SITL + Gazebo: vehicle follows a 500m NED waypoint sequence. OFFBOARD mode held for 60s. | Live demo: start MicroMind, start SITL, engage OFFBOARD, watch Gazebo vehicle move. Screen-recorded. |
| 'Does it stay in control if comms degrade?' | Failure injection: VIO outage injected mid-flight. Gazebo vehicle switches to degraded mode, continues on INS, recovers when VIO resumes. | Live demo: show mode change in terminal log + Gazebo visual + drift_envelope_m growing then recovering. |
| 'How fast is the control loop?' | Latency log: timestamps at each pipeline stage (sensor → state → decision → MAVLink send). P95 latency <50ms. Setpoint rate ≥20Hz confirmed. | CSV log played back as chart: latency per step, setpoint rate, CPU usage over 60s run. |
| 'Can I replace the IMU with my sensor?' | Config switch demo: change pipeline.yaml imu_source from sim to stub. Re-run. System starts, produces clean error pointing to unconnected hardware. No code change. | Terminal output showing config load, driver substitution, clean error message. Code diff showing zero changes to ESKF. |
| 'Is the frame handling correct? We use NED.' | Frame assertion test: inject ENU vector at ESKF input, show FrameViolationError. Then inject NED vector, show it passes. | Unit test run live. One command, two inputs, two outputs. Readable test name. |
| 'What compute does it need?' | CPU/memory profiling during 60s live pipeline run. Reported as: avg CPU%, peak CPU%, avg memory MB, worst-case step latency. | profiling_report.txt or terminal output during demo. Compared against Jetson Orin spec. |
| 'How do timestamps work with our hardware?' | TimeReference module: show monotonicity gate catching a non-monotonic timestamp, logging IFM-01, and continuing on valid data. Show PX4 sync call on connection. | Unit test + live log showing IFM-01 event with injected bad timestamp. Clean timestamps confirmed. |
| 'Is the audit log complete?' | Live logger demo: run 60s pipeline, show live log file growing, confirm schema 08.1 event structure, show drop rate <0.5%. | tail -f logfile during demo. Post-run: event count, drop count, schema validation. |
| 'Can this work without ROS2 on our platform?' | Direct-call architecture: demonstrate full pipeline with zero ROS2 dependency. python -c 'import core.ekf.error_state_ekf' with no ROS2 installed. Clean import = no dependency. | Clean import confirmed. If thin bridge chosen: show Python core runs standalone, bridge is additive. |
| 'What does BCMP-1 actually prove?' | Two-theatre dossier: 11/11 KPIs, six failure injections, enforcement events logged, both theatres passing. | MicroMind_BCMP1_TwoTheatre_TechnicalDossier.docx already complete. Walk through Annex A evidence chain. |
| 'Who owns control authority? Is MicroMind modifying PX4?' | Continuous OFFBOARD control driven solely by MicroMind setpoints. PX4 internal mission logic disabled. Control authority boundary visible in Section 1.7 and live in Gazebo. | Step 1: Show PX4 in OFFBOARD mode with MicroMind as sole setpoint source. Step 2: Disable MicroMind setpoint stream — PX4 exits OFFBOARD (expected, correct). Step 3: Resume MicroMind — vehicle re-engages trajectory. Step 4: Show zero modifications to any PX4 source file (git diff on PX4 repo = empty). Authority boundary is unambiguous. |

# **Part 3 — Complete Software Pipeline Definition**

## **A. Input Pipelines**

| **Module** | **Current state** | **Missing** | **Deliverable** |
| --- | --- | --- | --- |
| A-1  IMU driver | STIM300 noise model. Fixed-step 200Hz simulation. | Abstract base class. Config-driven source switch. Real SPI/UART driver stub (NotImplementedError). Interface: {timestamp_ns, gyro_xyz_rad_s, acc_xyz_m_s2}. | imu_driver.py: IMUDriver(ABC) + SimIMUDriver + RealIMUDriver(stub). Validated: ESKF receives identical structure from both paths. |
| A-2  VIO live ingestion | EuRoC .npy files loaded offline. frame_utils.py ENU→NED exists. | OfflineVIODriver wrapping existing .npy loader. LiveVIODriver stub. Per frame: ENU→NED applied, covariance guard, monotonicity gate per IFM-01. | vio_driver.py: VIODriver(ABC) + OfflineVIODriver + LiveVIODriver(stub). All S-NEP-04..S-NEP-09 tests pass through OfflineVIODriver. |
| A-3  GNSS ingestion | GNSS spoof injector. BIM trust scorer. No NMEA parser. | SimGNSSDriver wrapping existing injector. NMEA stub parser. GNSSMeasurement struct. BIM hook unchanged. Real driver stub. | gnss_driver.py: GNSSDriver(ABC) + SimGNSSDriver + RealGNSSDriver(NMEA stub). BIM.update() called from driver. |
| A-4  TRN / RADALT interface | TRNStub + DEMProvider. No RADALT abstraction. | RADALTDriver separating altitude source from terrain correlation. Interface: {timestamp_ns, agl_m, validity_flag}. TRN core unchanged. | radalt_driver.py: RADALTDriver(ABC) + SimRADALTDriver (wraps DEMProvider) + RealRADALTDriver(stub). |
| A-5  EO/IR pipeline placeholder | generate_synthetic_scene() in dmrl_stub.py. | Video ingest interface contract only. Expected input: {frame_timestamp_ns, thermal_frame_16bit, width, height, radiometric_scale}. No ML. Synthetic scene remains default. | eoir_driver.py: EOIRDriver(ABC) + SyntheticEOIRDriver + RealEOIRDriver(stub, NotImplementedError). No DMRL changes. |
| A-6  SDR pipeline placeholder | _EWEngineStub: random latency draw. | RF ingest interface contract only. Expected input: {iq_sample_rate_hz, snapshot_bytes, trigger_timestamp_ns}. No DSP. EW stub remains default. | sdr_driver.py: SDRDriver(ABC) + SimSDRDriver + RealSDRDriver(stub, NotImplementedError). |

## **B. Output Pipelines**

| **Module** | **Current state** | **Missing** | **Deliverable** |
| --- | --- | --- | --- |
| B-1  PX4 MAVLink OFFBOARD | Interface designed (V6 §5.1). Previous attempt failed Dec. | Heartbeat thread (2Hz, independent). Pre-stream before OFFBOARD request. POSITION_TARGET_LOCAL_NED with coordinate_frame=MAV_FRAME_LOCAL_NED. PX4 mode monitor. Arming sequence with explicit state checks. | mavlink_output.py: MAVLinkBridge. connect(), arm_and_offboard(), send_setpoint(x,y,z,yaw), heartbeat_thread(), monitor_mode(). Validated: OFFBOARD held 60s in SITL. Setpoint confirmed in Gazebo. |
| B-2  State vector publisher | State in Python dicts. JSON KPI log post-run. | Real-time publisher at ESKF update rate. Non-blocking. Two modes: in_process callback (default) and UDP socket for external consumers. | state_publisher.py: StatePublisher. Modes: in_process, udp_socket. Published at ≥190Hz. Overhead <1ms measured. |
| B-3  Live event logger | JSON KPI log and HTML debrief (S7) post-run. | Non-blocking live logger. Producer-consumer queue. Background writer thread. Schema 08.1 compatible. File rotation. Drop rate <0.5% at 200Hz. | live_logger.py: LiveLogger. log_event(), flush(), close(). Drop counter exposed. Validated: 200Hz stream 60s, <0.5% drop. |

## **C. Integration Layer**

| **Module** | **Current state** | **Missing** | **Deliverable** |
| --- | --- | --- | --- |
| C-1  ROS2 bridge (if chosen) | No ROS2 in codebase. | Thin bridge only. Publishes /micromind/state, /micromind/vio_mode, /micromind/setpoint. Subscribes /mavros/state. Python core never becomes ROS2 callback-driven. | ros2_bridge.py (if ROS2): publishes standard message types. No core module imports. |
| C-2  TimeReference | SimClock (fixed-step). S-NEP-04 §1.5 timestamp contracts documented. | Unified time reference. now_ns(). sync_to_px4(px4_boot_us). apply_imu_offset(offset_ns). Monotonicity guard raising IFM-01 on violation. PX4 boot time sync on connection. | time_reference.py: TimeReference. Validated: 1000 mixed-source timestamps, zero violations. IFM-01 on injected bad timestamp. |
| C-3  Frame assertions | frame_utils.py: rotate_pos_enu_to_ned, rotate_cov_enu_to_ned. Validated S-NEP-04. | assert_ned(vector, label) raising FrameViolationError. Called at ESKF VIO input gate. Called at MAVLink setpoint output gate (assert NED before send). | frame_assertions.py. Validated: ENU input raises FrameViolationError. NED input passes. MAVLink setpoint validated before send. |
| C-4  Config-driven source switching | No config file. Hard-coded sim paths. | pipeline.yaml: all source keys. DriverFactory selects at startup. No hard-coded paths in runner. | driver_config.py: DriverConfig + DriverFactory. config/pipeline.yaml. --config flag. Validated: source swap without code change. |

## **D. Gazebo Simulation Stack**

| **Module** | **Current state** | **Missing** | **Deliverable** |
| --- | --- | --- | --- |
| D-1  PX4 SITL + Gazebo | PX4 SITL referenced in programme documents. Not installed or configured. | PX4 SITL with Gazebo Garden. iris quadrotor model. OFFBOARD mode available. Local position estimate available without GPS (EKF2 with VIO fusion stub). | Validated: px4_sitl launch, vehicle arms in Gazebo, OFFBOARD mode accepted from external controller. |
| D-2  MicroMind → SITL setpoint loop | No connection. | MAVLink bridge (B-1) feeding setpoints to SITL. Trajectory: takeoff to 5m, hold 10s, fly 50m North, return. Setpoint rate ≥20Hz. Heartbeat active. | Validated: Gazebo vehicle executes 50m trajectory. Position confirms waypoints reached within 2m. |
| D-3  VIO outage demo scenario | VIO outage exercised in BCMP-1 two-theatre. No Gazebo visual. | Inject VIO outage mid-flight (suspend OfflineVIODriver updates for 10s). Show: mode→OUTAGE, drift_envelope grows, vehicle continues on INS, VIO resumes, mode→RESUMPTION→NOMINAL. | Demo script: inject_outage.py. Outputs: mode log, drift_envelope chart, Gazebo screen recording. |
| D-4  Post-flight report generation | HTML debrief (S7) runs post-run in batch. | Post-flight HTML report generated from live logger output after Gazebo run. Shows: KPI summary, mode transitions, latency histogram, VIO outage events. | run_demo.sh: starts MicroMind + SITL + Gazebo, runs trajectory + outage injection, generates report on exit. |

## **E. System Boundary Definition**

The MicroMind system boundary ends at the MAVLink interface.  Inside the boundary (MicroMind responsibility): All sensor drivers (A-1..A-6), ESKF, TRN, VIO fusion, BIM, VIO navigation mode, enforcement layer, setpoint generator, MAVLink bridge, all logging and audit artefacts.  Outside the boundary: PX4 flight controller and all internal PX4 modules, autopilot inner-loop control, vehicle hardware, sensor hardware.  Consequence: MicroMind can be integrated onto any PX4-compatible platform without modifying the flight controller, the sensor hardware, or the estimator internals.

**Single-line architecture flow:**

Sensors → Drivers → ESKF → Enforcement → Setpoint Generator → MAVLink → PX4 → Vehicle  Each arrow is a clean module boundary. All arrows before MAVLink are internal to MicroMind. The MAVLink → PX4 arrow is the integration interface. MicroMind never crosses the MAVLink boundary in the PX4-inward direction.

## **F. Integration Envelope**

This section defines the complete input/output contract for integrating MicroMind as an autonomy payload.

**Inputs expected by MicroMind**

| **Input** | **Interface** | **Required?** | **Notes** |
| --- | --- | --- | --- |
| IMU data | Abstract IMUDriver — sim: STIM300 noise model; real: SPI/UART stream at 200Hz | Required | No real-time OS requirement. Python threading sufficient for SIL/SITL. |
| VIO pose estimates | Abstract VIODriver — sim: EuRoC .npy files; real: ROS2 /odomimu topic or socket at 124Hz | Required | Must be ENU frame. Driver applies rotate_pos_enu_to_ned before ESKF. LiveVIODriver stub prepared. |
| GNSS fix | Abstract GNSSDriver — sim: injector; real: NMEA/UBX serial parser | Optional | Used only for BIM trust scoring. Never used as primary navigation source in GNSS-denied mode. |
| Terrain altitude | Abstract RADALTDriver — sim: DEMProvider; real: RADALT serial | Optional | Required for TRN corrections. If absent: TRN corrections suppressed. NAV-01 still met via VIO on flat terrain. |
| EO/IR video | Abstract EOIRDriver — sim: synthetic scene; real: thermal camera GigE/USB3 | Optional (placeholder) | Interface contract defined. Real implementation: Phase 3+ post-TASL. |
| SDR / EW spectrum | Abstract SDRDriver — sim: random draw; real: USB3/PCIe SDR | Optional (placeholder) | Interface contract defined. Real implementation: Phase 2+ post-TASL. |

**Outputs provided by MicroMind**

| **Output** | **Interface** | **Rate** | **Notes** |
| --- | --- | --- | --- |
| MAVLink setpoints | POSITION_TARGET_LOCAL_NED via MAVLink OFFBOARD to PX4 | 20Hz ±2Hz constant | NED frame. coordinate_frame=MAV_FRAME_LOCAL_NED always set. |
| Unified state vector | StatePublisher: in_process callback or UDP JSON | ~190Hz | Includes: p (NED), v, q, ba, bg, vio_mode, drift_envelope_m, bim_trust, innovation_spike_alert. |
| Mission event log | LiveLogger: structured JSON events | Real-time streaming | Schema 08.1 compatible. Contains all enforcement events. |
| Post-flight HTML report | run_demo.sh output | Generated on mission end | KPI summary, mode transitions, latency histogram, outage events. Self-contained file. |

**What does NOT need to change for integration**

| **Component** | **Status** | **Evidence** |
| --- | --- | --- |
| PX4 flight controller firmware | No modification required | MicroMind uses standard MAVLink OFFBOARD. PX4 stock firmware supported. |
| Sensor hardware (IMU, camera, GNSS) | No modification required | Abstract driver layer means MicroMind accepts any sensor that implements the driver interface. |
| ESKF estimator | No modification required | Frozen per S-NEP-07. All integration work is in the driver layer. |
| Enforcement layer (E-1..E-5) | No modification required | S-NEP-10. Active and validated. Enforcement gates are structural. |
| BCMP-1 acceptance gates | No modification required | 332/332 tests remain the SIL regression baseline throughout integration work. |

**Summary: MicroMind integration requires only three things from an OEM: (1) sensor data routed to the abstract driver interfaces, (2) a MAVLink cable from the MicroMind compute to the autopilot, and (3) OFFBOARD mode enabled on the autopilot. Nothing else changes.**

*Estimator boundary clarification: PX4 EKF2 remains the active estimator for flight control. MicroMind provides external navigation state and setpoints but does not replace or modify PX4 EKF2.*

# **Part 4 — PX4 OFFBOARD Integration Playbook**

**This section is the direct response to the December failure. Every rule here is motivated by a specific, documented PX4 OFFBOARD failure mode. Follow in order. Do not skip steps.**

## **4.1  Root Cause Analysis: December Failure**

| **Probable cause** | **Evidence indicator** | **Likelihood** |
| --- | --- | --- |
| Heartbeat not running or blocked | PX4 mode changed back from OFFBOARD with no external command. No heartbeat visible in PX4 log. | **High — most common** |
| Setpoints started after OFFBOARD request | Mode accepted then immediately rejected. Vehicle never moved. | **High** |
| Setpoint rate too low (event-driven, not time-driven) | Vehicle briefly entered OFFBOARD then exited. Appears as heartbeat issue. | **Medium** |
| Wrong coordinate frame | Vehicle moved in unexpected direction or altitude inverted. coordinate_frame field not explicitly set. | **Medium** |
| PX4 state not verified before arming | ARM command sent without checking PRE_FLIGHT checks complete. Vehicle did not arm. | **Medium** |

## **4.1a  PX4 Configuration Baseline**

Before writing any MicroMind MAVLink code, the following PX4 SITL configuration must be verified. These are not MicroMind settings — they are PX4 settings that must be correct for OFFBOARD to work at all.

| **Parameter** | **Required value** | **Where to set** | **Why it matters** |
| --- | --- | --- | --- |
| EKF2_AID_MASK | Bit 3 set (vision position fusion enabled) for SITL without GPS. OR bit 0 set (GPS) if GPS is active in SITL. | QGroundControl → Parameters → EKF2_AID_MASK OR px4_sitl launch file | If no position source is fused, EKF2 local position is invalid. PX4 accepts OFFBOARD mode but ignores setpoints. This is ADD-05 failure mode. |
| COM_RCL_EXCEPT | Bit 2 set (OFFBOARD exception to RC loss failsafe) | QGroundControl → Parameters → COM_RCL_EXCEPT | Without this, RC loss in SITL (no RC connected) triggers failsafe that overrides OFFBOARD mode. |
| NAV_RCL_ACT | 0 (disabled) for SITL | QGroundControl → Parameters → NAV_RCL_ACT | Disables RC loss failsafe action. Required for external OFFBOARD control without RC. |
| COM_OBL_ACT | 1 (return) or 0 (hold) — NOT 2 (land) for demo | QGroundControl → Parameters → COM_OBL_ACT | Defines behaviour on OFFBOARD link loss. Land is too aggressive for a demo with intentional MAVLink gaps. |
| CBRK_USB_CHK | 197848 (disable USB check) | QGroundControl → Parameters → CBRK_USB_CHK | Required for SITL operation via UDP. Without this, pre-arm check fails. |
| MAV_SYS_ID | Match system_id in pymavlink connection | px4_sitl launch file — default 1 | System ID mismatch causes pymavlink to silently ignore PX4 heartbeats. |

**Checkpoint: Before running S-PX4-01, verify each parameter above in QGroundControl connected to SITL. Screenshot and log the parameter values. A misconfigured PX4 SITL parameter is not a MicroMind bug — but it produces identical symptoms.**

*Additional validation: verify that LOCAL_POSITION_NED is actively updating (position changes over time **>**0.1 m). A static or NaN position indicates EKF2 is not aligned and OFFBOARD control will not be effective.*

## **4.2  Minimum Reliable Architecture**

**MicroMind → MAVLink bridge → PX4 SITL → Gazebo**

| **Layer** | **Component** | **Requirement** | **Failure if violated** |
| --- | --- | --- | --- |
| Thread 1 | Heartbeat thread | Send HEARTBEAT at exactly 2Hz. Never blocked by compute. Separate daemon thread. Does not stop if navigation loop is slow. | **PX4 exits OFFBOARD silently** |
| Thread 2 | Setpoint stream | Send POSITION_TARGET_LOCAL_NED at ≥20Hz. Interpolate between waypoints. Rate is CONSTANT, not event-driven. Independent of decision loop rate. | **PX4 exits OFFBOARD on rate drop** |
| Thread 3 | State monitor | Receive HEARTBEAT from PX4. Monitor mode, arm state, system status. Raise alert on mode change. Only way to detect silent OFFBOARD exit. | **Silent failure undetectable** |
| Startup sequence | Arming + mode | (1) Connect. (2) Wait LOCAL_POSITION_NED valid. (3) Start setpoint stream at 20Hz. (4) Wait 2s (pre-stream minimum). (5) Send ARM. (6) Send OFFBOARD mode. (7) Confirm mode in monitor before sending mission setpoints. | **Any skip = mode rejection** |
| Frame contract | NED enforcement | All setpoints in NED. coordinate_frame=MAV_FRAME_LOCAL_NED (value=1) explicitly set. Asserted by C-3 frame_assertions before every send call. Z is positive DOWN. | **Vehicle altitude inverted or heading wrong** |
| Timing | time_boot_ms | Sync to PX4 boot time on connection. Use SYSTEM_TIME response to compute offset. All MAVLink messages carry PX4-relative time_boot_ms. | **Messages rate-limited or ignored** |

## **4.2a  Control Loop Definition**

The following defines the exact thread architecture of the MicroMind → PX4 control loop. This definition must be implemented before any PX4 validation step begins.

| **Thread** | **Name** | **Rate** | **Blocked by** | **Blocks anything?** | **Failure if absent** |
| --- | --- | --- | --- | --- | --- |
| T-HB | Heartbeat daemon | 2Hz fixed | Nothing (daemon thread). Uses threading.Event.wait(0.5) not time.sleep(). | No | PX4 exits OFFBOARD within 1s of last heartbeat. |
| T-SP | Setpoint loop | 20Hz fixed | Nothing. Uses threading.Event or time.perf_counter_ns for rate control. Never waits on navigation decision. | No | PX4 exits OFFBOARD on rate drop below 15Hz. |
| T-MON | State monitor | 10Hz poll or event-driven receive | Nothing (async receive). Alerts T-NAV on OFFBOARD exit. | No (raises event only) | Silent OFFBOARD exit undetectable. |
| T-NAV | Navigation loop | 200Hz IMU rate | IMU driver read (should be <1ms). ESKF propagation (<10ms P95). Enforcement (<5ms). Never blocked by T-HB, T-SP, or T-MON. | No: publishes setpoints to T-SP queue. T-SP reads independently. | Navigation state not updated. Setpoints become stale but continue transmitting. |
| T-LOG | Logger background writer | Async (queue consumer) | File I/O. Deliberately isolated from all other threads. | No (queue write from T-NAV is O(1) non-blocking) | Events lost. Drop rate metric available. |

**Critical rule: T-HB and T-SP must never share a thread with T-NAV. If T-NAV stalls (e.g. slow ESKF step), T-HB and T-SP continue independently. PX4 must not lose OFFBOARD because the navigation loop was slow.**

*Queue constraint: the interface between T-NAV and T-SP must be implemented as a non-blocking queue with bounded size. Under no condition should setpoint generation be blocked by navigation loop execution or queue backpressure.*

## **4.3  Incremental Validation Steps (must follow in order)**

**Do not proceed to the next step until the current step passes. No exceptions.**

| **Step** | **Task** | **Pass condition** | **Stop if** |
| --- | --- | --- | --- |
| S-PX4-01 | Install PX4 SITL + Gazebo. Verify 4.1a parameter baseline. Launch iris model. Confirm vehicle visible in Gazebo. | Vehicle appears in Gazebo. px4 log shows EKF initialised. ESTIMATOR_STATUS local position bit set. | Vehicle does not appear after 60s. EKF not initialised. |
| S-PX4-02 | Connect pymavlink to SITL. Send HEARTBEAT only (no setpoints, no arm). Observe for 30s. | PX4 log shows external heartbeat received. No mode changes. | PX4 does not acknowledge heartbeat. |
| S-PX4-03 | Start setpoint stream at 20Hz (vehicle still disarmed, grounded). Observe for 10s. | No crash. PX4 receives setpoints. Rate confirmed ≥20Hz in send loop counter. | Rate drops below 20Hz consistently. |
| S-PX4-04 | Send ARM command. Confirm arm state in HEARTBEAT response. Do not request OFFBOARD yet. | PX4 HEARTBEAT reports MAV_STATE_ACTIVE. Gazebo vehicle shows armed state. | ARM rejected. Check pre-flight status and COM_RCL_EXCEPT parameter. |
| S-PX4-05 | Request OFFBOARD mode. Check MAV_RESULT in command acknowledgement. Observe mode in monitor thread. | Mode monitor reports OFFBOARD. No mode reversion within 10s. | Mode reverts immediately. Diagnose pre-stream gap. |
| S-PX4-06 | Send takeoff setpoint (z = -5m in NED = 5m altitude). Observe Gazebo. | Vehicle climbs to 5m AGL in Gazebo. No crash. OFFBOARD maintained. | Vehicle descends (z sign wrong — frame error). Vehicle does not move (rate issue). |
| S-PX4-07 | Send 50m North trajectory (y=0, x=50m NED). Observe Gazebo. | Vehicle moves North in Gazebo. Reaches target within 2m. OFFBOARD maintained throughout. | Vehicle moves in wrong direction (frame). Waypoints incorrect. |
| S-PX4-08 | Kill setpoint stream for 3s. Observe mode monitor. | PX4 exits OFFBOARD (expected — stream dropped). Mode monitor raises alert. Confirms monitor is working. | Mode does not exit (stream not monitored). Monitor alert not fired. |
| S-PX4-09 | Run full 60s trajectory with MicroMind navigation loop feeding setpoints. VIO active. Measure latency. | OFFBOARD maintained 60s. Latency P95 <50ms. No mode exits. Position within 2m of intended trajectory. | OFFBOARD exits during run. Diagnose: heartbeat gap or rate drop. |
| S-PX4-10 | Inject VIO outage (10s). Observe: mode→OUTAGE, drift_envelope grows, recovery on VIO resume. | Mode transitions correct. Drift envelope grows during outage. Recovery to NOMINAL after resume. Setpoints continue during outage from INS. | Mode fault (unknown mode). No recovery. Vehicle crashes. |

## **4.4  Failure Detection Signals**

| **Signal** | **Source** | **Action required** |
| --- | --- | --- |
| PX4 HEARTBEAT: base_mode bit 7 (OFFBOARD) drops | State monitor thread | Log OFFBOARD_EXIT event immediately with timestamp. Raise alert to navigation loop. Do not continue sending mission setpoints. |
| Setpoint send rate drops below 18Hz | Setpoint loop counter | Log SETPOINT_RATE_WARN. Investigate loop blocking. Check logger queue depth. |
| Command acknowledgement: MAV_RESULT_DENIED or MAV_RESULT_FAILED | MAVLink receive | Log CMD_REJECTED with command ID and result code. Stop and diagnose before retrying. |
| Gazebo vehicle moves in wrong direction on first NED setpoint | Visual observation | Frame error in setpoint. Check coordinate_frame field value. Check Z-sign convention. |
| IFM-01: timestamp gap >200ms at ESKF input | TimeReference monotonicity guard | Log IFM-01. ESKF continues on IMU alone. Investigate sensor clock source. |
| FrameViolationError raised at ESKF input | C-3 frame assertions | Log FRAME_VIOLATION with vector label. Block measurement. Investigate driver frame handling. |
| OFFBOARD active but vehicle not responding to setpoints (stationary in Gazebo despite correct setpoints sent) | PX4 EKF2 status. Monitor LOCAL_POSITION_NED and ESTIMATOR_STATUS flags. | Verify LOCAL_POSITION_NED.x/y/z are non-NaN and changing. Check ESTIMATOR_STATUS.flags: bit 0 (attitude) and bit 5 (local position) must both be set. If EKF2 not aligned (no GPS or VIO fusion active in SITL), PX4 will accept OFFBOARD mode but refuse to track setpoints. Fix: ensure VIO fusion plugin active in SITL or GNSS available to EKF2 before engaging OFFBOARD. |

# **Part 5 — Latency ****&**** Compute Validation**

## **5.1  Measurement Plan**

| **Stage** | **What to measure** | **Instrument** | **Output** |
| --- | --- | --- | --- |
| Sensor → ESKF | Time from driver.read() call to ESKF propagation complete. Includes driver overhead, timestamp check, frame assertion, update_vio() or update_gnss() call. | time.perf_counter_ns() before and after each update call. Stored in live logger. | Per-step latency in nanoseconds. P50, P95, P99 over 1000 steps. |
| ESKF → decision | Time from ESKF update complete to NanoCorteX FSM and enforcement decision. | time.perf_counter_ns() before and after decision block in runner. | Per-step latency. P95 target: <5ms. |
| Decision → MAVLink send | Time from setpoint computed to POSITION_TARGET_LOCAL_NED sent on socket. | time.perf_counter_ns() in setpoint thread before send call. | Per-send latency. P95 target: <2ms. |
| End-to-end | Total: sensor timestamp to MAVLink send. | Structured log event with sensor_ts and send_ts. Post-run analysis. | E2E P95 target: <50ms (engineering judgement; operationally, <100ms acceptable for SITL). |
| Setpoint rate | Actual Hz of POSITION_TARGET_LOCAL_NED send loop. | Counter incremented per send. Rate computed over 5s windows. | Should be stable at 20Hz ±2Hz. Any drop below 18Hz is flagged. |
| CPU usage | % CPU during 60s navigation run. | psutil.cpu_percent(interval=1) in monitoring thread. | Avg and peak CPU%. Target: <60% avg on development hardware. |
| Memory footprint | RSS memory during run. | psutil.Process().memory_info().rss at 1Hz. | Avg and peak MB. Target: <500MB (engineering estimate for Jetson-class hardware). |
| Loop jitter | Variation in step-to-step timing of IMU propagation loop. | Histogram of step durations. Std dev target: <1ms. | Jitter histogram. High jitter indicates GIL contention or blocking calls in loop. |

## **5.2  Acceptable Thresholds (Engineering Judgement)**

| **Metric** | **Target** | **Rationale** | **Action if exceeded** |
| --- | --- | --- | --- |
| ESKF update latency P95 | <10ms | 200Hz loop requires each step <5ms. 10ms P95 gives headroom for occasional spikes. | Profile update_vio() and update_gnss() calls. Check covariance computation cost. |
| End-to-end latency P95 | <50ms | SITL does not require hard real-time. <50ms provides adequate control responsiveness for Gazebo demo. | Check for blocking calls in decision or logging path. |
| Setpoint rate | 20 ±2 Hz | PX4 OFFBOARD minimum. 20Hz is minimum; 50Hz is better. | Investigate setpoint thread blocking. Separate from navigation loop if needed. |
| CPU average | <60% on dev hardware | Leaves margin for OS and monitoring. Jetson Orin has more cores; estimate will be lower. | Profile which module consumes most CPU. Candidate: NCC correlation in TRN. |
| Memory RSS | <500MB | Conservative estimate. EuRoC data + DEM tiles + state history. | Profile object creation in hot loops. Check for list growth in logging. |
| Heartbeat thread jitter | ±100ms on 2Hz send | Heartbeat must never miss a 500ms window. ±100ms is safe margin. | Ensure heartbeat thread has no blocking dependencies. Use threading.Event.wait() not sleep(). |

## **5.3  Deterministic Runtime System Properties**

MicroMind maintains three runtime properties simultaneously. These are system-level invariants, not aspirational targets. Violation of any one constitutes an integration defect.

| **Property** | **Statement** | **Measurement** | **Failure mode** |
| --- | --- | --- | --- |
| Bounded latency | End-to-end P95 latency from sensor input to MAVLink setpoint ≤50ms. | Measured continuously during all Gazebo runs. Reported in post-flight HTML report. | Blocking call in navigation path. Logger contention. Check jitter histogram. |
| Stable control rate | Setpoint transmission rate is 20Hz ±2Hz at all times. TIME-driven, not event-driven. Does not drop when navigation decision is unchanged or when OUTAGE/RESUMPTION mode is active. | Setpoint loop counter over 5s windows. Any drop below 18Hz flagged as SETPOINT_RATE_WARN. | T-SP and T-NAV sharing a thread. T-NAV slow step blocking T-SP. |
| Deterministic loop execution | Identical inputs produce identical outputs. Mode transitions occur in exactly 1 processing cycle (confirmed S-NEP-09 G-09-02). Loop behaviour does not depend on wall-clock conditions or OS scheduling. | 332 SIL gates serve as regression baseline. Any non-determinism manifests as test failure. | Non-monotonic timestamps. OS preemption inside critical section. |

**No blocking operation is permitted inside the IMU propagation loop (T-NAV) or the setpoint thread (T-SP).  Blocking operations that must NEVER appear in the control loop: •  File I/O (disk write, log flush) — use LiveLogger queue instead •  Network calls with blocking timeout — use non-blocking UDP send •  Mutex acquisition that could block on logger contention •  time.sleep() calls of any duration •  Any call that can block indefinitely  Enforcement: LiveLogger (B-3) uses producer-consumer queue to remove disk I/O from control path. StatePublisher (B-2) uses non-blocking UDP send. Heartbeat thread (T-HB) is an independent daemon. Any future addition to the control loop path must be reviewed for blocking potential before integration.**

*Determinism verification: repeat-run consistency must be validated by comparing mode transition timestamps and first_post_outage_innov values across identical runs. This behaviour is established in S-NEP-09 and must remain invariant under integration.*

# **Part 6 — Execution Plan**

Six phases. No phase starts until the previous phase exit gate is met. Phase 1.5 (PX4 minimal link) is deliberately pulled forward to fail-fast on the highest risk.

## **Phase 0 — Architecture Lock (Day 1, no code)**

| **Decision** | **Options** | **Recommendation** | **Gate** |
| --- | --- | --- | --- |
| Middleware | (a) Thin ROS2 bridge only  (b) Direct Python + UDP socket  (c) Full ROS2 refactor | (a) or (b). Never (c). Recommendation: (b) for Phase 1–2 (fastest path to PX4 demo); add thin ROS2 bridge in Phase 4 if needed for Gazebo MAVROS. | Written decision document. Reviewed by TD. |
| Time reference | (a) MicroMind owns master clock, sync PX4 on connect  (b) PX4 owns master | (a). MicroMind authoritative. PX4 sync via SYSTEM_TIME on connection. | Interface documented in time_reference.py docstring before code. |
| Driver depth | (a) Abstract base class per driver  (b) Config flags inline | (a). Abstract base class. Matches S-NEP-04 modular design. | Module boundary diagram: new files only, no existing files modified. |
| PX4 OFFBOARD failure modes | Review Part 4 of this document | All five failure modes and PX4 configuration baseline (Section 4.1a) reviewed and understood before B-1 implementation. | Signed acknowledgement before Phase 1.5. |

## **Phase 1 — Drivers + Config (Days 2–5)**

| **Module** | **Task** | **Exit gate** |
| --- | --- | --- |
| D-1 DriverFactory + DriverConfig | Abstract base classes. Factory. Config loading. All source keys. | 332 tests pass. Factory returns correct class per config value. |
| D-2 pipeline.yaml | YAML config file. All source keys. Default: all sim. | Swap one source to stub. Confirm module substitution without code change. |
| A-1 SimIMUDriver | Wrap existing noise model. Implement IMUDriver ABC. | Existing IMU tests pass via new driver path. |
| A-3 SimGNSSDriver | Wrap existing injector. NMEA stub. BIM hook unchanged. | S2 BIM tests pass via new driver. |
| A-4 SimRADALTDriver | Wrap DEMProvider. Implement RADALTDriver ABC. | TRN tests pass via new driver. |
| A-5 SyntheticEOIRDriver | Wrap generate_synthetic_scene. EOIRDriver ABC. | DMRL tests pass via new driver. |
| A-6 SimSDRDriver | Wrap _EWEngineStub. SDRDriver ABC. | EW tests pass via new driver. |
| C-3 Frame assertions | assert_ned() + FrameViolationError. Called at ESKF VIO input gate. | ENU input raises FrameViolationError. NED input passes. One test. |
| Phase 1 gate | Run 332 tests. | 332/332 pass. No existing test modified. |

## **Phase 1.5 — PX4 Minimal Link (Days 6–8) — PULLED FORWARD**

**This phase is pulled forward specifically because PX4 OFFBOARD failed in December. We must fail-fast on this before investing in the full output pipeline. If OFFBOARD cannot be established reliably, the integration architecture needs to be reconsidered before proceeding.  IF PX4 OFFBOARD FAILS AT PHASE 1.5: STEP 1 — STOP. Do not attempt to patch forward. Do not modify the heartbeat thread or setpoint rate without completing the diagnosis. STEP 2 — Isolate using Section 4.4 failure detection signals in order: (a) heartbeat visible in PX4 logs? (b) setpoint rate ≥20Hz? (c) mode acknowledgement MAV_RESULT_ACCEPTED? (d) LOCAL_POSITION_NED valid before mode switch? (e) coordinate_frame field set to 1? STEP 3 — Fix one cause at a time. Regression test after each change: rerun S-PX4-01 through S-PX4-05 before attempting S-PX4-06. STEP 4 — Escalate to TD if failure is not resolved in 2 days. Do not attempt further patching.**

| **Step** | **Task** | **Exit gate** |
| --- | --- | --- |
| Install PX4 SITL + Gazebo | Install px4_sitl_default, Gazebo Garden, iris model. Verify parameter baseline (Section 4.1a). Confirm vehicle spawns. | Vehicle visible in Gazebo. EKF initialised. ESTIMATOR_STATUS local position bit set. |
| B-1 MAVLink bridge (minimal) | Heartbeat thread. connect(). arm_and_offboard(). send_setpoint(). monitor_mode(). Follow S-PX4-01 through S-PX4-09 in strict order. | OFFBOARD held for 60s in SITL. Setpoint confirmed in Gazebo. Vehicle executes 50m trajectory. |
| C-2 TimeReference (minimal) | now_ns(). sync_to_px4(). Monotonicity guard. IFM-01 log. | 1000 timestamps: zero violations. IFM-01 fired on injected bad timestamp. |
| Phase 1.5 gate | OFFBOARD demonstration. | Vehicle executes waypoint sequence in Gazebo. OFFBOARD maintained throughout. Mode monitor confirms. Screen recorded. |

## **Phase 2 — Full Output Pipeline + Timing (Days 9–12)**

| **Module** | **Task** | **Exit gate** |
| --- | --- | --- |
| B-3 LiveLogger | Non-blocking queue. Background writer. Schema 08.1. Drop counter. | 200Hz events 60s. Drop rate <0.5%. Schema validated. |
| B-2 StatePublisher | in_process + UDP socket modes. Non-blocking. | State at ≥190Hz. Overhead <1ms. |
| Latency instrumentation | Timestamp at each pipeline stage per Section 5.1. P95 computation. CSV export. | Latency log produced for 60s run. P95 end-to-end <50ms. |
| Phase 2 gate | Full pipeline run with timing. | Latency log produced. Setpoint rate confirmed. No OFFBOARD exits during 60s run. |

## **Phase 3 — Live Input Stubs + VIO Adapter (Days 13–15)**

| **Module** | **Task** | **Exit gate** |
| --- | --- | --- |
| A-2 OfflineVIODriver + LiveVIODriver stub | Wrap existing .npy loader. Apply ENU→NED. Covariance guard. Monotonicity gate per IFM-01. LiveVIODriver: stub with NotImplementedError. | All S-NEP-04..S-NEP-09 tests pass through OfflineVIODriver. ENU input rejected. Covariance guard active. |
| Real driver stubs | RealIMUDriver, RealGNSSDriver, RealRADALTDriver: NotImplementedError + interface doc. Config=real produces clean error. | Clean error message pointing to unconnected hardware. No ESKF change required. |
| C-1 ROS2 bridge (if chosen) | Thin bridge. Publish state/setpoint topics. No core imports. | ROS2 topics receive data. Python core runs standalone without ROS2. |
| Phase 3 gate | Full driver substitution test. | All driver slots: sim path and real stub. source=sim runs full SIL. source=real gives clean error. Zero core module changes. |

## **Phase 4 — Gazebo Demo Environment (Days 16–19)**

| **Module** | **Task** | **Exit gate** |
| --- | --- | --- |
| D-3 VIO outage demo | inject_outage.py. Suspend OfflineVIODriver for 10s. Observe mode transitions in Gazebo + log. | Mode→OUTAGE visible in log and terminal. Drift envelope grows. Vehicle continues on INS. Recovery to NOMINAL. Screen recorded. |
| D-4 Post-flight report | Live logger output → HTML report. KPI summary. Mode transitions. Latency histogram. Outage events. | HTML report generated from Gazebo run. All expected events present. |
| run_demo.sh | Single script: start MicroMind + SITL + Gazebo, run trajectory + outage injection, generate report on exit. | One command. Reproducible. Clean exit. Report generated. |
| Phase 4 gate / Demo gate | OEM demonstration scenario. | run_demo.sh executes end-to-end. Vehicle trajectory visible in Gazebo. Outage injected. Recovery visible. HTML report generated. Reproducible 3x in a row. |

# **Part 7 — Time ****&**** Effort Estimate**

| **Module** | **Effort** | **Days** | **Parallelisable?** | **Notes** |
| --- | --- | --- | --- | --- |
| Phase 0: Arch decisions + PX4 failure review + config baseline | **Low** | 1 | No | 4 decisions + 4.1a parameter baseline documented before code |
| D-1 DriverFactory + DriverConfig | **Med** | 1.5 | No — gates Phase 1 | Abstract bases; factory; config parsing |
| D-2 pipeline.yaml | **Low** | 0.5 | With D-1 | YAML load + validation |
| A-1 SimIMUDriver | **Low** | 0.5 | With A-3/A-4/A-5/A-6 | Wrap noise model |
| A-3 SimGNSSDriver | **Low** | 0.5 | With A-1 | NMEA stub + BIM hook |
| A-4 SimRADALTDriver | **Low** | 0.5 | With A-1 | Wrap DEMProvider |
| A-5 SyntheticEOIRDriver | **Low** | 0.5 | With A-1 | Wrap scene generator |
| A-6 SimSDRDriver | **Low** | 0.5 | With A-1 | Wrap EW stub |
| C-3 Frame assertions | **Low** | 0.5 | With D-1 | assert_ned + one test |
| Phase 1 gate | **Low** | 0.5 | No | 332 tests + driver wiring |
| PX4 SITL + Gazebo install + verify + parameter baseline | **Med** | 1 | No — gates Phase 1.5 | Environment setup + 4.1a verification |
| B-1 MAVLink bridge (full OFFBOARD) | [object Object][object Object] | 3 | No — critical path | Heartbeat + pre-stream + mode monitor + control loop threads + S-PX4-01..10 |
| C-2 TimeReference | **Med** | 1.5 | After Phase 0 | Monotonicity guard; PX4 sync; IFM-01 |
| Phase 1.5 gate (OFFBOARD 60s) | **Med** | 0.5 | No | OFFBOARD + Gazebo confirmation |
| B-3 LiveLogger | **Med** | 1.5 | After C-2 | Queue + background writer |
| B-2 StatePublisher | **Low** | 1 | After C-2, with B-3 | in_process + UDP |
| Latency instrumentation | **Med** | 1 | With B-2/B-3 | perf_counter_ns at each stage; CSV export |
| Phase 2 gate | **Low** | 0.5 | No | Latency log + setpoint rate confirmed |
| A-2 OfflineVIODriver + LiveVIODriver stub | **Med** | 2 | After Phase 2 | ENU→NED; covariance guard; NEP tests pass |
| Real driver stubs (IMU, GNSS, RADALT) | **Low** | 1 | With A-2 | NotImplementedError + interface doc |
| C-1 ROS2 bridge (if chosen) | **Med** | 2 | With A-2 | Thin bridge only; optional |
| Phase 3 gate | **Low** | 0.5 | No | All driver slots wired |
| D-3 VIO outage demo script | **Med** | 1 | After Phase 3 | inject_outage.py + log validation |
| D-4 Post-flight HTML report | **Med** | 1 | With D-3 | Live logger → HTML |
| run_demo.sh + reproducibility test | **Low** | 1 | No | 3x reproducible |
| Phase 4 gate | **Low** | 0.5 | No | End-to-end demo confirmed |

| **Summary** | **Value** |
| --- | --- |
| Total effort (single developer, nominal) | ~25 days |
| Risk-adjusted total (+20% for PX4 OFFBOARD uncertainty) | ~30 days |
| Critical path | Phase 0 → D-1/D-2 → B-1 (MAVLink OFFBOARD) → C-2 → Phase 1.5 gate — ~8 days sequential |
| Parallelisable work | A-1..A-6 drivers (Phase 1, ~2 days); B-2/B-3 with each other (Phase 2); real stubs with A-2 (Phase 3) |
| Compressible to (if drivers batched) | ~20 days nominal, ~24 risk-adjusted |
| If PX4 OFFBOARD fails again at Phase 1.5 | STOP. 2-day diagnosis sprint per Section 4.4 + ADD-08 diagnostic protocol. Do not proceed to Phase 2 until OFFBOARD confirmed reliable. |

# **Part 8 — Pre-HIL Readiness Criteria**

All eleven criteria must pass before declaring the system Pre-HIL ready. No partial credit.

| **#** | **Criterion** | **Verification method** | **Pass condition** |
| --- | --- | --- | --- |
| RC-1 | PX4 SITL accepts and executes setpoints | S-PX4-09: run full trajectory | Vehicle executes 50m waypoint sequence in Gazebo. Position within 2m. OFFBOARD held throughout. |
| RC-2 | OFFBOARD mode stable for 60s under live navigation loop | S-PX4-09: 60s run | Zero OFFBOARD exits. Mode monitor confirms. Heartbeat thread log visible. |
| RC-3 | Heartbeat thread independent of navigation loop | Kill navigation loop for 3s while heartbeat runs. Monitor PX4 mode. | PX4 stays in OFFBOARD during navigation gap (heartbeat maintained). Exits when heartbeat explicitly stopped. |
| RC-4 | Input interfaces swappable without core code change | Swap each driver in pipeline.yaml one at a time sim→stub | No change to ESKF, BIM, vio_mode, enforcement. Only driver file substituted. 332 tests pass. |
| RC-5 | 332 SIL gates pass unchanged | pytest after all integration modules added | 332/332. No existing test modified. No test result changed. |
| RC-6 | ENU→NED enforced at ESKF input and MAVLink output | C-3 frame assertion tests: ENU in, NED in, MAVLink setpoint | FrameViolationError on ENU at ESKF gate. NED passes. MAVLink setpoint passes NED assertion. |
| RC-7 | Timestamp monotonicity enforced | Inject non-monotonic timestamp sequence | IFM-01 logged. ESKF continues. Specific event ID and timestamp recorded. |
| RC-8 | Live logger non-blocking at 200Hz | 200Hz event stream for 60s alongside navigation loop | Drop rate <0.5%. ESKF step rate unchanged (≥190Hz measured). |
| RC-9 | Latency P95 <50ms end-to-end | Latency log from 60s Gazebo run | P95 latency across all steps <50ms. Setpoint rate stable at 20±2 Hz. |
| RC-10 | No architectural rewrites required for real hardware | Architecture review: add real IMU = implement abstract base only? | Zero changes to ESKF, BIM, vio_mode, enforcement required. Driver substitution only. |
| RC-11 | Control loop independence. MicroMind continues generating valid NED setpoints at 20Hz even when PX4 temporarily rejects commands or when navigation mode is OUTAGE or RESUMPTION. | (a) Inject OUTAGE mode (suspend VIO 10s). Confirm setpoints continue at 20Hz throughout. (b) Disconnect MAVLink 5s. Confirm setpoint generation continues internally. Reconnect. Confirm OFFBOARD re-engages cleanly. | Setpoint generation rate does not drop during OUTAGE or RESUMPTION modes. Setpoint stream resumes correctly after MAVLink reconnect. No crash or exception during disconnect window. |

# **Part 9 — OEM Demonstration Definition**

This is the exact demo script for an OEM/TASL reviewer. It is reproducible by running: ./run_demo.sh

| **Step** | **Action** | **Expected output** | **What it proves** |
| --- | --- | --- | --- |
| 1 | ./run_demo.sh — starts MicroMind, PX4 SITL, Gazebo in sequence with 5s gaps | Gazebo window opens. Vehicle spawns. Terminal shows 'MicroMind: NOMINAL mode'. PX4: EKF initialised. | System starts cleanly from one command. No manual steps. |
| 2 | MAVLink bridge connects. TimeReference syncs to PX4 boot time. | Terminal: 'MAVLink connected. PX4 time synced. Delta: Xms'. Heartbeat thread starts. | Time synchronisation active from first connection. |
| 3 | Pre-stream: 2s of hold setpoints at current position, 20Hz. | Setpoint rate confirmed in log: 20Hz. PX4 receiving setpoints. | Pre-stream requirement met before mode switch. |
| 4 | ARM command sent. OFFBOARD requested. | Terminal: 'Armed. OFFBOARD engaged.' Gazebo: vehicle lifts off to 5m. | Clean OFFBOARD engagement. No mode rejection. |
| 5 | Trajectory: hold 5m for 5s, fly 50m North, hold, return, land. | Gazebo: vehicle executes full trajectory. Mode monitor: OFFBOARD throughout. Latency log: P95 <50ms. | Navigation loop commands vehicle correctly. Frame handling correct. |
| 6 | VIO outage injected at T+30s for 10s. | Terminal: 'VIO_OUTAGE: mode→OUTAGE. drift_envelope=8.0m'. Gazebo: vehicle continues (INS only). No crash. | Degraded mode correct. E-1..E-5 enforcement visible. Vehicle does not fall out of the sky. |
| 7 | VIO resumes at T+40s. | Terminal: 'VIO_RESUMPTION. spike_alert=True. mode→NOMINAL'. Gazebo: vehicle back under full navigation. drift_envelope decreasing. | Recovery from outage is clean and auditable. |
| 8 | Vehicle completes trajectory and lands. run_demo.sh generates HTML report. | HTML report opens in browser: KPI summary, mode transitions chart, latency histogram, VIO outage events, enforcement log. | Audit trail complete. Reviewer can verify every event. |
| 9 | Reviewer asks: 'swap the IMU source' | Operator changes imu_source=real in pipeline.yaml. Restarts. Terminal: 'RealIMUDriver: not connected. Check SPI interface.' No crash. No code change. | Replaceability claim demonstrated live. |
| 10 | Reviewer asks: 'show me the BCMP-1 results' | Open MicroMind_BCMP1_TwoTheatre_TechnicalDossier.docx. Walk through Annex A evidence chain. All claims traceable to commits. | SIL validation results support the integration demo. |

**Critical: run_demo.sh must be tested 3 times in a row before any OEM meeting. All three runs must succeed with identical outputs. If any run fails, diagnose and fix before scheduling the meeting.**

## **9.11  Mandatory On-Screen Overlays**

During the OEM demonstration, the following information must be visible on screen simultaneously. The purpose is to ensure the OEM observer can immediately understand system behaviour without reading log files.

| **Overlay** | **Source** | **Purpose** |
| --- | --- | --- |
| vio_mode | VIONavigationMode.current_mode.name — live from navigation loop | Shows NOMINAL/OUTAGE/RESUMPTION state at all times. Observer sees mode change immediately on VIO outage injection. |
| drift_envelope_m | vio_nav.drift_envelope_m — live from navigation loop | Shows conservative drift estimate growing during outage, resetting on recovery. Demonstrates conservative-by-design position confidence model. |
| Setpoint trajectory vs actual trajectory | Setpoint NED (x,y,z) from setpoint thread vs LOCAL_POSITION_NED from PX4 state monitor | Shows that MicroMind-commanded trajectory matches vehicle trajectory. Visual proof of control authority. |
| Mode transition log | Live event log from LiveLogger: TERMINAL_ZONE_ENTERED, OUTAGE, RESUMPTION, NOMINAL events | Observer sees events as they occur. Connects Gazebo visual to system state. Provides audit evidence in real time. |
| Setpoint rate (Hz) | Setpoint loop counter — computed over 1s windows | Confirms 20±2Hz continuously. Shows rate does not drop on mode change. Directly addresses 'how fast is the control loop?' question. |
| PX4 flight mode | State monitor (HEARTBEAT base_mode field) — live from MAVLink receive thread | Shows OFFBOARD / HOLD / POSCTL / STABILIZE in real time. Confirms PX4 is accepting and executing MicroMind commands. Immediate visual indicator if OFFBOARD mode exits. |

Implementation: these overlays can be achieved with a Python curses terminal panel or a tee to a separate monitoring process. They do not require a GUI framework. The run_demo.sh script must launch the overlay display as part of its startup sequence.

# **Part 10 — Execution Discipline**

## **10.1  Pre-Code Validation Checklist**

| **#** | **Validation** | **Method** | **Must complete before** |
| --- | --- | --- | --- |
| V-1 | Phase 0 architecture decisions documented | Written decisions reviewed by TD | Any Phase 1 code |
| V-2 | PX4 OFFBOARD failure modes reviewed and signed off | Part 4 of this document reviewed; five modes acknowledged | Phase 1.5 MAVLink code |
| V-3 | Module boundary diagram: new files only, no existing file modified | Diagram produced and approved | Any Phase 1 code |
| V-4 | 332 tests pass on clean clone of main | python -m pytest tests/ — 332/332 | Any integration code |
| V-5 | PX4 SITL installs and vehicle spawns on dev machine | px4_sitl_default + Gazebo: vehicle visible | B-1 MAVLink code |
| V-6 | frame_utils.py ENU→NED functions pass in isolation | python -c 'from core.fusion.frame_utils import rotate_pos_enu_to_ned; ...' | A-2 VIO driver |
| V-7 | PX4 OFFBOARD works from MAVSDK or QGroundControl (baseline) | Independent sanity check: OFFBOARD from standard tool, not MicroMind | B-1 MAVLink code. Establishes SITL is not broken before MicroMind is blamed. |
| V-8 | PX4 parameter baseline (Section 4.1a) verified in QGroundControl | Screenshot each parameter value. Log to file. | B-1 MAVLink code. Prevents misdiagnosis of PX4 config issues as MicroMind bugs. |

## **10.2  What Must NOT Be Changed**

| **Module** | **Constraint** |
| --- | --- |
| core/ekf/error_state_ekf.py | No changes. Frozen constants. Any modification requires TD approval and S-NEP-09 re-run. |
| core/fusion/vio_mode.py | No changes. Mode Integrity Invariant. Changing transition logic invalidates S-NEP-09 characterisation. |
| core/fusion/frame_utils.py | Do not rewrite. New driver code calls rotate_pos_enu_to_ned and rotate_cov_enu_to_ned. Never re-implement the transform. |
| scenarios/bcmp1/bcmp1_runner.py (E-1..E-5) | Enforcement blocks not modified. Integration adds drivers that feed existing paths. |
| Frozen estimator constants | _ACC_BIAS_RW, _GYRO_BIAS_RW, _POS_DRIFT_PSD, _GNSS_R_NOMINAL. No changes outside TD process. |
| tests/ (332 gates) | No gate modified to pass. No test result changed. Run after every module addition. |

## **10.3  Mandatory Review Checkpoints**

| **Checkpoint** | **Trigger** | **Action** | **Pass condition** |
| --- | --- | --- | --- |
| CP-0: Architecture lock | Phase 0 complete | TD reviews 4 decisions + module diagram. Signs off on PX4 failure mode list and parameter baseline. | All 4 decisions documented. Module diagram approved. PX4 failures acknowledged. Parameter baseline logged. |
| CP-1: Phase 1 gate | Phase 1 complete | Run 332 tests. Review driver registry. Confirm no core module imports changed. | 332/332. Factory correct. Frame assertion works. |
| CP-1.5: PX4 OFFBOARD gate (CRITICAL) | Phase 1.5 complete | Observe OFFBOARD demo in Gazebo. Check heartbeat thread in logs. Verify S-PX4-01..10 completed in order. | Vehicle executes 50m trajectory. OFFBOARD maintained. No reversion. Screen recorded. |
| CP-2: Phase 2 gate | Phase 2 complete | Latency log from 60s run. Setpoint rate confirmed. Logging validated. | P95 latency <50ms. Setpoint rate 20±2Hz. Log complete. |
| CP-3: Pre-HIL readiness | RC-1..RC-11 all pass | Run full readiness checklist (11 criteria). Generate readiness report. | All 11 criteria pass. Document issued. |

## **10.4  Coding Rules for This Phase**

- Every new module is a new file. No code added to existing core modules (ESKF, vio_mode, frame_utils, bcmp1_runner enforcement blocks, or tests/).

- Every driver inherits from an abstract base class. The ABC defines the interface contract as method signatures and docstrings. No duck typing.

- No ROS2 imports in any file under core/. ROS2 belongs in the bridge layer only.

- Every module has at least one test. Test validates minimum acceptable version, not ideal version.

- Run 332 tests after every module addition. Do not batch modules. If a gate breaks, stop and fix before writing next module.

- The MAVLink heartbeat thread (T-HB) must be the first functional element of B-1 tested. Do not write the setpoint loop before the heartbeat runs successfully for 30s in isolation.

- No speculative features. If a module is not in the scope definition in Part 3, it does not get built in this phase.

- Phase 1.5 is a go/no-go gate. If OFFBOARD cannot be held for 60s after following S-PX4-01..10 exactly, stop and escalate to TD before proceeding.

# **Part 11 — Hardware Readiness Statement**

This section defines the current hardware posture and what is required for each phase transition. It exists to prevent the programme from overstating hardware readiness to a partner.

## **11.1  Current Hardware State**

| **Component** | **State** | **Impact on Pre-HIL phase** | **Path to change** |
| --- | --- | --- | --- |
| Development workstation (micromind-node01) | Available. Ubuntu 24.04.4. Ryzen 7 9700X. RTX 5060 Ti 16GB. Python 3.12.3. | Full SIL and Gazebo SITL capability. All Phases 0–4 execute on this hardware. | No change required for Pre-HIL phase. |
| Compute payload board (e.g. Jetson Orin NX) | Not procured. | Phases 0–4 do not require it. Compute profiling targets Jetson Orin spec as reference. | Procurement decision gated on TASL partnership outcome. Post-TASL. |
| IMU hardware (STIM300 or ADIS16505-3) | Not procured. | Simulated via noise model. RealIMUDriver stub documents interface. Replacement is config change only. | Post-TASL. Requires RealIMUDriver implementation (~2 days once hardware available). |
| RADALT hardware | Not procured. | Simulated via DEMProvider. RADALT reclassified as MVP-required (Tech Review v1.1 R-03). | Post-TASL. Requires RealRADALTDriver implementation (~1 day once hardware available). |
| Thermal camera (EO/IR) | Not procured. | Stub only. DMRL uses synthetic scene. Interface contract defined in A-5. | Post-TASL + Phase 3 CNN work. Full integration blocked on ITL clearance. |
| SDR hardware | Not procured. | Stub only. EW engine uses random draw. Interface contract defined in A-6. | Phase 2+ post-TASL. SDR spec in V6 §4.5.1 (USB3/PCIe). |
| PX4-compatible autopilot (for HIL) | Not procured. | PX4 SITL (software) is available. Physical autopilot board required for HIL phase only. | Post-TASL hardware partnership. Pixhawk 6X or equivalent. |
| Vehicle airframe | Not procured. | Not required for SIL or SITL phases. | Post-TASL. ALS-250 class platform target. |

## **11.2  What Pre-HIL Phases Require vs What Is Available**

| **Phase** | **Hardware required** | **Hardware available?** | **Comment** |
| --- | --- | --- | --- |
| Phase 0 (Architecture lock) | Workstation only | **Yes** | No hardware dependency. |
| Phase 1 (Drivers + Config) | Workstation only | **Yes** | SimDrivers run on workstation. No real hardware needed. |
| Phase 1.5 (PX4 SITL) | Workstation only (PX4 SITL is software) | **Yes — software only** | PX4 SITL runs on the same workstation. No autopilot board required. |
| Phase 2 (Output pipelines + Timing) | Workstation only | **Yes** | Profiling targets Jetson Orin spec but runs on dev machine. |
| Phase 3 (Live input stubs) | Workstation only | **Yes** | Real driver stubs only. No hardware needed. |
| Phase 4 (Gazebo demo) | Workstation only | **Yes** | Full demo runs on workstation in SITL. |
| HIL validation | Jetson Orin + STIM300 + PX4 autopilot board + ALS-250 class airframe | **Not available — post-TASL** | Gated on TASL partnership decision and hardware procurement. |

**All Phases 0–4 (the entire Pre-HIL completion scope) execute on the development workstation. No hardware procurement is required to complete this phase. Hardware procurement begins post-TASL and does not block the integration software work.**

## **11.3  Observable Failure Behaviour Under Hardware Absence**

When a real hardware driver is selected (config source=real) but hardware is not connected, the following behaviour is required. This is part of the replaceability claim.

| **Driver** | **Config setting** | **Expected behaviour** | **Unacceptable behaviour** |
| --- | --- | --- | --- |
| RealIMUDriver | imu_source=real | Raises NotImplementedError on first read. Log entry: 'RealIMUDriver: device not connected. Check SPI interface.' System halts driver thread cleanly. Navigation loop is notified. | Silent failure. Navigation loop continues with stale or zero IMU data. |
| RealGNSSDriver | gnss_source=real | Raises NotImplementedError. Log entry: 'RealGNSSDriver: NMEA port not open. Check /dev/ttyUSB0.' BIM receives no GNSS updates. BIM remains at last trust score. | BIM trust score corrupted by null GNSS data. |
| RealRADALTDriver | radalt_source=real | Raises NotImplementedError. Log entry: 'RealRADALTDriver: RADALT not responding. TRN corrections suppressed.' TRN receives validity_flag=False. | TRN attempts correction with zero altitude. Navigation position corrupted. |
| RealEOIRDriver | eoir_source=real | Raises NotImplementedError. Log entry: 'RealEOIRDriver: camera interface not initialised.' DMRL receives no frames. Synthetic scene remains active. | DMRL receives null frames. Lock confidence falls to zero. Mission aborts spuriously. |
| RealSDRDriver | sdr_source=real | Raises NotImplementedError. Log entry: 'RealSDRDriver: SDR device not found.' EW engine receives no spectrum data. EW stub remains active. | EW engine receives null spectrum. Cost map corrupted. |
| MAVLink bridge (px4_output=real) | px4_output=real | connect() raises ConnectionError after 10s timeout. Log entry: 'MAVLink: PX4 not responding on udp:14550. Check SITL or hardware connection.' Navigation loop notified. Setpoint thread terminates cleanly. | Setpoint thread loops indefinitely on failed connection. CPU spike. |

**Observable failure behaviour is part of the integration claim. An OEM reviewer who runs config=real and sees a clean, informative error message understands that the system is genuinely replaceable. An OEM reviewer who sees a silent hang, a crash, or a corrupted state has lost confidence in the integration architecture.**

**Status: FINAL — Pre-HIL Software Integration Plan Closed. All architectural, integration, and validation requirements are defined. Ready for execution without further document changes.**
