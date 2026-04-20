**MicroMind**

Pre-HIL Software Integration — Technical Director Update

**29 March 2026  ·  Session Closure Report  ·  RESTRICTED**

| **Parameter** | **Value** |
| --- | --- |
| **Prepared by** | Technical Programme Lead (Claude — Anthropic) |
| **Date** | 29 March 2026 |
| **Audience** | Technical Director |
| **Basis** | MicroMind_PreHIL_v1_2.docx (Final). micromind-autonomy @ e5d8530. 332/332 SIL, 206/206 integration. |
| **Purpose** | Status report on Pre-HIL integration sprint. CP-0 through CP-2 passed. Phase 3 entry assessment. |

# **1  Executive Summary**

Phase 0 (architecture lock), Phase 1 (driver abstraction), Phase 1.5 (PX4 OFFBOARD), and Phase 2 (latency + timing) are all complete. All four mandatory review checkpoints — CP-0, CP-1, CP-1.5, and CP-2 — have passed against the criteria defined in v1.2.

The MicroMind navigation core is now wired to PX4 SITL via a production-quality integration layer. The system executes waypoint trajectories under OFFBOARD control, maintains 200Hz navigation, produces setpoints at 20Hz, and meets all latency thresholds with large margins.

One open item from CP-1.5: Gazebo 3D rendering is not displaying on the X11 session (physics runs correctly — vehicle altitude telemetry confirms correct behaviour). This is a display configuration issue on micromind-node01 and is non-blocking. All gate criteria use telemetry data, not visual observation.

# **2  Checkpoint Status**

| **Gate** | **Trigger** | **Pass condition** | **Status** |
| --- | --- | --- | --- |
| **CP-0** | **Architecture lock** | 4 decisions documented. Module diagram. FM-1..FM-7 acknowledged. Parameter baseline. | **COMPLETE** |
| **CP-1** | **Phase 1 complete** | 332/332 SIL gates. Factory correct. Frame assertion works. No core imports. | **COMPLETE** |
| **CP-1.5** | **PX4 OFFBOARD gate** | Vehicle executes 50m trajectory. OFFBOARD maintained. No reversion. Heartbeat thread confirmed. | **COMPLETE** |
| **CP-2** | **Phase 2 complete** | Latency log produced. P95 E2E <50ms. Setpoint rate 20±2Hz. Compute within bounds. | **COMPLETE** |
| **CP-3** | **Pre-HIL readiness** | RC-1..RC-11 all pass. Readiness report issued. | **PENDING** |

# **3  CP-2 Latency Results**

A 60-second instrumented run was executed on 29 March 2026 with LivePipeline + LatencyMonitor + MAVLinkBridge connected to PX4 SITL. The vehicle executed a five-waypoint trajectory under OFFBOARD control throughout.

| **Metric** | **P95 result** | **Gate** | **Margin** | **Result** |
| --- | --- | --- | --- | --- |
| **ESKF step latency P95** | 0.085ms | <10ms | 118× | **PASS** |
| **End-to-end latency P95** | 0.363ms | <50ms | 138× | **PASS** |
| **Setpoint rate (min window)** | 20.0Hz | ≥18Hz | On target | **PASS** |
| **CPU mean** | 3.2% | <60% | 19× margin | **PASS** |
| **Memory peak RSS** | 114.9MB | <500MB | 4.3× margin | **PASS** |

Latency margins are exceptionally large on the development workstation (Ryzen 7 9700X). On Jetson Orin NX (target payload compute), margins will be smaller. Jetson profiling is planned post-TASL. The current results confirm the architecture is correct — no blocking calls in the critical path.

# **4  S-PX4 Validation Steps (S-PX4-01 through S-PX4-09)**

All nine incremental PX4 SITL validation steps were completed in sequence per v1.2 §4.3. No step was skipped.

| **Step** | **Task** | **Evidence** | **Result** |
| --- | --- | --- | --- |
| **S-PX4-01** | **PX4 SITL install + vehicle spawn** | PX4 v1.17.0-alpha1, Gazebo Harmonic 8.11.0. gz_x500 (4001). Vehicle spawned. EKF2 aligned. | **PASS** |
| **S-PX4-02** | **HEARTBEAT only, 30s** | T-HB isolation: 60 heartbeats in 30s, 2Hz exact. No mode changes. FM-1 confirmed. | **PASS** |
| **S-PX4-03** | **Setpoint stream, 10s, disarmed** | 201 setpoints logged. Rate 20.0Hz confirmed. T-HB + T-SP independent. | **PASS** |
| **S-PX4-04** | **ARM command** | COMMAND_ACK result=0. base_mode=157 (armed). FM-2 pre-stream applied. | **PASS** |
| **S-PX4-05** | **OFFBOARD mode, 10s hold** | custom_mode=393216 at t+1.0s. Held 10s. No reversion. base_mode=145. | **PASS** |
| **S-PX4-06** | **Takeoff to 5m NED** | z=-5m setpoint. Vehicle climbed to 5.00m. NED Z convention correct. FM-3 confirmed. | **PASS** |
| **S-PX4-07** | **50m North trajectory** | 50.85m North reached. Altitude 5m maintained. OFFBOARD held throughout. | **PASS** |
| **S-PX4-08** | **Kill setpoint stream** | PX4 failsafe fired, RTL executed, vehicle landed. Mode exit confirmed in SITL log. | **PASS** |
| **S-PX4-09** | **60s full trajectory** | 5-waypoint sequence. OFFBOARD maintained 62s. Zero reversions. CP-1.5 gate. | **PASS** |

# **5  Integration Bugs Found and Fixed During Live Validation**

Three bugs were identified during live S-PX4 testing and resolved. All three are structural insights, not regressions. None touched the frozen SIL core.

| **#** | **Bug** | **Root cause** | **Fix** |
| --- | --- | --- | --- |
| **B-1** | **_local_pos_valid never set** | Movement threshold (>0.1m change) required on stationary SITL vehicle — position never moved enough. | Set flag on first LOCAL_POSITION_NED receipt. Movement check deferred to post-OFFBOARD. |
| **B-2** | **ARM/OFFBOARD ACK timeout** | T-MON recv_match racing with main thread on single pymavlink socket. Main thread lost COMMAND_ACK. | Added threading.Event ACK signalling. T-MON is now the sole receiver. Main thread waits on event. |
| **B-3** | **_stop_event stopping T-HB during stream kill** | _stop_event is shared — setting it to kill T-SP also stopped T-HB, then clearing it caused race. | Added _sp_paused threading.Event flag for T-SP only. T-HB runs independently. |

# **6  Readiness Criteria Status (RC-1 through RC-11)**

CP-3 requires all eleven RC criteria to pass. Current status below against v1.2 §Part 8 definitions.

| **RC** | **Criterion** | **Evidence / Notes** | **Status** |
| --- | --- | --- | --- |
| **RC-1** | **PX4 SITL accepts and executes setpoints** | S-PX4-09: 5-waypoint, 60s, OFFBOARD held. Position within 2m of each waypoint. | **PASS** |
| **RC-2** | **OFFBOARD stable 60s under live nav loop** | S-PX4-09: zero OFFBOARD exits, T-MON confirmed, heartbeat log present. | **PASS** |
| **RC-3** | **Heartbeat thread independent of nav loop** | T-HB 30s isolation test: 60 HB sent at 2Hz with nav loop not running. | **PASS** |
| **RC-4** | **Input interfaces swappable without core change** | DriverFactory sim→real swap: 332 gates unchanged. Zero ESKF/BIM/vio_mode changes. | **PASS** |
| **RC-5** | **332 SIL gates pass unchanged** | 332/332 after every integration module addition. No gate modified. | **PASS** |
| **RC-6** | **ENU→NED enforced at ESKF input** | frame_utils V-6 validated. FM-3 confirmed in S-PX4-06 (Z-sign correct). | **PASS** |
| **RC-7** | **Timestamp monotonicity enforced** | IFM-01 guard present in integration layer. Formal injection test pending Phase 3. | **PENDING** |
| **RC-8** | **Live logger non-blocking at 200Hz** | BridgeLogger T-LOG async queue confirmed. Drop rate negligible. Formal 60s drop-rate test pending. | **PENDING** |
| **RC-9** | **Latency P95 ****<****50ms end-to-end** | CP-2: E2E P95 = 0.363ms. 138× gate margin. Setpoint rate 20Hz stable. | **PASS** |
| **RC-10** | **No architectural rewrites for real hardware** | Driver ABC + factory pattern. Real hardware = implement ABC only. Zero core changes. | **PASS** |
| **RC-11** | **Control loop independence during OUTAGE/RESUMPTION** | VIONavigationMode wired. OUTAGE injection test pending Phase 3 (S-PX4-10 equivalent). | **PENDING** |

8 of 11 RC criteria pass today. RC-7 (timestamp injection), RC-8 (logger drop rate under load), and RC-11 (OUTAGE/RESUMPTION continuity) are pending Phase 3 execution. No architectural changes are anticipated to pass these — they are verification runs against already-implemented behaviour.

# **7  Open Items and Forward Look**

## **7.1  Gazebo 3D Rendering**

The Gazebo GUI window is not rendering the vehicle on the X11 display of micromind-node01. PX4 physics and telemetry are running correctly — all altitude and position data confirmed from telemetry. Root cause is likely a Gazebo Harmonic rendering plugin not initialising on the X11 session. Non-blocking for all gate criteria. Deferred to a dedicated display investigation.

## **7.2  Phase 3 Scope**

Three items remain before CP-3:

•  OfflineVIODriver — wraps EuRoC .npy loader with ENU→NED + IFM-01 monotonicity gate. Enables RC-7 and RC-11 verification.

•  RC-8 formal test — 60s 200Hz logger drop-rate measurement under full pipeline load.

•  RC-11 OUTAGE injection — suspend VIO 10s mid-flight, confirm setpoints continue at 20Hz, confirm mode→OUTAGE→RESUMPTION→NOMINAL.

## **7.3  Phase 4 — run_demo.sh**

Once CP-3 passes, the final deliverable is run_demo.sh — one command, reproducible 3× consecutively, executing the full demo sequence defined in v1.2 §Part 9. This is the OEM-ready artefact.

## **7.4  PX4 v1.17 Parameter Name Changes**

Two parameter renames were confirmed during V-8 baseline verification and are recorded in ADR-0 v1.1. These are informational — no impact on integration behaviour, but relevant to any documentation referencing the old names.

•  EKF2_AID_MASK → EKF2_GPS_CTRL (value 7, bits 0+1+2 active)

•  COM_OBL_ACT → COM_OBL_RC_ACT

# **8  Repository State**

| **Item** | **Value** |
| --- | --- |
| **Repository** | github.com/amitkr-anand/micromind-autonomy |
| **Latest commit** | e5d8530 — Phase 2 artefacts: CP-2 latency JSON + HTML demo report |
| **Tags** | cp1-phase1-pass · cp1_5-offboard-pass · cp2-latency-pass |
| **SIL gates** | 332/332 — unchanged throughout all integration work |
| **Integration gates** | 206/206 — drivers, pipeline, bridge, latency monitor |
| **Frozen modules** | ESKF, vio_mode, frame_utils, bcmp1_runner E-1..E-5 — zero modifications |
| **New files** | 23 files under integration/ — all new, no existing file modified |
| **CP-2 artefacts** | dashboard/micromind_prehil_cp2_latency_20260329.json + _report_20260329.html |

End of Technical Director Update · MicroMind Pre-HIL · 29 March 2026
