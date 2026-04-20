**MicroMind**

Pre-HIL Phase 0 Architecture Decision Record

**v1.0  ·  28 March 2026  ·  BASELINED  ·  RESTRICTED**

| **Parameter** | **Value** |
| --- | --- |
| **Document** | MicroMind Pre-HIL — Phase 0 Architecture Decision Record |
| **Version** | v1.0 — BASELINED |
| **Date** | 28 March 2026 |
| **Status** | BASELINED — Phase 0 architecture decisions locked. Pre-code validation in progress. |
| **Basis** | MicroMind_PreHIL_v1_2.docx (26 March 2026, Final). SIL baseline: micromind-autonomy @ 9eac06b, 332/332 tests passing. |
| **Scope** | Integration-layer software only. No core module modifications. No hardware. No ML. |
| **Purpose** | Formal record of all Phase 0 architectural decisions before any integration code is written. Required for CP-0 gate review. |

This document must be completed and reviewed at CP-0 before Phase 1 code begins. It is not a design document — it records decisions already made. No decision in this record may be changed during Phase 1–4 execution without a new revision and TD sign-off.

# **1  Architecture Decisions (D-1 through D-4)**

Four decisions were locked in the Phase 0 session on 28 March 2026. All decisions follow v1.2 recommendations unless explicitly noted as refinements.

| **ID** | **Decision** | **Resolution** | **Rationale** | **Constraints** |
| --- | --- | --- | --- | --- |
| **D-1** | **Middleware** | Direct Python + pymavlink over UDP socket for the MAVLink bridge in Phases 1–2. No ROS2 in the bridge layer. | Avoids a second debugging layer before OFFBOARD is proven stable. Fastest path to Phase 1.5 PX4 demo. | ROS2 remains isolated to the openvins_humble Docker container for VIO only. No ROS2 imports under integration/. |
| **D-2** | **Time reference** | MicroMind owns the internal monotonic mission clock. All outgoing MAVLink timestamps derive from that clock. SYSTEM_TIME sent at connect as courtesy sync only. | Prevents timestamp mismatch failure mode (FM-5). MicroMind clock is the authority; PX4 adoption of SYSTEM_TIME is not required or assumed. | REFINEMENT vs v1.2: no dependency on PX4 adopting SYSTEM_TIME. Consistent outgoing timestamping from MicroMind side is the sole requirement. |
| **D-3** | **Driver depth** | Python ABCs for all driver classes. Each ABC exposes: read/publish methods + health(), last_update_time(), is_stale(), source_type(). | Matches S-NEP-04 modular design. Enables OEM replaceability claim: real hardware slots in by implementing the ABC. No duck typing. | ADDITION vs v1.2: four health/status methods added to all ABCs. Sim and Real implementations must both implement all methods. |
| **D-4** | **PX4 failure modes** | All 7 failure modes (FM-1 through FM-7) acknowledged as governing constraints before any MAVLink code is written. | December PX4 failure root-cause analysis. Each FM has a definitive fix. No code proceeds until the fix is implemented and verified. | ADDITION vs v1.2: FM-6 (wrong sysid/compid) and FM-7 (OFFBOARD accepted but vehicle not armed) added. See Section 2. |

# **2  PX4 Failure Mode Register (FM-1 through FM-7)**

The following failure modes are governing constraints for Phase 1.5 implementation. No MAVLink code may be written without all seven modes and their fixes understood by the implementer. FM-6 and FM-7 were added in this session beyond the v1.2 baseline.

| **FM** | **Failure mode** | **Why it is hard to see** | **Detection signal** | **Definitive fix** |
| --- | --- | --- | --- | --- |
| **FM-1** | **Heartbeat not running or blocked** | PX4 logs show mode change but not why. From MicroMind side: setpoints sent, no error returned. Looks like commands ignored. | T-MON observes mode reversion within <2s of OFFBOARD entry. | Dedicated T-HB daemon at 2 Hz, completely independent of nav loop. Never blocked by compute. Validated: OFFBOARD held 60s in isolation first. |
| **FM-2** | **Setpoints sent before OFFBOARD engaged** | Mode switch appears to succeed (MAV_RESULT_ACCEPTED), but vehicle immediately exits OFFBOARD due to setpoint gap at critical moment. | T-MON observes immediate reversion after successful ACK. | Pre-stream: send at least 2s of setpoints at ≥20 Hz before requesting mode change. Explicit wait for LOCAL_POSITION_NED valid. Only then send mode change. |
| **FM-3** | **Wrong coordinate frame** | Vehicle climbs when it should descend. Developer assumes flight controller bug. | Vehicle moves in wrong direction vs Gazebo expectation. Z-sign inversion is definitive. | Explicitly set coordinate_frame=1 (MAV_FRAME_LOCAL_NED). Z positive DOWN. frame_utils.rotate_pos_enu_to_ned() is the validated transform — call it, do not reimplement. |
| **FM-4** | **Setpoint rate too low** | Vehicle enters OFFBOARD briefly then exits. Appears as heartbeat issue. Rate logging required to distinguish. | T-LOG setpoint_hz < 18 Hz in bridge log. | T-SP loop is time-driven at 20 Hz constant, never event-driven. Independent of navigation decision rate. |
| **FM-5** | **Timestamp mismatch** | No visible error. PX4 receives messages but does not act. Looks like command rejection. | PX4 COMMAND_ACK silent. Setpoints received at PX4 but vehicle stationary. | Use MicroMind monotonic clock for time_boot_ms. Call SYSTEM_TIME at connect to compute boot offset. D-2 decision governs. |
| **FM-6** | **Wrong target_system / target_component IDs** | Commands silently ignored by PX4 if sysid/compid do not match. No error returned. | No COMMAND_ACK received for arm or mode commands within timeout. | Derive target_system and target_component from first received HEARTBEAT before sending any command. Verify against MAV_SYS_ID parameter baseline (V-8). |
| **FM-7** | **OFFBOARD accepted but vehicle not armed or commander rejects setpoints** | Mode transition ACK succeeds. Vehicle appears in OFFBOARD. But setpoints have no effect — vehicle does not move. | Three-part check after mode entry: (1) armed state flag, (2) mode transition confirmed in T-MON, (3) LOCAL_POSITION_NED changing in Gazebo within 2s of first setpoint. | Three-part explicit verification before declaring OFFBOARD stable: arm state confirmed, mode confirmed, setpoint consumption confirmed via position change. |

# **3  Driver ABC Interface Contract**

Every driver in the integration layer inherits from a Python ABC. The ABC defines the interface contract as method signatures and docstrings. No duck typing. Sim and Real implementations must both implement all methods.

## **3.1  Mandatory methods on every driver ABC**

| **Method** | **Return type** | **Contract** |
| --- | --- | --- |
| **health()** | DriverHealth (enum: OK / DEGRADED / FAILED) | Returns current health state. Must never raise. Called by LivePipeline health watchdog at 10 Hz. |
| **last_update_time()** | float (monotonic seconds) | Returns timestamp of last successful read. Used by is_stale() and bridge logger. |
| **is_stale()** | bool | Returns True if (now - last_update_time()) > driver-specific stale threshold. Thresholds defined per driver type in MissionConfig. |
| **source_type()** | str: 'sim' │ 'real' | Returns the source type string. Used by BridgeLogger to annotate log entries. DriverFactory sets this at construction. |
| **read()** | driver-specific dataclass | Primary data read. Raises DriverReadError on hardware fault. Returns last valid data + DEGRADED health on transient fault. |
| **close()** | None | Clean shutdown. Must be idempotent. Called by LivePipeline on any exit path including exception. |

RealXxxDriver implementations raise NotImplementedError on first read() if hardware is not connected. The error message must identify the interface path (e.g. SPI, /dev/ttyUSB0). Navigation loop is notified. System halts driver thread cleanly. This is the OEM replaceability claim — a clean, informative error is evidence the interface is real.

# **4  Telemetry Watchlist**

T-MON subscribes to and monitors the following MAVLink messages from PX4 SITL. Any message absent beyond its staleness threshold triggers a bridge health alert.

| **Message** | **Staleness threshold** | **Purpose** |
| --- | --- | --- |
| **HEARTBEAT** | 500 ms | PX4 alive check. Source of target_system and target_component (FM-6). Mode field monitors for unexpected OFFBOARD reversion. |
| **LOCAL_POSITION_NED** | 200 ms | EKF2 alignment verification (FM-7). Must be actively updating (position change >0.1 m) before OFFBOARD is requested. Also used for setpoint consumption confirmation. |
| **ATTITUDE** | 200 ms | Vehicle attitude. Used for Gazebo visual validation and post-run analysis. |
| **ESTIMATOR_STATUS** | 1 s | EKF2 health. local_position bit must be set before OFFBOARD engagement (S-PX4-01 requirement). |
| **EXTENDED_SYS_STATE** | 1 s | Armed state (FM-7). landed_state. vtol_state. |
| **COMMAND_ACK** | on-demand | Response to arm and mode change commands. Must arrive within 3 s of command send. Absence = FM-6 (wrong sysid/compid) or FM-7 condition. |
| **BATTERY_STATUS** | 5 s | Voltage monitor. Not a gating condition for Phase 1.5 but logged for completeness. |
| **SYSTEM_TIME** | on-demand | Received at connect. Used to compute MicroMind clock boot offset for D-2 time reference implementation. |

# **5  Bridge Logging Contract**

BridgeLogger records every TX/RX event on the MAVLink bridge. Log entries are written to a JSON-lines file by T-LOG (async queue consumer). The log is the primary diagnostic tool for Phase 1.5 failures.

## **5.1  Per-event fields (all events)**

| **Field** | **Type** | **Description** |
| --- | --- | --- |
| **t_monotonic** | float | MicroMind monotonic clock timestamp at event creation. Authoritative time reference (D-2). |
| **msg_type** | str | MAVLink message type string, e.g. HEARTBEAT, SET_POSITION_TARGET_LOCAL_NED. |
| **direction** | str | 'TX' or 'RX'. |
| **seq** | int│null | MAVLink sequence number if present in message header, else null. |
| **source_type** | str | Driver source type from source_type() — 'sim' or 'real'. Carried through to all bridge log entries. |

## **5.2  Event-specific fields**

| **Event** | **Additional fields logged** |
| --- | --- |
| **OFFBOARD mode request** | target_system, target_component, requested_mode, result (from COMMAND_ACK), ack_latency_ms |
| **ARM command** | target_system, target_component, arm_state_before, arm_state_after (from EXTENDED_SYS_STATE), ack_result |
| **COMMAND_ACK received** | command_id, result_code, result_str, latency_ms from matched TX event |
| **Setpoint TX** | x_m (NED North), y_m (NED East), z_m (NED Down, positive down), setpoint_hz (rolling 1s average), coordinate_frame (must be 1) |
| **Mode transition observed** | old_mode, new_mode, t_monotonic, direction (expected / unexpected). Unexpected transitions trigger immediate alert. |
| **Heartbeat RX** | base_mode, custom_mode, system_status, mavlink_version, derived target_system, derived target_component |
| **Staleness alert** | msg_type, last_seen_t, elapsed_ms, threshold_ms |

# **6  V-3 File Inventory — integration/ Directory**

All Pre-HIL work goes into the new integration/ directory. Zero existing files are modified. The directory is entirely new and does not exist at the SIL baseline commit (9eac06b).

Every subdirectory requires an __init__.py. All driver ABCs and implementations live under integration/drivers/. All MAVLink bridge threads and logging live under integration/bridge/. The navigation pipeline wrapper lives under integration/pipeline/. Mission configuration lives under integration/config/.

| **File path** | **Phase** | **Description** |
| --- | --- | --- |
| integration/__init__.py | Phase 1 | Package root. |
| integration/drivers/__init__.py | Phase 1 | Drivers subpackage root. |
| integration/drivers/base.py | Phase 1 | SensorDriver ABC — base class with health(), last_update_time(), is_stale(), source_type(), read(), close(). |
| integration/drivers/imu.py | Phase 1 | IMUDriver ABC — extends SensorDriver. read() returns IMUReading(accel_mss, gyro_rads, temp_c, t). |
| integration/drivers/gnss.py | Phase 1 | GNSSDriver ABC — extends SensorDriver. read() returns GNSSReading(lat, lon, alt, hdop, fix_type, t). |
| integration/drivers/radalt.py | Phase 1 | RADALTDriver ABC — extends SensorDriver. read() returns RADALTReading(alt_agl_m, validity_flag, t). |
| integration/drivers/eoir.py | Phase 1 | EOIRDriver ABC — extends SensorDriver. read() returns EOIRFrame(frame_np, timestamp, validity_flag). |
| integration/drivers/sim_imu.py | Phase 1 | SimIMUDriver — implements IMUDriver using existing imu_model.py noise model. source_type() = 'sim'. |
| integration/drivers/sim_gnss.py | Phase 1 | SimGNSSDriver — implements GNSSDriver using gnss_spoof_injector.py. source_type() = 'sim'. |
| integration/drivers/sim_radalt.py | Phase 1 | SimRADALTDriver — implements RADALTDriver using DEMProvider. source_type() = 'sim'. |
| integration/drivers/sim_eoir.py | Phase 1 | SimEOIRDriver — implements EOIRDriver with synthetic frame generation. source_type() = 'sim'. |
| integration/drivers/sim_sdr.py | Phase 1 | SimSDRDriver — implements SDRDriver with random EW draw. source_type() = 'sim'. |
| integration/drivers/real_imu.py | Phase 1 | RealIMUDriver — stub. raise NotImplementedError with SPI interface path on first read(). source_type() = 'real'. |
| integration/drivers/real_gnss.py | Phase 1 | RealGNSSDriver — stub. raise NotImplementedError with /dev/ttyUSB0 path on first read(). source_type() = 'real'. |
| integration/drivers/real_radalt.py | Phase 1 | RealRADALTDriver — stub. raise NotImplementedError. TRN receives validity_flag=False. source_type() = 'real'. |
| integration/drivers/real_eoir.py | Phase 1 | RealEOIRDriver — stub. raise NotImplementedError. DMRL receives no frames. source_type() = 'real'. |
| integration/drivers/real_sdr.py | Phase 1 | RealSDRDriver — stub. raise NotImplementedError. EW stub remains active. source_type() = 'real'. |
| integration/drivers/factory.py | Phase 1 | DriverFactory — reads MissionConfig, returns correct Sim or Real implementation per driver type. |
| integration/config/__init__.py | Phase 1 | Config subpackage root. |
| integration/config/mission_config.py | Phase 1 | MissionConfig dataclass — imu_source, gnss_source, radalt_source, eoir_source, sdr_source, px4_output, stale_thresholds. |
| integration/pipeline/__init__.py | Phase 1 | Pipeline subpackage root. |
| integration/pipeline/live_pipeline.py | Phase 1 | LivePipeline — 200 Hz navigation loop. Reads from driver registry, feeds ESKF. Exposes T-NAV thread. Produces setpoints to non-blocking queue for T-SP. |
| integration/bridge/__init__.py | Phase 1.5 | Bridge subpackage root. |
| integration/bridge/mavlink_bridge.py | Phase 1.5 | MAVLinkBridge — owns T-HB (2 Hz), T-SP (20 Hz), T-MON (10 Hz/event) threads. Implements FM-1..FM-7 fixes. Manages OFFBOARD engagement sequence. |
| integration/bridge/time_reference.py | Phase 1.5 | TimeReference — MicroMind monotonic clock, boot offset computation from SYSTEM_TIME, time_boot_ms derivation for all outgoing messages. |
| integration/bridge/bridge_logger.py | Phase 1.5 | BridgeLogger — TX/RX log contract from Section 5. JSON-lines output. Consumed by T-LOG async queue. |
| integration/tests/__init__.py | Phase 1 | Tests subpackage root. |
| integration/tests/test_prehil_drivers.py | Phase 1 | Driver ABC conformance tests. One test per driver type verifying all 6 ABC methods present and callable. Sim driver smoke tests. |
| integration/tests/test_prehil_bridge.py | Phase 1.5 | Bridge unit tests: T-HB isolation 30s, time_reference offset, bridge_logger field completeness. |

# **7  Thread Architecture — MAVLink Bridge**

Five threads. Thread separation is mandatory per v1.2 §Phase 1.5. T-NAV and T-SP must never share a thread. T-HB must never be blocked by any other thread.

| **Thread** | **Name** | **Rate** | **Blocking constraint** |
| --- | --- | --- | --- |
| **T-HB** | **Heartbeat daemon** | 2 Hz fixed | MUST NEVER BLOCK. Independent daemon. No dependency on nav loop state, setpoint queue, or any other thread. First functional element tested in isolation (30s) before T-SP is written. |
| **T-SP** | **Setpoint loop** | 20 Hz fixed | MUST NEVER BLOCK. Time-driven, never event-driven. Reads from non-blocking bounded queue fed by T-NAV. Sends NED setpoints with coordinate_frame=1. Continues at 20 Hz even if queue is empty (sends hold setpoint). |
| **T-MON** | **State monitor** | 10 Hz + on-event | Async receive only. Monitors telemetry watchlist (Section 4). Alerts on mode reversion, staleness, and COMMAND_ACK timeout. Read-only — never sends commands. |
| **T-NAV** | **Navigation loop** | 200 Hz (IMU rate) | Must never block T-HB, T-SP, or T-MON. Produces position setpoints to non-blocking queue for T-SP. Nav computation is independent of bridge state. |
| **T-LOG** | **Logger** | Async queue consumer | Isolated from all other threads. Reads from log queue, writes JSON-lines to disk. Never blocks navigation or bridge threads. |

T-NAV → T-SP interface is a non-blocking queue with bounded size. If the queue is full when T-NAV attempts to enqueue, T-NAV drops the setpoint silently and logs a queue-full event. T-SP never waits on T-NAV. This is the non-blocking queue constraint from v1.2 ADD-08.

# **8  Pre-Code Validation Checklist (V-1 through V-8)**

All eight items must be green before Phase 1 code begins. Items V-4 through V-8 require hands-on execution on micromind-node01. Status as of 28 March 2026 session.

| **#** | **Validation** | **Evidence / Notes** | **Method** | **Status** |
| --- | --- | --- | --- | --- |
| **V-1** | **Phase 0 decisions documented** | D-1 through D-4+FM-6+FM-7 recorded in this document. Session 28 March 2026. | Written decisions, this document | **PASS** |
| **V-2** | **PX4 failure modes reviewed and signed off** | FM-1 through FM-7 all acknowledged as governing constraints. See Section 2. | Section 2 review | **PASS** |
| **V-3** | **Module boundary diagram — new files only** | integration/ directory defined. 28 files. Zero overlap with core/, tests/, scenarios/. See Section 6. | Diagram + file inventory | **PASS** |
| **V-4** | **332 tests pass on clean state** | 332 passed in 111.86s. Commit 9eac06b. __pycache__ untracked. Remote sync confirmed. | python3 -m pytest tests/ -q | **PASS** |
| **V-5** | **PX4 SITL installs and vehicle spawns** | Pending execution on micromind-node01. | make px4_sitl + heartbeat check | **PENDING** |
| **V-6** | **frame_utils ENU→NED passes in isolation** | Pending execution on micromind-node01. | python3 inline assertion script | **PENDING** |
| **V-7** | **PX4 OFFBOARD works from QGC baseline** | Pending execution on micromind-node01. | QGroundControl OFFBOARD test | **PENDING** |
| **V-8** | **PX4 parameter baseline verified in QGC** | Pending. Six parameters: EKF2_AID_MASK, COM_RCL_EXCEPT, NAV_RCL_ACT, COM_OBL_ACT, CBRK_USB_CHK, MAV_SYS_ID. | QGC parameter screenshot + param show | **PENDING** |

Phase 0 exit gate (CP-0): all 8 items PASS. V-5 through V-8 require hands-on execution. Report results back to record in this document before Phase 1 code begins.

# **9  Frozen Baseline — Do Not Modify**

The following files and constants are frozen at the SIL baseline (9eac06b). No Pre-HIL work may modify them. Any change requires TD approval and S-NEP-09 re-run.

| **Item** | **Why frozen** |
| --- | --- |
| **core/ekf/error_state_ekf.py** | Frozen constants and validated interface. 332 gates calibrated to this baseline. S-NEP-09 anchored here. |
| **core/fusion/vio_mode.py** | Mode Integrity Invariant. VIONavigationMode transition latency confirmed 1-step. S-NEP-09 re-run required if changed. |
| **core/fusion/frame_utils.py** | ENU→NED transform validated in S-NEP-04. Do not rewrite. Call existing functions only. Never re-implement the transform. |
| **scenarios/bcmp1/bcmp1_runner.py (E-1..E-5)** | Enforcement blocks committed and validated two-theatre. TERM KPIs depend on this layer being active. |
| **_ACC_BIAS_RW, _GYRO_BIAS_RW, _POS_DRIFT_PSD, _GNSS_R_NOMINAL** | Frozen estimator constants. TD approval required for any modification. |
| **tests/ (332 gates)** | No gate modified to pass. All 332 pass unchanged after every integration addition. Run after every new module. |

End of ADR-0 · MicroMind Pre-HIL Phase 0 Architecture Decision Record v1.0
