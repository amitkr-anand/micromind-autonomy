**MicroMind**

Pre-HIL Phase 0 Architecture Decision Record

**v1.1  ·  28 March 2026  ·  BASELINED  ·  ALL V-1..V-8 PASS  ·  RESTRICTED**

v1.1 supersedes v1.0 (28 March 2026). Changes: SIL baseline commit updated to 46fd9e7; V-5 Gazebo target corrected to gz_x500 / Gazebo Harmonic; V-8 parameter table updated with PX4 v1.17 renames and live values; FM-6 compid=0 confirmed from live heartbeat; T-MON OFFBOARD custom_mode=393216 added; all V-1..V-8 checklist items updated to PASS; V-9 added per Sensor Integration Architecture addendum (MicroMind_SIA_v1_0).

| **Parameter** | **Value** |
| --- | --- |
| **Document** | MicroMind Pre-HIL — Phase 0 Architecture Decision Record |
| **Version** | v1.1 — BASELINED. Supersedes v1.0. |
| **Date** | 28 March 2026 |
| **Status** | BASELINED — All V-1..V-8 validated. Phase 0 gate PASS. Phase 1 code authorised to begin. |
| **Basis** | MicroMind_PreHIL_v1_2.docx (Final). SIL baseline: micromind-autonomy @ 46fd9e7, 332/332 tests passing. PX4 v1.17.0-alpha1, Gazebo Harmonic 8.11.0. |
| **Supersedes** | ADR-0 v1.0 (28 March 2026, commit 9eac06b). v1.0 remains in project knowledge for audit trail. |

# **1  Architecture Decisions (D-1 through D-4)**

All four decisions unchanged from v1.0. Recorded here for completeness.

| **ID** | **Decision** | **Resolution** | **Rationale** | **Constraints** |
| --- | --- | --- | --- | --- |
| **D-1** | **Middleware** | Direct Python + pymavlink over UDP socket. No ROS2 in bridge layer. | Avoids second debugging layer before OFFBOARD proven stable. | ROS2 isolated to openvins_humble Docker. No ROS2 imports under integration/. |
| **D-2** | **Time reference** | MicroMind monotonic clock authoritative. All outgoing MAVLink timestamps derive from it. SYSTEM_TIME sent at connect as courtesy only. | Prevents FM-5. MicroMind clock is authority; PX4 adoption of SYSTEM_TIME not required. | REFINEMENT vs v1.2: no dependency on PX4 adopting SYSTEM_TIME. |
| **D-3** | **Driver depth** | Python ABCs for all driver classes. Mandatory methods: read/publish + health(), last_update_time(), is_stale(), source_type(). | Matches S-NEP-04 modular design. OEM replaceability claim depends on this. | ADDITION vs v1.2: four health/status methods added. All implementations must expose all six methods. |
| **D-4** | **PX4 failure modes** | FM-1 through FM-7 all acknowledged as governing constraints before MAVLink code written. | December failure root-cause. Each FM has definitive fix. | ADDITION vs v1.2: FM-6 (wrong sysid/compid) and FM-7 (OFFBOARD accepted, vehicle not armed) added. |

# **2  PX4 Failure Mode Register (FM-1 through FM-7)**

FM-6 updated with live finding from V-5 validation: compid=0, not compid=1. All other failure modes unchanged from v1.0.

| **FM** | **Failure mode** | **Why it is hard to see** | **Detection signal** | **Definitive fix** |
| --- | --- | --- | --- | --- |
| **FM-1** | **Heartbeat not running or blocked** | PX4 logs show mode change but not why. Looks like commands ignored. | T-MON observes mode reversion within <2s of OFFBOARD entry. | Dedicated T-HB daemon at 2 Hz, independent of nav loop. Validated 30s isolation before T-SP written. |
| **FM-2** | **Setpoints sent before OFFBOARD engaged** | Mode switch ACK succeeds but vehicle immediately exits OFFBOARD. | T-MON observes immediate reversion after successful ACK. | Pre-stream: 2s of setpoints at ≥20 Hz before mode change request. Wait for LOCAL_POSITION_NED valid first. |
| **FM-3** | **Wrong coordinate frame** | Vehicle climbs when should descend. Developer assumes flight controller bug. | Vehicle moves wrong direction vs Gazebo expectation. Z-sign inversion definitive. | coordinate_frame=1 (MAV_FRAME_LOCAL_NED). Z positive DOWN. Call frame_utils.rotate_pos_enu_to_ned(). Never reimplement. |
| **FM-4** | **Setpoint rate too low** | Vehicle enters OFFBOARD briefly then exits. Appears as heartbeat issue. | T-LOG setpoint_hz < 18 Hz in bridge log. | T-SP time-driven at 20 Hz constant, never event-driven. Independent of navigation decision rate. |
| **FM-5** | **Timestamp mismatch** | No visible error. PX4 receives messages but does not act. | COMMAND_ACK silent. Setpoints received but vehicle stationary. | MicroMind monotonic clock for time_boot_ms. SYSTEM_TIME at connect for boot offset. D-2 decision governs. |
| **FM-6** | **Wrong target_system / target_component IDs  [UPDATED v1.1: compid=0 confirmed live]** | Commands silently ignored. No error returned. | No COMMAND_ACK received for arm or mode commands within timeout. | LIVE FINDING: sysid=1, compid=0 (not 1). Derive both from first received HEARTBEAT. Never hardcode. Hardcoded compid=1 will cause silent rejection. |
| **FM-7** | **OFFBOARD accepted but vehicle not armed or commander rejects setpoints** | Mode transition ACK succeeds but setpoints have no effect. | Three-part check: (1) armed state flag, (2) mode confirmed in T-MON, (3) LOCAL_POSITION_NED changing >0.1m within 2s. | Explicit three-part verification: arm confirmed, mode confirmed, setpoint consumption confirmed via position change. |

v1.1 FM-6 update: live V-5 validation confirmed sysid=1, compid=0. The v1.2 governing document assumed compid=1. All bridge code must derive both IDs from the first received HEARTBEAT. Any hardcoded compid=1 will cause silent command rejection.

# **3  Driver ABC Interface Contract**

Unchanged from v1.0. All six methods mandatory on every driver ABC.

| **Method** | **Return type** | **Contract** |
| --- | --- | --- |
| **health()** | DriverHealth (enum: OK / DEGRADED / FAILED) | Returns current health state. Must never raise. Called by LivePipeline health watchdog at 10 Hz. |
| **last_update_time()** | float (monotonic seconds) | Returns timestamp of last successful read. |
| **is_stale()** | bool | True if (now - last_update_time()) > driver-specific stale threshold from MissionConfig. |
| **source_type()** | str: 'sim' │ 'real' | Returns source type. Used by BridgeLogger. Set at construction by DriverFactory. |
| **read()** | driver-specific dataclass | Primary data read. Raises DriverReadError on hardware fault. Returns last valid + DEGRADED health on transient fault. |
| **close()** | None | Clean shutdown. Idempotent. Called on any exit path including exception. |

# **4  Telemetry Watchlist**

Unchanged from v1.0.

| **Message** | **Staleness** | **Purpose** |
| --- | --- | --- |
| **HEARTBEAT** | 500 ms | PX4 alive check. Source of target_system and target_component (FM-6). Mode field monitors for OFFBOARD reversion. OFFBOARD custom_mode=393216 (confirmed V-7). |
| **LOCAL_POSITION_NED** | 200 ms | EKF2 alignment verification (FM-7). Must be actively updating (>0.1 m change) before OFFBOARD. Setpoint consumption confirmation. |
| **ATTITUDE** | 200 ms | Vehicle attitude. Gazebo visual validation and post-run analysis. |
| **ESTIMATOR_STATUS** | 1 s | EKF2 health. local_position bit must be set before OFFBOARD (S-PX4-01). |
| **EXTENDED_SYS_STATE** | 1 s | Armed state (FM-7). landed_state. vtol_state. |
| **COMMAND_ACK** | on-demand | Response to arm and mode commands. Must arrive within 3s. Absence = FM-6 or FM-7. |
| **BATTERY_STATUS** | 5 s | Voltage monitor. Logged for completeness. |
| **SYSTEM_TIME** | on-demand | Received at connect. Boot offset computation for D-2 time reference. |
| **HIGHRES_IMU** | 5 ms (200 Hz) | [ADDED v1.1 per SIA] Primary IMU listener path. Force to 200 Hz via PX4 stream config. P95 jitter measured in S-PX4-09. |
| **GPS_RAW_INT** | 200 ms (5 Hz) | [ADDED v1.1 per SIA] GNSS listener path for BIM integrity scoring. Force to 5 Hz. |
| **DISTANCE_SENSOR** | 100 ms (10 Hz) | [ADDED v1.1 per SIA] RADALT listener path if fitted on PX4. Enable stream if available. |

# **5  Bridge Logging Contract**

Unchanged from v1.0. See v1.0 Section 5 for full field definitions.

# **6  V-3 File Inventory — integration/ Directory**

Unchanged from v1.0. 28 files across integration/drivers/, integration/config/, integration/pipeline/, integration/bridge/, integration/tests/. Zero overlap with core/, tests/, or scenarios/.

# **7  Thread Architecture — MAVLink Bridge**

Thread structure unchanged from v1.0. T-MON note updated with live OFFBOARD custom_mode value.

| **Thread** | **Name** | **Rate** | **Blocking constraint** |
| --- | --- | --- | --- |
| **T-HB** | **Heartbeat daemon** | 2 Hz fixed | MUST NEVER BLOCK. Independent daemon. First functional element tested in isolation (30s) before T-SP is written. |
| **T-SP** | **Setpoint loop** | 20 Hz fixed | MUST NEVER BLOCK. Time-driven, never event-driven. Reads non-blocking bounded queue from T-NAV. Sends hold setpoint if queue empty. coordinate_frame=1 always. |
| **T-MON** | **State monitor** | 10 Hz + on-event | Async receive only. Monitors telemetry watchlist. Watches custom_mode field of HEARTBEAT — OFFBOARD = 393216 (confirmed V-7). Alerts on reversion, staleness, COMMAND_ACK timeout. |
| **T-NAV** | **Navigation loop** | 200 Hz (IMU rate) | Must never block T-HB, T-SP, or T-MON. Produces position setpoints to non-blocking bounded queue for T-SP. |
| **T-LOG** | **Logger** | Async queue consumer | Isolated from all threads. JSON-lines to disk. Never blocks navigation or bridge threads. |

v1.1: T-MON must watch for custom_mode=393216 to confirm OFFBOARD engagement. This value was confirmed from live V-7 validation on PX4 v1.17.0-alpha1 gz_x500.

# **8  Pre-Code Validation Checklist (V-1 through V-9)**

All V-1 through V-8 items confirmed PASS. V-9 added per MicroMind_SIA_v1_0 addendum.

| **#** | **Validation** | **Evidence / Notes** | **Method** | **Status** |
| --- | --- | --- | --- | --- |
| **V-1** | **Phase 0 decisions documented** | D-1..D-4+FM-6+FM-7 recorded in this document. Session 28 March 2026. | Written decisions | **PASS** |
| **V-2** | **PX4 failure modes reviewed and signed off** | FM-1 through FM-7 acknowledged as governing constraints. See Section 2. | Section 2 review | **PASS** |
| **V-3** | **Module boundary diagram — new files only** | integration/ directory: 28 files. Zero overlap with core/, tests/, scenarios/. See Section 6. | Diagram + inventory | **PASS** |
| **V-4** | **332 tests pass on clean state** | 332 passed in 111.86s. Commit 9eac06b. __pycache__ untracked. Remote synced to 46fd9e7. | python3 -m pytest tests/ -q | **PASS** |
| **V-5** | **PX4 SITL installs and vehicle spawns** | PX4 v1.17.0-alpha1. Gazebo Harmonic 8.11.0. Target: gz_x500 (4001). Heartbeat: sysid=1, compid=0. pymavlink 2.4.49. NOTE: Gazebo Classic not present — gz_x500 used, not gazebo-classic_iris. | make px4_sitl gz_x500 + heartbeat check | **PASS** |
| **V-6** | **frame_utils ENU→NED passes in isolation** | North [1,0,0] ✓  East [0,1,0] ✓  Z-flip [0,0,-1] ✓  Cov eigvals [0.01,0.02,0.04] ✓ | python3 assertion script | **PASS** |
| **V-7** | **PX4 OFFBOARD works via pymavlink** | Pre-stream 3s FM-2 fix confirmed. COMMAND_ACK result=0. Mode=393216 at t+1.0s. Held 10s no reversion. base_mode=145 (armed+OFFBOARD). | pymavlink OFFBOARD script | **PASS** |
| **V-8** | **PX4 parameter baseline verified** | EKF2_GPS_CTRL=7 (replaces EKF2_AID_MASK), COM_RCL_EXCEPT=4 (set), NAV_RCL_ACT=0 (set), COM_OBL_RC_ACT=0 (replaces COM_OBL_ACT), CBRK_USB_CHK=197848, MAV_SYS_ID=1. Persisted to px4-rc.params. | pymavlink param read + set | **PASS** |
| **V-9** | **PX4 sensor stream rates verified (SIA addendum)** | Pending: HIGHRES_IMU at 200 Hz, GPS_RAW_INT at 5 Hz, ESTIMATOR_STATUS at 5 Hz. Execute before S-PX4-01 in Phase 1.5. | mavlink status streams | **PENDING** |

v1.1: V-5 corrected — Gazebo Harmonic gz_x500, not Gazebo Classic iris. V-8 updated — two parameter renames confirmed (EKF2_AID_MASK → EKF2_GPS_CTRL, COM_OBL_ACT → COM_OBL_RC_ACT). All V-1..V-8 PASS. V-9 added per SIA addendum — PENDING execution in Phase 1.5.

# **9  Frozen Baseline — Do Not Modify**

Unchanged from v1.0.

| **Item** | **Why frozen** |
| --- | --- |
| **core/ekf/error_state_ekf.py** | Frozen constants and validated interface. 332 gates calibrated to this baseline. |
| **core/fusion/vio_mode.py** | Mode Integrity Invariant. Transition latency confirmed 1-step. S-NEP-09 re-run required if changed. |
| **core/fusion/frame_utils.py** | ENU→NED transform validated in S-NEP-04. Call existing functions. Never reimplement. |
| **scenarios/bcmp1/bcmp1_runner.py (E-1..E-5)** | Enforcement blocks committed and validated two-theatre. |
| **_ACC_BIAS_RW, _GYRO_BIAS_RW, _POS_DRIFT_PSD, _GNSS_R_NOMINAL** | Frozen estimator constants. TD approval required. |
| **tests/ (332 gates)** | No gate modified to pass. Run after every new module addition. |

End of ADR-0 v1.1 · MicroMind Pre-HIL Phase 0 Architecture Decision Record · Supersedes v1.0
