**MicroMind**

**Sensor Integration Architecture**

*v1.0  —  Listener-First Integration Position*

| **Parameter** | **Value** |
| --- | --- |
| Version | 1.0 |
| Date | 28 March 2026 |
| Status | BASELINED — governs Pre-HIL and HIL sensor integration decisions |
| Applies to | MicroMind Pre-HIL phase, HIL phase, and all OEM integration discussions |
| Relationship to Pre-HIL v1.2 | Addendum. Does not modify v1.2. Governs sensor integration decisions within the v1.2 execution plan. |
| Key principle | Listener-first, not listener-only. PX4 listens and performance evidence drives upgrade. |

**This document defines the authoritative sensor integration architecture for MicroMind.**

# **1  Guiding Principle: Listener-First, Not Listener-Only**

The initial position of 'listener-only' was architecturally clean but operationally compromised. A pure listener model sacrifices performance for integration simplicity at points where the performance cost is unacceptable. The adopted position is listener-first: subscribe to what PX4 already broadcasts wherever performance is adequate, and take direct ownership only where performance demands it or where no listener path exists.

**The decision rule is explicit:**

- If PX4 broadcasts the data at the required rate with acceptable latency and fidelity, MicroMind subscribes. No direct connection is added.

- If PX4 does not broadcast the data, or if transport jitter demonstrably degrades ESKF performance, MicroMind takes direct ownership of the sensor.

- If the sensor is intrinsically a MicroMind sensor (camera for VIO, SDR for EW), MicroMind connects directly regardless of whether a listener path theoretically exists.

*This position is consistent with the Hivemind-class integration model: not always pure plug-and-play, but not requiring full autopilot replacement or deep rewrites at any phase.*

# **2  Responsibility Split**

The following responsibility assignment is fixed. It defines what each system owns and does not own.

## **2.1  PX4 Retains Full Responsibility For**

| **Function** | **Why MicroMind does not own it** |
| --- | --- |
| Flight stability and attitude control | PX4's primary function. MicroMind produces position setpoints only. No actuator-level commands. |
| Actuator mixing and motor control | Flight-critical. Any interference invalidates platform certification. |
| EKF2 internal state estimation | PX4 EKF2 remains the active flight estimator. MicroMind does not replace or modify it. |
| Barometer | Slow sensor (10-50Hz). MAVLink listener path carries negligible performance penalty. PX4 uses it for its own altitude hold. |
| Magnetometer | PX4 needs it for EKF2 and motor interference calibration. Dual ownership creates interference risk. |
| Airspeed sensor | PX4 uses it for flight envelope management. Listener path is fully adequate for dead reckoning. |
| Optical flow (if fitted) | PX4 owns this for its own hover and landing modes. MicroMind subscribes as supplementary AGL aid. |
| GNSS primary fusion | PX4 EKF2 fuses GNSS for its own position estimate. MicroMind receives GNSS data separately for integrity scoring only. |

## **2.2  MicroMind Owns Directly**

| **Sensor / function** | **Rationale for direct ownership** | **Interface** |
| --- | --- | --- |
| EO camera (navigation and terminal) | No MAVLink path for video exists. MIPI is point-to-point. At ALS-250 class there is typically no companion computer already processing camera frames. Direct connection is the only path. | MIPI CSI-2 (preferred) or USB3 |
| SDR / EW payload | No autopilot interfaces with SDR. PX4 broadcasts no RF spectrum data. MicroMind is the only consumer. Direct is the only architecture. | USB3 or M.2 PCIe |
| Thermal / IR camera (Phase 2) | Same logic as EO camera. Thermal cameras never connect to autopilots. DMRL requires raw 16-bit radiometric frames that a compressed ROS2 stream may not carry. | USB3 or GigE |

## **2.3  MicroMind Adds: Navigation and Mission Layer**

MicroMind's functional contribution sits entirely above PX4's flight control layer:

- Navigation integrity and GNSS-denied position estimation (ESKF, TRN, VIO)

- Spoof detection and GNSS trust scoring (BIM)

- EW environment sensing and route replanning

- Mission envelope enforcement and ROE gate (L10s-SE)

- Setpoint generation and transmission (position commands via MAVLink OFFBOARD)

- Mission event logging and audit trail

# **3  Listener Path: Required MAVLink Messages and Rates**

The following MAVLink messages must be broadcast by PX4 at the specified minimum rates for MicroMind to function correctly. These are configuration requests, not PX4 modifications. They are set via QGroundControl or parameter upload before integration.

*Important: PX4 stock configuration often broadcasts many of these messages at lower rates than required. The PX4 Configuration Baseline in Section 5 specifies the exact parameter settings.*

| **MAVLink message** | **MicroMind use** | **Min rate required** | **PX4 default** | **Action needed** |
| --- | --- | --- | --- | --- |
| HIGHRES_IMU | Raw IMU data (accel + gyro + mag) for ESKF propagation | 200 Hz | 50 Hz | Must force to 200Hz. See Section 5. |
| ATTITUDE_QUATERNION | Attitude cross-check for VIO frame validation | 50 Hz | 50 Hz | **Default sufficient** |
| LOCAL_POSITION_NED | PX4 position estimate for pre-OFFBOARD validation. EKF2 alignment check. | 10 Hz | 10 Hz | **Default sufficient** |
| GPS_RAW_INT | Raw GNSS data for BIM spoof detection and integrity scoring | 5 Hz | 1 Hz | Force to 5Hz. See Section 5. |
| DISTANCE_SENSOR | RADALT AGL altitude for TRN corrections (if RADALT on PX4) | 10 Hz | Disabled by default | Enable if RADALT fitted. See Section 5. |
| SCALED_PRESSURE | Barometric altitude for ESKF vertical stabilisation | 10 Hz | 10 Hz | **Default sufficient** |
| HEARTBEAT (from PX4) | PX4 flight mode, arm state, system status. Required for OFFBOARD mode monitor. | 1 Hz | 1 Hz | **Default sufficient** |
| SYS_STATUS | General health, sensor availability flags | 1 Hz | 1 Hz | **Default sufficient** |
| VFR_HUD | Airspeed for dead reckoning augmentation | 5 Hz | 5 Hz | **Default sufficient** |
| ESTIMATOR_STATUS | EKF2 alignment flags. Bit 0 (attitude) and Bit 5 (local pos) must be set before OFFBOARD. | 5 Hz | 5 Hz | **Default sufficient** |

# **4  Sensor-by-Sensor Integration Decision**

The following table defines the integration path for each sensor, the reasoning, and the performance penalty if the listener path is used instead of direct ownership.

| **Sensor** | **Phase 1 path** | **Long-term path** | **Performance penalty if listener-only** |
| --- | --- | --- | --- |
| IMU | Listener via HIGHRES_IMU at 200Hz. Measure P95 jitter. Accept if P95 < 0.5ms. | Direct SPI if P95 jitter > 0.5ms sustained across a 60s run. | High if jitter is significant. Double-fusion risk if PX4 broadcasts processed attitude instead of raw samples. Must confirm HIGHRES_IMU carries raw accel/gyro, not EKF2 outputs. |
| EO camera (VIO + terminal) | **Direct MIPI CSI-2 or USB3. Always direct. No listener path for video.** | **Direct. Unchanged.** | No listener path exists. MAVLink cannot carry video frames. If a companion computer publishes /camera/image_raw over ROS2, that is an alternative direct path — not a listener path. |
| Barometer | **Listener via SCALED_PRESSURE at 10Hz.** | **Listener. Unchanged.** | Negligible. Barometer is a slow bias correction source. MAVLink jitter at 10Hz is irrelevant for ESKF vertical stabilisation. |
| GNSS | Listener via GPS_RAW_INT at 5Hz for BIM trust scoring. | Optional direct UART Y-cable for raw UBX/NMEA. Required if spoofing scenarios demand pre-filter signal. | Moderate. GPS_RAW_INT carries post-filter data. Sophisticated spoofing that passes PX4's filter will not be detectable on the listener path. For Phase 1 demo scenarios, listener is sufficient. |
| RADALT | Listener via DISTANCE_SENSOR at 10Hz (if RADALT connected to PX4). Direct UART if platform RADALT does not route through PX4. | Direct UART/CAN for terrain-following, low-level ingress, flare, and terminal guidance precision. | Low for navigation. High for terminal phase. Listener-only is acceptable for Phase 1 corridor navigation. For precision low-altitude approach and terminal alignment, direct wiring is required in later HIL. |
| Magnetometer | **Listener via HIGHRES_IMU mag fields.** | **Listener. Unchanged.** | Negligible. Magnetometer is a slow heading reference used only during camera outage. PX4 must own it for its own EKF2 and motor interference calibration. |
| Airspeed | **Listener via VFR_HUD at 5Hz.** | **Listener. Unchanged.** | None. Airspeed is a dead-reckoning supplement. Rate and latency via MAVLink are fully adequate. |
| Optical flow | **Listener via OPTICAL_FLOW_RAD (if fitted).** | **Listener. Unchanged.** | None. Low-altitude, low-speed aid. PX4 owns this for its landing modes. MicroMind uses it as a supplementary input below 10m AGL only. |
| SDR / EW | **Direct USB3 or M.2 PCIe. Always direct. No PX4 path exists.** | **Direct. Unchanged.** | No listener path exists. SDR is a MicroMind-native sensor. |
| Thermal / IR camera | Not in Phase 1. Stub driver only. | Direct USB3 or GigE. Raw 16-bit radiometric frames required for DMRL thermal inertia analysis. | High if via compressed ROS2 stream. Raw radiometric data (16-bit) is required for DMRL. Compressed 8-bit visual stream is insufficient for thermal inertia analysis. |

# **5  PX4 Configuration Baseline for Listener Path**

These parameters must be set on the host autopilot before MicroMind integration. This is a configuration request, not a modification. Any GCS (QGroundControl) or parameter upload tool sets these values.

*Verification: screenshot each parameter value in QGroundControl and log to file. A misconfigured PX4 parameter produces identical symptoms to a MicroMind integration failure. Establish this baseline before any MicroMind code is run.*

| **Parameter** | **Required value** | **Default** | **Purpose** |
| --- | --- | --- | --- |
| MAV_1_RATE | 0 (maximum) | Default stream rates apply | Sets stream channel 1 to maximum. Individual message rates then override. |
| SER_TEL1_BAUD or equivalent | 921600 | 57600 | Serial baud rate to companion computer. Low baud rate throttles 200Hz HIGHRES_IMU. |
| HIGHRES_IMU rate via mavlink stream | 200 Hz | 50 Hz | Force HIGHRES_IMU to 200Hz. Set via: mavlink stream -d /dev/ttyS1 -s HIGHRES_IMU -r 200 |
| GPS_RAW_INT rate | 5 Hz | 1 Hz | Force GPS_RAW_INT for BIM. Set via: mavlink stream -s GPS_RAW_INT -r 5 |
| DISTANCE_SENSOR rate | 10 Hz | Disabled | Enable if RADALT is fitted on PX4. Set via: mavlink stream -s DISTANCE_SENSOR -r 10 |
| EKF2_AID_MASK | Bit 3 (vision) or Bit 0 (GPS) | GPS only | EKF2 must have a position source. Without it, LOCAL_POSITION_NED is invalid and OFFBOARD is accepted but setpoints are ignored. |
| COM_RCL_EXCEPT | Bit 2 set | 0 | Prevents RC loss failsafe from overriding OFFBOARD mode in SITL and HIL. |
| NAV_RCL_ACT | 0 (disabled) | Varies | Disables RC loss failsafe action for external OFFBOARD control. |
| COM_OBL_ACT | 1 (return) or 0 (hold) | 2 (land) | Defines behaviour on OFFBOARD link loss. Land is too aggressive for demo gaps. |
| CBRK_USB_CHK | 197848 | 0 | Disables USB safety check. Required for SITL UDP operation. |
| MAV_SYS_ID | Match pymavlink system_id | 1 | Mismatch causes pymavlink to silently ignore PX4 heartbeats. |

# **6  Upgrade Gates: When to Move from Listener to Direct**

The following gates define the conditions under which each sensor is upgraded from the listener path to direct ownership. Gates are evaluated from measured data, not from assumption.

## **6.1  IMU Upgrade Gate**

| **Condition** | **Measurement method** | **Decision** |
| --- | --- | --- |
| P95 jitter below 0.5ms across a 60s run | Timestamp HIGHRES_IMU receipt against hardware reference clock. Compute P95 latency distribution. | **Listener path confirmed. No action.** |
| P95 jitter 0.5-1.0ms, std deviation < 0.3ms | Same measurement. | **Acceptable with monitoring. Flag for HIL phase review. Log jitter histogram.** |
| P95 jitter above 1.0ms, or std deviation above 0.3ms sustained | Same measurement. | **Plan direct SPI for next HIL phase. Stub RealIMUDriver already exists in Phase 3 plan.** |
| PX4 broadcasts EKF2 attitude instead of raw accel/gyro | Inspect HIGHRES_IMU xacc, yacc, zacc fields. Zero values or smoothed values indicate EKF2 output, not raw sensor. | **Listener path is unusable regardless of jitter. Direct SPI required immediately.** |

## **6.2  RADALT Upgrade Gate**

| **Phase** | **Path** | **Trigger for upgrade** |
| --- | --- | --- |
| Phase 1 (SITL + first HIL) | **Listener via DISTANCE_SENSOR if RADALT on PX4.** | No trigger unless RADALT not on PX4. |
| Pre-terminal and low-altitude HIL | **Evaluate direct UART. Required for terrain-following, flare, and terminal approach precision.** | When low-altitude corridor ingress or terminal alignment precision becomes a test objective. |
| Production / operational | Direct UART/CAN RADALT strongly recommended. | At production integration stage. |

## **6.3  GNSS Upgrade Gate**

| **Phase** | **Path** | **Trigger for upgrade** |
| --- | --- | --- |
| Phase 1-2 (demo and early HIL) | **Listener via GPS_RAW_INT at 5Hz.** | Sufficient for demo spoof scenarios. |
| Production spoof testing | **Direct UART Y-cable for raw UBX/NMEA.** | When GNSS spoof test scenarios require detection of pre-filter signals that pass PX4's integrity checks. |

# **7  Hardware Implications for 150g SWaP Claim**

The 150g SWaP target applies exclusively to: the autonomy compute board, fan, daughter cards, and wire harness connecting to platform power and the PX4 autopilot. Sensors are platform-owned and not counted in this figure.

**MicroMind does NOT provide sensors. Sensors belong to the host UAV platform.**

Under the listener-first architecture, the additional hardware MicroMind requires beyond the compute board is:

| **Item** | **Weight estimate** | **Interface used** | **Notes** |
| --- | --- | --- | --- |
| Wire harness to PX4 UART (MAVLink) | ~5g | UART | One serial cable from compute board to autopilot UART port. |
| Wire harness to power rail | ~8g | 12-28V DC | Platform provides regulated power. Harness only. |
| MIPI CSI-2 ribbon to camera | ~3g | MIPI CSI-2 | Camera is platform-owned. Ribbon connects it to compute board MIPI port. |
| USB3 cable to SDR (if fitted) | ~10g | USB3 | SDR is MicroMind-owned. USB3 Type-C or Type-A. |
| NVMe SSD (M.2 2230) | ~8g | M.2 M-key PCIe Gen4 | Mission logs, DEM tiles, mission envelope storage. |
| Compute board (AIR6N0 class with Orin NX 8GB SOM + fan) | ~130g | All interfaces onboard | Primary compute. Fan required for 0-60C operational range. |

*Total MicroMind integration hardware weight estimate: ~164g (compute + harness + SSD). This is within the 150-180g range discussed for the first demonstrator. Sensors, camera, and any platform-side cabling are not included.*

# **8  Integration Architecture Summary**

Single reference table for OEM integration planning discussions.

| **Sensor** | **Owner** | **Phase 1 interface** | **Long-term interface** | **MicroMind action** |
| --- | --- | --- | --- | --- |
| IMU | Platform (PX4) | **Listen: HIGHRES_IMU 200Hz** | **Direct SPI if jitter ****>**** gate** | Measure jitter on platform. Upgrade if gate triggers. |
| EO camera | Platform (connects to MicroMind) | **Direct: MIPI CSI-2 / USB3** | **Direct: unchanged** | Verify MIPI connector compatibility with platform camera. |
| Barometer | Platform (PX4) | **Listen: SCALED_PRESSURE 10Hz** | **Listen: unchanged** | Verify PX4 broadcasts at 10Hz. No further action. |
| GNSS | Platform (PX4 + MicroMind integrity) | **Listen: GPS_RAW_INT 5Hz** | **Optional direct UART Y-cable** | Force GPS_RAW_INT to 5Hz. Plan Y-cable for production. |
| RADALT | Platform (PX4 if fitted) | **Listen: DISTANCE_SENSOR 10Hz** | **Direct UART for terminal phase** | Verify DISTANCE_SENSOR enabled if RADALT on PX4. |
| Magnetometer | Platform (PX4) | **Listen: HIGHRES_IMU mag fields** | **Listen: unchanged** | No action. PX4 owns. |
| Airspeed | Platform (PX4) | **Listen: VFR_HUD 5Hz** | **Listen: unchanged** | No action. PX4 owns. |
| Optical flow | Platform (PX4) | **Listen: OPTICAL_FLOW_RAD** | **Listen: unchanged** | Subscribe if message available. No action required. |
| SDR / EW | MicroMind | **Direct: USB3 / M.2 PCIe** | **Direct: unchanged** | USB3 or M.2 E-key slot on compute board. No platform dependency. |
| Thermal camera | Platform (Phase 2+) | Not in Phase 1 | **Direct: USB3 / GigE** | Plan USB3 port allocation. Raw 16-bit radiometric required. |

# **9  Relationship to Pre-HIL v1.2 Plan**

This document is an addendum to MicroMind Pre-HIL Software Completion Plan v1.2. It does not modify v1.2. It governs sensor integration decisions within the v1.2 execution phases as follows:

| **Pre-HIL v1.2 phase** | **Sensor integration impact** |
| --- | --- |
| Phase 0 (Architecture lock) | Add sensor integration path decision to Phase 0 documentation. Confirm listener path message list and PX4 configuration baseline (Section 5 of this document) as pre-code validation items. |
| Phase 1 (Driver abstraction) | SimIMUDriver, SimGNSSDriver, SimRADALTDriver remain unchanged. RealIMUDriver stub documents direct SPI interface. PX4 listener drivers (MAVLinkIMUDriver, MAVLinkGNSSDriver) are the Phase 1 real implementations. |
| Phase 1.5 (PX4 minimal link) | Add HIGHRES_IMU jitter measurement to S-PX4-09 validation step. Record P95 jitter. Evaluate against Section 6.1 gate. Add ESTIMATOR_STATUS validation check before OFFBOARD. |
| Phase 2 (Output pipeline + timing) | Latency measurement includes HIGHRES_IMU receipt-to-ESKF timestamp delta as a separate metric. Added to latency instrumentation plan. |
| Phase 3 (Live input stubs) | LiveIMUDriver implements HIGHRES_IMU subscriber. LiveGNSSDriver implements GPS_RAW_INT subscriber. LiveRADALTDriver implements DISTANCE_SENSOR subscriber. All use the MAVLink listener path by default. Direct-path stubs remain as upgrade targets. |
| HIL (post-TASL) | IMU upgrade gate (Section 6.1) evaluated on real hardware. RADALT direct wiring evaluated for terminal phase scenarios. GNSS Y-cable evaluated for production spoof detection. |

**Status: BASELINED. This document governs all sensor integration decisions from Pre-HIL through**
