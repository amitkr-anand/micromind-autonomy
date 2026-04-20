**MicroMind Navigation System**

**S-NEP-04 — VIO Integration Plan**

OpenVINS Integration with MicroMind Fusion Layer

NEP Programme  |  21 March 2026  |  CONFIDENTIAL

| **Programme Context** Stage-1 Complete (Selection) → Stage-2 Complete (Endurance) → S-NEP-04 (Integration Validation) OpenVINS received GO — System-Level Clearance under VIO Selection Standard v1.2 |
| --- |

| **S-NEP-04 Objective: Establish clean, stable, and verifiable integration of OpenVINS into the MicroMind navigation pipeline.** The question this sprint answers: **Can OpenVINS operate correctly as a component inside MicroMind?** This is a system validation phase, not a development phase. OpenVINS is not modified. The MicroMind architecture is not redesigned. New algorithms are not introduced. The candidate set is not expanded. |
| --- |

# **1. Interface Definition**

This section defines the contract between OpenVINS and the MicroMind fusion layer. It must be satisfied before any integration run is valid.

## **1.1 Input Dependencies**

| **Input** | **Source** | **Rate** | **Requirement** |
| --- | --- | --- | --- |
| Stereo images | /cam0/image_raw /cam1/image_raw | 20 Hz (EuRoC) | Synchronised stereo pair. Timestamps must be identical within 0.5 ms. Raw (unrectified) images — OpenVINS performs internal rectification using calibration parameters. |
| IMU | /imu0 | 200 Hz (EuRoC) | Linear acceleration (m/s²) and angular velocity (rad/s) in body frame. Hardware-stamped timestamps. Rate must be ≥4x camera rate for proper VIO initialisation. |
| Camera calibration | estimator_config.yaml | Static | Intrinsics (fu, fv, cu, cv), distortion coefficients, stereo baseline, camera-IMU extrinsics (T_cam_imu), and camera-IMU time offset. EuRoC calibration verified in Stage-2 (cam0: fu=458.654, cam1: fu=457.587, baseline=0.110 m). |

## **1.2 Output Interface**

OpenVINS publishes pose estimates on the /odomimu topic. The MicroMind fusion layer consumes this as a VIO measurement input to the ESKF.

| **Output Field** | **ROS2 Topic** | **Specification** |
| --- | --- | --- |
| Pose + covariance | /odomimu | nav_msgs/Odometry. pose.pose contains position (x,y,z) and orientation (quaternion). pose.covariance is a 6×6 row-major matrix [x,y,z,roll,pitch,yaw]. Validated in Stage-2: covariance non-zero, non-degenerate, NIS consistent. |
| Timestamp | header.stamp | Sensor time in nanoseconds. Monotonically increasing. Must match IMU hardware clock. Maximum observed gap in Stage-2: 25 ms (MH_03), 70 ms (V1_01). Both within EKF propagation tolerance. |
| Frame ID | header.frame_id | Odometry frame ("odom"). child_frame_id is the IMU body frame. These must be registered in the MicroMind TF tree before fusion begins. See Section 1.3. |
| Update rate | /odomimu | 122–126 Hz observed in Stage-2. MicroMind fusion layer input specification: 25–50 Hz (Part Two V7.1 FR). OpenVINS significantly exceeds this. No downsampling required — ESKF will process at arrival rate, up to its propagation limit. |

## **1.3 Coordinate Frame Contract**

This is the highest-risk interface item. Frame misalignment produces silent incorrect fusion — the EKF accepts the measurement without error but the fused state is corrupted.

| **Frame** | **Definition** | **Integration Requirement** |
| --- | --- | --- |
| OpenVINS world frame | Gravity-aligned. X/Y horizontal, Z up. Set at initialisation from first IMU measurement. | Must be registered as a fixed frame in the MicroMind TF tree. OpenVINS world frame origin is arbitrary (set at first pose). This is an odometric, not absolute, reference. |
| IMU body frame | Right-hand, NED or ENU depending on IMU orientation. EuRoC: IMU frame is body frame. | The ESKF state vector is expressed in the body frame. OpenVINS pose is in the odometry frame with child_frame_id=imu. The TF transform odom→imu must be published continuously at the OpenVINS output rate. |
| MicroMind navigation frame | NED (North-East-Down) per Part Two V7.1 convention. | **CRITICAL: OpenVINS uses ENU (East-North-Up) by default. A static rotation transform must be applied before the VIO measurement enters the ESKF. Failure to apply this transform produces heading corruption that the EKF cannot detect.** |
| **ENU→NED transform** | R = [[0,1,0],[1,0,0],[0,0,-1]] | Apply to position, velocity, and the 3×3 position block of the covariance matrix. The rotation block of the 6×6 covariance must also be transformed consistently. This must be verified as part of the integration validation tests (Section 4). |

## **1.4 Covariance Structure for EKF Fusion**

The 6×6 pose covariance matrix from OpenVINS must be consumed correctly by the ESKF. Incorrect indexing is a common integration defect that produces no runtime error but causes incorrect measurement weighting.

pose.covariance layout (row-major, 6×6):

  [0:3, 0:3] → position covariance (x,y,z) in metres²

  [3:6, 3:6] → orientation covariance (roll,pitch,yaw) in rad²

  [0:3, 3:6] → cross-terms (position-orientation)

| **Requirement** | **Detail** |
| --- | --- |
| Position block extraction | ESKF update_vio() must extract indices [0,7,14] (diagonal of position block) as R_pos for measurement noise. Full 3×3 block is preferred for correlated noise. |
| Frame rotation on covariance | When applying ENU→NED rotation R: R_ned = rot @ R_enu_pos @ rot.T for the position block. Must be applied before the measurement enters the ESKF. Not optional. |
| Inflation guard | If NIS p95 exceeds 4.5 during any integration run, apply a conservative inflation factor of 2.0 to the position block diagonal. This is a runtime guard, not a design parameter. |
| **Zero covariance guard** | **If any diagonal element of the position block is ≤0, reject the measurement and log an FM event. This indicates an OpenVINS reset or initialisation failure. Never fuse a zero-covariance measurement.** |

## **1.5 Timestamp Contract**

| **Requirement** | **Specification** |
| --- | --- |
| Monotonicity | All incoming timestamps must be strictly increasing. A non-monotonic timestamp must trigger measurement rejection and an IFM-01 (timestamp fault) log entry. |
| Maximum gap | Gaps >200 ms (observed max: 70 ms in Stage-2) must trigger an IFM-02 (gap fault) event. The ESKF continues propagating on IMU alone during the gap. If gap >500 ms, flag as VIO timeout and downweight subsequent measurement for 1 second. |
| IMU-camera time offset | OpenVINS estimates camera-IMU time offset online. The converged value in Stage-2 runs was approximately -0.0005 s. The ESKF must account for this offset when associating VIO measurements with the IMU propagation timeline. |
| Latency budget | OpenVINS processing latency observed in Stage-2: ~8 ms (update time). Total pipeline latency budget to ESKF input: ≤50 ms. Latency >50 ms must be logged for review. |

# **2. Fusion Readiness Validation**

Stage-2 validated OpenVINS in isolation. S-NEP-04 validates its behaviour inside the fusion pipeline. The properties tested here are system-level — they emerge from the interaction of OpenVINS with the ESKF, not from OpenVINS alone.

| **The validation question is different from Stage-2.** Stage-2 asked: Is OpenVINS accurate enough? **S-NEP-04 asks: Does OpenVINS behave correctly inside the MicroMind state estimator?** |
| --- |

## **2.1 Covariance Integrity Inside Fusion**

| **Property** | **Validation Method** | **Pass Condition** |
| --- | --- | --- |
| Covariance is positive definite at ESKF input | Log eigenvalues of position block per measurement. Assert min eigenvalue > 0. | Zero rejections in baseline replay run. |
| Fused state covariance does not grow unboundedly | Monitor trace of ESKF position covariance over the full replay sequence. | Trace remains bounded and decreases or stabilises after VIO updates. No monotonic growth. |
| NIS at fusion layer consistent with Stage-2 | Compute NIS using ESKF innovation (z - Hx) and innovation covariance S = HPHᵀ + R. Compare to Stage-2 standalone NIS. | Fusion-layer NIS within [0.3, 3.0]. If substantially higher than Stage-2 NIS, indicates frame or covariance transformation error. |
| No measurement rejection storms | Count consecutive measurement rejections. A storm is defined as >5 consecutive rejections. | Zero storms in baseline replay. Isolated rejections (≤2) acceptable. |

## **2.2 Temporal Consistency Under Pipeline Conditions**

| **Property** | **Validation Method** | **Pass Condition** |
| --- | --- | --- |
| IMU propagation and VIO update interleave correctly | Log ESKF update sequence: each entry is either (PROPAGATE, t) or (VIO_UPDATE, t, NIS). Inspect ordering. | No VIO update precedes its corresponding IMU propagation. No timestamp inversion in the update log. |
| VIO measurement latency is within budget | Measure time delta between VIO measurement timestamp and ESKF processing time. | P95 latency ≤50 ms. No measurement arrives after the ESKF has already propagated past its timestamp by more than 100 ms. |
| Update rate is stable under pipeline load | Monitor /odomimu message rate at the fusion node input over the replay sequence. | Rate ≥100 Hz sustained. No gaps >200 ms. Consistent with Stage-2 observations (121–126 Hz). |

## **2.3 Absence of Discontinuities**

| **Property** | **Detection Method** | **Pass Condition** |
| --- | --- | --- |
| No pose jumps at fusion output | Compute frame-to-frame delta of fused position. Flag if δpos/δt > 5 m/s (FM-02 equivalent at fusion layer). | Zero jump events in baseline replay. |
| OpenVINS resets do not corrupt ESKF state | Inject a simulated OpenVINS reset (publish null/zero pose) mid-sequence. Observe ESKF response. | ESKF continues propagating on IMU alone. Fused state does not jump. Recovery on next valid VIO measurement within 2 s. |

# **3. Integration Failure Modes**

These failure modes are distinct from Stage-2 estimator failures. They arise from the interaction of OpenVINS with the fusion pipeline — they cannot be detected by evaluating OpenVINS in isolation.

| **ID** | **Failure Mode** | **Detection Signal** | **System Impact** | **Mitigation** |
| --- | --- | --- | --- | --- |
| **IFM-01** | Timestamp misalignment — VIO timestamps not synchronised with IMU hardware clock | ESKF innovation covariance S grows unboundedly. NIS at fusion layer >> Stage-2 NIS. Fused state diverges from ground truth despite valid VIO output. | Heading drift accumulates silently. Estimator reports high confidence on a corrupted state. Catastrophic if undetected. | Assert timestamp source consistency at startup. Log camera-IMU offset convergence. Reject measurements with │offset│ > 10 ms from converged value. |
| **IFM-02** | Frame inconsistency — ENU/NED rotation not applied before ESKF update | Fused position X/Y axes swapped or inverted. NIS at fusion layer elevated (>3.0) on position components. Trajectory visually wrong but ESKF does not flag an error. | Navigation commands computed in wrong frame. Route tracking fails silently. High consequence in operational use. | Unit test: publish a known VIO pose (e.g. 1.0m North), verify ESKF fused position is 1.0m North. Fail-fast on mismatch at integration startup. |
| **IFM-03** | Covariance misuse — incorrect matrix indexing or no frame rotation on covariance | ESKF overweights or underweights VIO measurements. Fused covariance shrinks to near-zero (overconfidence) or remains large (underconfidence). NIS at fusion layer biased systematically. | Overconfidence: fused state is pulled toward a corrupted measurement. Underconfidence: VIO measurement ignored despite good quality. | Log the extracted R matrix per measurement. Verify diagonal is positive. Monitor trace(R) for consistency with Stage-2 σ_position range (0.09–0.22 m). |
| **IFM-04** | Estimator reset propagation — OpenVINS reinitialization not handled by fusion layer | VIO pose discontinuity (large jump). ESKF receives valid-looking measurement at a position far from current state estimate. Innovation gate triggered or ESKF state corrupted. | Fused state jumps to VIO pose. Navigation commands discontinuous. Autopilot may respond to invalid trajectory command. | Monitor consecutive pose delta at ESKF input. If delta > 1.0 m in one step, classify as reset event, reject measurement, set VIO_TIMEOUT flag. Resume on consistent measurements. |
| **IFM-05** | Latency-induced instability — VIO measurements arrive after ESKF has propagated past their timestamp | ESKF applies a stale correction. Fused state oscillates at VIO update rate. NIS is erratic rather than smoothly distributed. | Fused state is never settled. If latency is periodic, oscillation frequency matches VIO rate. Subtle and difficult to diagnose post-hoc. | Implement a latency budget check: if measurement.header.stamp < ekf.current_time - 100ms, log IFM-05 and apply a reduced trust weight (0.5x) rather than full rejection. |
| **IFM-06** | Trust weighting conflict — ESKF trust score mechanism not updated for VIO source | The existing update_gnss(state, pos, trust_score) interface uses a trust score to scale measurement noise. If VIO is routed through this interface, the BIM-derived GNSS trust score must not be applied to VIO measurements. | VIO measurements incorrectly downweighted during GNSS denial (exactly when VIO is most critical). Or VIO measurements receive incorrect trust when GNSS is available. | Implement a separate update_vio(state, pose, covariance) interface. Do not reuse the GNSS trust score path for VIO. VIO trust is encoded in the covariance matrix itself. |

# **4. Minimal Validation Test Set**

Six tests. Each is surgical — designed to confirm one specific property with the minimum required evidence. No large test matrices.

| **ID** | **Test Name** | **Method** | **Pass Condition** | **IFM Addressed** |
| --- | --- | --- | --- | --- |
| **T-01** | **Frame alignment sanity check** | Publish a single known VIO pose (position: [1.0, 0.0, 0.0] in ENU) to the fusion node. Read the fused ESKF state. | Fused state shows position [0.0, 1.0, 0.0] in NED (East→North axis swap). Any other result fails. | IFM-02 (frame inconsistency). Must pass before any replay test is run. |
| **T-02** | **Covariance extraction and transformation check** | Log the R matrix (extracted measurement noise) per VIO measurement during a 30-second replay. Compute trace(R) statistics. | trace(R) in range [0.01, 0.20] m². All diagonal elements > 0. Consistent with Stage-2 σ_position range 0.09–0.22 m. | IFM-03 (covariance misuse). |
| **T-03** | **Baseline replay — EuRoC MH_01_easy** | Full replay of MH_01_easy through the integrated pipeline. Log: fused trajectory, NIS at fusion layer, ESKF covariance trace, update sequence log. | Fused ATE ≤2x standalone ATE (0.087 m). Fusion-layer NIS within [0.3, 4.0]. No IFM events. Covariance trace bounded. | IFM-01, IFM-03, IFM-05, IFM-06. Primary integration health check. |
| **T-04** | **Reset injection test** | During MH_01_easy replay, inject a simulated reset event at t=60s: publish a zero-covariance pose. Observe ESKF response and recovery. | ESKF rejects the zero-covariance measurement. Fused state continues on IMU propagation. Recovery on next valid measurement within 2 s. No state jump. | IFM-04 (reset propagation). |
| **T-05** | **Timestamp gap injection** | Pause the bag playback for 300 ms mid-sequence. Observe ESKF behaviour during gap and recovery behaviour. | ESKF propagates on IMU alone during gap. IFM-02 gap event logged. On VIO resumption, NIS spike acceptable for ≤5 measurements then returns to baseline. No state divergence. | IFM-01 (timestamp), IFM-05 (latency). |
| **T-06** | **Repeatability check** | Run T-03 (baseline replay) three times on the same sequence. Compute std(ATE) and std(NIS mean) across runs. | std(ATE) < 0.020 m (< 10% of mean ATE). std(NIS mean) < 0.15. Results are reproducible. | General integration stability. Required for exit criteria. |

Tests T-01 and T-02 are pre-flight checks — they must pass before T-03 through T-06 are executed. Failure of T-01 or T-02 means the interface contract is not satisfied and integration cannot proceed.

# **5. Observability Plan**

We must see inside the fusion behaviour, not just inputs and outputs. These are the signals that must be logged and monitored throughout S-NEP-04 execution.

| **Principle: If it is not logged, it cannot be diagnosed. All signals below must be present in every integration run.** **A run without complete observability is not a valid integration run.** |
| --- |

## **5.1 Required Log Signals**

| **#** | **Signal** | **Source** | **Why It Matters** |
| --- | --- | --- | --- |
| O-01 | **ESKF update sequence log** | Fusion node (custom) | Each entry: (type, timestamp, NIS). Type is PROPAGATE, VIO_UPDATE, or REJECTION. Enables post-hoc reconstruction of the full filter timeline. Required for diagnosing IFM-01 and IFM-05. |
| O-02 | **Fusion-layer NIS time series** | Fusion node (computed) | NIS = innovationᵀ S⁻¹ innovation per VIO update. Distinct from Stage-2 standalone NIS. Elevations here indicate frame or covariance transformation issues, not estimator quality. |
| O-03 | **ESKF position covariance trace** | Fusion node (ESKF state) | trace(P_pos) = P[0,0] + P[1,1] + P[2,2] at each propagation step. Must decrease or stabilise after VIO updates. Monotonic growth indicates VIO measurements are not being incorporated correctly. |
| O-04 | **Pose vs fused state divergence** | Post-processed vs ground truth | Fused ATE compared to standalone Stage-2 ATE. Fusion must not degrade estimator quality. If fused ATE > 2x standalone ATE, indicates a fusion integration defect. |
| O-05 | **VIO measurement R matrix log** | Fusion node (per measurement) | Extracted and frame-rotated measurement noise matrix. Confirms covariance transformation is being applied correctly. Log trace(R) and min eigenvalue per measurement. |
| O-06 | **Update rate at fusion node input** | ROS2 topic monitor | Message rate on /odomimu as seen by the fusion subscriber. Must match Stage-2 observations (121–126 Hz). Drops indicate pipeline bottleneck. |
| O-07 | **IFM event log** | Fusion node (event-triggered) | One log entry per IFM event: (IFM-ID, timestamp, details). Must be present even if zero events are observed. Zero events is the expected pass state. |
| O-08 | **Camera-IMU time offset convergence** | /odomimu or OpenVINS debug topic | Online estimated camera-IMU offset from OpenVINS. Must converge to a stable value within 10 s of initialisation. Divergence indicates timing inconsistency (IFM-01 precursor). |

## **5.2 Observability Implementation**

All observability signals must be implemented as a lightweight logging wrapper on the fusion node — not embedded in the ESKF core. This preserves the ESKF boundary constants and avoids any risk of modifying validated code.

| **Implementation Rule** | **Rationale** |
| --- | --- |
| Log wrapper, not ESKF instrumentation | The ESKF boundary constants are frozen. Adding instrumentation inside ekf/ would require a spec update. All logging is in the fusion node wrapper that calls the ESKF. |
| Structured JSON log per run | Each run produces one JSON file containing all O-01 through O-08 signals. File is named by sequence and run number. Required for T-06 repeatability analysis. |
| No realtime display required | Observability is post-hoc. No dashboard or realtime visualisation is required in S-NEP-04. Logs are analysed after each run. |

# **6. Exit Criteria**

S-NEP-04 is complete when all of the following conditions are satisfied simultaneously. Partial satisfaction is not sufficient.

| **All six conditions must be met. No substitution or partial credit.** |
| --- |

| **#** | **Exit Condition** | **Evidence Required** | **Source Test** |
| --- | --- | --- | --- |
| **EC-01** | **Stable fusion without divergence** | Fused ATE ≤2x standalone ATE on MH_01_easy baseline. No IFM events of types IFM-01, IFM-02, or IFM-04 in any run. | T-03, T-04 |
| **EC-02** | **Consistent covariance behaviour** | trace(P_pos) bounded and non-monotonically-increasing throughout baseline replay. Fusion-layer NIS within [0.3, 4.0]. R matrix trace consistent with Stage-2 σ_position. | T-02, T-03 (O-02, O-03, O-05) |
| **EC-03** | **Frame contract satisfied** | T-01 (frame sanity check) passes with zero tolerance. Published known ENU pose produces correct NED fused state. | T-01 |
| **EC-04** | **Reset and gap resilience** | T-04 (reset injection) and T-05 (gap injection) both pass. ESKF continues on IMU during disruption. Recovery within 2 s. | T-04, T-05 |
| **EC-05** | **No integration failure modes active** | Zero IFM events in three consecutive baseline replay runs. IFM event log present and complete in all run logs. | T-03 ×3, T-06 (O-07) |
| **EC-06** | **Reproducible results** | T-06 repeatability: std(ATE) < 0.020 m and std(NIS mean) < 0.15 across three independent runs. Results are deterministic under identical inputs. | T-06 |

## **6.1 On Partial Failures**

| **Scenario** | **Programme Response** |
| --- | --- |
| **T-01 or T-02 fail** | **Interface contract not satisfied. Stop S-NEP-04 execution. Fix frame or covariance transformation. Re-run from T-01. Do not proceed to replay tests.** |
| T-03 passes, T-04 or T-05 fail | Baseline is healthy but fault handling is incomplete. Implement the reset and gap handling rules defined in IFM-04 and IFM-05. Re-run failing tests only. |
| EC-06 fails (non-reproducible) | Indicates a race condition or non-deterministic path in the fusion node. Must be resolved before declaring integration complete. This is a programme blocker. |
| IFM events present but isolated | Single IFM events that do not affect fused state quality are documented as known limitations. They do not block exit if EC-01 through EC-05 are otherwise satisfied. |

# **7. Sprint Structure**

S-NEP-04 is scoped as a single sprint. The sequence below is the recommended execution order — each step gates the next.

| **Step** | **Task** | **Deliverable** | **Gates** |
| --- | --- | --- | --- |
| **04-A** | Interface implementation | update_vio() interface in error_state_ekf.py. Frame rotation utility (ENU→NED). Covariance extraction and validation. Logging wrapper (O-01 through O-08). | T-01, T-02 must pass before 04-B |
| **04-B** | Baseline replay validation | T-03 run (3 times). Full observability log per run. ATE, fusion-layer NIS, covariance trace, IFM event log. | EC-01, EC-02 must pass before 04-C |
| **04-C** | Fault injection tests | T-04 (reset injection), T-05 (gap injection). IFM event logs per test. | EC-04 must pass before 04-D |
| **04-D** | Repeatability and exit | T-06 (3 independent runs). Repeatability statistics. All EC checks confirmed. Sprint closure report. | All 6 EC conditions satisfied |

# **8. Out of Scope**

| **The following are explicitly excluded from S-NEP-04:**   •  Modifications to OpenVINS algorithm or configuration   •  Changes to ESKF boundary constants (frozen per Phase 1 closure)   •  New algorithm introduction or architecture changes   •  Expansion of the VIO candidate set   •  Mission-scale (km-range) validation — this is a post-integration milestone   •  Outdoor or operational environment testing   •  Closed-loop hardware-in-the-loop testing   •  Optimisation of fusion performance beyond what the interface contract requires If any of the above appear necessary during S-NEP-04 execution, they must be deferred to a named follow-on sprint. They do not block S-NEP-04 exit. |
| --- |

# **9. Next Phase After S-NEP-04**

| **S-NEP-04 Outcome** | **Programme Action** |
| --- | --- |
| **All 6 EC conditions met** | Proceed to system-level navigation validation. Scope includes: mission-scale drift validation on longer sequences, two-theatre architecture validation (TRN-primary eastern corridor, VIO-primary western corridor), and BCMP-1 corridor simulation with VIO in the loop. |
| EC-01 through EC-05 met, EC-06 fails | Resolve non-determinism in fusion node before declaring integration complete. EC-06 is a programme blocker. One additional targeted sprint to identify and fix the non-deterministic path. |
| **Critical IFM found (IFM-01, IFM-02, IFM-04)** | Stop execution. Fix the interface defect. Re-run from the relevant step. Do not proceed to system-level validation with an active critical IFM. |

MicroMind Navigation System  |  S-NEP-04 VIO Integration Plan  |  OpenVINS → MicroMind Fusion Layer  |  21 March 2026  |  CONFIDENTIAL
