**MicroMind**

Pre-HIL Programme State Assessment

**Part 12 · Technical Director Review · 29 March 2026 · RESTRICTED**

| **Document** | **Value** |
| --- | --- |
| **Status** | Formal appendix to MicroMind_PreHIL_v1_2.docx. This document constitutes Part 12. |
| **Basis** | TD Update 29 March 2026. MicroMind_PreHIL_v1_2.docx. MicroMind_SIA_v1_0.docx. Live telemetry from S-PX4-01..09. |
| **Author** | Technical Programme Lead |
| **Audience** | Technical Director. Not for OEM distribution in current form. |
| **Constraint** | No new architectural rewrites. No speculative hardware. No re-opening of closed decisions. Candid where needed. |

# **Part 12.1 — What Is Genuinely De-Risked**

The following risks are effectively retired by evidence produced during the 29 March 2026 integration sprint. They should not be re-opened without a specific failure event.

## **12.1.1  Thread Architecture**

The December failure was T-HB sharing a thread with T-SP, causing PX4 to drop OFFBOARD when the navigation loop stalled. The current architecture has T-HB, T-SP, T-MON, T-NAV, and T-LOG all independent. Three bugs found during this sprint were direct manifestations of thread boundary violations — socket contention, shared stop event, and movement threshold. All three were found and fixed during live testing under real conditions, not during design review. That is the correct failure mode. The architecture is now validated under load, not just in theory.

## **12.1.2  PX4 Integration**

S-PX4-01 through S-PX4-09 completed in strict order. The FM-6 compid=0 finding, FM-3 NED Z-sign, FM-2 pre-stream requirement, and OFFBOARD custom_mode=393216 — all confirmed from live telemetry. These were the principal unknowns entering Phase 1.5. They are no longer unknowns.

## **12.1.3  Driver Abstraction**

DriverFactory producing all five driver types from a single config, with sim→real substitution producing a clean error pointing to the hardware interface, is now demonstrated code, not design intent. The RC-4 and RC-10 claims rest on this, and it works.

## **12.1.4  Latency Architecture**

ESKF P95 at 0.085ms and E2E P95 at 0.363ms are not gate passes — they are structural confirmation that there are no blocking calls anywhere in the critical path. A blocking call would show in P95. The 138× gate margin means even a 100× degradation on Jetson Orin still passes. This is a strong result.

## **12.1.5  December Failure Mode**

S-PX4-09 held OFFBOARD for 62 seconds across a five-waypoint trajectory with zero reversions. That is the direct counter-evidence to the December failure. It is closed.

# **Part 12.2 — What Remains High-Risk or Weakly Proven**

## **12.2.1  RC-11 — Control Loop Independence**

RC-11 is the most important unverified criterion and is significantly underweighted in the programme to date. The pass condition requires setpoint generation at 20Hz during both OUTAGE mode and a 5-second MAVLink disconnect, with clean OFFBOARD re-engagement after reconnect. Section 12.4 decomposes RC-11 into five sub-criteria.

The critical unverified behaviour is whether LivePipeline continues producing valid, non-crashing setpoints when vio_mode transitions to OUTAGE and the ESKF receives no VIO updates for 10 seconds. The ESKF divergence behaviour under prolonged zero-update conditions is characterised in S-NEP-06 and S-NEP-07, but those results were established against the SIL baseline — not against the current integration layer with drivers and bridge running simultaneously. There is a non-trivial risk that the ESKF produces NaN or unbounded state during a 10-second VIO blackout, causing `state.p` to yield a non-finite setpoint, causing PX4 to exit OFFBOARD.

RC-11 is not a verification run against existing behaviour. It may expose a new failure mode. This is the highest-risk Phase 3 step and must not be rushed.

## **12.2.2  RC-7 — Timestamp Monotonicity**

The IFM-01 guard exists. What has not been verified is whether injecting a non-monotonic timestamp causes the guard to fire and log the event without disrupting the 200Hz loop. The concern is not whether the guard is present — it is whether the guard's response (log and continue, not crash) is correct under actual live timing conditions. If IFM-01 fires mid-loop and the logging path blocks even briefly, it could introduce a jitter spike invisible in organic P95 results.

## **12.2.3  RC-8 — Logger Non-Blocking at 200Hz**

The BridgeLogger T-LOG queue has been confirmed non-blocking in unit tests. The formal gate requires <0.5% drop rate under 200Hz load for 60 seconds — approximately 12,000 events. The queue is bounded at 10,000 items; at 200Hz it fills in 50 seconds if T-LOG falls behind disk. The test confirms whether T-LOG write throughput is sufficient. On micromind-node01 this will pass. On Jetson Orin with slower storage, this is not guaranteed. See Section 12.2.4 for Jetson-oriented validation note.

## **12.2.4  Hidden Assumption in CP-2 Evidence**

The mark_send instrumentation in the CP-2 run was patched into the bridge at runtime using types.MethodType. This means the latency measurement was taken on a modified T-SP implementation, not the canonical one. The canonical and patched versions are functionally identical, but the CP-2 artefact is technically a measurement of an instrumented variant. This should be closed by integrating mark_send natively into mavlink_bridge._setpoint_loop before CP-3. It is a one-line change but it removes an asterisk from the CP-2 evidence. This is listed as a CP-3 exit requirement.

## **12.2.5  Jetson Orin Validation Note**

RC-8 should be run once under artificial CPU load or reduced process priority to approximate Jetson-class performance. Recommended method: renice the MicroMind process to +10 or run with taskset limiting to 2 cores before the 60-second logger test. If drop rate remains <0.5% under this condition, the logger queue is safe for Jetson deployment. If not, the queue size or T-LOG write strategy requires review before HIL. This does not block CP-3 on micromind-node01, but the result should be recorded as a Jetson risk indicator in the CP-3 readiness report.

# **Part 12.3 — Gazebo Visualisation Assessment**

## **12.3.1  Why Non-Blocking Technically**

Every gate criterion in v1.2 is defined in terms of telemetry — altitude from LOCAL_POSITION_NED, mode from HEARTBEAT, position from NED coordinates. The Gazebo window renders the same physics that produces those values. S-PX4-06 through S-PX4-09 were all validated against telemetry and are correct. The physics simulation is running. The rendering failure does not affect any gate.

## **12.3.2  Why Commercially Critical**

The OEM demonstration definition in v1.2 §Part 9 is built around Gazebo as the visual anchor. If the Gazebo window is blank during a TASL review, the technical lead will spend the first ten minutes explaining why. That explanation — however accurate — undermines confidence in the entire demonstration. A defence OEM evaluating a payload will not independently verify that telemetry numbers correspond to actual vehicle motion. They need to see it. Without Gazebo, the demo is a log-reading exercise.

## **12.3.3  Recommended Fix Path**

The Gazebo GUI is launching (INFO [init] Starting gz gui is visible in SITL output) but rendering nothing. The most likely cause is OGRE2 trying a Vulkan or Metal backend and failing silently on the X11 session. Recommended diagnostic sequence:

•  Attempt 1: Set LIBGL_ALWAYS_SOFTWARE=1 and MESA_GL_VERSION_OVERRIDE=3.3 before launching PX4 SITL. If vehicle appears, the fix is permanent.

•  Attempt 2: Add --render-engine ogre flag to force OGRE1. Set export QT_QPA_PLATFORM=xcb before launch to resolve X11 compositor conflicts.

•  If neither resolves within two attempts: stop and escalate. Do not debug indefinitely.

Gazebo rendering should be fixed in parallel with Phase 3 execution, not after. It does not block any Phase 3 deliverable. The risk of leaving it until after CP-3 is that it becomes the last item before the demo, gets rushed, and introduces instability.

## **12.3.4  Priority Relative to CP-3**

Elevate Gazebo rendering to active work immediately. Assign one focused session. If resolved, it clears a significant commercial risk. If not resolved in two attempts, escalate and document — but do not let the Gazebo investigation consume Phase 3 execution time.

# **Part 12.4 — RC-11 Decomposition**

RC-11 as defined in v1.2 contains five logically distinct verification requirements. These are decomposed here for engineering clarity. All five must pass before RC-11 is considered closed. They should be tested in order — RC-11a through RC-11e — since each depends on the previous.

| **Sub-RC** | **Criterion** | **Pass condition** | **Status** |
| --- | --- | --- | --- |
| **RC-11a** | **Setpoint continuity during OUTAGE** | LivePipeline produces setpoints at 20Hz for the full 10-second VIO blackout. No rate drop. Queue drop count increases (expected and correct) — but setpoint generation does not stop. | **PENDING** |
| **RC-11b** | **No NaN / non-finite setpoints** | All Setpoint.x_m, y_m, z_m values are finite (math.isfinite) throughout the OUTAGE window. ESKF does not diverge to NaN under 10s zero-update condition. | **PENDING** |
| **RC-11c** | **OFFBOARD retained during VIO outage** | PX4 custom_mode remains 393216 throughout the OUTAGE window. T-MON does not detect a mode reversion. Heartbeat and setpoint stream both continue uninterrupted. | **PENDING** |
| **RC-11d** | **OFFBOARD recovered after MAVLink disconnect** | After a 5-second deliberate MAVLink disconnect, OFFBOARD re-engages cleanly when bridge reconnects. Setpoint stream resumes. No crash or exception during the disconnect window. | **PENDING** |
| **RC-11e** | **Mode transition chain logged and visible** | Log contains OUTAGE event with timestamp, RESUMPTION event with timestamp, spike_alert flag, and return to NOMINAL. All five fields present and correctly ordered. VIO mode sequence is auditable. | **PENDING** |

RC-11b is the highest-risk sub-criterion. If the ESKF produces NaN during OUTAGE, it will not be caught by the existing 332 SIL gates because those gates exercise the ESKF in the SIL context, not through the integration layer under live timing conditions. RC-11b requires a specific finite-value assertion added to the instrumented run — it cannot be inferred from RC-11c passing alone.

# **Part 12.5 — CP-3 Exit Definition**

CP-3 is the Pre-HIL readiness gate. It requires all eleven readiness criteria to pass. Based on the analysis in Parts 12.2 through 12.4, the following explicit exit conditions are required before CP-3 can be declared. No partial credit.

| **#** | **Exit condition** | **Evidence required** | **Status** |
| --- | --- | --- | --- |
| **1** | RC-7 pass — IFM-01 fires on injected non-monotonic timestamp, ESKF continues, event ID and timestamp logged | Formal test run committed. Log entry present with specific event fields. | **PENDING** |
| **2** | RC-8 pass — Logger drop rate <0.5% at 200Hz for 60s | Formal test run committed. Drop rate computed and recorded. | **PENDING** |
| **2a** | RC-8 Jetson note — RC-8 repeated under reduced process priority (renice +10 or 2-core taskset) | Result recorded as Jetson risk indicator in CP-3 readiness report. | **PENDING** |
| **3** | RC-11a through RC-11e all pass — see Part 12.4 | All five sub-criteria verified in a single live SITL run with OfflineVIODriver and MAVLink bridge active. | **PENDING** |
| **4** | Gazebo rendering functional — vehicle visible in Gazebo GUI during live SITL run | Screen visible with vehicle entity rendered. SITL terminal confirms EKF aligned. | **PENDING** |
| **5** | mark_send integrated natively into mavlink_bridge._setpoint_loop | CP-2 asterisk removed. Single commit. 332 + integration gates still passing. | **PENDING** |
| **6** | CP-3 readiness report committed to repo | Document records pass evidence for all 11 RC criteria. Tagged commit. Committed to dashboard/. | **PENDING** |

# **Part 12.6 — Phase 3 Discipline Review**

The defined Phase 3 scope is tight and correct: OfflineVIODriver, RC-7, RC-8, RC-11. There is no ambiguity in the exit gate. The following scope boundaries must be enforced.

## **12.6.1  Scope Creep Traps**

The most likely trap is OfflineVIODriver pulling in NEP rework. v1.2 requires OfflineVIODriver to pass S-NEP-04 through S-NEP-09. If RC-11b reveals ESKF divergence during OUTAGE, the temptation will be to patch the ESKF. This must be refused. If RC-11b fails, the correct response is to log it, document it, and escalate to TD. The ESKF is frozen. Any change requires a full S-NEP re-run.

The second trap is beginning run_demo.sh before CP-3 is closed. Once RC-11 passes, the instinct will be to move immediately to Phase 4. Hold the gate. Starting Phase 4 before CP-3 risks carrying unverified criteria into demo rehearsal.

## **12.6.2  What Must NOT Be Added Before CP-3**

•  ROS2 bridge — optional per v1.2, not a gate item

•  Any sensor driver work beyond OfflineVIODriver

•  Terminal overlay / curses panel implementation — Phase 4

•  HTML report wiring to live log — Phase 4

•  Any modification to ESKF, BIM, vio_mode, or enforcement blocks

•  Jetson Orin profiling — post-TASL

•  run_demo.sh — Phase 4

# **Part 12.7 — OEM-Readiness Assessment**

## **12.7.1  What Makes the Programme Look Serious**

The programme has five things most sub-scale defence autonomy efforts do not: a frozen regression-tested SIL baseline (332 gates, untouched across eight months); a formal gate structure with tagged commits and documented evidence; a live PX4 SITL integration with telemetry-backed results; a published latency measurement with margins that are not close calls; and a driver abstraction layer that genuinely demonstrates hardware replaceability without core code changes. The evidence chain from claim to commit hash is complete and independently verifiable.

## **12.7.2  What Still Prevents ****'****Ready to Test Now****'**

Three things. First, the Gazebo window is blank — the first thing any reviewer will see. Second, there is no VIO outage demonstration. The programme's primary claim is resilience under GNSS denial. Until step 6 of the §Part 9 script (outage injection, mode→OUTAGE, drift_envelope growing, recovery) is working and visible, the programme is demonstrating a well-engineered autopilot interface, not an autonomous navigation payload. Third, run_demo.sh does not exist. A single-command reproducible demo is a credibility item, not a polish item.

## **12.7.3  Single Highest-Value Additional Artefact**

A screen recording of the full §Part 9 demo sequence — from ./run_demo.sh through VIO outage injection, recovery, landing, and HTML report generation — with the six mandatory overlays visible simultaneously. Not a production video. A raw recording from micromind-node01. It answers the question every OEM technical lead will have before committing to a partnership: 'Can I watch it work?' A document answers it worked. A recording answers watch it work.

# **Part 12.8 — Demo Readiness Ladder**

The following ladder defines the ordered progression from CP-3 closure to OEM-ready demonstration. Each rung must be achieved before the next begins. No partial credit. No reordering.

| **Rung** | **Label** | **Exit condition** | **Status** |
| --- | --- | --- | --- |
| **DR-1** | **CP-3 passed** | All 11 RC criteria pass. All 6 CP-3 exit conditions met (Part 12.5). Readiness report committed to dashboard/. Tagged commit. | PENDING — Phase 3 |
| **DR-2** | **Gazebo rendering** | Vehicle entity visible in Gazebo GUI during live SITL run. Confirmed in RC-11 test run or dedicated session. No blank window at demo time. | PENDING — parallel to Phase 3 |
| **DR-3** | **run_demo.sh reproducible 3×** | ./run_demo.sh executes complete §Part 9 sequence end-to-end. Vehicle trajectory. VIO outage injected and recovered. HTML report generated. Three consecutive runs, identical outputs. No manual steps. | PENDING — Phase 4 |
| **DR-4** | **Overlays working** | All six mandatory overlays from v1.2 §9.11 visible simultaneously during run_demo.sh: vio_mode, drift_envelope_m, setpoint trajectory vs actual, mode transition log, setpoint rate Hz, PX4 flight mode. | PENDING — Phase 4 |
| **DR-5** | **Screen recording captured** | Raw screen recording from micromind-node01 covering full §Part 9 sequence with overlays visible. Not edited. Reproducible from the same run_demo.sh that passed DR-3. | PENDING — Phase 4 |
| **DR-6** | **OEM-ready demonstration** | DR-1 through DR-5 all achieved. run_demo.sh tested 3× within 48 hours of any OEM meeting. All evidence traceable to tagged commits. BCMP-1 SIL dossier available for step 10. | PENDING — pre-TASL |

DR-2 (Gazebo rendering) is marked parallel to Phase 3. It does not block any Phase 3 gate criterion. However it must be resolved before DR-3 begins, because run_demo.sh without a working Gazebo window fails the §Part 9 sequence at step 1.

# **Part 12.9 — Recommended Next Sequence**

The following order is disciplined, bounded, and respects the gate structure. Nothing in Phase 3 bleeds into Phase 4 scope.

| **Step** | **Item** | **Action** | **Phase** |
| --- | --- | --- | --- |
| **1** | **mark_send native** | Integrate mark_send into mavlink_bridge._setpoint_loop natively. One-line addition. Commit. Verify 332 + integration gates still pass. Removes CP-2 asterisk. | Pre-Phase 3 |
| **2** | **Gazebo rendering fix** | One focused session, max two attempts. LIBGL_ALWAYS_SOFTWARE=1 first, then --render-engine ogre. If unresolved in two attempts, escalate and continue Phase 3 without blocking. | Parallel to Phase 3 |
| **3** | **OfflineVIODriver** | Wrap EuRoC .npy loader with ENU→NED via frame_utils and IFM-01 monotonicity gate. Verify S-NEP-04..S-NEP-09 pass through new driver. Gate for RC-7 and RC-11. | Phase 3 |
| **4** | **RC-7** | Formal timestamp injection test inside 200Hz loop. Confirm IFM-01 fires, event logged, loop continues. Record specific event ID and timestamp. | Phase 3 |
| **5** | **RC-8** | 60s logger drop-rate test. Measure drop rate, confirm <0.5%, confirm ESKF step rate ≥190Hz. Then repeat with renice +10 or 2-core taskset for Jetson note. | Phase 3 |
| **6** | **RC-11a–e** | OUTAGE injection test in live SITL. All five sub-criteria in order. Highest-risk step — allocate full session, do not rush. | Phase 3 |
| **7** | **CP-3 declaration** | All 11 RC criteria pass. All 6 exit conditions met. Readiness report committed. Tag. | Phase 3 close |
| **8** | **run_demo.sh + overlays** | Phase 4 begins only after CP-3 is formally closed. Implement six mandatory overlays. Wire HTML report to live log. Write run_demo.sh. Test 3× consecutively. | Phase 4 |
| **9** | **Screen recording** | Capture raw recording from DR-3 run. No editing. Archive with tagged commit. | Phase 4 |
| **10** | **EO/IR and sensor pipeline** | Do not open before TASL partnership is confirmed. Hardware-blocked. Opening before TASL consumes engineering time with no gate deliverable. | Post-TASL |

End of Part 12 · MicroMind Pre-HIL Programme State Assessment · 29 March 2026
