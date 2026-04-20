**MicroMind**

NEP Navigation Enhancement Programme

**S-NEP-07 (Rev 3 — Baseline)**

**Engineering Implications and Decision Framework**

*Derived from S-NEP-06 Final Analytical Closure Report — Sections 5 and 6*

22 March 2026 — Revision 3 (final tightening — baseline-ready)

This document is a system engineering decision brief, not an analytical report.
All statements derive directly from S-NEP-06 Sections 5 and 6.
No new hypotheses are introduced. No extrapolation beyond validated data.

# **Section 1 — System Reality (Derived Constraints)**

The following are design constraints — not observations. They must be treated as fixed properties of the current system configuration until an explicit architectural change is made and validated.

## **1.1  Velocity is not directly observable**

The ESKF velocity state is updated only through the off-diagonal covariance coupling from position-only VIO corrections. This coupling is weak. In the S-NEP-06 controlled runs, the velocity state remained unconverged throughout a 60-second segment window from zero-initialisation. Velocity estimation error is present and decreasing, but the steady-state magnitude is not quantified.

DESIGN CONSTRAINT: The system does not reliably know its own velocity. Any function that depends on a velocity estimate — including dead-reckoning, trajectory prediction, or drift forecasting — operates on an unreliable input.

## **1.2  Drift behaviour during outage depends on trajectory geometry, not time alone**

Drift during a VIO outage is not a function of outage duration alone. The error trajectory shape (monotonic rise vs loopback) is determined by which interval of the physical trajectory the outage covers. For a fixed outage placement, the initial drift rate is constant regardless of outage duration.

DESIGN CONSTRAINT: A longer outage does not necessarily produce more error at resumption. Error at outage end is determined by the interaction of outage duration and trajectory geometry. This cannot be predicted without knowing the trajectory in advance.

## **1.3  Peak error during outage is bounded by trajectory divergence, not outage duration**

For the tested segment, peak incremental drift saturated at approximately 1230 mm regardless of outage duration beyond 10 seconds. The trajectory's maximum divergence from the outage-start position — reached at approximately 5.6 seconds — acts as a ceiling on peak error for outages that extend beyond that point in curvilinear motion.

DESIGN CONSTRAINT: Peak error during outage has a trajectory-determined upper bound in curvilinear motion. For straight-line motion this bound does not apply — drift grows without saturation. This constraint is specific to the tested conditions and does not generalise to all trajectories.

## **1.4  Geometry affects error trajectory shape, not initial drift rate**

The initial drift rate in the first 2 seconds of outage is identical across all outage durations at a given trajectory placement (std = 29 mm/s across 2s–30s). Trajectory geometry determines whether error peaks-and-falls (loopback) or continues rising. The initial rate is a property of the filter state at outage start.

DESIGN CONSTRAINT: The initial seconds of an outage are the most informative for estimating drift rate. Geometry effects only become visible after several seconds of outage.

## **1.5  Recovery at VIO resumption is rapid but produces a state discontinuity**

In the original S-NEP-06 full-sequence runs, the causal rolling mean error returned within the recovery band (τ = 111 mm) within 5 seconds of VIO resumption for all tested outage durations (2s–20s). This is consistent with Kalman gain near saturation (P >> R after even short outages), which causes the correction at resumption to be applied in one or a few large steps.

DESIGN CONSTRAINT: VIO resumption produces a position state discontinuity, not smooth convergence. The state.p value steps by the magnitude of the accumulated drift in a single inject() call. Downstream systems must treat resumption as a re-alignment event, not a continuous correction.

## **1.6  NIS is a diagnostic signal only under current operating regime**

At ~124 Hz VIO, innovations reflect inter-frame relative motion (~2–66 mm), which is much smaller than the absolute measurement noise (σ ≈ 87 mm). NIS is structurally suppressed and does not satisfy the assumptions of the [0.5, 8.0] evaluation band. NIS responds correctly to rate changes but cannot be used as a consistency gate.

DESIGN CONSTRAINT: NIS cannot be used as a filter health gate in the current configuration. A low NIS reading does not confirm filter consistency; it is the expected steady-state condition.

## **1.7  The current system is partially observable**

This is a design condition, not an insight. It determines what the system can and cannot do under the current architecture, and must be treated as a fixed constraint until an architectural change is validated.

| **State** | **Observability** | **Mechanism** | **Consequence** |
| --- | --- | --- | --- |
| Position (state.p) | Strongly observed | Direct VIO position measurement at ~124 Hz | Reliable during active VIO. Degrades to INS propagation during outage. |
| Velocity (state.v) | Weakly observed | Off-diagonal P coupling from position corrections only. No direct velocity measurement. | Unconverged from zero-init within 60s. Drift rate not predictable from velocity estimate. |
| Accel bias (state.ba) | Indirectly observed | Covariance coupling, slow convergence | Residual bias accumulates as velocity error during outage. Magnitude at outage start not reliably known. |
| Gyro bias (state.bg) | Indirectly observed | Same mechanism as accel bias | Attitude error contributes to spurious acceleration, compounding velocity error. |

**All decisions in this document must be consistent with partial observability. Any engineering action that implicitly assumes reliable velocity or bias knowledge is invalid under the current architecture.**

## **1.8  Velocity must not be used as a primary input to control, planning, or decision logic**

This is a system rule, not a recommendation. It follows directly from the partial observability condition (Section 1.7) and the velocity unobservability constraint (Section 1.1).

**SYSTEM RULE: No control, planning, or decision logic shall rely on state.v (velocity) as a primary input under the current architecture. Velocity may be logged and monitored as a diagnostic signal, but must not be used as an input to position prediction, dead-reckoning, route planning, terminal guidance sequencing, or any function whose correctness depends on an accurate velocity estimate.**

This rule applies until velocity observability is explicitly improved via a validated architectural change (e.g. direct velocity measurement — L-02) and the change is formally incorporated into the programme baseline.

# **Section 2 — Risk Identification**

| **Risk ID** | **Description** | **Trigger Condition** | **Operational Impact** | **Detectability** |
| --- | --- | --- | --- | --- |
| R-01 | Velocity unobservability causes dead-reckoning failure during VIO outage | Any VIO outage > 1s where velocity estimate has not converged | INS drift rate cannot be predicted or bounded without reliable velocity. Dead-reckoning error grows at an unknown rate. | Low — vel_err is not a usable real-time signal. No direct velocity observable. |
| R-02 | Loopback masking produces false confidence in outage severity | Outage over a curvilinear trajectory segment where GT returns toward outage-start position | System appears to have low drift at outage end. Actual accumulated displacement may be large but cancelled by geometry. Error can re-emerge if trajectory subsequently diverges. | Low — loopback is not detectable without GT. Innovation spike at resumption may be small, giving no warning. |
| R-03 | Drift during straight-line high-speed outage is unconstrained | VIO outage during straight-line motion | No trajectory-imposed bound on peak error. Drift grows linearly. Rate depends on velocity error magnitude, which is not reliably known. | Moderate — innovation spike at resumption is proportional to accumulated drift and is observable. |
| R-04 | Large Kalman correction at VIO resumption causes state discontinuity | Any outage with accumulated drift > ~200mm at resumption | A large correction applied in one step may cause downstream consumers of state.p to see a position jump. This is correct filter behaviour but may trigger logic errors in dependent systems. | High — innovation spike magnitude is logged and directly observable. |
| R-05 | NIS cannot detect filter inconsistency | Any condition where filter covariance mismatches actual error | Standard filter health monitoring via NIS is ineffective. Filter divergence or miscalibration may go undetected. | Low — NIS is structurally suppressed in this operating regime (PF-03). |
| R-06 | Velocity-dependent trajectory planning uses unreliable velocity estimate | Any mission phase requiring velocity knowledge | Planned trajectories based on velocity estimate may be inconsistent with actual vehicle motion. | Low — velocity estimation error is not observable without external reference. |

# **Section 3 — Observability Gaps**

## **3.1  Velocity**

Why it exists: VIO provides position measurements only. The ESKF observation matrix H = [I₃ | 0₁₂] has no velocity columns. Velocity is updated only through weak off-diagonal covariance coupling P[3:6, 0:3]. The S-NEP-06 controlled runs confirmed the velocity state had not converged after 60 seconds from zero-initialisation.

What breaks because of it:

- Dead-reckoning accuracy during VIO outage is unknown — drift rate is not predictable

- Drift attribution is impossible — relative contributions of velocity error, attitude error, and bias cannot be separated

- Trajectory-dependent drift prediction requires velocity knowledge that the system does not reliably have

## **3.2  Accelerometer Bias**

Why it exists: Accelerometer bias (state.ba) is estimated through the same indirect covariance coupling mechanism as velocity. In the MH_03 controlled run, ba magnitude grew from near-zero to 0.23 m/s² during the segment, indicating partial but slow convergence. Bias is not directly observable from position measurements.

What breaks because of it:

- Uncompensated bias accumulates as velocity error during outage

- Bias state at outage start is unknown in magnitude — cannot be used for drift prediction

## **3.3  Long-horizon drift accumulation**

Why it exists: S-NEP-05 established bounded drift within 138 seconds with active VIO. S-NEP-06 established drift behaviour during single outages. The behaviour under repeated or extended outages — including how state.v accuracy evolves after large corrections — has not been characterised.

What breaks because of it:

- Multiple sequential outages may allow drift to accumulate across recovery cycles

- Post-resumption velocity state accuracy is not established

# **Section 4 — Engineering Levers**

## **4A — Fusion Architecture Changes**

**L-01: Velocity smoothing aid from position differences (diagnostic only — NOT a state update)**

Mechanism: Compute a velocity proxy from consecutive VIO position corrections (Δp/Δt over a short window). Use as a diagnostic signal to monitor whether the velocity state is tracking the computed proxy. Do NOT inject this as a measurement into the ESKF.

**CLASSIFICATION: Diagnostic smoothing aid only. This quantity is derived from position measurements and is therefore correlated with the ESKF position state. Injecting it as a pseudo-measurement would introduce correlated noise into the estimator, violating the independence assumption. It must NOT be used as an ESKF measurement input under any circumstance.**

Scope: Addition to fusion_logger.py as a logged diagnostic field. Zero impact on filter. Requires no approval for implementation — it is outside the ESKF boundary.

**L-02: Add wheel odometry or barometric altitude as supplementary measurements**

Mechanism: Wheel odometry provides direct velocity in the body frame. Barometric altitude provides direct D-axis position. Either would substantially improve velocity observability and reduce bias accumulation. Both are well-established in fixed-wing and multirotor navigation.

Scope: New sensor interfaces, new update functions in ESKF (analogous to update_vio()). Requires hardware changes. Outside current SIL scope.

**L-03: Velocity-aware covariance initialisation**

Mechanism: Initialise P[3:6, 3:6] (velocity covariance) with a value that correctly represents the actual uncertainty at zero-init, rather than the current value which underestimates it. This does not improve observability but makes the uncertainty representation honest from the start.

Scope: Single change to ESKF initialisation. Requires explicit TD approval as it touches covariance initialisation.

## **4B — Estimation Strategy**

**L-04: Innovation magnitude gating for large corrections at resumption**

Mechanism: When innovation magnitude at VIO resumption exceeds a threshold, spread the correction over multiple update steps rather than applying it in a single inject() call. This converts the position discontinuity (R-04) into a bounded ramp, reducing impact on downstream position consumers.

Scope: Modification to inject() behaviour. Requires design decision on spreading function and threshold. Must not alter the long-run converged position. Requires TD approval.

Limitation: Introduces a transient where state.p is between old and corrected positions. Duration must be bounded and logged.

**L-05: Outage duration tracking with conservative drift uncertainty estimate**

Mechanism: During VIO outage, log elapsed outage time and compute a conservative drift uncertainty estimate based on the initial drift rate observed in the first 2 seconds of outage (C-02 from S-NEP-06). Report this to downstream consumers as a confidence degradation signal, not as a position bound.

CONSERVATISM REQUIREMENT: The drift envelope is an uncertainty estimate, not a guaranteed maximum. The initial drift rate is itself state-dependent and imprecisely known (velocity state is weakly observed). The estimate must err toward over-estimation. Downstream systems must treat it as a signal that position confidence is degraded — not as a precision error bound. Any system logic that treats the envelope as a hard maximum is unsafe.

Scope: Addition to fusion_logger.py. No filter changes. Zero approval required.

**L-06: Adaptive process noise during outage — classified as low priority**

Mechanism: During VIO outage, increase Q to reflect the growing uncertainty in the unobserved velocity state. This causes P to grow faster, producing a larger Kalman gain at resumption.

LIMITATION: This lever addresses a symptom, not the root cause. The correct scaling of Q depends on the unknown velocity error — which is the core observability gap. Until velocity observability is resolved, adaptive Q is heuristic and may produce miscalibrated covariance. Classified as low priority for this reason.

## **4C — System-Level Controls**

**L-07: Outage detection and explicit mode switching**

Mechanism: Track dt_since_last_vio. When it exceeds a threshold, transition to OUTAGE mode. In OUTAGE mode: flag state.p as low-confidence, suppress dependent functions requiring reliable position, begin drift envelope computation (L-05). When VIO resumes, enter RESUMPTION mode before returning to NOMINAL.

Scope: New FSM states (OUTAGE and RESUMPTION) in core/state_machine/. Threshold is a programme decision, not an analytical one. Aligned with the operating doctrine defined in Section 4D below.

**L-08: Innovation spike alerting at VIO resumption**

Mechanism: Log and flag innovation magnitude at the first VIO update after an outage. If innovation exceeds a threshold (e.g. 1m), emit an alert. The mission system decides whether to accept the correction or enter a safety hold. Makes R-04 (state discontinuity) detectable in real time.

Scope: One-line addition to the VIO update path. Innovation magnitude is already computed. Zero filter impact.

**L-09: Trajectory-aware outage classification — deferred**

Requires mission planning integration (MicroMind-OS interface). Not actionable in current programme phase.

# **Section 4D — System Operating Doctrine**

This section defines how the system is intended to behave across operating modes. It is derived from validated S-NEP-06 findings and the partial observability condition established in Section 1.7. It does not introduce new assumptions.

## **4D.1  Operating Modes**

The navigation subsystem operates in one of three modes at all times. Mode transitions are triggered by VIO availability status, not by accuracy metrics.

**NOMINAL — VIO active and providing updates**

What is trusted:  state.p (position) — strongly observed, corrections applied at each VIO update

What is degraded: state.v (velocity) — weakly observed, converging but not reliable for precision use

What is suppressed: nothing — full navigation function available

Authoritative signal: VIO position measurement, innovation_mag for anomaly detection

NIS: diagnostic only (PF-03). Do not use as health gate.

**OUTAGE — VIO absent (dt_since_last_vio ****>**** threshold)**

What is trusted:  state.p is conditionally usable but uncertain. Drift magnitude is not predictable

  in real time — trajectory geometry (loopback vs divergence) is not observable without GT.

  R-02: error masking due to loopback cannot be detected during outage. State.p may appear

  low-error at outage end due to geometry, then diverge if the trajectory subsequently changes.

What is degraded: state.p confidence degrades with outage duration. Rate of degradation is unknown

  (velocity unobservable). Drift envelope (L-05) provides a conservative uncertainty estimate only.

What must be suppressed: all functions requiring accurate position (terminal guidance engagement,

  precision route constraint checking, position-gated decisions). These are inhibited until

  RESUMPTION is confirmed and stabilisation complete.

Authoritative signal: dt_since_last_vio (time-triggered, no trajectory knowledge required).

  Drift envelope as confidence degradation signal — not a guaranteed error bound.

**RESUMPTION — VIO has returned after outage**

State correction is DISCONTINUOUS: state.p steps by the accumulated drift in one inject() call.

This is re-alignment, not smooth convergence.

What is trusted:  state.p after the first successful inject() — position is corrected to current VIO measurement.

What is degraded: state.v and state.ba remain weakly observed. The large position correction does

  not directly update velocity or bias. These states require additional VIO update cycles to reflect

  the post-correction trajectory. The number of cycles required is not quantified by current data.

What must be suppressed: position-dependent functions remain suppressed during a post-resumption

  stabilisation window. The system must not immediately return to NOMINAL — it must complete at

  least one additional VIO update cycle in stabilisation before full trust is restored.

Authoritative signal: innovation_mag at first post-outage VIO update (observable, logged).

  If innovation > threshold, emit alert (L-08) and hold before accepting correction.

Full trust restoration: requires one or more successful VIO update cycles post-correction.

  The exact count is not specified (data does not support a precise number).

## **4D.2  Mode Transition Behaviour**

| **Transition** | **Trigger** | **System Action** | **Evidence Basis** |
| --- | --- | --- | --- |
| NOMINAL → OUTAGE | dt_since_last_vio > threshold (threshold to be set per programme decision) | Flag state.p low-confidence. Begin drift envelope computation. Suppress position-dependent functions. | C-02: Initial drift rate is stable and characterised within 2s of outage. L-05/L-07. |
| OUTAGE → RESUMPTION | First VIO update received after outage | Apply Kalman correction (single inject() step). Log innovation_mag. Emit alert if > threshold. Enter post-resumption stabilisation: suppress position-dependent functions for at least one additional VIO update cycle. State.v and state.ba remain degraded. | C-07: Recovery within 5s for 2s–20s outages in full-sequence runs. R-04: correction is discontinuous. Directive 4: velocity/bias not corrected by position step. |
| RESUMPTION → NOMINAL | Stabilisation complete: at least one post-correction VIO update cycle elapsed, no active alert or alert acknowledged by mission system | Resume full navigation function. Clear outage flag. Continue drift envelope logging for one cycle. Velocity and bias resume slow convergence from post-correction state. | C-07, C-09: suppression accurate, correction valid. Velocity convergence not quantified — continues from Section 1.1 constraint. |

## **4D.3  State Correction is Re-alignment, Not Convergence**

This must be stated explicitly as a doctrine requirement:

**At VIO resumption, state.p is re-aligned to the current VIO measurement in a single correction step. The position change is discontinuous. The magnitude of the step equals the accumulated drift. Downstream systems — routing, terminal guidance, corridor enforcement — must be designed to tolerate this step, or must be suppressed during the resumption stabilisation window.**

Velocity and bias states are NOT corrected by the position re-alignment step. They remain in their pre-correction state and continue the same slow convergence process described in Sections 1.1 and 1.7. The number of VIO update cycles required before velocity and bias reflect the post-correction trajectory is not established by current data and must not be assumed.

# **Section 5 — Decision Matrix**

| **Lever** | **Addresses** | **Complexity** | **Impact** | **Risk Reduction** | **Priority** |
| --- | --- | --- | --- | --- | --- |
| L-01: Velocity diagnostic (NOT state update) | Gap 3.1 (diagnostic only) | Very Low | Low (observability only) | None to filter; aids diagnosis | MEDIUM |
| L-02: Odometry / baro sensor | R-01, R-06, Gaps 3.1, 3.2 | High (hardware) | High | R-01, R-06 | DEFERRED (hardware dependency) |
| L-03: Velocity covariance init | Gap 3.1 (honest uncertainty) | Very Low | Low–Moderate | R-01 minor | MEDIUM |
| L-04: Innovation spreading | R-04 | Moderate | Moderate | R-04 | MEDIUM |
| L-05: Drift envelope logging | R-01, R-02, R-03 | Very Low | Low (observability) | Situational awareness | HIGH (zero filter cost) |
| L-06: Adaptive process noise | Symptom of R-01/R-03 | Moderate–High | Low–Moderate | Partial, heuristic | LOW (symptom, not root cause) |
| L-07: Outage mode switching | R-01, R-03, R-04, Doctrine | Low | High | R-01, R-03, R-04 | HIGH |
| L-08: Innovation spike alerting | R-04 | Very Low | Moderate | R-04 | HIGH (zero filter cost) |
| L-09: Trajectory-aware classification | R-02 | Very High | Moderate | R-02 | DEFERRED |

# **Section 6 — Recommended Decisions (Programme Level)**

Each decision is bounded by validated S-NEP-06 evidence. No extrapolation beyond tested conditions.

**D-01  ****Accept that drift behaviour during VIO outage is trajectory-dependent and cannot be declared safe or unsafe by duration alone.**

*Justified by: C-02, C-03, Constraint 1.2: Drift depends on trajectory geometry, not time alone. No universal outage duration can be declared operationally bounded without trajectory context. Operational thresholds must be defined per mission regime and validated separately against mission-specific trajectory profiles.*

**D-02  ****Treat straight-line motion during VIO outage as a condition with unbounded drift growth. Operational thresholds for straight-line outage must be defined per mission regime and validated separately.**

*Justified by: C-06: MH_03 straight-line outage produced 7.06 m incremental drift at 1.3 m/s under a 10s outage, linear model R²=0.941. Straight-line drift growth is confirmed as unbounded within the tested window. Extrapolation to other speeds or platforms is not justified by current data and is explicitly removed from this decision.*

**D-03  ****Implement L-07 (outage mode switching), L-08 (innovation spike alerting), and the operating doctrine defined in Section 4D as the immediate next engineering actions.**

*Justified by: R-04 (state discontinuity at resumption) is detectable at zero filter cost (L-08). L-07 requires one FSM state addition. Operating doctrine formalises the mode behaviour derived from C-07 and Section 1.5.*

**D-04  ****Implement L-05 (drift envelope logging) in the next integration sprint.**

*Justified by: C-02: Initial drift rate is stable and observable within the first 2 seconds of outage. This enables conservative uncertainty quantification without filter changes. Minimum cost, direct benefit to downstream position consumers.*

**D-05  ****Do not use vel_err_m_s as an operational health signal until velocity state convergence is characterised over a longer window (minimum 5 minutes from initialisation).**

*Justified by: S-NEP-06 Section 5: velocity state is unconverged in 60s windows from zero-init. The vel_err signal in this state reflects initial condition, not estimator quality.*

**D-06  ****Do not use NIS as a filter consistency gate in the current operating regime.**

*Justified by: PF-03: NIS is structurally suppressed at 124 Hz VIO. No NIS-based alerting to be implemented until measurement model or operating regime changes.*

# **Section 7 — What NOT to Do**

## **7.1  Do not declare a safe outage duration**

No outage duration can be declared universally safe. Drift is trajectory-dependent. A 5-second outage over a loopback trajectory segment produces low peak error; the same duration over a straight high-speed segment may produce metres of drift. Any attempt to set a universal safe duration without trajectory context is not supported by S-NEP-06 data.

**NOT JUSTIFIED: Any system parameter or operational rule that specifies a universally safe VIO outage duration without trajectory qualification.**

## **7.2  Do not extrapolate EuRoC results to operational UAV speeds**

The S-NEP-06 controlled experiments were conducted on EuRoC MAV data at speeds of 0.37–1.49 m/s. Extrapolating drift rates or error magnitudes to tactical UAV speeds (10–50 m/s) is not supported by current data. Quantitative drift thresholds for operational deployment must be derived from mission-specific characterisation.

**NOT JUSTIFIED: Any numerical drift specification for operational speeds derived from EuRoC measurements alone.**

## **7.3  Do not inject L-01 velocity proxy into the ESKF as a measurement**

A pseudo-velocity derived from consecutive position differences is correlated with the ESKF position state. Injecting it as an independent measurement would violate the independence assumption of the Kalman filter, introduce biased covariance updates, and potentially destabilise the estimator under poor VIO conditions. L-01 is classified as a diagnostic signal only.

**NOT JUSTIFIED: Injecting any position-derived velocity estimate into the ESKF as a measurement input.**

## **7.4  Do not tune Q or R to reduce NIS**

NIS suppression is a measurement model property (PF-03), not a tuning artefact. Modifying Q or R to bring NIS into the [0.5, 8.0] band would misrepresent uncertainty and could cause rejection of valid measurements.

**NOT JUSTIFIED: Any Q/R modification motivated by NIS compliance.**

## **7.5  Do not classify the 5s→10s transition as a filter regime change**

The transition observed in Block C is a trajectory geometry threshold, not a filter state change. Engineering logic built around this specific threshold would respond to a test-dataset artefact, not a general system property.

**NOT JUSTIFIED: Any engineering decision premised on a filter regime change at 5–10 seconds of outage.**

## **7.6  Do not characterise resumption as smooth convergence**

VIO resumption produces a state discontinuity. The position correction is applied in one step. Downstream systems must be designed to tolerate this or be suppressed during resumption. Describing it as smooth or gradual convergence understates the step magnitude and creates incorrect design assumptions in dependent subsystems.

**NOT JUSTIFIED: Any system design that assumes smooth, gradual convergence at VIO resumption.**

## **7.7  Do not implement adaptive process noise as a priority action**

L-06 addresses a symptom — growing position uncertainty during outage. The root cause is velocity unobservability. Scaling Q heuristically without knowing the true velocity error may produce either over-confident or under-confident covariance during outage. The expected benefit does not justify the architectural risk at this stage.

**NOT JUSTIFIED: Adaptive process noise as a substitute for velocity observability improvement.**

# **Programme Status and Immediate Next Actions**

| **Sprint** | **Status** | **Next Action** |
| --- | --- | --- |
| S-NEP-04 through S-NEP-06 | ✅ CLOSED | No action |
| S-NEP-07 (Rev 2) | ✅ THIS DOCUMENT — under TD review | TD sign-off required before S-NEP-08 scope is finalised |
| S-NEP-08 (proposed) | TO BE SCOPED | Implement L-07, L-08, L-05 — outage mode, alerting, drift envelope |

**Immediate next actions on TD approval (from D-03 and D-04):**

- Add OUTAGE and RESUMPTION states to FSM (L-07) — aligned with Section 4D operating doctrine

- Add innovation spike alerting to VIO update path (L-08) — threshold = 1m, zero filter impact

- Add drift envelope logging to fusion_logger.py (L-05) — log dt_since_vio and estimated drift bound per update

- Remove vel_err_m_s from any operational health display until velocity convergence is characterised (D-05)

*All decisions in this document are grounded in S-NEP-06 validated findings. No decision extrapolates beyond tested conditions. Decisions marked DEFERRED require hardware procurement or mission planning integration outside the current SIL scope.*
