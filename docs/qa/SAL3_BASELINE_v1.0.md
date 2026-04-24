SAL-3 BASELINE v1.0
Authority: Deputy 1 (Architect Lead) | Date: 24 April 2026 | Status: FROZEN — no further design changes

PART A — BASELINE FREEZE
Activation logic
SAL-3 activates when EITHER condition is met:

Elapsed distance since last ACCEPT correction (any source) exceeds SAL3_ACTIVATION_WINDOW_KM
EKF position covariance trace exceeds SAL3_COV_ACTIVATION_THRESHOLD

AND SAL-2 output is SUPPRESS in the current cycle.
Deactivation: SAL-2 returns ACCEPT or CAUTION; or vehicle exits GNSS-denied phase. SAL-3 may be active while producing no valid output. That is correct behaviour.
Correction validity gates (sequential — all must pass)

Gate 1 Temporal consistency: correction vector must be directionally consistent with the previous valid candidate within SAL3_CONSISTENCY_WINDOW_KM. First observation after activation is a candidate only. Injection requires two directionally consistent observations separated by meaningful displacement.
Gate 2 Geometric validity: minimum two landmarks, angular separation exceeding minimum bearing separation. Single-landmark output unconditionally suppressed.
Gate 3 Residual character: correction vector must not directionally oppose the established INS drift vector. Magnitude is not the criterion — direction relative to drift is.

Any gate failure → correction suppressed → SAL3_GATE_REJECTED logged with gate identifier.
Source selection
SAL-2 has unconditional priority. SAL-2 injects whenever output is ACCEPT or CAUTION. SAL-3 injects only when SAL-2 is SUPPRESS. Simultaneous injection never occurs. On same-cycle conflict, SAL-2 injects and SAL-3 logs SAL3_PREEMPTED.
Trust model
SAL-3 corrections carry anisotropic uncertainty: tighter cross-track (bearing-constrained), looser along-track (range uncertainty). SAL-3 uncertainty is always larger in absolute terms than SAL-2 at equivalent confidence band. SAL-3 corrections must never dominate EKF state — bounded influence is the design intent. Co-directional map staleness error is an accepted limitation: the uncertainty ellipse size is the only real-time protection.
Safety invariants — see Part B.

PART B — NON-NEGOTIABLE INVARIANTS

A correction failing any gate must never reach the EKF, regardless of drift magnitude or activation state.
SAL-2 and SAL-3 must never inject in the same navigation cycle.
A single-landmark observation must never produce an injected correction.
SAL-3 must never inject when SAL-2 is producing ACCEPT or CAUTION output.
A correction vector directionally opposing the established INS drift vector must never be injected without temporal consistency confirmation from a second independent observation.


PART C — SIMULATION VALIDATION PLAN
ScenarioCondition InjectedExpected BehaviourFeatureless terrainNo landmarks in map within search radiusSAL-3 active, Gate 2 fails, INS-only, NAV_DRIFT_UNBOUND at window thresholdRepetitive terrainLandmark candidate count exceeds ambiguity limitGate 2 fails, SAL3_GATE_REJECTED logged, INS-onlyMap staleness (biased)Landmark positions offset by fixed vector aligned with driftGate 3 passes (co-directional), correction injected with large uncertainty ellipse — known limitation confirmed observable in logEO degradationFeature contrast below detection floorGate 1 or Gate 2 fails, INS-only, silentSAL-2 recovery transitionSAL-3 active and valid; SAL-2 transitions SUPPRESS→CAUTION mid-cycleSAL-2 injects, SAL-3 logs SAL3_PREEMPTED, no dual injectionHigh drift + late correctionLong SAL-2 absence, first SAL-3 observation arrivesFirst observation: candidate only. Second consistent observation: injection permitted. One-cycle delay confirmed correct

PART D — OBSERVABILITY / LOGGING REQUIREMENTS

Activation and deactivation events with trigger reason (distance condition or covariance condition)
Gate rejection events: gate identifier, cycle timestamp, candidate correction vector
Correction candidate vs accepted correction count per corridor segment
SAL3_PREEMPTED events with SAL-2 confidence band at preemption
Source selection decision per cycle: which source injected, which suppressed
EKF position covariance trace at activation trigger
NAV_DRIFT_UNBOUND events with elapsed distance and estimated drift
Post-injection EKF residual (pre- and post-correction state difference) for accepted SAL-3 corrections


PART E — IMPLEMENTATION CLARIFICATIONS

Directional consistency definition: requires a quantitative angular bound for Gate 1. Must be specified before coding. Candidate principle: correction vector bearing must fall within the angular cone subtended by the prior drift vector. Exact cone angle is a parameter, not a design question.
EKF covariance trace access: NavigationManager must expose the current position covariance trace to the SAL-3 activation check each cycle. Confirm this field is readable without modifying the ESKF frozen file.
Drift vector availability: Gate 3 requires an established INS drift vector. Source must be confirmed — either NavigationManager running drift estimate or EKF velocity-integrated delta. Must be the same source used consistently across all Gate 3 evaluations.
SAL3_CONSISTENCY_WINDOW_KM scope: must be defined as a named constant before implementation, not inferred from activation window. These are independent parameters.


PART F — FINAL CONFIRMATION
SAL-3 v1.0 is internally consistent. All prior contradictions (activation timing, residual gate logical trap, SAL-2 interaction instability, single-frame validity) have been explicitly resolved. The co-directional map staleness error case is an accepted limitation — bounded EKF influence is the documented mitigation. The design is non-blocking, integrates with existing NavigationManager schema, and produces no hidden state beyond the temporal consistency candidate buffer.
SAL-3 v1.0 is ready for simulation-based validation.
