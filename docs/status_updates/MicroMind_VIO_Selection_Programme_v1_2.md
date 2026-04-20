**MicroMind Navigation System**

**VIO Selection Programme**

Two-Stage Evaluation Execution Plan

Version 1.2  |  21 March 2026  |  NEP Programme Discipline  |  CONFIDENTIAL

| Programme objective: Produce a high-confidence, system-level selection of a single VIO algorithm with defensible evidence of estimator quality, endurance behaviour, and integration readiness. The output is one selected algorithm. Nothing less constitutes programme completion. |
| --- |

| **Stage 1** Fast Screening All 4 candidates | **Stage 2** Endurance Validation Top 2 shortlisted only | **Final Decision** Single algorithm selected Mission viability basis |
| --- | --- | --- |

# **1. Programme Objective**

The VIO selection programme answers a single question with a single output:

| Which VIO algorithm should enter the MicroMind navigation fusion layer, and is there sufficient evidence to defend that choice at system level? |
| --- |

The programme evaluates two distinct properties that cannot be assessed on the same sequence:

| **Property A — Estimator Quality (Stage 1)** | **Property B — Operational Endurance (Stage 2)** |
| --- | --- |
| Short-horizon accuracy and stability measured on a controlled indoor sequence. Determines whether an algorithm is worth advancing to endurance testing. | Long-horizon drift behaviour, covariance stability, and tracking robustness across varied sequences. Determines mission viability. This is the primary selection driver. |
| Metrics: ATE, RPE 5s, NIS, FM events, update rate | Metrics: Drift (primary), NIS stability, tracking robustness, failure accumulation |

Stage 1 filters for estimator quality. Stage 2 validates endurance and robustness. Final decision is based on mission viability, not convenience or minimum error.

# **2. Stage 1 — Fast Screening**

Purpose: Eliminate weak estimators quickly. Identify the top 2 candidates for endurance validation. Stage 1 does not select a winner — it eliminates losers.

## **2.1 Candidates**

| **#** | **Algorithm** | **Integration Path** | **Output Topic** | **Status** |
| --- | --- | --- | --- | --- |
| 1 | **OpenVINS** | Docker (Humble) + rosbag2 replay | /odomimu (nav_msgs/Odometry) | Complete — baseline established |
| 2 | **ORB-SLAM3** | Docker (Humble) + rosbag2 replay | /pose (geometry_msgs/PoseStamped) | Pending integration |
| 3 | **VINS-Fusion** | Docker (Humble) + rosbag2 replay | /vins_estimator/odometry (nav_msgs/Odometry) | Pending integration |
| 4 | **Kimera-VIO** | Docker (Humble) + rosbag2 replay | TBD — stereo-IMU calibration required | Optional — programme pacing rule: if >2 sessions to integrate, deferred to next cycle (not eliminated) |

Topic normalisation contract: each algorithm has a dedicated pose parser producing PoseEstimate objects. The evaluation pipeline downstream is identical for all candidates.

Programme pacing rule: if integrating any candidate exceeds 2 sessions, that candidate is deferred to the next evaluation cycle. This is a scheduling decision, not a technical elimination. Deferred candidates remain eligible for reconsideration if performance justifies. Integration difficulty must not bias final technical selection — it only affects evaluation scheduling.

## **2.2 Dataset and Conditions**

- Dataset: EuRoC MH_01_easy — fixed for all candidates

- Identical Docker environment (osrf/ros:humble-desktop)

- Identical rosbag2 file, identical estimator config per algorithm

- Single run per candidate — no variance testing at Stage 1

- A candidate that cannot run on this dataset is disqualified on integration grounds

## **2.3 Stage 1 Metrics**

Exactly five metrics. No additions. Tracking loss is recorded but does not affect shortlisting — it is a proxy metric at this stage. Drift is recorded as a proxy only; the sequence is too short for a valid measurement.

| **Metric** | **GO Threshold** | **Eliminates if** | **Scoring** | **Notes** |
| --- | --- | --- | --- | --- |
| ATE RMSE (m) | < 1.0 m | > 5.0 m | +1 if < 1.0 m | Full sequence incl. init transient |
| RPE 5s mean (m) | < 0.3 m | > 1.0 m | +1 if < 0.3 m | 5s sliding window |
| NIS mean | 0.3 – 3.0 | < 0.1 or > 5.0 | +1 if consistent | Aligned error vs cov diagonal |
| FM events (FM-02 + FM-03) | 0 | Any event | +1 if zero events | Hard elimination — no exceptions |
| Update rate (Hz) | >= 20 Hz | < 20 Hz | Not scored | Hard elimination gate only |

## **2.4 Shortlisting Rules**

Applied in order. First matching rule wins.

| **Rule** | **Condition** | **Action** |
| --- | --- | --- |
| 1 | Any FM-02 or FM-03 event on any sequence | **Eliminated — no exceptions regardless of other scores** |
| 2 | Update rate < 20 Hz | **Eliminated — hard integration gate** |
| 3 | Composite score < 2 (out of 4) | **Eliminated — insufficient estimator quality** |
| 4 | Remaining candidates ranked by score, then ATE tiebreaker | Top 2 proceed to Stage 2. Do not lower the bar if fewer than 2 survive. |

## **2.5 Stage 1 Comparison Table**

Populated after each run. One row per candidate.

| **Candidate** | **ATE RMSE (m)** | **RPE 5s (m)** | **NIS mean** | **NIS class** | **FM events** | **Rate (Hz)** | **Eliminated** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **OpenVINS** | 0.087 | 0.068 | 0.426 | CONSISTENT | 0 | 126 | **NO** |
| **ORB-SLAM3** | — | — | — | — | — | — | Pending |
| **VINS-Fusion** | — | — | — | — | — | — | Pending |
| **Kimera-VIO** | — | — | — | — | — | — | Optional |

# **3. Stage 2 — Endurance Validation**

| Stage 2 is an endurance validation stage. Drift behaviour is the primary decision driver. An algorithm that minimises short-horizon error but accumulates drift is not viable for GNSS-denied corridor operations. Applied only to Stage 1 shortlist — maximum 2 candidates. |
| --- |

## **3.1 Datasets**

| **Sequence** | **Length** | **Duration** | **Difficulty** | **Primary Purpose** |
| --- | --- | --- | --- | --- |
| **EuRoC MH_03_medium** | ~130 m | ~132 s | Medium | PRIMARY — drift proxy with longer baseline, dynamic motion stress |
| **EuRoC V1_01_easy** | ~58 m | ~144 s | Easy | Vicon room — different texture profile, lower feature density, tracking stress |
| **EuRoC MH_05_difficult** | ~100 m | ~111 s | Difficult | CONDITIONAL — added only if both candidates survive MH_03 + V1_01 |

Note on drift measurement: even MH_03 at 130 m does not provide a 1 km waypoint. Drift is reported as ATE/sequence_length (proxy) with confidence increasing with sequence length. Cross-sequence consistency of the drift proxy is a stronger signal than any single value.

| Drift proxy interpretation rule: Proxy values derived from short sequences (≤130 m) are not decision-sufficient in isolation. They must be interpreted jointly with drift stability (linearity, R²) and cross-sequence consistency (std of proxy across sequences). A single-sequence drift proxy must never drive selection. Consistency of drift behaviour across sequences is the primary signal at this stage. |
| --- |

## **3.2 Stage 2 Metrics**

The five Stage 1 metrics are retained. Three additional metrics become valid in Stage 2.

| **Pri.** | **Metric** | **Measurement** | **Decision Role** |
| --- | --- | --- | --- |
| **1** | **Drift proxy (m/km)** | ATE_mean / sequence_length_km per sequence | **PRIMARY driver. Compared across candidates and sequences. Must show consistent behaviour, not just low values.** |
| **1** | **Drift stability (linearity, cross-sequence consistency)** | Linear fit R² on drift curve; std(drift proxy) across sequences | **PRIMARY driver. A single-sequence drift proxy value must never drive selection alone. Non-linear drift (R² ****<**** 0.5) or high cross-sequence variance flags unpredictable behaviour and overrides a favourable single-sequence result.** |
| 2 | NIS stability over time | NIS time series per sequence; mean and std across sequences | Secondary. A consistent NIS range across all sequences confirms the estimator is honest over varied conditions. |
| 3 | Tracking robustness (proxy) | Covariance-spike proxy per sequence; compared across candidates | Secondary. Re-enters decision at Stage 2. An estimator with stable drift but collapsing tracking is not operationally viable. |
| 4 | FM event accumulation | Total FM-02 + FM-03 events across all sequences | Any FM event on any sequence is a hard elimination. Zero across all sequences is required. |
| 5 | ATE RMSE (tie-breaker only) | Per sequence; mean across sequences | Used as tie-breaker only when all higher-priority metrics are equal. Does not override endurance properties. |

## **3.3 Run Protocol**

- Three runs per sequence per candidate — identical config, record std(ATE)

- A candidate with std(ATE) > 20% of mean(ATE) on any sequence is flagged as non-deterministic

- Non-determinism does not auto-eliminate but is included in the final selection evidence

- Acquire MH_03 and V1_01 rosbag2 files in parallel with Stage 1 runs — do not wait

# **4. Final Selection Logic**

| Selection is driven by endurance stability, not minimum error. The priority hierarchy defines the order of evaluation, not a mechanical override. A marginal advantage in a higher-priority metric does not automatically defeat a substantial degradation in a lower-priority metric. Magnitude matters. |
| --- |

## **4.1 Selection Priority Hierarchy**

| **Pri.** | **Property** | **What it measures** | **Override rule** |
| --- | --- | --- | --- |
| **1** | **Drift behaviour** | Mission viability — how far can the UAV fly without TRN correction? | Primary driver. But a marginal drift advantage does not override a substantial tracking failure. |
| 2 | NIS stability | Covariance honesty — is the estimator safe to fuse? | A candidate whose covariance is inconsistent across sequences cannot be safely fused regardless of drift performance. |
| 3 | FM absence | Stability — does the estimator fail catastrophically? | Any FM event on any sequence is a hard elimination. No exceptions. |
| 4 | Tracking robustness | Operational continuity — does the estimator hold tracking under varied conditions? | Low ATE with unstable tracking is not usable in real operations. A substantial tracking advantage can outweigh a marginal drift difference. |
| 5 | ATE RMSE | Short-horizon accuracy | Tie-breaker only. Never overrides endurance properties. |

## **4.2 Integration Readiness as a Selection Dimension**

The sandbox is not only evaluating algorithm performance. It is exposing integration complexity, topic compatibility, runtime behaviour, and resource usage patterns. These are valid selection inputs.

| **Dimension** | **Role in Selection** |
| --- | --- |
| Topic compatibility | A candidate publishing pose in a non-standard format requires a custom parser. Acceptable — but adds programme risk. |
| ROS environment stability | A candidate that requires build patches or non-standard dependencies carries ongoing maintenance cost. Noted in evidence. |
| Resource usage | CPU, memory, and GPU usage recorded. High resource demand at 1x playback speed implies edge deployment risk. |
| Integration cost (pacing rule) | If integration exceeds 2 sessions, candidate is deferred to the next evaluation cycle — not eliminated. Deferred candidates remain eligible. Integration difficulty is a secondary discriminator that breaks ties between technically equal candidates. It must not bias technical selection. |

Integration difficulty must not override performance. It is a secondary discriminator. Two candidates with equal endurance performance — the simpler integration wins.

## **4.3 Covariance Requirement — Hard Constraint for Fusion**

| The MicroMind fusion layer requires: measurement = state + covariance. Without usable covariance, NIS cannot be computed and safe EKF fusion is not possible. This is a non-negotiable system requirement, not an evaluation preference. |
| --- |

Any candidate that does not provide a usable covariance estimate is classified as NOT READY for fusion and is CONDITIONALLY EXCLUDED from final selection, unless one of the following is satisfied:

- Covariance is sourced from a verified alternative topic on the same algorithm (e.g. a separate pose-with-covariance publisher)

- Covariance is derived through a validated method (e.g. windowed position variance from the pose stream) with documented assumptions

ORB-SLAM3 specific: geometry_msgs/PoseStamped carries no covariance. Before ORB-SLAM3 is accepted as a Stage 1 candidate, the team must confirm that a covariance-bearing topic is available and parseable. If no covariance source is confirmed, ORB-SLAM3 is excluded from the current selection cycle on fusion incompatibility grounds — not integration difficulty grounds. It remains eligible for reconsideration if a covariance source is identified.

# **5. Decision Resolution Framework — Stage 2**

Stage 2 may produce conflicting results across sequences. This framework ensures the final selection remains deterministic and defensible regardless of conflict type.

## **5.1 Sequence Weighting**

Not all sequences carry equal weight in the decision. MH_03 is the primary endurance sequence — its result on drift and FM accumulation carries higher weight than V1_01. V1_01 primarily stresses tracking robustness under lower feature density.

| **Sequence** | **Weight** | **Primary property assessed** |
| --- | --- | --- |
| **MH_03_medium** | **PRIMARY** | Drift proxy with longer baseline, dynamic motion, endurance |
| **V1_01_easy** | SECONDARY | Tracking robustness, low feature density, environment generalisation |
| **MH_05_difficult** | CONDITIONAL | High-agility stress — used only if both candidates survive MH_03 + V1_01 |

## **5.2 Conflict Classification**

| **Type** | **Definition** | **Resolution** | **Escalation** |
| --- | --- | --- | --- |
| **Type 1** | Soft conflict: both candidates pass all gates across all sequences but metrics favour different candidates on different sequences | Apply priority hierarchy with sequence weighting. Magnitude of differences determines outcome — a marginal drift advantage does not override a substantial tracking degradation. | No escalation required — framework resolves |
| **Type 2** | Hard conflict: one candidate fails a hard gate on one sequence (FM event or covariance collapse) but not another | Candidate that failed a hard gate on any sequence is eliminated. Hard gates are non-negotiable regardless of performance elsewhere. | No escalation required — rule resolves |
| **Type 3** | Programme conflict: both candidates fail different hard gates on different sequences, or drift behaviour cannot be resolved by hierarchy + weighting | Cannot be resolved by the evaluation pipeline alone. Structured escalation package submitted to programme director. | **Escalation required — see 5.3** |

## **5.3 Type 3 Escalation Protocol**

Escalation is not an open-ended conversation. It is a structured submission with four mandatory elements.

| **#** | **Element** | **Required Content** |
| --- | --- | --- |
| **1** | **Conflict Statement** | Which metric on which sequence produces the conflict — one sentence per candidate. No analysis, no narrative. |
| **2** | **Consequence Table** | What each candidate's weakness means operationally — translated into mission terms, not metric values. Example: 'Candidate A drift of 18 m/km limits GNSS-denied legs to 2.8 km at 50m TRN accuracy.' |
| **3** | **Team Recommendation** | The evaluation team's selection recommendation with rationale, even if evidence is not conclusive. The programme does not escalate without a recommendation. |
| **4** | **Resolution Options** | Two or three concrete options for the director, each with a defined action and consequence. Options may include: accept a candidate with a documented operating envelope, commission an additional sequence run, or defer pending Phase-2 outdoor data. |

The director's decision against this package is recorded as the programme decision with the escalation package as its evidence base. Both the decision and the package are committed to the programme archive.

## **5.4 Determinism Guarantee**

Every conflict either resolves through the priority hierarchy and sequence weighting (Types 1 and 2), or escalates with a structured package and a team recommendation (Type 3). There is no path through Stage 2 that produces an unresolved ambiguity. Both resolved selection and structured escalation are defensible programme outputs.

# **6. Programme Decision Tree**

| **STAGE 1: Run all candidates on EuRoC MH_01_easy** Evaluate: ATE, RPE 5s, NIS, FM events, update rate Eliminate: FM events → rate < 20Hz → score < 2 Shortlist: top 2 by composite score, ATE tiebreaker **↓** **STAGE 2: Run shortlist on MH_03_medium + V1_01_easy (+ MH_05 if both survive)** Evaluate: drift (primary), NIS stability, tracking robustness, FM accumulation, ATE (tiebreaker) Conflict: Type 1/2 → resolve via hierarchy + weighting │ Type 3 → structured escalation **↓** **FINAL DECISION: One selected algorithm** Evidence package: Stage 1 + Stage 2 results, conflict resolution record, integration readiness assessment **Output: Algorithm name + documented operating envelope + Phase-2 integration clearance** |
| --- |

# **7. Immediate Next Actions**

| **#** | **Action** | **Detail** | **Priority** |
| --- | --- | --- | --- |
| 1 | Acquire Stage 2 rosbag2 files | Download MH_03_medium and V1_01_easy from ETH Research Collection. Do not wait for Stage 1 completion. | **Immediate** |
| 2 | Assess VINS-Fusion integration cost | Publishes nav_msgs/Odometry — same format as OpenVINS. Parser reuse likely. Lowest integration cost after OpenVINS. | High |
| 3 | Assess ORB-SLAM3 integration cost | **Publishes geometry_msgs/PoseStamped — no covariance in this message type. SELECTION CONSTRAINT: must confirm covariance is available on an alternative topic before Stage 1 run is valid.** | High |
| 4 | Assess Kimera-VIO integration cost | Requires additional stereo-IMU calibration input. If >2 sessions to integrate, defer to Phase-2. | Medium |
| 5 | Commit Stage 1 OpenVINS results to programme archive | benchmark/ directory with metrics.json, plots, and benchmark_report.md. Commit hash recorded. | **Immediate** |

MicroMind Navigation System — VIO Selection Programme v1.2  |  NEP Evaluation Pipeline  |  21 March 2026  |  CONFIDENTIAL
