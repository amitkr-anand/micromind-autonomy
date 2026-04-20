**MicroMind VIO Selection Programme — Stage-1 Evaluation Record**

Governed by VIO Selection Standard v1.2  |  21 March 2026  |  CONFIDENTIAL

**Candidate: RTAB-Map (stereo_odometry node)  |  Sequence: EuRoC MH_01_easy**

| **NO-GO — Stage-1 Disqualification** Performance Failure + Architectural Incompatibility with v1.2 Covariance Requirement |
| --- |

# **1. Disqualification Reasons**

Two independent reasons, both sufficient for disqualification. Both are recorded.

## **1A. Performance Failure**

| **RPE 5s = 1.58 m — exceeds NO-GO threshold of 1.0 m** ATE RMSE = 3.13 m — exceeds GO threshold of 1.0 m (CONDITIONAL range) Drift proxy = 37.3 m/km — exceeds GO threshold of 10 m/km (CONDITIONAL range) **Conclusion: fails Stage-1 estimator quality criteria independent of covariance issue** |
| --- |

## **1B. Architectural Incompatibility — Covariance Type Mismatch (Primary Finding)**

| **RTAB-Map stereo_odometry publishes: incremental odometry covariance (frame-to-frame uncertainty)** **MicroMind fusion layer requires: global state covariance (absolute pose uncertainty)** These are fundamentally different quantities. They cannot be compared or substituted. Consequence 1: NIS cannot be computed meaningfully (error ÷ wrong covariance type) Consequence 2: Covariance is not usable for EKF fusion Consequence 3: Fusion compatibility cannot be established or assessed **Verdict: fails v1.2 covariance requirement (hard constraint)** |
| --- |

# **2. Programme Finding — Covariance Type Requirement for Fusion**

| **FINDING PF-01: Not all odometry covariance is equivalent for MicroMind fusion purposes.** **For MicroMind EKF integration:**   •  Covariance must represent global state uncertainty (absolute pose error bounds)   •  Incremental odometry covariance (frame-to-frame) is not acceptable   •  NIS computation requires covariance semantically aligned with absolute pose error **This finding becomes a selection filter for all future candidates under v1.2.** Any candidate that publishes only incremental covariance must be classified as NOT READY for fusion unless a validated conversion method is documented. |
| --- |

This finding is appended to VIO Selection Standard v1.2 as a programme-level clarification. It does not alter existing rules — it makes an implicit requirement explicit.

# **3. NIS Status**

| **Item** | **Record** |
| --- | --- |
| NIS computed value | **551,179 (observed — not a valid measurement)** |
| NIS validity | **NOT VALID** |
| Reason for invalidity | Covariance semantic mismatch — incremental covariance divided by absolute position error. Not numerical instability. Not a missing metric. |
| Action taken | NIS excluded from decision. Structural incompatibility recorded as Programme Finding PF-01. NIS requirement is NOT downgraded. |
| **Decision basis** | **Verdict driven by performance failure (RPE NO-GO) + covariance incompatibility. NIS invalidity is a corroborating finding, not the sole basis.** |

# **4. Metric Record (Reference Only)**

Values recorded for programme completeness. Final verdict is not driven by these values in isolation — it is driven by RPE failure and covariance incompatibility.

| **Metric** | **Value** | **Threshold** | **Interpretation** |
| --- | --- | --- | --- |
| ATE RMSE (m) | 3.1306 | GO < 1.0 m | CONDITIONAL — weak global accuracy; 36x worse than OpenVINS baseline |
| Drift proxy (m/km) | 37.28 | GO < 10 m/km | CONDITIONAL — limits GNSS-denied leg to 1.3 km at 50m TRN accuracy |
| **RPE 5s mean (m)** | **1.5788** | NO-GO > 1.0 m | **NO-GO — local accuracy insufficient for 5s dead-reckoning** |
| Tracking loss (proxy) | 2.7% | GO < 5% | GO — tracking continuity acceptable |
| Update rate (Hz) | 20.0 | GO ≥ 20 Hz | GO — at minimum acceptable threshold |
| **NIS** | **NOT VALID** | CONSISTENT 0.3–3.0 | **Structural covariance type mismatch — see Section 3. Not computed.** |
| FM-02 divergence events | 0 | 0 required | No pose divergence observed |
| **Fusion compatibility** | **NOT READY** | All checks must pass | **Covariance type mismatch — see Programme Finding PF-01** |

# **5. Exclusion from Current Selection Cycle**

| **RTAB-Map is excluded from the current VIO selection cycle.** Reason A: Not a VIO-equivalent estimator under v1.2 definition.   RTAB-Map is a full SLAM system. It was admitted as a pragmatic comparator only.   Its odometry-mode performance does not meet Stage-1 estimator quality criteria. Reason B: Does not meet fusion-layer covariance contract.   Incremental covariance is architecturally incompatible with MicroMind EKF.   This is a structural constraint, not a configuration limitation. |
| --- |

# **6. Reclassification and Future Revisit**

| **Classification** | **Detail** |
| --- | --- |
| **Status** | **DEFERRED — Architectural Mismatch (SLAM vs VIO)** |
| This is not | A rejection of RTAB-Map as a system. It is a recognition that RTAB-Map in its current stereo_odometry configuration does not satisfy v1.2 requirements for MicroMind fusion. |

Revisit ONLY if one of the following conditions is achieved:

|  | **Revisit Condition** |
| --- | --- |
| **R1** | A configuration or mode is identified that provides global pose covariance consistent with EKF fusion expectations (absolute uncertainty, not incremental). |
| **R2** | A validated method exists to convert RTAB-Map incremental covariance into global state uncertainty, with documented assumptions and validation evidence. |
| **R3** | Programme scope expands to include SLAM-based navigation architectures as a distinct evaluation track, with separate covariance and fusion requirements. |

# **7. Programme Integrity Statement**

| **The following programme standards are NOT modified by this evaluation outcome:**   •  Covariance requirement for NIS computation — retained   •  NIS as a mandatory decision metric — retained   •  Covariance hard constraint for fusion compatibility — retained   •  v1.2 metric hierarchy and decision logic — retained unchanged RTAB-Map is not treated as a partial comparator. Its metrics are recorded for reference only. **The covariance requirement is not downgraded to accommodate this or any future candidate.** |
| --- |

MicroMind VIO Selection Programme  |  RTAB-Map Stage-1 Evaluation Record  |  Governed by VIO Selection Standard v1.2  |  21 March 2026  |  CONFIDENTIAL
