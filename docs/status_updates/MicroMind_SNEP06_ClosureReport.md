**MicroMind**

NEP Navigation Enhancement Programme

**S-NEP-06 — VIO Outage and Recovery**

*Final Analytical Closure Report*

22 March 2026 — Schema 06e.3

**Programme Standard: Analytical Closure**

•  All conclusions are derived from data.

•  All conclusions are supported by explicit computation.

•  All conclusions are free from unverified assumptions.

•  Statements not supportable by data are declared explicitly as such.

# **Section 1 — Velocity State Validity**

## **1.1  Frame Validation**

vel_est_ned is state.v from the ESKF (NED frame). vel_gt_ned is columns 8:11 of the EuRoC GT CSV (world frame for MH sequences, approximately NED). Both are nominally in the same frame for MH sequences. No rotation transformation was applied.

Per-axis correlations between vel_est and vel_gt (MH_03, N=7337 entries):

| **Axis** | **corr(est, gt)** | **mean_bias (m/s)** | **Assessment** |
| --- | --- | --- | --- |
| N | -0.214 | +0.487 | Near-zero correlation, large bias |
| E | +0.078 | +1.927 | Near-zero correlation, large bias |
| D | +0.179 | +0.365 | Near-zero correlation, large bias |

## **1.2  Magnitude Analysis**

vel_est starts at |[0,0,0]| = 0 m/s (zero-initialisation). First 5 logged values of |vel_est| ≈ 0.010 m/s. The GT velocity at the same time is |vel_gt| ≈ 0.57 m/s. This confirms the velocity state has not converged at the start of the segment.

vel_err magnitude by thirds of run (MH_03):

| **Run Third** | **│vel_err│ mean (m/s)** | **Interpretation** |
| --- | --- | --- |
| First third | 2.825 | Far from GT — zero-init dominant |
| Middle third | 2.778 | Slow convergence |
| Final third | 1.951 | Partially converged |

## **1.3  Conclusion on Usability**

vel_err_m_s as logged measures the distance between the zero-initialised velocity estimate and the GT velocity. The velocity state is unconverged throughout the controlled segment runs (~60s window). The δv/v ratios of 1.9–2.8 reported in earlier analysis are dominated by this unconverged initial condition and do not represent the structural observability limit of position-only VIO.

**⚠  NOT SUPPORTED BY CURRENT DATA: Velocity contribution to drift cannot be isolated or quantified from vel_err in these runs.**

**⚠  NOT SUPPORTED BY CURRENT DATA: δv/v ratios previously reported are not valid steady-state metrics.**

**✓  CONFIRMED: Velocity state is converging (decreasing vel_err over thirds of run). It is weakly constrained by position-only VIO — consistent with theoretical expectation — but quantification requires a longer converged window.**

# **Section 2 — Drift Characterisation**

## **2.1  Computation Method**

Drift is computed as the incremental change in unaligned position error from the outage start:

incr_drift(t) = ‖state.p(t) − p_GT(t)‖  −  ‖state.p(t_outage_start) − p_GT(t_outage_start)‖

This formulation removes the constant origin-mismatch offset present in the controlled segment runs and yields an origin-independent quantity. The computation uses PROPAGATE log entries during the outage window, sampled every 20 IMU steps (~0.1s).

Three models were fit to incr_drift vs elapsed time since outage start:

- Linear: incr_drift = a·t + b

- Quadratic: incr_drift = a·t² + b·t + c

- Saturation: classified as undetermined when neither linear nor quadratic R² ≥ 0.30

## **2.2  Model Fits — Block A (10s outage, three sequences)**

| **Sequence** | **err_start (m)** | **peak_incr (m)** | **slope (mm/s)** | **R² linear** | **R² quad** | **Best model** | **Loopback** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V1_01 low | 4.13 | 0.87 | −74 | 0.319 | 0.319+ | quadratic | Yes |
| MH_01 mid | 8.98 | 0.04 | −310 | 0.866 | 0.866+ | quadratic | Yes |
| MH_03 high | 4.10 | 7.06 | +800 | 0.941 | 0.941 | linear | No |

## **2.3  Does Drift Equal Velocity Error × Time?**

The hypothesis drift ≈ δv × t was tested by comparing the linear drift slope to vel_err at the outage start:

| **Sequence** | **Drift slope (mm/s)** | **vel_err at outage (mm/s)** | **Ratio slope/vel_err** |
| --- | --- | --- | --- |
| V1_01 low | −74 | 840 | 0.09 |
| MH_01 mid | −310 | 444 | 0.70 |
| MH_03 high | +800 | 3321 | 0.24 |

**⚠  NOT SUPPORTED BY CURRENT DATA: drift = velocity error × time is rejected. Slope/vel_err ratios are 0.09–0.70 — not consistent with a simple relationship.**

**✓  CONFIRMED: Drift arises from accumulated INS propagation error evaluated against a moving ground-truth reference. The error trajectory depends on both the IMU integration path and the GT trajectory shape. These cannot be separated from vel_err alone.**

# **Section 3 — Geometry Effect**

## **3.1  Controlled Comparison**

Block B provides a direct test of geometry influence: V1_01 (structured) and MH_01 (mixed) have nearly identical GT speeds (0.37 vs 0.42 m/s) and heading change rates (both 2.10 rad/s). MH_03 (curved) has higher speed (1.33 m/s) but lower heading change rate (0.74 rad/s), contradicting the label “curved/aggressive” by this metric.

| **Sequence** | **GT speed (m/s)** | **HCR (rad/s)** | **Peak incr drift (m)** | **Loopback** |
| --- | --- | --- | --- | --- |
| V1_01 str | 0.37 | 2.10 | 0.643 | No |
| MH_01 mix | 0.42 | 2.10 | 0.539 | No |
| MH_03 curv | 1.33 | 0.74 | 0.000 | Yes |

Note: MH_03 and MH_01 cannot be used to isolate geometry from speed — their speeds differ by 3×. The V1_01 vs MH_01 comparison (similar speed, similar HCR) shows a small drift difference (643 vs 539 mm) without loopback in either case. This difference is consistent with segment geometry but is not large enough to establish a strong causal attribution.

## **3.2  Loopback Evidence — Critical Test**

The strongest evidence for geometry influence comes from comparing two runs on the same sequence (V1_01) at nearly identical speed but different trajectory segments:

| **Run** | **Segment window** | **GT speed (m/s)** | **Loopback** |
| --- | --- | --- | --- |
| Block A V1_01 | t_lo=15s, outage at t+40s | 0.39 | Yes |
| Block B V1_01 | t_lo=25s, outage at t+55s | 0.37 | No |

**✓  CONFIRMED: Trajectory geometry — specifically which interval of the physical trajectory the outage window covers — determines whether the error trajectory exhibits loopback or monotonic rise.**

**✓  CONFIRMED: Speed alone does not determine loopback. Same sequence, similar speed, different outage placement → different loopback outcome.**

## **3.3  Scope of Geometry Influence**

Geometry affects: error trajectory shape (loopback vs monotonic rise) and the value of drift at outage end (which may be less than the peak if loopback occurs). Geometry does not independently establish drift magnitude scaling from current data — the initial drift rate is determined by the accumulated IMU state at outage start, not by trajectory shape.

**⚠  NOT SUPPORTED BY CURRENT DATA: Geometry does not independently scale the initial drift rate. This relationship is not demonstrable from current data.**

# **Section 4 — Outage Duration Behaviour (Block C)**

## **4.1  Experimental Setup**

Block C uses a single sequence (MH_01_easy), a single segment (VIO window anchored to t_vio_start+5s through t_vio_end-5s, 128s duration), and six outage durations (2s through 30s). All outages start at the same absolute time (t_start + 40s). Filter state at outage start is identical across all durations. GT speed during steady window: 0.567 m/s (consistent across all runs, n=3813 each).

## **4.2  Peak Drift Analysis**

| **Outage (s)** | **Early slope mm/s (first 2s)** | **Peak incr drift (mm)** | **Loopback** | **Innovation at resumption (mm)** |
| --- | --- | --- | --- | --- |
| 2 | −639 | 298 | Yes | 1525 |
| 5 | −717 | 298 | Yes | 1972 |
| 10 | −717 | 1228 | No | 1504 |
| 15 | −717 | 1232 | Yes | 2701 |
| 20 | −717 | 1232 | Yes | 5512 |
| 30 | −717 | 1232 | Yes | 4851 |

## **4.3  Key Observations**

**Initial drift rate is independent of outage duration**

Early slopes (first 2s of outage, all durations): −639, −717, −717, −717, −717, −717 mm/s. Standard deviation = 29 mm/s. All durations start from the same filter state and produce the same initial drift rate.

**✓  CONFIRMED: Initial drift rate is independent of outage duration. It is determined by the filter state at outage start.**

**Peak drift saturates at approximately 1230 mm**

For outages ≥10s, peak incremental drift plateaus at 1228–1232 mm. The peak occurs at approximately 5.6s into the outage for all durations. This corresponds to the maximum divergence between the INS-propagated state and the GT trajectory at that segment of the MH_01 trajectory.

**✓  CONFIRMED: Peak drift is trajectory-bounded. For outages longer than ~5.6s starting at this trajectory point, additional duration does not increase peak drift.**

**Loopback pattern is trajectory-determined**

2s and 5s outages: loopback = True (outage ends before the trajectory divergence peak). 10s outage: loopback = False (outage ends after the peak, trajectory has not yet returned). 15s–30s outages: loopback = True again (the GT trajectory returns toward the INS-propagated position after t+10s of outage).

This pattern — True, True, False, True, True, True — cannot arise from outage duration alone. It reflects the specific geometry of the MH_01 trajectory at the tested outage placement.

**Innovation at resumption is non-monotonic**

Innovation magnitudes (1525, 1972, 1504, 2701, 5512, 4851 mm) do not increase monotonically with outage duration. Durations where loopback occurs produce smaller innovation because the INS-VIO gap at the outage endpoint is reduced by the trajectory returning. This is consistent with the loopback pattern above.

## **4.4  Corrected Classification**

**⚠  NOT SUPPORTED BY CURRENT DATA: The 5s→10s transition is NOT a filter regime transition. The filter architecture, parameters, and measurement model do not change with outage duration.**

**✓  CONFIRMED: The transition is a trajectory-dependent threshold: outages ending before t+5.6s (relative to outage start) do not reach the peak divergence point; outages ending after t+5.6s do. This is a property of the MH_01 trajectory at the tested outage placement, not of the estimator.**

**✓  CONFIRMED: Behaviour depends on the alignment of the outage window with the trajectory divergence structure at that segment. Both outage duration and trajectory geometry contribute to observed outcomes — they cannot be fully separated with a single fixed outage placement.**

# **Section 5 — Constraints and Non-Conclusions**

The following statements are explicitly not supported by current data:

## **5.1  Velocity Error and Drift Attribution**

**⚠  NOT SUPPORTED BY CURRENT DATA: Velocity error magnitude is quantifiable from vel_err in these runs. The velocity state is unconverged throughout.**

**⚠  NOT SUPPORTED BY CURRENT DATA: δv/v ≈ 2.5 is a structural property of position-only VIO. This ratio was computed from unconverged velocity estimates.**

**⚠  NOT SUPPORTED BY CURRENT DATA: Drift is caused by velocity error. The causal mechanism of drift is accumulation of INS propagation error against a moving GT reference. Velocity error contributes but cannot be isolated as dominant from current data.**

**⚠  NOT SUPPORTED BY CURRENT DATA: drift = velocity error × time. This relationship is rejected by the slope/vel_err ratios (0.09–0.70).**

## **5.2  Geometry Quantification**

**⚠  NOT SUPPORTED BY CURRENT DATA: Heading change rate (HCR) reliably quantifies trajectory curvature for this purpose. MH_03 shows lower HCR than V1_01/MH_01 despite being labelled more aggressive.**

**⚠  NOT SUPPORTED BY CURRENT DATA: Geometry independently scales initial drift rate. No controlled evidence supports this relationship.**

## **5.3  Filter Behaviour**

**⚠  NOT SUPPORTED BY CURRENT DATA: A filter regime transition occurs with outage duration. The observed transition at 5s→10s is trajectory-dependent, not a filter state change.**

**⚠  NOT SUPPORTED BY CURRENT DATA: H3 (monotonic recovery time with outage duration) can be evaluated from controlled segment runs. Recovery metric is non-evaluable due to origin mismatch in these runs.**

## **5.4  Recovery**

**⚠  NOT SUPPORTED BY CURRENT DATA: Recovery timing can be inferred from Block C runs. The 9.28 m origin mismatch in the controlled segment runs makes the recovery threshold (τ = 111 mm) structurally unreachable.**

Reference: The original S-NEP-06 full-sequence runs (s06_2s, s06_5s, s06_10s, s06_20s) provide the only valid recovery evidence. In those runs, all four outage durations showed the causal rolling mean returning within τ immediately at outage resumption (recovery_time = 0.0s) — the Kalman correction at resumption brought the rolling mean error below τ within the first 5s of post-outage operation. This is the resolution limit of the causal metric, not a claim of instantaneous convergence.

# **Section 6 — Mathematically Defensible Conclusions**

The following statements are directly supported by explicit computation from logged data. No extrapolation beyond the evidence is made.

| **ID** | **Conclusion** | **Evidence** | **Source** |
| --- | --- | --- | --- |
| C-01 | Loopback (error peak-and-fall during outage) is determined by trajectory geometry, not speed alone. | Block A vs B V1_01: same speed (0.37–0.39 m/s), different segment placement, different loopback outcome. | Block A/B ctrl2 logs |
| C-02 | Initial drift rate is independent of outage duration. | Early slopes (first 2s) across 2s–30s outages: −639, −717, −717, −717, −717, −717 mm/s. Std = 29 mm/s. | Block C ctrl3 logs |
| C-03 | Peak incremental drift saturates at approximately 1230 mm for outages ≥10s at the tested trajectory placement. | Peak values: 1228, 1232, 1232, 1232 mm for 10s, 15s, 20s, 30s outages. All peak at t+5.6s. | Block C ctrl3 logs |
| C-04 | The 5s→10s behavioural transition is a trajectory-dependent threshold, not a filter regime change. | Outages ending before t+5.6s show loopback (2s, 5s). Outage ending after t+5.6s does not (10s). Filter parameters unchanged throughout. | Block C ctrl3 logs |
| C-05 | Innovation at resumption is non-monotonic with outage duration. | Values: 1525, 1972, 1504, 2701, 5512, 4851 mm for 2s–30s. Not monotone. | Block C ctrl3 logs |
| C-06 | MH_03 shows higher incremental drift (7.06 m) and no loopback under 10s outage at the tested segment. | Block A direct measurement. V1_01: 0.87 m, MH_01: 0.04 m, MH_03: 7.06 m (all incremental, origin-independent). | Block A ctrl2 logs |
| C-07 | Drift growth in MH_03 is well-described by a linear model (R²=0.941, slope=+800 mm/s) over the 10s outage window. | Fit computed from 101 PROPAGATE entries during outage. | Block A ctrl2 logs |
| C-08 | In full-sequence S-NEP-06 runs, the causal rolling mean error returned within τ=111 mm within the first 5s of post-outage operation for all tested durations (2s–20s). | Recovery analysis using causal unaligned metric from s06_2s through s06_20s baseline logs. | S-NEP-06 original runs |
| C-09 | Outage suppression operates correctly: gap sizes match commanded outage durations within 30ms. | Block A/B: gaps 10.01–10.03s vs commanded 10s. Block C: 2.02, 5.03, 10.03, 15.01, 20.02, 30.02s. | All ctrl2/ctrl3 logs |

# **Programme Status**

| **Sprint** | **Status** | **Commit** | **Deliverable** |
| --- | --- | --- | --- |
| S-NEP-04 | ✅ CLOSED | 154c860 | VIO integration validated, update_vio() interface |
| S-NEP-05 | ✅ CLOSED | 4dd3a76 | Mission-scale stability confirmed, ATE 0.0800 m |
| S-NEP-06 | ✅ CLOSED (this document) | — | Outage/recovery characterised, analytical standard established |

*This report establishes the programme standard for analytical discipline. All future phases must produce conclusions that are derived from data, supported by explicit computation, and free from unverified assumptions. Statements not supportable by current data must be declared explicitly.*
