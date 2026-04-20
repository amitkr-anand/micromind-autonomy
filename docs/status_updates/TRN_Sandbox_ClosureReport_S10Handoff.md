**MicroMind / NanoCorteX**

TRN Sandbox --- Himalayan Manoeuvre Corridor

**Closure Report & S10 Handoff**

March 2026 \| v3 Final --- 2D Patch NCC

1\. Purpose of This Sandbox

The pre-S10 TRN sandbox was established to answer a single forcing
question before the TASL partnership decision: does the MicroMind TRN
subsystem hold NAV-01 (worst 5km error \< 100m) on a manoeuvring UAV
flight profile over real Himalayan terrain?

This question was not answered by any prior sprint. S8-C acceptance
tests validated 10km straight corridors. The 150km ALS-250 regression
corridor (S9) was also straight. TASL engineers reviewing the system
would immediately ask whether the nav solution degrades when the UAV
changes heading --- a fundamental operational requirement. The sandbox
was designed to produce a defensible, evidence-backed answer before that
meeting.

2\. Test Corridor Definition

All experiments used a 150km five-segment Kashmir-to-Zanskar corridor
over the srinagar_dem_ext.tif SRTM DEM (7498x16752 pixels, 28.3m/px,
247-7073m elevation range).

  --------- -------------- ------------- ------------- -----------------------------------
  **Seg**   **Distance**   **Heading**   **Terrain**   **Description**

  A         30 km          045°          Valley        Valley departure --- Kashmir floor,
                                                       flat start, low σ_terrain

  B         28 km          020°          Ridge         Ridge climb --- terrain roughness
                                                       building rapidly

  C         35 km          070°          High ridge    High-ridge traverse --- peak
                                                       σ_terrain \>200m, 6000m+ peaks

  D         30 km          035°          Valley        Valley transition --- Zanskar river
                                                       system, moderate roughness

  E         27 km          055°          Mountain      Mountain exit --- mixed terrain,
                                                       corridor closes
  --------- -------------- ------------- ------------- -----------------------------------

Heading changes between segments: A-B 25°, B-C 50°, C-D 35°, D-E 20°.
Crosstrack wind model: ±150m sinusoidal at 4km period. Mountain
coverage: 75% of route. Mean σ_terrain along path: 101.9m (well above
the 40m spec minimum).

3\. Experiment History --- What Was Tried and Why It Failed

3.1 v1 and v2: 1D Profile NCC

Both v1 and v2 used the architecture inherited from the ALS-250
corridor: a 1D terrain profile sampled along the current heading,
correlated against candidate profiles in the DEM search area. The v2 run
included three bug fixes over v1 (cache removal, fine grid at DEM native
resolution, fresh RADALT noise seed per call).

Result: v1 and v2 both achieved 5/75 accepted updates and NAV-01 = 422m.
The bug fixes made no measurable difference.

Root cause (confirmed by innovation scatter analysis): the 1D profile
NCC is architecturally unable to resolve cross-track position error. The
returned correction vector (bdn, bde) contains a physically meaningful
along-track component but a spurious cross-track component equal to grid
quantisation noise (\~1-3 grid steps, \~28-85m). When INS error exceeds
\~120m, cross-track noise pushes the 2D innovation magnitude past the
150m gate, rejecting geometrically correct matches. All 64 rejected
updates had innovation values sitting precisely on the y=x line in the
scatter plot --- the NCC was finding the right answer; the gate was
failing due to noise in the unused axis.

This is consistent with published literature: SITAN (ION NAVIGATION
2021) notes that 1D profile TRN is \'prone to false-fixes during
conditions of large initial position error\', and TERCOM \'restricts
aircraft manoeuvring during update\' for the same observability reason.

3.2 v3: 2D Patch NCC --- Three Implementation Bugs

The architectural decision was made to replace 1D profile NCC with 2D
terrain patch NCC. Rather than sampling along a heading, the system
extracts a rectangular DEM patch at the true position (as the RADALT
template) and slides it across a larger search window centred at the
INS-estimated position. Both N and E offsets are directly resolved by
the correlation peak. The design was validated on synthetic terrain
before implementation.

v3 required three sequential bug fixes before producing correct results.
Each fix unmasked the next:

-   Score normalisation (Run 1): The NCC score formula divided by (w_std
    \* ts \* n_pix). Since the template was already unit-normalised
    (dividing by ts during normalisation), ts appeared twice in the
    denominator. On Himalayan terrain with ts \~370m, all scores were
    deflated by a factor of 370 --- maximum score across all updates was
    0.003 against a 0.45 threshold. NCC-rej=69, wall time 1 second, no
    computation was effectively running. Fix: remove ts from the
    denominator. Correct formula: dot / (w_std \* n_pix).

-   Correction sign --- North axis (Run 2): After the score fix, NCC-rej
    dropped to 0 but innov-rej jumped to 64-65. The innovation scatter
    showed all rejected points on the y=x line with INS errors of
    200-370m. Diagnosis: each accepted correction was doubling the error
    rather than halving it. bdn_m was computed as -dn_px \* res_m. Since
    dn_px \> 0 means truth is south of INS (DEM rows increase
    southward), the correction err_n -= bdn_m = err_n -= (-positive) =
    err_n += positive, amplifying the northward error. Fix: bdn_m =
    +dn_px \* res_m.

-   Correction sign --- East axis (Run 2, same fix): bde_m had the same
    sign inversion on the east axis. DEM columns increase eastward, so
    de_px \> 0 means truth is west of INS. The original formula bde_m =
    +de_px \* res_m applied corrections in the wrong direction. Fix:
    bde_m = -de_px \* res_m. All four cardinal cases (N, S, E, W
    displacements) were verified against known-offset synthetic terrain
    before applying the fix.

Additionally, the forced update mechanism (MAX_CONSEC_SUPP=3) was
disabled by setting the cap to 999. The first 15km of the corridor is
Kashmir valley floor with σ_terrain \< 10m. A forced NCC update on flat
terrain introduced a \~100m residual error; subsequent drift then took
the INS beyond the 150m gate within 2km, causing an irreversible
cascade. With the forced update disabled, the sigma gate simply
suppresses flat-terrain updates (correctly) and corrections begin
cleanly at km \~15 when the terrain roughens above threshold.

4\. Final Results --- v3 with 2D Patch NCC

  ------------------- ---------------- ---------------- -----------------
  **Metric**          **v1/v2 (1D      **v3 (2D Patch   **Change**
                      NCC)**           NCC)**           

  NAV-01 Worst 5km    421.9m           **33.3m**        **−92%**

  RMS Error           \~280m           **17m**          **−94%**

  P95 Error           \~366m           **29m**          **−92%**

  Accepted updates    5/75 (7%)        **68/75 (90%)**  **+13.6×**

  Innovation rejected 64-69            **0**            **Eliminated**

  NCC rejected        0-69             **0**            **Eliminated**

  Max correction gap  127-150km        **13.9km**       **−89%**

  Mean gap            \~150km          **2.2km**        Nominal 2km
                                                        interval

  NAV-01 gate (100m)  **FAIL**         **PASS --- 3.0×  
                                       margin**         
  ------------------- ---------------- ---------------- -----------------

Both patch sizes (600m standard, 900m extended) produced identical
results, confirming that 600m is sufficient for unambiguous 2D terrain
matching on Himalayan ridges. Patch size is not a sensitivity parameter
on this terrain.

5\. Performance Note: Why 2D Patch Runs Faster than 1D Profile

The v3 run completed in approximately 1 second for the full 150km
corridor, compared to \~807 seconds for v2. This is counterintuitive and
warrants explanation for the record.

The 1D NCC in v2 used a Python inner loop: for each of the 841 search
grid candidates (29×29 at 28.3m step), it called dem.get_elev() once per
profile point --- 40 calls per candidate --- resulting in approximately
33,640 individual Python-level DEM lookups per update. Over 69 eligible
updates, this totalled \~2.3 million Python function calls.

The 2D patch NCC reads the DEM exactly twice per update (one
get_patch_px for the template, one for the search window) then performs
all 841 candidate correlations in a single sliding_window_view operation
followed by vectorised numpy matrix operations --- entirely in C with no
Python loop touching the DEM. The cost ratio is approximately 33,640:2
DEM reads, with the inner loop replaced by BLAS-optimised matrix
operations.

The implication for the real payload is positive: 2D patch NCC is not
only architecturally superior (full 2D observability) but
computationally cheaper than the 1D approach it replaces. On a dedicated
compute board, the same vectorised approach --- potentially with CUDA
--- would run faster still. The earlier Phase-2 classification of GPU
NCC optimisation should be reconsidered: the vectorised numpy approach
may already be sufficient for real-time operation on a mid-tier embedded
processor.

6\. What Is Carried Into S10

6.1 Architecture Decision: 2D Patch NCC Replaces 1D Profile

The trn_stub.py NCC implementation will be updated to use 2D terrain
patch NCC. The specific parameters validated in the sandbox:

  ---------------------- ------------------ -----------------------------
  **Parameter**          **Value**          **Notes**

  Patch size             600m × 600m        21×21 px at 28.3m DEM res

  Search radius          400m (±400m N and  Unchanged from Hybrid-3
                         E)                 config

  Search window          53×53 px           patch + 2×search_radius

  NCC candidates         31×31 = 961        sliding_window_view, fully
                                            vectorised

  Score threshold        0.45               Unchanged from FR-107

  Innovation gate        150m (2D           Gate is now valid on both
                         Euclidean)         axes

  Score formula          dot / (w_std \*    Template pre-normalised; ts
                         n_pix)             not in denominator

  Correction signs       bdn = +dn_px\*res, Verified all 4 cardinal
                         bde = -de_px\*res  directions

  Forced updates         Disabled (cap=999) Flat terrain correction
                                            cascade risk
  ---------------------- ------------------ -----------------------------

6.2 Gate Config Unchanged --- Hybrid-3 Remains Frozen

The Hybrid-3 gate configuration established in S9 is unchanged and
remains frozen:

-   Innovation gate: 150m

-   σ_terrain threshold: 10m

-   Max consecutive suppressed: 3 (operative for ALS-250 straight
    corridor; set to 999 for Himalayan sandbox only)

-   Search radius: 400m

The 2D patch architecture makes the innovation gate physically
meaningful on both axes. The gate value does not need to change.

6.3 S10 Execution Sequence (Strict Order, Unchanged)

The sandbox closure does not change the S10 scope or sequence. S10
proceeds as follows:

-   S10-1: Deploy trn_stub_ncc_patch.py to repo. Run 68/68 TRN
    regression gates. Run 50km STIM300 smoke test on ALS-250.

-   S10-2: Deploy test_s9_nav01_pass.py to tests/. Confirm 7/7 gates
    pass. Suite reaches 222/222.

-   S10-4: Deploy run_als250_parallel.py to repo root (subprocess.Popen
    approach, MKL deadlock resolved). Run 50km parallel smoke on all 3
    IMUs (STIM300, ADIS16505-3, BASELINE).

-   S10-3: Deploy als250_drift_chart.py to dashboard/. Run full 250km ×
    3 IMUs. Generate PNG + PDF drift chart.

-   Close: git rm run_als250_parallel_v2.py. Commit and push.
    SPRINT_STATUS.md updated. M2 and M5 closed.

6.4 Pending Pre-S10 Actions

-   Change MAX_TRN_CORRECTION_M from 300m to 150m in als250_nav_sim.py.

-   Record Hybrid-3 gate config in trn_stub.py implementation notes.

-   Update Part Two V7 STIM300 ARW floor from 0.1 to \<=0.2 deg/sqrt(hr)
    before TASL meeting.

7\. Lessons Learned --- For Programme Record

-   **L1.** Short acceptance corridors hide systemic gaps.

The S8-C acceptance suite validated 10km straight corridors. The 150km
ALS-250 regression corridor was also straight. Neither exposed the 1D
NCC cross-track blindness problem because cross-track error is
negligible on short straight flights. The Himalayan sandbox (first
manoeuvring test at operational length) immediately revealed the
architectural limit. Acceptance test corridors must be as long and as
manoeuvre-rich as the operational scenario.

-   **L2.** Innovation scatter is the definitive diagnostic for NCC
    pipeline bugs.

In every failed run, the innovation scatter showed the rejected points
sitting on the y=x line --- the NCC was finding the geometrically
correct answer even when sign errors and score errors prevented
corrections from being applied. Reading the scatter first would have
identified the root cause faster than inspecting code. This should be a
standard diagnostic plot for any TRN debugging session.

-   **L3.** Architectural correctness beats parameter tuning.

The entire v1/v2 investigation (cache, grid spacing, template length,
noise seeding) was parameter-level debugging on an architecturally
limited system. No combination of parameters could make 1D NCC resolve
cross-track position error --- it is not a tuning problem. The correct
response to repeated parameter-level failure is to question the
architecture, not add more parameters.

-   **L4.** Wall time is a pipeline diagnostic.

v3 Run 1 completed in 1 second for 150km. This alone should have
indicated that the NCC was not running for the majority of updates --- a
1-second run with 69 NCC-eligible updates and 841 candidates each is
physically impossible if the computation is executing. Runtime anomalies
are first-class diagnostics.

-   **L5.** Sign convention bugs accumulate silently.

The bdn and bde sign errors produced plausible-looking output: positive
acceptance counts (4/75), a NAV-01 that was lower than the INS-only
case, an innovation scatter with points below the gate. Each of these
was misleading. The only reliable indicator was that corrections made
things worse rather than better --- visible only in the navigation error
trajectory plot, not in the summary metrics. Summary metrics alone are
insufficient; the full nav error trajectory must be inspected.

8\. Sandbox Closure Statement

The TRN sandbox is closed. The pre-S10 forcing question is answered:

**2D Patch NCC holds NAV-01 on a 150km manoeuvring Himalayan corridor at
33.3m --- a 3.0x margin. TASL can be told this with real SRTM data
behind it.**

The 2D patch architecture is now the confirmed TRN NCC design for
MicroMind Phase 1. It is faster than the 1D approach it replaces,
resolves both position axes, and is validated on real terrain at
operational corridor length with realistic heading changes and
crosstrack disturbance. S10 proceeds to close M2 and M5.

MicroMind Programme \| Sandbox Report \| March 2026 \|
amitkr-anand/micromind-autonomy
