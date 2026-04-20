**Navigation Evaluation Platform**

Platform Contract Specification

*Phase-0 Design Freeze*

  ---------------------------- ------------------------------------------
  **Document Status**          Phase-0 Baseline (Frozen)

  **Version**                  1.1

  **Programme**                MicroMind / NanoCorteX

  **Platform Version**         NEP_1.0

  **Architecture**             v1.2
  ---------------------------- ------------------------------------------

+-----------------------------------------------------------------------+
| **Design Freeze Notice --- Phase-0 Baseline (Frozen)**                |
|                                                                       |
| This document is the Phase-0 Baseline of the Navigation Evaluation    |
| Platform.                                                             |
|                                                                       |
| The architecture it describes is frozen.                              |
|                                                                       |
| No block responsibilities, interface contracts, or information flow   |
| sequences may                                                         |
|                                                                       |
| be altered without a formal versioned revision to this specification. |
|                                                                       |
| Future implementations must be verified against the contracts defined |
| here.                                                                 |
|                                                                       |
| Future architectural changes must occur through versioned updates to  |
| this specification.                                                   |
|                                                                       |
| Development follows a strict cycle: Specification → Implementation →  |
| Verification.                                                         |
|                                                                       |
| No architectural reinterpretation is permitted during implementation. |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Version 1.1 --- Phase-0 Baseline**                                  |
|                                                                       |
| Version 1.1 incorporates five final architectural clarifications and  |
| marks the                                                             |
|                                                                       |
| Phase-0 Baseline as frozen. The architecture is unchanged from v1.0.  |
|                                                                       |
| Added: runner timestamp ownership rule (§5.6), latency domain         |
| invariant (INV-PE-05),                                                |
|                                                                       |
| minimum trajectory condition (§6.1), archive immutability boundary    |
| (§2.7, §8.4),                                                         |
|                                                                       |
| and experiment validity classification (§8.3).                        |
+-----------------------------------------------------------------------+

**1 System Intent**

**1.1 Core Purpose**

The Navigation Evaluation Platform (NEP) runs navigation algorithms on
labelled datasets, measures their accuracy and performance, and produces
immutable, traceable experiment records. It is a benchmarking
infrastructure, not a navigation system.

**1.2 What the Platform Does**

-   Loads experiment configuration from validated YAML files.

-   Loads dataset ground-truth trajectories and image timestamp
    registries from disk.

-   Instantiates and dispatches algorithm runners via a registration
    registry.

-   Collects pose estimates from runners as a streaming generator.

-   Computes navigation accuracy, latency, update rate, and stability
    metrics from the pose stream.

-   Produces immutable, write-once experiment archives containing
    metrics, metadata, and logs.

-   Validates platform pipeline integrity through infrastructure runners
    before real algorithm integration.

**1.3 What the Platform Explicitly Does NOT Do**

-   Implement navigation algorithms. Algorithm logic lives entirely
    outside the platform.

-   Download or manage dataset files. Datasets are assumed present on
    disk before any experiment runs.

-   Modify ground truth data. Ground truth is read-only input to the
    evaluation layer.

-   Generate visualisation plots. VisualizationModule (B-06) is outside
    Phase-0 scope.

-   Schedule or distribute experiments across multiple machines.

-   Perform real-time evaluation. All evaluation is offline,
    post-pose-stream.

-   Provide algorithm configuration or parameter tuning. Parameters are
    loaded verbatim from file.

**1.4 Expected Usage Model**

A user provides a dataset on disk, a configuration YAML file describing
the algorithm and parameters, and optionally an ExperimentPlan for
multi-algorithm benchmarks. The platform loads the configuration, runs
the algorithm, evaluates the output against ground truth, and commits
the results. The user reads the archived MetricSet and metadata to
assess algorithm performance.

+-----------------------------------------------------------------------+
| **Architectural Principle**                                           |
|                                                                       |
| The platform evaluates algorithms. It does not implement them.        |
|                                                                       |
| Every algorithm-specific concern lives in a registered Runner module. |
|                                                                       |
| The platform infrastructure never changes when a new algorithm is     |
| added.                                                                |
+-----------------------------------------------------------------------+

**2 Block Architecture**

**2.1 Block Map**

The platform is composed of seven major blocks. Each block owns a single
responsibility domain. Dependencies flow in one direction only:
ExperimentRunner (B-01) is the sole orchestration root; all other blocks
are called by it but never call it back.

  ------------------------------------------------------------------------------------------
  **Block   **Name**               **Responsibility**    **Output**            **Phase-0**
  ID**                                                                         
  --------- ---------------------- --------------------- --------------------- -------------
  B-01      ExperimentRunner       Orchestration root    ExperimentRecord      Complete
                                   --- coordinates all                         
                                   other blocks.                               

  B-02      ConfigurationManager   Loads and validates   ExperimentConfig      Complete
                                   experiment YAML.                            
                                   Returns                                     
                                   ExperimentConfig.                           

  B-03      DatasetManager         Loads dataset         GroundTruth           Partial
                                   ground-truth and                            (EuRoC)
                                   image timestamps from                       
                                   disk. Returns                               
                                   GroundTruth.                                

  B-04      Runner Layer           Algorithm execution   PoseEstimate stream   Interface
                                   --- reads dataset                           defined
                                   files, emits                                
                                   PoseEstimate stream.                        

  B-05      MetricsEngine          Streaming evaluation  MetricSet             Complete
                                   --- computes                                
                                   MetricSet from pose                         
                                   stream and ground                           
                                   truth.                                      

  B-06      VisualizationModule    Generates trajectory  Plot files            Out of scope
                                   and metric plots from                       
                                   MetricSet.                                  

  B-07      ResultsArchive         Write-once experiment archive on disk       Complete
                                   storage. Owns                               
                                   results/ directory.                         
  ------------------------------------------------------------------------------------------

**2.2 ExperimentRunner (B-01) --- Orchestration Root**

ExperimentRunner is the only block permitted to import and call other
blocks. No other block imports ExperimentRunner. This invariant is
absolute and must be verifiable by import analysis.

-   Responsibilities: configuration loading, dataset loading, runner
    dispatch, metric computation loop, archive commit.

-   Inputs: config file path, nav_code string, or ExperimentPlan.

-   Outputs: ExperimentRecord.

-   Forbidden dependencies: none --- it is the only block with
    cross-block visibility.

-   Dry-run mode: when ExperimentConfig.dry_run is True, SyntheticRunner
    is always used regardless of algorithm field. No real dataset or
    runner is required.

**2.3 ConfigurationManager (B-02)**

-   Responsibilities: parse YAML, validate against schema.yaml, resolve
    paths to absolute, compute SHA-256 hash of parameters file,
    construct ExperimentConfig.

-   Inputs: config file path, experiment_id (supplied by
    ExperimentRunner), platform_root.

-   Outputs: ExperimentConfig (ICD-01).

-   Forbidden dependencies: evaluation/, runners/, results_archive.

-   experiment_id is supplied by the caller --- ConfigurationManager
    does not generate IDs. If present in the YAML file it is accepted
    but overridden by the runtime value.

**2.4 DatasetManager (B-03)**

-   Responsibilities: read dataset CSV files from disk, validate
    monotonicity, compute path length, construct GroundTruth.

-   Inputs: dataset_name in source/sequence format.

-   Outputs: GroundTruth (evaluation layer type).

-   Forbidden dependencies: runners/, evaluation/metrics_engine,
    results_archive, config/.

-   Phase-0 scope: EuRoC ASL CSV format is fully implemented. KITTI and
    TartanAir are registered stubs that raise NotImplementedError.

-   DatasetManager does not produce DatasetRef in Phase-0. DatasetRef
    construction for real runners is a Phase-1 integration step.

**2.5 Runner Layer (B-04)**

-   Responsibilities: execute the algorithm, read dataset files via
    DatasetRef, emit PoseEstimate objects one at a time via a generator.

-   Inputs: ExperimentConfig, DatasetRef, PlatformLogger.

-   Outputs: PoseEstimate stream (generator), metadata dict.

-   Forbidden dependencies: evaluation/ (MetricsEngine, GroundTruth,
    MetricSet), results_archive, experiment_runner, experiment_plan,
    config/.

-   Runners are registered in ExperimentRunner.\_RUNNER_REGISTRY. Adding
    an algorithm requires only adding a registration entry and
    implementing BaseRunner.

**2.6 MetricsEngine (B-05)**

-   Responsibilities: incremental evaluation --- accumulate positions
    and timestamps from PoseEstimate stream, compute
    ATE/RPE/drift/latency/tracking metrics, return MetricSet.

-   Inputs: PoseEstimate stream (via update()), GroundTruth (via
    finalise()).

-   Outputs: MetricSet (ICD-08/09).

-   Forbidden dependencies: runners/, config/, results_archive,
    experiment_runner.

-   MetricsEngine must not perform file I/O of any kind. All data
    arrives via method arguments.

**2.7 ResultsArchive (B-07)**

-   Responsibilities: write-once experiment archive, atomic commit
    (EXP_ID.tmp rename), global experiment counter, experiment ID
    generation.

-   Inputs: ExperimentRecord, metadata dict, metrics dict, plot and log
    file bytes.

-   Outputs: archived directory tree under results/.

-   Forbidden dependencies: evaluation/, runners/, config/.

-   ResultsArchive is the sole owner of the results/ directory. No other
    block creates directories under results/.

-   next_experiment_id() is the only source of experiment IDs.
    ExperimentRunner calls it once per experiment immediately before
    execution.

-   Once ResultsArchive.commit() successfully completes, the archived
    record is immutable. No subsequent process may modify the experiment
    directory, its metrics, metadata, logs, or plots. The archive
    functions as a benchmark ledger.

**2.8 Interfaces Layer**

-   Responsibilities: define frozen data contracts between blocks as
    Python dataclasses with \_\_post_init\_\_ validation.

-   ICD types are immutable (frozen=True). They are the only permitted
    communication channel between blocks.

-   interfaces/constants.py is the single source of truth for
    VALID_NAV_TYPES, VALID_ALGORITHMS, and VALID_NAV_CODES. No block may
    define its own copy of these sets.

**3 Information Flow**

**3.1 Single-Experiment Lifecycle**

The following sequence defines the complete lifecycle of one experiment.
Each step is mandatory. The sequence is enforced by
ExperimentRunner.\_run_from_config().

  --------------------------------------------------------------------------------
  **Step**   **Name**       **Description**
  ---------- -------------- ------------------------------------------------------
  1          ID reservation ExperimentRunner calls
                            ResultsArchive.next_experiment_id(nav_code). The
                            returned ID is used for all subsequent steps. ID
                            reservation happens before any file I/O.

  2          Config loading ExperimentRunner calls
                            ConfigurationManager.load(config_path, experiment_id,
                            platform_root). Returns ExperimentConfig. All paths
                            are absolute. Parameters file is hashed.

  3          Logger init    ExperimentRunner creates an experiment-scoped
                            PlatformLogger writing to a temporary directory. The
                            logger scope ends at step 7.

  4          GroundTruth    ExperimentRunner calls
             loading        DatasetManager.load(dataset_name). Returns
                            GroundTruth. For dry_run=True, a synthetic GroundTruth
                            matched to SyntheticRunner output is used.

  5          DatasetRef     ExperimentRunner constructs DatasetRef for the runner.
             construction   For dry_run=True, a placeholder DatasetRef is used.
                            For dry_run=False, real DatasetRef construction is a
                            Phase-1 integration step; the current code raises
                            NotImplementedError at this step.

  6          Runner         ExperimentRunner calls \_dispatch(config, dataset_ref,
             dispatch       logger). For dry_run=True, SyntheticRunner is always
                            used. For dry_run=False, (navigation_type, algorithm)
                            is looked up in \_RUNNER_REGISTRY. Missing entry
                            raises RunnerDispatchError.

  7          Pose streaming ExperimentRunner drives: MetricsEngine.begin(); for
             loop           pose in runner.stream(): MetricsEngine.update(pose).
                            Each PoseEstimate is consumed immediately. Poses are
                            not accumulated. MetricsEngine.update() raises
                            TimestampRegressionError on non-increasing timestamps.

  8          Metric         ExperimentRunner calls
             finalisation   MetricsEngine.finalise(groundtruth). Returns
                            MetricSet. If runner produced zero poses,
                            EmptyStreamError is raised before finalise() executes.

  9          Runner         ExperimentRunner calls runner.metadata() to collect
             metadata       algorithm-specific fields. Called after stream()
                            exhausts, before runner.close().

  10         Runner close   ExperimentRunner calls runner.close() in a finally
                            block --- guaranteed even on error. Idempotent by
                            contract.

  11         Record         ExperimentRunner constructs ExperimentRecord from
             construction   experiment_id, status, duration_s, and
                            acceptance_pass. All paths are relative.

  12         Archive commit ExperimentRunner calls ResultsArchive.commit(record,
                            metadata, metrics, plot_files, log_files). Atomically
                            writes EXP_ID.tmp then renames to EXP_ID.
  --------------------------------------------------------------------------------

**3.2 Data Objects Exchanged**

  -------------------------------------------------------------------------
  **Object**         **Flow**     **Description**
  ------------------ ------------ -----------------------------------------
  ExperimentConfig   B-02 → B-01  Validated, resolved experiment
  (ICD-01)                        configuration. All paths absolute.

  DatasetRef         B-01 → B-04  Validated dataset file paths. Runners use
  (ICD-02)                        this to locate dataset files.

  GroundTruth        B-03 → B-05  Reference trajectory and image timestamp
                                  registry. Owned by DatasetManager.

  PoseEstimate       B-04 → B-05  Single pose estimate from algorithm
  (ICD-05/06)                     runner. Strictly increasing timestamps
                                  required.

  MetricSet          B-05 → B-01  Complete evaluation result. Immutable.
  (ICD-08/09)                     All gate fields internally consistent.

  ExperimentRecord   B-01 → B-07  Complete experiment record for archive
  (ICD-11)                        commit. Relative paths only.
  -------------------------------------------------------------------------

**3.3 Multi-Experiment: ExperimentPlan**

ExperimentPlan defines a benchmarking scenario over multiple algorithms
on a single dataset. ExperimentRunner.execute(plan) iterates over
AlgorithmSpec objects from plan.iter_algorithm_specs() and calls
\_run_from_config() for each. ID reservation, execution, and archive
commit occur independently per algorithm. The shared counter in
ResultsArchive ensures globally unique IDs across both run() and
execute() entry points.

**4 Interface Contracts**

**4.1 PoseEstimate (ICD-05/06)**

Produced by algorithm runners. Consumed by MetricsEngine.

  --------------------------------------------------------------------------
  **Field**           **Contract**
  ------------------- ------------------------------------------------------
  timestamp_s         Float64. Dataset clock time of algorithm output.
                      Enforced \>= frame_timestamp_s at construction.

  frame_timestamp_s   Float64. Dataset clock time of the source image frame.
                      Populated from DatasetRef or image timestamp registry.

  position_x/y/z      Float64. ENU world frame, metres. World origin defined
                      by dataset ground-truth frame.

  quat_w/x/y/z        Float64. Hamilton quaternion. Unit norm enforced at
                      construction (atol=1e-3).

  covariance_6x6      6x6 float64 numpy array. \[x, y, z, roll, pitch, yaw\]
                      uncertainty. Must not be identity matrix or all-zeros
                      (EP-04).

  tracking_valid      Bool. False if pose is propagated only (no visual
                      update this frame).

  feature_count       Non-negative integer. Number of tracked visual
                      features this frame.

  source_algorithm    Non-empty string. Algorithm identifier for archiving
                      and comparison.
  --------------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **PoseEstimate Invariants**                                           |
|                                                                       |
| INV-PE-01: timestamp_s \>= frame_timestamp_s. Enforced at             |
| construction.                                                         |
|                                                                       |
| INV-PE-02: Successive PoseEstimates in any stream must have strictly  |
| increasing timestamp_s.                                               |
|                                                                       |
| INV-PE-03: covariance_6x6 must be neither identity nor all-zeros.     |
|                                                                       |
| INV-PE-04: timestamp_s must originate from the dataset clock, not     |
| system wall-clock.                                                    |
|                                                                       |
| INV-PE-05: Latency (timestamp_s − frame_timestamp_s) must be \>= 0.   |
| Negative latency is                                                   |
|                                                                       |
| invalid and is rejected by PoseEstimate construction. This invariant  |
| is                                                                    |
|                                                                       |
| enforced at the runner boundary, not by MetricsEngine.                |
+-----------------------------------------------------------------------+

**4.2 ExperimentConfig (ICD-01)**

Produced by ConfigurationManager. Consumed by ExperimentRunner and
passed to runners.

  -------------------------------------------------------------------------
  **Field**          **Contract**
  ------------------ ------------------------------------------------------
  experiment_id      Format: EXP\_{TYPE}\_{NNN}\_{YYYYMMDD}\_{HHMMSS}.
                     Supplied by ExperimentRunner, not
                     ConfigurationManager.

  navigation_type    One of: vio \| trn \| fusion \| terminal. Defined in
                     interfaces/constants.py.

  algorithm          Registered algorithm identifier. Defined in
                     interfaces/constants.py.

  parameters_path    Absolute Path. File must exist at load time.

  parameters_hash    SHA-256 hex digest of parameters file. Format:
                     sha256:\<64 hex chars\>.

  dry_run            Bool. True means infrastructure validation only.
                     SyntheticRunner is always used.

  platform_root      Absolute Path. Repository root. Used to resolve
                     relative paths.

  platform_version   String starting with NEP\_. Current: NEP_1.0.
  -------------------------------------------------------------------------

**4.3 DatasetRef (ICD-02)**

Passed to algorithm runners. Contains only dataset file paths and
metadata. Runners use this to locate files on disk.

+-----------------------------------------------------------------------+
| **DatasetRef Invariants**                                             |
|                                                                       |
| INV-DR-01: sequence_path, groundtruth_path, calibration_path must all |
| be absolute Paths.                                                    |
|                                                                       |
| INV-DR-02: dataset_hash format: sha256:\<64 hex chars\>.              |
|                                                                       |
| INV-DR-03: camera_rate_hz \<= imu_rate_hz.                            |
|                                                                       |
| INV-DR-04: duration_s \> 0, trajectory_length_m \> 0.                 |
|                                                                       |
| INV-DR-05: Runners may read dataset files via these paths. Runners    |
| must not access GroundTruth objects.                                  |
+-----------------------------------------------------------------------+

**4.4 MetricSet (ICD-08/09)**

Produced by MetricsEngine. Frozen. All gate fields are computed from
metric values --- they cannot be set independently.

  ---------------------------------------------------------------------------
  **Field**           **Type**   **Description**
  ------------------- ---------- --------------------------------------------
  ate_rmse_m          Float \>=  Absolute Trajectory Error RMSE, metres.
                      0          

  ate_mean_m          Float \>=  ATE mean, metres.
                      0          

  ate_max_m           Float \>=  ATE maximum, metres.
                      0          

  rpe_mean_m          Float \>=  Relative Pose Error mean over 1-second
                      0          windows, metres.

  rpe_max_m           Float \>=  RPE maximum, metres.
                      0          

  drift_m_per_km      Float \>=  Mean drift per km. Gate threshold: \<= 50.0
                      0          m/km.

  update_rate_hz      Float \>=  Mean pose output rate. Gate threshold: \>=
                      0          20.0 Hz.

  latency_mean_ms     Float \>=  Mean frame-to-pose latency. Gate threshold:
                      0          \<= 100.0 ms.

  latency_p95_ms      Float \>=  95th-percentile latency.
                      0          

  latency_max_ms      Float \>=  Maximum latency.
                      0          

  tracking_loss_pct   Float \>=  Loss frames / total frames. Gate threshold:
                      0          \< 5.0%.

  acceptance_pass     Bool       True iff all four gate fields are True
                                 simultaneously.
  ---------------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **MetricSet Invariants**                                              |
|                                                                       |
| INV-MS-01: gate_drift = (drift_m_per_km \<= 50.0). Enforced at        |
| construction.                                                         |
|                                                                       |
| INV-MS-02: gate_update_rate = (update_rate_hz \>= 20.0). Enforced at  |
| construction.                                                         |
|                                                                       |
| INV-MS-03: gate_latency = (latency_mean_ms \<= 100.0). Enforced at    |
| construction.                                                         |
|                                                                       |
| INV-MS-04: gate_tracking = (tracking_loss_pct \< 5.0). Enforced at    |
| construction.                                                         |
|                                                                       |
| INV-MS-05: acceptance_pass = gate_drift AND gate_update_rate AND      |
| gate_latency AND gate_tracking.                                       |
|                                                                       |
| INV-MS-06: MetricSet.compute_gates_and_build() is the preferred       |
| constructor. Direct construction with inconsistent gates raises       |
| ValueError.                                                           |
+-----------------------------------------------------------------------+

**4.5 ExperimentRecord (ICD-11)**

+-----------------------------------------------------------------------+
| **ExperimentRecord Invariants**                                       |
|                                                                       |
| INV-ER-01: experiment_id format:                                      |
| EXP\_{TYPE}\_{NNN}\_{YYYYMMDD}\_{HHMMSS}.                             |
|                                                                       |
| INV-ER-02: metadata_path and metrics_path must be relative paths.     |
|                                                                       |
| INV-ER-03: acceptance_pass must be False when status != \'complete\'. |
|                                                                       |
| INV-ER-04: status must be one of: complete \| failed \| dry_run.      |
|                                                                       |
| INV-ER-05: duration_s \>= 0.0.                                        |
+-----------------------------------------------------------------------+

**5 Runner Contract**

**5.1 BaseRunner Interface**

All algorithm runners must inherit from BaseRunner and implement
stream(). The constructor signature is fixed --- ExperimentRunner
instantiates runners by class reference from the registry and always
passes the same three arguments.

> BaseRunner.\_\_init\_\_(config: ExperimentConfig, dataset_ref:
> DatasetRef, logger: PlatformLogger)
>
> BaseRunner.stream() -\> Iterator\[PoseEstimate\]
>
> BaseRunner.metadata() -\> Dict\[str, Any\] \# optional, default
> returns {}
>
> BaseRunner.close() -\> None \# optional, default is no-op

**5.2 Runner Lifecycle**

ExperimentRunner drives the following lifecycle. Each step is mandatory
in this order.

  -------------------------------------------------------------------------
  **Step**       **Call**               **Requirement**
  -------------- ---------------------- -----------------------------------
  1\.            runner =               Runner initialises internal state.
  Construction   RunnerClass(config,    File I/O for configuration is
                 dataset_ref, logger)   permitted.

  2\. Streaming  for pose in            Runner emits PoseEstimate objects
                 runner.stream():       one at a time. Poses must be in
                 engine.update(pose)    strictly increasing timestamp
                                        order.

  3\. Metadata   meta =                 Called after stream() exhausts.
                 runner.metadata()      Must not be called after close().
                                        Returns JSON-serialisable dict.

  4\. Close      runner.close() \# in   Guaranteed call. Releases
                 finally block          subprocess handles, file
                                        descriptors, GPU contexts. Must be
                                        idempotent.
  -------------------------------------------------------------------------

**5.3 Runner Obligations**

-   Populate frame_timestamp_s from the dataset image timestamp (not
    system wall-clock).

-   Populate timestamp_s from the algorithm output timestamp (dataset
    clock or ROS2 message timestamp).

-   Ensure timestamp_s \>= frame_timestamp_s for every PoseEstimate.

-   Ensure timestamp_s is strictly increasing across the stream.

-   Log all algorithm stdout/stderr through
    self.\_log.algorithm_output(), not directly to stdout.

**5.4 Runner Forbidden Dependencies**

-   Must not import MetricsEngine, GroundTruth, or MetricSet.

-   Must not import ResultsArchive or ExperimentRunner.

-   Must not import ConfigurationManager or ExperimentPlan.

-   Must not compute evaluation metrics internally.

-   Must not write files to the results/ directory.

**5.5 Runner Registration**

Runners are registered in ExperimentRunner.\_RUNNER_REGISTRY as a dict
mapping (navigation_type, algorithm) tuples to runner classes.
Registration is performed at module level in experiment_runner.py via
\_register_runners(). Adding a new algorithm requires: implementing a
BaseRunner subclass, adding it to VALID_ALGORITHMS in
interfaces/constants.py, adding it to schema.yaml allowed list, and
adding a registration entry in \_register_runners().

+-----------------------------------------------------------------------+
| **Dispatch Rule**                                                     |
|                                                                       |
| config.dry_run == True -\> SyntheticRunner is always used, registry   |
| is ignored.                                                           |
|                                                                       |
| config.dry_run == False -\> (navigation_type, algorithm) looked up in |
| registry.                                                             |
|                                                                       |
| Missing entry -\> RunnerDispatchError is raised and propagated.       |
|                                                                       |
| Placeholder DatasetRef must not reach a real runner.                  |
+-----------------------------------------------------------------------+

**5.6 Runner Timestamp Ownership**

Strictly increasing timestamp ordering is the runner\'s responsibility,
not the evaluation engine\'s. This rule applies without exception.

-   The runner must produce PoseEstimate objects with strictly
    increasing timestamp_s across the entire stream.

-   MetricsEngine verifies this property via TimestampRegressionError
    but must never attempt to correct, reorder, or buffer poses in
    response.

-   If a pose arrives with a timestamp not strictly greater than the
    previous pose timestamp, TimestampRegressionError is raised and the
    experiment terminates as a runner failure.

-   The platform treats timestamp ordering violations as a runner
    contract violation, not an evaluation problem. The runner is the
    source of authority on pose timing.

+-----------------------------------------------------------------------+
| **Timestamp Ownership Rule**                                          |
|                                                                       |
| The runner owns timestamp ordering. MetricsEngine owns timestamp      |
| verification.                                                         |
|                                                                       |
| These responsibilities must never be reversed or merged.              |
|                                                                       |
| A runner that produces out-of-order timestamps has a defect.          |
|                                                                       |
| MetricsEngine that silently accepts or reorders out-of-order          |
| timestamps has a defect.                                              |
+-----------------------------------------------------------------------+

**6 Evaluation Contract**

**6.1 MetricsEngine Assumptions**

MetricsEngine makes the following assumptions about the pose stream it
receives. Violation of these assumptions produces incorrect metrics or
raises exceptions.

  -----------------------------------------------------------------------
  **Assumption**   **Specification**
  ---------------- ------------------------------------------------------
  Timestamp        Poses must arrive in strictly increasing timestamp_s
  ordering         order. TimestampRegressionError is raised on
                   violation. See §5.6 for ownership rules.

  begin() before   MetricsEngine.begin() must be called before the first
  update()         update(). Calling begin() resets all state, including
                   the timestamp guard, so previous experiments cannot
                   affect subsequent runs.

  Empty streams    If finalise() is called with zero poses received,
                   EmptyStreamError is raised.

  Latency domain   Latency is defined as: latency = (timestamp_s −
                   frame_timestamp_s) \* 1000 ms. The valid domain is
                   latency \>= 0. Negative latency is invalid and is
                   rejected by PoseEstimate construction before reaching
                   MetricsEngine. MetricsEngine may defensively clamp
                   floating-point noise around zero, but this must not
                   mask real violations. Any negative value that passes
                   PoseEstimate construction is a platform defect.

  Minimum          ATE and RPE trajectory alignment requires sufficient
  trajectory       spatial displacement and a minimum number of
  condition        associated pose pairs. Trajectories with fewer than 3
                   associated poses after temporal matching cannot be
                   aligned and will cause alignment to fall back to the
                   nearest-neighbour method. Degenerate trajectories
                   (all-zero positions, stationary runs) may cause
                   alignment algorithms to produce undefined results.
                   Such conditions must be treated as experiment failures
                   rather than valid metric values.

  ATE alignment    ATE is computed with SE3 Umeyama alignment via the evo
                   library. The fallback (nearest-neighbour, no
                   alignment) is used if evo fails or the trajectory is
                   degenerate.

  Drift metric     drift_m_per_km uses absolute error at km boundaries.
                   For sequences shorter than 1 km, the value is
                   approximate and not directly comparable to standard
                   benchmark figures.

  Update rate      Computed as (N-1) / (t_last - t_first) --- span-based
                   to prevent float64 accumulation error.

  Profiling        CPU/memory profiling runs in a background thread at 1
                   Hz. GPU profiling runs if pynvml is available.
  -----------------------------------------------------------------------

**6.2 Metric Definitions**

  ------------------------------------------------------------------------------
  **Metric**          **Unit**   **Definition**
  ------------------- ---------- -----------------------------------------------
  ATE RMSE            metres     Root-mean-square of position error after SE3
                                 alignment over full sequence.

  ATE mean            metres     Mean position error after alignment.

  RPE mean            metres     Mean relative pose error over 1-second
                                 trajectory segments.

  drift_m_per_km      m/km       Mean absolute position error at 1 km
                                 boundaries. See note on sub-km sequences.

  update_rate_hz      Hz         Span-based: (N-1) / (t_last - t_first).

  latency_mean_ms     ms         Mean of (timestamp_s - frame_timestamp_s) \*
                                 1000, clamped to 0.

  latency_p95_ms      ms         95th percentile of clamped latency values.

  tracking_loss_pct   \%         Frames with tracking_valid=False or
                                 feature_count \< 20 / total frames.

  ttfp_s              seconds    Time from first frame_timestamp_s to first pose
                                 with tracking_valid=True.
  ------------------------------------------------------------------------------

**7 Dataset Contract**

**7.1 DatasetManager Guarantees**

DatasetManager guarantees the following properties of the GroundTruth
object it produces. MetricsEngine may rely on these without
re-validating.

  -----------------------------------------------------------------------
  **Guarantee**    **Specification**
  ---------------- ------------------------------------------------------
  Sorted           timestamps_s is strictly increasing. Enforced after
  timestamps       CSV load via sort and monotonicity guard.

  Normalised       All orientations_wxyz entries have unit norm. Enforced
  quaternions      after load.

  Known path       total_length_m \> 0. Enforced by GroundTruth
  length           constructor.

  Validated file   groundtruth_path and image_timestamps_s are populated
  paths            from verified on-disk files.

  Skip-ratio guard If more than 10% of CSV rows are skipped during
                   parsing, DatasetLoadError is raised.

  Duplicate guard  If duplicate timestamps remain after sorting,
                   DatasetLoadError is raised.

  Sequence name    For dry-run GroundTruth, sequence_name is prefixed
                   with \'synthetic:\' to distinguish from real dataset
                   runs in archived metadata.
  -----------------------------------------------------------------------

**7.2 DatasetRef Construction (Phase-0 Status)**

In Phase-0, DatasetRef is constructed by ExperimentRunner using a
placeholder for dry-run experiments. For real runs (dry_run=False),
DatasetRef construction raises NotImplementedError because
DatasetManager does not yet return DatasetRef alongside GroundTruth.
This is the documented Phase-1 integration step.

+-----------------------------------------------------------------------+
| **Phase-1 Integration Step**                                          |
|                                                                       |
| DatasetManager.build_dataset_ref(dataset_name) must be implemented.   |
|                                                                       |
| It should return a DatasetRef whose groundtruth_path points to the    |
| verified on-disk CSV.                                                 |
|                                                                       |
| ExperimentRunner.\_run_from_config() step 5 must be updated to call   |
| it for real runs.                                                     |
|                                                                       |
| Until this step is complete, real runners cannot receive a valid      |
| DatasetRef.                                                           |
+-----------------------------------------------------------------------+

**8 Error Classification and Handling**

**8.1 Three Error Categories**

  ---------------------------------------------------------------------------
  **Category**   **Exception Types**    **Handling**
  -------------- ---------------------- -------------------------------------
  Integration    RunnerDispatchError,   Configuration or integration is
  Error          NotImplementedError    incomplete. Missing registry entry,
                                        missing dataset on disk, or Phase-1
                                        DatasetRef step not implemented.
                                        Propagated to caller without
                                        archiving. Experiment ID has been
                                        reserved but no record is committed.

  Runner Failure EmptyStreamError,      Runner was dispatched but produced no
                 RunnerExecutionError   output or terminated abnormally.
                                        ExperimentRunner catches these and
                                        commits a status=failed
                                        ExperimentRecord with failure_cause
                                        in metadata.

  Platform Error Any other Exception    Unexpected failure inside a platform
                                        block (MetricsEngine, ResultsArchive,
                                        logger). ExperimentRunner catches
                                        these and commits a status=failed
                                        ExperimentRecord. Error details are
                                        in the experiment log.
  ---------------------------------------------------------------------------

**8.2 ExperimentRunner Error Handling Logic**

The exception handling in \_run_from_config() follows this decision
tree:

-   RunnerDispatchError or NotImplementedError: re-raise to caller. No
    archive record is written. ID has been reserved.

-   EmptyStreamError: catch, log detailed error, write status=failed
    record with failure_cause=\'empty_stream\' in metadata_extra.

-   Any other Exception: catch, log, write status=failed record with
    failure_cause omitted. Exception detail is in the experiment log
    file.

-   In all cases: runner.close() is called in a finally block regardless
    of which path is taken.

-   In all cases: the experiment logger is flushed and log bytes are
    collected before the archive commit.

**8.3 Experiment Validity Classification**

Every experiment committed to the archive must be classified as either
VALID or INVALID. This classification is independent of acceptance
gates, which measure performance, not execution validity.

  -----------------------------------------------------------------------------
  **Classification**   **Definition**
  -------------------- --------------------------------------------------------
  VALID                The experiment executed successfully and produced a
                       complete MetricSet. status = \'complete\'. The MetricSet
                       is a meaningful benchmark result regardless of whether
                       acceptance gates pass or fail. A VALID experiment with
                       poor performance (e.g., ATE too large, drift above
                       threshold) is still a legitimate benchmark measurement.

  INVALID              Execution failed before producing a complete MetricSet.
                       status = \'failed\'. Causes include: integration errors,
                       runner failures, dataset errors, degenerate
                       trajectories, or unexpected platform exceptions. INVALID
                       experiments must not be interpreted as benchmark
                       results. They are execution records only.
  -----------------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **Validity vs. Performance --- Critical Distinction**                 |
|                                                                       |
| Acceptance gates (drift_m_per_km \<= 50, update_rate_hz \>= 20,       |
| latency_mean_ms \<= 100,                                              |
|                                                                       |
| tracking_loss_pct \< 5) measure algorithm PERFORMANCE.                |
|                                                                       |
| They do not determine experiment VALIDITY.                            |
|                                                                       |
| Example: ATE = 200 m → VALID experiment, poor performance.            |
|                                                                       |
| Example: Runner produced zero poses → INVALID experiment, not a       |
| benchmark result.                                                     |
|                                                                       |
| INVALID experiments are archived for traceability but must be         |
| excluded from any                                                     |
|                                                                       |
| performance comparison, ranking, or acceptance decision.              |
+-----------------------------------------------------------------------+

**8.4 Archive Immutability Boundary**

The archive immutability boundary is defined precisely by the successful
completion of ResultsArchive.commit(). Once commit() returns without
error:

-   The experiment record (ExperimentRecord) is immutable.

-   The associated metrics (metrics.json) are immutable.

-   The associated metadata (metadata.yaml) is immutable.

-   The archived logs and plots are immutable.

-   No subsequent process, script, or platform operation may modify,
    overwrite, or delete any file within a committed experiment
    directory.

This boundary makes the archive function as a benchmark ledger. Archived
results represent the state of the algorithm and platform at the time of
the experiment run. Re-running an experiment always produces a new
experiment ID and a new archive record --- it does not modify the
existing one.

+-----------------------------------------------------------------------+
| **Archive Immutability Rule**                                         |
|                                                                       |
| A committed experiment directory is a permanent, read-only record.    |
|                                                                       |
| Modifying archived records invalidates the reproducibility guarantee  |
| of the platform.                                                      |
|                                                                       |
| If a result must be corrected, a new experiment must be run.          |
|                                                                       |
| The original flawed result is retained in the archive with its        |
| original ID.                                                          |
+-----------------------------------------------------------------------+

**9 Determinism and Reproducibility**

  -----------------------------------------------------------------------
  **Property**        **Specification**
  ------------------- ---------------------------------------------------
  Experiment ID       Each experiment ID is globally unique.
  uniqueness          ResultsArchive.next_experiment_id() increments a
                      persistent counter under results/counter.json. The
                      counter is shared between run() and execute() entry
                      points.

  ID format           EXP\_{NAV_CODE}\_{NNN}\_{YYYYMMDD}\_{HHMMSS}. NNN
                      is zero-padded counter. Date and time are UTC.

  Write-once archives ResultsArchive refuses to write to an existing
                      experiment directory. Re-runs must produce a new
                      ID. Stale .tmp directories from interrupted runs
                      are cleaned on the next commit.

  Archive             Once commit() completes, the archived record is
  immutability        permanent. No post-commit modification of any
                      archived file is permitted. See §8.4.

  Parameters hash     parameters_hash (SHA-256 of parameters file) is
                      stored in every experiment metadata. Identical hash
                      means identical algorithm configuration.

  Atomic commits      Archives are written to EXP_ID.tmp then renamed to
                      EXP_ID. Any directory in results/ without a .tmp
                      suffix is complete and trustworthy.

  Replay runner       GroundTruthReplayRunner produces an identical pose
  determinism         stream when given the same parameters file and the
                      same RNG seed. Seed is stored in parameters YAML,
                      not in ExperimentConfig.

  Dataset hash        dataset_hash in DatasetRef enables future
                      verification that the dataset on disk has not
                      changed between runs.
  -----------------------------------------------------------------------

**10 GroundTruthReplayRunner Specification**

**10.1 Purpose**

GroundTruthReplayRunner is an infrastructure validation runner. It
replays dataset ground truth with controlled noise, latency, and dropout
to validate the complete evaluation pipeline under analytically
predictable conditions. It is not a navigation algorithm and must never
appear in real algorithm evaluation archives unless explicitly marked as
infrastructure validation.

Registration: ExperimentRunner.\_RUNNER_REGISTRY\[(\"vio\",
\"groundtruth_replay\")\]. Algorithm identifier: groundtruth_replay.

**10.2 Data Access**

-   Reads ground truth from DatasetRef.groundtruth_path using
    DatasetManager.\_read_euroc_groundtruth().

-   Does not access GroundTruth objects. Does not import from
    evaluation/.

-   All replay behaviour is controlled by the parameters YAML at
    ExperimentConfig.parameters_path.

**10.3 Parameters File**

All parameters must be present in the YAML file. Default values are
applied for missing fields.

  -----------------------------------------------------------------------------------
  **Parameter**                  **Type**   **Description**
  ------------------------------ ---------- -----------------------------------------
  noise.position_stddev_m        Float \>=  Gaussian std dev added to x, y, z
                                 0          independently, metres. 0.0 = no noise.

  noise.orientation_stddev_rad   Float \>=  Std dev of small-angle rotation noise,
                                 0          radians. 0.0 = no noise.

  latency.mean_ms                Float \>=  Mean algorithm latency added to each
                                 0          frame, milliseconds.

  latency.stddev_ms              Float \>=  Std dev of latency jitter, milliseconds.
                                 0          

  dropout.rate                   Float in   Fraction of frames randomly dropped.
                                 \[0.0,     
                                 1.0)       

  seed                           Integer    RNG seed. Identical seed + identical file
                                            = identical pose stream.
  -----------------------------------------------------------------------------------

**10.4 Pose Generation Behaviour**

For each ground truth frame at timestamp t:

-   frame_timestamp_s = t (dataset clock timestamp).

-   latency_s = max(\|N(mean, stddev)\|, 0.1 ms). Always positive ---
    ICD constraint enforced by construction.

-   timestamp_s = frame_timestamp_s + latency_s.

-   position = gt_position + N(0, position_stddev_m) per axis.

-   orientation = gt_quaternion perturbed by small-angle rotation noise
    of stddev orientation_stddev_rad.

-   If dropout.rate \> 0 and uniform random sample \< dropout.rate,
    frame is skipped.

-   Covariance diagonal = \[position_var, position_var, position_var,
    orientation_var, orientation_var, orientation_var\] where var =
    max(stddev\^2, 1e-6). Never identity, never all-zeros.

**10.5 Timestamp Ordering Guarantee**

timestamp_s is strictly increasing because: (1) ground truth timestamps
are sorted and strictly increasing after DatasetManager loading; (2)
latency_s \>= 0.1 ms \> 0 on every frame; therefore timestamp_s\[i+1\]
\> timestamp_s\[i\] always holds. A monotonicity guard inside stream()
provides a second line of defence.

**10.6 Validation Invariants**

The following relationships must hold when the platform pipeline is
correct. Deviations are platform bugs, not algorithm behaviours.

  ---------------------------------------------------------------------------------
  **Condition**             **Expected        **Notes**
                            Result**          
  ------------------------- ----------------- -------------------------------------
  noise.position_stddev_m = ATE RMSE ≈ 0      Any residual is from timestamp
  0                                           association, not position error.

  noise.position_stddev_m = ATE RMSE ≈ sigma  Isotropic Gaussian noise.
  sigma                                       Approximation holds over large
                                              samples.

  latency.mean_ms = L,      latency_mean_ms = Zero jitter means deterministic
  stddev = 0                L exactly         latency on every frame.

  dropout.rate = 0          All GT frames     No frames dropped, all poses emitted.
                            yielded           

  dropout.rate = d          \~(1-d) \* N      Approximate --- exact count depends
                            frames yielded    on RNG seed.

  dropout.rate = 0, noise = All acceptance    Full platform acceptance: drift,
  0                         gates pass        update rate, latency, tracking.
  ---------------------------------------------------------------------------------

**11 Non-Goals of Phase-0**

**11.1 Explicitly Out of Scope**

The following capabilities are not part of Phase-0 and must not be
assumed by any component that claims compliance with this specification.

  ------------------------------------------------------------------------------
  **Non-Goal**          **Specification**
  --------------------- --------------------------------------------------------
  Real VIO integration  No real navigation algorithm (OpenVINS, ORB-SLAM3,
                        VINS-Fusion, Kimera) is integrated. Only
                        GroundTruthReplayRunner and SyntheticRunner exist.

  DatasetRef from       DatasetManager does not yet return DatasetRef. Real runs
  DatasetManager        raise NotImplementedError at step 5. This is a Phase-1
                        integration step.

  Dataset downloading   Datasets must be present on disk before any experiment
                        runs. The platform provides no download capability.

  VisualizationModule   Plot generation is out of scope. plot_files in
  (B-06)                ExperimentRecord is always an empty tuple in Phase-0.
                        ResultsArchive creates plots/ directory but it is empty.

  Distributed           Experiments run sequentially. No parallelism, job
  scheduling            queuing, or distributed compute support.

  KITTI and TartanAir   DatasetManager is implemented for EuRoC ASL format only.
  loaders               KITTI and TartanAir sources raise NotImplementedError.

  ROS2 runtime          No ROS2 node management. Runners in Phase-0 operate
                        without a live ROS2 runtime.

  Real-time evaluation  All evaluation is offline. The platform cannot be used
                        as a live navigation monitor.

  GPU-accelerated       Metric computation uses CPU numpy. GPU profiling is
  metrics               read-only (psutil/pynvml).

  Dataset hash          dataset_hash in DatasetRef is populated but not verified
  verification          against current on-disk files in Phase-0.
  ------------------------------------------------------------------------------

**11.2 Phase-1 Integration Steps**

The following steps are required before the first real algorithm runner
can be integrated. Each item corresponds to a specific code change
location.

-   Implement DatasetManager.build_dataset_ref(dataset_name) returning a
    DatasetRef with real on-disk paths.

-   Update ExperimentRunner.\_run_from_config() step 5 to call
    DatasetManager.build_dataset_ref() for dry_run=False experiments.

-   Implement the first real runner (OpenVINS or equivalent) under
    runners/vio/, inheriting BaseRunner.

-   Register the runner in ExperimentRunner.\_RUNNER_REGISTRY.

-   Add the algorithm identifier to VALID_ALGORITHMS in
    interfaces/constants.py.

-   Add the algorithm identifier to schema.yaml allowed list.

-   Run GroundTruthReplayRunner validation suite and confirm all
    invariants hold before connecting the real runner.

*End of NEP Platform Contract Specification --- Version 1.1 --- Phase-0
Baseline (Frozen)*
