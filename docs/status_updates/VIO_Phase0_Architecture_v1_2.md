MicroMind Navigation Programme

**VIO Sandbox**

**Phase-0 Architecture Definition**

*Navigation Evaluation Platform --- Blueprint, Block Definitions & ICD*

  ------------- ---------------------------------------------------------
  **Status**    Phase-0 --- Architecture Definition. Implementation gated
                on this review.

  ------------- ---------------------------------------------------------

  ------------- ---------------------------------------------------------
  **Version**   1.1 \| March 2026 \| Incorporates 12-item review feedback

  ------------- ---------------------------------------------------------

  --------------- ---------------------------------------------------------
  **Programme**   MicroMind / NanoCorteX --- Indigenous GNSS-Denied
                  Navigation, India

  --------------- ---------------------------------------------------------

  ------------- ---------------------------------------------------------
  **Compute**   Phase-0: Azure VM \| Benchmarking: i5-14400F + 32 GB
                DDR5 + RTX 5060 Ti

  ------------- ---------------------------------------------------------

  ------------- ---------------------------------------------------------
  **ROS**       ROS2 Humble \| nav_msgs/Odometry with covariance

  ------------- ---------------------------------------------------------

  ------------- ---------------------------------------------------------
  **Camera**    Stereo + IMU \| EuRoC stereo sequences

  ------------- ---------------------------------------------------------

***Implementation begins only after Phase-0 artefacts are reviewed and
confirmed.***

**PART I**

**Navigation Evaluation Platform --- Architecture Blueprint**

**1. Platform Design Philosophy**

The Navigation Evaluation Platform (NEP) is designed around one
governing principle: the evaluation infrastructure must outlast any
single algorithm evaluation campaign. Phase-0 creates a platform, not a
script.

Three architectural decisions flow from this:

-   Navigation-type agnostic. The platform evaluates VIO first. It will
    evaluate TRN variants, sensor fusion configurations, spoof
    detection, and terminal guidance later. No VIO-specific assumption
    is hardcoded into the core platform layers. Algorithm-specific logic
    lives only in algorithm runners.

-   Traceability is structural, not optional. Experiment IDs, git commit
    hashes, parameter snapshots, and metric archives are produced by the
    platform infrastructure. An algorithm runner cannot produce a result
    without a valid experiment record. Traceability cannot be bypassed.

-   Portability is a first-class constraint. All paths are relative. All
    environment dependencies are declared in config. The repository
    moves from Azure VM to the benchmarking workstation without code
    changes.

**2. Block Architecture Overview**

The platform is composed of seven blocks. Each block owns a single
responsibility domain. No two blocks duplicate responsibilities.
Dependencies flow in one direction: orchestration downward, results
upward.

  ----------------------------------------------------------------------------------
  **Block   **Block Name**         **Role Layer**  **Phase-0 Status**
  ID**                                             
  --------- ---------------------- --------------- ---------------------------------
  B-01      ExperimentRunner       Orchestration   Implemented --- core of Phase-0

  B-02      ConfigurationManager   Configuration   Implemented --- schema + registry

  B-03      DatasetManager         Data Pipeline   Stub --- registry only in Phase-0

  B-04      AlgorithmRunner        Execution       Interface-defined --- no runners
                                                   yet

  B-05      MetricsEngine          Analysis        Implemented --- metric pipeline

  B-06      VisualizationModule    Reporting       Implemented --- plot generators

  B-07      ResultsArchive         Storage         Implemented --- immutable archive

  ---       interfaces/ (shared    Cross-cutting   Implemented --- ICD structures in
            type library)                          code (v1.1 addition)

  ---       platform/logger.py     Cross-cutting   Implemented --- structured
            (shared utility)                       logging for all blocks (v1.1
                                                   addition)
  ----------------------------------------------------------------------------------

**3. Block Dependency Graph**

ExperimentRunner is the orchestration root. It calls all other blocks
--- no block calls ExperimentRunner. The graph below shows control flow
(solid lines: ExperimentRunner invokes) and data flow (dashed lines:
data streams between blocks). DatasetManager and AlgorithmRunner are
invoked sequentially, not concurrently.

*v1.1 correction: The previous diagram implied DatasetManager and
ConfigurationManager push data to ExperimentRunner. The corrected
diagram shows ExperimentRunner pulling from all blocks. Control flows
outward from the root; results flow back.*

+-----------------------------------------------------------------------+
| ExperimentRunner (orchestration root)                                 |
|                                                                       |
| / \| \| \| \| \\                                                      |
|                                                                       |
| invokes ────────/ \| \| \| \| \\──────── invokes                      |
|                                                                       |
| / \| \| \| \| \\                                                      |
|                                                                       |
| ConfigMgr DatasetMgr \| \| \| \| ResultsArchive                       |
|                                                                       |
| \| \| \|                                                              |
|                                                                       |
| AlgorithmRunner \| \|                                                 |
|                                                                       |
| \| \| \|                                                              |
|                                                                       |
| PoseStream ──────► MetricsEngine                                      |
|                                                                       |
| PoseStream ──────► VisualizationModule                                |
|                                                                       |
| MetricSet ──────────► ExperimentRunner                                |
|                                                                       |
| PlotFiles ──────────► ExperimentRunner                                |
|                                                                       |
| Invariant: No block imports or calls ExperimentRunner.                |
|                                                                       |
| Invariant: AlgorithmRunner does not import evaluation/ or             |
| visualization/.                                                       |
|                                                                       |
| Invariant: Only ResultsArchive creates directories under results/.    |
|                                                                       |
| Orchestration sequence:                                               |
|                                                                       |
| Step 1 ExperimentRunner calls ConfigurationManager.load(config)       |
|                                                                       |
| Step 2 ExperimentRunner calls DatasetManager.resolve(dataset_name)    |
|                                                                       |
| Step 3 ExperimentRunner calls AlgorithmRunner.run(sequence, params)   |
|                                                                       |
| Step 4 AlgorithmRunner emits PoseStream → MetricsEngine +             |
| VisualizationModule                                                   |
|                                                                       |
| Step 5 ExperimentRunner collects MetricSet + PlotFiles                |
|                                                                       |
| Step 6 ExperimentRunner calls ResultsArchive.commit(ExperimentRecord) |
+-----------------------------------------------------------------------+

**4. Directory Structure**

The directory structure is the physical expression of block boundaries.
Each top-level directory maps to exactly one block. Shared utilities
live in platform/. No cross-directory imports except through defined
interfaces.

+-----------------------------------------------------------------------+
| sandbox_vio/ \# Repository root                                       |
|                                                                       |
| platform/ \# B-01: ExperimentRunner core                              |
|                                                                       |
| experiment_runner.py \# Orchestration engine                          |
|                                                                       |
| experiment_id.py \# EXP_NAV_NNN_YYYYMMDD_HHMMSS generator             |
|                                                                       |
| git_capture.py \# git commit hash capture at runtime                  |
|                                                                       |
| environment_capture.py \# OS, Python, ROS2, CPU, GPU fingerprint      |
|                                                                       |
| logger.py \# Structured platform logger (v1.1) --- all blocks log     |
| through this                                                          |
|                                                                       |
| config/ \# B-02: ConfigurationManager                                 |
|                                                                       |
| schema.yaml \# Metadata schema (validated on every run)               |
|                                                                       |
| platform_config.yaml \# Global paths and environment settings         |
|                                                                       |
| datasets/ \# Dataset registries (registry only, no data)              |
|                                                                       |
| euroc_registry.yaml                                                   |
|                                                                       |
| kitti_registry.yaml                                                   |
|                                                                       |
| tartanair_registry.yaml                                               |
|                                                                       |
| algorithms/ \# Per-algorithm parameter templates                      |
|                                                                       |
| openvins_default.yaml                                                 |
|                                                                       |
| vins_fusion_default.yaml                                              |
|                                                                       |
| orb_slam3_default.yaml                                                |
|                                                                       |
| kimera_default.yaml                                                   |
|                                                                       |
| datasets/ \# B-03: DatasetManager                                     |
|                                                                       |
| dataset_manager.py \# Registry loader + sequence resolver             |
|                                                                       |
| loaders/ \# Format-specific loaders                                   |
|                                                                       |
| euroc_loader.py                                                       |
|                                                                       |
| kitti_loader.py                                                       |
|                                                                       |
| tartanair_loader.py                                                   |
|                                                                       |
| preprocessing/ \# Standardisation (timestamp sync, IMU align)         |
|                                                                       |
| timestamp_sync.py                                                     |
|                                                                       |
| imu_align.py                                                          |
|                                                                       |
| runners/ \# B-04: AlgorithmRunner                                     |
|                                                                       |
| base_runner.py \# Abstract base class (navigation-type agnostic)      |
|                                                                       |
| vio/ \# VIO-specific runners                                          |
|                                                                       |
| openvins_runner.py                                                    |
|                                                                       |
| vins_fusion_runner.py                                                 |
|                                                                       |
| orb_slam3_runner.py                                                   |
|                                                                       |
| kimera_runner.py                                                      |
|                                                                       |
| trn/ \# Future: TRN evaluation runners                                |
|                                                                       |
| fusion/ \# Future: Sensor fusion runners                              |
|                                                                       |
| evaluation/ \# B-05: MetricsEngine                                    |
|                                                                       |
| metrics_engine.py \# Metric orchestrator                              |
|                                                                       |
| metrics/ \# Pluggable metric modules                                  |
|                                                                       |
| trajectory/ \# Navigation accuracy metrics                            |
|                                                                       |
| ate.py \# Absolute Trajectory Error (wraps evo)                       |
|                                                                       |
| rpe.py \# Relative Pose Error                                         |
|                                                                       |
| drift.py \# m/km over distance                                        |
|                                                                       |
| performance/ \# System performance metrics                            |
|                                                                       |
| update_rate.py \# Pose output Hz                                      |
|                                                                       |
| latency.py \# Frame → pose timestamp delta                            |
|                                                                       |
| cpu_profiler.py \# CPU utilisation (psutil)                           |
|                                                                       |
| memory_profiler.py \# RSS memory (psutil)                             |
|                                                                       |
| gpu_profiler.py \# VRAM / utilisation (pynvml)                        |
|                                                                       |
| stability/ \# Tracking stability metrics                              |
|                                                                       |
| tracking_loss.py \# Loss frames / total                               |
|                                                                       |
| ttfp.py \# Time to first pose                                         |
|                                                                       |
| recovery_time.py \# Recovery after tracking loss                      |
|                                                                       |
| visualization/ \# B-06: VisualizationModule                           |
|                                                                       |
| plot_trajectory.py \# Ground truth vs estimate overlay                |
|                                                                       |
| plot_drift.py \# Drift m/km over distance                             |
|                                                                       |
| plot_ate_rpe.py \# Error distribution histograms                      |
|                                                                       |
| plot_performance.py \# CPU / latency / Hz time series                 |
|                                                                       |
| plot_comparison.py \# Multi-candidate comparison (Phase-5)            |
|                                                                       |
| results/ \# B-07: ResultsArchive (write-once)                         |
|                                                                       |
| EXP_NAV_000_20260320_000000/ \# Smoke test (Phase-0 acceptance gate)  |
|                                                                       |
| metadata.yaml                                                         |
|                                                                       |
| metrics.json                                                          |
|                                                                       |
| plots/                                                                |
|                                                                       |
| logs/                                                                 |
|                                                                       |
| interfaces/ \# Shared type library --- ICD definitions in code (v1.1) |
|                                                                       |
| pose_estimate.py \# ICD-05/06 PoseEstimate dataclass                  |
|                                                                       |
| tracking_status.py \# ICD-07 TrackingStatus dataclass                 |
|                                                                       |
| metric_set.py \# ICD-08/09 MetricSet dataclass                        |
|                                                                       |
| experiment_config.py \# ICD-01 ExperimentConfig dataclass             |
|                                                                       |
| experiment_record.py \# ICD-11 ExperimentRecord dataclass             |
|                                                                       |
| dataset_ref.py \# ICD-02 DatasetRef dataclass                         |
|                                                                       |
| reports/ \# Final selection reports (Phase-5)                         |
|                                                                       |
| tests/ \# Unit and smoke tests                                        |
|                                                                       |
| test_experiment_runner.py \# Phase-0 acceptance gate                  |
|                                                                       |
| test_metrics_engine.py \# Metric module unit tests                    |
|                                                                       |
| test_interfaces.py \# P0-12: ICD field name and type verification     |
| (v1.1)                                                                |
|                                                                       |
| fixtures/ \# Synthetic pose data for testing                          |
+-----------------------------------------------------------------------+

**5. Experiment ID Scheme**

Experiment IDs encode navigation type, a zero-padded sequence counter,
and a UTC timestamp. This makes experiments sortable, searchable by
type, and unambiguous.

+-----------------------------------------------------------------------+
| Format: EXP\_{NAV_TYPE}\_{NNN}\_{YYYYMMDD}\_{HHMMSS}                  |
|                                                                       |
| Examples:                                                             |
|                                                                       |
| EXP_VIO_001_20260320_153015 VIO evaluation, experiment 1              |
|                                                                       |
| EXP_VIO_002_20260321_091200 VIO evaluation, experiment 2 (re-run =    |
| new ID)                                                               |
|                                                                       |
| EXP_TRN_001_20260401_110000 Future: TRN evaluation experiment         |
|                                                                       |
| EXP_FUS_001_20260501_140000 Future: Fusion evaluation experiment      |
|                                                                       |
| EXP_NAV_000_20260320_000000 Phase-0 smoke test (synthetic data)       |
|                                                                       |
| Counter: global, not per-type. Managed by ResultsArchive.             |
|                                                                       |
| Uniqueness: guaranteed by timestamp + counter combination.            |
|                                                                       |
| Write-once: ResultsArchive refuses to overwrite an existing ID.       |
+-----------------------------------------------------------------------+

**6. Metadata Schema**

Every experiment produces a metadata.yaml that is the single source of
truth for reproducibility. An experiment is reproducible if and only if
its metadata.yaml contains enough information to recreate the exact
conditions.

+-----------------------------------------------------------------------+
| \# metadata.yaml --- required fields (schema.yaml enforces this)      |
|                                                                       |
| experiment_id: EXP_VIO_001_20260320_153015                            |
|                                                                       |
| navigation_type: vio \# vio \| trn \| fusion \| terminal              |
|                                                                       |
| algorithm: openvins \# exact algorithm identifier                     |
|                                                                       |
| algorithm_version: 2.6.3 \# from git tag or release                   |
|                                                                       |
| dataset: euroc/MH_01_easy                                             |
|                                                                       |
| dataset_hash: sha256:a1b2c3\... \# prevents silent dataset changes    |
|                                                                       |
| parameters_file: config/algorithms/openvins_default.yaml              |
|                                                                       |
| parameters_hash: sha256:d4e5f6\... \# parameters frozen at run time   |
|                                                                       |
| git_commit: 3c37d82 \# repo commit at launch                          |
|                                                                       |
| timestamp_utc: 2026-03-20T15:30:15Z                                   |
|                                                                       |
| platform_version: NEP_1.0 \# v1.1: identifies platform version that   |
| produced this record                                                  |
|                                                                       |
| requirement_refs: \[NAV-01, FR-VIO-01, FR-VIO-02, FR-VIO-03\]         |
|                                                                       |
| environment:                                                          |
|                                                                       |
| os: Ubuntu 22.04.3 LTS                                                |
|                                                                       |
| python: 3.10.12                                                       |
|                                                                       |
| ros2: Humble (22.04)                                                  |
|                                                                       |
| cpu_model: Intel Core i5-14400F                                       |
|                                                                       |
| cpu_cores: 16 (6P + 10E)                                              |
|                                                                       |
| ram_gb: 32                                                            |
|                                                                       |
| gpu_model: NVIDIA RTX 5060 Ti                                         |
|                                                                       |
| gpu_vram_gb: 16                                                       |
|                                                                       |
| cuda_version: 12.x                                                    |
+-----------------------------------------------------------------------+

**7. Future Navigation Types --- Extension Contract**

To add a new navigation type (e.g., TRN re-evaluation, fusion, terminal
guidance), a developer must:

1.  Create a new runner subdirectory under runners/ (e.g.,
    runners/trn/).

2.  Implement one or more runners inheriting from BaseRunner.

3.  Add a dataset registry YAML for any new datasets.

4.  Add algorithm parameter templates under config/algorithms/.

5.  Register the new navigation_type string in schema.yaml.

6.  Write a smoke test using synthetic data in tests/.

Nothing in platform/, evaluation/, visualization/, or results/ requires
modification. The ExperimentRunner, MetricsEngine, and ResultsArchive
are navigation-type agnostic by design.

*This is the architectural contract enforced by Phase-0. Any
implementation that requires modifying platform/ to add a new navigation
type has violated the design.*

**PART II**

**Block Responsibility Definitions**

Seven blocks. Each block owns a single responsibility domain.
Definitions below are the implementation contract.

+-----------+----------------------------------------------------------+
| **Block:  |                                                          |
| B-01      |                                                          |
| Experimen |                                                          |
| tRunner** |                                                          |
+-----------+----------------------------------------------------------+
| **        | Orchestrate the end-to-end execution of a single         |
| Purpose** | experiment. Enforce traceability. Produce the experiment |
|           | record. This is the only block with visibility into all  |
|           | other blocks.                                            |
+-----------+----------------------------------------------------------+
| *         | -   ExperimentConfig (from ConfigurationManager) ---     |
| *Inputs** |     resolved dataset, algorithm, parameters              |
|           |                                                          |
|           | -   DatasetRef (from DatasetManager) --- validated       |
|           |     sequence path and preprocessing spec                 |
|           |                                                          |
|           | -   CLI arguments --- experiment type, algorithm name,   |
|           |     dataset name, dry-run flag                           |
+-----------+----------------------------------------------------------+
| **        | -   ExperimentRecord --- the complete, immutable archive |
| Outputs** |     written to ResultsArchive                            |
|           |                                                          |
|           | -   ExperimentID --- returned on completion for logging  |
|           |                                                          |
|           | -   Exit status --- 0 on success, non-zero with          |
|           |     structured error on failure                          |
+-----------+----------------------------------------------------------+
| **Respon- | -   Generate ExperimentID before any execution begins    |
| sib       |                                                          |
| ilities** | -   Capture git commit hash at launch time via           |
|           |     git_capture.py                                       |
|           |                                                          |
|           | -   Capture environment fingerprint at launch time via   |
|           |     environment_capture.py                               |
|           |                                                          |
|           | -   Resolve dataset path via DatasetManager before       |
|           |     launching runner                                     |
|           |                                                          |
|           | -   Launch the appropriate AlgorithmRunner subprocess    |
|           |                                                          |
|           | -   Collect PoseStream from AlgorithmRunner and route to |
|           |     MetricsEngine and VisualizationModule                |
|           |                                                          |
|           | -   Collect MetricSet from MetricsEngine and PlotFiles   |
|           |     from VisualizationModule                             |
|           |                                                          |
|           | -   Assemble ExperimentRecord and commit to              |
|           |     ResultsArchive                                       |
|           |                                                          |
|           | -   Validate that the written record is complete before  |
|           |     returning success                                    |
|           |                                                          |
|           | -   Refuse to overwrite an existing ExperimentID         |
+-----------+----------------------------------------------------------+
| **Depends | -   ConfigurationManager (B-02)                          |
| on**      |                                                          |
|           | -   DatasetManager (B-03)                                |
|           |                                                          |
|           | -   AlgorithmRunner (B-04)                               |
|           |                                                          |
|           | -   MetricsEngine (B-05)                                 |
|           |                                                          |
|           | -   VisualizationModule (B-06)                           |
|           |                                                          |
|           | -   ResultsArchive (B-07)                                |
+-----------+----------------------------------------------------------+

+-----------+----------------------------------------------------------+
| **Block:  |                                                          |
| B-02      |                                                          |
| Conf      |                                                          |
| iguration |                                                          |
| Manager** |                                                          |
+-----------+----------------------------------------------------------+
| **        | Own all configuration: platform paths, dataset           |
| Purpose** | registries, algorithm parameter templates, and the       |
|           | metadata schema. Validate all configuration at load      |
|           | time. Be the single source of ground truth for what the  |
|           | platform knows about datasets and algorithms.            |
+-----------+----------------------------------------------------------+
| *         | -   YAML files from config/ directory tree               |
| *Inputs** |                                                          |
|           | -   CLI overrides (parameter key=value pairs)            |
|           |                                                          |
|           | -   Environment variable overrides (e.g.,                |
|           |     SANDBOX_VIO_ROOT)                                    |
+-----------+----------------------------------------------------------+
| **        | -   ExperimentConfig --- resolved, validated             |
| Outputs** |     configuration struct passed to ExperimentRunner      |
|           |                                                          |
|           | -   DatasetRegistry --- the set of known datasets and    |
|           |     their sequence lists                                 |
|           |                                                          |
|           | -   AlgorithmParameterSet --- frozen parameter snapshot  |
|           |     for a given algorithm                                |
|           |                                                          |
|           | -   ValidationResult --- pass / fail with error messages |
|           |     on schema violations                                 |
+-----------+----------------------------------------------------------+
| **Respon- | -   Load and parse all YAML files in config/             |
| sib       |                                                          |
| ilities** | -   Validate experiment metadata against schema.yaml     |
|           |     before any experiment runs                           |
|           |                                                          |
|           | -   Resolve all paths relative to platform root          |
|           |     (portability requirement)                            |
|           |                                                          |
|           | -   Hash parameter files at load time and include hash   |
|           |     in ExperimentConfig                                  |
|           |                                                          |
|           | -   Reject unknown algorithm names or dataset references |
|           |     (fail-fast)                                          |
|           |                                                          |
|           | -   Expose no hardcoded paths --- all paths come from    |
|           |     platform_config.yaml                                 |
+-----------+----------------------------------------------------------+
| **Depends | -   None (root configuration block --- no runtime        |
| on**      |     dependencies)                                        |
+-----------+----------------------------------------------------------+

+-----------+----------------------------------------------------------+
| **Block:  |                                                          |
| B-03      |                                                          |
| Dataset   |                                                          |
| Manager** |                                                          |
+-----------+----------------------------------------------------------+
| **        | Own dataset discovery, validation, and preprocessing.    |
| Purpose** | Ensure that every algorithm runner receives data in a    |
|           | standardised format regardless of the source dataset     |
|           | format. Be the firewall between raw dataset formats and  |
|           | the evaluation platform.                                 |
+-----------+----------------------------------------------------------+
| *         | -   DatasetRegistry (from ConfigurationManager) ---      |
| *Inputs** |     known datasets and sequences                         |
|           |                                                          |
|           | -   DatasetRef request (from ExperimentRunner) --- which |
|           |     dataset and sequence to load                         |
|           |                                                          |
|           | -   Raw dataset files on disk --- ROS2 bags, binary      |
|           |     files, CSV ground truth                              |
+-----------+----------------------------------------------------------+
| **        | -   DatasetRef --- validated sequence path, ground truth |
| Outputs** |     path, calibration, IMU spec                          |
|           |                                                          |
|           | -   StandardisedSequence --- preprocessed data ready for |
|           |     AlgorithmRunner consumption                          |
|           |                                                          |
|           | -   DatasetHash --- SHA-256 of sequence files at load    |
|           |     time                                                 |
|           |                                                          |
|           | -   PreprocessingReport --- timestamps aligned, IMU rate |
|           |     confirmed, stereo calibration verified               |
+-----------+----------------------------------------------------------+
| **Respon- | -   Phase-0: Implement registry loader and sequence      |
| sib       |     resolver only --- no actual data loading             |
| ilities** |                                                          |
|           | -   Phase-1+: Implement format-specific loaders          |
|           |     (euroc_loader, kitti_loader, tartanair_loader)       |
|           |                                                          |
|           | -   Validate that requested sequence exists on disk      |
|           |     before reporting success to ExperimentRunner         |
|           |                                                          |
|           | -   Standardise timestamps to float64 seconds from epoch |
|           |     across all dataset formats                           |
|           |                                                          |
|           | -   Synchronise stereo image timestamps with IMU         |
|           |     timestamps (EuRoC-specific requirement)              |
|           |                                                          |
|           | -   Compute and return SHA-256 hash of sequence files    |
|           |                                                          |
|           | -   Never modify raw dataset files --- preprocessing     |
|           |     produces derived artefacts only                      |
|           |                                                          |
|           | -   PHASE-0 SCOPE (v1.1): In Phase-0, DatasetManager     |
|           |     performs only registry loading, sequence path        |
|           |     resolution, and dataset existence validation. Actual |
|           |     data loading and preprocessing are Phase-1           |
|           |     responsibilities.                                    |
+-----------+----------------------------------------------------------+
| **Depends | -   ConfigurationManager (B-02)                          |
| on**      |                                                          |
+-----------+----------------------------------------------------------+

+-----------+----------------------------------------------------------+
| **Block:  |                                                          |
| B-04      |                                                          |
| Algorith  |                                                          |
| mRunner** |                                                          |
+-----------+----------------------------------------------------------+
| **        | Execute a single VIO (or future navigation) algorithm    |
| Purpose** | against a prepared dataset and stream pose estimates to  |
|           | the metrics and visualisation layers. Be the sole block  |
|           | with knowledge of algorithm-specific interfaces. The     |
|           | base class is navigation-type agnostic; concrete runners |
|           | are algorithm-specific.                                  |
+-----------+----------------------------------------------------------+
| *         | -   StandardisedSequence (from DatasetManager) ---       |
| *Inputs** |     preprocessed camera + IMU data                       |
|           |                                                          |
|           | -   AlgorithmParameterSet (from ConfigurationManager)    |
|           |     --- frozen parameter snapshot                        |
|           |                                                          |
|           | -   ExperimentID (from ExperimentRunner) --- for log     |
|           |     file naming                                          |
+-----------+----------------------------------------------------------+
| **        | -   PoseStream --- continuous stream of PoseEstimate     |
| Outputs** |     records (timestamped position + orientation +        |
|           |     covariance)                                          |
|           |                                                          |
|           | -   TrackingStatusStream --- per-frame feature count and |
|           |     tracking status flag                                 |
|           |                                                          |
|           | -   AlgorithmLog --- raw algorithm stdout/stderr,        |
|           |     written to logs/                                     |
|           |                                                          |
|           | -   RunSummary --- total frames processed, tracking loss |
|           |     events, algorithm exit status                        |
+-----------+----------------------------------------------------------+
| **Respon- | -   Implement BaseRunner abstract interface --- all      |
| sib       |     concrete runners must inherit from it                |
| ilities** |                                                          |
|           | -   Launch the algorithm process (ROS2 node or           |
|           |     subprocess) with the provided parameters             |
|           |                                                          |
|           | -   Translate PoseEstimate from algorithm-native format  |
|           |     to platform PoseEstimate interface                   |
|           |                                                          |
|           | -   Emit PoseEstimate at the algorithm\'s native rate    |
|           |     --- do not buffer or downsample                      |
|           |                                                          |
|           | -   Emit TrackingStatus per frame with feature_count and |
|           |     status flag                                          |
|           |                                                          |
|           | -   Write all algorithm stdout/stderr to logs/ without   |
|           |     filtering                                            |
|           |                                                          |
|           | -   Report RunSummary on completion regardless of        |
|           |     success or failure                                   |
|           |                                                          |
|           | -   Never write to results/ directly --- all output goes |
|           |     through ExperimentRunner                             |
|           |                                                          |
|           | -   ISOLATION RULE (v1.1): AlgorithmRunner modules must  |
|           |     not import from evaluation/ or visualization/ ---    |
|           |     any import from those packages is an architectural   |
|           |     violation                                            |
|           |                                                          |
|           | -   Log through platform/logger.py --- do not write raw  |
|           |     output streams directly to files                     |
+-----------+----------------------------------------------------------+
| **Depends | -   DatasetManager (B-03)                                |
| on**      |                                                          |
|           | -   ConfigurationManager (B-02)                          |
|           |                                                          |
|           | -   platform/logger.py (shared utility)                  |
+-----------+----------------------------------------------------------+

+-----------+----------------------------------------------------------+
| **Block:  |                                                          |
| B-05      |                                                          |
| Metric    |                                                          |
| sEngine** |                                                          |
+-----------+----------------------------------------------------------+
| **        | Compute all evaluation metrics from the PoseStream and   |
| Purpose** | TrackingStatusStream. Produce a structured MetricSet.    |
|           | Apply acceptance gate logic. Be the authority on whether |
|           | a candidate passes or fails.                             |
+-----------+----------------------------------------------------------+
| *         | -   PoseStream (from AlgorithmRunner) --- timestamped    |
| *Inputs** |     pose estimates with covariance                       |
|           |                                                          |
|           | -   GroundTruth (from DatasetManager via DatasetRef) --- |
|           |     reference trajectory                                 |
|           |                                                          |
|           | -   TrackingStatusStream (from AlgorithmRunner) ---      |
|           |     per-frame feature count                              |
|           |                                                          |
|           | -   AcceptanceCriteria (from ConfigurationManager) ---   |
|           |     hard gate thresholds                                 |
+-----------+----------------------------------------------------------+
| **        | -   MetricSet --- all computed metrics in a structured   |
| Outputs** |     dict, written to metrics.json                        |
|           |                                                          |
|           | -   AcceptanceReport --- pass/fail status per acceptance |
|           |     criterion                                            |
|           |                                                          |
|           | -   MetricSummary --- human-readable one-line summary    |
|           |     for logs                                             |
+-----------+----------------------------------------------------------+
| **Respon- | -   Compute ATE (RMSE) using the evo library trajectory  |
| sib       |     alignment                                            |
| ilities** |                                                          |
|           | -   Compute RPE over 1-second windows                    |
|           |                                                          |
|           | -   Compute trajectory drift in m/km over each 1 km      |
|           |     segment                                              |
|           |                                                          |
|           | -   Compute update rate as mean pose outputs per second  |
|           |                                                          |
|           | -   Compute per-frame latency as timestamp delta between |
|           |     image and pose                                       |
|           |                                                          |
|           | -   Profile CPU utilisation and RSS memory via psutil    |
|           |     during run                                           |
|           |                                                          |
|           | -   Profile GPU VRAM and utilisation via pynvml if GPU   |
|           |     is available                                         |
|           |                                                          |
|           | -   Compute tracking loss percentage from                |
|           |     TrackingStatusStream                                 |
|           |                                                          |
|           | -   Compute time-to-first-pose from sequence start to    |
|           |     first valid pose                                     |
|           |                                                          |
|           | -   Compute recovery time after each tracking loss event |
|           |                                                          |
|           | -   Apply acceptance gate thresholds and emit pass/fail  |
|           |     per criterion                                        |
|           |                                                          |
|           | -   Never reject data silently --- log all anomalies     |
|           |     (NaN poses, timestamp inversions)                    |
+-----------+----------------------------------------------------------+
| **Depends | -   AlgorithmRunner (B-04) via PoseStream                |
| on**      |                                                          |
|           | -   DatasetManager (B-03) via GroundTruth                |
+-----------+----------------------------------------------------------+

+-----------+----------------------------------------------------------+
| **Block:  |                                                          |
| B-06      |                                                          |
| Vis       |                                                          |
| ualizatio |                                                          |
| nModule** |                                                          |
+-----------+----------------------------------------------------------+
| **        | Produce all plots for an experiment. Write plot files to |
| Purpose** | the experiment results directory. Be stateless ---       |
|           | produce the same plots given the same inputs.            |
+-----------+----------------------------------------------------------+
| *         | -   PoseStream (from AlgorithmRunner) --- estimated      |
| *Inputs** |     trajectory                                           |
|           |                                                          |
|           | -   GroundTruth (from DatasetManager via DatasetRef) --- |
|           |     reference trajectory                                 |
|           |                                                          |
|           | -   MetricSet (from MetricsEngine) --- computed metrics  |
|           |     for annotation                                       |
|           |                                                          |
|           | -   ExperimentID (from ExperimentRunner) --- for plot    |
|           |     titling                                              |
+-----------+----------------------------------------------------------+
| **        | -   PlotFiles --- a set of PNG files written to          |
| Outputs** |     results/{EXP_ID}/plots/                              |
|           |                                                          |
|           | -   PlotManifest --- list of generated plot filenames    |
|           |     for inclusion in ExperimentRecord                    |
+-----------+----------------------------------------------------------+
| **Respon- | -   Generate trajectory_comparison.png --- ground truth  |
| sib       |     (grey) vs estimate (coloured by error magnitude)     |
| ilities** |                                                          |
|           | -   Generate drift_over_distance.png --- m/km over       |
|           |     distance with NAV-01 limit line                      |
|           |                                                          |
|           | -   Generate ate_rpe_distribution.png --- error          |
|           |     histograms                                           |
|           |                                                          |
|           | -   Generate performance_timeseries.png --- CPU%,        |
|           |     latency ms, update Hz over time                      |
|           |                                                          |
|           | -   Annotate all plots with ExperimentID, algorithm      |
|           |     name, dataset, and key metrics                       |
|           |                                                          |
|           | -   Write all plots to plots/ subdirectory only ---      |
|           |     never to a shared output location                    |
|           |                                                          |
|           | -   Produce Phase-5 comparison plots when called with    |
|           |     multiple MetricSets                                  |
+-----------+----------------------------------------------------------+
| **Depends | -   MetricsEngine (B-05) via MetricSet                   |
| on**      |                                                          |
|           | -   DatasetManager (B-03) via GroundTruth                |
+-----------+----------------------------------------------------------+

+-----------+----------------------------------------------------------+
| **Block:  |                                                          |
| B-07      |                                                          |
| Results   |                                                          |
| Archive** |                                                          |
+-----------+----------------------------------------------------------+
| **        | Own the immutable storage of all experiment results.     |
| Purpose** | Enforce write-once semantics. Be the authority on        |
|           | experiment existence and the global experiment counter.  |
+-----------+----------------------------------------------------------+
| *         | -   ExperimentRecord (from ExperimentRunner) ---         |
| *Inputs** |     complete record to be archived                       |
|           |                                                          |
|           | -   ExperimentID --- to verify uniqueness before writing |
+-----------+----------------------------------------------------------+
| **        | -   Committed archive directory ---                      |
| Outputs** |     EXP_NAV_NNN_YYYYMMDD_HHMMSS/ written to results/     |
|           |                                                          |
|           | -   ExistenceCheck result --- boolean: does this ID      |
|           |     already exist?                                       |
|           |                                                          |
|           | -   ExperimentList --- sorted list of all experiment IDs |
|           |     in results/                                          |
|           |                                                          |
|           | -   NextCounter --- next available sequence number       |
+-----------+----------------------------------------------------------+
| **Respon- | -   Refuse to write if the ExperimentID already exists   |
| sib       |     --- raise, do not silently skip                      |
| ilities** |                                                          |
|           | -   Validate ExperimentRecord completeness before        |
|           |     writing (metadata.yaml and metrics.json must be      |
|           |     present)                                             |
|           |                                                          |
|           | -   Write metadata.yaml, metrics.json, plots/, and logs/ |
|           |     atomically (write to temp dir, then rename)          |
|           |                                                          |
|           | -   Maintain a global counter file to assign the next    |
|           |     NNN without collision                                |
|           |                                                          |
|           | -   Provide a read-only query interface to list all      |
|           |     experiments and retrieve records                     |
|           |                                                          |
|           | -   Never delete or modify existing records --- archive  |
|           |     is append-only                                       |
|           |                                                          |
|           | -   OWNERSHIP RULE (v1.1): Only ResultsArchive may       |
|           |     create directories under results/. No other block    |
|           |     may write to or create directories in results/       |
|           |     directly.                                            |
+-----------+----------------------------------------------------------+
| **Depends | -   None (terminal block --- no outgoing dependencies)   |
| on**      |                                                          |
+-----------+----------------------------------------------------------+

**Shared Infrastructure: interfaces/ and platform/logger.py**

The interfaces/ directory and platform/logger.py are cross-cutting
support modules, not runtime blocks. They have no orchestration
responsibilities and no block IDs. Every runtime block imports from
interfaces/; no block defines its own local copy of an ICD data
structure.

**interfaces/ --- ICD Definitions in Code**

The interfaces/ directory is the physical implementation of the ICD.
Each module contains exactly one Python dataclass corresponding to one
ICD entry. Field names, types, and default values must match the ICD
exactly. This is the single authoritative definition of all platform
data structures.

  ----------------------------------------------------------------------------------
  **Module**             **ICD         **Contents**
                         Reference**   
  ---------------------- ------------- ---------------------------------------------
  pose_estimate.py       ICD-05/06     PoseEstimate dataclass --- 14 fields, float64
                                       enforced, ordering constraint documented

  tracking_status.py     ICD-07        TrackingStatus dataclass --- 6 fields,
                                       camera-rate stream

  metric_set.py          ICD-08/09     MetricSet dataclass --- all metric fields
                                       including sequence_length_m and gate booleans

  experiment_config.py   ICD-01        ExperimentConfig dataclass --- resolved
                                       config struct from ConfigurationManager

  dataset_ref.py         ICD-02        DatasetRef dataclass --- validated dataset
                                       reference from DatasetManager

  experiment_record.py   ICD-11        ExperimentRecord dataclass --- complete
                                       archive record passed to ResultsArchive
  ----------------------------------------------------------------------------------

*Rule: If a field exists in interfaces/, it must not be redefined in any
block module. All blocks import the shared type. Duplication is an
architectural violation caught by P0-12.*

**platform/logger.py --- Structured Platform Logger**

All blocks log through platform/logger.py. No block writes raw output
streams directly to files. The logger produces experiment-scoped,
timestamped, structured log entries. This ensures consistent formatting,
easier debugging, and a clean archive structure.

  ---------------------------------------------------------------------------
  **Responsibility**   **Detail**
  -------------------- ------------------------------------------------------
  Experiment scoping   Log file path includes ExperimentID --- all log output
                       for one experiment goes to one file

  Structured format    Each entry: ISO timestamp \| block_id \| level \|
                       message --- machine-parseable

  Level support        DEBUG, INFO, WARNING, ERROR --- consistent across all
                       blocks

  Algorithm            AlgorithmRunner routes algorithm stdout/stderr through
  pass-through         logger at DEBUG level rather than writing raw streams

  No external          logger.py uses Python stdlib logging only --- no
  dependency           third-party logging framework
  ---------------------------------------------------------------------------

**PART III**

**Interface Control Document (ICD)**

This ICD defines every inter-block data structure. These definitions
become the implementation baseline. Field names, types, and units must
not diverge from this document without a formal revision.

  -----------------------------------------------------------------------------------------------------
  **ICD-ID**   **Interface Name**      **Source**             **Destination**       **Cardinality**
  ------------ ----------------------- ---------------------- --------------------- -------------------
  ICD-01       ExperimentConfig        B-02                   B-01 ExperimentRunner 1 per experiment
                                       ConfigurationManager                         launch

  ICD-02       DatasetRef              B-03 DatasetManager    B-01 ExperimentRunner 1 per experiment

  ICD-03       AlgorithmParameterSet   B-02                   B-04 AlgorithmRunner  1 per experiment
                                       ConfigurationManager                         

  ICD-04       StandardisedSequence    B-03 DatasetManager    B-04 AlgorithmRunner  1 per experiment

  ICD-05       PoseEstimate            B-04 AlgorithmRunner   B-05 MetricsEngine    N per experiment
                                                                                    (streaming)

  ICD-06       PoseEstimate            B-04 AlgorithmRunner   B-06                  N per experiment
                                                              VisualizationModule   (streaming)

  ICD-07       TrackingStatus          B-04 AlgorithmRunner   B-05 MetricsEngine    1 per frame

  ICD-08       MetricSet               B-05 MetricsEngine     B-01 ExperimentRunner 1 per experiment

  ICD-09       MetricSet               B-05 MetricsEngine     B-06                  1 per experiment
                                                              VisualizationModule   

  ICD-10       PlotManifest            B-06                   B-01 ExperimentRunner 1 per experiment
                                       VisualizationModule                          

  ICD-11       ExperimentRecord        B-01 ExperimentRunner  B-07 ResultsArchive   1 per experiment
  -----------------------------------------------------------------------------------------------------

**ICD-01 ExperimentConfig**

Resolved, validated configuration struct produced by
ConfigurationManager and consumed by ExperimentRunner at launch.

+-----------------------------------------------------------------------+
| **Interface: ICD-01 ExperimentConfig**                                |
+-----------------------------------------------------------------------+
| **Source:** B-02 ConfigurationManager **→ Destination:** B-01         |
| ExperimentRunner                                                      |
+-----------------------------------------------------------------------+
|   ------                                                              |
| ------------- ------------- ---------- ---------- ------------------- |
|   **                                                                  |
| Field**           **Type**      **Unit**   **Rate**   **Description** |
|                                                                       |
|                                                                       |
|  experiment_id       str           ---        1x launch  Pre-assigned |
|                                                                       |
|                                                     EXP_NAV_NNN\_\... |
|                                                           identifier  |
|                                                                       |
|                                                                       |
| navigation_type     str enum      ---        1x         vio \| trn \| |
|                                                                       |
|                                                    fusion \| terminal |
|                                                                       |
|   algorithm           str           ---        1x         openvins \| |
|                                                                       |
|                                                        vins_fusion \| |
|                                                                       |
|                                                   orb_slam3 \| kimera |
|                                                                       |
|   algo                                                                |
| rithm_version   str           ---        1x         Semver or git tag |
|                                                                       |
|   datas                                                               |
| et_name        str           ---        1x         Registry key, e.g. |
|                                                                       |
|                                                      euroc/MH_01_easy |
|                                                                       |
|   par                                                                 |
| ameters_path     Path          ---        1x         Absolute path to |
|                                                                       |
|                                                        parameter YAML |
|                                                                       |
|   para                                                                |
| meters_hash     str           SHA-256    1x         Hash of parameter |
|                                                                       |
|                                                     file at load time |
|                                                                       |
|   r                                                                   |
| equirement_refs    List\[str\]   ---        1x         e.g. \[NAV-01, |
|                                                           FR-VIO-01\] |
|                                                                       |
|   dry_                                                                |
| run             bool          ---        1x         If true, validate |
|                                                                       |
|                                                       only --- do not |
|                                                                       |
|                                                         launch runner |
|                                                                       |
|   platf                                                               |
| orm_root       Path          ---        1x         Absolute repo root |
|                                                                       |
|                                                   --- all other paths |
|                                                                       |
|                                                      relative to this |
|   ------                                                              |
| ------------- ------------- ---------- ---------- ------------------- |
+-----------------------------------------------------------------------+

**ICD-02 DatasetRef**

Validated dataset reference produced by DatasetManager. Confirms the
sequence exists on disk and provides all metadata the runner and metrics
engine need.

+-----------------------------------------------------------------------+
| **Interface: ICD-02 DatasetRef**                                      |
+-----------------------------------------------------------------------+
| **Source:** B-03 DatasetManager **→ Destination:** B-01               |
| ExperimentRunner                                                      |
+-----------------------------------------------------------------------+
|   -----                                                               |
| ---------------- ------------ ---------- ---------- ----------------- |
|   **F                                                                 |
| ield**             **Type**     **Unit**   **Rate**   **Description** |
|                                                                       |
|                                                                       |
| dataset_name          str          ---        1x         Registry key |
|                                                                       |
|   sequ                                                                |
| ence_path         Path         ---        1x         Absolute path to |
|                                                                       |
|                                                         sequence root |
|                                                                       |
|   grou                                                                |
| ndtruth_path      Path         ---        1x         Absolute path to |
|                                                                       |
|                                                     ground truth file |
|                                                                       |
|                                                                       |
| calibration_path      Path         ---        1x         Camera + IMU |
|                                                                       |
|                                                      calibration file |
|                                                                       |
|                                                                       |
|  dataset_hash          str          SHA-256    1x         Hash of key |
|                                                                       |
|                                                        sequence files |
|                                                                       |
|   imu_r                                                               |
| ate_hz           float        Hz         1x         Declared IMU rate |
|                                                                       |
|                                                          (e.g. 200.0) |
|                                                                       |
|   cam                                                                 |
| era_rate_hz        float        Hz         1x         Declared camera |
|                                                                       |
|                                                      rate (e.g. 20.0) |
|                                                                       |
|   st                                                                  |
| ereo                bool         ---        1x         True if stereo |
|                                                            sequence   |
|                                                                       |
|   durat                                                               |
| ion_s            float        s          1x         Sequence duration |
|                                                            in seconds |
|                                                                       |
|                                                                       |
| trajectory_length_m   float        m          1x         Total ground |
|                                                                       |
|                                                     truth path length |
|   -----                                                               |
| ---------------- ------------ ---------- ---------- ----------------- |
+-----------------------------------------------------------------------+

**ICD-05 / ICD-06 PoseEstimate**

The primary data stream from AlgorithmRunner to MetricsEngine and
VisualizationModule. This is the highest-frequency interface in the
platform --- every pose output from the VIO algorithm flows through it.

+-----------------------------------------------------------------------+
| **Interface: ICD-05/06 PoseEstimate**                                 |
+-----------------------------------------------------------------------+
| **Source:** B-04 AlgorithmRunner **→ Destination:** B-05              |
| MetricsEngine / B-06 VisualizationModule                              |
+-----------------------------------------------------------------------+
|   ------                                                              |
| ------------- --------------- ---------- ---------- ----------------- |
|   **Fi                                                                |
| eld**           **Type**        **Unit**   **Rate**   **Description** |
|                                                                       |
|   timestamp_s         float64         s          ≥20 Hz     UTC epoch |
|                                                                       |
|                                                      seconds. float64 |
|                                                                       |
|                                                         mandatory --- |
|                                                             float32   |
|                                                                       |
|                                                          insufficient |
|                                                             precision |
|                                                                       |
|   frame                                                               |
| _timestamp_s   float64         s          ≥20 Hz     Timestamp of the |
|                                                                       |
|                                                      image frame that |
|                                                                       |
|                                                        triggered this |
|                                                             pose      |
|                                                                       |
|   posit                                                               |
| ion_x          float64         m          ≥20 Hz     Position East in |
|                                                                       |
|                                                      world frame (ENU |
|                                                                       |
|                                                           convention) |
|                                                                       |
|   positi                                                              |
| on_y          float64         m          ≥20 Hz     Position North in |
|                                                                       |
|                                                      world frame (ENU |
|                                                                       |
|                                                           convention) |
|                                                                       |
|   pos                                                                 |
| ition_z          float64         m          ≥20 Hz     Position Up in |
|                                                                       |
|                                                      world frame (ENU |
|                                                                       |
|                                                           convention) |
|                                                                       |
|                                                                       |
| quat_w              float64         ---        ≥20 Hz     Orientation |
|                                                                       |
|                                                        quaternion --- |
|                                                                       |
|                                                           scalar part |
|                                                                       |
|                                                                       |
| quat_x              float64         ---        ≥20 Hz     Orientation |
|                                                                       |
|                                                      quaternion --- i |
|                                                             component |
|                                                                       |
|                                                                       |
| quat_y              float64         ---        ≥20 Hz     Orientation |
|                                                                       |
|                                                      quaternion --- j |
|                                                             component |
|                                                                       |
|                                                                       |
| quat_z              float64         ---        ≥20 Hz     Orientation |
|                                                                       |
|                                                      quaternion --- k |
|                                                             component |
|                                                                       |
|   co                                                                  |
| variance_6x6      float64\[36\]   m²,rad²    ≥20 Hz     Row-major 6x6 |
|                                                                       |
|                                                      pose covariance. |
|                                                                       |
|                                                           Must not be |
|                                                                       |
|                                                     identity --- real |
|                                                                       |
|                                                     estimate required |
|                                                                       |
|   track                                                               |
| ing_valid      bool            ---        ≥20 Hz     False if pose is |
|                                                                       |
|                                                        extrapolated / |
|                                                                       |
|                                                       propagated only |
|                                                                       |
|   featur                                                              |
| e_count       int             ---        ≥20 Hz     Number of tracked |
|                                                                       |
|                                                       visual features |
|                                                                       |
|                                                            this frame |
|                                                                       |
|   source_algorithm    str             ---        1x         Algorithm |
|                                                                       |
|                                                        identifier --- |
|                                                             for       |
|                                                                       |
|                                                       multi-candidate |
|                                                                       |
|                                                      comparison plots |
|   ------                                                              |
| ------------- --------------- ---------- ---------- ----------------- |
+-----------------------------------------------------------------------+

*Convention: All positions in ENU (East-North-Up) world frame. World
frame origin is defined by the dataset ground-truth coordinate frame.
Quaternion in Hamilton convention (w, x, y, z). Covariance must reflect
real estimator uncertainty --- a diagonal identity matrix is a protocol
violation.*

*v1.1 ordering constraint: timestamp_s \>= frame_timestamp_s must always
hold. The pose is estimated after the frame is captured. A violation
indicates a clock synchronisation error or incorrect timestamp
assignment and must be flagged as a protocol error by MetricsEngine.
Clock source: timestamp_s must originate from the dataset clock or ROS2
message timestamp, not from system wall-clock time. System wall-clock
timestamps produce non-reproducible latency measurements across runs.*

**ICD-07 TrackingStatus**

Per-frame tracking health signal, emitted at camera frame rate.

+-----------------------------------------------------------------------+
| **Interface: ICD-07 TrackingStatus**                                  |
+-----------------------------------------------------------------------+
| **Source:** B-04 AlgorithmRunner **→ Destination:** B-05              |
| MetricsEngine                                                         |
+-----------------------------------------------------------------------+
|   --                                                                  |
| ---------------- ------------ ---------- ---------- ----------------- |
|                                                                       |
| **Field**          **Type**     **Unit**   **Rate**   **Description** |
|                                                                       |
|                                                                       |
| timestamp_s        float64      s          Camera     Frame timestamp |
|                                              rate                     |
|                                                                       |
|                                                                       |
| feature_count      int          ---        Camera     Tracked feature |
|                                                                       |
|                                           rate       count this frame |
|                                                                       |
|   tracking_ok        bool         ---        Camera     True if       |
|                                                                       |
|                                          rate       feature_count \>= |
|                                                         20 (FR-VIO    |
|                                                         threshold)    |
|                                                                       |
|   lo                                                                  |
| ss_event         bool         ---        Camera     True on the first |
|                                              rate       frame of a    |
|                                                         tracking loss |
|                                                         event         |
|                                                                       |
|   re                                                                  |
| covery_event     bool         ---        Camera     True on the first |
|                                              rate       frame after   |
|                                                         recovery      |
|                                                                       |
|   ke                                                                  |
| yframe           bool         ---        Camera     True if algorithm |
|                                                                       |
|                                            rate       designated this |
|                                                                       |
|                                                      frame a keyframe |
|   --                                                                  |
| ---------------- ------------ ---------- ---------- ----------------- |
+-----------------------------------------------------------------------+

**ICD-08 / ICD-09 MetricSet**

Complete metric results for one experiment. Written to metrics.json and
passed to VisualizationModule for plot annotation.

+-----------------------------------------------------------------------+
| **Interface: ICD-08/09 MetricSet**                                    |
+-----------------------------------------------------------------------+
| **Source:** B-05 MetricsEngine **→ Destination:** B-01                |
| ExperimentRunner / B-06 VisualizationModule                           |
+-----------------------------------------------------------------------+
|   --------                                                            |
| ----------- ------------- ---------- ---------- --------------------- |
|   **                                                                  |
| Field**           **Type**      **Unit**   **Rate**   **Description** |
|                                                                       |
|   ate_rm                                                              |
| se_m          float64       m          1x         Absolute Trajectory |
|                                                                       |
|                                                   Error --- RMSE over |
|                                                                       |
|                                                         full sequence |
|                                                                       |
|   a                                                                   |
| te_mean_m          float64       m          1x         ATE mean error |
|                                                                       |
|   ate_                                                                |
| max_m           float64       m          1x         ATE maximum error |
|                                                                       |
|   rpe                                                                 |
| _mean_m          float64       m          1x         RPE mean over 1s |
|                                                           windows     |
|                                                                       |
|   rpe_ma                                                              |
| x_m           float64       m          1x         RPE maximum over 1s |
|                                                           windows     |
|                                                                       |
|   drift_m_                                                            |
| per_km      float64       m/km       1x         Mean drift per km --- |
|                                                                       |
|                                                    PRIMARY acceptance |
|                                                           metric      |
|                                                                       |
|   dri                                                                 |
| ft_per_segment   float64\[\]   m/km       1x/km      Per-km drift for |
|                                                                       |
|                                                   drift_over_distance |
|                                                           plot        |
|                                                                       |
|   update_r                                                            |
| ate_hz      float64       Hz         1x         Mean pose output rate |
|                                                                       |
|                                                   --- gate: \>= 20 Hz |
|                                                                       |
|   laten                                                               |
| cy_mean_ms     float64       ms         1x         Mean frame-to-pose |
|                                                                       |
|                                                 latency --- gate: \<= |
|                                                           100 ms      |
|                                                                       |
|   la                                                                  |
| tency_p95_ms      float64       ms         1x         95th percentile |
|                                                           latency     |
|                                                                       |
|   la                                                                  |
| tency_max_ms      float64       ms         1x         Maximum latency |
|                                                           observed    |
|                                                                       |
|   cpu_mea                                                             |
| n_pct        float64       \%         1x         Mean CPU utilisation |
|                                                           during run  |
|                                                                       |
|   cpu_pea                                                             |
| k_pct        float64       \%         1x         Peak CPU utilisation |
|                                                                       |
|   me                                                                  |
| mory_peak_mb      float64       MB         1x         Peak RSS memory |
|                                                                       |
|   gpu_mea                                                             |
| n_pct        float64       \%         1x         Mean GPU utilisation |
|                                                                       |
|                                                         (0 if no GPU) |
|                                                                       |
|   gpu_pe                                                              |
| ak_vram_mb    float64       MB         1x         Peak GPU VRAM usage |
|                                                                       |
|   trackin                                                             |
| g_loss_pct   float64       \%         1x         Tracking loss frames |
|                                                                       |
|                                                  / total --- gate: \< |
|                                                           5%          |
|                                                                       |
|   ttfp_s                                                              |
|               float64       s          1x         Time to first valid |
|                                                           pose        |
|                                                                       |
|   recov                                                               |
| ery_mean_s     float64       s          1x         Mean recovery time |
|                                                                       |
|                                                   after tracking loss |
|                                                           events      |
|                                                                       |
|   seque                                                               |
| nce_length_m   float64       m          1x         v1.1: Total ground |
|                                                                       |
|                                                      truth trajectory |
|                                                           length ---  |
|                                                                       |
|                                                       denominator for |
|                                                                       |
|                                                 drift_m_per_km sanity |
|                                                           check       |
|                                                                       |
|   acceptan                                                            |
| ce_pass     bool          ---        1x         True only if ALL four |
|                                                                       |
|                                                       hard gates pass |
|                                                                       |
|   gate_dri                                                            |
| ft          bool          ---        1x         drift_m_per_km \<= 50 |
|                                                                       |
|   gate_upd                                                            |
| ate_rate    bool          ---        1x         update_rate_hz \>= 20 |
|                                                                       |
|   gate_l                                                              |
| atency        bool          ---        1x         latency_mean_ms \<= |
|                                                           100         |
|                                                                       |
|   gate_tr                                                             |
| acking       bool          ---        1x         tracking_loss_pct \< |
|                                                           5           |
|   --------                                                            |
| ----------- ------------- ---------- ---------- --------------------- |
+-----------------------------------------------------------------------+

**ICD-11 ExperimentRecord**

The complete, immutable record written to ResultsArchive. This is
everything the platform knows about one experiment.

+-----------------------------------------------------------------------+
| **Interface: ICD-11 ExperimentRecord**                                |
+-----------------------------------------------------------------------+
| **Source:** B-01 ExperimentRunner **→ Destination:** B-07             |
| ResultsArchive                                                        |
+-----------------------------------------------------------------------+
|   ------------------ ---                                              |
| ----------------- ---------- ---------- ----------------------------- |
|   **Field*                                                            |
| *          **Type**             **Unit**   **Rate**   **Description** |
|                                                                       |
|   experiment_id      s                                                |
| tr                  ---        1x         EXP_NAV_NNN_YYYYMMDD_HHMMSS |
|                                                                       |
|   metadata                                                            |
|   ExperimentConfig +   ---        1x         Written to metadata.yaml |
|                      environment snapshot                             |
|                                                                       |
|   metric_set                                                          |
|    MetricSet            ---        1x         Written to metrics.json |
|                                                                       |
|   plot_manifest                                                       |
|   List\[Path\]         ---        1x         PNG filenames written to |
|                                                                       |
|                                                                plots/ |
|                                                                       |
|   log_files                                                           |
| List\[Path\]         ---        1x         Algorithm stdout/stderr in |
|                                                                 logs/ |
|                                                                       |
|   status             str                                              |
|  enum             ---        1x         complete \| failed \| dry_run |
|                                                                       |
|   duration_s         fl                                               |
| oat64              s          1x         Wall time from runner launch |
|                                                                       |
|                                                     to archive commit |
|                                                                       |
|   acce                                                                |
| ptance_pass    bool                 ---        1x         Copied from |
|                                                                       |
|                                             MetricSet.acceptance_pass |
|   ------------------ ---                                              |
| ----------------- ---------- ---------- ----------------------------- |
+-----------------------------------------------------------------------+

**ICD Naming and Unit Conventions**

  -------------------------------------------------------------------------
  **Convention**   **Rule**                       **Rationale**
  ---------------- ------------------------------ -------------------------
  Field names      snake_case throughout. No      Consistent grep-ability
                   abbreviations except           across codebase
                   established acronyms (ate,     
                   rpe, imu, enu).                

  Position units   Metres (m) always. Never cm,   Single unit, no
                   mm, or km in data structures.  conversion bugs

  Time units       Seconds (s) as float64.        float64 seconds
                   Milliseconds (ms) only in      sufficient for 0.1 ms
                   human-readable summary fields  precision over 24h
                   (latency_mean_ms).             

  Angle units      Radians in all data            Consistent with ROS2
                   structures. Degrees only in    nav_msgs convention
                   plot labels.                   

  Quaternion order Hamilton convention: (w, x, y, Avoid w-last vs w-first
                   z). Matches ROS2               confusion
                   geometry_msgs/Quaternion.      

  Coordinate frame ENU world frame                Consistent with future
                   (East-North-Up). Matches ROS2  ROS2 integration
                   REP-105.                       

  Boolean gates    gate\_{criterion}: bool ---    Allows partial pass
                   explicit per-criterion gate    analysis and debugging
                   fields.                        

  Covariance       Always 6x6 row-major float64   Preserves orientation
                   array (36 elements). Never 3x3 uncertainty for ESKF
                   position-only.                 fusion

  File paths in    Relative to experiment         Ensures portability when
  records          directory root, not platform   archive is moved
                   root.                          

  SHA-256 hashes   Stored as hex string with      Unambiguous hash
                   sha256: prefix, e.g.           algorithm identification
                   sha256:a1b2c3\...              
  -------------------------------------------------------------------------

**PART IV**

**Phase-0 Acceptance Gate & Engineering Proposals**

**Phase-0 Implementation Checklist**

Phase-0 is complete --- and Phase-1 implementation is unblocked --- when
all items below are checked:

  -----------------------------------------------------------------------------------
  **Item**   **Deliverable**               **Acceptance Evidence**
  ---------- ----------------------------- ------------------------------------------
  P0-01      ExperimentRunner core         Produces EXP_NAV_000 with valid
             (experiment_runner.py)        metadata.yaml from synthetic data

  P0-02      ExperimentID generator        IDs are unique, sortable, and follow
                                           EXP\_{TYPE}\_{NNN}\_{YYYYMMDD}\_{HHMMSS}
                                           format

  P0-03      Git commit capture            Returns current HEAD hash; returns
             (git_capture.py)              \'untracked\' gracefully if not in git
                                           repo

  P0-04      Environment fingerprint       Captures OS, Python, ROS2, CPU, GPU, CUDA
             (environment_capture.py)      --- no hard failures if GPU absent

  P0-05      ConfigurationManager          Loads schema.yaml, validates a valid
                                           config, rejects an invalid config with
                                           error

  P0-06      Dataset registries (YAML only euroc_registry.yaml, kitti_registry.yaml,
             --- no downloads)             tartanair_registry.yaml parseable

  P0-07      DatasetManager stub           Resolves sequence path from registry;
                                           returns error if path not on disk

  P0-08      MetricsEngine with synthetic  Accepts synthetic PoseStream, returns
             data                          valid MetricSet with all fields populated

  P0-09      ResultsArchive                Writes EXP_NAV_000 atomically; refuses to
                                           overwrite on second run

  P0-10      Smoke test                    pytest passes; EXP_NAV_000 directory
             (test_experiment_runner.py)   produced with all required files

  P0-11      Portability verified          Repository path changes do not break any
                                           test (all paths relative to platform_root)

  P0-12      Interface validation (v1.1    All ICD data structures implemented in
             addition)                     interfaces/. pytest test_interfaces.py
                                           confirms field names and types match ICD
                                           definitions. No block defines its own
                                           local copy of PoseEstimate, MetricSet, or
                                           ExperimentConfig.
  -----------------------------------------------------------------------------------

**Engineering Proposals**

The following proposals improve upon the implementation prompt. They are
presented for review --- none modifies a program guardrail.

**EP-01 Navigation-Agnostic Experiment IDs**

Proposal: Use EXP_NAV_NNN (navigation-agnostic) for the smoke test, and
EXP_VIO_NNN for VIO experiments, rather than a single global EXP_VIO\_
prefix for everything.

Rationale: When the platform expands to TRN re-evaluation or fusion
experiments, having EXP_TRN_001 alongside EXP_VIO_001 makes the results/
directory self-documenting. The counter is global (never resets), so
ordering is always unambiguous.

Cost: Zero. The change is in the ID generator and schema only.

**EP-02 Dataset Hash Verification**

Proposal: Compute and store SHA-256 of the sequence files at experiment
launch time, stored in metadata.yaml as dataset_hash.

Rationale: A silent dataset change (corrupted download, partial
re-download) would produce incorrect metrics with no indication that the
data changed. The hash makes this detectable. If the hash in
metadata.yaml does not match the current file, the experiment cannot be
reproduced.

Cost: One hash computation per experiment launch --- negligible on SSD.

**EP-03 Atomic Archive Write**

Proposal: ResultsArchive writes to a temp directory (EXP_ID.tmp) and
renames to the final directory only on successful completion.

Rationale: If the experiment runner crashes mid-write, a partial archive
is worse than no archive --- it can appear valid but contain corrupt
data. The atomic rename guarantees that any directory in results/
without the .tmp suffix is complete.

Cost: One directory rename operation --- zero performance cost.

**EP-04 Covariance Quality Validation**

Proposal: MetricsEngine checks that PoseEstimate.covariance_6x6 is not a
diagonal identity matrix and not all-zeros. Flag as a protocol violation
if either condition is detected.

Rationale: Several VIO systems default to publishing identity covariance
when the estimator is uncertain. If MicroMind\'s ESKF fusion relies on
covariance to weight the VIO measurement, a false identity covariance is
more dangerous than no covariance. Detecting this in the sandbox
prevents a latent fusion bug.

Cost: One matrix check per pose --- negligible.

**EP-05 i5-14400F Hybrid Core Warning**

Proposal: environment_capture.py checks for Intel hybrid core
architecture (P-cores + E-cores) and logs a warning recommending
explicit MKL thread count: export MKL_NUM_THREADS=6 (P-cores only).

Rationale: The i5-14400F has 6 P-cores and 10 E-cores. MKL dispatches
inconsistently on hybrid cores --- the same failure mode observed on the
Azure VM in the TRN sandbox. NumPy operations used in the MetricsEngine
(ATE/RPE computation) will exhibit this. The warning is automatic and
requires no action from the user, but prevents silent performance
inconsistency across benchmark runs.

Cost: Zero runtime cost. Warning on first run only.

*MicroMind VIO Sandbox --- Phase-0 Architecture Definition v1.1 \| March
2026 \| Incorporates 12-item review \| PROGRAMME CONFIDENTIAL*

