**Navigation Evaluation Platform**

Phase-1 Execution Plan

*Replay Runner Validation → First Algorithm Integration*

  -------------------------- --------------------------------------------
  **Document Status**        Baseline Version 1.0 (Frozen)

  **Linked Spec**            NEP Platform Contract Specification v1.1
                             (Phase-0 Frozen)

  **Architecture**           v1.2 (Phase-0 Frozen)

  **Platform**               NEP_1.0

  **Scope**                  Sprint S-NEP-01 → S-NEP-03

  **Prerequisites**          384/384 tests passing · Phase-0 design
                             freeze complete
  -------------------------- --------------------------------------------

+-----------------------------------------------------------------------+
| **Architecture Discipline**                                           |
|                                                                       |
| This plan governs implementation of the Navigation Evaluation         |
| Platform Phase-1.                                                     |
|                                                                       |
| The development cycle is fixed: Specification → Implementation →      |
| Verification.                                                         |
|                                                                       |
| Any proposed change to block responsibilities, interfaces, or         |
| information flow                                                      |
|                                                                       |
| requires a versioned update to the Platform Contract Specification    |
| before any                                                            |
|                                                                       |
| code is written. No architectural reinterpretation during             |
| implementation.                                                       |
+-----------------------------------------------------------------------+

**1 Context and Starting Point**

**1.1 Current State**

Phase-0 is complete and frozen. The platform has:

-   All seven blocks implemented: ExperimentRunner,
    ConfigurationManager, DatasetManager, Runner Layer, MetricsEngine,
    ResultsArchive, Interfaces.

-   384/384 tests passing. Architecture v1.2 frozen in Platform Contract
    Specification v1.1.

-   GroundTruthReplayRunner implemented but with 5 failing tests and a
    blocking DatasetRef gap.

-   DatasetManager.build_dataset_ref() not yet implemented --- real runs
    raise NotImplementedError at step 5.

-   OpenVINS, ORB-SLAM3, and other real runners: not yet integrated.

**1.2 Failing Tests That Must Be Resolved First**

Five tests in test_groundtruth_replay_runner.py currently fail. These
are not new work --- they are diagnostic failures from the Phase-0
implementation session that were not resolved. They must be the first
item in Phase-1.

  -----------------------------------------------------------------------------------------------------------------------
  **Test**                                                                  **Likely Root Cause**
  ------------------------------------------------------------------------- ---------------------------------------------
  TestZeroNoiseATE::test_zero_noise_produces_low_ate                        ATE computation path; likely DatasetManager
                                                                            fixture or MetricsEngine ATE alignment
                                                                            mismatch on short EuRoC fixture trajectory.

  TestNoiseATE::test_larger_noise_produces_larger_ate                       Statistical property test. Short fixture (100
                                                                            poses) may produce insufficient spread.
                                                                            Tolerance or fixture adjustment needed.

  TestDeterminism::test_different_seeds_produce_different_positions         RNG isolation test. Seeds may be producing
                                                                            identical streams ---
                                                                            numpy.random.default_rng seeding issue.

  TestFullPipelineIntegration::test_produces_committed_archive_record       Full pipeline test. Requires patched
                                                                            \_make_dry_run_dataset_ref + real GT CSV
                                                                            path. Staticmethod patching issue.

  TestFullPipelineIntegration::test_archive_metadata_contains_runner_type   Same root cause as above.
  -----------------------------------------------------------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **Rule**                                                              |
|                                                                       |
| These 5 tests must pass before any other Phase-1 work begins.         |
|                                                                       |
| A sprint that starts with 5 known failures carries hidden technical   |
| debt into every subsequent step.                                      |
+-----------------------------------------------------------------------+

**1.3 Phase-1 Objective**

Phase-1 has two sequential objectives separated by a clear verification
gate:

  -----------------------------------------------------------------------
  **Objective**      **Definition**
  ------------------ ----------------------------------------------------
  Objective 1 ---    Prove that the evaluation pipeline produces correct,
  Replay Validation  predictable, verifiable metrics from a controlled
                     pose stream. All seven validation invariants from
                     §10.6 of the spec must pass analytically.

  Objective 2 ---    Connect one real VIO algorithm (OpenVINS
  First Algorithm    recommended) through the full pipeline end-to-end.
  Integration        Produce a committed archive record with real
                     MetricSet values on a real EuRoC sequence.
  -----------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **Gate**                                                              |
|                                                                       |
| Objective 2 cannot begin until Objective 1 is fully verified.         |
|                                                                       |
| Attempting algorithm integration before the platform pipeline is      |
| proven is the primary                                                 |
|                                                                       |
| source of difficult-to-diagnose failures in evaluation platform       |
| projects.                                                             |
+-----------------------------------------------------------------------+

**2 Sprint Structure**

**2.1 Three-Sprint Plan**

Phase-1 is structured as three sprints. Each sprint has a single,
unambiguous goal and a binary exit gate. Sprint N+1 does not start until
Sprint N\'s exit gate passes.

  -----------------------------------------------------------------------------------------
  **Sprint**   **Name**         **Goal**                             **Exit Gate**
  ------------ ---------------- ------------------------------------ ----------------------
  S-NEP-01     Replay Runner    Prove the pipeline is correct under  29/29 replay runner
               Validation       controlled, analytically verifiable  tests pass. All 7
                                conditions.                          invariants verified.
                                                                     384+→N/N total suite.

  S-NEP-02     DatasetRef       Implement                            NotImplementedError
               Integration      DatasetManager.build_dataset_ref()   eliminated. Real run
                                so real runners receive verified     dispatches without
                                on-disk paths.                       raising. Full suite
                                                                     clean.

  S-NEP-03     First Algorithm  Connect OpenVINS through the full    status=complete in
               Integration      pipeline. Produce a committed        archive. MetricSet has
                                MetricSet on EuRoC MH_01_easy.       meaningful ATE and
                                                                     latency values. Suite
                                                                     clean.
  -----------------------------------------------------------------------------------------

**2.2 What Is Not in Phase-1**

  -------------------------------------------------------------------------
  **Non-Goal**          **Classification**
  --------------------- ---------------------------------------------------
  ORB-SLAM3,            Phase-2. Second and subsequent algorithms add after
  VINS-Fusion, Kimera   OpenVINS pipeline is proven end-to-end.

  Multi-sequence        Phase-2. Phase-1 proves one algorithm on one
  benchmarking          sequence.

  VisualizationModule   Phase-2. Not in Phase-0 scope. plot_files remains
  (B-06)                empty tuple.

  Distributed           Phase-2. Sequential execution only.
  experiment scheduling 

  KITTI, TartanAir      Phase-2. EuRoC ASL only in Phase-1.
  dataset loaders       

  ROS2 live node        Phase-3. All Phase-1 work is offline evaluation.
  integration           

  Stereo camera         Out of scope. DatasetRef.calibration_path is
  calibration pipeline  present but validation is runner responsibility.
  -------------------------------------------------------------------------

**3 Sprint S-NEP-01 --- Replay Runner Validation**

+-----------------------------------------------------------------------+
| **S-NEP-01 --- Replay Runner Validation**                             |
|                                                                       |
| **Goal:** *Close the 5 failing tests and verify all 7 platform        |
| invariants analytically.*                                             |
+-----------------------------------------------------------------------+

  --------------------------------------------------------------------------
  **ID**   **Deliverable**    **Description**                 **Acceptance
                                                              Gate**
  -------- ------------------ ------------------------------- --------------
  01-A     Fix 5 failing      Diagnose and fix each of the 5  29/29 replay
           tests              failing replay runner tests.    runner tests
                              Root causes identified in §1.2. pass.
                              No new architecture --- test    
                              fixes only. Each fix must be    
                              verified in isolation before    
                              moving to the next.             

  01-B     Invariant          Run the full replay runner test Tabulated
           verification run   suite with zero noise, known    invariant
                              sigma noise, fixed latency, and results match
                              fixed dropout in separate       spec
                              parameter files. Verify all 7   tolerances.
                              invariants from spec §10.6      Attached to
                              numerically.                    sprint
                                                              handoff.

  01-C     Full regression    Run complete test suite.        384+ / N tests
                              Confirm no regressions from the pass. Zero
                              test fixes.                     failures
                                                              permitted.
  --------------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **Exit Gate**                                                         |
|                                                                       |
| All 29 tests in test_groundtruth_replay_runner.py pass.               |
|                                                                       |
| All 7 validation invariants confirmed with numerical evidence.        |
|                                                                       |
| Full suite clean. Sprint handoff document committed.                  |
+-----------------------------------------------------------------------+

**3.1 Invariant Verification Protocol**

Each invariant must be verified with a dedicated parameter file and a
measurement. The results must be recorded in the sprint handoff document
--- not just asserted as passing tests.

  ------------------------------------------------------------------------------
  **INV**   **Condition**      **Parameter File**         **Acceptance**
  --------- ------------------ -------------------------- ----------------------
  INV-1     zero noise → ATE ≈ params:                    ate_rmse_m \< 0.01 m
            0                  position_stddev_m=0,       
                               orientation_stddev_rad=0   

  INV-2     σ=0.5 m noise →    params:                    0.1 σ ≤ ate_rmse_m ≤
            ATE ≈ 0.5 m        position_stddev_m=0.5,     3.0 σ
                               seed=42                    

  INV-3     latency=80 ms,     params: mean_ms=80.0,      latency_mean_ms = 80.0
            jitter=0           stddev_ms=0.0              ± 2.0 ms

  INV-4     dropout=0 → all    params: dropout.rate=0.0   pose count = GT row
            frames                                        count

  INV-5     dropout=0.5 →      params: dropout.rate=0.5   0.35 N ≤ pose_count ≤
            \~50% frames                                  0.65 N

  INV-6     no dropout, no     params: all zero           gate_latency,
            noise → gates pass                            gate_tracking,
                                                          gate_update_rate all
                                                          True

  INV-7     fixed seed →       run twice with seed=42;    All positions equal to
            identical stream   compare position arrays    float64 tolerance
  ------------------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **Fixture Freeze Rule**                                               |
|                                                                       |
| The EuRoC test fixture (tests/fixtures/euroc/MH_01_easy/) used for    |
| replay validation                                                     |
|                                                                       |
| must be treated as frozen for the duration of Phase-1.                |
|                                                                       |
| The fixture currently has 100 GT poses over \~0.25 m path and 50      |
| image frames.                                                         |
|                                                                       |
| If the fixture trajectory length, pose count, or timestamps are       |
| modified for any reason                                               |
|                                                                       |
| (e.g. to fix the ATE tests in §3.2), the invariant tolerances in the  |
| table above                                                           |
|                                                                       |
| must be recalculated and the table updated before re-running the      |
| verification protocol.                                                |
|                                                                       |
| A tolerance derived from the old fixture applied to a new fixture is  |
| not a valid verification.                                             |
+-----------------------------------------------------------------------+

**3.2 Debug Protocol for the 5 Failing Tests**

Apply the standing programme debug rule: if a test is not fixed in 3--4
attempts, stop patching. Run diagnostics, inspect live state, identify
root cause fully, then make one correct fix.

**01-A-1 ATE tests (test_zero_noise_produces_low_ate,
test_larger_noise_produces_larger_ate)**

Likely cause: the EuRoC fixture has 100 GT poses over 0.25 m total path
(short fixture). ATE alignment via evo requires ≥3 associated pairs with
non-degenerate spatial displacement. Check
MetricsEngine.\_compute_trajectory_errors() fallback path --- if evo
fails the fallback returns nearest-neighbour ATE which on a short
trajectory may be dominated by timestamp-association error, not position
noise.

-   Diagnostic: print total pose count, total_length_m, and ate_rmse_m
    values in a debug run.

-   If the issue is fixture length: extend the fixture to 500+ poses at
    the existing 200 Hz rate (\~2.5 seconds), moving at 0.5 m/s → 1.25 m
    total path. This stays above the 1.0 m floor in
    \_compute_path_length.

-   If the issue is evo alignment failure on short trajectories:
    document in spec §6.1 minimum trajectory note and ensure the
    fallback produces a physically sensible ATE for the test to assert
    against.

**01-A-2 Determinism test
(test_different_seeds_produce_different_positions)**

Likely cause: numpy.random.default_rng(seed) with seed=1 vs seed=2 may
produce identical first values for the noise distribution. Check that
the RNG state is per-runner-instance and not shared. Verify the noise is
actually being applied (position_stddev_m \> 0 in both parameter files).

**01-A-3 Pipeline integration tests**

Root cause identified: \_patched_make_ref is assigned as a plain
function to a staticmethod slot --- Python\'s descriptor protocol does
not wrap it correctly. The fix is to patch at the instance level on the
runner object, not the class staticmethod. Use
runner.\_make_dry_run_dataset_ref = lambda config: DR(\...) or use
unittest.mock.patch.object with a new_callable that wraps the lambda
correctly.

**4 Sprint S-NEP-02 --- DatasetRef Integration**

+-----------------------------------------------------------------------+
| **S-NEP-02 --- DatasetRef Integration**                               |
|                                                                       |
| **Goal:** *Implement DatasetManager.build_dataset_ref() so real runs  |
| receive verified on-disk paths instead of raising                     |
| NotImplementedError.*                                                 |
+-----------------------------------------------------------------------+

  -----------------------------------------------------------------------------------------------------
  **ID**   **Deliverable**       **Description**                                  **Acceptance Gate**
  -------- --------------------- ------------------------------------------------ ---------------------
  02-A     Implement             Add                                              DatasetRef returned
           build_dataset_ref()   DatasetManager.build_dataset_ref(dataset_name) → with valid absolute
                                 DatasetRef. Must load the sequence directory,    paths for EuRoC
                                 locate groundtruth_path (via existing            sequences.
                                 \_find_euroc_groundtruth()), calibration_path    
                                 (sensor.yaml under cam0/), compute duration_s    
                                 and trajectory_length_m from the already-loaded  
                                 GroundTruth data, and populate dataset_hash      
                                 (SHA-256 of groundtruth CSV). Re-use all         
                                 existing DatasetManager methods --- no new       
                                 parsing logic.                                   

  02-B     Update                Replace the NotImplementedError branch in        NotImplementedError
           ExperimentRunner step \_run_from_config() step 5 with a call to        no longer raised for
           5                     DatasetManager.build_dataset_ref(). Both         real runs with
                                 GroundTruth and DatasetRef are now produced from dry_run=False.
                                 DatasetManager before runner dispatch. Remove    
                                 \_make_dry_run_dataset_ref() or restrict it to   
                                 dry_run=True branch only.                        

  02-C     Add DatasetRef tests  Add test_dataset_ref_from_dataset_manager.py.    New test file passes.
                                 Test that build_dataset_ref() returns correct    Full suite clean.
                                 paths, non-zero trajectory_length_m, valid       
                                 dataset_hash format, and that runner             
                                 construction with the returned DatasetRef does   
                                 not raise.                                       
  -----------------------------------------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **Exit Gate**                                                         |
|                                                                       |
| DatasetManager.build_dataset_ref() implemented and tested.            |
|                                                                       |
| ExperimentRunner step 5 no longer raises NotImplementedError.         |
|                                                                       |
| Real run with dry_run=False and EuRoC fixture proceeds to runner      |
| dispatch.                                                             |
|                                                                       |
| Full suite clean.                                                     |
+-----------------------------------------------------------------------+

**4.1 DatasetRef Construction Contract**

build_dataset_ref() must satisfy the following from the Platform
Contract Specification §7.1:

  ----------------------------------------------------------------------------
  **Field**             **Construction Rule**
  --------------------- ------------------------------------------------------
  sequence_path         Absolute path to the sequence root directory. The
                        directory must exist on disk.

  groundtruth_path      Absolute path to the ground truth CSV. Located via
                        \_find_euroc_groundtruth(mav0_dir). Must exist.

  calibration_path      Absolute path to cam0/sensor.yaml. If not present, use
                        cam0/data.csv as the path (calibration loading is a
                        runner responsibility; the path must be a valid
                        absolute path that exists).

  dataset_hash          SHA-256 of the groundtruth CSV file. Format:
                        sha256:\<64 hex chars\>. Reuse
                        ConfigurationManager.hash_file() or equivalent.

  camera_rate_hz        Read from the image list CSV: (last_image_timestamp -
                        first_image_timestamp) / (N_images - 1). Rounded to
                        nearest Hz.

  imu_rate_hz           Fixed 200.0 for EuRoC. Hard-coded constant is
                        acceptable in Phase-1.

  duration_s            gt.timestamps_s\[-1\] - gt.timestamps_s\[0\] from the
                        already-loaded GroundTruth.

  trajectory_length_m   gt.total_length_m from the already-loaded GroundTruth.
  ----------------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **Design note**                                                       |
|                                                                       |
| DatasetManager.build_dataset_ref() should accept a GroundTruth as an  |
| optional second argument.                                             |
|                                                                       |
| ExperimentRunner has already called DatasetManager.load() at step 4   |
| and holds the GroundTruth object.                                     |
|                                                                       |
| Passing it in avoids re-reading the CSV. Signature:                   |
| build_dataset_ref(dataset_name, groundtruth=None).                    |
|                                                                       |
| If groundtruth is None, load it internally.                           |
+-----------------------------------------------------------------------+

**5 Sprint S-NEP-03 --- First Algorithm Integration**

+-----------------------------------------------------------------------+
| **S-NEP-03 --- First Algorithm Integration**                          |
|                                                                       |
| **Goal:** *Connect OpenVINS through the full pipeline end-to-end and  |
| produce a committed archive record with a real MetricSet on EuRoC     |
| MH_01_easy.*                                                          |
+-----------------------------------------------------------------------+

  -------------------------------------------------------------------------------------------------------------------
  **ID**   **Deliverable**    **Description**                             **Acceptance Gate**
  -------- ------------------ ------------------------------------------- -------------------------------------------
  03-A     Download EuRoC     Download the full EuRoC MH_01_easy sequence DatasetManager.load(\'euroc/MH_01_easy\')
           MH_01_easy         (ASL format) to the platform datasets       succeeds on full sequence. gt.num_gt_poses
                              directory. Verify DatasetManager.load()     \> 10000. gt.total_length_m \> 80.0 m.
                              produces a valid GroundTruth. Verify        
                              DatasetManager.build_dataset_ref() produces 
                              a valid DatasetRef with correct paths.      

  03-B     OpenVINS           Implement runners/vio/openvins_runner.py    runner.stream() yields at least 1000
           subprocess runner  inheriting BaseRunner. Uses subprocess to   PoseEstimate objects on MH_01_easy. All
                              launch OpenVINS against the EuRoC sequence. timestamps strictly increasing.
                              Reads OpenVINS pose output (TUM or EuRoC    
                              format) and yields PoseEstimate objects.    
                              frame_timestamp_s populated from dataset    
                              image timestamps via DatasetRef. Handles    
                              process cleanup in close(). Parameters file 
                              configures OpenVINS config path.            

  03-C     Registration and   Register OpenVINSRunner in                  ExperimentRunner.\_RUNNER_REGISTRY contains
           parameter file     \_RUNNER_REGISTRY\[(\"vio\",                (\"vio\", \"openvins\") entry.
                              \"openvins\")\]. Add \'openvins\' to        
                              VALID_ALGORITHMS in interfaces/constants.py 
                              and schema.yaml. Create                     
                              config/algorithms/openvins_mh01_easy.yaml   
                              parameter file pointing to OpenVINS config. 

  03-D     End-to-end run     Execute runner.run() with dry_run=False,    Committed archive record with
                              algorithm=openvins,                         status=complete. ate_rmse_m \> 0 and \< 50
                              dataset_name=euroc/MH_01_easy. Verify       m (plausible VIO result). latency_mean_ms
                              status=complete, MetricSet produced,        \> 0.
                              archive committed.                          

  03-E     Replay vs real     Run GroundTruthReplayRunner and             Replay ATE \< 0.01 m. OpenVINS ATE \> 0 m.
           comparison         OpenVINSRunner on the same sequence.        Both records committed. Results logged in
                              Compare MetricSet values. Replay with zero  sprint handoff.
                              noise should show ATE near zero; OpenVINS   
                              should show realistic ATE. This confirms    
                              the pipeline correctly distinguishes        
                              algorithm quality.                          
  -------------------------------------------------------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **Exit Gate**                                                         |
|                                                                       |
| EuRoC MH_01_easy full sequence loaded successfully by DatasetManager. |
|                                                                       |
| OpenVINSRunner.stream() yields a valid pose stream.                   |
|                                                                       |
| End-to-end run produces committed archive with status=complete.       |
|                                                                       |
| Replay vs real comparison logged. Pipeline validated end-to-end.      |
|                                                                       |
| Full test suite clean.                                                |
+-----------------------------------------------------------------------+

**5.1 OpenVINS Runner Implementation Notes**

The OpenVINS runner is a subprocess runner. The following contracts from
the Platform Contract Specification apply:

  -----------------------------------------------------------------------
  **Concern**      **Rule**
  ---------------- ------------------------------------------------------
  Timestamp source frame_timestamp_s must come from the EuRoC image list
                   timestamps --- not from system wall-clock. Use
                   DatasetRef.groundtruth_path to locate the sequence
                   root, then find cam0/data.csv for image timestamps.

  Clock alignment  Some OpenVINS builds publish pose timestamps using ROS
                   system clock rather than dataset clock. If this is the
                   case, the runner must translate system timestamps into
                   dataset time before yielding. Method: record the
                   wall-clock time at which the first image is fed to
                   OpenVINS, record the dataset timestamp of that image
                   from cam0/data.csv, then offset = dataset_t0 -
                   wall_t0. Apply: pose_dataset_ts = pose_wall_ts +
                   offset. Without this correction, latency values will
                   be meaningless (typically very large or negative after
                   clamping) and ATE timestamp association will fail.

  Latency          timestamp_s = OpenVINS output timestamp, translated to
  definition       dataset clock if needed. frame_timestamp_s = dataset
                   image timestamp from cam0/data.csv. The difference
                   must be \>= 0. If after clock alignment any pose still
                   has timestamp_s \< frame_timestamp_s, that pose must
                   be dropped --- it indicates a dataset clock /
                   algorithm clock misalignment that cannot be corrected
                   by offset alone.

  Ordering         OpenVINS via ROS2 bag may produce slightly
                   out-of-order timestamps due to topic buffering. The
                   runner must sort or drop before yielding. Never pass
                   out-of-order poses to MetricsEngine.

  close()          The subprocess must be terminated in close(). close()
  obligation       must be idempotent --- safe to call if process already
                   exited. Use subprocess.Popen and .terminate() /
                   .wait().

  metadata()       Return at minimum: algorithm_version (from
  fields           ExperimentConfig), config_path (absolute path to
                   OpenVINS config YAML), total_poses_yielded,
                   clock_alignment_applied (bool), clock_offset_s (float,
                   0.0 if not applied).
  -----------------------------------------------------------------------

+-----------------------------------------------------------------------+
| **Clock Alignment Rule --- OpenVINS**                                 |
|                                                                       |
| OpenVINS may publish poses with system wall-clock timestamps (ROS     |
| clock) rather than                                                    |
|                                                                       |
| dataset-clock timestamps. This is not an error in OpenVINS --- it is  |
| a known behaviour                                                     |
|                                                                       |
| of some build configurations and ROS2 launch setups.                  |
|                                                                       |
| The runner is responsible for detecting and correcting this before    |
| yielding any pose.                                                    |
|                                                                       |
| Detection: if the first pose timestamp is within 1 second of the      |
| current system time                                                   |
|                                                                       |
| and more than 1000 seconds from the dataset t0, system-clock          |
| timestamps are in use.                                                |
|                                                                       |
| Without correction: latency_mean_ms will be either very large (system |
| time \>\> dataset                                                     |
|                                                                       |
| time) or negative after clamping (system time \<\< dataset time on    |
| Azure/VM). ATE                                                        |
|                                                                       |
| timestamp association will produce degenerate matches. The archive    |
| record will be VALID                                                  |
|                                                                       |
| but the MetricSet values will be physically meaningless.              |
+-----------------------------------------------------------------------+

**5.2 EuRoC Sequence Download and Validation**

Before 03-B begins, verify the full EuRoC sequence is correctly loaded
by the existing DatasetManager:

> python3 -c \"from datasets.dataset_manager import DatasetManager; from
> pathlib import Path; mgr =
> DatasetManager(Path(\'/path/to/datasets\')); gt =
> mgr.load(\'euroc/MH_01_easy\'); print(gt.num_gt_poses,
> gt.total_length_m)\"

Expected output: num_gt_poses \> 10000, total_length_m \> 80.0. If
total_length_m returns 1.0 (the floor), the trajectory data has not
loaded correctly --- check the CSV path and the mav0/ directory
structure.

**6 Verification Requirements**

**6.1 Spec Compliance Checks**

Each sprint must produce evidence that the implementation conforms to
the Platform Contract Specification. The following checks must be
performed at the end of each sprint, not just at the end of Phase-1.

  ---------------------------------------------------------------------------------
  **Sprint**   **Check**      **Method**                       **Spec Ref**
  ------------ -------------- -------------------------------- --------------------
  S-NEP-01     Timestamp      Run replay with high latency     Spec §5.6
               ordering       jitter (stddev=50 ms). Confirm   
                              TimestampRegressionError is      
                              never raised.                    

  S-NEP-01     Empty stream   Register ZeroRunner, run         Spec §8.3
                              dry_run=False (with fixture      
                              DatasetManager). Confirm         
                              failure_cause=\'empty_stream\'   
                              in metadata.                     

  S-NEP-01     INV-PE-05      Attempt to construct             Spec §4.1
                              PoseEstimate with timestamp_s \< 
                              frame_timestamp_s. Confirm       
                              construction raises.             

  S-NEP-02     DatasetRef     Call build_dataset_ref() and     Spec §4.3
               invariants     confirm all INV-DR checks pass:  
                              absolute paths, valid hash       
                              format, camera_rate_hz \<=       
                              imu_rate_hz.                     

  S-NEP-02     Step-5 gate    With real runner registered and  Spec §3.1 step 5
                              dataset on disk, confirm run()   
                              completes without                
                              NotImplementedError.             

  S-NEP-03     Archive        After commit(), attempt to write Spec §8.4
               immutability   to the archive directory.        
                              Confirm ResultsArchive refuses.  

  S-NEP-03     Validity vs    Produce a VALID experiment with  Spec §8.3
               gates          poor performance (large noise).  
                              Confirm status=complete,         
                              acceptance_pass=False. Confirm   
                              it is a VALID result.            

  S-NEP-03     Runner         Static analysis: confirm         Spec §5.4
               boundary       openvins_runner.py imports       
                              nothing from evaluation/,        
                              results_archive, or              
                              experiment_runner.               
  ---------------------------------------------------------------------------------

**6.2 Non-Regression Rule**

Every deliverable in every sprint must leave the full test suite clean.
The rule is identical to the standing programme rule for MicroMind:

+-----------------------------------------------------------------------+
| Before starting any deliverable: run full suite, confirm N/N passing. |
|                                                                       |
| After completing any deliverable: run full suite, confirm N+delta /   |
| N+delta passing.                                                      |
|                                                                       |
| A deliverable that introduces a new failure in an existing test is    |
| not complete.                                                         |
|                                                                       |
| If a bug is not resolved in 3--4 attempts: STOP. Diagnose. One        |
| correct fix.                                                          |
+-----------------------------------------------------------------------+

**7 Sprint Handoff Template**

Every sprint close produces a handoff document committed to the
repository under Daily Logs/. The following template is mandatory.

  -----------------------------------------------------------------------
  **Field**          **Content**
  ------------------ ----------------------------------------------------
  Sprint ID          S-NEP-0N

  Date               YYYY-MM-DD

  Commit hash        git log \--oneline main \| head -1

  Tests at close     N/N passing (list any skips)

  Deliverables       List each with gate result
  completed          

  Invariants         Tabulated results for each INV (S-NEP-01 only)
  verified           

  Known issues       Any deferred items with rationale
  carried            

  Next sprint        Exact commands to verify clean state before starting
  checklist          
  -----------------------------------------------------------------------

**7.1 Session Start Checklist (every sprint)**

Run these commands at the start of every session, regardless of sprint
stage. Do not begin implementation until all pass.

> cd sandbox_vio
>
> python3 -m pytest tests/ -q \# must be N/N before touching anything
>
> git log \--oneline main \| head -5 \# confirm latest commit
>
> git status \# confirm clean working tree

**7.2 Spec Conformance Declaration**

Each sprint handoff must include an explicit declaration:

+-----------------------------------------------------------------------+
| I confirm that all changes made in this sprint are consistent with    |
|                                                                       |
| NEP Platform Contract Specification v1.1 (Phase-0 Baseline).          |
|                                                                       |
| No block responsibilities, interface contracts, or information flow   |
| sequences                                                             |
|                                                                       |
| were altered without a versioned spec update.                         |
|                                                                       |
| All new code was implemented after the relevant spec section was      |
| re-read.                                                              |
+-----------------------------------------------------------------------+

*End of NEP Phase-1 Execution Plan --- Baseline Version 1.0 (Frozen) ---
Linked to Platform Contract Specification v1.1*

