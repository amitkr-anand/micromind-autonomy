# Sprint Handoff — S-NEP-02 → S-NEP-03

| Field | Content |
|---|---|
| Sprint ID | S-NEP-02 |
| Date | 2026-03-20 |
| Commit hash | f3c83a3 |
| Tests at close | 424/424 passing — 0 skips |
| Spec conformance | NEP Platform Contract Specification v1.1 (Phase-0 Baseline) |

---

## Spec Conformance Declaration

I confirm that all changes made in this sprint are consistent with
NEP Platform Contract Specification v1.1 (Phase-0 Baseline).

No block responsibilities, interface contracts, or information flow
sequences were altered without a versioned spec update.

All new code was implemented after the relevant spec section was re-read
(§4.3 DatasetRef, §4.1 DatasetRef Construction Contract, §11.2 Phase-1
Integration Steps).

---

## Deliverables Completed

| ID | Deliverable | Gate | Result |
|---|---|---|---|
| 02-A | `DatasetManager.build_dataset_ref(dataset_name, groundtruth=None) → DatasetRef` | DatasetRef returned with valid absolute paths for EuRoC sequences | ✅ PASS |
| 02-B | Update `ExperimentRunner._run_from_config()` step 3 | `NotImplementedError` no longer raised for `dry_run=False` | ✅ PASS |
| 02-C | `tests/test_dataset_ref_from_dataset_manager.py` — 11 tests | New test file passes; full suite clean | ✅ PASS |

---

## Exit Gate Verification

| Check | Result |
|---|---|
| `DatasetManager.build_dataset_ref()` implemented and tested | ✅ |
| `ExperimentRunner` step 3 no longer raises `NotImplementedError` | ✅ |
| Real run with `dry_run=False` and EuRoC fixture proceeds to runner dispatch | ✅ |
| Full suite clean: 424/424 | ✅ |

---

## DatasetRef Invariants Verified (INV-DR-01 through INV-DR-05)

| Invariant | Condition | Test | Result |
|---|---|---|---|
| INV-DR-01 | `sequence_path`, `groundtruth_path`, `calibration_path` all absolute Paths | `test_inv_dr01_all_paths_absolute` | ✅ PASS |
| INV-DR-02 | `dataset_hash` format: `sha256:<64 hex chars>` | `test_inv_dr02_dataset_hash_format` | ✅ PASS |
| INV-DR-03 | `camera_rate_hz <= imu_rate_hz` | `test_inv_dr03_camera_rate_not_exceed_imu_rate` | ✅ PASS |
| INV-DR-04 | `duration_s > 0`, `trajectory_length_m > 0` | `test_inv_dr04_duration_and_length_positive` | ✅ PASS |
| INV-DR-05 | Paths exist on disk | `test_inv_dr05_paths_exist_on_disk` | ✅ PASS |

---

## Implementation Notes

- `build_dataset_ref()` inserted at line 417 of `datasets/dataset_manager.py`, inside the `DatasetManager` class, immediately before the `DatasetLoadError` class boundary.
- `imu_rate_hz` hard-coded to `200.0` for EuRoC — Phase-1 constant per spec §4.1.
- `camera_rate_hz` derived from image list: `round((N-1) / (t_last - t_first))`. Falls back to `20` for degenerate fixtures with fewer than 2 frames.
- `calibration_path` resolves to `cam0/sensor.yaml` if present, else falls back to `cam0/data.csv` (per spec §4.1).
- `dataset_hash` is SHA-256 of the ground truth CSV file, computed in 64 KB chunks.
- Passing `groundtruth` avoids re-reading the CSV — ExperimentRunner passes the already-loaded `GroundTruth` from step 2.
- Phase-0 guard test `test_real_run_without_dataset_integration_raises` retired and replaced with `test_real_run_dataset_ref_construction_no_longer_raises` in `tests/test_experiment_runner.py`.

---

## Known Issues / Deferred Items

None. Sprint scope fully delivered.

---

## S-NEP-03 Entry Checklist

S-NEP-03 goal: Connect OpenVINS through the full pipeline and produce a committed `MetricSet` on EuRoC MH_01_easy.

```bash
# Verify clean state before starting S-NEP-03
cd ~/micromind/repos/nep-vio-sandbox
git log --oneline main | head -3          # must show f3c83a3 at top
git status                                 # must be clean
python3 -m pytest tests/ -q               # must be 424/424
```

Pre-work required before implementation begins:
1. Download EuRoC MH_01_easy (ASL format) — deliverable 03-A
2. Verify `DatasetManager.load('euroc/MH_01_easy')` succeeds on full sequence
3. Confirm `gt.num_gt_poses > 10000` and `gt.total_length_m > 80.0 m`

S-NEP-03 deliverables (from Phase-1 Execution Plan §5):
- **03-A** Download EuRoC MH_01_easy and verify DatasetManager loads it
- **03-B** Implement `runners/vio/openvins_runner.py` (subprocess runner, yields PoseEstimates)
- **03-C** Register runner, run end-to-end, produce committed archive record with real MetricSet

