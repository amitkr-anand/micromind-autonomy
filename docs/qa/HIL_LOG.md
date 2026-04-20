
---

## HIL H-1 — Orin Nano Super Environment Setup
**Date:** 19 April 2026
**Status:** PASS
**Hardware:** NVIDIA Jetson Orin Nano Super Engineering Reference Dev Kit
**JetPack:** R36.4.7 | Ubuntu 22.04 | CUDA 12.6 | cuDNN 9.3.0 | TensorRT 10.3
**SSH:** mmuser-orin@192.168.1.53 (key-based, no password)
**Dev machine:** mmuser@192.168.1.44

### Environment
- Miniforge ARM64, conda env: micromind-autonomy, Python 3.11.15
- Core deps: numpy 2.4.3, scipy 1.17.1, rasterio 1.4.4, cv2 4.13.0
- Repository: cloned via SSH from dev machine, HEAD 233781a

### Certified Baseline
- Result: 483/483 PASS
- Runtime: 18m25s at max clocks (nvpmodel -m 0 + jetson_clocks)
- Frozen files: error_state_ekf.py md5=7021ff952454474c3bc289acd63ed480 MATCH
                bcmp1_runner.py    md5=3ea4416da572e20a0cf4c558ad1b3c00 MATCH

### OI-25 — ESKF Latency
- Measured at max clocks (nvpmodel -m 0)
- P50: 0.0957ms | P95: 0.1014ms | P99: 0.1136ms | Budget: 50ms
- Margin: 49.89ms (99.8%) — 440× inside budget
- Status: CLOSED — Orin Nano sufficient, no escalation to Orin NX required

### Terrain Data on Orin (minimum set)
- shimla_corridor/SHIMLA-1_COP30.tif                         (29MB)
- shimla_manali_corridor/shimla_tile.tif                     (29MB)
- shimla_manali_corridor/manali_tile.tif                     (82MB)
- Jammu_leh_corridor_COP30/TILE1/rasters_COP30/output_hh.tif (42MB)
- Jammu_leh_corridor_COP30/TILE2/rasters_COP30/output_hh.tif (58MB)
- Jammu_leh_corridor_COP30/TILE3/rasters_COP30/output_hh.tif (93MB)
- Shimla_Manali_Corridor/rasters_COP30/output_hh.tif         (82MB)
- Total: ~415MB (down from 5.4GB blind transfer — 92% reduction)
- Excluded: viz outputs, sentinel2 tiles, .SAFE archives — not needed for SIL tests

### sudo Configuration
- /etc/sudoers.d/micromind-hil: nvpmodel + jetson_clocks passwordless
- Applied via Orin console (keyboard required for bootstrap)

### Next: HIL H-2
- Full certified baseline at max clocks: COMPLETE (483/483 above)
- HIL H-3: LightGlue latency profiling on Orin GPU (upcoming)

---

## HIL H-3 — LightGlue Latency Profiling on Orin GPU
**Date:** 19 April 2026
**Status:** PASS (with Python version flag — see below)

### Results
- GPU: Orin Nano Super iGPU (1024-core Ampere, 918MHz max)
- CUDA: confirmed active (torch 2.5.0a0 Jetson build)
- Model load: 455ms (cold), cached after first call

| Frame | kp_u | kp_s | matches | inliers | total_ms | match_ms |
|-------|------|------|---------|---------|---------|---------|
| 0 (JIT warmup) | 2048 | 2048 | 541 | 203 | 1659 | 1613 |
| 1 | 2048 | 2048 | 569 | 246 | 915 | 902 |
| 2 | 2048 | 2048 | 763 | 347 | 624 | 611 |
| 3 | 2048 | 2048 | 768 | 345 | 628 | 615 |
| 4 | 2048 | 2048 | 736 | 333 | 628 | 614 |

### Latency (steady-state frames 1-4)
- Median: 628ms | P95: 1511ms | P99: 1630ms | Max: 1659ms
- Match-only mean: 871ms
- Slowdown vs dev machine (RTX 5060 Ti, 72ms): 12.4×

### Operational Assessment
- Slow-loop budget (2km at 27m/s): 74,000ms
- P99 worst case: 1,630ms
- Margin: 45× inside budget at P99
- TensorRT optimisation: NOT REQUIRED at current correction interval
- Verdict: VIABLE for operational L2 correction loop

### Python Version Flag (programme action required)
- Production env (micromind-autonomy): Python 3.11
- Jetson CUDA PyTorch wheels: cp310 (Python 3.10) only
- H-3 benchmark ran in separate hil-h3 conda env (Python 3.10)
- LightGlue CANNOT be imported in micromind-autonomy without resolution

### Recommended Resolution — Option C (subprocess IPC)
Run LightGlue as a separate process in hil-h3 (Python 3.10) environment.
Main micromind-autonomy process sends UAV frame + GPS prior via IPC.
LightGlue process returns position correction + confidence.
Interface contract: docs/interfaces/L2_LIGHTGLUE_IPC.md (to be created).
Consistent with SAD AD-03 (subprocess for process isolation).

### Next HIL Steps
- H-4: LightGlue subprocess IPC design and latency verification ✅ DONE
- H-5: End-to-end L2 correction integration test on Orin

---

## HIL H-4 — LightGlue Subprocess IPC Bridge — Orin Verification
**Date:** 19 April 2026  
**Status:** FULL PASS (Site 04 confirmed)  
**Commits:** 33c0d40 (bridge) + b523f59 (qa docs) + 26407a1 (cuda fix) + f0d7cc3 (extra tiles) + see final commit

### Environment
- Server: hil-h3 (Python 3.10.20) on Orin Nano Super
- Client: micromind-autonomy (Python 3.11.15) on Orin Nano Super
- IPC: Unix socket `/tmp/micromind_lightglue.sock`
- Model load: 484 ms (SuperPoint + LightGlue on CUDA, cached after first call)
- CUDA fix: `libcusparseLt.so.0` resolved from `nvidia/cusparselt` conda package;
  `LIGHTGLUE_LD_LIBRARY_PATH_EXTRA` injected by client at subprocess spawn

### Test Results
| Test | Result | Detail |
|------|--------|--------|
| T1 — server ping | **PASS** | status=pong, version=1.0, lightglue_available=True, round_trip=0.7ms |
| T2 — real GPU match | **PASS** | dlat=-0.036576° dlon=0.036270° conf=0.826 match_ms=1386 ipc=1.0ms |
| T3 — invalid coords | **PASS** | status=no_match reason=invalid_coordinates, 0.5ms |

### Latency Benchmark (5 frames, shimla same-modal, steady-state)
| Frame | match_ms | ipc_ms | total_ms |
|-------|----------|--------|----------|
| 0 (warmup) | 1386 | 1.0 | 1387 |
| 1 | 635 | 0.9 | 636 |
| 2 | 542 | 1.1 | 543 |
| 3 | 539 | 1.1 | 540 |
| 4 | 541 | 1.1 | 542 |

**Steady-state (frames 1-4): mean 564 ms  IPC overhead mean: 1.0 ms**  
Budget (2km @ 27m/s): 74,000ms — **131× margin at mean, consistent with H-3 628ms median**

### T2 Site 04 — Real UAV Frame (04_0001.JPG) Verification
After diagnostic revealed tile_resolver returned None for Site 04 GPS (119.9°E not in built-in regions), `satellite04.tif` was located at `/home/mmuser-orin/hil_benchmark/satellite04.tif` and registered via `LIGHTGLUE_EXTRA_TILES` env var (bounds: 119.906–119.955°E / 32.151–32.254°N, confirmed covering frame GPS).

| Item | Value |
|------|-------|
| Frame | 04_0001.JPG (real UAV nadir, Site 04) |
| GPS prior | 32.15556°N 119.928901°E |
| Tile | satellite04.tif (RGB, 38408×18093, EPSG:4326) |
| confidence | 0.743 |
| correction | 93.9 m |
| match_ms | 3,192 ms (first call — includes JIT + 3-band patch) |
| ipc_overhead | 2.2 ms |
| T2 result | **PASS** |

Note: 3,192 ms is elevated vs. 635 ms steady-state because satellite04.tif is 3-band RGB at very high resolution (38k×18k px); patch extraction and initial JIT combine. Within 74,000 ms budget (23× margin at this call). Steady-state will be faster for repeated GPS priors.

### Notes
- IPC overhead (1.0–2.2 ms) is higher than dev machine (0.35 ms) due to ARM pipeline; both are operationally negligible.
- stub_mode=False confirmed across all T1/T2/T3 runs.

### Next HIL Steps
- H-5: End-to-end L2 correction integration test — wire lightglue_client.match()
  into NavigationManager.update_trn() and verify correction is applied to ESKF
  on a shimla corridor replay mission on Orin.

### HIL H-4 — Addendum: Site 04 tile resolver fix and closure
**Date:** 19 April 2026

#### Tile resolver fix
- satellite04.tif actual bounds confirmed: 119.905980–119.954509°E / 32.151018–32.254036°N
- Built-in regions cover India only; satellite04.tif is outside all built-in bounds (119.9°E)
- Fix: LIGHTGLUE_EXTRA_TILES env var accepts JSON array of absolute-path tile entries
- Frame 0 GPS (32.155560°N 119.928901°E) confirmed covered after fix

#### T2 — Real Site 04 UAV frame (04_0001.JPG) final result
- conf=0.743, match_ms=3192ms (cold-start JIT + 3-band 38k×18k tile), correction=93.9m
- Steady-state latency 539–635ms confirmed from H-3; 23× budget margin at cold-start
- ipc_overhead=2.2ms (Unix socket); stub_mode=False confirmed

#### H-4 gate closure
- T1 server ping:        PASS
- T2 real Site 04 match: PASS (conf=0.743, real GPU, real UAV frame)
- T3 invalid coords:     PASS (returns None)
- HIL H-4:               FULL PASS

---

## HIL H-5 — NavigationManager + LightGlue Integration
**Date:** 21 April 2026
**Node:** Orin Nano Super (mmuser-orin@192.168.1.53)
**Env:** micromind-autonomy (Python 3.11) → lightglue_bridge → hil-h3 (Python 3.10, CUDA 12.6)
**HEAD:** ff6cbb8
**Deputy 1 ruling:** PASS (partial — H5-AC-3 deferred to H-6)

### Test scope
End-to-end integration of OI-48 (LightGlue wired into NavigationManager) and
OI-49 (SAL-2 terrain-class thresholds) on Orin GPU hardware.

### Results

| AC | Description | Result |
|---|---|---|
| H5-AC-1 | LightGlue server starts without exception | PASS |
| H5-AC-2 | match() returns server response (None acceptable at H-5) | PASS — None returned (geographic mismatch Site 04 / Shimla tile — expected) |
| H5-AC-3 | confidence in [0.0, 1.0] if match returned | DEFERRED — no geographic overlap at H-5; confirmed at H-6 |
| H5-AC-4 | Latency reported | PASS — 23,655ms wall-clock (cold-start); H-3 warm = 628ms median |
| H5-AC-5 | SAL-2 thresholds correct on Orin | PASS — ACCEPT=0.35, CAUTION=0.40, SUPPRESS=None |
| H5-AC-6 | lightglue_client in NavigationManager.__init__ | PASS — default=None confirmed |

### Frozen file verification
- core/ekf/error_state_ekf.py: 7021ff952454474c3bc289acd63ed480 ✅
- scenarios/bcmp1/bcmp1_runner.py: 3ea4416da572e20a0cf4c558ad1b3c00 ✅

### Open items raised
- OI-51 (NEW): HIL H-6 — geographically matched corridor replay. Requires UAV frames
  from Shimla corridor (or Site 04 tile loaded for Site 04 frames). Confirms H5-AC-3
  (confidence in [0,1]) and measures warm-path latency end-to-end through
  NavigationManager. Prerequisite: frame/tile geographic alignment confirmed before run.
