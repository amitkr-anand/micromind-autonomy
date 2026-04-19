
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
- H-4: LightGlue subprocess IPC design and latency verification
- H-5: End-to-end L2 correction integration test on Orin
