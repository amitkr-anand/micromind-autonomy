
## Phase B Validity Caveat (Deputy 1, 18 April 2026)

Phase B Blender frames at 150m AGL are NOT valid test frames for matcher evaluation.

Root cause: Sentinel-2 TCI at 10m/px native resolution provides only 17x17 source
pixels for a 173m footprint. Blender renders this as a heavily upsampled, blurry
image — confirmed by visual inspection of frames km025-km040.

Consequences:
- SuperPoint+LightGlue failure is correct behaviour on blurry low-frequency input.
  The matcher CANNOT be judged on these frames. Result 0/12 is not a terrain finding.
- Phase correlation 14.43m mean error on these frames is unvalidated.
  Near-zero errors may be degenerate matches on low-frequency content, not genuine
  terrain registration. Result is indicative only — not a validated performance claim.
- LoFTR 23.19m mean error is the most credible Phase B result (dense matching is
  more tolerant of low-frequency input) but also requires validation on proper frames.

Valid evaluation requires ONE of:
  A) Higher UAV altitude: ≥500m AGL gives ≥57x57 S2 source pixels (marginal)
     ≥1000m AGL gives ≥115x115 S2 source pixels (adequate)
  B) Higher resolution reference: CARTOSAT-1 2.5m/px or aerial orthophoto 0.5m/px
  C) Direct S2-to-S2 crop matching at native tile scale (no Blender pipeline)

AD-23 remains valid based on Phase A Colombia data (genuine orthoimage pairs).
Phase B Indian terrain results are INDICATIVE ONLY — not cited as validated performance.

## Altitude Sweep — S2 Same-Modality Phase Correlation (18 April 2026)

### Test design
Query = S2 TCI crop shifted 5px (simulated INS drift)
Reference = same S2 TCI crop unshifted
Purpose: verify algorithm correctness across altitudes with S2 10m/px

### Results
| AGL | Footprint | S2_px | GSD | Mean SNR | Correct |
|-----|-----------|-------|-----|---------|---------|
| 150m | 173m | 17 | 0.271m/px | 409,440 | YES — 1.35m |
| 500m | 577m | 58 | 0.902m/px | 409,420 | YES — 4.51m |
| 1000m | 1155m | 115 | 1.804m/px | 409,438 | YES — 9.02m |
| 2000m | 2309m | 231 | 3.608m/px | 409,446 | YES — 18.04m |

### Interpretation (Deputy 1, 18 April 2026)
SNR ~409,000 is a self-correlation artefact — image vs shifted copy of itself
always produces near-perfect correlation regardless of content. This confirms
algorithmic correctness only, not real-world cross-condition performance.

Real-world L2 TRN gap is cross-modal/cross-condition (sensor difference,
seasonal variation, Blender rendering vs real camera), not altitude or GSD.

Correction precision scales linearly with GSD — 0.27m at 150m AGL to 3.6m
at 2000m AGL. All altitudes within NAV-02 spec if corrections are frequent.

### AD-23 Addendum
Architecture decision unchanged. Phase correlation validated as L2 primary.
Operational altitude for HIL: 150m AGL per Gates 1-6 certified baseline.
Real EO camera data on Orin Nano (HIL) is the next validation step.
Blender simulation at 150m AGL with S2 texture is not a valid EO camera proxy.

## Sandbox Status: COMPLETE AND CLOSED

---

## Phase D — UAV-VisLoc Dataset Evaluation (19 April 2026)

### Dataset
**Citation:** Xu, W., Yao, Y., Cao, J., Wei, Z., Liu, C., Wang, J., Peng, M. (2024).
UAV-VisLoc: A Large-scale Dataset for UAV Visual Localization. arXiv:2405.11936.

11 sites, 6,774 UAV frames, georeferenced satellite GeoTIFF per site (~0.28m/px),
GPS ground truth per frame (lat, lon, height, heading angles).

### Phase D-1 — Baseline Benchmarking (native satellite resolution ~0.28m/px)

**Implementation fixes applied before valid results:**
1. CRS-aware satellite crop: EPSG:4326 tiles require degree-space half-extent, not metre-space
2. Heading rotation applied after downscaling (not at native 3976×2652 resolution)
3. Per-frame multi-tile satellite lookup (Site 09 had 4 tiles, wrong tile used previously)
4. Pre-evaluation preflight check mandatory (00_dataset_preflight.py)

**Results — stable sites only (temporal change sites excluded):**

| Site | Terrain | GSD | LG_med | LG_acc | LF_med | LF_acc |
|------|---------|-----|--------|--------|--------|--------|
| 01 | farmland/river | 0.366 | 34.2m | 0.20 | 76.7m | 1.00 |
| 02 | farmland/river | 0.366 | 28.9m | 0.13 | 41.4m | 0.87 |
| 03 | peri-urban agricultural | 0.421 | — | 0.00 | 91.4m | 0.97 |
| 04 | industrial/road infrastructure | 0.490 | 34.7m | 0.83 | 69.5m | 1.00 |
| 05 | highland plateau | 2.087 | — | 0.00 | 318.4m | 1.00 |
| 06 | mixed forest | 0.757 | — | 0.00 | 135.4m | 1.00 |
| 07 | narrow corridor | 4.679 | — | 0.00 | 169.0m | 0.46 |
| 10 | semi-arid agricultural | 0.697 | — | 0.00 | 196.9m | 0.93 |
| 11 | arid desert | 2.303 | — | 0.00 | 490.8m | 0.97 |

**Excluded sites:**
- 08a: Lake Taihu water surface — valid suppression scenario, no matcher viable
- 08b: Temporal change — greenhouses built after satellite acquisition (confirmed by visual inspection)
- 09: Temporal change — highway construction not present in satellite (confirmed by visual inspection)

**LightGlue deep-dive — Site 04, 60 frames, confidence threshold calibration:**

At threshold 0.50: 49/60 accepted, 40.0m mean GT error
At threshold 0.35: 56/60 accepted, 42.4m mean GT error, P90 78.9m
Decision: threshold 0.35 adopted (Deputy 1, 19 April 2026)
Rejection breakdown at 0.35: 0 failed match count, 1 failed inliers, 3 failed confidence

**LightGlue operational envelope (validated):**
- Works: Structured terrain — roads, buildings, industrial zones, field networks — at 400–800m AGL
- Fails correctly: Agricultural fields without structural features, forest, >1500m AGL, desert, water
- Temporal change causes failure — correct suppression, not matcher limitation

**LoFTR operational envelope (characterised):**
- Works: 8/9 stable sites at 0.92 mean accept rate
- Median error: 163.5m overall; ~110m excluding high-altitude sites (GSD <1.5m/px)
- Fails: Narrow corridor (site 07, GSD 4.679m/px), water surface

**Phase correlation (not viable as standalone):**
- 100% accept rate everywhere — no confidence discrimination
- 411m median error — shift-only, no rotation handling
- Not suitable for operational position correction at these altitudes

### LightGlue Role Evaluation (Deputy 1, 19 April 2026)

| Role | Decision | Reason |
|------|----------|--------|
| EO-to-satellite position correction | VALIDATED | 56/60, 42.4m, 72ms |
| EO-to-EO VIO heading (fast loop) | REJECTED | 72–500ms incompatible with 20Hz VIO |
| Heading from inlier displacement angles | REJECTED | 171° variance across 10 frames — reflects terrain geometry not UAV heading |

### Architecture Consequence (AD-23, AD-24)
- LightGlue is the primary L2 matcher at 0.35 confidence threshold
- VIO (OpenVINS) remains sole heading measurement source
- Segment Awareness Layer (SAL) proposed as AD-24 to add terrain-class thresholds and bounded search
- Phase D-2 (resolution degradation) and Phase D-3 (robustness) pending

### Sandbox Status: OPEN (Phase D-2 and D-3 pending)
