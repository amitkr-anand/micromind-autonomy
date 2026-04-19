
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

## Phase D-2: Reference Resolution Degradation (19 April 2026)
Site 04, 60 frames, structured terrain, 540m AGL

Degradation method: INTER_AREA downsample → INTER_LINEAR upsample to 1280px

| Resolution | Accept | Rate  | Mean GT err | Mean conf | Mean inliers |
|-----------|--------|-------|------------|-----------|-------------|
| 0.28m/px  | 56/60  | 93.3% | 42.4m      | baseline  | baseline    |
| 1.0m/px   | 50/60  | 83.3% | 48.9m      | 0.608     | 192.3       |
| 2.5m/px   | 46/60  | 76.7% | 44.1m      | 0.568     | 136.8       |
| 5.0m/px   | 19/60  | 31.7% | 48.8m      | 0.440     | 73.7        |
| 10.0m/px  | 0/60   | 0.0%  | —          | —         | —           |

**Crossover (50% accept): between 2.5m/px and 5.0m/px**
**Minimum viable satellite reference resolution: ≤3m/px**
**Key finding: mean GT error stable (42-49m) across accepted frames at all resolutions
  — LightGlue either accepts cleanly or rejects entirely, no graceful degradation.**
**Programme implication: Sentinel-2 10m/px NOT viable for LightGlue.
  CARTOSAT-1 2.5m/px is viable (76.7%). Google Earth 0.28m/px is optimal.**

## Phase D-3: Robustness Testing (19 April 2026)
Site 04, 30 frames, structured terrain, 540m AGL

### D-3a: Heading Mismatch Tolerance

| Offset | Accept | Rate  | Mean err | Note |
|--------|--------|-------|---------|------|
| -45°   | 1/30   | 3.3%  | 77.9m   | collapse |
| -20°   | 17/30  | 56.7% | 55.4m   | borderline |
| -10°   | 22/30  | 73.3% | 51.8m   | degraded |
| -5°    | 27/30  | 90.0% | 49.3m   | acceptable |
| 0°     | 28/30  | 93.3% | 43.1m   | baseline |
| +5°    | 28/30  | 93.3% | 38.0m   | nominal |
| +10°   | 28/30  | 93.3% | 34.2m   | nominal |
| +20°   | 28/30  | 93.3% | 32.9m   | dataset artifact* |
| +45°   | 26/30  | 86.7% | 39.3m   | degrading |

*Positive offset improvement at +10/+20° is a dataset artifact — southward-flying
frames interact with north-alignment rotation. Not a genuine capability gain.

**VIO heading budget: ±10° for reliable operation (73-93% accept rate range)**
**Negative heading errors are more damaging than positive due to site heading distribution**

### D-3b: FOV Sensitivity

| FOV | Accept | Rate  | Mean err | eff_gsd   |
|-----|--------|-------|---------|-----------|
| 30° | 26/30  | 86.7% | 29.4m   | 0.299m/px |
| 45° | 26/30  | 86.7% | 35.6m   | 0.352m/px |
| 60° | 28/30  | 93.3% | 43.1m   | 0.491m/px |
| 75° | 21/30  | 70.0% | 55.6m   | 0.652m/px |
| 90° | 15/30  | 50.0% | 89.4m   | 0.849m/px |

**Operational setting: FOV 60° (max accept rate criterion)**
**Note: FOV 30° gives 29.4m mean error — better accuracy but 7pp lower accept rate**

## Consolidated LightGlue Operating Parameters (Deputy 1, 19 April 2026)
| Parameter           | Value  | Basis       |
|--------------------|--------|-------------|
| Confidence threshold| ≥0.35  | Phase D-1   |
| FOV setting        | 60°    | Phase D-3b  |
| VIO heading budget | ±10°   | Phase D-3a  |
| Min satellite res  | ≤3m/px | Phase D-2   |
| Working resolution | 1280px | TASL camera |
| Min terrain class  | Structured (roads/buildings) | Phase D-1 |

## HIL Validation (19 April 2026)

### H-3: LightGlue on Orin Nano Super GPU
- Steady-state: 628ms median, 1630ms P99
- Budget (2km@27m/s): 74,000ms — 45× margin at P99
- Slowdown vs dev (RTX 5060 Ti): 12.4×
- TensorRT optimisation: NOT required

### H-4: LightGlue IPC Bridge
- Mechanism: Unix socket AF_UNIX, 1.0ms IPC overhead
- Interface: match(frame_path, lat, lon, alt, heading) → (dlat, dlon, conf, ms)
- T2 real Site 04: conf=0.743, 3192ms cold-start, 93.9m correction
- satellite04.tif bounds: 119.906–119.955E / 32.151–32.254N
- Status: FULL PASS

## Sandbox Status: OPEN
Phase D-1/D-2/D-3: COMPLETE
Pending: SAL-3 sandbox (Jammu-Leh SUPPRESS zones)
