
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
