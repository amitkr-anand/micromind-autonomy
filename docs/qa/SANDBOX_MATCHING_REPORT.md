
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
