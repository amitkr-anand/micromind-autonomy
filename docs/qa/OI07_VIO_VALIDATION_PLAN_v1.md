# OI-07 Outdoor VIO Validation Plan v1.0
Authority: Deputy 1 | Date: 26 April 2026

## SRS Reference
NAV-03, PX4-03 §8.3
Pass criterion: VIO drift ≤ 0.5% of distance over any 1km segment.

## Current Evidence State
S-NEP tests confirm VIO mode switching and outage recovery.
No km-scale outdoor validation exists.
Compliance matrix carries NAV-03 as PARTIAL.
This gap blocks HIL readiness certification.

## Updated Assessment — UAV VisLoc Dataset
The UAV VisLoc dataset (11 sites, satellite imagery pairs) is
available at:
/home/mmuser/micromind_data/raw/imagery/uav_visloc/UAV_VisLoc/
This dataset provides ground-truth satellite reference images with
known positions — directly usable for VIO accuracy measurement
without requiring a physical outdoor rig test.

## What Constitutes Valid Outdoor Evidence

### Tier 1 — SIL-adjacent (UAV VisLoc dataset):
- Use UAV VisLoc satellite imagery as reference
- Run VIO stack on synthetic trajectory over known imagery
- Measure drift vs ground truth at 100m, 250m, 500m intervals
- Pass criterion: drift ≤ 0.5% of distance at each checkpoint
- Estimated effort: 1 session

### Tier 2 — HIL-required (real outdoor):
- Jetson Orin NX running production VIO stack
- Real EO camera at AVP-02 GSD (≤0.5 m/px at 150m AGL)
- 5km minimum run, 3 independent runs (varied lighting)
- GPS RTK ground truth
- Blocked until OI-43 camera driver resolved

## Blocking Dependencies
- OI-43: camera driver (Tier 2 only — Tier 1 can proceed now)
- Orin SSH confirmed stable at 451b92b (Tier 2 ready when OI-43 resolved)

## Proposed Sequence
1. Tier 1: UAV VisLoc SIL validation — unblocked, next sprint
2. Tier 2: Orin outdoor — after OI-43 camera driver resolved
3. Full HIL VIO: Shimla corridor — after Tier 2 passes

## Status
PLANNING COMPLETE. Tier 1 unblocked. Tier 2 blocked on OI-43.
