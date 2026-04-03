# MicroMind Architecture Decisions Register
**Format:** One entry per decision. Immutable once recorded — amendments create new entries.  
**Reference:** Complements DD-01 and DD-02 in Part Two V7.

---

## AD-01 — Navigation Ingress Correction Mechanism
**Date:** 03 April 2026  
**Status:** ADOPTED  
**Owner:** Programme Director

### Decision
Replace RADALT-NCC TRN as the primary ingress correction mechanism with Orthophoto Image Matching against preloaded satellite imagery tiles.

### Context
Part Two V7 specified TRN via radar altimeter (0.5–50 m AGL range) + Normalised Cross-Correlation against DEM elevation profile. This specification was found to be inconsistent with AVP-02/03/04 cruise altitudes (100–2,000 m AGL), where a conventional RADALT beam footprint spans 9–13 DEM grid cells, averaging out terrain relief and degrading correction accuracy.

Research review (03 April 2026) established that orthophoto image matching achieves MAE < 7 m across 20 experimental scenarios including night-time LWIR operation, with no additional hardware required. The system uses the existing downward-facing EO/LWIR camera already specified for DMRL.

### Architecture
**Three-layer navigation stack:**
- L1 (Relative): IMU + VIO (OpenVINS) — high-rate pose, ~1 m/km drift
- L2 (Absolute Reset): Orthophoto matching vs preloaded satellite tiles — hard position reset every 2–5 km over textured terrain, MAE < 7 m, no accumulated error
- L3 (Vertical Stability): Baro-INS — damps vertical channel divergence, no terrain sensing function

**RADALT retained** for terminal phase only: 0–300 m AGL, final 5 km, altitude input for aimpoint offset computation.

**LWIR camera dual-use:** orthophoto matching during ingress (L2); DMRL decoy rejection during terminal.

### Consequences
- `trn_stub.py` must be updated to reflect orthophoto matching as the L2 mechanism (OI-05)
- SRS NAV-02 test cases must be rewritten: correction mechanism is image matching, not NCC altimetry. Pass criterion (< 50 m CEP-95) unchanged.
- Route planner (hybrid_astar.py) requires terrain-texture cost term to penalise featureless zones (OI-08)
- V7 RADALT spec must be scoped to terminal phase only (remove from ingress navigation section)
- Storage: ~10–15 GB satellite tiles for 150 km radius at 1 m resolution — within 32 GB eMMC spec

### Risks
- Image matching fails over featureless terrain (Thar Desert flat, Himalayan snowfield). Mitigated by route planner texture cost and VIO bridging.
- Night operation requires LWIR image matching against visible-light orthophotos — cross-spectral matching tested and demonstrated in literature but not yet validated in MicroMind SIL.

### References
- Yao et al. (2024), GNSS-Denied Geolocalization using Terrain-Weighted Constraint Optimization
- OKSI OMNInav system architecture
- ICRA 2001 Sinopoli et al. (hierarchical DEM + probabilistic navigation)

---

## AD-02 — IMU Specification Floor Correction
**Date:** 03 April 2026 (flagged in S8, 27 February 2026)  
**Status:** PENDING SPEC UPDATE  
**Owner:** Spec author

### Decision
Update Part Two V7 IMU ARW floor from ≤ 0.1°/√hr to ≤ 0.2°/√hr.

### Context
S8 sensor characterisation established that the STIM300 (primary tactical IMU candidate) has a typical ARW of 0.15°/√hr, which exceeds the V7 spec floor of 0.1°/√hr. The ADIS16505-3 (MEMS candidate) is at 0.22°/√hr. The BASELINE model used for BCMP-2 C-2 envelope calibration is 0.05°/√hr — a value no real candidate sensor achieves.

### Consequences
- V7 spec update required before TASL meeting
- C-2 envelopes must be re-validated with STIM300 noise profile once ALS-250 overnight run data is available (OI-03)
- Any BCMP-2 results presented externally must note that C-2 envelopes were calibrated on BASELINE IMU, not STIM300

---
*Append new decisions above the final line.*
