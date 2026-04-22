# Synthetic Terrain Data — Infrastructure Caveat

**Status:** SYNTHETIC TEST INFRASTRUCTURE — NOT REAL TERRAIN DATA
**Created:** 22 April 2026
**Ref:** git commit 3dc15b8 (git filter-repo terrain blob removal)
**Governance ref:** MicroMind_SRS_v1_3.docx §15 Synthetic Terrain Note; MICROMIND_PROJECT_CONTEXT.md §4

## What This Directory Contains

The `.tif` files in this directory are synthetic EPSG:4326 replacement
tiles created after `git filter-repo` removed the original 3.7 GB GLO-30
real terrain blobs from repository history.

These synthetic tiles exist solely to allow the SIL regression suite
(`run_certified_baseline.sh`) to load tile files without `FileNotFoundError`.
They do NOT represent real terrain elevation data.

## What These Tiles Are NOT

- NOT real GLO-30 (Copernicus DEM 30m) elevation data
- NOT real CARTOSAT or equivalent satellite imagery
- NOT valid inputs for navigation performance assessment
- NOT representative of the Shimla-Manali, Jammu-Leh, or any other
  Indian corridor terrain profile

## Navigation Performance Claims

All navigation performance claims in programme documents
(including `PREHIL_NAV_SPECIFICATION.md`, BCMP-2 reports, and SRS §14
traceability entries for NAV-01, NAV-02, NAV-06) were validated using
**real terrain data** on `micromind-node01` before the git filter-repo
operation. The programme evidence base is unaffected.

## Bias Warning

The LightGlue L2 matcher (AD-23, AD-24) was characterised on real
structured terrain (UAV-VisLoc dataset, Shimla corridor). Its acceptance
rates and confidence thresholds are calibrated for:
- Structured terrain: roads, buildings, field boundaries, industrial zones
- Altitude: 400–800 m AGL
- GSD: approximately 0.3–1.5 m/px at operational altitudes

Synthetic tiles introduce an uncharacterised structured-terrain bias.
Tests that exercise the LightGlue or NCC matching pipeline using
synthetic tiles do NOT provide evidence of real-terrain matching
performance. Gate 7 (`test_gate7_sal_corridor.py`) validates SAL
suppression logic and search-radius scaling — it does not validate
matching accuracy.

## Prohibited Uses of Synthetic Tiles

1. Citing SIL test results using these tiles as evidence of navigation accuracy
2. Presenting tile-dependent test outputs in TASL-facing or DRDO-facing materials
3. Deriving drift envelopes, CEP figures, or matching confidence statistics

## Real Terrain Acquisition

Real GLO-30 tiles for the Shimla-Manali and Jammu-Leh corridors must
be re-acquired from Copernicus Open Access Hub (or equivalent) for
any navigation performance validation run. Tiles must NOT be
committed to the repository — they are listed in `.gitignore`.
