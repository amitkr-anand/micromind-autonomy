# TECHNICAL_NOTES — scenarios/bcmp2

**Last Updated:** 11 April 2026 (Prompt 9 housekeeping — OI-23)

---

## AD-19 Velocity Check — System Rule 1.8

**System Rule 1.8:** No velocity-dependent control logic may exist in mission
runners. Navigation commands are issued as position/bearing intent; PX4 owns
velocity control.

**Result:** CLEAN — no violations found.

**Checked:**
- `scenarios/bcmp1/bcmp1_runner.py`
- `scenarios/bcmp2/bcmp2_runner.py`
- `scenarios/bcmp2/bcmp2_scenario.py`
- `scenarios/bcmp2/bcmp2_drift_envelopes.py`
- `scenarios/bcmp2/baseline_nav_sim.py`
- `scenarios/bcmp2/bcmp2_report.py`
- `scenarios/bcmp2/bcmp2_terrain_gen.py`

**Pattern searched:** `state\.v\b` and `\.velocity`

**Date:** 11 April 2026

**Authority:** OI-23, Code Governance Manual v3.2 §1.4, AD-19
