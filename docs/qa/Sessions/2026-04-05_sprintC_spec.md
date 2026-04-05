# Sprint C — Architecture Specification
**Date:** 2026-04-05  
**Type:** SPRINT  
**Open Items Resolved:** OI-05, OI-08, OI-11  
**Status:** Specification only — awaiting implementation

---

## What Sprint C Builds

Three deliverables, in strict dependency order:

1. `core/ins/orthophoto_matching_stub.py` — replaces the sensor model in `trn_stub.py`. The ESKF interface (measurement-provider-only pattern, `TRNCorrection` dataclass) is preserved exactly. Only the internals change.
2. `core/route_planner/hybrid_astar.py` — add terrain texture cost term. One new cost component, no change to the A* algorithm itself. Existing tests must continue to pass unchanged.
3. `tests/test_sprint_c_om_stub.py` — acceptance tests for the new stub, including the featureless terrain scenario (OI-11).

---

## Deliverable 1 — OrthophotoMatchingStub

**File:** `core/ins/orthophoto_matching_stub.py`  
**What it replaces:** The sensor model inside `trn_stub.py`. Specifically: `RadarAltimeterSim`, `DEMProvider`, and the elevation-strip NCC logic. The `TRNCorrection` dataclass and the measurement-provider-only pattern are preserved.  
**Design principle (AD-03):** The stub is a measurement provider only. It computes a position correction and returns it. The caller applies it via `eskf.update_vio()`. No internal Kalman filter.

### Synthetic Satellite Tile Provider

The stub must simulate preloaded satellite imagery tiles. It does not need real imagery — it needs a synthetic tile generator that produces texture scores and controlled match confidence values. This is analogous to how `DEMProvider` used sinusoidal terrain — we replace it with a texture-based provider.

```python
class SatelliteTileProvider:
    """
    Synthetic preloaded tile provider for SIL testing.
    
    Produces tiles characterised by sigma_terrain (texture score).
    High sigma_terrain = textured terrain = high match probability.
    Low sigma_terrain = featureless terrain = match suppressed.
    
    Parameters
    ----------
    sigma_terrain : float
        Terrain texture standard deviation in metres.
        >= 30 m : high texture (preferred by route planner)
        10-30 m : medium texture
        < 10 m  : featureless (match suppressed)
    seed : int
        Random seed for reproducibility.
    """
```

```
match_confidence = clip(
    gauss(mu_confidence(sigma_terrain), 0.05),
    0.0, 1.0
)
```

where `mu_confidence(sigma_terrain)`:

| Condition | mu |
|---|---|
| `sigma_terrain >= 30 m` | `mu = 0.82` (high texture, reliable match) |
| `10 <= sigma < 30 m` | `mu = 0.60` (medium texture, marginal) |
| `sigma_terrain < 10 m` | `mu = 0.25` (featureless, below threshold) |

### OM Parameters (named constants, not magic numbers)

```python
OM_MATCH_THRESHOLD = 0.65
OM_CORRECTION_INTERVAL_MIN_KM = 2.0
OM_CORRECTION_INTERVAL_MAX_KM = 5.0
OM_MAX_CONSECUTIVE_SUPPRESSED = 3
OM_FEATURELESS_SIGMA_THRESHOLD = 10.0   # m — below this: match suppressed
OM_PREFERRED_SIGMA_THRESHOLD = 30.0     # m — above this: high confidence
OM_R_NORTH = 9.0 ** 2                   # m² — measurement noise, NAV-02 v1.3
OM_R_EAST  = 9.0 ** 2                   # m² — replaces RADALT-NCC R=225 m²
```

`match_threshold = 0.65` — calibration placeholder per SRS v1.3 NAV-02 frozen constants — to be updated from SIL data, but must be a named constant, not a magic number.

### Interface

Must exactly match the existing `TRNCorrection` consumer pattern:

```python
@dataclass
class OMCorrection:
    """
    Output of OrthophotoMatchingStub.update().
    Consumed by caller via eskf.update_vio() or equivalent.
    """
    timestamp_s: float
    correction_north_m: float
    correction_east_m: float
    match_confidence: float          # 0.0–1.0
    correction_applied: bool
    consecutive_suppressed_count: int
    om_last_fix_km_ago: float        # 999.0 if no fix ever achieved
    sigma_terrain: float             # terrain texture of current tile
    r_matrix: np.ndarray             # 2x2 measurement noise, diag[OM_R_NORTH, OM_R_EAST]
```

`update()` method signature:

```python
def update(
    self,
    pos_north_m: float,
    pos_east_m: float,
    mission_km: float,
    sigma_terrain: float,     # provided by route planner / terrain model
) -> OMCorrection:
```

### Internal Logic

```
1. Check correction interval gate:
   if mission_km - last_fix_km < OM_CORRECTION_INTERVAL_MIN_KM:
       return OMCorrection(correction_applied=False, ...)

2. Sample match_confidence from tile provider using sigma_terrain

3. If match_confidence >= OM_MATCH_THRESHOLD:
   - Compute correction as small Gaussian noise on true position
     (in SIL, the "true" position is the input — correction is
      the residual from accumulated drift simulation)
   - Reset consecutive_suppressed_count to 0
   - Update last_fix_km
   - correction_applied = True
   
4. If match_confidence < OM_MATCH_THRESHOLD:
   - consecutive_suppressed_count += 1
   - correction_applied = False
   - Log OM_SUPPRESSED event (return in OMCorrection fields)
   
5. Return OMCorrection
```

### What NOT to Include

- No `RadarAltimeterSim` class
- No `DEMProvider` class
- No elevation strip NCC
- No internal Kalman filter (AD-03)
- No reference to RADALT anywhere in the file

### File Header

Must state:

```
Implements L2 Absolute Reset per Part Two V7.2 §1.7.2
Replaces RADALT-NCC TRN stub (AD-01, 03 April 2026)
Measurement-provider-only pattern (AD-03)
SIL stub — satellite tile provider is synthetic
OI-05 resolved by this file
```

---

## Deliverable 2 — Terrain Texture Cost Term in Route Planner

**File:** `core/route_planner/hybrid_astar.py`  
**What changes:** One new cost component added to the existing cost function. The A* algorithm itself is unchanged. All existing tests must continue to pass.

### New Constants (add to existing constants block)

```python
TEXTURE_COST_WEIGHT = 2.0         # multiplier for featureless terrain penalty
FEATURELESS_SIGMA_THRESHOLD = 10.0   # m — must match OM_FEATURELESS_SIGMA_THRESHOLD
PREFERRED_SIGMA_THRESHOLD = 30.0     # m — must match OM_PREFERRED_SIGMA_THRESHOLD
```

### New Cost Term Logic

```python
def terrain_texture_cost(sigma_terrain: float) -> float:
    """
    Returns a cost penalty for featureless terrain.
    
    Featureless terrain (sigma < 10 m) reduces orthophoto match
    frequency, increasing drift accumulation between OM resets.
    Route planner penalises these zones to prefer textured corridors.
    
    Returns
    -------
    float
        0.0  for high-texture terrain (sigma >= 30 m)
        0.5  for medium-texture terrain (10 <= sigma < 30 m)
        1.0  for featureless terrain (sigma < 10 m)
    """
    if sigma_terrain >= PREFERRED_SIGMA_THRESHOLD:
        return 0.0
    elif sigma_terrain >= FEATURELESS_SIGMA_THRESHOLD:
        return 0.5
    else:
        return 1.0
```

### Integration into Existing Cost Function

The existing cost function takes `ew_cost` as an input. Add `sigma_terrain` as a new optional parameter with default `30.0` (high texture — safe default that does not change existing test behaviour):

```python
def compute_cost(
    self,
    node,
    ew_cost: float,
    sigma_terrain: float = 30.0,   # NEW — default preserves existing behaviour
) -> float:
    existing_cost = ... # unchanged
    texture_penalty = TEXTURE_COST_WEIGHT * terrain_texture_cost(sigma_terrain)
    return existing_cost + texture_penalty
```

**Critical constraint:** The default `sigma_terrain=30.0` means all existing tests that do not pass `sigma_terrain` will behave exactly as before — zero texture penalty. This is mandatory. No existing test may change its result.

---

## Deliverable 3 — Acceptance Tests

**File:** `tests/test_sprint_c_om_stub.py`

### Required Test Cases — 8 minimum

| Test ID | Description | Key assertion |
|---|---|---|
| OM-01 | High texture terrain — match applied | `correction_applied=True`, `match_confidence >= 0.65` |
| OM-02 | Featureless terrain — match suppressed | `correction_applied=False`, `consecutive_suppressed_count >= 1` |
| OM-03 | Three consecutive suppressions — count correct | `consecutive_suppressed_count == 3` after 3 featureless calls |
| OM-04 | Correction interval gate — no correction before min interval | `correction_applied=False` when `mission_km` delta < 2.0 |
| OM-05 | OMCorrection dataclass fields all present | All 8 fields populated, `r_matrix.shape == (2,2)` |
| OM-06 | R matrix values correct | `r_matrix[0,0] == OM_R_NORTH`, `r_matrix[1,1] == OM_R_EAST` — confirms R=9² m² not old 225 m² |
| OM-07 | Featureless terrain — route planner cost higher than textured | `compute_cost(sigma=5) > compute_cost(sigma=40)` with same EW cost |
| OM-08 | Featureless terrain integration — no match for 10+ km (OI-11) | Run 20 consecutive updates over featureless zone (sigma=5), assert `correction_applied=False` throughout, `om_last_fix_km_ago > 10.0` |

**OM-06 is the critical QA gate.** The old `R_TRN_NORTH = 15.0**2 = 225 m²` was calibrated for RADALT-NCC. The new `OM_R_NORTH = 9.0**2 = 81 m²` reflects orthophoto MAE < 7 m accuracy. Any test that passes with R=225 but should use R=81 would silently underweight the correction. OM-06 catches this.

**OM-08 is the OI-11 closure gate.** This is the first test that exercises the featureless terrain failure mode — the scenario that was structurally untestable with the synthetic DEM (which was always textured). It must assert that no correction is applied over the featureless zone AND that `om_last_fix_km_ago` correctly accumulates.

---

## Acceptance Criteria for Sprint C

Sprint C is complete when all of the following are true:

| Gate | Criterion |
|---|---|
| SC-01 | `tests/test_sprint_c_om_stub.py` — 8/8 tests pass |
| SC-02 | `run_s5_tests.py` — 111/111 unchanged |
| SC-03 | `run_s8_tests.py` — 68/68 unchanged |
| SC-04 | `run_bcmp2_tests.py` — 90/90 unchanged |
| SC-05 | `tests/test_s5_l10s_se_adversarial.py` — 6/6 unchanged |
| SC-06 | `orthophoto_matching_stub.py` contains zero references to RADALT, DEMProvider, RadarAltimeterSim, or elevation strip |
| SC-07 | `hybrid_astar.py` existing tests all pass with default `sigma_terrain=30.0` |
| SC-08 | `OM_R_NORTH` and `OM_R_EAST` are `9.0**2`, not `15.0**2` |

---

## What Claude Code Must NOT Do

- Do not modify `trn_stub.py` — it stays as a frozen historical artefact. The new stub is a new file.
- Do not modify any frozen file.
- Do not modify any existing test.
- Do not import from `trn_stub.py` in the new stub.
- Do not add any internal Kalman filter to the stub (AD-03).
- Do not use `datetime.utcnow()` — use `datetime.now(timezone.utc)` (Python 3.12).
