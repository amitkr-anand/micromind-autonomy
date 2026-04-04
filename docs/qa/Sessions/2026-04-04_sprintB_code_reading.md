# Sprint B Step 1 — Code Reading Session
**Date:** 2026-04-04  
**Type:** Code reading only — no code changes, no test changes  
**Scope:** `core/l10s_se/l10s_se.py`, `core/dmrl/dmrl_stub.py`, `tests/test_s5_l10s_se.py`, `tests/test_s5_dmrl.py`

---

## 1. `core/l10s_se/l10s_se.py` — L10s-SE Decision Engine

### Inputs (`L10sInputs` dataclass)

| Field | Type | Source | Notes |
|---|---|---|---|
| `lock_acquired` | bool | DMRL | True if confidence ≥ 0.85 was sustained |
| `lock_confidence` | float | DMRL | 0–1 |
| `is_decoy` | bool | DMRL | True if 3 consecutive frames ≥ 0.80 decoy probability |
| `decoy_confidence` | float | DMRL | 0–1 |
| `lock_lost_timeout` | bool | DMRL | True if re-acq timed out (> 1.5 s) |
| `civilian_confidence` | float | Mission envelope / real-time sense | 0–1; ≥ 0.70 in any frame → abort |
| `corridor_violation` | bool | Mission envelope | True if predicted trajectory exits envelope |
| `pre_terminal_zpi_complete` | bool | ZPI/DD-02 | Mandatory pre-terminal burst confirmation |
| `activation_timestamp` | float | Caller | Monotonic time at activation |

`civilian_confidence` is explicitly **not produced by DMRL**. It arrives from a separate sensor/mission-envelope channel. The `inputs_from_dmrl()` helper defaults it to 0.0 unless the caller passes it.

### Decision Tree (exact gate order)

```
Gate 0:  pre_terminal_zpi_complete == False  → ABORT / NO_LOCK
Gate 1a: lock_lost_timeout == True           → ABORT / LOCK_LOST_TIMEOUT
Gate 1b: lock_acquired == False              → ABORT / NO_LOCK
Gate 1c: lock_confidence < 0.85             → ABORT / NO_LOCK
Gate 2:  is_decoy == True                   → ABORT / DECOY_DETECTED
Gate 3:  civilian_confidence >= 0.70        → ABORT / CIVILIAN_DETECTED
Gate 4:  corridor_violation == True         → ABORT / CORRIDOR_VIOLATION
Gate 5:  elapsed > 2.0 s                    → ABORT / DECISION_TIMEOUT
All clear                                   → CONTINUE / NONE
```

Priority is strict: each gate short-circuits before the next is evaluated.  
A later gate's condition is never visible if an earlier gate fires.

### PASS vs ABORT

- **CONTINUE** (PASS): all 6 gates clear, timing within 2 s.
- **ABORT**: any single gate condition is true. The `abort_reason` field records which gate fired first.
- `l10s_compliant` is True only when both `decision_compliance` (latency ≤ 2 s) and `window_compliance` (latency ≤ 10 s) are both satisfied.

### Civilian Detection Path

Gate 3 is the sole civilian check. Trigger condition: `civilian_confidence >= 0.70` (inclusive).  
This fires **after** lock and decoy gates — meaning a civilian abort is only reachable if:
1. ZPI was confirmed (Gate 0 passed)
2. Lock was not timed out (Gate 1a passed)
3. Lock was acquired with confidence ≥ 0.85 (Gates 1b/1c passed)
4. No decoy was flagged (Gate 2 passed)

Consequence: **a scenario where lock_confidence < 0.85 AND civilian_confidence ≥ 0.70 simultaneously will abort at Gate 1c, never reaching Gate 3.** The civilian gate is structurally unreachable in that combined condition.

---

## 2. `core/dmrl/dmrl_stub.py` — DMRL Processor

### `generate_synthetic_scene()` output

Produces a `list[ThermalTarget]`. Each `ThermalTarget` has:
- `target_id`, `is_decoy`, `thermal_signature`, `thermal_decay_rate`, `initial_roi_px`, `bearing_deg`, `range_m`

**Does not produce `civilian_confidence`.** That field has no representation in the scene generator.

### Parameters

| Parameter | Default | Effect |
|---|---|---|
| `n_targets` | 1 | Number of real (non-decoy) targets |
| `n_decoys` | 1 | Number of thermal decoys |
| `seed` | None | RNG seed for reproducibility |

### Clean vs Adversarial

- **Clean:** `n_decoys=0` — only real targets, no thermal countermeasures.
- **Adversarial:** `n_decoys >= 1` and/or `inject_lock_loss_at` used in `process_target()`.
- Note: there is no `civilian` parameter. Civilian presence is never modelled in the scene generator.

### Lock Confidence Calculation (real target)

```python
sig = thermal_signature - thermal_decay_rate * frame_id   # thermal at this frame
temporal_factor = min(1.0, dwell_frames / 5)              # ramps 0→1 over first 5 frames
roi_px = max(1, initial_roi_px - frame_id // 4)
roi_factor = min(1.0, roi_px / 8)
noise = gauss(0.0, 0.025)
raw = min(1.0, sig + noise)
confidence = raw * temporal_factor * roi_factor           # ∈ [0, 1]
```

Lock is acquired when: `dwell_frames >= 5` AND `lock_confidence >= 0.85`.  
For decoys the formula includes an additional multiplier of `uniform(0.72, 0.84)` and a triple-rate decay penalty, driving confidence below threshold as frames accumulate.

---

## 3. `tests/test_s5_l10s_se.py` — Test Inventory

All tests use manually constructed `L10sInputs` via `_valid_inputs(**overrides)`.  
Base valid state: `lock_acquired=True, lock_confidence=0.92, is_decoy=False, decoy_confidence=0.10, lock_lost_timeout=False, civilian_confidence=0.05, corridor_violation=False, pre_terminal_zpi_complete=True`.  
All inputs are **clean synthetic** (manually set, no DMRL pipeline, no adversarial EO conditions).

### TestL10sBoundaryConstants (5 tests)

| Test | Stimulus | Asserts | Inputs |
|---|---|---|---|
| `test_decision_timeout_is_2s` | Read constant | `DECISION_TIMEOUT_S == 2.0` | — |
| `test_window_is_10s` | Read constant | `L10S_WINDOW_S == 10.0` | — |
| `test_lock_threshold_matches_dmrl` | Read constant | `LOCK_CONFIDENCE_THRESHOLD == 0.85` | — |
| `test_civilian_threshold_is_0_7` | Read constant | `CIVILIAN_DETECT_THRESHOLD == 0.70` | — |
| `test_reacquisition_timeout_is_1_5s` | Read constant | `REACQUISITION_TIMEOUT_S == 1.5` | — |

### TestL10sGate0ZPI (2 tests)

| Test | Stimulus | Asserts | Inputs |
|---|---|---|---|
| `test_gate0_abort_if_zpi_not_complete` | `pre_terminal_zpi_complete=False` | ABORT | Clean (override ZPI only) |
| `test_gate0_pass_if_zpi_complete` | `pre_terminal_zpi_complete=True` | CONTINUE | Clean |

### TestL10sGate1LockAcquired (6 tests)

| Test | Stimulus | Asserts | Inputs |
|---|---|---|---|
| `test_gate1_abort_if_no_lock` | `lock_acquired=False, lock_confidence=0.70` | ABORT/NO_LOCK | Clean |
| `test_gate1_abort_if_lock_confidence_below_threshold` | `lock_acquired=True, lock_confidence=0.84` | ABORT/NO_LOCK | Clean |
| `test_gate1_abort_exactly_at_boundary_minus_epsilon` | `lock_confidence=0.8499` | ABORT | Clean |
| `test_gate1_continue_at_threshold` | `lock_confidence=0.850` | CONTINUE | Clean |
| `test_gate1_abort_if_lock_lost_timeout` | `lock_lost_timeout=True, lock_confidence=0.95` | ABORT/LOCK_LOST_TIMEOUT | Clean |
| `test_gate1_lock_lost_timeout_takes_precedence_over_good_lock` | `lock_lost_timeout=True, lock_confidence=0.99` | ABORT/LOCK_LOST_TIMEOUT | Clean |

### TestL10sGate2Decoy (3 tests)

| Test | Stimulus | Asserts | Inputs |
|---|---|---|---|
| `test_gate2_abort_if_decoy_detected` | `is_decoy=True, decoy_confidence=0.88` | ABORT/DECOY_DETECTED | Clean |
| `test_gate2_continue_if_no_decoy` | `is_decoy=False, decoy_confidence=0.10` | CONTINUE | Clean |
| `test_gate2_abort_even_with_high_lock_confidence` | `is_decoy=True, lock_confidence=0.97` | ABORT/DECOY_DETECTED | Clean |

### TestL10sGate3Civilian (5 tests)

| Test | Stimulus | Asserts | Inputs |
|---|---|---|---|
| `test_gate3_abort_if_civilian_confidence_at_threshold` | `civilian_confidence=0.70` | ABORT/CIVILIAN_DETECTED | Clean (good lock, no decoy) |
| `test_gate3_abort_if_civilian_confidence_above_threshold` | `civilian_confidence=0.85` | ABORT/CIVILIAN_DETECTED | Clean (good lock, no decoy) |
| `test_gate3_continue_below_civilian_threshold` | `civilian_confidence=0.69` | CONTINUE | Clean |
| `test_gate3_abort_at_0_699_boundary` | `civilian_confidence=0.699` | CONTINUE | Clean |
| `test_gate3_abort_at_exactly_0_700` | `civilian_confidence=0.700` | ABORT | Clean |

### TestL10sGate4Corridor (3 tests)

| Test | Stimulus | Asserts | Inputs |
|---|---|---|---|
| `test_gate4_abort_if_corridor_violation` | `corridor_violation=True` | ABORT/CORRIDOR_VIOLATION | Clean |
| `test_gate4_continue_if_corridor_clear` | `corridor_violation=False` | CONTINUE | Clean |
| `test_gate4_corridor_abort_overrides_high_confidence` | `corridor_violation=True, lock_confidence=0.99, civilian_confidence=0.01` | ABORT/CORRIDOR_VIOLATION | Clean |

### TestL10sTimingCompliance (5 tests)

| Test | Stimulus | Asserts | Inputs |
|---|---|---|---|
| `test_decision_latency_within_2s_continue` | Valid CONTINUE inputs | latency ≤ 2 s | Clean |
| `test_decision_latency_within_2s_abort_no_lock` | `lock_acquired=False` | latency ≤ 2 s | Clean |
| `test_decision_latency_within_2s_abort_decoy` | `is_decoy=True` | latency ≤ 2 s | Clean |
| `test_l10s_compliant_flag_true_when_timing_met` | Valid CONTINUE | `l10s_compliant=True` | Clean |
| `test_100_consecutive_runs_all_timing_compliant` | 100× valid CONTINUE | 100% compliance | Clean |

### TestL10sAuditLog (7 tests)

All clean inputs. Tests audit log structure, non-emptiness, timestamps, event strings, activation record, abort reason recording, and log independence between evaluations.

### TestL10sDecisionPriority (4 tests)

| Test | Stimulus | Asserts | Notes |
|---|---|---|---|
| `test_zpi_abort_beats_all_other_conditions` | `zpi=False`, everything else valid | ABORT | Clean overrides |
| `test_lock_lost_timeout_beats_decoy_flag` | `lock_lost_timeout=True, is_decoy=True` | LOCK_LOST_TIMEOUT | Two concurrent faults |
| `test_decoy_abort_beats_civilian_detection` | `is_decoy=True, civilian_confidence=0.80` | DECOY_DETECTED | `civilian_confidence=0.80 ≥ 0.70`, but decoy fires first at Gate 2 |
| `test_civilian_abort_beats_corridor` | `civilian_confidence=0.85, corridor_violation=True` | CIVILIAN_DETECTED | Two concurrent faults |

### TestInputsFromDMRL (2 tests)

| Test | Stimulus | Asserts | Inputs |
|---|---|---|---|
| `test_helper_builds_valid_inputs` | Mock DMRL + `civilian_confidence=0.04` | Fields correctly propagated | Clean |
| `test_helper_defaults_are_safe` | Mock DMRL with `lock_acquired=False, lock_lost_timeout=True` | ABORT | Clean/unsafe state |

---

## 4. `tests/test_s5_dmrl.py` — Test Inventory

Tests use manually constructed `ThermalTarget` objects or `generate_synthetic_scene()`.  
No test injects `civilian_confidence`; DMRL tests have no civilian path.

### TestDMRLBoundaryConditions (9 tests)

| Test | Stimulus | Asserts | Inputs |
|---|---|---|---|
| `test_bc01_lock_confidence_threshold_is_correct` | Read constant | `LOCK_CONFIDENCE_THRESHOLD == 0.85` | — |
| `test_bc01_lock_not_acquired_below_threshold` | `thermal_signature=0.40` (weak target) | `lock_acquired=False` | Clean (no decoy) |
| `test_bc01_lock_acquired_above_threshold` | `thermal_signature=0.90`, 50 seeds | lock rate ≥ 80% | Clean |
| `test_bc02_decoy_abort_threshold_is_correct` | Read constants | `DECOY_ABORT_THRESHOLD==0.80`, `DECOY_ABORT_CONSECUTIVE==3` | — |
| `test_bc02_decoy_detected_and_flagged` | `is_decoy=True, decay_rate=0.030`, 50 seeds | detection rate ≥ 85% | Adversarial (decoy) |
| `test_bc02_real_target_not_flagged_as_decoy` | `is_decoy=False, thermal_signature=0.85`, 50 seeds | false-positive rate ≤ 15% | Clean |
| `test_bc03_min_dwell_frames_constant` | Read constant | `MIN_DWELL_FRAMES == 5` | — |
| `test_bc03_frame_rate_is_25fps` | Read constant | `FRAME_RATE_HZ == 25.0` | — |
| `test_bc03_lock_not_acquired_before_min_dwell` | Normal target | `dwell_frames >= 5` if lock acquired | Clean |
| `test_bc04_min_roi_constant` | Read constant | `MIN_THERMAL_ROI_PX == 8` | — |
| `test_bc04_small_roi_target_handles_gracefully` | `initial_roi_px=4` | No exception; returns `DMRLResult` | Edge case |
| `test_bc05_reacquisition_timeout_constant` | Read constant | `REACQUISITION_TIMEOUT_S == 1.5` | — |
| `test_bc05_lock_loss_timeout_sets_flag` | `inject_lock_loss_at=0`, 50 frames | `lock_lost_timeout=True` | Adversarial (lock loss) |
| `test_bc06_aimpoint_correction_limit_constant` | Read constant | `AIMPOINT_CORRECTION_LIMIT == 15.0` | — |
| `test_bc06_all_frame_corrections_within_limit` | `bearing_deg=12.0`, 30 frames | all `|correction| ≤ 15°` | Clean |

### TestDMRLSceneProcessing (5 tests)

| Test | Stimulus | Asserts | Inputs |
|---|---|---|---|
| `test_scene_generation_correct_counts` | `n_targets=1, n_decoys=1, seed=0` | 1 real, 1 decoy | Adversarial scene |
| `test_scene_decoy_has_higher_decay_rate` | 20 seeds, 1+1 scenes | `decoy.decay_rate > real.decay_rate` | Adversarial scene |
| `test_process_scene_returns_all_targets` | `seed=7`, 1+1 scene | result has key for every target | Adversarial scene |
| `test_select_primary_target_returns_non_decoy` | 20 seeds, 1+1 scenes | selected target `is_decoy=False` | Adversarial scene |
| `test_select_primary_returns_none_when_all_decoys` | Manually forced all-decoy results | returns `None` | Adversarial (manual override) |

### TestDMRLKPIRequirements (2 tests)

| Test | Stimulus | Asserts | Inputs |
|---|---|---|---|
| `test_kpi_t01_terminal_lock_rate_50_runs` | `generate_synthetic_scene()`, 50 seeds | lock rate ≥ 85% | Adversarial (1 real + 1 decoy per run) |
| `test_kpi_t02_decoy_rejection_rate_50_runs` | Manually constructed decoy-only scene, 50 seeds | rejection rate ≥ 90% | Adversarial (decoy only) |

### TestDMRLResultStructure (3 tests)

All clean inputs (real target, no decoy). Tests field types, temporal ordering of frames, and confidence value ranges.

---

## Answers to Precise Questions

### Q1: Range of `civilian_confidence` values that `generate_synthetic_scene()` can produce, and under what conditions?

**None. `generate_synthetic_scene()` never produces `civilian_confidence`.**

The function returns `list[ThermalTarget]`. `ThermalTarget` has no `civilian_confidence` field. Civilian confidence is a separate `L10sInputs` field that arrives from the mission envelope / real-time EO sensor path entirely outside the DMRL pipeline. In the `inputs_from_dmrl()` helper it defaults to `0.0`. In all existing tests it is injected manually by the test author. The scene generator has no awareness of civilian presence at all.

### Q2: Minimum `lock_confidence` value that causes L10s-SE to proceed (not abort)?

**Exactly 0.85** (`LOCK_CONFIDENCE_THRESHOLD = 0.85`).

Gate 1c check is `if inputs.lock_confidence < LOCK_CONFIDENCE_THRESHOLD → ABORT`. The boundary is inclusive: `0.8499` aborts, `0.850` proceeds. This is verified by:
- `test_gate1_abort_exactly_at_boundary_minus_epsilon` (0.8499 → ABORT)  
- `test_gate1_continue_at_threshold` (0.850 → CONTINUE)

### Q3: Is there any existing test where `civilian_confidence` ≥ 0.70 AND a genuine target is present simultaneously?

**Partially yes — but only in isolated L10s-SE unit tests, not through the DMRL pipeline.**

In `test_s5_l10s_se.py`, the Gate 3 tests inject `civilian_confidence=0.70/0.85/0.700` while the base state includes `lock_acquired=True, is_decoy=False, lock_confidence=0.92` (i.e. a genuine target state). These tests confirm the abort fires correctly.

However: these are hand-constructed `L10sInputs` — no DMRL processing, no `generate_synthetic_scene()`, no end-to-end integration. There is no test that runs a real DMRL scene (genuine thermal target) **and** feeds a non-zero `civilian_confidence` into the subsequent L10s-SE evaluation. The integration path (DMRL result → `inputs_from_dmrl()` → L10s-SE) is only tested with `civilian_confidence=0.04` in `test_helper_builds_valid_inputs`, which is well below the abort threshold.

**This is a gap.** OI-26 ("L10s-SE adversarial EO condition tests absent") correctly identifies this as a QA standing rule violation.

### Q4: Is there any existing test where `lock_confidence` is below threshold AND a decoy is present AND a civilian is within the L10s window simultaneously?

**No. This three-way adversarial combination is not tested anywhere.**

Surveying all tests:
- Tests with `is_decoy=True` use `lock_confidence=0.88–0.97` (all at or above threshold).
- `test_decoy_abort_beats_civilian_detection` has `is_decoy=True` and `civilian_confidence=0.80`, but `lock_confidence` defaults to `0.92` (above threshold).
- No test combines `lock_confidence < 0.85` + `is_decoy=True` + `civilian_confidence ≥ 0.70`.

Furthermore, the decision tree makes this three-way combination partially moot: if `lock_confidence < 0.85`, Gate 1c aborts before Gate 2 (decoy) or Gate 3 (civilian) are evaluated. The civilian and decoy checks are structurally unreachable in that scenario. But this exact structural assumption — that the gate ordering prevents simultaneous evaluation — is itself never explicitly tested. A test asserting "low lock + decoy + civilian → abort at Gate 1c (not Gate 2 or 3)" does not exist.

---

## QA Observations

1. **No adversarial EO inputs in any DMRL or L10s-SE test** (OI-26): all civilian_confidence values are either 0.05 (safe baseline) or injected as clean scalars in isolation. The EO sensor path that produces civilian_confidence in flight is not modelled.

2. **`civilian_confidence` has no generator**: the absence of this field from `generate_synthetic_scene()` means integration tests between DMRL and L10s-SE always require manual injection. The risk is that developers default to 0.0 and the civilian abort gate is never exercised in mission-level tests.

3. **Gate 3 is only reachable under a very specific precondition set**: civilian abort can only fire after Gates 0–2 all pass. No test exercises the full pass-through path (Gates 0–2 clear, Gate 3 fires) with realistic DMRL-generated inputs.

4. **DMRL is a stub** (OI-06): all lock confidence and decoy probability values are rule-based approximations, not CNN outputs. All test results referencing terminal guidance are stub-based.

---

*No code modified. No tests modified. Session note only.*
