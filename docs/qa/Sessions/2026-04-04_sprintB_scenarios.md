# Sprint B — Adversarial Test Scenario Specification
**Date:** 2026-04-04  
**Type:** Specification only — no test code  
**Closes:** OI-26 (L10s-SE adversarial EO condition tests absent)  
**Follows:** `2026-04-04_sprintB_code_reading.md`

---

## Context

Step 1 reading established:

- `generate_synthetic_scene()` never produces `civilian_confidence`. That field exists only on `L10sInputs`, not on any DMRL data structure.
- `inputs_from_dmrl()` defaults `civilian_confidence` to `0.0` unless the caller overrides it.
- Every Gate 3 (civilian) test in `test_s5_l10s_se.py` is hand-constructed: it bypasses the DMRL pipeline entirely.
- No test has ever run `generate_synthetic_scene()` → `DMRLProcessor.process_target()` → `inputs_from_dmrl()` → `L10sSafetyEnvelope.evaluate()` with `civilian_confidence > 0.0`.
- Gate 3 has therefore never been reached through the full integration path in any existing test.

These six scenarios close that gap. Each drives the full pipeline: DMRL processes a synthetic scene, the result is handed to `inputs_from_dmrl()` with an explicit `civilian_confidence` override, and L10s-SE evaluates it.

---

## SRS Requirement Mapping

| ID | Requirement | Text (summary) |
|---|---|---|
| TERM-01 | FR-105 §1.12 | Civilian detection confidence ≥ 0.70 in any frame → abort |
| TERM-02 | FR-103 §1.9.3 | Lock confidence ≥ 0.85 required to proceed |
| TERM-03 | FR-103 §1.9.3 | Decoy flagged over 3 consecutive frames → abort before civilian gate |

---

## Adversarial Scenario Specifications

---

### ADV-01 — Civilian Detected via Full DMRL Pipeline

**Gap it closes:** OI-26; QA standing rule 2 — Gate 3 has never been exercised through the DMRL integration path  
**SRS requirement:** TERM-01 (civilian abort gate)

**Stimulus:**

| Parameter | Value | Rationale |
|---|---|---|
| `generate_synthetic_scene()` | `n_targets=1, n_decoys=0, seed=42` | Single clean real target; no decoy so Gates 0–2 all pass |
| DMRL processing | `DMRLProcessor.process_target(target, max_frames=30)` | Full pipeline run; lock should be acquired (thermal_sig ≥ 0.87) |
| `civilian_confidence` injected into `inputs_from_dmrl()` | `0.82` | Well above 0.70 threshold; represents civilian thermal signature in terminal frame |
| `corridor_violation` | `False` | Isolates Gate 3 — no other abort trigger present |
| `pre_terminal_zpi_complete` | `True` | Gate 0 passes |

**Expected outcome:** ABORT  
**Expected abort reason:** `CIVILIAN_DETECTED`  
**Gate that fires:** Gate 3  

**Why adversarial:** A genuine target is locked and all pre-civilian gates pass — the system must abort solely on civilian confidence, not on any DMRL fault.  
**Why not covered before:** `inputs_from_dmrl()` has always been called with the default `civilian_confidence=0.0`; no integration test ever injected a non-zero value.

---

### ADV-02 — Civilian Below Threshold, Full Pipeline, Must Continue

**Gap it closes:** OI-26; establishes the negative control for ADV-01 through the integration path  
**SRS requirement:** TERM-01 (below-threshold must not abort)

**Stimulus:**

| Parameter | Value | Rationale |
|---|---|---|
| `generate_synthetic_scene()` | `n_targets=1, n_decoys=0, seed=42` | Same scene as ADV-01 for direct comparison |
| DMRL processing | `DMRLProcessor.process_target(target, max_frames=30)` | Full pipeline; lock acquired |
| `civilian_confidence` injected | `0.65` | Below 0.70 threshold; should not trigger abort |
| `corridor_violation` | `False` | No additional abort triggers |
| `pre_terminal_zpi_complete` | `True` | Gate 0 passes |

**Expected outcome:** CONTINUE  
**Expected abort reason:** `NONE`  
**Gate that fires:** All clear

**Why adversarial:** Civilian confidence is elevated (0.65) but sub-threshold — system must discriminate and proceed, not false-abort.  
**Why not covered before:** The integration path has only ever used `civilian_confidence=0.0`; no test has verified that a sub-threshold value correctly permits CONTINUE through the full pipeline.

---

### ADV-03 — Civilian Confidence at Exactly 0.70 Boundary, Full Pipeline

**Gap it closes:** OI-26; boundary condition for TERM-01 through the integration path  
**SRS requirement:** TERM-01 (exactly ≥ 0.70 is the abort condition; inclusive boundary)

**Stimulus:**

| Parameter | Value | Rationale |
|---|---|---|
| `generate_synthetic_scene()` | `n_targets=1, n_decoys=0, seed=42` | Single real target; no decoy |
| DMRL processing | `DMRLProcessor.process_target(target, max_frames=30)` | Full pipeline; lock acquired |
| `civilian_confidence` injected | `0.70` | Exactly at threshold — must abort per `≥` condition |
| `corridor_violation` | `False` | No additional abort triggers |
| `pre_terminal_zpi_complete` | `True` | Gate 0 passes |

**Expected outcome:** ABORT  
**Expected abort reason:** `CIVILIAN_DETECTED`  
**Gate that fires:** Gate 3

**Why adversarial:** The boundary value `0.70` must produce abort — a boundary error (treating `≥` as `>`) would produce CONTINUE and silently violate ROE.  
**Why not covered before:** The boundary tests in `test_s5_l10s_se.py` (`test_gate3_abort_at_exactly_0_700`) use hand-constructed inputs and never run through the DMRL pipeline.

---

### ADV-04 — Decoy Present AND Civilian Present, Gate Priority Verification

**Gap it closes:** OI-26; gate ordering under concurrent adversarial conditions through the full pipeline  
**SRS requirement:** TERM-01 + TERM-03 (decoy abort at Gate 2 must precede civilian check at Gate 3)

**Stimulus:**

| Parameter | Value | Rationale |
|---|---|---|
| `generate_synthetic_scene()` | `n_targets=1, n_decoys=1, seed=7` | Real target plus thermal decoy; adversarial scene |
| DMRL processing | `DMRLProcessor.process_target(decoy_target, max_frames=30)` | Process the **decoy** target — expect `is_decoy=True` result |
| `civilian_confidence` injected | `0.82` | Above threshold; would trigger Gate 3 if reached |
| `corridor_violation` | `False` | Isolates Gate 2 vs Gate 3 ordering |
| `pre_terminal_zpi_complete` | `True` | Gate 0 passes |

**Expected outcome:** ABORT  
**Expected abort reason:** `DECOY_DETECTED`  
**Gate that fires:** Gate 2 (Gate 3 is never evaluated)

**Why adversarial:** Both decoy and civilian conditions are simultaneously true — the system must fire at Gate 2 and must not reach Gate 3, proving the priority order holds under dual-fault conditions.  
**Why not covered before:** `test_decoy_abort_beats_civilian_detection` in `test_s5_l10s_se.py` tests this priority but with hand-constructed inputs; no test runs actual DMRL decoy detection and then checks which gate fires.

---

### ADV-05 — Civilian Present, Lock Degraded to 0.83 (Simulated Haze), Gate Priority Verification

**Gap it closes:** OI-26; confirms Gate 1c aborts before Gate 3 when lock is degraded and civilian is present  
**SRS requirement:** TERM-01 + TERM-02 (lock threshold at Gate 1c takes priority over civilian abort at Gate 3)

**Stimulus:**

| Parameter | Value | Rationale |
|---|---|---|
| `generate_synthetic_scene()` | Not used for the target itself | Target is manually constructed with degraded signature to produce lock_confidence ≈ 0.83 |
| Manual `ThermalTarget` | `is_decoy=False, thermal_signature=0.62, thermal_decay_rate=0.001, initial_roi_px=24, bearing_deg=1.0, range_m=1200.0` | Degraded signature produces lock_confidence < 0.85 after full dwell |
| DMRL processing | `DMRLProcessor.process_target(target, max_frames=30)` | Full pipeline; expect `lock_acquired=False`, `lock_confidence ≈ 0.83` |
| `civilian_confidence` injected | `0.82` | Above threshold; would trigger Gate 3 if reached |
| `corridor_violation` | `False` | No additional abort triggers |
| `pre_terminal_zpi_complete` | `True` | Gate 0 passes |

**Expected outcome:** ABORT  
**Expected abort reason:** `NO_LOCK`  
**Gate that fires:** Gate 1c (Gate 3 is never evaluated)

**Why adversarial:** Two simultaneous abort conditions (low lock + civilian) — system must abort at the earlier gate (1c) and must not evaluate the later gate (3), proving no gate is skipped or reordered under stress.  
**Why not covered before:** This concurrent condition (low lock + elevated civilian) has never been assembled; existing tests exercise each abort reason in isolation.

---

### ADV-06 — Civilian Present, Lock Exactly at Boundary (0.85), Gate 3 Must Fire

**Gap it closes:** OI-26; the critical integration path where Gate 1c narrowly passes and Gate 3 must catch the abort  
**SRS requirement:** TERM-01 + TERM-02 (boundary lock passes Gate 1c, then civilian gate must evaluate and abort)

**Stimulus:**

| Parameter | Value | Rationale |
|---|---|---|
| Manual `ThermalTarget` | `is_decoy=False, thermal_signature=0.75, thermal_decay_rate=0.001, initial_roi_px=22, bearing_deg=1.0, range_m=1200.0` | Produces lock_confidence close to 0.85 boundary after min dwell |
| DMRL processing | `DMRLProcessor.process_target(target, max_frames=30)` | Full pipeline |
| **Assertion prerequisite** | `result.lock_confidence >= 0.85` and `result.lock_acquired == True` | Must be verified before L10s-SE is called; if lock not acquired, test is invalid |
| `civilian_confidence` injected | `0.75` | Above 0.70 threshold; Gate 3 must fire |
| `is_decoy` from DMRL | `False` | Gate 2 passes — no decoy |
| `corridor_violation` | `False` | Gate 4 does not fire |
| `pre_terminal_zpi_complete` | `True` | Gate 0 passes |

**Expected outcome:** ABORT  
**Expected abort reason:** `CIVILIAN_DETECTED`  
**Gate that fires:** Gate 3 (Gates 0–2 all clear)

**Why adversarial:** This is the only integration path where all pre-civilian gates pass and Gate 3 is the sole abort trigger — it is the direct test of the civilian ROE gate operating as intended under realistic DMRL conditions.  
**Why not covered before:** Reaching Gate 3 via the full integration path requires lock_acquired=True through DMRL, which was never combined with a non-zero civilian_confidence injection; `inputs_from_dmrl()` has always defaulted to 0.0.

---

## Implementation Notes for Step 3

### Pipeline structure for each scenario

```python
# Canonical integration pattern (not test code — specification only)
scene = generate_synthetic_scene(n_targets=N, n_decoys=M, seed=S)
processor = DMRLProcessor(verbose=False)
dmrl_result = processor.process_target(target, max_frames=30)

inp = inputs_from_dmrl(
    dmrl_result,
    civilian_confidence=<injected value>,   # THE KEY OVERRIDE
    corridor_violation=False,
    pre_terminal_zpi_complete=True,
)
engine = L10sSafetyEnvelope(verbose=False)
result = engine.evaluate(inp)
```

### ADV-06 prerequisite guard

ADV-06 requires a Monte Carlo outer loop or seed selection to guarantee `lock_acquired=True` before the L10s-SE call. If DMRL does not acquire lock (stochastic), the test must be skipped or retried with a different seed — it must not assert Gate 3 firing when the actual abort reason is Gate 1c.

Recommended implementation: run 10 seeds, assert Gate 3 fires in at least one run where `dmrl_result.lock_acquired == True`.

### Seed discipline

All scenarios using `generate_synthetic_scene()` with a fixed seed are deterministic. ADV-01 through ADV-04 use `seed=42` or `seed=7` as specified. ADV-05 and ADV-06 use manually constructed `ThermalTarget` objects and are not seed-dependent at the scene level (stochasticity enters via `_compute_lock_confidence()` and `_compute_decoy_probability()`).

---

## Gate Coverage Summary

| Scenario | Gate 0 | Gate 1a | Gate 1b | Gate 1c | Gate 2 | Gate 3 | Gate 4 | Outcome |
|---|---|---|---|---|---|---|---|---|
| ADV-01 | PASS | PASS | PASS | PASS | PASS | **FIRE** | — | ABORT / CIVILIAN_DETECTED |
| ADV-02 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | CONTINUE |
| ADV-03 | PASS | PASS | PASS | PASS | PASS | **FIRE** | — | ABORT / CIVILIAN_DETECTED |
| ADV-04 | PASS | PASS | PASS | PASS | **FIRE** | — | — | ABORT / DECOY_DETECTED |
| ADV-05 | PASS | PASS | PASS | **FIRE** | — | — | — | ABORT / NO_LOCK |
| ADV-06 | PASS | PASS | PASS | PASS | PASS | **FIRE** | — | ABORT / CIVILIAN_DETECTED |

ADV-01 and ADV-06 together are the first tests that reach Gate 3 through the full DMRL integration path.

---

*Specification only. No test code written. Step 3 will implement these as pytest cases in `tests/test_s5_l10s_se_adversarial.py`.*
