# HANDOFF — S-NEP-08 Closure
**Date:** 25 March 2026
**Status:** S-NEP-08 COMPLETE
**From:** S-NEP-08 implementation session (25 March 2026)
**To:** Next programme phase session

---

## Repo State at Closure

| Repo | Commit | Branch | Test suite |
|---|---|---|---|
| `micromind-autonomy` | `542867e` | main (clean) | **311/311 passed** |
| `nep-vio-sandbox` | `30c2d56` | main (clean) | 443/443 |

---

## Programme Invariants (Do Not Violate)

1. **Frozen estimator constants — immutable**
   - `_ACC_BIAS_RW`, `_GYRO_BIAS_RW`, `_POS_DRIFT_PSD`, `_GNSS_R_NOMINAL`
   - TD approval required for any modification

2. **EKF interface stability**
   - `update_vio()` now returns `(nis, rejected, innov_mag)` — three values
   - `propagate()`, `inject()`, `update_gnss()` — unchanged
   - No changes to correction logic or filter state

3. **Fusion/mission separation — non-negotiable**
   - `VIONavigationMode` lives in `core/fusion/vio_mode.py` — fusion layer only
   - `core/state_machine/state_machine.py` — zero modifications permitted
   - Mission layer reads `vio_nav.current_mode.name` as a string signal only
   - No import of ESKF internals from fusion modules

4. **NIS interpretation constraint (PF-03)**
   - NIS must NOT be tuned, reinterpreted, or used as a health gate
   - Diagnostic signal only under current operating regime

5. **Drift envelope is a confidence degradation signal — NOT a hard bound**
   - `drift_envelope_m = VIO_DRIFT_RATE_CONSERVATIVE_M_S × dt_since_vio`
   - Conservative over-estimate per S-NEP-07 L-05
   - Downstream systems must not treat it as a guaranteed maximum

6. **vel_err_m_s is removed from operational logs (D-05)**
   - `vel_err_diagnostic` available only when `emit_vel_diagnostic=True`
   - Default: off. Do not restore as operational signal without velocity convergence characterisation

7. **Baseline anchoring**
   - All future evaluations reference S-NEP-04 through S-NEP-08 logs
   - Schema 08.1 for new runs; old logs (06e.3) remain valid and unaffected

8. **Mode Integrity Invariant — structural constraint**
   - Navigation mode transitions are determined solely by `VIONavigationMode` inputs: `dt_since_vio` and VIO update events
   - Transitions must be deterministic given identical input streams — same input sequence always produces the same mode sequence
   - No external module (including the mission FSM) may override or force mode transitions
   - Mission layer consumes `current_mode` as a read-only signal only
   - Any code that writes to or overrides `VIONavigationMode` internal state outside of `tick()` and `on_vio_update()` is a violation of this invariant

---

## What S-NEP-08 Delivered

### New files — micromind-autonomy

**`core/fusion/vio_mode.py`**
Three-state VIO navigation mode tracker. Module-level constants configurable at construction.
```python
from core.fusion.vio_mode import VIONavigationMode, VIOMode

vio_nav = VIONavigationMode()         # uses module defaults
vio_nav.tick(dt)                       # call every IMU step
mode_changed, spike = vio_nav.on_vio_update(accepted=True, innov_mag=innov_mag)

vio_nav.current_mode        # VIOMode.NOMINAL / OUTAGE / RESUMPTION
vio_nav.current_mode.name   # "NOMINAL" / "OUTAGE" / "RESUMPTION" (for logging/mission)
vio_nav.dt_since_vio        # float seconds
vio_nav.drift_envelope_m    # float | None (None outside OUTAGE)
vio_nav.in_outage           # bool — True during OUTAGE or RESUMPTION
```

**`core/fusion/fusion_logger.py`** — schema 08.1
```python
logger = FusionLogger(log_path="run.json", label="my_run")
logger.log_vio_update(t=, nis=, innov_mag=, trace_P=, vio_mode=,
                       dt_since_vio=, drift_envelope_m=, innovation_spike_alert=,
                       error_m=, ba_est=)
logger.log_propagate(t=, trace_P=, vio_mode=, dt_since_vio=, error_m=)
logger.log_rejection(t=, nis=, innov_mag=)
logger.close(vio_nav_mode=vio_nav)    # writes JSON; incorporates mode summary stats
```

**`core/ekf/error_state_ekf.py`** — additive change only
```python
nis, rejected, innov_mag = eskf.update_vio(state, pos_ned, cov_pos_ned)
# innov_mag = ‖pos_ned - state.p‖ (metres, computed pre-gate, always returned)
# Early-rejection paths return innov_mag=0.0
```

**Logging context extension (observability only — no logic depends on these):**
`innovation_spike_alert` log entries may optionally carry:
- `outage_duration_s` — `dt_since_vio` at the moment the spike was detected
- `mode_at_trigger` — mode string at trigger (always "OUTAGE" by definition, but explicit for analysis)

These fields are for post-hoc analysis only. No system logic may use them as inputs.

**`tests/test_s_nep_08.py`** — 30 unit gates (G-01..G-08), no EuRoC data required

### New files — nep-vio-sandbox

**`fusion/run_08_mode_validation.py`** — full pipeline validation
```bash
cd ~/micromind/repos/micromind-autonomy
python3 ../nep-vio-sandbox/fusion/run_08_mode_validation.py
# Produces: fusion/logs/run_08_a_5s_08.json, run_08_b_20s_08.json
```

### Configurable constants (vio_mode.py)

| Constant | Default | Source | Change requires |
|---|---|---|---|
| `VIO_OUTAGE_THRESHOLD_S` | 2.0 s | S-NEP-07 L-07 | Programme decision |
| `VIO_INNOVATION_SPIKE_THRESHOLD_M` | 1.0 m | S-NEP-07 L-08 | Programme decision |
| `VIO_DRIFT_RATE_CONSERVATIVE_M_S` | 0.800 m/s | S-NEP-06 C-07 | TD approval (links to validated data) |
| `VIO_RESUMPTION_CYCLES` | 1 | S-NEP-07 4D doctrine | Programme decision |

---

## Pre-existing Bugs Fixed in This Sprint

These were failing before S-NEP-08 started. Fixed opportunistically:

| File | Fix |
|---|---|
| `tests/test_sprint_s3_acceptance.py` | TRNStub.update() old calling convention (`ins=`, `dt=` kwargs removed); correction applied manually via `corr.delta_north_m` |
| `tests/test_s9_nav01_pass.py` | `_get_Q()` now calls `propagate(dt=0.01)` before reading Q (Q is built dynamically) |
| `sim/nav_scenario.py` | Same TRNStub.update() calling convention fix |

---

## Operating Doctrine (S-NEP-07 Rev 3, Section 4D — implemented)

| Mode | Trigger | What is trusted | What is suppressed |
|---|---|---|---|
| NOMINAL | dt_since_vio = 0 | state.p (strongly observed) | Nothing |
| OUTAGE | dt_since_vio ≥ 2.0s | state.p conditionally (geometry unknown) | Position-dependent functions |
| RESUMPTION | First accepted VIO after outage | state.p (corrected) | Position-dependent functions until stabilisation complete |

**State correction at resumption is discontinuous** — `state.p` steps by accumulated drift in one `inject()` call. `state.v` and `state.ba` are NOT corrected. Mission layer must treat resumption as re-alignment, not smooth convergence.

**System Rule (S-NEP-07 Section 1.8):** No control, planning, or decision logic shall rely on `state.v` as a primary input under the current architecture.

---

## What Is NOT Done (deferred per S-NEP-07)

| Lever | Reason deferred |
|---|---|
| L-04 Innovation spreading | Requires TD approval — modifies inject() behaviour |
| L-03 Velocity covariance init | Requires TD approval — touches covariance initialisation |
| L-06 Adaptive process noise | Low priority — heuristic, symptom not root cause |
| L-02 Odometry/baro sensors | Hardware dependency — outside SIL scope |
| L-09 Trajectory-aware classification | Mission planning integration — outside current scope |

---

## Entry Checklist for Next Session

```bash
# micromind-autonomy
cd ~/micromind/repos/micromind-autonomy
git log --oneline main | head -3   # expect 542867e at top
python3 -m pytest tests/ -q 2>&1 | tail -3  # expect 311 passed

# Verify new module is importable
python3 -c "from core.fusion.vio_mode import VIONavigationMode; print('OK')"

# nep-vio-sandbox
cd ~/micromind/repos/nep-vio-sandbox
git log --oneline main | head -3   # expect 30c2d56 at top
python3 -m pytest tests/ -q 2>&1 | tail -3  # expect 443/443
```

---

## Forward Look — S-NEP-09

S-NEP-08 closes the VIO navigation mode implementation. The system now has:
- Structured outage detection and mode tracking
- Innovation spike alerting at resumption
- Conservative drift uncertainty quantification during outage
- Clean fusion/mission separation with Mode Integrity Invariant

**Next phase:** S-NEP-09 — Operational Behaviour Validation
Objective: validate system behaviour under realistic operating variability before any optimisation is considered. Scope defined in `HANDOFF_S-NEP-08_CLOSURE_v2.md` and `NEP_SPRINT_STATUS.md`.

The principle: **the system must be understood in operation before it is improved.**

**Status:**
```
S-NEP-08: CLOSED — VIO NAVIGATION MODE FRAMEWORK IMPLEMENTED
311/311 tests passing
Next phase: S-NEP-09 — Operational Behaviour Validation (scope defined)
