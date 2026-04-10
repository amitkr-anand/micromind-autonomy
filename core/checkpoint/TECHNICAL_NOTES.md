# core/checkpoint/TECHNICAL_NOTES.md
**Module:** core/checkpoint (Checkpoint v1.2, CheckpointStore)  
**Last updated:** 10 April 2026  
**SRS ref:** §10.15, PX4-05, EC-02  
**Corrections implemented:** P-01 (SHM persistence), P-02 (operator clearance gate)

---

## OODA-Loop Rationale — P-01 (SHM Persistence)

**Why must a vehicle that was in SHM at checkpoint time re-enter SHM on reboot
rather than resume autonomous flight?**

Silent Harvest Mode (SHM, ST-05) is entered when the vehicle is inside the
terminal engagement zone with zero RF and full L10s-SE enforcement.  At that
point the vehicle has already crossed into an area where the operator cannot
observe it in real time.

**Threat model:**

A SIGKILL or power-cycle event inside the terminal zone is not a random fault —
it is a credible adversary action.  An adversary who can cut power to an airborne
platform (via directed EMP, proximity kill, or controlled shutdown of a captured
ground station) may do so precisely *because* the vehicle is in SHM and the
adversary wants to force a reboot into a more vulnerable, autonomous-flight state
where the vehicle will re-establish RF contact and become trackable.

If the vehicle were to resume autonomous flight (ST-01 NOMINAL) on reboot:

1. **RF exposure window:** The vehicle would attempt to re-acquire GNSS and RF
   contact, broadcasting its position to any listener.
2. **Corridor escape risk:** The mission envelope was computed before the terminal
   zone was entered.  After an uncontrolled reboot, the vehicle has no way to
   verify that the re-entered flight path is still within the original signed
   envelope.  Corridor violations during autonomous recovery could carry civilian
   exposure risk.
3. **State integrity loss:** The pre-reboot SHM entry may have been triggered by
   conditions (EO lock achieved, target confirmed) that no longer hold after the
   reboot.  Resuming as NOMINAL would bypass the L10s-SE gate that originally
   justified the engagement.

**Design consequence:**  
When `shm_active=True` is written to a checkpoint, the CheckpointStore preserves
it across SIGKILL.  On restore, `MissionManager.resume()` (via
`pending_operator_clearance_required`, which is set together with `shm_active`
when the vehicle enters the terminal zone) blocks autonomous resumption until an
operator explicitly clears the vehicle.  This forces a human-in-the-loop review
before the vehicle can act on stale terminal-zone state.

No new log event is required: existing `CHECKPOINT_WRITTEN` and
`CHECKPOINT_RESTORED` events carry `shm_active` as a structured field that can
be audited post-incident.

---

## OODA-Loop Rationale — P-02 (Operator Clearance Gate)

**Why does autonomous mission resume after a reboot require explicit operator
clearance in certain conditions?  What failure mode does the D8a gate prevent?**

The D8a gate is named for the failure mode it prevents: **D**ead **r**eckoning
**r**esume with **s**tale **t**arget **a**cquisition state ("D8a" from the SRS
failure mode taxonomy).

**The failure mode:**

Without the P-02 gate, a vehicle that reboots after achieving terminal-zone EO
lock on a target would resume autonomous flight using the last-written
`fsm_state="SHM_ACTIVE"` checkpoint.  The ESKF position at the time of the
checkpoint may be hours-stale if the reboot was delayed (e.g., a hibernation
event or a delayed power recovery).  The target may have moved.  The corridor
may now contain civilians.  The original operator intent may no longer apply.

**What the gate enforces:**

When `pending_operator_clearance_required=True` is restored from a checkpoint,
`MissionManager.resume()` must:

1. **Log `AWAITING_OPERATOR_CLEARANCE`** with `req_id='PX4-05'`,
   `severity='WARNING'`, `module_name='MissionManager'`, and `timestamp_ms`
   from the mission clock.  This creates an auditable record that the vehicle
   is in a blocked state visible to any connected monitoring infrastructure.

2. **Enter `MissionState.SHM`** — the vehicle is logically in Silent Harvest
   Mode; no autonomous navigation commands are issued.

3. **Refuse to resume** until `grant_clearance()` is called.  This requires
   a deliberate operator action via the ground-station command channel —
   it cannot happen automatically, by timeout, or by sensor event.

**Why this is not an availability trade-off failure:**

The P-02 gate is expected to fire only when:
  - The vehicle has already entered the terminal zone (`shm_active=True`), AND
  - A fault (SIGKILL, power-cycle) occurred at that point.

This represents a scenario where the vehicle's continuation is already
non-nominal.  Requiring operator clearance is the legally and ethically correct
response: a human must confirm that the engagement conditions which led to SHM
entry still hold before autonomous engagement can continue.

---

## Design Decision — Six New Fields (v1.2)

The six fields were added in a single schema bump from v1.1 to v1.2 to avoid
incremental partial deployments.  All six are captured by `dataclasses.asdict()`
without special handling; no serialisation path modification was required.

| Field | Type | Default | What goes wrong if lost across SIGKILL |
|---|---|---|---|
| `shm_active` | `bool` | `False` | Vehicle reboots from SHM into ST-01 NOMINAL; RF exposure window opens; adversary can track reboot event; L10s-SE context lost. **(P-01)** |
| `pending_operator_clearance_required` | `bool` | `False` | Vehicle resumes autonomous flight from terminal-zone checkpoint without operator review; D8a failure mode (stale target acquisition on stale position); potential civilian exposure. **(P-02)** |
| `mission_abort_flag` | `bool` | `False` | An abort that was commanded before the SIGKILL would be silently cancelled; vehicle resumes a mission that the operator intended to terminate. |
| `eta_to_destination_ms` | `int` | `0` | Post-reboot route planning cannot account for deadline pressure; the vehicle may re-plan a longer route that violates a time-critical corridor window. |
| `terrain_corridor_phase` | `str` | `""` | Vehicle loses track of which route segment it was in; route planner may assign a terrain-texture cost penalty for a phase already completed, causing unnecessary detour. |
| `route_corridor_half_width_m` | `float` | `0.0` | Vehicle uses the default corridor width (which may be wider or narrower than the signed mission envelope); corridor violation checks after reboot may be mis-calibrated. |

---

## Serialisation Guarantee

`Checkpoint.to_dict()` calls `dataclasses.asdict()`, which recursively converts
all fields.  `Checkpoint.from_dict()` filters to `dataclasses.fields(Checkpoint)`
and constructs the object directly.  This guarantees:

- No field can be silently dropped by a `to_dict()` → JSON → `from_dict()` cycle.
- Adding new fields to `Checkpoint` is automatically covered without any change
  to serialisation code (P-01 guarantee: new fields added = new fields persisted).
- Legacy checkpoint files (schema < v1.2) can be loaded; missing fields take
  their defined defaults.

Atomic write pattern in `CheckpointStore.write()`:
```
tmp = dest.with_suffix(".tmp")
tmp.write_text(...)
tmp.rename(dest)          ← atomic on POSIX; SIGKILL before this leaves only .tmp
```
A SIGKILL between `write_text` and `rename` leaves `<name>.tmp` on disk but never
produces a corrupt `<name>.json`.  The last successful rename is always a valid
checkpoint file.

---

## Rolling Purge

`CheckpointStore` retains at most `max_retained=5` checkpoint files (per PX4-05
§10.15).  After each `write()`, files are sorted lexicographically (which equals
chronological sort due to the `{timestamp_ms:020d}` prefix) and all but the five
most recent are deleted.  Each deleted file generates a `CHECKPOINT_PURGED` event
in the shared `event_log`.

The rolling purge prevents unbounded disk consumption during long endurance
missions (AT-6 endurance baseline: 1483 missions, 0 crashes, slope 1.135 MB/hr).
