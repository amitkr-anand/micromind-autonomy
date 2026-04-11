# integration/TECHNICAL_NOTES.md
**Module:** integration/bridge (MAVLinkBridge, RebootDetector, BridgeLogger, TimeReference)  
**Last updated:** 10 April 2026  
**SRS ref:** IT-PX4-02, §10.15 PX4-04, §16 Recovery Ownership Matrix  
**Implementations:** PX4-04 reboot detection (SA-05), D8a gate wiring

---

## OODA-Loop Rationale — Reboot Detection

### Why sequence-number-reset rather than connection-loss detection?

**Connection-loss detection** (heartbeat timeout, link drop) fires when the
channel between MicroMind and PX4 is interrupted.  This covers RF jamming,
cable disconnection, and GCS link loss — but it does NOT distinguish between:

1. A **genuine PX4 reboot** (MCU power-cycle, watchdog reset, deliberate
   maintenance reboot), and
2. A **transient link interruption** where PX4 kept running and the
   mission state is intact.

If MicroMind responds to a transient link interruption with a full D8a gate
evaluation (operator clearance required, SHM entry), it will abort or block a
mission that never needed intervention.  False-positive rate on link-loss
detection is high in operationally realistic conditions: terrain masking,
antenna shadowing during manoeuvres, and EW-induced link degradation all
produce transient dropouts without any change to PX4 internal state.

**Sequence-number-reset detection** is definitive: PX4's outgoing MAVLink
sequence counter is a monotonic 0–255 rolling counter that **resets to 0 only
on process restart**.  A backward jump larger than the detection threshold
(5 positions, configurable) that is not attributable to the normal 255→0
rollover cannot occur without a genuine MCU or software restart.

The rollover guard uses modular arithmetic:
```
backward_dist = (last_seq – new_seq) % 256  >  threshold
forward_dist  = (new_seq – last_seq) % 256  >  threshold
```
Both must hold.  On a normal rollover (last≈255, new≈0): `forward_dist ≈ 1–4`
fails the second condition.  On a reboot (last arbitrary, new≈0): both distances
are large (≥ 200) and the event fires.

### The operational scenario requiring detection within 3 s

**Scenario:** Vehicle is in GNSS-denied flight (ST-03 / ST-04) at km 80 of a
150 km mission.  An EW event causes a momentary overvoltage spike that resets
the PX4 FCU.  PX4 restarts in default MANUAL/STABILISE mode with no knowledge
of the previous OFFBOARD setpoints.  The vehicle has no GNSS and is relying on
MicroMind's VIO + ESKF for state.

**Without 3 s detection:** The MAVLink setpoint stream is still live from T-SP.
PX4 (now in STABILISE) ignores the POSITION_TARGET setpoints.  The vehicle
diverges from its planned corridor while MicroMind believes it is tracking.
After the divergence grows beyond the corridor half-width, the mission reaches
a point where re-entry to OFFBOARD with the original trajectory is no longer
safe — and potentially outside the signed mission envelope.

**With ≤ 3 s detection:** Within one to three heartbeat periods (0.5 s each),
the sequence reset is observed.  `PX4_REBOOT_DETECTED` is logged.  The
integration layer notifies `MissionManager.on_reboot_detected()`, which
evaluates the D8a gate before any OFFBOARD re-engagement attempt.  The vehicle
is never re-engaged autonomously without the gate check.

The 3 s budget is derived from the 2 Hz HEARTBEAT rate (T-HB interval 0.5 s):
worst case is one full heartbeat period before the reset sequence number is
observed plus one period to propagate detection — giving 1.0 s detection latency.
The 3 s budget is ×3 the theoretical minimum to account for T-MON receive
scheduling jitter and queue drain latency under load.

---

## OODA-Loop Rationale — D8a Gate

### Why does autonomous resume require operator clearance after a PX4 reboot?

The D8a failure mode is: **D**ead-reckoning **r**esume with **s**tale target
**a**cquisition state on a stale ESKF position estimate.

**Failure mode anatomy:**

After a PX4 reboot, the ESKF position stored in the checkpoint is the last
known NED position before the crash.  The vehicle was in motion; its actual
position at the moment of reboot may differ from the checkpoint position by:

- Drift accumulated since the last checkpoint write (checkpoint interval may
  be several seconds during high-rate operations).
- Physical displacement during the uncontrolled interval after PX4 lost OFFBOARD
  (vehicle may have entered a mode with non-zero velocity, drifted with wind,
  or descended under failsafe).

If `pending_operator_clearance_required=True` was set at checkpoint time, it
means the vehicle was in a condition where **operator review was already required
before the reboot occurred** (typically: SHM entered, terminal zone crossed, or
an abort had been commanded but not yet confirmed).  In these conditions:

1. **ESKF state is stale:** the pre-reboot checkpoint position may not reflect
   where the vehicle actually is.  Blind re-engagement on a stale position puts
   the vehicle on a trajectory that was valid for a different position, potentially
   exiting the signed mission envelope.

2. **Target acquisition state is lost:** if `shm_active=True` was stored, the
   vehicle had previously confirmed an EO lock on a terminal target.  Post-reboot,
   that lock is gone.  Resuming L10s-SE based on stale confirmation constitutes
   an autonomous engagement decision without current sensor evidence — a clear
   violation of the ethical gate under SRS §5.2 (non-combatant protection).

3. **Corridor half-width may no longer be valid:** `route_corridor_half_width_m`
   stored in the checkpoint was computed from pre-reboot route geometry.  A
   reboot during a route replanning event (EW avoidance) may leave the stored
   value inconsistent with the current threat picture.

**What the gate enforces:**

`MissionManager.on_reboot_detected()` calls `resume()` which evaluates
`pending_operator_clearance_required` before any autonomous action.  When True:

- `AWAITING_OPERATOR_CLEARANCE` is logged immediately (auditable record).
- State is set to `MissionState.SHM` — no navigation outputs.
- `grant_clearance()` requires a deliberate, explicit operator command.

This cannot be satisfied by a sensor event, a timeout, or automatic inference.
It requires a human being to assess the situation (via available telemetry or
visual observation if the vehicle is still in RF range) and issue an explicit
command.

**§1.3 boundary enforcement:**

The D8a gate decision lives exclusively in `MissionManager`.  `MAVLinkBridge`
(via `RebootDetector`) detects and logs; it does not make mission decisions.
`CheckpointStore` serialises and restores; it does not evaluate clearance state.
This separation ensures the mission-critical gate cannot be bypassed by a
module that lacks authorisation to make engagement decisions.

---

## Module Boundary Summary

| Component | Responsibility | What it must NOT do |
|---|---|---|
| `RebootDetector` | Detect seq-number reset; log `PX4_REBOOT_DETECTED` | Issue mission commands; evaluate gate conditions |
| `MAVLinkBridge` | Wire `RebootDetector` into T-MON; notify integration layer | Make mission resume decisions; load checkpoints |
| `MissionManager` | Restore checkpoint; evaluate D8a / P-02 gate; log outcomes | Send MAVLink commands; touch PX4 directly |
| `CheckpointStore` | Persist and restore checkpoint state | Evaluate clearance conditions; make resume decisions |

**§1.3 rule:** PX4 Bridge must contain detection and logging only.  Mission
logic (resume/block/abort decisions) lives in `core/`, not `integration/bridge/`.

---

## OODA-Loop Rationale — PX4-01 OFFBOARD Continuity

### Why is 99.5 % the continuity threshold and not 100 %?

SRS §6.1 sets the OFFBOARD continuity threshold at 99.5 % (≤ 0.5 % loss) over
a 30-minute OFFBOARD engagement window.  This translates to a maximum cumulative
loss of **9 seconds** per mission.  The threshold is not 100 % because:

1. **Link-layer transients are operationally inevitable.**  In a MAVLink
   UDP-over-serial architecture (TELEM port → ground), brief link
   interruptions of 1–3 s occur from terrain masking, antenna shadowing
   during banking manoeuvres, and EW-induced link degradation.  A 100 %
   continuity requirement would make the system fail-safe on any link hiccup
   during a live GNSS-denied approach — exactly when the operator cannot
   intervene.

2. **PX4's internal safety governor.**  If OFFBOARD setpoints are absent for
   > 500 ms, PX4 transitions to its own failsafe (loiter/land depending on
   vehicle config).  The 9 s budget is ×18 this internal governor threshold.
   A single transient of ≤ 9 s allows the link to recover and OFFBOARD to
   be re-engaged before the PX4 failsafe causes an unrecoverable divergence.

3. **Empirical evidence from live SITL (97b2f5a).**  The 10 April 2026 live
   SITL run recorded one link transient of ≈ 3 s during the lap-1 turn.
   OFFBOARD continuity was ≈ 99.97 % — well within the 99.5 % gate.  The 0.5 %
   budget provides 3× headroom over the observed worst-case transient.

4. **Operational scenario: GNSS-denied ingress at km 80.**  At this point the
   vehicle is autonomous and no operator intervention is possible.  The 9 s
   budget allows for one full PX4 heartbeat timeout + recovery handshake
   without triggering the D8a gate (reboot detection), which would otherwise
   block mission resume and require operator clearance.

### What is the operational scenario that allows for up to 0.5 % loss?

The scenario is: **antenna body-masking during a 30° coordinated turn** in
SILENT_INGRESS (ST-04).  The LOS to the relay node is broken for 2–4 s as the
fuselage blocks the antenna.  During this window, T-HB continues sending
heartbeats (FM-1 independent daemon) and T-SP continues dispatching setpoints
(FM-4 time-driven).  PX4 holds the last OFFBOARD setpoint; the vehicle
maintains its heading.  When LOS is restored, T-MON detects HEARTBEAT resumption,
confirms OFFBOARD mode (custom_mode == 393216), and OFFBOARD_RESTORED is logged
with gap_duration_ms and stale_setpoints_discarded=True.  The 0.5 % budget
absorbs two such turns in a 30-minute mission without gate failure.

---

## Design Decision — Stale Setpoint Discard on Link Recovery

### Why stale setpoints must be discarded on link recovery

When the MAVLink link drops, T-SP continues dispatching setpoints from the last
`update_setpoint()` call — the frozen NED position at the moment of link loss.
During the gap, the vehicle may have:

- **Drifted with wind:** Under nominal OFFBOARD, PX4's position controller
  compensates.  During OFFBOARD loss, PX4 may engage its own failsafe attitude
  hold, allowing wind drift of 1–5 m/s depending on conditions.
- **Changed heading:** Under SILENT_INGRESS, the route planner may have issued
  a new waypoint during the link gap.  The frozen setpoint references the
  pre-gap position, not the updated trajectory.
- **Experienced VIO drift:** The ESKF runs continuously.  After a 5 s link gap,
  accumulated VIO drift (≈1 m/km) may move the vehicle's estimated position
  away from the frozen setpoint origin.

### What is the navigation hazard if a stale setpoint is replayed after a 5 s gap?

If a 200 ms stale setpoint is replayed after a 5 s gap, the hazard depends on
the gap context:

| Gap duration | Vehicle state during gap | Hazard on stale replay |
|---|---|---|
| 200 ms | Nominal cruising, no replanning | Negligible — position drift < 0.1 m |
| 1 s | Banking turn, changing heading | Setpoint is for old heading; PX4 commands a transient toward stale target before next fresh setpoint arrives |
| 5 s | SILENT_INGRESS, route replanning in progress | Frozen setpoint is 25–50 m behind current ESKF position; PX4 commands vehicle back along its own track — opposing the live trajectory |
| 9 s (budget limit) | EW-active, BIM Amber, heading change | Stale setpoint may point toward a previously-computed waypoint that is now inside an EW threat zone updated during the gap |

The 5 s case is the design-driving scenario.  A 200 ms T-SP interval means that
after a 5 s gap, **25 stale setpoints accumulate in the pipeline**.  Replaying
them would command a 1–3 s backward correction that opposing the live route,
visible as a position spike in EKF2 and a temporary corridor exceedance.  In
terminal phase (ST-05 SHM_ACTIVE) with a 100 m corridor half-width, this spike
could trigger `CORRIDOR_VIOLATION → ABORT` — aborting a correctly-executing
mission.

**The discard action:** On link recovery (`record_offboard_restored()`),
`PX4ContinuityMonitor` calls `self._sp_timestamps.clear()`.  In the wired
`MAVLinkBridge`, the setpoint buffer (`_setpoint_x_m`, `_setpoint_y_m`,
`_setpoint_z_m`) must be refreshed from the current nav state before the next
T-SP tick.  This guarantees the first post-recovery setpoint references the live
ESKF position rather than the pre-gap frozen coordinate.

---

## Known limitations (as of 10 April 2026)

| Item | Description | OI |
|---|---|---|
| Single detection threshold | `seq_threshold=5` is hardcoded default; not in any config file | Pending config governance |
| No deliberate-reboot exemption | `RebootDetector.reset()` exists but is not wired to any planned-reboot command path | Phase B |
| No cross-vehicle reboot correlation | Two-vehicle scenario (OI-20) has independent `RebootDetector` instances; simultaneous reboots are treated as independent events | Phase C |
