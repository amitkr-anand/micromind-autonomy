# TECHNICAL_NOTES — core/mission_manager
**Module:** MissionManager + MissionEventBus
**Last updated:** 11 April 2026
**SRS ref:** §5.4 MM-04, §10.6 PX4-04, §16
**Implementations:** P-02 operator clearance gate,
D8a reboot recovery, MissionEventBus queue
latency instrumentation

---

## OODA-Loop Rationale — P-02 Operator Clearance

When pending_operator_clearance_required=True is
restored from a checkpoint, autonomous resume is
blocked. This is the D8a safety gate.

**Why operator clearance cannot be automatic:**
After a PX4 reboot, the restored ESKF position
is the last checkpoint value — which may be
seconds to minutes stale. The vehicle's actual
position at reboot may differ from the checkpoint
by accumulated VIO drift plus uncontrolled
displacement during the reboot interval.

If MissionManager resumed autonomously:
- The trajectory is computed from a stale
  position estimate
- Any target acquisition state (shm_active=True)
  stored in the checkpoint is no longer current
  — the vehicle's sensor picture has changed
- Route corridor half-width may be inconsistent
  with current threat geometry

The operator clearance requirement forces a human
to assess the situation before autonomous
re-engagement. This cannot be satisfied by a
timeout, a sensor event, or automatic inference.
grant_clearance() must be a deliberate human
action — SRS §10.6 post-reboot state requirement.

## OODA-Loop Rationale — MissionEventBus Design

The MissionEventBus introduces a producer/consumer
queue between event sources (BIM, Navigation
Manager, L10s-SE, PX4 Bridge) and MissionManager.

**Why a queue rather than direct method calls:**
Direct synchronous calls from event sources to
MissionManager create coupling across the Logic
Box boundary (§2.3). An event source in the
Navigation Core Box would need a direct reference
to MissionManager in core/mission_manager/ — a
dependency that the Logic Box framework prohibits.

The queue decouples production from consumption:
event sources enqueue typed events; MissionManager
drains the queue on its own scheduler tick.
This preserves module sovereignty and makes the
inter-module communication path auditable.

**Why 100 ms is the delivery threshold:**
See core/state_machine/TECHNICAL_NOTES.md §MM-04.
The 100 ms budget is a legal/ethical gate, not a
performance target. A SHM_TRIGGER delayed beyond
100 ms means the L10s-SE ROE gate may not execute
before the vehicle crosses a terminal boundary.

## Design Decision — Queue Priority and Overflow

CRITICAL events are never dropped under queue
pressure. INFO events are dropped when utilisation
exceeds 80%. See core/state_machine/
TECHNICAL_NOTES.md for the full operational
consequence table.

QUEUE_CAPACITY = 50 was chosen to hold 5 seconds
of critical events at 10 Hz arrival rate — the
maximum sustained critical event rate under
compound fault injection (FI-01 through FI-07
simultaneously). This provides headroom while
keeping the queue footprint small enough for
embedded deployment.

---

*Document governed by Code Governance Manual
v3.2 §2.5 and §9.1.*
