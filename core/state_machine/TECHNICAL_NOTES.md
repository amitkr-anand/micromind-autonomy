# TECHNICAL_NOTES — core/state_machine

**Module:** NanoCorteXFSM (`state_machine.py`) + MissionEventBus (`mission_manager/mission_manager.py`)
**Last Updated:** 11 April 2026 (SB-5 Phase B — MM-04, SB-06 PASS)

---

## OODA-Loop Rationale — MM-04 Queue Latency

### Why is 100 ms the critical event delivery threshold?

SRS §5.4 establishes 100 ms as the maximum permissible latency for critical
event delivery through the Mission Manager internal event bus. SRS §6.4 grounds
this figure in the OODA-loop budget for the terminal engagement phase:

- **Observe → Orient:** sensor fusion (BIM, EO/IR lock confidence) updates
  arrive at ~50 Hz (20 ms per cycle).
- **Orient → Decide:** NanoCorteXFSM evaluates guards and may issue a
  SHM/ABORT trigger within one tick (≤ 20 ms).
- **Decide → Act:** The event bus must deliver the trigger to the Mission
  Manager within 100 ms so that the downstream abort/SHM path executes
  before the vehicle crosses a no-go boundary or violates ROE constraints.

A 100 ms budget corresponds to **5 sensor ticks at 50 Hz** — sufficient to
absorb a single queue drain cycle under moderate CPU load (ST-CPU-01) while
still completing the full OODA loop within the 200 ms NFR-002 transition
budget allocated for safety-critical transitions.

### What happens to vehicle safety if a SHM trigger event is delayed beyond 100 ms?

A SHM_TRIGGER event is emitted when L10s-SE activates (`ST-05 SHM_ACTIVE`).
The L10s-SE is the terminal safety gate — the deterministic decision tree that
enforces ROE and civilian abort obligations. If delivery is delayed:

| Delay | Consequence |
|---|---|
| 100–200 ms | NFR-002 transition budget consumed; state machine may not reach SHM_ACTIVE before EO lock confidence falls below 0.3, causing EO_LOCK_LOSS abort — correct fallback, but bypasses intended SHM path |
| 200–500 ms | Vehicle continues terminal approach in SILENT_INGRESS without L10s-SE gate active; civilian abort conditions go unevaluated for ≥ 10 sensor ticks |
| > 500 ms | Terminal engagement executes without ROE check — unacceptable; mission must abort at system level |

The 100 ms threshold is therefore a **legal/ethical gate** (§2 in the L10s-SE
legal framework), not merely a performance target. Exceeding it during a live
engagement may constitute a failure to exercise adequate human-machine
authority over a lethal decision — a SRS hard requirement, not a KPI.

---

## Design Decision — Degraded Mode Priorities

### Why INFO events are dropped before critical events under queue pressure

Under queue utilisation > 80 % (`QUEUE_HIGH`), `MissionEventBus.enqueue()`
drops `EventPriority.INFO` events and passes `EventPriority.CRITICAL` events
through to the queue. This priority inversion is intentional (SRS §5.4):

**Safety argument:**

- `INFO` events carry diagnostic data (e.g., periodic health log lines,
  module-status ticks). They are produced at high rate by multiple modules
  and are the dominant source of queue pressure during high-CPU episodes.
  Dropping them degrades observability but does not impair vehicle safety.

- `CRITICAL` events carry mission-phase transitions (`SHM_TRIGGER`,
  `CORRIDOR_BREACH`, `ABORT`, `EO_LOCK_LOSS`). Every critical event
  represents a state change that the Mission Manager and NanoCorteXFSM must
  act on within the 100 ms OODA budget. Dropping any of these breaks the
  control loop and may allow a legally/operationally dangerous condition to
  go undetected.

### Operational consequence: CORRIDOR_BREACH vs. diagnostic log line

| Event | If dropped | Operational outcome |
|---|---|---|
| Diagnostic INFO (health tick) | Observability gap in post-mission log | **Acceptable.** Operator reviews partial log; no safety impact. |
| `CORRIDOR_BREACH` (CRITICAL) | NanoCorteXFSM never transitions to ABORT from NOMINAL/EW_AWARE/GNSS_DENIED/SILENT_INGRESS | **Unacceptable.** Vehicle continues through a predicted corridor violation. Depending on mission geometry this may mean: (a) flying into a geo-fenced exclusion zone; (b) approaching a non-combatant area without abort authority; (c) violating mission envelope hash — which the operator cryptographically signed. All three outcomes breach §16 of the SRS recovery ownership table and the L10s-SE ROE gate. |

The asymmetry is absolute: one missed diagnostic line vs. one unchecked
corridor breach. The priority ordering reflects this asymmetry directly.

### Why the queue overflow threshold is 80 % and not 100 %

A 20 % headroom band (80 %→100 %) acts as a temporal buffer. At 80 %
utilisation, `QUEUE_HIGH` fires and INFO events are dropped, reducing the
fill rate immediately. The remaining 20 % capacity (10 slots at `QUEUE_CAPACITY=50`)
absorbs the burst of critical events that may arrive during the window
between `QUEUE_HIGH` detection and the queue fully draining. This prevents
the scenario where the protective drop policy triggers too late and a
critical event still encounters a full queue.

---

*Document governed by Code Governance Manual v3.2 §9.1 (technical rationale
must be committed alongside implementation).*
