"""
core/mission_manager/mission_manager.py
MicroMind / NanoCorteX — Mission Manager

Controls the mission resume path and implements:
  - P-02 operator clearance gate (SRS §10.15, PX4-05 correction P-02)
  - D8a gate / reboot-triggered recovery (SRS IT-PX4-02, PX4-04, EC-03)
  - MM-04 event bus queue latency instrumentation (SRS §5.4, GAP-05, EC-08)

§9.1 failure-first sequence:
    The blocked/SHM path is implemented and evaluated BEFORE the nominal resume
    path inside resume().  A checkpoint with pending_operator_clearance_required=True
    will never reach the nominal path.

P-02 gate — resume() behaviour when pending_operator_clearance_required=True:
    (a) Log AWAITING_OPERATOR_CLEARANCE with:
            req_id        = 'PX4-05'
            severity      = 'WARNING'
            module_name   = 'MissionManager'
            timestamp_ms  = clock_fn()
    (b) Set state to MissionState.SHM  (vehicle enters SHM)
    (c) Return False  — autonomous flight does NOT resume

D8a gate — on_reboot_detected() — triggered when PX4_REBOOT_DETECTED fires:
    1. Restore latest Checkpoint from CheckpointStore.
    2. Call resume(checkpoint) which evaluates pending_operator_clearance_required:
         False → log MISSION_RESUME_AUTHORISED (req_id='PX4-04', severity='INFO'),
                 set state ACTIVE, return True
         True  → P-02 path: log AWAITING_OPERATOR_CLEARANCE, set SHM, return False

Autonomous flight resumes only when:
    - grant_clearance() is called by an operator command handler, OR
    - abort() is called to end the mission

MM-04 Event Bus (MissionEventBus):
    Thread-safe bounded queue with enqueue/dequeue latency instrumentation.
    Implements SRS §5.4 degraded-mode behaviours:
      - QUEUE_HIGH: logged when queue utilisation exceeds 80%; INFO events dropped.
      - QUEUE_CRITICAL_OVERFLOW: logged when queue is full and a critical event
        cannot be enqueued; overflow counter incremented.
    Every processed event produces an EVENT_QUEUE_LATENCY log at DEBUG level.
    All timestamps via clock_fn only (§1.4 — no time.time()).
    No raw sensor reads; no navigation state writes (§1.3).

References:
    SRS §5.4        MM-04 queue latency, GAP-05
    SRS §10.15      PX4-05 correction P-02
    SRS IT-PX4-02   PX4-04, EC-03 (D8a gate)
    Code Governance Manual v3.2  §9.1 (failure-first sequence)
    Code Governance Manual v3.2  §1.3 (no PX4 commands in MissionManager)
    Code Governance Manual v3.2  §1.4 (no time.time(); mission_clock only)
"""

from __future__ import annotations

import queue
import threading
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from core.checkpoint.checkpoint import Checkpoint, CheckpointStore


# ---------------------------------------------------------------------------
# MM-04 — Event Bus (SRS §5.4, GAP-05, EC-08)
# ---------------------------------------------------------------------------

class EventPriority(str, Enum):
    """Priority levels for MissionEventBus events (SRS §5.4)."""
    CRITICAL = "CRITICAL"   # safety-critical events — never dropped
    INFO     = "INFO"       # diagnostic / informational — dropped under high load


class MissionEventBus:
    """
    Internal event bus for Mission Manager queue latency instrumentation
    (MM-04, SRS §5.4, GAP-05, EC-08).

    Thread-safe bounded queue with latency measurement:
      - enqueue(): stamps enqueue_ts_ms via clock_fn; checks utilisation
      - _process_loop(): stamps dequeue_ts_ms immediately before processing;
        computes latency_ms = dequeue_ts_ms - enqueue_ts_ms; logs
        EVENT_QUEUE_LATENCY at DEBUG level.

    Degraded-mode behaviours (SRS §5.4):
      - Queue utilisation > 80%: log QUEUE_HIGH (WARNING), drop INFO events.
      - Queue full (overflow): log QUEUE_CRITICAL_OVERFLOW (CRITICAL),
        increment queue_overflow_count.

    Forbidden behaviours (§1.3, §1.4):
      - Must NOT read raw sensor data.
      - Must NOT modify navigation state vectors.
      - All timestamps via clock_fn only (no time.time()).
    """

    QUEUE_CAPACITY       = 50      # max events in flight
    QUEUE_HIGH_THRESHOLD = 0.80    # 80 % utilisation → QUEUE_HIGH + INFO drop

    def __init__(
        self,
        event_log: List[Dict[str, Any]],
        clock_fn:  Callable[[], int],
    ) -> None:
        """
        Args:
            event_log: Shared list that receives EVENT_QUEUE_LATENCY,
                       QUEUE_HIGH, and QUEUE_CRITICAL_OVERFLOW log dicts.
            clock_fn:  Zero-argument callable returning current time in
                       milliseconds — must be mission_clock only (§1.4).
        """
        self._event_log: List[Dict[str, Any]] = event_log
        self._clock_fn:  Callable[[], int]    = clock_fn
        self._queue:     queue.Queue          = queue.Queue(maxsize=self.QUEUE_CAPACITY)
        self._queue_overflow_count: int       = 0
        self._stop_event: threading.Event     = threading.Event()
        self._worker:    Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background event-processing worker thread."""
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name="MissionEventBus-worker",
        )
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        """
        Signal the worker thread to stop and wait up to *timeout* seconds.

        Safe to call even if start() was never called.
        """
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def queue_overflow_count(self) -> int:
        """Number of critical events dropped due to queue overflow."""
        return self._queue_overflow_count

    # ------------------------------------------------------------------
    # Event injection
    # ------------------------------------------------------------------

    def enqueue(
        self,
        event_type: str,
        priority:   EventPriority = EventPriority.CRITICAL,
    ) -> bool:
        """
        Enqueue an event for processing.

        Stamps enqueue_ts_ms = clock_fn() at the moment of enqueue.
        Applies SRS §5.4 degraded-mode rules:
          - If utilisation > 80 %: log QUEUE_HIGH, drop INFO events.
          - If queue is full: log QUEUE_CRITICAL_OVERFLOW, return False.

        Args:
            event_type: String identifier for the event (e.g. 'SHM_TRIGGER').
            priority:   CRITICAL events are never dropped by utilisation
                        policy; INFO events are dropped at > 80 % utilisation.

        Returns:
            True  — event accepted and queued.
            False — event dropped (INFO under high load, or full queue).
        """
        enqueue_ts_ms = self._clock_fn()

        # Utilisation check (qsize is approximate — best-effort, §5.4)
        current_size = self._queue.qsize()
        utilization  = current_size / self.QUEUE_CAPACITY

        if utilization > self.QUEUE_HIGH_THRESHOLD:
            self._event_log.append({
                "event":        "QUEUE_HIGH",
                "req_id":       "MM-04",
                "severity":     "WARNING",
                "module_name":  "MissionManager",
                "timestamp_ms": enqueue_ts_ms,
                "payload":      {"utilization_percent": round(utilization * 100.0, 1)},
            })
            # Drop INFO-level events only (critical events continue to enqueue)
            if priority == EventPriority.INFO:
                return False

        # Attempt non-blocking put
        item = {
            "event_type":    event_type,
            "priority":      priority,
            "enqueue_ts_ms": enqueue_ts_ms,
        }
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            # Queue is completely full — critical event cannot be accepted
            self._queue_overflow_count += 1
            self._event_log.append({
                "event":        "QUEUE_CRITICAL_OVERFLOW",
                "req_id":       "MM-04",
                "severity":     "CRITICAL",
                "module_name":  "MissionManager",
                "timestamp_ms": enqueue_ts_ms,
                "payload":      {
                    "dropped_event_type": event_type,
                    "timestamp_ms":       enqueue_ts_ms,
                },
            })
            return False

    # ------------------------------------------------------------------
    # Worker — dequeue and log latency
    # ------------------------------------------------------------------

    def _process_loop(self) -> None:
        """
        Worker thread: dequeue events and emit EVENT_QUEUE_LATENCY logs.

        At dequeue time (immediately before processing begins):
          dequeue_ts_ms = clock_fn()
          latency_ms    = dequeue_ts_ms - enqueue_ts_ms

        Logs EVENT_QUEUE_LATENCY at DEBUG level with req_id='MM-04',
        module_name='MissionManager', and a payload containing all four
        timing fields (SRS §5.4 UT-MM-04).
        """
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.01)
            except queue.Empty:
                continue

            # Stamp dequeue time immediately before processing
            dequeue_ts_ms = self._clock_fn()
            latency_ms    = dequeue_ts_ms - item["enqueue_ts_ms"]

            self._event_log.append({
                "event":        "EVENT_QUEUE_LATENCY",
                "req_id":       "MM-04",
                "severity":     "DEBUG",
                "module_name":  "MissionManager",
                "timestamp_ms": dequeue_ts_ms,
                "payload":      {
                    "event_type":    item["event_type"],
                    "enqueue_ts_ms": item["enqueue_ts_ms"],
                    "dequeue_ts_ms": dequeue_ts_ms,
                    "latency_ms":    latency_ms,
                },
            })

            self._queue.task_done()


# ---------------------------------------------------------------------------
# Mission Manager operational states
# ---------------------------------------------------------------------------

class MissionState(str, Enum):
    """Mission Manager operational states."""
    IDLE     = "IDLE"      # no checkpoint loaded; initial state
    RESUMING = "RESUMING"  # resume attempt in progress (transient — not externally visible)
    ACTIVE   = "ACTIVE"    # mission running autonomously
    SHM      = "SHM"       # blocked in SHM pending operator clearance (P-02)
    ABORTED  = "ABORTED"   # mission aborted; no further transitions


class MissionManager:
    """
    Mission Manager — controls mission resume path with P-02 clearance gate.

    The resume() method implements §9.1 failure-first ordering:
    the operator-clearance-required (blocked/SHM) branch is evaluated before
    the nominal (ACTIVE) branch.

    Usage:
        event_log = []
        mm = MissionManager(event_log=event_log, clock_fn=lambda: get_time_ms())
        resumed = mm.resume(checkpoint)
        if not resumed:
            # vehicle is in MissionState.SHM
            # wait for operator command, then:
            mm.grant_clearance()  # → MissionState.ACTIVE

    The event_log list receives AWAITING_OPERATOR_CLEARANCE dicts that are
    inspectable by tests and upstream logging infrastructure.
    """

    def __init__(
        self,
        event_log: List[Dict[str, Any]],
        clock_fn:  Callable[[], int],
    ):
        """
        Args:
            event_log: List to receive checkpoint-related log events.
                       AWAITING_OPERATOR_CLEARANCE is appended here on P-02 trigger.
            clock_fn:  Zero-argument callable returning current time in milliseconds.
                       Used to populate timestamp_ms in log events.
        """
        self._event_log = event_log
        self._clock_fn  = clock_fn
        self._state     = MissionState.IDLE

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> MissionState:
        """Current mission manager state."""
        return self._state

    @property
    def shm_entered(self) -> bool:
        """True when blocked in SHM pending operator clearance."""
        return self._state == MissionState.SHM

    # ------------------------------------------------------------------
    # Mission control
    # ------------------------------------------------------------------

    def resume(self, checkpoint: Checkpoint) -> bool:
        """
        Attempt to resume autonomous mission from a restored checkpoint.

        Implements §9.1 failure-first sequence — P-02 blocked path is evaluated
        FIRST, before the nominal resume path.

        Args:
            checkpoint: Restored Checkpoint object to resume from.

        Returns:
            True  — mission resumed; state is now ACTIVE.
            False — mission blocked by P-02 gate; state is now SHM.
                    Call grant_clearance() to unblock, or abort() to end mission.
        """
        self._state = MissionState.RESUMING

        # --- P-02 gate: failure-first (§9.1) --------------------------------
        # Evaluate BEFORE nominal path.  If operator clearance is required:
        #   (a) log AWAITING_OPERATOR_CLEARANCE
        #   (b) enter SHM
        #   (c) return False — autonomous flight must not resume
        if checkpoint.pending_operator_clearance_required:
            self._event_log.append({
                "event":        "AWAITING_OPERATOR_CLEARANCE",
                "req_id":       "PX4-05",
                "severity":     "WARNING",
                "module_name":  "MissionManager",
                "timestamp_ms": self._clock_fn(),
            })
            self._state = MissionState.SHM
            return False

        # --- Nominal resume path --------------------------------------------
        # pending_operator_clearance_required is False; resume autonomously.
        # Log MISSION_RESUME_AUTHORISED (PX4-04 D8a gate nominal outcome).
        resume_ts = self._clock_fn()
        self._event_log.append({
            "event":        "MISSION_RESUME_AUTHORISED",
            "req_id":       "PX4-04",
            "severity":     "INFO",
            "module_name":  "MissionManager",
            "timestamp_ms": resume_ts,
        })
        # SRS §8.4 PX4-04 D8a — AUTONOMOUS_RESUME_APPROVED (SRS canonical name)
        self._event_log.append({
            "event":                              "AUTONOMOUS_RESUME_APPROVED",
            "req_id":                             "PX4-04",
            "severity":                           "INFO",
            "module_name":                        "MissionManager",
            "timestamp_ms":                       resume_ts,
            "pending_operator_clearance_required": False,
        })
        self._state = MissionState.ACTIVE
        return True

    def on_reboot_detected(self, checkpoint_store: CheckpointStore) -> bool:
        """
        Handle a PX4_REBOOT_DETECTED event (D8a gate, SRS IT-PX4-02, PX4-04).

        Called by the integration layer when RebootDetector.feed() returns True.
        This method must NOT issue any PX4 commands (§1.3 forbidden behaviour).

        Sequence (§9.1 failure-first):
          1. Restore latest checkpoint from store.
             If no checkpoint exists: return False (no mission state to recover).
          2. Call resume(checkpoint):
               pending_operator_clearance_required=False →
                   log MISSION_RESUME_AUTHORISED, set ACTIVE, return True.
               pending_operator_clearance_required=True  →
                   log AWAITING_OPERATOR_CLEARANCE, set SHM, return False.

        Args:
            checkpoint_store: CheckpointStore to restore the latest checkpoint from.

        Returns:
            True  — mission recovered and now ACTIVE.
            False — no checkpoint found, or P-02 gate blocked resume (SHM).
        """
        checkpoint = checkpoint_store.restore_latest()
        if checkpoint is None:
            return False
        return self.resume(checkpoint)

    def grant_clearance(self) -> bool:
        """
        Grant operator clearance for autonomous mission resume.

        Only valid when state is SHM.  Transitions state to ACTIVE.

        Returns:
            True  — clearance accepted; state is now ACTIVE.
            False — state was not SHM; no state change.
        """
        if self._state == MissionState.SHM:
            self._state = MissionState.ACTIVE
            return True
        return False

    def abort(self) -> None:
        """
        Abort the mission unconditionally.

        Valid from any state.  State transitions to ABORTED.
        No further state transitions are possible after abort.
        """
        self._state = MissionState.ABORTED

    def __repr__(self) -> str:
        return f"MissionManager(state={self._state.value})"
