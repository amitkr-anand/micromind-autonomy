"""
core/mission_manager/mission_manager.py
MicroMind / NanoCorteX — Mission Manager

Controls the mission resume path and implements the P-02 operator clearance gate
(SRS §10.15, PX4-05 correction P-02).

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

Autonomous flight resumes only when:
    - grant_clearance() is called by an operator command handler, OR
    - abort() is called to end the mission

References:
    SRS §10.15  PX4-05 correction P-02
    Code Governance Manual v3.2  §9.1 (failure-first sequence)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List

from core.checkpoint.checkpoint import Checkpoint


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
        self._state = MissionState.ACTIVE
        return True

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
