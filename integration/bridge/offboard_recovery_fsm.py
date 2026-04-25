"""
integration/bridge/offboard_recovery_fsm.py
MicroMind / NanoCorteX — D1→D2→D3→D6 OFFBOARD Recovery FSM
(SRS §8.1 PX4-01; Appendix D D1..D6)

Owns the full OFFBOARD timeout chain invoked when PX4 exits OFFBOARD
mode unexpectedly (heartbeat loss, mode reversion):

  D1 — loss detected; log OFFBOARD_LOSS
  D2 — 5 s reconnect window at 1 Hz (send SET_MODE OFFBOARD)
  D3 — 5 s additional window with SHM activated
  D6 — timeout exhausted; log OFFBOARD_UNRECOVERED; call abort_fn()

Design rules (CGM v3.4):
  - T-MON remains read-only (CGM §1.3). on_offboard_loss() spawns a
    daemon thread; T-MON itself never blocks.
  - SR-01: all waits in daemon thread use threading.Event().wait().
    No time.sleep(). No simulation clock (clock_fn is monotonic wall
    time only).
  - abort_fn() is called by the FSM on D6; the caller owns ABORT_MISS.
    The FSM does not emit ABORT_MISS.

Log events emitted:
  OFFBOARD_LOSS         WARNING  req_id=PX4-01  — D1 entry
  OFFBOARD_RESTORED     INFO     req_id=PX4-01  — D2 or D3 success
  SHM_ENTRY             WARNING  req_id=MM-03   — D3 entry
  SHM_EXIT              INFO     req_id=MM-03   — D3 success exit
  OFFBOARD_UNRECOVERED  WARNING  req_id=PX4-01  — D6

References:
  SRS §8.1 PX4-01, Appendix D D1..D6
  Code Governance Manual v3.4 §1.3, SR-01
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List


class OffboardRecoveryFSM:
    """
    D1→D2→D3→D6 OFFBOARD timeout chain (SRS §8.1 PX4-01, Appendix D).

    Args:
        send_set_mode_fn: Sends MAV_CMD_DO_SET_MODE; returns True on ACK.
                          Called with (base_mode=209, custom_mode=393216).
        event_log:        External list for structured log events.
        clock_fn:         Returns current monotonic wall time as float seconds.
                          Must NOT be the simulation clock (SR-01).
        abort_fn:         Called once on D6 to signal ABORT_MISS to the caller.
        d2_timeout_s:     D2 reconnect window in seconds (default 5.0).
        d3_timeout_s:     D3 additional window in seconds (default 5.0).
        retry_interval_s: Wait between SET_MODE attempts (default 1.0 — 1 Hz).
    """

    def __init__(
        self,
        send_set_mode_fn: Callable[[int, int], bool],
        event_log:        List[Dict[str, Any]],
        clock_fn:         Callable[[], float],
        abort_fn:         Callable[[], None],
        d2_timeout_s:     float = 5.0,
        d3_timeout_s:     float = 5.0,
        retry_interval_s: float = 1.0,
    ) -> None:
        self._send_set_mode_fn = send_set_mode_fn
        self._event_log        = event_log
        self._clock_fn         = clock_fn
        self._abort_fn         = abort_fn
        self._d2_timeout_s     = d2_timeout_s
        self._d3_timeout_s     = d3_timeout_s
        self._retry_interval_s = retry_interval_s
        self._wait_event       = threading.Event()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def on_offboard_loss(self, ts_ms: int) -> None:
        """Entry point: spawn daemon thread to run D1→D2→D3→D6 chain.

        Called by T-MON callback on OFFBOARD loss detection.
        Returns immediately — T-MON never blocks.

        Args:
            ts_ms: Monotonic timestamp (ms) at which loss was detected.
        """
        t = threading.Thread(
            target=self._run_chain,
            args=(ts_ms,),
            daemon=True,
            name="d6_recovery_fsm",
        )
        t.start()

    # ------------------------------------------------------------------
    # D1→D2→D3→D6 chain (runs in daemon thread)
    # ------------------------------------------------------------------

    def _run_chain(self, ts_ms: int) -> None:
        """Full D1→D2→D3→D6 timeout chain. Runs in daemon thread only."""
        start_s = self._clock_fn()

        # D1 — log OFFBOARD_LOSS
        self._event_log.append({
            "event":        "OFFBOARD_LOSS",
            "req_id":       "PX4-01",
            "severity":     "WARNING",
            "module_name":  "OffboardRecoveryFSM",
            "timestamp_ms": ts_ms,
            "gap_ms":       0,
        })

        attempts_made = 0

        # D2 — 5 s reconnect window
        while (self._clock_fn() - start_s) < self._d2_timeout_s:
            attempts_made += 1
            ack = self._send_set_mode_fn(209, 393216)
            if ack:
                elapsed_ms = int((self._clock_fn() - start_s) * 1000)
                self._event_log.append({
                    "event":          "OFFBOARD_RESTORED",
                    "req_id":         "PX4-01",
                    "severity":       "INFO",
                    "module_name":    "OffboardRecoveryFSM",
                    "timestamp_ms":   ts_ms + elapsed_ms,
                    "recovery_phase": "D2",
                    "elapsed_ms":     elapsed_ms,
                })
                return
            self._wait_event.wait(timeout=self._retry_interval_s)
            self._wait_event.clear()

        # D3 — SHM activated, 5 s additional window
        d3_entry_ms = int((self._clock_fn() - start_s) * 1000)
        self._event_log.append({
            "event":               "SHM_ENTRY",
            "req_id":              "MM-03",
            "severity":            "WARNING",
            "module_name":         "OffboardRecoveryFSM",
            "timestamp_ms":        ts_ms + d3_entry_ms,
            "shm_trigger_reason":  "OFFBOARD_TIMEOUT_D2",
        })

        d3_deadline_s = self._d2_timeout_s + self._d3_timeout_s
        while (self._clock_fn() - start_s) < d3_deadline_s:
            attempts_made += 1
            ack = self._send_set_mode_fn(209, 393216)
            if ack:
                elapsed_ms = int((self._clock_fn() - start_s) * 1000)
                self._event_log.append({
                    "event":        "SHM_EXIT",
                    "req_id":       "MM-03",
                    "severity":     "INFO",
                    "module_name":  "OffboardRecoveryFSM",
                    "timestamp_ms": ts_ms + elapsed_ms,
                })
                self._event_log.append({
                    "event":          "OFFBOARD_RESTORED",
                    "req_id":         "PX4-01",
                    "severity":       "INFO",
                    "module_name":    "OffboardRecoveryFSM",
                    "timestamp_ms":   ts_ms + elapsed_ms,
                    "recovery_phase": "D3",
                    "elapsed_ms":     elapsed_ms,
                })
                return
            self._wait_event.wait(timeout=self._retry_interval_s)
            self._wait_event.clear()

        # D6 — timeout exhausted
        total_elapsed_ms = int((self._clock_fn() - start_s) * 1000)
        self._event_log.append({
            "event":            "OFFBOARD_UNRECOVERED",
            "req_id":           "PX4-01",
            "severity":         "WARNING",
            "module_name":      "OffboardRecoveryFSM",
            "timestamp_ms":     ts_ms + total_elapsed_ms,
            "total_elapsed_ms": total_elapsed_ms,
            "attempts_made":    attempts_made,
        })
        self._abort_fn()
