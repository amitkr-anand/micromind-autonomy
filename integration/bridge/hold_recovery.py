"""
integration/bridge/hold_recovery.py
MicroMind / NanoCorteX — D10 HOLD Mode Recovery Handler (SRS §8.4 PX4-04)

Implements the D10 OFFBOARD re-entry sequence when PX4 returns to HOLD
(AUTO_LOITER) mode after a reboot, per SRS Appendix D D10:

  "Command SET_MODE OFFBOARD within 1s. Retry 3× at 2s.
   Proceed to D8a on success. Total window: 6s. Failure path: D6."

Design:
    - Called from a dedicated daemon thread spawned by T-MON.
      T-MON itself remains read-only per CGM §1.3.
    - All inter-attempt waits use threading.Event().wait() — SR-01 compliant.
    - clock_fn must supply monotonic wall time, NOT the simulation clock.

Log events emitted (appended to event_log):
    D10_HOLD_DETECTED      WARNING — at entry, before any attempt
    D10_OFFBOARD_RESTORED  INFO    — on successful ACK
    D10_RETRY              WARNING — after each failed attempt
    D10_RECOVERY_FAILED    WARNING — after window/retries exhausted

References:
    SRS §8.4 PX4-04, Appendix D D10
    Code Governance Manual v3.4 §1.3, SR-01
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List


class HoldRecoveryHandler:
    """
    D10: OFFBOARD re-entry from HOLD mode.

    Retries SET_MODE OFFBOARD up to max_retries times within total_window_s.
    All inter-attempt waits use threading.Event().wait() (SR-01 compliant).
    """

    def __init__(
        self,
        send_set_mode_fn: Callable[[int, int], bool],
        event_log:        List[Dict[str, Any]],
        clock_fn:         Callable[[], int],
        max_retries:      int   = 3,
        retry_interval_s: float = 2.0,
        total_window_s:   float = 6.0,
    ) -> None:
        """
        Args:
            send_set_mode_fn: Sends MAV_CMD_DO_SET_MODE; returns True on ACK.
                              Invoked with (base_mode=209, custom_mode=393216).
            event_log:        External list for D10 structured log events.
            clock_fn:         Returns current monotonic time as integer ms.
                              Must NOT be the simulation clock (SR-01).
            max_retries:      Maximum SET_MODE attempts (default 3).
            retry_interval_s: Wait between retries in seconds (default 2.0).
            total_window_s:   Total recovery window in seconds (default 6.0).
        """
        self._send_set_mode_fn = send_set_mode_fn
        self._event_log        = event_log
        self._clock_fn         = clock_fn
        self._max_retries      = max_retries
        self._retry_interval_s = retry_interval_s
        self._total_window_s   = total_window_s
        self._wait_event       = threading.Event()

    def attempt_offboard_recovery(self, ts_ms: int) -> bool:
        """
        D10: attempt OFFBOARD re-entry from HOLD mode.

        Runs in caller's thread (must be a non-monitor thread).
        Returns True if OFFBOARD restored within window. False → D6.

        Args:
            ts_ms: Monotonic timestamp (ms) at which HOLD was detected.
        """
        start_ms = self._clock_fn()

        self._event_log.append({
            "event":        "D10_HOLD_DETECTED",
            "req_id":       "PX4-04",
            "severity":     "WARNING",
            "module_name":  "HoldRecoveryHandler",
            "timestamp_ms": ts_ms,
        })

        attempts_made = 0
        for attempt in range(1, self._max_retries + 1):
            attempts_made = attempt
            ack = self._send_set_mode_fn(209, 393216)

            if ack:
                self._event_log.append({
                    "event":        "D10_OFFBOARD_RESTORED",
                    "req_id":       "PX4-04",
                    "severity":     "INFO",
                    "module_name":  "HoldRecoveryHandler",
                    "timestamp_ms": self._clock_fn(),
                    "attempt":      attempt,
                })
                return True

            elapsed_ms   = self._clock_fn() - start_ms
            remaining_ms = max(0, int(self._total_window_s * 1000) - elapsed_ms)

            self._event_log.append({
                "event":              "D10_RETRY",
                "req_id":             "PX4-04",
                "severity":           "WARNING",
                "module_name":        "HoldRecoveryHandler",
                "timestamp_ms":       self._clock_fn(),
                "attempt":            attempt,
                "remaining_window_ms": remaining_ms,
            })

            if elapsed_ms / 1000.0 >= self._total_window_s:
                break

            if attempt < self._max_retries:
                # SR-01 compliant wait — threading.Event().wait(), not time.sleep()
                self._wait_event.wait(timeout=self._retry_interval_s)
                self._wait_event.clear()
                if (self._clock_fn() - start_ms) / 1000.0 >= self._total_window_s:
                    break

        elapsed_ms = self._clock_fn() - start_ms
        self._event_log.append({
            "event":        "D10_RECOVERY_FAILED",
            "req_id":       "PX4-04",
            "severity":     "WARNING",
            "module_name":  "HoldRecoveryHandler",
            "timestamp_ms": self._clock_fn(),
            "attempts_made": attempts_made,
            "elapsed_ms":   elapsed_ms,
        })
        return False
