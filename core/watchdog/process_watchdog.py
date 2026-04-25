"""
core/watchdog/process_watchdog.py
MicroMind / NanoCorteX — ProcessWatchdog (SRS §8.3 RS-03)

Owns the restartability classification registry and recovery dispatch
logic. Implements the decision path that a real watchdog would execute
on process heartbeat miss or unexpected exit.

SIL-level only: restart_fn / checkpoint_restore_fn are injected
callables — real SIGKILL stimulus test (ST-RESTART-01) is Phase D.

Restartability classes:
  RESTARTABLE_WITHOUT_SHM — restart immediately; no checkpoint needed.
  RESTARTABLE_WITH_SHM    — activate SHM, restart, restore checkpoint.
  NOT_RESTARTABLE         — ABORT_MISS immediately (e.g., ESKF core).

Log events emitted:
  PROCESS_FAILURE          WARNING   req_id=RS-03  — on_process_failure entry
  PROCESS_UNKNOWN          WARNING   req_id=RS-03  — unregistered process
  ESKF_CORE_FAILURE        CRITICAL  req_id=RS-03  — NOT_RESTARTABLE path
  SHM_ENTRY                WARNING   req_id=MM-03  — RESTARTABLE_WITH_SHM
  PROCESS_RESTARTED        INFO      req_id=RS-03  — per restart attempt
  NAV_RESTORED             INFO      req_id=RS-03  — checkpoint restore success
  SHM_EXIT                 INFO      req_id=MM-03  — RESTARTABLE_WITH_SHM success
  PROCESS_RESTART_FAILED   WARNING   req_id=RS-03  — all attempts exhausted

References:
  SRS §8.3 RS-03; Code Governance Manual v3.4 SR-01
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Restartability class constants
# ---------------------------------------------------------------------------

RESTARTABLE_WITHOUT_SHM: str = "RESTARTABLE_WITHOUT_SHM"
RESTARTABLE_WITH_SHM:    str = "RESTARTABLE_WITH_SHM"
NOT_RESTARTABLE:         str = "NOT_RESTARTABLE"


class ProcessWatchdog:
    """
    RS-03 restartability classification registry and recovery dispatcher.

    Args:
        event_log:              External list for structured log events.
        clock_fn:               Returns current monotonic wall time (seconds).
        abort_fn:               Called on NOT_RESTARTABLE or restart exhaustion.
        restart_fn:             Callable[[process_name], bool] — True on success.
        checkpoint_restore_fn:  Callable[[], bool] — True on success.
        registry:               Classification map {name → class constant}.
                                Defaults to PROCESS_REGISTRY if None.
        restart_timeout_s:      Per-attempt restart timeout (seconds).
        restore_timeout_s:      Per-attempt checkpoint restore timeout (seconds).
        max_restart_attempts:   Maximum restart loop iterations before abort.
    """

    PROCESS_REGISTRY: Dict[str, str] = {
        "EWManager":         RESTARTABLE_WITHOUT_SHM,
        "LogBus":            RESTARTABLE_WITHOUT_SHM,
        "NavigationManager": RESTARTABLE_WITH_SHM,
        "MissionManager":    RESTARTABLE_WITH_SHM,
        "PX4Bridge":         RESTARTABLE_WITH_SHM,
        "ESKFCore":          NOT_RESTARTABLE,
    }

    def __init__(
        self,
        event_log:             List[Dict[str, Any]],
        clock_fn:              Callable[[], float],
        abort_fn:              Callable[[], None],
        restart_fn:            Callable[[str], bool],
        checkpoint_restore_fn: Callable[[], bool],
        registry:              Optional[Dict[str, str]] = None,
        restart_timeout_s:     float = 2.0,
        restore_timeout_s:     float = 10.0,
        max_restart_attempts:  int   = 3,
    ) -> None:
        self._event_log             = event_log
        self._clock_fn              = clock_fn
        self._abort_fn              = abort_fn
        self._restart_fn            = restart_fn
        self._checkpoint_restore_fn = checkpoint_restore_fn
        self._registry              = registry if registry is not None else self.PROCESS_REGISTRY
        self._restart_timeout_s     = restart_timeout_s
        self._restore_timeout_s     = restore_timeout_s
        self._max_restart_attempts  = max_restart_attempts

    # -----------------------------------------------------------------------
    # Public entry point — SR-01: never blocks T-MON
    # -----------------------------------------------------------------------

    def on_process_failure(
        self, process_name: str, exit_code: int, ts_ms: int
    ) -> None:
        """
        Entry point: called when a process heartbeat is missed or exits.
        Dispatches recovery logic to a daemon thread (T-MON never blocks).

        Args:
            process_name: Registered process name string.
            exit_code:    Exit code / signal number reported by OS.
            ts_ms:        Monotonic timestamp (ms) at failure detection.
        """
        t = threading.Thread(
            target=self._handle_failure,
            args=(process_name, exit_code, ts_ms),
            daemon=True,
            name=f"watchdog_{process_name}",
        )
        t.start()

    # -----------------------------------------------------------------------
    # Recovery dispatch (daemon thread)
    # -----------------------------------------------------------------------

    def _handle_failure(
        self, process_name: str, exit_code: int, ts_ms: int
    ) -> None:
        """Full recovery chain. Runs in daemon thread only."""

        restartability_class = self._registry.get(process_name)

        # Step 1 — log PROCESS_FAILURE
        self._event_log.append({
            "event":                "PROCESS_FAILURE",
            "req_id":               "RS-03",
            "severity":             "WARNING",
            "module_name":          "ProcessWatchdog",
            "timestamp_ms":         ts_ms,
            "process_name":         process_name,
            "exit_code":            exit_code,
            "restartability_class": restartability_class or "UNKNOWN",
        })

        # Step 2 — unknown process
        if restartability_class is None:
            self._event_log.append({
                "event":        "PROCESS_UNKNOWN",
                "req_id":       "RS-03",
                "severity":     "WARNING",
                "module_name":  "ProcessWatchdog",
                "timestamp_ms": ts_ms,
                "process_name": process_name,
            })
            return

        # Step 3 — NOT_RESTARTABLE → ABORT_MISS immediately
        if restartability_class == NOT_RESTARTABLE:
            self._event_log.append({
                "event":        "ESKF_CORE_FAILURE",
                "req_id":       "RS-03",
                "severity":     "CRITICAL",
                "module_name":  "ProcessWatchdog",
                "timestamp_ms": ts_ms,
                "process_name": process_name,
            })
            self._abort_fn()
            return

        # Step 4 — RESTARTABLE_WITH_SHM: activate SHM before restart loop
        if restartability_class == RESTARTABLE_WITH_SHM:
            self._event_log.append({
                "event":               "SHM_ENTRY",
                "req_id":              "MM-03",
                "severity":            "WARNING",
                "module_name":         "ProcessWatchdog",
                "timestamp_ms":        ts_ms,
                "shm_trigger_reason":  "PROCESS_FAILURE",
            })

        # Steps 5–6 — restart loop
        wait_ev = threading.Event()
        for attempt in range(1, self._max_restart_attempts + 1):
            attempt_start_s = self._clock_fn()
            ok = self._restart_fn(process_name)
            restart_latency_ms = int((self._clock_fn() - attempt_start_s) * 1000)

            self._event_log.append({
                "event":              "PROCESS_RESTARTED",
                "req_id":             "RS-03",
                "severity":           "INFO",
                "module_name":        "ProcessWatchdog",
                "timestamp_ms":       ts_ms,
                "process_name":       process_name,
                "restart_attempt":    attempt,
                "restart_latency_ms": restart_latency_ms,
            })

            if restartability_class == RESTARTABLE_WITH_SHM:
                restored = self._checkpoint_restore_fn()
                if restored:
                    self._event_log.append({
                        "event":        "NAV_RESTORED",
                        "req_id":       "RS-03",
                        "severity":     "INFO",
                        "module_name":  "ProcessWatchdog",
                        "timestamp_ms": ts_ms,
                        "process_name": process_name,
                    })
                    self._event_log.append({
                        "event":        "SHM_EXIT",
                        "req_id":       "MM-03",
                        "severity":     "INFO",
                        "module_name":  "ProcessWatchdog",
                        "timestamp_ms": ts_ms,
                    })
                    return   # success
                # restore failed — continue to next attempt
                wait_ev.wait(timeout=0.01)
                wait_ev.clear()
                continue

            # RESTARTABLE_WITHOUT_SHM — single restart, no checkpoint
            return   # success

        # Step 7 — all attempts failed → abort
        self._event_log.append({
            "event":        "PROCESS_RESTART_FAILED",
            "req_id":       "RS-03",
            "severity":     "WARNING",
            "module_name":  "ProcessWatchdog",
            "timestamp_ms": ts_ms,
            "process_name": process_name,
            "attempts":     self._max_restart_attempts,
        })
        self._abort_fn()
