"""
tests/test_ut_rs03.py
MicroMind — UT-RS-03: ProcessWatchdog Restartability Decision Logic

Requirements: SRS §8.3 RS-03; Code Governance Manual v3.4

Gates:
    test_not_restartable_triggers_abort       — 5 assertions
    test_restartable_with_shm_success         — 6 assertions
    test_restartable_without_shm_success      — 5 assertions
    test_restartable_all_attempts_fail        — 5 assertions
    test_unknown_process_ignored              — 3 assertions
    test_registry_classification_correct      — 6 assertions

Total: 30 assertions.

Note: ST-RESTART-01 (real SIGKILL stimulus) is Phase D.
These tests validate decision logic only (SIL level).
"""

from __future__ import annotations

import threading
import time as _time
from typing import Any, Dict, List

import pytest

from core.watchdog.process_watchdog import (
    NOT_RESTARTABLE,
    RESTARTABLE_WITH_SHM,
    RESTARTABLE_WITHOUT_SHM,
    ProcessWatchdog,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_watchdog(
    event_log:              List[Dict[str, Any]],
    abort_fn,
    restart_fn,
    checkpoint_restore_fn=None,
    registry=None,
    max_restart_attempts:   int = 3,
) -> ProcessWatchdog:
    if checkpoint_restore_fn is None:
        checkpoint_restore_fn = lambda: False
    return ProcessWatchdog(
        event_log=event_log,
        clock_fn=_time.monotonic,
        abort_fn=abort_fn,
        restart_fn=restart_fn,
        checkpoint_restore_fn=checkpoint_restore_fn,
        registry=registry,
        max_restart_attempts=max_restart_attempts,
    )


def _events_of(log: List[Dict[str, Any]], event_name: str) -> List[Dict[str, Any]]:
    return [e for e in log if e.get("event") == event_name]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUTRS03:

    def test_not_restartable_triggers_abort(self) -> None:
        """NOT_RESTARTABLE → ESKF_CORE_FAILURE + abort_fn; restart_fn never called."""
        event_log: List[Dict[str, Any]] = []
        abort_fired = threading.Event()
        restart_call_count: List[int] = [0]

        def _restart_fn(name: str) -> bool:
            restart_call_count[0] += 1
            return False

        wdog = _make_watchdog(
            event_log=event_log,
            abort_fn=abort_fired.set,
            restart_fn=_restart_fn,
            registry={"ESKFCore": NOT_RESTARTABLE},
        )
        wdog.on_process_failure("ESKFCore", -9, ts_ms=1000)
        abort_fired.wait(timeout=0.5)

        pf = _events_of(event_log, "PROCESS_FAILURE")
        ekf = _events_of(event_log, "ESKF_CORE_FAILURE")

        assert abort_fired.is_set(), "a. abort_fn must be called for NOT_RESTARTABLE"
        assert pf and pf[-1]["restartability_class"] == NOT_RESTARTABLE, (
            "b. PROCESS_FAILURE must be logged with restartability_class=NOT_RESTARTABLE"
        )
        assert ekf, "c. ESKF_CORE_FAILURE must be logged"
        assert restart_call_count[0] == 0, "d. restart_fn must NOT be called"
        assert not _events_of(event_log, "NAV_RESTORED"), "e. NAV_RESTORED must not appear"

    def test_restartable_with_shm_success(self) -> None:
        """RESTARTABLE_WITH_SHM: success path — SHM_ENTRY → restart → restore → SHM_EXIT."""
        event_log: List[Dict[str, Any]] = []
        abort_called: List[bool] = [False]

        wdog = _make_watchdog(
            event_log=event_log,
            abort_fn=lambda: abort_called.__setitem__(0, True),
            restart_fn=lambda name: True,
            checkpoint_restore_fn=lambda: True,
            registry={"NavigationManager": RESTARTABLE_WITH_SHM},
        )
        done_ev = threading.Event()

        def _watch() -> None:
            while not _events_of(event_log, "SHM_EXIT"):
                done_ev.wait(timeout=0.01)
                done_ev.clear()
            done_ev.set()

        t = threading.Thread(target=_watch, daemon=True)
        t.start()
        wdog.on_process_failure("NavigationManager", -9, ts_ms=1000)
        done_ev.wait(timeout=0.5)

        assert not abort_called[0], "a. abort_fn must NOT be called on success"
        assert _events_of(event_log, "PROCESS_FAILURE"), "b. PROCESS_FAILURE must be logged"
        assert _events_of(event_log, "SHM_ENTRY"), "c. SHM_ENTRY must be logged"
        assert _events_of(event_log, "PROCESS_RESTARTED"), "d. PROCESS_RESTARTED must be logged"
        assert _events_of(event_log, "NAV_RESTORED"), "e. NAV_RESTORED must be logged"
        assert _events_of(event_log, "SHM_EXIT"), "f. SHM_EXIT must be logged"

    def test_restartable_without_shm_success(self) -> None:
        """RESTARTABLE_WITHOUT_SHM: restart only; no SHM or checkpoint."""
        event_log: List[Dict[str, Any]] = []
        abort_called: List[bool] = [False]
        done_ev = threading.Event()

        def _restart_fn(name: str) -> bool:
            done_ev.set()
            return True

        wdog = _make_watchdog(
            event_log=event_log,
            abort_fn=lambda: abort_called.__setitem__(0, True),
            restart_fn=_restart_fn,
            registry={"EWManager": RESTARTABLE_WITHOUT_SHM},
        )
        wdog.on_process_failure("EWManager", -9, ts_ms=1000)
        done_ev.wait(timeout=0.5)

        # Give daemon thread time to append PROCESS_RESTARTED after restart_fn returns
        threading.Event().wait(timeout=0.05)

        assert not abort_called[0], "a. abort_fn must NOT be called"
        assert _events_of(event_log, "PROCESS_FAILURE"), "b. PROCESS_FAILURE must be logged"
        assert not _events_of(event_log, "SHM_ENTRY"), "c. SHM_ENTRY must NOT be logged"
        assert _events_of(event_log, "PROCESS_RESTARTED"), "d. PROCESS_RESTARTED must be logged"
        assert not _events_of(event_log, "NAV_RESTORED"), (
            "e. NAV_RESTORED must NOT appear (no checkpoint restore for RESTARTABLE_WITHOUT_SHM)"
        )

    def test_restartable_all_attempts_fail(self) -> None:
        """RESTARTABLE_WITH_SHM: all restart attempts fail → abort_fn called."""
        event_log: List[Dict[str, Any]] = []
        abort_fired = threading.Event()

        wdog = _make_watchdog(
            event_log=event_log,
            abort_fn=abort_fired.set,
            restart_fn=lambda name: False,
            checkpoint_restore_fn=lambda: False,
            registry={"NavigationManager": RESTARTABLE_WITH_SHM},
            max_restart_attempts=3,
        )
        wdog.on_process_failure("NavigationManager", -9, ts_ms=1000)
        abort_fired.wait(timeout=1.0)

        prf = _events_of(event_log, "PROCESS_RESTART_FAILED")

        assert abort_fired.is_set(), "a. abort_fn must be called after all attempts fail"
        assert _events_of(event_log, "PROCESS_FAILURE"), "b. PROCESS_FAILURE must be logged"
        assert prf, "c. PROCESS_RESTART_FAILED must be logged"
        assert _events_of(event_log, "SHM_ENTRY"), "d. SHM_ENTRY must be logged before restart loop"
        assert not _events_of(event_log, "NAV_RESTORED"), "e. NAV_RESTORED must NOT appear"

    def test_unknown_process_ignored(self) -> None:
        """Unregistered process → PROCESS_UNKNOWN logged; abort_fn and restart_fn not called."""
        event_log: List[Dict[str, Any]] = []
        abort_called: List[bool] = [False]
        restart_call_count: List[int] = [0]
        done_ev = threading.Event()

        wdog = _make_watchdog(
            event_log=event_log,
            abort_fn=lambda: abort_called.__setitem__(0, True),
            restart_fn=lambda name: restart_call_count.__setitem__(0, restart_call_count[0] + 1) or False,
            registry={},
        )
        wdog.on_process_failure("UnknownProcess", -9, ts_ms=1000)
        # Give daemon thread time to run
        done_ev.wait(timeout=0.1)
        done_ev.clear()

        assert _events_of(event_log, "PROCESS_UNKNOWN"), "a. PROCESS_UNKNOWN must be logged"
        assert not abort_called[0], "b. abort_fn must NOT be called"
        assert restart_call_count[0] == 0, "c. restart_fn must NOT be called"

    def test_registry_classification_correct(self) -> None:
        """Default PROCESS_REGISTRY contains the correct classification for all 6 entries."""
        reg = ProcessWatchdog.PROCESS_REGISTRY

        assert reg["ESKFCore"]          == NOT_RESTARTABLE,         "a. ESKFCore → NOT_RESTARTABLE"
        assert reg["NavigationManager"] == RESTARTABLE_WITH_SHM,    "b. NavigationManager → RESTARTABLE_WITH_SHM"
        assert reg["MissionManager"]    == RESTARTABLE_WITH_SHM,    "c. MissionManager → RESTARTABLE_WITH_SHM"
        assert reg["PX4Bridge"]         == RESTARTABLE_WITH_SHM,    "d. PX4Bridge → RESTARTABLE_WITH_SHM"
        assert reg["EWManager"]         == RESTARTABLE_WITHOUT_SHM, "e. EWManager → RESTARTABLE_WITHOUT_SHM"
        assert reg["LogBus"]            == RESTARTABLE_WITHOUT_SHM, "f. LogBus → RESTARTABLE_WITHOUT_SHM"
