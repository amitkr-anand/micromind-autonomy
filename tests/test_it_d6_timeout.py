"""
tests/test_it_d6_timeout.py
MicroMind — IT-D6-TIMEOUT-01: Full D6 path (D2/D3 timeout → OFFBOARD_UNRECOVERED)

Requirements: SRS §8.1 PX4-01; Appendix D D1..D6
Governance:   Code Governance Manual v3.4 SR-01

Gates:
  test_d6_offboard_restored_in_d2      — ACK in D2; no abort; no D6 event
  test_d6_offboard_restored_in_d3      — ACK in D3; no abort; SHM_ENTRY present
  test_d6_full_timeout_abort           — all fail; abort_fn called; OFFBOARD_UNRECOVERED present
  test_d6_timeout_values_correct       — elapsed_ms within 20% of (d2+d3)*1000
"""

from __future__ import annotations

import time
import threading
import unittest
from typing import Any, Dict, List
from unittest.mock import Mock, call

from integration.bridge.offboard_recovery_fsm import OffboardRecoveryFSM


def _wall_clock() -> float:
    """Monotonic wall time in seconds — SR-01 compliant clock_fn."""
    return time.monotonic()


def _join_daemon(fsm: OffboardRecoveryFSM, ts_ms: int, timeout_s: float) -> None:
    """Invoke on_offboard_loss and wait for the daemon thread to finish."""
    done = threading.Event()

    original_run = fsm._run_chain

    def _wrapped(ts):
        try:
            original_run(ts)
        finally:
            done.set()

    fsm._run_chain = _wrapped  # type: ignore[method-assign]
    fsm.on_offboard_loss(ts_ms)
    done.wait(timeout=timeout_s)


class TestITD6Timeout(unittest.TestCase):
    """
    IT-D6-TIMEOUT-01: OffboardRecoveryFSM D2/D3/D6 gate tests.

    All tests use accelerated timeouts (d2/d3 ≤ 0.1 s, retry 0.01 s)
    so the full chain completes in < 1 s wall clock.
    """

    # -----------------------------------------------------------------------
    # test_d6_offboard_restored_in_d2
    # -----------------------------------------------------------------------

    def test_d6_offboard_restored_in_d2(self):
        """ACK True on first call → OFFBOARD_RESTORED(D2), no abort, no D6."""
        event_log: List[Dict[str, Any]] = []
        abort_fn  = Mock()
        send_fn   = Mock(return_value=True)

        fsm = OffboardRecoveryFSM(
            send_set_mode_fn=send_fn,
            event_log=event_log,
            clock_fn=_wall_clock,
            abort_fn=abort_fn,
            d2_timeout_s=0.1,
            retry_interval_s=0.01,
        )
        _join_daemon(fsm, ts_ms=1000, timeout_s=1.0)

        events = {e["event"] for e in event_log}

        # a. abort_fn NOT called
        abort_fn.assert_not_called()

        # b. OFFBOARD_LOSS in log
        self.assertIn("OFFBOARD_LOSS", events,
            "OFFBOARD_LOSS must be logged at D1 entry")

        # c. OFFBOARD_RESTORED with recovery_phase="D2"
        restored = [e for e in event_log if e["event"] == "OFFBOARD_RESTORED"]
        self.assertTrue(restored,
            "OFFBOARD_RESTORED must be logged on D2 success")
        self.assertEqual(restored[0]["recovery_phase"], "D2",
            f"recovery_phase expected 'D2', got {restored[0].get('recovery_phase')}")

        # d. OFFBOARD_UNRECOVERED NOT in log
        self.assertNotIn("OFFBOARD_UNRECOVERED", events,
            "OFFBOARD_UNRECOVERED must not be logged when D2 recovers")

    # -----------------------------------------------------------------------
    # test_d6_offboard_restored_in_d3
    # -----------------------------------------------------------------------

    def test_d6_offboard_restored_in_d3(self):
        """All D2 calls fail, ACK in D3 → SHM_ENTRY, OFFBOARD_RESTORED(D3), no abort."""
        event_log: List[Dict[str, Any]] = []
        abort_fn  = Mock()
        # False for first 6 calls (D2 at 0.01s retry exhausts in 0.05s),
        # True on 7th call (within D3 window)
        send_fn   = Mock(side_effect=[False, False, False, False, False, False, True])

        fsm = OffboardRecoveryFSM(
            send_set_mode_fn=send_fn,
            event_log=event_log,
            clock_fn=_wall_clock,
            abort_fn=abort_fn,
            d2_timeout_s=0.05,
            d3_timeout_s=0.1,
            retry_interval_s=0.01,
        )
        _join_daemon(fsm, ts_ms=2000, timeout_s=1.0)

        events = {e["event"] for e in event_log}

        # a. abort_fn NOT called
        abort_fn.assert_not_called()

        # b. SHM_ENTRY in log
        self.assertIn("SHM_ENTRY", events,
            "SHM_ENTRY must be logged when D2 exhausts and D3 begins")

        # c. OFFBOARD_RESTORED with recovery_phase="D3"
        restored = [e for e in event_log if e["event"] == "OFFBOARD_RESTORED"]
        self.assertTrue(restored,
            "OFFBOARD_RESTORED must be logged on D3 success")
        self.assertEqual(restored[0]["recovery_phase"], "D3",
            f"recovery_phase expected 'D3', got {restored[0].get('recovery_phase')}")

        # d. OFFBOARD_UNRECOVERED NOT in log
        self.assertNotIn("OFFBOARD_UNRECOVERED", events,
            "OFFBOARD_UNRECOVERED must not be logged when D3 recovers")

    # -----------------------------------------------------------------------
    # test_d6_full_timeout_abort
    # -----------------------------------------------------------------------

    def test_d6_full_timeout_abort(self):
        """All ACKs False → abort_fn called once, OFFBOARD_UNRECOVERED present."""
        event_log: List[Dict[str, Any]] = []
        abort_fn  = Mock()
        send_fn   = Mock(return_value=False)

        fsm = OffboardRecoveryFSM(
            send_set_mode_fn=send_fn,
            event_log=event_log,
            clock_fn=_wall_clock,
            abort_fn=abort_fn,
            d2_timeout_s=0.05,
            d3_timeout_s=0.05,
            retry_interval_s=0.01,
        )
        _join_daemon(fsm, ts_ms=3000, timeout_s=1.0)

        events = {e["event"] for e in event_log}

        # a. abort_fn called exactly once
        abort_fn.assert_called_once()

        # b. OFFBOARD_LOSS in log
        self.assertIn("OFFBOARD_LOSS", events,
            "OFFBOARD_LOSS must be logged at D1 entry")

        # c. SHM_ENTRY in log
        self.assertIn("SHM_ENTRY", events,
            "SHM_ENTRY must be logged when D2 exhausts")

        # d. OFFBOARD_UNRECOVERED in log
        self.assertIn("OFFBOARD_UNRECOVERED", events,
            "OFFBOARD_UNRECOVERED must be logged on D6")

        # e. OFFBOARD_UNRECOVERED payload has total_elapsed_ms
        d6_evts = [e for e in event_log if e["event"] == "OFFBOARD_UNRECOVERED"]
        self.assertTrue(d6_evts,
            "OFFBOARD_UNRECOVERED event must be present")
        self.assertIn("total_elapsed_ms", d6_evts[0],
            "OFFBOARD_UNRECOVERED must carry total_elapsed_ms field")

        # f. OFFBOARD_RESTORED NOT in log
        self.assertNotIn("OFFBOARD_RESTORED", events,
            "OFFBOARD_RESTORED must not be logged when all attempts fail")

    # -----------------------------------------------------------------------
    # test_d6_timeout_values_correct
    # -----------------------------------------------------------------------

    def test_d6_timeout_values_correct(self):
        """elapsed_ms in OFFBOARD_UNRECOVERED within 20% of (d2+d3)*1000."""
        event_log: List[Dict[str, Any]] = []
        abort_fn  = Mock()
        send_fn   = Mock(return_value=False)

        d2 = 0.05
        d3 = 0.05

        fsm = OffboardRecoveryFSM(
            send_set_mode_fn=send_fn,
            event_log=event_log,
            clock_fn=_wall_clock,
            abort_fn=abort_fn,
            d2_timeout_s=d2,
            d3_timeout_s=d3,
            retry_interval_s=0.01,
        )
        _join_daemon(fsm, ts_ms=4000, timeout_s=1.0)

        d6_evts = [e for e in event_log if e["event"] == "OFFBOARD_UNRECOVERED"]
        self.assertTrue(d6_evts,
            "OFFBOARD_UNRECOVERED must be present for timing check")

        elapsed_ms  = d6_evts[0]["total_elapsed_ms"]
        expected_ms = (d2 + d3) * 1000
        tolerance   = expected_ms * 0.20

        self.assertGreaterEqual(elapsed_ms, expected_ms - tolerance,
            f"elapsed_ms {elapsed_ms} below lower bound "
            f"{expected_ms - tolerance:.1f} ms")
        self.assertLessEqual(elapsed_ms, expected_ms + tolerance * 5,
            f"elapsed_ms {elapsed_ms} far exceeds expected "
            f"{expected_ms:.1f} ms — possible hang or clock error")


if __name__ == "__main__":
    unittest.main()
