"""
tests/test_it_d10_gnss.py
MicroMind / NanoCorteX — IT-D10-GNSS-01: HOLD Mode Recovery Gates

IT-D10-GNSS-01  test_d10_hold_recovery_success        — ACK on first attempt
IT-D10-GNSS-02  test_d10_hold_recovery_retry_then_success — ACK on 3rd attempt
IT-D10-GNSS-03  test_d10_hold_recovery_all_fail        — all retries exhausted → D6
IT-D10-GNSS-04  test_d10_hold_constant_value           — PX4_HOLD_CUSTOM_MODE == 50_593_792

Requirements: SRS §8.4 PX4-04, Appendix D D10
SRS ref:      D10 recovery window (3× retry, 2s, total 6s)
Governance:   Code Governance Manual v3.4 SR-01
"""

from __future__ import annotations

import time
import unittest
from typing import List
from unittest.mock import Mock

from integration.bridge.hold_recovery import HoldRecoveryHandler
from integration.bridge.mavlink_bridge import PX4_HOLD_CUSTOM_MODE


def _wall_clock() -> int:
    """Monotonic wall-clock ms — compliant clock_fn for HoldRecoveryHandler."""
    return int(time.monotonic() * 1000)


class TestITD10GNSSDenied(unittest.TestCase):
    """
    IT-D10-GNSS-01: HoldRecoveryHandler unit gates.

    All tests exercise HoldRecoveryHandler directly with mocked
    send_set_mode_fn.  No PX4 connection or MAVLink required.
    """

    # -----------------------------------------------------------------------
    # IT-D10-GNSS-01 — success on first attempt
    # -----------------------------------------------------------------------

    def test_d10_hold_recovery_success(self):
        """ACK True on first call → D10_OFFBOARD_RESTORED(attempt=1), no retry."""
        event_log: List = []
        mock_send = Mock(return_value=True)

        handler = HoldRecoveryHandler(
            send_set_mode_fn=mock_send,
            event_log=event_log,
            clock_fn=_wall_clock,
            # use defaults: max_retries=3, retry_interval_s=2.0, total_window_s=6.0
        )

        result = handler.attempt_offboard_recovery(ts_ms=1000)

        # --- Return value -------------------------------------------------------
        self.assertTrue(result,
            "attempt_offboard_recovery must return True on first-call ACK")

        # --- D10_HOLD_DETECTED present -------------------------------------------
        detected = [e for e in event_log if e["event"] == "D10_HOLD_DETECTED"]
        self.assertEqual(len(detected), 1,
            "Expected exactly 1 D10_HOLD_DETECTED event")

        # --- D10_OFFBOARD_RESTORED present with attempt=1 -----------------------
        restored = [e for e in event_log if e["event"] == "D10_OFFBOARD_RESTORED"]
        self.assertEqual(len(restored), 1,
            "Expected exactly 1 D10_OFFBOARD_RESTORED event")
        self.assertEqual(restored[0]["attempt"], 1,
            f"D10_OFFBOARD_RESTORED attempt: expected 1, got {restored[0].get('attempt')}")

        # --- D10_RETRY absent ---------------------------------------------------
        retry_events = [e for e in event_log if e["event"] == "D10_RETRY"]
        self.assertEqual(len(retry_events), 0,
            f"Expected no D10_RETRY events on first-call success, got {len(retry_events)}")

        # --- send_set_mode_fn called exactly once with correct args -------------
        mock_send.assert_called_once_with(209, 393216)

    # -----------------------------------------------------------------------
    # IT-D10-GNSS-02 — fail twice, succeed on third attempt
    # -----------------------------------------------------------------------

    def test_d10_hold_recovery_retry_then_success(self):
        """ACK False × 2 then True → 2 D10_RETRY events, D10_OFFBOARD_RESTORED(attempt=3)."""
        event_log: List = []
        mock_send = Mock(side_effect=[False, False, True])

        handler = HoldRecoveryHandler(
            send_set_mode_fn=mock_send,
            event_log=event_log,
            clock_fn=_wall_clock,
            retry_interval_s=0.01,   # fast for test speed
            total_window_s=1.0,
        )

        result = handler.attempt_offboard_recovery(ts_ms=1000)

        # --- Return value -------------------------------------------------------
        self.assertTrue(result,
            "attempt_offboard_recovery must return True when 3rd attempt ACKs")

        # --- D10_RETRY count == 2 (attempts 1 and 2 failed) -------------------
        retry_events = [e for e in event_log if e["event"] == "D10_RETRY"]
        self.assertEqual(len(retry_events), 2,
            f"Expected 2 D10_RETRY events (2 failed attempts), got {len(retry_events)}")

        # --- D10_OFFBOARD_RESTORED with attempt=3 --------------------------------
        restored = [e for e in event_log if e["event"] == "D10_OFFBOARD_RESTORED"]
        self.assertEqual(len(restored), 1,
            "Expected exactly 1 D10_OFFBOARD_RESTORED event")
        self.assertEqual(restored[0]["attempt"], 3,
            f"D10_OFFBOARD_RESTORED attempt: expected 3, got {restored[0].get('attempt')}")

        # --- D10_RECOVERY_FAILED absent -----------------------------------------
        self.assertNotIn("D10_RECOVERY_FAILED",
                         {e["event"] for e in event_log},
                         "D10_RECOVERY_FAILED must not be logged on successful recovery")

        # --- send_set_mode_fn called exactly 3 times ---------------------------
        self.assertEqual(mock_send.call_count, 3,
            f"send_set_mode_fn should be called 3 times, got {mock_send.call_count}")

    # -----------------------------------------------------------------------
    # IT-D10-GNSS-03 — all retries exhausted → D6 path
    # -----------------------------------------------------------------------

    def test_d10_hold_recovery_all_fail(self):
        """All ACKs False → return False, D10_RECOVERY_FAILED, D10_RETRY × max_retries."""
        event_log: List = []
        mock_send = Mock(return_value=False)

        handler = HoldRecoveryHandler(
            send_set_mode_fn=mock_send,
            event_log=event_log,
            clock_fn=_wall_clock,
            retry_interval_s=0.01,   # fast for test speed
            total_window_s=1.0,
            # max_retries=3 (default)
        )

        result = handler.attempt_offboard_recovery(ts_ms=1000)

        # --- Return value -------------------------------------------------------
        self.assertFalse(result,
            "attempt_offboard_recovery must return False when all retries fail")

        # --- D10_RECOVERY_FAILED present ----------------------------------------
        failed = [e for e in event_log if e["event"] == "D10_RECOVERY_FAILED"]
        self.assertEqual(len(failed), 1,
            "Expected exactly 1 D10_RECOVERY_FAILED event")

        # --- D10_RETRY count == max_retries (3) ---------------------------------
        retry_events = [e for e in event_log if e["event"] == "D10_RETRY"]
        self.assertEqual(len(retry_events), 3,
            f"Expected D10_RETRY count == max_retries (3), got {len(retry_events)}")

        # --- D10_OFFBOARD_RESTORED absent ----------------------------------------
        self.assertNotIn("D10_OFFBOARD_RESTORED",
                         {e["event"] for e in event_log},
                         "D10_OFFBOARD_RESTORED must not be logged when all retries fail")

        # --- send_set_mode_fn called exactly max_retries times -----------------
        self.assertEqual(mock_send.call_count, 3,
            f"send_set_mode_fn should be called 3 times, got {mock_send.call_count}")

    # -----------------------------------------------------------------------
    # IT-D10-GNSS-04 — PX4_HOLD_CUSTOM_MODE constant value
    # -----------------------------------------------------------------------

    def test_d10_hold_constant_value(self):
        """PX4_HOLD_CUSTOM_MODE == 50_593_792 (AUTO_LOITER sub=3<<24 | main=4<<16)."""
        self.assertEqual(PX4_HOLD_CUSTOM_MODE, 50_593_792,
            f"PX4_HOLD_CUSTOM_MODE: expected 50593792, got {PX4_HOLD_CUSTOM_MODE}")


if __name__ == "__main__":
    unittest.main()
