"""
tests/test_sb5_ec01.py
MicroMind / NanoCorteX — SB-5 Phase A IT-PX4-01 Formal Gate

EC01-01  test_ec01_01_offboard_continuity    ≥ 99.5 % continuity over 30-min mission
EC01-02  test_ec01_02_offboard_loss_count    ≤ 1 OFFBOARD_LOSS event per mission
EC01-03  test_ec01_03_setpoint_rate          ≥ 20 Hz setpoint dispatch rate

Requirements: PX4-01, EC-01
SRS ref:      §6.1, IT-PX4-01
Governance:   Code Governance Manual v3.2 §1.3, §1.4, §9.1

All three gates are independent.  Each uses a mock clock — no live PX4 required.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

import pytest

from integration.bridge.offboard_monitor import PX4ContinuityMonitor


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_monitor(ts_ms: int = 0) -> tuple[PX4ContinuityMonitor, List[Dict[str, Any]]]:
    """Return (PX4ContinuityMonitor, event_log) with a frozen mock clock."""
    event_log: List[Dict[str, Any]] = []
    monitor = PX4ContinuityMonitor(
        event_log=event_log,
        clock_fn=lambda: ts_ms,   # frozen at ts_ms — tests pass explicit ts_ms args
    )
    return monitor, event_log


def _events(log: List[Dict], name: str) -> List[Dict]:
    """Return all log entries matching event name."""
    return [e for e in log if e.get("event") == name]


# ---------------------------------------------------------------------------
# EC01-01 — OFFBOARD continuity ≥ 99.5 %
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEC0101OffboardContinuity(unittest.TestCase):
    """
    EC01-01: 30-minute mission (1 800 000 ms) with one OFFBOARD loss of exactly
    8 s.  Continuity must be ≥ 99.5 %.

    SRS §6.1 threshold: offboard_continuity_percent ≥ 99.5
    Operational margin: 8 000 ms loss / 1 800 000 ms mission = 0.444 % < 0.5 %
    """

    # Mission parameters
    MISSION_MS       = 1_800_000   # 30 minutes
    LOSS_START_MS    = 300_000     # loss at T+300 s
    LOSS_DURATION_MS = 8_000       # 8 s gap (≤ 9 s = ≤ 0.5 % budget)
    MIN_CONTINUITY   = 99.5

    def test_ec01_01_offboard_continuity(self) -> None:
        """
        OFFBOARD continuity ≥ 99.5 % after one 8 s loss in a 30-min mission.
        """
        monitor, event_log = _make_monitor()

        # Inject one OFFBOARD loss event
        loss_ts    = self.LOSS_START_MS
        restore_ts = self.LOSS_START_MS + self.LOSS_DURATION_MS

        monitor.record_offboard_loss(ts_ms=loss_ts)
        monitor.record_offboard_restored(ts_ms=restore_ts)

        # Compute continuity over full 30-minute mission
        continuity = monitor.compute_continuity(total_mission_ms=self.MISSION_MS)

        self.assertGreaterEqual(
            continuity, self.MIN_CONTINUITY,
            f"OFFBOARD continuity {continuity:.4f} % < SRS §6.1 threshold "
            f"{self.MIN_CONTINUITY} %",
        )

        # Verify OFFBOARD_LOSS logged
        loss_events = _events(event_log, "OFFBOARD_LOSS")
        self.assertEqual(len(loss_events), 1, "OFFBOARD_LOSS must be logged once")
        self.assertEqual(loss_events[0]["req_id"],      "PX4-01")
        self.assertEqual(loss_events[0]["severity"],    "WARNING")
        self.assertEqual(loss_events[0]["module_name"], "PX4Bridge")

        # Verify OFFBOARD_RESTORED logged with correct gap and discard flag
        restored_events = _events(event_log, "OFFBOARD_RESTORED")
        self.assertEqual(len(restored_events), 1, "OFFBOARD_RESTORED must be logged once")
        payload = restored_events[0]["payload"]
        self.assertEqual(payload["gap_duration_ms"],           self.LOSS_DURATION_MS)
        self.assertTrue(payload["stale_setpoints_discarded"],
                        "stale_setpoints_discarded must be True on recovery")


# ---------------------------------------------------------------------------
# EC01-02 — OFFBOARD_LOSS count ≤ 1
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEC0102OffboardLossCount(unittest.TestCase):
    """
    EC01-02: A nominal 30-minute mission must record ≤ 1 OFFBOARD_LOSS event.

    Tolerance of exactly 1 event accommodates the single transient link gap
    documented in the live SITL verification at 97b2f5a.
    """

    MISSION_MS       = 1_800_000
    LOSS_START_MS    = 600_000
    LOSS_DURATION_MS = 8_000

    def test_ec01_02_offboard_loss_count(self) -> None:
        """
        OFFBOARD_LOSS count ≤ 1 for nominal 30-min mission with one gap.
        """
        monitor, event_log = _make_monitor()

        monitor.record_offboard_loss(ts_ms=self.LOSS_START_MS)
        monitor.record_offboard_restored(ts_ms=self.LOSS_START_MS + self.LOSS_DURATION_MS)

        self.assertLessEqual(
            monitor.offboard_loss_count, 1,
            f"offboard_loss_count {monitor.offboard_loss_count} exceeds limit of 1",
        )

        # Accumulated loss must be exactly the injected gap
        self.assertEqual(
            monitor.total_offboard_loss_ms, self.LOSS_DURATION_MS,
            "total_offboard_loss_ms must equal the injected 8 s gap",
        )

        # Continuity is still ≥ 99.5 %
        continuity = monitor.compute_continuity(total_mission_ms=self.MISSION_MS)
        self.assertGreaterEqual(continuity, 99.5)


# ---------------------------------------------------------------------------
# EC01-03 — Setpoint rate ≥ 20 Hz
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEC0103SetpointRate(unittest.TestCase):
    """
    EC01-03: 20 setpoints dispatched at 20 Hz over a 1 s window must produce
    measured setpoint_rate_hz ≥ 20.0.  SETPOINT_RATE_LOW must NOT be logged.

    SRS §6.1: T-SP dispatches at 20 Hz (50 ms interval).  Rate log at 1 Hz.
    """

    # Inject 20 setpoints at 20 Hz (50 ms apart) starting at t=0 ms
    SETPOINT_COUNT    = 20
    SETPOINT_PERIOD   = 50        # ms → 20 Hz
    MEASURE_TS_MS     = 1_000     # measure at t=1 000 ms (end of window)
    MIN_RATE_HZ       = 20.0

    def test_ec01_03_setpoint_rate(self) -> None:
        """
        20 setpoints at 50 ms spacing → measured rate ≥ 20.0 Hz.
        SETPOINT_RATE_LOW must NOT be logged.
        """
        monitor, event_log = _make_monitor()

        # Inject 20 setpoints at 20 Hz — timestamps 0, 50, 100, ..., 950 ms
        for i in range(self.SETPOINT_COUNT):
            monitor.record_setpoint(ts_ms=i * self.SETPOINT_PERIOD)

        # Measure rate at end of 1 s window
        rate_hz = monitor.measure_rate_hz(ts_ms=self.MEASURE_TS_MS)
        self.assertGreaterEqual(
            rate_hz, self.MIN_RATE_HZ,
            f"setpoint_rate_hz {rate_hz:.1f} < SRS §6.1 minimum {self.MIN_RATE_HZ} Hz",
        )

        # Log rate at 1 Hz — must not produce SETPOINT_RATE_LOW
        monitor.log_setpoint_rate(ts_ms=self.MEASURE_TS_MS)

        low_rate = _events(event_log, "SETPOINT_RATE_LOW")
        self.assertEqual(
            len(low_rate), 0,
            "SETPOINT_RATE_LOW must NOT be logged when rate ≥ 20 Hz",
        )

        # SETPOINT_RATE_LOG must be present with correct fields
        rate_log = _events(event_log, "SETPOINT_RATE_LOG")
        self.assertEqual(len(rate_log), 1, "SETPOINT_RATE_LOG must be logged once")
        ev = rate_log[0]
        self.assertEqual(ev["req_id"],      "PX4-01")
        self.assertEqual(ev["severity"],    "DEBUG")
        self.assertEqual(ev["module_name"], "PX4Bridge")
        self.assertGreaterEqual(ev["payload"]["setpoint_rate_hz"], self.MIN_RATE_HZ)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
