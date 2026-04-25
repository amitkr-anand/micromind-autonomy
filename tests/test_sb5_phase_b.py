"""
tests/test_sb5_phase_b.py
MicroMind / NanoCorteX — SB-5 Phase B Gates

SB-01  test_sb01_retask_rejected_in_ins_only       UT-PLN-02 nav mode rejection
SB-02  test_sb02_retask_succeeds_gnss_denied       IT-PLN-01 retask under GNSS_DENIED
SB-03  test_sb03_rollback_restores_ew_and_terrain  R-03 rollback completeness
SB-04  test_sb04_retask_timeout_triggers_rollback  R-06 timeout + rollback
SB-05  test_sb05_dead_end_returns_to_last_waypoint PLN-03 dead-end recovery
SB-06  test_sb06_ut_mm04_queue_latency_under_load  UT-MM-04 queue latency (MM-04, SRS §5.4)

Requirements: PLN-02, PLN-03, EC-04, MM-04, EC-08
SRS ref:      §5.2, §5.4, IT-PLN-01, IT-PLN-02, UT-PLN-02, UT-MM-04,
              GAP-02, GAP-03, GAP-05
Governance:   Code Governance Manual v3.2 §1.3, §1.4, §9.1

All six gates are independent; setUp/tearDown isolate state between tests.
"""

from __future__ import annotations

import threading
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np

from core.ew_engine.ew_engine import EWEngine
from core.mission_manager.mission_manager import EventPriority, MissionEventBus
from core.route_planner.hybrid_astar import ReplanResult
from core.route_planner.route_planner import (
    RETASK_TIMEOUT_S,
    EW_MAP_STALENESS_THRESHOLD_S,
    RetaskNavMode,
    RoutePlanner,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_engine() -> EWEngine:
    """Return a zero-cost EWEngine (default state — A* always finds a path)."""
    return EWEngine()


def _make_clock(return_value: float = 0.0):
    """Return a mock clock whose .now() always returns return_value."""
    clock = MagicMock()
    clock.now.return_value = return_value
    return clock


def _make_planner(
    engine:           Optional[EWEngine] = None,
    clock=None,
    terrain_regen_fn=None,
    ew_refresh_fn=None,
    px4_upload_fn=None,
) -> Tuple[RoutePlanner, List[Dict[str, Any]]]:
    """
    Build a RoutePlanner with an isolated event_log.

    Returns (planner, event_log) so tests can inspect logged events.
    """
    if engine is None:
        engine = _make_engine()
    if clock is None:
        clock = _make_clock(0.0)

    event_log: List[Dict[str, Any]] = []
    planner = RoutePlanner(
        ew_engine        = engine,
        mission_clock    = clock,
        event_log        = event_log,
        terrain_regen_fn = terrain_regen_fn,
        ew_refresh_fn    = ew_refresh_fn,
        px4_upload_fn    = px4_upload_fn,
    )
    return planner, event_log


def _logged_events(event_log: List[Dict], event_name: str) -> List[Dict]:
    """Return all log entries with the given event name."""
    return [e for e in event_log if e.get("event") == event_name]


# Start / goal positions within the EWEngine grid (zero-cost → A* always succeeds)
_START_WP    = (30_000.0, 0.0, 4_000.0)
_GOAL_NORTH  = 60_000.0
_GOAL_EAST   = 0.0
_CRUISE_ALT  = 4_000.0


# ---------------------------------------------------------------------------
# SB-01 — UT-PLN-02 Nav mode rejection
# ---------------------------------------------------------------------------

class TestSB01RetaskRejectedInINSOnly(unittest.TestCase):
    """
    SB-01: Retask is accepted in CRUISE, rejected in TERMINAL and INS_ONLY.
    RETASK_REJECTED_INS_ONLY must be logged only for INS_ONLY, not for CRUISE.
    """

    def setUp(self):
        self.engine    = _make_engine()
        self.clock     = _make_clock(0.0)
        self.planner, self.event_log = _make_planner(
            engine = self.engine,
            clock  = self.clock,
        )
        self.planner.load_route([_START_WP])

    def tearDown(self):
        del self.planner, self.event_log, self.engine, self.clock

    # ------------------------------------------------------------------

    def test_sb01_retask_accepted_in_cruise(self):
        """Retask in CRUISE mode: returns True, RETASK_REJECTED_INS_ONLY NOT logged."""
        self.planner.nav_mode = RetaskNavMode.CRUISE
        result = self.planner.retask(
            new_goal_north_m = _GOAL_NORTH,
            new_goal_east_m  = _GOAL_EAST,
            cruise_alt_m     = _CRUISE_ALT,
        )
        self.assertTrue(result, "Retask in CRUISE mode must return True")
        self.assertFalse(
            _logged_events(self.event_log, "RETASK_REJECTED_INS_ONLY"),
            "RETASK_REJECTED_INS_ONLY must NOT be logged in CRUISE mode",
        )

    def test_sb01_retask_rejected_in_terminal(self):
        """Retask in TERMINAL mode: returns False."""
        self.planner.nav_mode = RetaskNavMode.TERMINAL
        result = self.planner.retask(
            new_goal_north_m = _GOAL_NORTH,
            new_goal_east_m  = _GOAL_EAST,
        )
        self.assertFalse(result, "Retask in TERMINAL mode must return False")

    def test_sb01_retask_rejected_in_ins_only(self):
        """
        Retask in INS_ONLY mode with cross_track_error_m exceeding corridor margin:
        returns False, RETASK_NAV_CONFIDENCE_TOO_LOW logged with correct fields.
        Threshold = half_width (500 m default) - 100 m = 400 m; 600 m exceeds it.
        """
        self.planner.nav_mode = RetaskNavMode.INS_ONLY
        result = self.planner.retask(
            new_goal_north_m    = _GOAL_NORTH,
            new_goal_east_m     = _GOAL_EAST,
            cross_track_error_m = 600.0,
        )
        self.assertFalse(result, "Retask in INS_ONLY mode with large XTE must return False")

        events = _logged_events(self.event_log, "RETASK_NAV_CONFIDENCE_TOO_LOW")
        self.assertEqual(len(events), 1, "RETASK_NAV_CONFIDENCE_TOO_LOW must be logged once")

        ev = events[0]
        self.assertEqual(ev["req_id"],              "PLN-02")
        self.assertEqual(ev["severity"],            "WARNING")
        self.assertEqual(ev["module_name"],         "RoutePlanner")
        self.assertIn("timestamp_ms", ev)
        self.assertEqual(ev["nav_mode"],            "INS_ONLY")
        self.assertEqual(ev["cross_track_error_m"], 600.0)
        self.assertEqual(ev["threshold_m"],         400.0)


# ---------------------------------------------------------------------------
# SB-02 — IT-PLN-01 Retask under GNSS_DENIED
# ---------------------------------------------------------------------------

class TestSB02RetaskSucceedsGnssDenied(unittest.TestCase):
    """
    SB-02: In GNSS_DENIED mode (nav degraded but not INS_ONLY), retask must
    complete successfully and log RETASK_COMPLETE.
    """

    def setUp(self):
        self.engine = _make_engine()
        self.clock  = _make_clock(0.0)
        self.planner, self.event_log = _make_planner(
            engine = self.engine,
            clock  = self.clock,
        )
        self.planner.load_route([_START_WP])
        self.planner.nav_mode = RetaskNavMode.GNSS_DENIED

    def tearDown(self):
        del self.planner, self.event_log, self.engine, self.clock

    def test_sb02_retask_succeeds_gnss_denied(self):
        """Retask in GNSS_DENIED mode completes; route is valid; RETASK_COMPLETE logged."""
        result = self.planner.retask(
            new_goal_north_m = _GOAL_NORTH,
            new_goal_east_m  = _GOAL_EAST,
            cruise_alt_m     = _CRUISE_ALT,
            trigger          = "IT-PLN-01",
        )
        self.assertTrue(result, "Retask in GNSS_DENIED mode must return True")

        # Route must be non-empty and contain valid (north, east, alt) tuples
        route = self.planner.waypoints
        self.assertGreater(len(route), 0, "Waypoint list must be non-empty after retask")
        for wp in route:
            self.assertEqual(len(wp), 3, "Each waypoint must be (north, east, alt)")

        # RETASK_COMPLETE must be logged
        events = _logged_events(self.event_log, "RETASK_COMPLETE")
        self.assertEqual(len(events), 1, "RETASK_COMPLETE must be logged once")
        ev = events[0]
        self.assertEqual(ev["req_id"],      "PLN-02")
        self.assertEqual(ev["severity"],    "INFO")
        self.assertEqual(ev["module_name"], "RoutePlanner")

        # RETASK_REJECTED_INS_ONLY must NOT be logged
        self.assertFalse(
            _logged_events(self.event_log, "RETASK_REJECTED_INS_ONLY"),
            "RETASK_REJECTED_INS_ONLY must NOT be logged in GNSS_DENIED mode",
        )


# ---------------------------------------------------------------------------
# SB-03 — R-03 Rollback completeness
# ---------------------------------------------------------------------------

class TestSB03RollbackRestoresEWAndTerrain(unittest.TestCase):
    """
    SB-03: On retask failure, rollback must restore EW map state, terrain
    corridor, and waypoint list to pre-retask snapshot values.
    """

    def setUp(self):
        self.engine = _make_engine()
        self.clock  = _make_clock(0.0)

        # Track whether terrain_regen and ew_refresh were called
        self.terrain_regen_called = False
        self.ew_refresh_called    = False

        def terrain_regen_fn():
            self.terrain_regen_called = True

        def ew_refresh_fn():
            # Simulate EW map refresh by setting some cells to 0.9
            self.ew_refresh_called = True
            self.engine.cost_map[:10, :10] = 0.9

        self.planner, self.event_log = _make_planner(
            engine           = self.engine,
            clock            = self.clock,
            terrain_regen_fn = terrain_regen_fn,
            ew_refresh_fn    = ew_refresh_fn,
        )

        # Establish initial state
        self.initial_waypoints = [_START_WP, (45_000.0, 0.0, 4_000.0)]
        self.planner.load_route(self.initial_waypoints)
        self.planner.nav_mode = RetaskNavMode.CRUISE

        # Set a non-trivial terrain corridor
        self.initial_terrain = np.ones((10, 10), dtype=np.float32) * 0.5
        self.planner.terrain_corridor = self.initial_terrain.copy()

        # Confirm EW map is zero before retask
        self.initial_ew_map = self.engine.cost_map.copy()

    def tearDown(self):
        del self.planner, self.event_log, self.engine, self.clock

    def test_sb03_rollback_restores_ew_and_terrain(self):
        """
        Force retask failure (patch replan to return success=False for all levels).
        After failure, EW map, terrain corridor, and waypoints must all match
        their pre-retask snapshot values.
        """
        # Build a ReplanResult stub for failure
        def _fail_replan(*args, **kwargs):
            return ReplanResult(
                replan_id            = "FAIL",
                trigger              = "test",
                mission_time_s       = 0.0,
                wall_latency_ms      = 1.0,
                success              = False,
                waypoints            = [],
                original_waypoints   = [],
                max_east_deviation_m = 0.0,
                kpi_ew02_pass        = True,
                nodes_explored       = 0,
            )

        with patch.object(self.planner._astar, "replan", side_effect=_fail_replan):
            result = self.planner.retask(
                new_goal_north_m = _GOAL_NORTH,
                new_goal_east_m  = _GOAL_EAST,
            )

        self.assertFalse(result, "Retask must return False when all replan attempts fail")

        # --- EW map restored ---
        np.testing.assert_array_equal(
            self.engine.cost_map,
            self.initial_ew_map,
            err_msg = "EW cost map must be restored to pre-retask snapshot",
        )

        # --- Terrain corridor restored ---
        restored_terrain = self.planner.terrain_corridor
        self.assertIsNotNone(restored_terrain, "Terrain corridor must not be None after rollback")
        np.testing.assert_array_equal(
            restored_terrain,
            self.initial_terrain,
            err_msg = "Terrain corridor must be restored to pre-retask snapshot",
        )

        # --- Waypoints: PLN-03 sets route to last valid waypoint after rollback ---
        # R-03 rollback restores waypoints internally; PLN-03 then sets the active
        # route to last_valid_waypoint so the vehicle always has a safe destination.
        route = self.planner.waypoints
        self.assertGreater(len(route), 0, "Route must not be empty after rollback + PLN-03")
        self.assertEqual(
            route[0],
            self.initial_waypoints[-1],   # last_valid_waypoint = final wp of initial route
            "PLN-03 must set route to last valid waypoint after dead-end",
        )

        # --- Terrain regen and EW refresh were called (R-01 ordering occurred) ---
        self.assertTrue(self.terrain_regen_called, "terrain_regen_fn must have been called")
        self.assertTrue(self.ew_refresh_called,    "ew_refresh_fn must have been called")


# ---------------------------------------------------------------------------
# SB-04 — R-06 Timeout + rollback
# ---------------------------------------------------------------------------

class TestSB04RetaskTimeoutTriggersRollback(unittest.TestCase):
    """
    SB-04: When the mission_clock advances past 15 s during the route search,
    RETASK_TIMEOUT_ROLLBACK must be logged and state must be rolled back.
    """

    def setUp(self):
        self.engine = _make_engine()

        # Mock clock: early calls return 0.0; call 6 returns 20.0 (timeout fires).
        # Call sequence inside retask():
        #   call 1: now_s = clock.now()               (initial ts, staleness check)
        #   call 2: clock.now() for snap_ew_age_ms    (IT-ROLLBACK-01 snapshot field)
        #   call 3: clock.now() for RETASK_TERRAIN_FIRST log timestamp
        #   call 4: clock.now() after ew_refresh (mark EW updated)
        #   call 5: retask_start_s = clock.now()
        #   call 6: elapsed = clock.now() - retask_start_s → 20.0 - 0.0 > 15 s → TIMEOUT
        #   call 7: final_ts_ms = clock.now()  (for RETASK_TIMEOUT_ROLLBACK log)
        self.clock = MagicMock()
        self.clock.now.side_effect = [0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0]

        self.initial_waypoints = [_START_WP]
        self.initial_terrain   = np.ones((5, 5), dtype=np.float32) * 0.3

        self.planner, self.event_log = _make_planner(
            engine = self.engine,
            clock  = self.clock,
        )
        self.planner.load_route(self.initial_waypoints)
        self.planner.nav_mode = RetaskNavMode.CRUISE
        self.planner.terrain_corridor = self.initial_terrain.copy()

        self.initial_ew_map = self.engine.cost_map.copy()

    def tearDown(self):
        del self.planner, self.event_log, self.engine, self.clock

    def test_sb04_retask_timeout_triggers_rollback(self):
        """
        Clock advancing past 15 s in the constraint loop must trigger
        RETASK_TIMEOUT_ROLLBACK and roll back EW map, terrain, and waypoints.
        """
        result = self.planner.retask(
            new_goal_north_m = _GOAL_NORTH,
            new_goal_east_m  = _GOAL_EAST,
            trigger          = "SB-04-TIMEOUT-TEST",
        )

        self.assertFalse(result, "Retask must return False on timeout")

        # --- RETASK_TIMEOUT_ROLLBACK logged ---
        events = _logged_events(self.event_log, "RETASK_TIMEOUT_ROLLBACK")
        self.assertEqual(len(events), 1, "RETASK_TIMEOUT_ROLLBACK must be logged once")
        ev = events[0]
        self.assertEqual(ev["req_id"],      "PLN-02")
        self.assertEqual(ev["severity"],    "WARNING")
        self.assertEqual(ev["module_name"], "RoutePlanner")
        self.assertIn("timestamp_ms", ev)

        # --- EW map rolled back ---
        np.testing.assert_array_equal(
            self.engine.cost_map,
            self.initial_ew_map,
            err_msg = "EW cost map must be restored after timeout rollback",
        )

        # --- Terrain corridor rolled back ---
        restored_terrain = self.planner.terrain_corridor
        self.assertIsNotNone(restored_terrain)
        np.testing.assert_array_equal(
            restored_terrain,
            self.initial_terrain,
            err_msg = "Terrain corridor must be restored after timeout rollback",
        )

        # --- Waypoints rolled back ---
        self.assertEqual(
            self.planner.waypoints,
            self.initial_waypoints,
            "Waypoints must be restored after timeout rollback",
        )

        # --- RETASK_COMPLETE must NOT be logged ---
        self.assertFalse(
            _logged_events(self.event_log, "RETASK_COMPLETE"),
            "RETASK_COMPLETE must NOT be logged when timeout fires",
        )


# ---------------------------------------------------------------------------
# SB-05 — PLN-03 Dead-end recovery
# ---------------------------------------------------------------------------

class TestSB05DeadEndReturnsToLastWaypoint(unittest.TestCase):
    """
    SB-05: When no valid route exists at any constraint level (and no timeout),
    DEAD_END_DETECTED must be logged and the route set to the last valid waypoint.
    The vehicle must never be left with an empty route.
    """

    def setUp(self):
        self.engine = _make_engine()
        self.clock  = _make_clock(0.0)

        self.last_valid_wp     = (50_000.0, 500.0, 4_000.0)
        self.initial_waypoints = [_START_WP, self.last_valid_wp]

        self.planner, self.event_log = _make_planner(
            engine = self.engine,
            clock  = self.clock,
        )
        self.planner.load_route(self.initial_waypoints)
        self.planner.nav_mode = RetaskNavMode.CRUISE

    def tearDown(self):
        del self.planner, self.event_log, self.engine, self.clock

    def test_sb05_dead_end_returns_to_last_waypoint(self):
        """
        All constraint levels return failure → DEAD_END_DETECTED logged,
        route set to last_valid_waypoint, no empty route returned.
        """
        def _fail_replan(*args, **kwargs):
            return ReplanResult(
                replan_id            = "DEAD",
                trigger              = "test",
                mission_time_s       = 0.0,
                wall_latency_ms      = 0.5,
                success              = False,
                waypoints            = [],
                original_waypoints   = [],
                max_east_deviation_m = 0.0,
                kpi_ew02_pass        = True,
                nodes_explored       = 0,
            )

        with patch.object(self.planner._astar, "replan", side_effect=_fail_replan):
            result = self.planner.retask(
                new_goal_north_m = _GOAL_NORTH,
                new_goal_east_m  = _GOAL_EAST,
                trigger          = "PLN-03-TEST",
            )

        self.assertFalse(result, "Retask must return False on dead-end")

        # --- DEAD_END_DETECTED logged ---
        events = _logged_events(self.event_log, "DEAD_END_DETECTED")
        self.assertEqual(len(events), 1, "DEAD_END_DETECTED must be logged once")
        ev = events[0]
        self.assertEqual(ev["req_id"],      "PLN-03")
        self.assertEqual(ev["severity"],    "WARNING")
        self.assertEqual(ev["module_name"], "RoutePlanner")
        self.assertIn("timestamp_ms", ev)

        # --- Vehicle route set to last valid waypoint ---
        route = self.planner.waypoints
        self.assertGreater(len(route), 0, "Route must never be empty after dead-end recovery")
        self.assertEqual(
            route[0],
            self.last_valid_wp,
            "Dead-end recovery must set route to last_valid_waypoint",
        )

        # --- RETASK_COMPLETE must NOT be logged ---
        self.assertFalse(
            _logged_events(self.event_log, "RETASK_COMPLETE"),
            "RETASK_COMPLETE must NOT be logged on dead-end",
        )

        # --- RETASK_TIMEOUT_ROLLBACK must NOT be logged (timeout did not fire) ---
        self.assertFalse(
            _logged_events(self.event_log, "RETASK_TIMEOUT_ROLLBACK"),
            "RETASK_TIMEOUT_ROLLBACK must NOT be logged on dead-end (no timeout)",
        )


# ---------------------------------------------------------------------------
# R-03 ETA rollback gate (OI-56) — PLN-02 Appendix B ROLLBACK action (5)
# ---------------------------------------------------------------------------

class TestR03ETARollback(unittest.TestCase):
    """
    R-03: On retask failure, _rollback() must restore _eta_s to the pre-retask
    value, and RETASK_ROLLBACK must be logged immediately after _rollback() with
    eta_s_restored in the payload.

    Requirements: PLN-02 R-03, OI-56
    SRS ref:      §4.2 Appendix B ROLLBACK action (5)
    Governance:   Code Governance Manual v3.4 §1.3, §1.4, §9.1
    """

    def setUp(self):
        self.engine = _make_engine()
        self.clock  = _make_clock(0.0)
        self.planner, self.event_log = _make_planner(
            engine = self.engine,
            clock  = self.clock,
        )
        self.planner.load_route([_START_WP])
        self.planner.nav_mode = RetaskNavMode.CRUISE
        # Establish a known pre-retask ETA (1 500 s — arbitrary non-zero sentinel)
        self._pre_retask_eta_s = 1500.0
        self.planner._eta_s    = self._pre_retask_eta_s

    def tearDown(self):
        del self.planner, self.event_log, self.engine, self.clock

    def test_r03_eta_rollback(self):
        """
        Force retask failure via dead-end (all replans return success=False,
        no timeout so mock clock stays at 0.0).

        Assertions:
          (a) _eta_s restored to pre-retask value after rollback
          (b) RETASK_ROLLBACK logged exactly once
          (c) RETASK_ROLLBACK payload contains eta_s_restored field
          (d) eta_s_restored equals the pre-retask _eta_s sentinel
        """
        def _fail_replan(*args, **kwargs) -> ReplanResult:
            return ReplanResult(
                replan_id            = "FAIL-R03",
                trigger              = "test",
                mission_time_s       = 0.0,
                wall_latency_ms      = 0.5,
                success              = False,
                waypoints            = [],
                original_waypoints   = [],
                max_east_deviation_m = 0.0,
                kpi_ew02_pass        = True,
                nodes_explored       = 0,
            )

        with patch.object(self.planner._astar, "replan", side_effect=_fail_replan):
            result = self.planner.retask(
                new_goal_north_m = _GOAL_NORTH,
                new_goal_east_m  = _GOAL_EAST,
                trigger          = "R03-ETA-TEST",
            )

        self.assertFalse(result, "Retask must return False on dead-end")

        # (a) _eta_s restored to pre-retask value
        self.assertAlmostEqual(
            self.planner._eta_s,
            self._pre_retask_eta_s,
            places=6,
            msg=f"_eta_s must be restored to {self._pre_retask_eta_s} s after rollback",
        )

        # (b) RETASK_ROLLBACK logged exactly once
        rollback_events = _logged_events(self.event_log, "RETASK_ROLLBACK")
        self.assertEqual(
            len(rollback_events), 1,
            "RETASK_ROLLBACK must be logged exactly once on retask failure",
        )

        # (c) RETASK_ROLLBACK payload contains eta_s_restored
        ev = rollback_events[0]
        self.assertIn(
            "eta_s_restored", ev,
            "RETASK_ROLLBACK event must contain eta_s_restored field",
        )
        self.assertEqual(ev["req_id"],      "PLN-02")
        self.assertEqual(ev["severity"],    "WARNING")
        self.assertEqual(ev["module_name"], "RoutePlanner")
        self.assertIn("timestamp_ms", ev)

        # (d) eta_s_restored equals the pre-retask sentinel
        self.assertAlmostEqual(
            ev["eta_s_restored"],
            self._pre_retask_eta_s,
            places=6,
            msg=(
                f"eta_s_restored must equal pre-retask _eta_s "
                f"({self._pre_retask_eta_s} s)"
            ),
        )


# ---------------------------------------------------------------------------
# SB-06 — UT-MM-04 queue latency under load
# ---------------------------------------------------------------------------

class TestSB06UTmm04QueueLatencyUnderLoad(unittest.TestCase):
    """
    SB-06: MM-04 event bus delivers 20 critical events within 100 ms latency
    while a background thread consumes ~70 % CPU (ST-CPU-01 load simulation).

    Requirements: MM-04, EC-08
    SRS ref:      §5.4, UT-MM-04, GAP-05
    Governance:   Code Governance Manual v3.2 §1.3, §1.4, §9.1
    """

    # Number of events to inject and expected delivery count
    _EVENT_COUNT    = 20
    # Injection rate: 50 Hz → one event every 20 ms
    _INJECT_PERIOD  = 0.020
    # Maximum allowed latency per event (SRS §5.4 / §6.4)
    _MAX_LATENCY_MS = 100.0
    # Wait budget for processing after last injection
    _DRAIN_TIMEOUT  = 5.0

    def setUp(self) -> None:
        # --- Busy-loop thread (~70 % CPU) ---
        self._busy_stop   = threading.Event()
        self._busy_thread = threading.Thread(
            target=self._busy_loop, daemon=True, name="SB06-busyloop"
        )
        self._busy_thread.start()

        # --- Event bus under test ---
        self._event_log: List[Dict[str, Any]] = []
        # Real wall-clock milliseconds — needed to measure actual delivery
        # latency in a real-time injection test.
        self._clock_fn = lambda: int(time.monotonic() * 1000)
        self._bus = MissionEventBus(
            event_log=self._event_log,
            clock_fn=self._clock_fn,
        )
        self._bus.start()

    def _busy_loop(self) -> None:
        """
        Simulate ST-CPU-01 load: consume ~70 % CPU on one thread.
        Pattern: busy-spin 7 ms, sleep 3 ms → 70 % duty cycle.
        Uses time.monotonic() for pacing (test infrastructure only).
        """
        while not self._busy_stop.is_set():
            deadline = time.monotonic() + 0.007
            while time.monotonic() < deadline:
                pass  # tight busy-spin
            time.sleep(0.003)  # yield to allow other threads to run

    def tearDown(self) -> None:
        # Stop event bus worker first
        self._bus.stop(timeout=2.0)
        # Stop busy-loop thread
        self._busy_stop.set()
        self._busy_thread.join(timeout=2.0)

    # ------------------------------------------------------------------

    def test_sb06_ut_mm04_queue_latency_under_load(self) -> None:
        """
        Inject 20 critical events at 50 Hz under ~70 % CPU load.
        Assert all 20 delivered within 100 ms latency (SRS §5.4).

        Gate assertions:
          (a) delivered_count == 20
          (b) max(latency_ms) <= 100 ms
          (c) len(EVENT_QUEUE_LATENCY log entries) == 20
          (d) queue_overflow_count == 0  (no critical events dropped)
        """
        # Inject 20 critical events at 50 Hz
        for i in range(self._EVENT_COUNT):
            self._bus.enqueue(f"CRITICAL_SHM_TRIGGER_{i}", EventPriority.CRITICAL)
            time.sleep(self._INJECT_PERIOD)

        # Wait for all events to be processed (poll up to _DRAIN_TIMEOUT)
        deadline = time.monotonic() + self._DRAIN_TIMEOUT
        while time.monotonic() < deadline:
            latency_entries = [
                e for e in self._event_log
                if e.get("event") == "EVENT_QUEUE_LATENCY"
            ]
            if len(latency_entries) >= self._EVENT_COUNT:
                break
            time.sleep(0.010)
        else:
            latency_entries = [
                e for e in self._event_log
                if e.get("event") == "EVENT_QUEUE_LATENCY"
            ]

        # (a) All 20 critical events delivered
        delivered_count = len(latency_entries)
        self.assertEqual(
            delivered_count, self._EVENT_COUNT,
            f"Expected {self._EVENT_COUNT} events delivered, got {delivered_count}",
        )

        # (b) All latency values ≤ 100 ms (SRS §5.4 / §6.4 critical event delivery budget)
        latencies = [e["payload"]["latency_ms"] for e in latency_entries]
        max_latency_ms = max(latencies)
        self.assertLessEqual(
            max_latency_ms, self._MAX_LATENCY_MS,
            f"Max latency {max_latency_ms:.1f} ms exceeds SRS §5.4 limit of "
            f"{self._MAX_LATENCY_MS} ms",
        )

        # (c) EVENT_QUEUE_LATENCY logged for all 20 events
        self.assertEqual(
            len(latency_entries), self._EVENT_COUNT,
            "EVENT_QUEUE_LATENCY must be logged for every delivered event",
        )

        # (d) No critical events dropped (overflow count == 0)
        self.assertEqual(
            self._bus.queue_overflow_count, 0,
            "queue_overflow_count must be 0 — no critical events may be dropped",
        )


# ---------------------------------------------------------------------------
# SB-07 — RS-04 Route fragment cleanup
# ---------------------------------------------------------------------------

class TestSB07RS04RouteFragmentCleanup(unittest.TestCase):
    """
    SB-07: RS-04 route fragment cleanup — intermediate waypoint lists generated
    during retask constraint-relaxation levels must be explicitly cleared after
    every retask operation.

    (a) Successful retask: ROUTE_FRAGMENT_CLEANUP logged with fragments_cleared >= 0.
    (b) Failed retask (forced rollback via impassable map): ROUTE_FRAGMENT_CLEANUP
        logged after rollback with fragments_cleared >= 0.
    (c) Memory stability: 10 consecutive failed retask operations do not grow
        _intermediate_fragments — count remains 0 after each operation.

    Requirements: RS-04, E-02
    SRS ref: §11.4, RS-04 v1.2
    Governance: Code Governance Manual v3.2 §1.3, §1.4, §9.1
    """

    def setUp(self):
        self.engine = _make_engine()
        self.clock  = _make_clock(0.0)
        self.planner, self.event_log = _make_planner(
            engine = self.engine,
            clock  = self.clock,
        )
        self.planner.load_route([_START_WP])
        self.planner.nav_mode = RetaskNavMode.CRUISE

    def tearDown(self):
        del self.planner, self.event_log, self.engine, self.clock

    # ------------------------------------------------------------------
    # Shared failure stub
    # ------------------------------------------------------------------

    @staticmethod
    def _fail_replan(*args, **kwargs) -> ReplanResult:
        """Stub replan: always returns success=False (impassable map simulation)."""
        return ReplanResult(
            replan_id            = "FAIL-SB07",
            trigger              = "test",
            mission_time_s       = 0.0,
            wall_latency_ms      = 1.0,
            success              = False,
            waypoints            = [],
            original_waypoints   = [],
            max_east_deviation_m = 0.0,
            kpi_ew02_pass        = True,
            nodes_explored       = 0,
        )

    # ------------------------------------------------------------------

    def test_sb07_rs04_route_fragment_cleanup(self):
        """
        Three sub-assertions:
          (a) Successful retask → ROUTE_FRAGMENT_CLEANUP logged, fragments_cleared >= 0.
          (b) Failed retask (forced rollback) → ROUTE_FRAGMENT_CLEANUP logged, fragments_cleared >= 0.
          (c) Memory stability: 10 consecutive failures → _intermediate_fragments == 0 each iteration.
        """
        # ── (a) Successful retask ─────────────────────────────────────────────
        result = self.planner.retask(
            new_goal_north_m = _GOAL_NORTH,
            new_goal_east_m  = _GOAL_EAST,
            cruise_alt_m     = _CRUISE_ALT,
            trigger          = "SB-07-SUCCESS",
        )
        self.assertTrue(result, "(a) Successful retask must return True")

        cleanup_events = _logged_events(self.event_log, "ROUTE_FRAGMENT_CLEANUP")
        self.assertGreaterEqual(
            len(cleanup_events), 1,
            "(a) ROUTE_FRAGMENT_CLEANUP must be logged after successful retask",
        )
        ev = cleanup_events[-1]
        self.assertGreaterEqual(
            ev["payload"]["fragments_cleared"], 0,
            "(a) fragments_cleared must be >= 0",
        )
        self.assertEqual(ev["req_id"],      "RS-04")
        self.assertEqual(ev["severity"],    "DEBUG")
        self.assertEqual(ev["module_name"], "RoutePlanner")
        self.assertIn("timestamp_ms", ev)

        # ── (b) Failed retask — forced rollback via impassable map ────────────
        self.event_log.clear()
        with patch.object(self.planner._astar, "replan", side_effect=self._fail_replan):
            result = self.planner.retask(
                new_goal_north_m = _GOAL_NORTH,
                new_goal_east_m  = _GOAL_EAST,
                trigger          = "SB-07-FAIL",
            )
        self.assertFalse(result, "(b) Failed retask must return False")

        cleanup_events = _logged_events(self.event_log, "ROUTE_FRAGMENT_CLEANUP")
        self.assertGreaterEqual(
            len(cleanup_events), 1,
            "(b) ROUTE_FRAGMENT_CLEANUP must be logged after failed retask rollback",
        )
        ev = cleanup_events[-1]
        self.assertGreaterEqual(
            ev["payload"]["fragments_cleared"], 0,
            "(b) fragments_cleared must be >= 0 after rollback",
        )
        self.assertEqual(ev["req_id"],      "RS-04")
        self.assertEqual(ev["severity"],    "DEBUG")
        self.assertEqual(ev["module_name"], "RoutePlanner")

        # ── (c) Memory stability — 10 consecutive failed retask operations ────
        for i in range(10):
            self.event_log.clear()
            with patch.object(self.planner._astar, "replan", side_effect=self._fail_replan):
                self.planner.retask(
                    new_goal_north_m = _GOAL_NORTH,
                    new_goal_east_m  = _GOAL_EAST,
                    trigger          = f"SB-07-STABILITY-{i}",
                )
            fragment_count = len(self.planner._intermediate_fragments)
            self.assertEqual(
                fragment_count, 0,
                f"(c) _intermediate_fragments must be 0 after iteration {i}, "
                f"got {fragment_count} — fragment accumulation detected",
            )


# ---------------------------------------------------------------------------
# W2-8 — IT-PLN-02 GNSS-denied retask integration
# ---------------------------------------------------------------------------

class TestW28GnssDeniedRetaskIntegration(unittest.TestCase):
    """
    W2-8 IT-PLN-02 GNSS-denied retask integration gates.

    Four tests covering the full IT-PLN-02 requirement surface:
      (1) nominal GNSS_DENIED retask — R-01..R-06 pass
      (2) GNSS_DENIED rollback — _eta_s restored, RETASK_ROLLBACK logged
      (3) INS_ONLY rejection — XTE above corridor margin → False
      (4) INS_ONLY permit   — XTE below corridor margin → True

    Requirements: PLN-02, R-01..R-06, R-05 conditional
    SRS ref:      §5.2, IT-PLN-02
    Governance:   Code Governance Manual v3.4 §1.3, §1.4, §9.1
    """

    def setUp(self):
        self.engine = _make_engine()
        self.clock  = _make_clock(0.0)
        self.planner, self.event_log = _make_planner(
            engine = self.engine,
            clock  = self.clock,
        )
        self.planner.load_route([_START_WP])

    def tearDown(self):
        del self.planner, self.event_log, self.engine, self.clock

    # -----------------------------------------------------------------------
    # (1) test_gnss_denied_retask_nominal
    # -----------------------------------------------------------------------

    def test_gnss_denied_retask_nominal(self):
        """
        IT-PLN-02 nominal: GNSS_DENIED retask completes with R-01..R-06 all pass.

        Assertions:
          (a) retask() returns True in GNSS_DENIED mode
          (b) RETASK_COMPLETE logged exactly once with req_id=PLN-02, severity=INFO
          (c) RETASK_COMPLETE module_name == RoutePlanner
          (d) Resulting route is non-empty with valid 3-tuple waypoints
          (e) RETASK_REJECTED_INS_ONLY NOT logged
          (f) RETASK_NAV_CONFIDENCE_TOO_LOW NOT logged
        """
        self.planner.nav_mode = RetaskNavMode.GNSS_DENIED

        result = self.planner.retask(
            new_goal_north_m = _GOAL_NORTH,
            new_goal_east_m  = _GOAL_EAST,
            cruise_alt_m     = _CRUISE_ALT,
            trigger          = "IT-PLN-02",
        )

        # (a)
        self.assertTrue(result, "Retask in GNSS_DENIED mode must return True")

        # (b)/(c)
        complete_events = _logged_events(self.event_log, "RETASK_COMPLETE")
        self.assertEqual(len(complete_events), 1,
            "RETASK_COMPLETE must be logged exactly once")
        ev = complete_events[0]
        self.assertEqual(ev["req_id"],      "PLN-02")
        self.assertEqual(ev["severity"],    "INFO")
        self.assertEqual(ev["module_name"], "RoutePlanner")

        # (d)
        route = self.planner.waypoints
        self.assertGreater(len(route), 0, "Route must be non-empty after GNSS_DENIED retask")
        for wp in route:
            self.assertEqual(len(wp), 3, "Each waypoint must be a (north, east, alt) 3-tuple")

        # (e)
        self.assertFalse(
            _logged_events(self.event_log, "RETASK_REJECTED_INS_ONLY"),
            "RETASK_REJECTED_INS_ONLY must NOT be logged in GNSS_DENIED mode",
        )

        # (f)
        self.assertFalse(
            _logged_events(self.event_log, "RETASK_NAV_CONFIDENCE_TOO_LOW"),
            "RETASK_NAV_CONFIDENCE_TOO_LOW must NOT be logged in GNSS_DENIED mode",
        )

    # -----------------------------------------------------------------------
    # (2) test_gnss_denied_retask_rollback_eta_restored
    # -----------------------------------------------------------------------

    def test_gnss_denied_retask_rollback_eta_restored(self):
        """
        IT-PLN-02 rollback: _eta_s restored in GNSS_DENIED mode on dead-end failure.

        Forces dead-end (all replans return success=False) while in GNSS_DENIED mode.

        Assertions:
          (a) retask() returns False on dead-end
          (b) _eta_s restored to pre-retask sentinel (1500.0 s)
          (c) RETASK_ROLLBACK logged exactly once
          (d) RETASK_ROLLBACK payload eta_s_restored == 1500.0
          (e) RETASK_ROLLBACK has req_id=PLN-02, severity=WARNING
        """
        self.planner.nav_mode = RetaskNavMode.GNSS_DENIED
        pre_retask_eta_s = 1500.0
        self.planner._eta_s = pre_retask_eta_s

        def _fail_replan(*args, **kwargs) -> ReplanResult:
            return ReplanResult(
                replan_id            = "FAIL-W28",
                trigger              = "test",
                mission_time_s       = 0.0,
                wall_latency_ms      = 0.5,
                success              = False,
                waypoints            = [],
                original_waypoints   = [],
                max_east_deviation_m = 0.0,
                kpi_ew02_pass        = True,
                nodes_explored       = 0,
            )

        with patch.object(self.planner._astar, "replan", side_effect=_fail_replan):
            result = self.planner.retask(
                new_goal_north_m = _GOAL_NORTH,
                new_goal_east_m  = _GOAL_EAST,
                trigger          = "IT-PLN-02-ROLLBACK",
            )

        # (a)
        self.assertFalse(result,
            "Retask must return False on dead-end in GNSS_DENIED mode")

        # (b)
        self.assertAlmostEqual(
            self.planner._eta_s, pre_retask_eta_s, places=6,
            msg=f"_eta_s must be restored to {pre_retask_eta_s} s after GNSS_DENIED rollback",
        )

        # (c)
        rollback_events = _logged_events(self.event_log, "RETASK_ROLLBACK")
        self.assertEqual(len(rollback_events), 1,
            "RETASK_ROLLBACK must be logged exactly once on GNSS_DENIED retask failure")

        # (d)/(e)
        ev = rollback_events[0]
        self.assertAlmostEqual(
            ev["eta_s_restored"], pre_retask_eta_s, places=6,
            msg=f"eta_s_restored must equal pre-retask _eta_s ({pre_retask_eta_s} s)",
        )
        self.assertEqual(ev["req_id"],   "PLN-02")
        self.assertEqual(ev["severity"], "WARNING")

    # -----------------------------------------------------------------------
    # (3) test_ins_only_retask_rejected
    # -----------------------------------------------------------------------

    def test_ins_only_retask_rejected(self):
        """
        IT-PLN-02 R-05: INS_ONLY retask rejected when XTE exceeds corridor margin.

        Threshold = half_width (500 m default) - 100 m = 400 m.
        XTE = 600 m > 400 m → RETASK_NAV_CONFIDENCE_TOO_LOW, return False.

        Assertions:
          (a) retask() returns False
          (b) RETASK_NAV_CONFIDENCE_TOO_LOW logged exactly once
          (c) Payload: cross_track_error_m=600.0, threshold_m=400.0, nav_mode=INS_ONLY
          (d) RETASK_COMPLETE NOT logged
        """
        self.planner.nav_mode = RetaskNavMode.INS_ONLY

        result = self.planner.retask(
            new_goal_north_m    = _GOAL_NORTH,
            new_goal_east_m     = _GOAL_EAST,
            cross_track_error_m = 600.0,
        )

        # (a)
        self.assertFalse(result,
            "Retask in INS_ONLY mode with XTE 600 m > 400 m threshold must return False")

        # (b)
        events = _logged_events(self.event_log, "RETASK_NAV_CONFIDENCE_TOO_LOW")
        self.assertEqual(len(events), 1,
            "RETASK_NAV_CONFIDENCE_TOO_LOW must be logged exactly once on INS_ONLY rejection")

        # (c)
        ev = events[0]
        self.assertEqual(ev["cross_track_error_m"], 600.0)
        self.assertEqual(ev["threshold_m"],         400.0)
        self.assertEqual(ev["nav_mode"],             "INS_ONLY")

        # (d)
        self.assertFalse(
            _logged_events(self.event_log, "RETASK_COMPLETE"),
            "RETASK_COMPLETE must NOT be logged when INS_ONLY XTE rejection fires",
        )

    # -----------------------------------------------------------------------
    # (4) test_ins_only_retask_permitted
    # -----------------------------------------------------------------------

    def test_ins_only_retask_permitted(self):
        """
        IT-PLN-02 R-05: INS_ONLY retask permitted when XTE is below corridor margin.

        Threshold = 500 - 100 = 400 m.  XTE = 300 m < 400 m → retask proceeds.

        Assertions:
          (a) retask() returns True
          (b) RETASK_COMPLETE logged with req_id=PLN-02
          (c) RETASK_NAV_CONFIDENCE_TOO_LOW NOT logged
        """
        self.planner.nav_mode = RetaskNavMode.INS_ONLY

        result = self.planner.retask(
            new_goal_north_m    = _GOAL_NORTH,
            new_goal_east_m     = _GOAL_EAST,
            cruise_alt_m        = _CRUISE_ALT,
            cross_track_error_m = 300.0,
        )

        # (a)
        self.assertTrue(result,
            "Retask in INS_ONLY mode with XTE 300 m < 400 m threshold must return True")

        # (b)
        complete_events = _logged_events(self.event_log, "RETASK_COMPLETE")
        self.assertEqual(len(complete_events), 1,
            "RETASK_COMPLETE must be logged exactly once when INS_ONLY XTE is within threshold")
        self.assertEqual(complete_events[0]["req_id"], "PLN-02")

        # (c)
        self.assertFalse(
            _logged_events(self.event_log, "RETASK_NAV_CONFIDENCE_TOO_LOW"),
            "RETASK_NAV_CONFIDENCE_TOO_LOW must NOT be logged when XTE is below threshold",
        )


# ---------------------------------------------------------------------------
# IT-ROLLBACK-01 — TERRAIN_GEN_FAIL, COMMIT_FAIL, RETASK_ROLLBACK payload
# ---------------------------------------------------------------------------

class TestITRollback01(unittest.TestCase):
    """
    IT-ROLLBACK-01: Three rollback trigger paths and full RETASK_ROLLBACK payload.

    (1) TERRAIN_GEN_FAIL → rollback: _terrain_regen_fn raises → RETASK_TERRAIN_GEN_FAILED
        + RETASK_ROLLBACK with reason=TERRAIN_GEN_FAIL + ETA restored.
    (2) COMMIT_FAIL → rollback: _px4_upload_fn raises after route found →
        RETASK_COMMIT_FAILED + RETASK_ROLLBACK with reason=COMMIT_FAIL + ETA restored.
    (3) Payload completeness: all SRS-required RETASK_ROLLBACK fields present on
        timeout path.

    Requirements: PLN-02, App-B ROUTING, App-B COMMITTING, App-B ROLLBACK
    SRS ref:      §4.2 Appendix B
    Governance:   Code Governance Manual v3.4 §1.3, §1.4, §9.1
    """

    def setUp(self):
        self.engine = _make_engine()
        self.clock  = _make_clock(0.0)

    def tearDown(self):
        del self.engine, self.clock

    # -----------------------------------------------------------------------
    # (1) test_terrain_gen_fail_triggers_rollback
    # -----------------------------------------------------------------------

    def test_terrain_gen_fail_triggers_rollback(self):
        """
        _terrain_regen_fn raises RuntimeError → RETASK_TERRAIN_GEN_FAILED logged,
        RETASK_ROLLBACK logged with reason=TERRAIN_GEN_FAIL, ETA restored.

        Assertions:
          (a) retask() returns False
          (b) RETASK_TERRAIN_GEN_FAILED in log
          (c) RETASK_ROLLBACK in log
          (d) rollback_event["reason"] == "TERRAIN_GEN_FAIL"
          (e) rollback_event["eta_s_restored"] == 1500.0
          (f) planner._eta_s == 1500.0 (ETA restored)
          (g) rollback_event["restored_terrain_phase"] is not None
        """
        def _fail_terrain():
            raise RuntimeError("terrain fail")

        event_log: List[Dict[str, Any]] = []
        planner = RoutePlanner(
            ew_engine        = self.engine,
            mission_clock    = self.clock,
            event_log        = event_log,
            terrain_regen_fn = _fail_terrain,
        )
        planner.load_route([_START_WP])
        planner.nav_mode = RetaskNavMode.CRUISE
        planner._eta_s   = 1500.0

        result = planner.retask(
            new_goal_north_m = _GOAL_NORTH,
            new_goal_east_m  = _GOAL_EAST,
        )

        # (a)
        self.assertFalse(result, "retask() must return False on terrain gen failure")

        # (b)
        terrain_events = _logged_events(event_log, "RETASK_TERRAIN_GEN_FAILED")
        self.assertEqual(len(terrain_events), 1,
            "RETASK_TERRAIN_GEN_FAILED must be logged exactly once")

        # (c)
        rollback_events = _logged_events(event_log, "RETASK_ROLLBACK")
        self.assertEqual(len(rollback_events), 1,
            "RETASK_ROLLBACK must be logged exactly once on terrain gen failure")

        # (d)
        ev = rollback_events[0]
        self.assertEqual(ev["reason"], "TERRAIN_GEN_FAIL",
            "RETASK_ROLLBACK reason must be TERRAIN_GEN_FAIL")

        # (e)
        self.assertAlmostEqual(ev["eta_s_restored"], 1500.0, places=6,
            msg="RETASK_ROLLBACK eta_s_restored must equal pre-retask ETA")

        # (f)
        self.assertAlmostEqual(planner._eta_s, 1500.0, places=6,
            msg="planner._eta_s must be restored to 1500.0 after rollback")

        # (g)
        self.assertIn("restored_terrain_phase", ev,
            "RETASK_ROLLBACK must contain restored_terrain_phase field")
        self.assertIsNotNone(ev["restored_terrain_phase"],
            "restored_terrain_phase must not be None")

    # -----------------------------------------------------------------------
    # (2) test_commit_fail_triggers_rollback
    # -----------------------------------------------------------------------

    def test_commit_fail_triggers_rollback(self):
        """
        _px4_upload_fn raises after route found → RETASK_COMMIT_FAILED logged,
        RETASK_ROLLBACK with reason=COMMIT_FAIL, ETA restored.

        Assertions:
          (a) retask() returns False
          (b) RETASK_COMMIT_FAILED in log
          (c) RETASK_ROLLBACK in log
          (d) rollback_event["reason"] == "COMMIT_FAIL"
          (e) rollback_event["eta_s_restored"] == 2000.0
          (f) planner._eta_s == 2000.0 (ETA restored)
          (g) RETASK_TERRAIN_GEN_FAILED NOT in log
        """
        def _fail_upload(wps):
            raise RuntimeError("upload fail")

        event_log: List[Dict[str, Any]] = []
        planner = RoutePlanner(
            ew_engine     = self.engine,
            mission_clock = self.clock,
            event_log     = event_log,
            px4_upload_fn = _fail_upload,
        )
        planner.load_route([_START_WP])
        planner.nav_mode = RetaskNavMode.CRUISE
        planner._eta_s   = 2000.0

        result = planner.retask(
            new_goal_north_m = _GOAL_NORTH,
            new_goal_east_m  = _GOAL_EAST,
            cruise_alt_m     = _CRUISE_ALT,
        )

        # (a)
        self.assertFalse(result, "retask() must return False on commit failure")

        # (b)
        commit_events = _logged_events(event_log, "RETASK_COMMIT_FAILED")
        self.assertEqual(len(commit_events), 1,
            "RETASK_COMMIT_FAILED must be logged exactly once")

        # (c)
        rollback_events = _logged_events(event_log, "RETASK_ROLLBACK")
        self.assertEqual(len(rollback_events), 1,
            "RETASK_ROLLBACK must be logged exactly once on commit failure")

        # (d)
        ev = rollback_events[0]
        self.assertEqual(ev["reason"], "COMMIT_FAIL",
            "RETASK_ROLLBACK reason must be COMMIT_FAIL")

        # (e)
        self.assertAlmostEqual(ev["eta_s_restored"], 2000.0, places=6,
            msg="RETASK_ROLLBACK eta_s_restored must equal pre-retask ETA")

        # (f)
        self.assertAlmostEqual(planner._eta_s, 2000.0, places=6,
            msg="planner._eta_s must be restored to 2000.0 after rollback")

        # (g)
        self.assertFalse(
            _logged_events(event_log, "RETASK_TERRAIN_GEN_FAILED"),
            "RETASK_TERRAIN_GEN_FAILED must NOT be logged — terrain regen succeeded",
        )

    # -----------------------------------------------------------------------
    # (3) test_rollback_payload_complete
    # -----------------------------------------------------------------------

    def test_rollback_payload_complete(self):
        """
        Timeout path rollback contains all SRS-required RETASK_ROLLBACK fields.

        Triggers timeout via mock clock advancing past RETASK_TIMEOUT_S.

        Assertions (all five SRS-required payload fields):
          (a) "reason" in rollback_event
          (b) "eta_s_restored" in rollback_event
          (c) "restored_terrain_phase" in rollback_event
          (d) "restored_ew_map_age_ms" in rollback_event
          (e) "previous_target" in rollback_event (may be None — field must exist)
        """
        # Clock: calls 1–5 return 0.0, call 6 returns 20.0 → timeout fires
        # (call 2 is new snap_ew_age_ms computation — IT-ROLLBACK-01)
        clock = MagicMock()
        clock.now.side_effect = [0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0]

        event_log: List[Dict[str, Any]] = []
        planner = RoutePlanner(
            ew_engine     = self.engine,
            mission_clock = clock,
            event_log     = event_log,
        )
        planner.load_route([_START_WP])
        planner.nav_mode = RetaskNavMode.CRUISE

        planner.retask(
            new_goal_north_m = _GOAL_NORTH,
            new_goal_east_m  = _GOAL_EAST,
        )

        rollback_events = _logged_events(event_log, "RETASK_ROLLBACK")
        self.assertEqual(len(rollback_events), 1,
            "RETASK_ROLLBACK must be logged exactly once on timeout")
        ev = rollback_events[0]

        # (a)
        self.assertIn("reason", ev,
            "RETASK_ROLLBACK must contain 'reason' field")
        # (b)
        self.assertIn("eta_s_restored", ev,
            "RETASK_ROLLBACK must contain 'eta_s_restored' field")
        # (c)
        self.assertIn("restored_terrain_phase", ev,
            "RETASK_ROLLBACK must contain 'restored_terrain_phase' field")
        # (d)
        self.assertIn("restored_ew_map_age_ms", ev,
            "RETASK_ROLLBACK must contain 'restored_ew_map_age_ms' field")
        # (e)
        self.assertIn("previous_target", ev,
            "RETASK_ROLLBACK must contain 'previous_target' field (may be None)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
