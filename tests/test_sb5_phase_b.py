"""
tests/test_sb5_phase_b.py
MicroMind / NanoCorteX — SB-5 Phase B Gates

SB-01  test_sb01_retask_rejected_in_ins_only       UT-PLN-02 nav mode rejection
SB-02  test_sb02_retask_succeeds_gnss_denied       IT-PLN-01 retask under GNSS_DENIED
SB-03  test_sb03_rollback_restores_ew_and_terrain  R-03 rollback completeness
SB-04  test_sb04_retask_timeout_triggers_rollback  R-06 timeout + rollback
SB-05  test_sb05_dead_end_returns_to_last_waypoint PLN-03 dead-end recovery

Requirements: PLN-02, PLN-03, EC-04
SRS ref:      §5.2, IT-PLN-01, IT-PLN-02, UT-PLN-02, GAP-02, GAP-03
Governance:   Code Governance Manual v3.2 §1.3, §1.4, §9.1

All five gates are independent; setUp/tearDown isolate state between tests.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np

from core.ew_engine.ew_engine import EWEngine
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
        Retask in INS_ONLY mode: returns False, RETASK_REJECTED_INS_ONLY logged
        with correct req_id, severity, and module_name.
        """
        self.planner.nav_mode = RetaskNavMode.INS_ONLY
        result = self.planner.retask(
            new_goal_north_m = _GOAL_NORTH,
            new_goal_east_m  = _GOAL_EAST,
        )
        self.assertFalse(result, "Retask in INS_ONLY mode must return False")

        events = _logged_events(self.event_log, "RETASK_REJECTED_INS_ONLY")
        self.assertEqual(len(events), 1, "RETASK_REJECTED_INS_ONLY must be logged once")

        ev = events[0]
        self.assertEqual(ev["req_id"],      "PLN-02")
        self.assertEqual(ev["severity"],    "WARNING")
        self.assertEqual(ev["module_name"], "RoutePlanner")
        self.assertIn("timestamp_ms", ev)


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

        # Mock clock: early calls return 0.0; call 5 returns 20.0 (timeout fires).
        # Call sequence inside retask():
        #   call 1: now_s = clock.now()               (initial ts, staleness check)
        #   call 2: clock.now() for RETASK_TERRAIN_FIRST log timestamp
        #   call 3: clock.now() after ew_refresh (mark EW updated)
        #   call 4: retask_start_s = clock.now()
        #   call 5: elapsed = clock.now() - retask_start_s → 20.0 - 0.0 > 15 s → TIMEOUT
        #   call 6: final_ts_ms = clock.now()  (for RETASK_TIMEOUT_ROLLBACK log)
        self.clock = MagicMock()
        self.clock.now.side_effect = [0.0, 0.0, 0.0, 0.0, 20.0, 20.0]

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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
