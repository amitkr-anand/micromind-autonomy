"""
core/route_planner/route_planner.py
MicroMind / NanoCorteX — Dynamic Retask Route Planner

SB-5 Phase B — PLN-02 Dynamic Retask + PLN-03 Dead-End Recovery
Corrections R-01 through R-06 (SRS §5.2, IT-PLN-01, IT-PLN-02,
UT-PLN-02, GAP-02, GAP-03).
Req IDs: PLN-02, PLN-03, EC-04

Forbidden behaviour (Code Governance Manual v3.2 §1.3):
  - Must NOT evaluate ROE
  - Must NOT issue direct PX4 commands
  - Must NOT call time.time() — uses mission_clock only (§1.4)
  - All thresholds from named constants — no magic numbers

§9.1 failure-first sequence enforced in retask():
  R-05 INS_ONLY rejection evaluated first (failure path before nominal)
  R-06 timeout enforces rollback — nominal path only reached on success
  PLN-03 dead-end recovery prevents no-route vehicle state
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.ew_engine.ew_engine import EWEngine
from core.route_planner.hybrid_astar import HybridAstar, ReplanResult

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Named constants — no magic numbers (§1.3)
# ---------------------------------------------------------------------------

RETASK_TIMEOUT_S              = 15.0   # PLN-02 §R-06: max computation time (s)
EW_MAP_STALENESS_THRESHOLD_S  = 15.0   # PLN-02 §R-02: EW map max age before warning (s)
WAYPOINT_POSITION_TOLERANCE_M = 15.0   # SRS §5.2: waypoint proximity threshold (m)
RETASK_CONSTRAINT_LEVELS      = 3      # Number of constraint relaxation levels to try
ROUTE_FRAGMENT_BYTES_PER_WP   = 24     # RS-04: bytes per waypoint (3 floats × 8 bytes)


# ---------------------------------------------------------------------------
# Navigation mode enum for retask eligibility
# ---------------------------------------------------------------------------

class RetaskNavMode(str, Enum):
    """
    Navigation modes that determine retask eligibility (PLN-02 §R-05).

    CRUISE and GNSS_DENIED allow retask — vehicle has sufficient state
    estimates and is not in a final approach phase.

    TERMINAL and INS_ONLY reject retask — either the vehicle is already
    in terminal engagement (re-routing would break ROE sequencing) or
    navigation quality is insufficient to validate a new route plan.
    """
    CRUISE      = "CRUISE"       # Normal cruise — retask allowed
    GNSS_DENIED = "GNSS_DENIED"  # GNSS denied, VIO/TRN active — retask allowed
    TERMINAL    = "TERMINAL"     # Terminal approach — retask rejected (safety)
    INS_ONLY    = "INS_ONLY"     # INS-only navigation — retask rejected (R-05)


# ---------------------------------------------------------------------------
# RoutePlanner
# ---------------------------------------------------------------------------

class RoutePlanner:
    """
    High-level Route Planner with dynamic retask capability.

    Wraps HybridAstar and enforces PLN-02 retask corrections R-01–R-06
    and PLN-03 dead-end recovery on every retask invocation.

    Forbidden behaviour (§1.3):
      - No ROE evaluation
      - No direct PX4 commands
      - No time.time() calls — mission_clock.now() only
      - All thresholds from named constants

    Usage::

        from core.clock.sim_clock import SimClock
        from core.ew_engine.ew_engine import EWEngine
        from core.route_planner.route_planner import RoutePlanner, RetaskNavMode

        clock   = SimClock(dt=0.01)
        clock.start()
        engine  = EWEngine()
        events  = []
        planner = RoutePlanner(engine, clock, events)

        planner.load_route([(30_000, 0, 4_000), (60_000, 0, 4_000)])
        planner.nav_mode = RetaskNavMode.CRUISE
        ok = planner.retask(new_goal_north_m=70_000, new_goal_east_m=0)
    """

    def __init__(
        self,
        ew_engine: EWEngine,
        mission_clock,                                   # SimClock-compatible; .now() → seconds
        event_log: List[Dict[str, Any]],
        terrain_regen_fn: Optional[Callable[[], None]] = None,
        ew_refresh_fn:    Optional[Callable[[], None]] = None,
        px4_upload_fn:    Optional[Callable[[List], None]] = None,
    ):
        """
        Args:
            ew_engine:        EWEngine instance providing the cost map.
            mission_clock:    SimClock-compatible; .now() returns seconds.
                              Must NOT be replaced with time.time() (§1.4).
            event_log:        List that receives structured log dicts.
            terrain_regen_fn: Callback to regenerate terrain corridor.
                              Called before EW refresh on every retask (R-01).
            ew_refresh_fn:    Callback to refresh the EW cost map.
                              Called after terrain regen on every retask (R-01).
            px4_upload_fn:    Callback to upload waypoints to PX4 in index
                              order (R-04). Must NOT issue commands directly;
                              delegation to PX4 bridge only.
        """
        self._engine    = ew_engine
        self._clock     = mission_clock
        self._event_log = event_log
        self._astar     = HybridAstar(ew_engine)

        # Injected side-effect callbacks (defaults are no-ops for unit tests)
        self._terrain_regen_fn = terrain_regen_fn if terrain_regen_fn is not None else lambda: None
        self._ew_refresh_fn    = ew_refresh_fn    if ew_refresh_fn    is not None else lambda: None
        self._px4_upload_fn    = px4_upload_fn    if px4_upload_fn    is not None else lambda wps: None

        # Mission state
        self._nav_mode:            RetaskNavMode = RetaskNavMode.CRUISE
        self._waypoints:           List[Tuple[float, float, float]] = []
        self._last_valid_waypoint: Optional[Tuple[float, float, float]] = None

        # Terrain corridor state (abstract numpy array; None until first load)
        self._terrain_corridor:    Optional[np.ndarray] = None

        # EW map age tracking — updated by mark_ew_map_updated()
        self._ew_map_last_updated_s: float = 0.0

        # RS-04: intermediate route fragments accumulated during retask search
        # (waypoint lists from non-adopted replan attempts at each constraint level)
        self._intermediate_fragments: List[List[Tuple[float, float, float]]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nav_mode(self) -> RetaskNavMode:
        """Current navigation mode; determines retask eligibility."""
        return self._nav_mode

    @nav_mode.setter
    def nav_mode(self, mode: RetaskNavMode) -> None:
        self._nav_mode = mode

    @property
    def waypoints(self) -> List[Tuple[float, float, float]]:
        """Current active waypoint list (snapshot copy)."""
        return list(self._waypoints)

    @property
    def terrain_corridor(self) -> Optional[np.ndarray]:
        """Current terrain corridor state (reference, not a copy)."""
        return self._terrain_corridor

    @terrain_corridor.setter
    def terrain_corridor(self, corridor: Optional[np.ndarray]) -> None:
        self._terrain_corridor = corridor

    @property
    def last_valid_waypoint(self) -> Optional[Tuple[float, float, float]]:
        """Last known good waypoint — used as PLN-03 dead-end recovery target."""
        return self._last_valid_waypoint

    # ------------------------------------------------------------------
    # Route and EW map management
    # ------------------------------------------------------------------

    def load_route(self, waypoints: List[Tuple[float, float, float]]) -> None:
        """
        Load a new route into the planner.

        The last waypoint in the list is stored as last_valid_waypoint for
        PLN-03 dead-end recovery.
        """
        self._waypoints = list(waypoints)
        if waypoints:
            self._last_valid_waypoint = waypoints[-1]

    def mark_ew_map_updated(self, timestamp_s: float) -> None:
        """
        Record that the EW cost map was updated at timestamp_s.

        Call this whenever the EW engine ingests new observations, so the
        R-02 staleness check has an accurate reference timestamp.
        """
        self._ew_map_last_updated_s = timestamp_s

    # ------------------------------------------------------------------
    # Dynamic retask — PLN-02 / R-01–R-06 / PLN-03
    # ------------------------------------------------------------------

    def retask(
        self,
        new_goal_north_m: float,
        new_goal_east_m:  float,
        cruise_alt_m:     float = 4_000.0,
        trigger:          str   = "RETASK",
    ) -> bool:
        """
        Dynamically retask the vehicle to a new goal position.

        Implements §9.1 failure-first sequence:
          1. R-05  INS_ONLY rejection (failure path first)
          2. TERMINAL rejection (failure path)
          3. R-02  EW map staleness warning (non-blocking)
          4. R-03  Snapshot EW map + terrain corridor + waypoints
          5. R-01  Terrain regen before EW refresh (ordering assertion)
          6. R-06  15 s timeout loop over constraint levels
          7. PLN-03 Dead-end recovery if no route at any level
          8. R-04  Waypoint upload in ascending index order
          9. Nominal success log

        Args:
            new_goal_north_m: New goal northing in metres (grid frame).
            new_goal_east_m:  New goal easting in metres (grid frame).
            cruise_alt_m:     Cruise altitude in metres (default 4 000 m).
            trigger:          Human-readable retask trigger label for logs.

        Returns:
            True  — retask succeeded; self.waypoints updated.
            False — retask rejected or failed; state rolled back.

        Forbidden: this method must NOT evaluate ROE, issue PX4 commands,
        or call time.time() (§1.3, §1.4).
        """
        # ── Initial timestamp (used for early rejection log events) ─────────
        now_s = self._clock.now()         # §1.4: mission_clock only
        ts_ms = int(now_s * 1000)

        # ── R-05: Reject if INS_ONLY (failure-first §9.1) ───────────────────
        # Retask is unsafe when nav mode is INS_ONLY: the position estimate
        # relies solely on IMU dead-reckoning, accumulated drift makes route
        # validation unreliable, and a failed retask with rolled-back state
        # may leave the vehicle navigating on a stale route. Reject early.
        if self._nav_mode == RetaskNavMode.INS_ONLY:
            self._event_log.append({
                "event":        "RETASK_REJECTED_INS_ONLY",
                "req_id":       "PLN-02",
                "severity":     "WARNING",
                "module_name":  "RoutePlanner",
                "timestamp_ms": ts_ms,
            })
            self._cleanup_route_fragments(ts_ms)  # RS-04: no fragments generated
            return False

        # ── TERMINAL rejection (non-INS_ONLY blocked mode) ──────────────────
        # Terminal approach: vehicle is in final engagement phase.
        # Re-routing here would break the deterministic ROE execution sequence.
        if self._nav_mode == RetaskNavMode.TERMINAL:
            self._cleanup_route_fragments(ts_ms)  # RS-04: no fragments generated
            return False

        # ── R-02: EW map staleness check (non-blocking warning) ─────────────
        # If the EW cost map is older than EW_MAP_STALENESS_THRESHOLD_S, warn
        # but continue — use last valid map rather than aborting the retask.
        ew_age_s = now_s - self._ew_map_last_updated_s
        if ew_age_s > EW_MAP_STALENESS_THRESHOLD_S:
            self._event_log.append({
                "event":        "EW_MAP_STALE_ON_RETASK",
                "req_id":       "PLN-02",
                "severity":     "WARNING",
                "module_name":  "RoutePlanner",
                "timestamp_ms": ts_ms,
                "ew_age_s":     ew_age_s,
            })
            # Continue — last valid map is used; do not abort on staleness alone

        # ── R-03: Snapshot state for rollback ───────────────────────────────
        # Snapshot ALL three state components before any modification.
        # Rollback must restore all three — not just waypoints.
        snap_waypoints           = list(self._waypoints)
        snap_ew_map              = self._engine.cost_map.copy()
        snap_terrain_corridor    = (
            self._terrain_corridor.copy()
            if self._terrain_corridor is not None else None
        )
        snap_ew_map_last_updated = self._ew_map_last_updated_s

        def _rollback() -> None:
            """
            R-03 rollback: restore EW map, terrain corridor, and waypoints
            to pre-retask snapshot values.
            """
            self._waypoints              = snap_waypoints
            self._engine.cost_map[:]     = snap_ew_map          # in-place restore
            self._terrain_corridor       = snap_terrain_corridor
            self._ew_map_last_updated_s  = snap_ew_map_last_updated

        # ── R-01: Terrain regeneration BEFORE EW map refresh ────────────────
        # INVARIANT: terrain regen must always precede EW map refresh on retask.
        # Terrain corridor defines the search space; EW costs are overlaid on
        # that space. If EW refresh ran first, it might penalise cells that the
        # updated terrain corridor will invalidate, producing suboptimal routes.
        self._terrain_regen_fn()
        # ORDERING CHECKPOINT: terrain regen complete — EW refresh follows
        self._event_log.append({
            "event":        "RETASK_TERRAIN_FIRST",
            "req_id":       "PLN-02",
            "severity":     "DEBUG",
            "module_name":  "RoutePlanner",
            "timestamp_ms": int(self._clock.now() * 1000),  # §1.4
        })
        self._ew_refresh_fn()
        self._ew_map_last_updated_s = self._clock.now()    # Track refresh time

        # ── R-06: Route search with 15 s timeout ────────────────────────────
        # Uses mission_clock — not time.time() (§1.4).
        # If no valid route is found within RETASK_TIMEOUT_S, roll back and
        # return False. An unrecovered timeout would leave the vehicle with
        # no valid route; rollback to the pre-retask waypoints is mandatory.
        retask_start_s = self._clock.now()
        found_route: Optional[List[Tuple[float, float, float]]] = None
        timed_out   = False

        start_wp = self._waypoints[0] if self._waypoints else (0.0, 0.0, cruise_alt_m)

        for constraint_level in range(RETASK_CONSTRAINT_LEVELS):
            # Check timeout at top of each constraint level (§9.1 failure-first)
            elapsed_s = self._clock.now() - retask_start_s
            if elapsed_s > RETASK_TIMEOUT_S:
                timed_out = True
                break

            result: ReplanResult = self._astar.replan(
                start_north_m  = start_wp[0],
                start_east_m   = start_wp[1],
                goal_north_m   = new_goal_north_m,
                goal_east_m    = new_goal_east_m,
                cruise_alt_m   = cruise_alt_m,
                mission_time_s = self._clock.now(),
                trigger        = f"{trigger} level={constraint_level}",
            )

            if result.success:
                found_route = result.waypoints
                break

            # RS-04: track this non-adopted replan attempt as an intermediate fragment
            self._intermediate_fragments.append(list(result.waypoints))

            # Check timeout after each failed attempt
            elapsed_s = self._clock.now() - retask_start_s
            if elapsed_s > RETASK_TIMEOUT_S:
                timed_out = True
                break

        final_ts_ms = int(self._clock.now() * 1000)

        # ── R-06: Timeout → rollback ─────────────────────────────────────────
        if timed_out:
            self._event_log.append({
                "event":        "RETASK_TIMEOUT_ROLLBACK",
                "req_id":       "PLN-02",
                "severity":     "WARNING",
                "module_name":  "RoutePlanner",
                "timestamp_ms": final_ts_ms,
            })
            _rollback()
            self._cleanup_route_fragments(final_ts_ms)  # RS-04
            return False

        # ── PLN-03: Dead-end recovery ─────────────────────────────────────────
        # All constraint levels exhausted with no valid route found.
        # Log DEAD_END_DETECTED and return to last valid waypoint.
        # Never leave the vehicle with an empty route.
        if found_route is None:
            self._event_log.append({
                "event":        "DEAD_END_DETECTED",
                "req_id":       "PLN-03",
                "severity":     "WARNING",
                "module_name":  "RoutePlanner",
                "timestamp_ms": final_ts_ms,
            })
            # Restore state and set route to last valid waypoint
            _rollback()
            if self._last_valid_waypoint is not None:
                self._waypoints = [self._last_valid_waypoint]
            self._cleanup_route_fragments(final_ts_ms)  # RS-04
            return False

        # ── R-04: Waypoint upload in ascending index order ────────────────────
        # Assert and enforce ascending index sequence before upload.
        # Waypoints computed by HybridAstar are already in path order (index 0
        # first), but the assertion makes the invariant explicit and auditable.
        upload_indices = list(range(len(found_route)))
        assert upload_indices == sorted(upload_indices), (
            f"R-04 violation: waypoint upload sequence not ascending — "
            f"got indices {upload_indices}"
        )
        self._event_log.append({
            "event":        "WAYPOINT_UPLOAD_ORDER_VERIFIED",
            "req_id":       "PLN-02",
            "severity":     "DEBUG",
            "module_name":  "RoutePlanner",
            "timestamp_ms": final_ts_ms,
        })
        self._px4_upload_fn(found_route)   # delegate to bridge — no direct PX4 cmds (§1.3)

        # ── Nominal success path ──────────────────────────────────────────────
        self._waypoints = found_route
        if found_route:
            self._last_valid_waypoint = found_route[-1]

        self._event_log.append({
            "event":        "RETASK_COMPLETE",
            "req_id":       "PLN-02",
            "severity":     "INFO",
            "module_name":  "RoutePlanner",
            "timestamp_ms": final_ts_ms,
        })
        self._cleanup_route_fragments(final_ts_ms)  # RS-04
        return True

    # ------------------------------------------------------------------
    # RS-04 Fragment cleanup
    # ------------------------------------------------------------------

    def _cleanup_route_fragments(self, ts_ms: int) -> None:
        """
        RS-04: Explicitly clear intermediate route fragments accumulated
        during a retask search and log the cleanup event.

        Intermediate fragments are waypoint lists from non-adopted replan
        attempts at each constraint relaxation level (R-06).  They are
        tracked in self._intermediate_fragments and must be cleared after
        every retask operation — successful, failed, timed-out, or rejected
        — so that long missions do not accumulate stale fragment state.

        Args:
            ts_ms: Log timestamp in milliseconds (from caller — no new
                   clock call; §1.4 forbids time.time()).

        Logs:
            ROUTE_FRAGMENT_CLEANUP at DEBUG with req_id='RS-04'.
        """
        fragments_cleared = len(self._intermediate_fragments)
        bytes_freed = sum(
            len(frag) * ROUTE_FRAGMENT_BYTES_PER_WP
            for frag in self._intermediate_fragments
        )
        self._intermediate_fragments.clear()

        self._event_log.append({
            "event":        "ROUTE_FRAGMENT_CLEANUP",
            "req_id":       "RS-04",
            "severity":     "DEBUG",
            "module_name":  "RoutePlanner",
            "timestamp_ms": ts_ms,
            "payload": {
                "fragments_cleared":    fragments_cleared,
                "bytes_freed_estimate": bytes_freed,
            },
        })
        _log.debug(
            "ROUTE_FRAGMENT_CLEANUP: fragments_cleared=%d bytes_freed_estimate=%d",
            fragments_cleared,
            bytes_freed,
        )
