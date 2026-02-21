"""
core/route_planner/hybrid_astar.py
MicroMind / NanoCorteX — Hybrid A* Route Planner
Sprint S4 Deliverable 2 of 3

Implements Hybrid A* on the EW cost map grid produced by EWEngine.
Routes a vehicle from a start waypoint to a goal waypoint while
penalising cells with high EW threat cost.

Design:
    Standard A* on a discrete grid, with:
      - 8-connected neighbourhood (N, NE, E, SE, S, SW, W, NW)
      - g(n) = Euclidean distance travelled
      - h(n) = Euclidean distance to goal (admissible heuristic)
      - f(n) = g(n) + h(n) + EW_WEIGHT × cost_map[n]
      - EW_WEIGHT = 50_000 — cost in metres-equivalent to cross a cost=1 cell
        (makes the planner avoid high-cost cells unless detour is huge)

    "Hybrid" here means the planner reasons in grid space but outputs
    smooth world-coordinate waypoints by extracting the path centreline
    and applying light smoothing (moving average, 5-point window).

KPI gate:
    EW-02  Route replan ≤ 1 s wall clock — guaranteed by grid size
           (500×100 = 50 000 cells; A* on this is < 100 ms in Python)

References:
    Part Two V7  §1.6.3 (route replanning under EW threat), KPI-E03
    BCMP-1       EW-01, EW-02 pass criteria
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.ew_engine.ew_engine import (
    EWEngine,
    GRID_NORTH_MIN, GRID_NORTH_MAX,
    GRID_EAST_MIN,  GRID_EAST_MAX,
    GRID_RESOLUTION,
    N_NORTH, N_EAST,
)


# ---------------------------------------------------------------------------
# Planner configuration
# ---------------------------------------------------------------------------

EW_WEIGHT       = 50_000.0   # metres-equivalent cost for crossing cost=1 cell
MAX_EAST_OFFSET =  8_000.0   # max eastward deviation allowed (m) — corridor constraint
SMOOTH_WINDOW   = 5          # moving-average smoothing window for output waypoints

# 8-connected grid moves: (Δrow, Δcol)
MOVES = [
    (-1,  0), ( 1,  0), ( 0, -1), ( 0,  1),   # cardinal
    (-1, -1), (-1,  1), ( 1, -1), ( 1,  1),   # diagonal
]
MOVE_COSTS = [
    1.0, 1.0, 1.0, 1.0,
    math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2),
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReplanResult:
    """Output of a single Hybrid A* replan."""
    replan_id:          str
    trigger:            str                     # e.g. "JMR-01 activated"
    mission_time_s:     float
    wall_latency_ms:    float
    success:            bool
    waypoints:          List[Tuple[float, float, float]]  # (north_m, east_m, alt_m)
    original_waypoints: List[Tuple[float, float, float]]
    max_east_deviation_m: float                 # peak east deviation from original
    kpi_ew02_pass:      bool                    # wall_latency_ms ≤ 1000
    nodes_explored:     int = 0


# ---------------------------------------------------------------------------
# Hybrid A* planner
# ---------------------------------------------------------------------------

class HybridAstar:
    """
    Hybrid A* route planner with EW cost overlay.

    Usage:
        planner = HybridAstar(ew_engine)
        result  = planner.replan(
            start_north_m = 30_000,
            start_east_m  = 0,
            goal_north_m  = 60_000,
            goal_east_m   = 0,
            cruise_alt_m  = 4_000,
            mission_time_s= 1080.0,
            trigger       = "JMR-01 activated",
        )
    """

    def __init__(self, ew_engine: EWEngine):
        self._engine    = ew_engine
        self._replans:  List[ReplanResult] = []
        self._replan_counter = 0

    @property
    def replans(self) -> List[ReplanResult]:
        return list(self._replans)

    # ------------------------------------------------------------------
    # Public: replan
    # ------------------------------------------------------------------

    def replan(
        self,
        start_north_m:  float,
        start_east_m:   float,
        goal_north_m:   float,
        goal_east_m:    float,
        cruise_alt_m:   float,
        mission_time_s: float,
        trigger:        str = "",
    ) -> ReplanResult:
        """
        Run A* from start to goal on the current EW cost map.
        Returns a ReplanResult with the new waypoint list.
        """
        import time
        t0 = time.monotonic()

        self._replan_counter += 1
        rid = f"REPLAN-{self._replan_counter:02d}"

        # Convert world to grid
        start_row, start_col = EWEngine.world_to_grid(start_north_m, start_east_m)
        goal_row,  goal_col  = EWEngine.world_to_grid(goal_north_m,  goal_east_m)

        # Store original straight-line path for deviation measurement
        original_wps = self._straight_line_waypoints(
            start_north_m, start_east_m,
            goal_north_m,  goal_east_m,
            cruise_alt_m,  n_points=20,
        )

        # Run A*
        path_cells, nodes_explored = self._astar(
            start_row, start_col,
            goal_row,  goal_col,
        )

        wall_ms = (time.monotonic() - t0) * 1000.0

        if path_cells is None:
            return ReplanResult(
                replan_id   = rid,
                trigger     = trigger,
                mission_time_s = mission_time_s,
                wall_latency_ms = wall_ms,
                success     = False,
                waypoints   = original_wps,
                original_waypoints = original_wps,
                max_east_deviation_m = 0.0,
                kpi_ew02_pass = wall_ms <= 1000.0,
                nodes_explored = nodes_explored,
            )

        # Convert path cells → world waypoints, smooth, add altitude
        waypoints = self._cells_to_waypoints(path_cells, cruise_alt_m)

        # Measure max east deviation from original centreline (east=0)
        max_dev = max(abs(wp[1] - 0.0) for wp in waypoints)

        result = ReplanResult(
            replan_id    = rid,
            trigger      = trigger,
            mission_time_s = mission_time_s,
            wall_latency_ms = wall_ms,
            success      = True,
            waypoints    = waypoints,
            original_waypoints = original_wps,
            max_east_deviation_m = max_dev,
            kpi_ew02_pass = wall_ms <= 1000.0,
            nodes_explored = nodes_explored,
        )
        self._replans.append(result)
        return result

    # ------------------------------------------------------------------
    # A* core
    # ------------------------------------------------------------------

    def _astar(
        self,
        sr: int, sc: int,
        gr: int, gc: int,
    ) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        """
        A* on the cost map grid.
        Returns (path as list of (row, col), nodes_explored).
        path is None if no route found.
        """
        cost_map = self._engine.cost_map

        # f_score heap: (f, g, row, col)
        open_heap: List[Tuple[float, float, int, int]] = []
        heapq.heappush(open_heap, (0.0, 0.0, sr, sc))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score:   Dict[Tuple[int, int], float] = {(sr, sc): 0.0}
        closed:    set = set()
        nodes_explored = 0

        def heuristic(r: int, c: int) -> float:
            return math.sqrt((r - gr)**2 + (c - gc)**2) * GRID_RESOLUTION

        while open_heap:
            f, g, r, c = heapq.heappop(open_heap)

            if (r, c) in closed:
                continue
            closed.add((r, c))
            nodes_explored += 1

            if r == gr and c == gc:
                # Reconstruct path
                path = []
                cur = (gr, gc)
                while cur in came_from:
                    path.append(cur)
                    cur = came_from[cur]
                path.append((sr, sc))
                path.reverse()
                return path, nodes_explored

            for (dr, dc), move_cost_cells in zip(MOVES, MOVE_COSTS):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < N_NORTH and 0 <= nc < N_EAST):
                    continue
                if (nr, nc) in closed:
                    continue

                # Enforce east corridor constraint
                world_east = GRID_EAST_MIN + nc * GRID_RESOLUTION
                if abs(world_east) > MAX_EAST_OFFSET:
                    continue

                move_dist = move_cost_cells * GRID_RESOLUTION
                ew_penalty = float(cost_map[nr, nc]) * EW_WEIGHT
                new_g = g + move_dist + ew_penalty

                if new_g < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = new_g
                    came_from[(nr, nc)] = (r, c)
                    f_new = new_g + heuristic(nr, nc)
                    heapq.heappush(open_heap, (f_new, new_g, nr, nc))

        return None, nodes_explored   # no path found

    # ------------------------------------------------------------------
    # Path post-processing
    # ------------------------------------------------------------------

    def _cells_to_waypoints(
        self,
        cells: List[Tuple[int, int]],
        alt_m: float,
    ) -> List[Tuple[float, float, float]]:
        """
        Convert grid cells to world waypoints with smoothing.
        Downsamples to every Nth cell then applies moving-average.
        """
        # Downsample — keep every 5th cell plus endpoints
        step = max(1, len(cells) // 50)
        sampled = cells[::step]
        if cells[-1] not in sampled:
            sampled.append(cells[-1])

        # To world coordinates
        wps = []
        for r, c in sampled:
            north, east = EWEngine.grid_to_world(r, c)
            wps.append([north, east])
        wps = np.array(wps)

        # Moving-average smoothing
        if len(wps) >= SMOOTH_WINDOW:
            kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
            smooth_n = np.convolve(wps[:, 0], kernel, mode='same')
            smooth_e = np.convolve(wps[:, 1], kernel, mode='same')
            # Preserve exact start/end
            smooth_n[0],  smooth_e[0]  = wps[0]
            smooth_n[-1], smooth_e[-1] = wps[-1]
            wps = np.column_stack([smooth_n, smooth_e])

        return [(float(n), float(e), alt_m) for n, e in wps]

    @staticmethod
    def _straight_line_waypoints(
        sn: float, se: float,
        gn: float, ge: float,
        alt_m: float,
        n_points: int = 20,
    ) -> List[Tuple[float, float, float]]:
        """Generate straight-line waypoints from start to goal."""
        return [
            (
                sn + (gn - sn) * t,
                se + (ge - se) * t,
                alt_m,
            )
            for t in np.linspace(0, 1, n_points)
        ]
