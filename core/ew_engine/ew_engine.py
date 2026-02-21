"""
core/ew_engine/ew_engine.py
MicroMind / NanoCorteX — EW Engine
Sprint S4 Deliverable 1 of 3

Responsibilities:
  1. Receive SDR observations (simulated signal-strength snapshots)
  2. Cluster observations into jammer hypotheses using DBSCAN
  3. Maintain a 2D cost map over the mission corridor
  4. Expose cost map to route planner (Hybrid A*)

KPI targets (Part Two V7):
  KPI-E01  Cost map update latency  < 500 ms from jammer activation
  NFR-003  SDR EW processing        < 500 ms snapshot → cost tile update
  NFR-005  Cost-map refresh rate    ≤ 500 ms
  NFR-006  Probability of detection  Pd ≥ 0.90
  NFR-007  Probability false alarm   Pfa ≤ 0.05

Signal model:
  confidence(r) = confidence_at_50m × (50 / r)²   clamped [0, 1]

DBSCAN (no sklearn — pure scipy + numpy):
  Uses cKDTree for ε-neighbourhood queries.
  Core point: ≥ min_samples neighbours within ε metres.
  Cluster centroid → jammer hypothesis position.

Cost map:
  2D numpy array  [N_north × N_east]
  Each cell = max EW threat cost in [0, 1]
  Updated in-place on each observation batch.
  Route planner reads cost map as navigational penalty.

References:
  Part Two V7  §1.6 (EW Engine), KPI-E01, KPI-E03, NFR-003/005/006/007
  BCMP-1       JMR-01 (T+18 min), JMR-02 (T+25 min)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Cost map grid parameters
# ---------------------------------------------------------------------------

# Corridor bounding box (ENU metres from mission origin)
GRID_NORTH_MIN =      0.0
GRID_NORTH_MAX = 100_000.0   # 100 km corridor
GRID_EAST_MIN  = -10_000.0   # ±10 km east of centreline
GRID_EAST_MAX  =  10_000.0
GRID_RESOLUTION =    200.0   # metres per cell

# Derived grid dimensions
N_NORTH = int((GRID_NORTH_MAX - GRID_NORTH_MIN) / GRID_RESOLUTION)  # 500
N_EAST  = int((GRID_EAST_MAX  - GRID_EAST_MIN)  / GRID_RESOLUTION)  # 100

# Cost decay — older observations lose weight
COST_DECAY_RATE   = 0.02    # cost units/second
COST_FLOOR        = 0.05    # minimum residual cost after decay

# DBSCAN parameters
DBSCAN_EPS_M      = 5_000.0  # cluster radius (m) — jammers are area emitters
DBSCAN_MIN_SAMPLES = 2        # min observations to form a cluster

# Jammer hypothesis thresholds (NFR-006/007)
HYPOTHESIS_CONFIDENCE_THRESHOLD = 0.50   # min confidence to raise hypothesis
DETECTION_CONFIDENCE_THRESHOLD  = 0.70   # Pd ≥ 0.90 target at this level


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EWObservation:
    """Single SDR signal-strength snapshot from the EW receiver."""
    timestamp_s:        float           # mission time
    bearing_deg:        float           # bearing to emitter (degrees from North)
    signal_strength_db: float           # received signal strength (dBm)
    estimated_range_m:  float           # range estimate from signal model
    position_enu:       np.ndarray      # observer position at time of obs (3,)
    frequency_mhz:      float = 1575.42 # L1 GPS band default


@dataclass
class JammerHypothesis:
    """
    Clustered jammer hypothesis produced by EW Engine.
    Confidence drives cost map intensity and FSM EW_AWARE transition guard.
    """
    hypothesis_id:  str
    position_enu:   np.ndarray      # estimated position (N, E, U) metres
    confidence:     float           # 0.0–1.0
    radius_m:       float           # estimated effective radius
    first_seen_s:   float           # mission time of first observation
    last_seen_s:    float           # mission time of most recent observation
    obs_count:      int = 1
    jammer_type:    str = "BROADBAND_GNSS"


@dataclass
class CostMapUpdate:
    """Record of a single cost map update — used for KPI-E01 latency measurement."""
    timestamp_s:        float       # mission time of triggering observation
    wall_latency_ms:    float       # actual compute time (ms)
    hypotheses_count:   int
    peak_cost:          float
    cells_updated:      int
    kpi_e01_pass:       bool        # wall_latency_ms < 500


# ---------------------------------------------------------------------------
# EW Engine
# ---------------------------------------------------------------------------

class EWEngine:
    """
    Electronic Warfare Engine — jammer detection, hypothesis management,
    and cost map maintenance.

    Usage:
        engine = EWEngine()

        # Each tick, feed observations:
        obs = EWObservation(...)
        update = engine.process_observations([obs], mission_time_s)

        # Route planner reads cost map:
        cost = engine.cost_map          # numpy array [N_NORTH × N_EAST]
        hyps = engine.hypotheses        # list[JammerHypothesis]
    """

    def __init__(self):
        # Cost map — initialised to zero (no known threats)
        self._cost_map: np.ndarray = np.zeros((N_NORTH, N_EAST), dtype=np.float32)

        # Observation buffer — retained for DBSCAN clustering
        self._observations: List[EWObservation] = []

        # Active jammer hypotheses
        self._hypotheses: Dict[str, JammerHypothesis] = {}
        self._hyp_counter = 0

        # Update history for KPI logging
        self._updates: List[CostMapUpdate] = []

        # Last update time (for decay computation)
        self._last_update_s: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def cost_map(self) -> np.ndarray:
        """Current 2D cost map [N_NORTH × N_EAST], values in [0, 1]."""
        return self._cost_map

    @property
    def hypotheses(self) -> List[JammerHypothesis]:
        return list(self._hypotheses.values())

    @property
    def updates(self) -> List[CostMapUpdate]:
        return list(self._updates)

    @property
    def active_hypothesis_count(self) -> int:
        return len(self._hypotheses)

    def process_observations(
        self,
        observations: List[EWObservation],
        mission_time_s: float,
    ) -> Optional[CostMapUpdate]:
        """
        Ingest a batch of SDR observations, run DBSCAN clustering,
        update hypotheses, and refresh cost map.

        Returns a CostMapUpdate record (always, even if no new hypotheses).
        KPI-E01 latency is measured as wall-clock time for this call.
        """
        t_wall_start = time.monotonic()

        if not observations:
            return None

        # 1. Add to buffer
        self._observations.extend(observations)

        # 2. Decay existing cost map
        self._decay_cost_map(mission_time_s)

        # 3. DBSCAN cluster all observations → hypotheses
        self._cluster_observations(mission_time_s)

        # 4. Paint hypotheses onto cost map
        cells_updated = self._paint_cost_map()

        # 5. Measure latency
        wall_ms = (time.monotonic() - t_wall_start) * 1000.0
        peak    = float(self._cost_map.max())

        update = CostMapUpdate(
            timestamp_s      = mission_time_s,
            wall_latency_ms  = wall_ms,
            hypotheses_count = len(self._hypotheses),
            peak_cost        = peak,
            cells_updated    = cells_updated,
            kpi_e01_pass     = wall_ms < 500.0,
        )
        self._updates.append(update)
        self._last_update_s = mission_time_s
        return update

    def signal_confidence(self, observer_pos: np.ndarray,
                          jammer_pos: np.ndarray,
                          confidence_at_50m: float) -> float:
        """
        Inverse-square signal model.
        confidence(r) = confidence_at_50m × (50/r)²  clamped [0, 1]
        """
        r = float(np.linalg.norm(observer_pos[:2] - jammer_pos[:2]))
        r = max(r, 1.0)   # prevent divide-by-zero at point-blank
        return min(1.0, confidence_at_50m * (50.0 / r) ** 2)

    # ------------------------------------------------------------------
    # Internal — DBSCAN clustering
    # ------------------------------------------------------------------

    def _cluster_observations(self, mission_time_s: float) -> None:
        """
        DBSCAN over estimated emitter positions derived from observations.
        Each observation contributes its estimated_range_m as a disc around
        the bearing line — simplified to a point estimate at range along bearing.
        """
        if len(self._observations) < DBSCAN_MIN_SAMPLES:
            return

        # Build point array: estimated emitter north/east positions
        points = []
        for obs in self._observations:
            bearing_rad = math.radians(obs.bearing_deg)
            est_north = obs.position_enu[0] + obs.estimated_range_m * math.cos(bearing_rad)
            est_east  = obs.position_enu[1] + obs.estimated_range_m * math.sin(bearing_rad)
            points.append([est_north, est_east])

        points_arr = np.array(points)

        # DBSCAN via cKDTree
        tree    = cKDTree(points_arr)
        labels  = np.full(len(points_arr), -1, dtype=int)   # -1 = noise
        cluster_id = 0

        visited = set()
        for i in range(len(points_arr)):
            if i in visited:
                continue
            visited.add(i)
            neighbours = tree.query_ball_point(points_arr[i], DBSCAN_EPS_M)
            if len(neighbours) < DBSCAN_MIN_SAMPLES:
                continue   # noise point
            # Expand cluster
            labels[i] = cluster_id
            queue = list(neighbours)
            while queue:
                j = queue.pop()
                if j not in visited:
                    visited.add(j)
                    nn = tree.query_ball_point(points_arr[j], DBSCAN_EPS_M)
                    if len(nn) >= DBSCAN_MIN_SAMPLES:
                        queue.extend(nn)
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1

        # Build / update hypotheses from clusters
        for cid in range(cluster_id):
            mask    = labels == cid
            cluster = points_arr[mask]
            centroid_n, centroid_e = cluster.mean(axis=0)
            centroid = np.array([centroid_n, centroid_e, 3500.0])   # assume plateau altitude

            # Confidence from cluster density
            obs_in_cluster = [self._observations[i] for i in range(len(points_arr)) if mask[i]]
            mean_confidence = np.mean([
                min(1.0, o.signal_strength_db / -60.0)   # −60 dBm = full confidence
                for o in obs_in_cluster
            ])
            mean_confidence = float(np.clip(mean_confidence, 0.0, 1.0))

            if mean_confidence < HYPOTHESIS_CONFIDENCE_THRESHOLD:
                continue

            # Radius = max distance from centroid to any cluster point
            dists  = np.linalg.norm(cluster - centroid[:2], axis=1)
            radius = max(float(dists.max()), DBSCAN_EPS_M)

            # Match to existing hypothesis or create new
            matched = False
            for hyp in self._hypotheses.values():
                if np.linalg.norm(hyp.position_enu[:2] - centroid[:2]) < DBSCAN_EPS_M:
                    # Update existing
                    hyp.position_enu = centroid
                    hyp.confidence   = max(hyp.confidence, mean_confidence)
                    hyp.radius_m     = radius
                    hyp.last_seen_s  = mission_time_s
                    hyp.obs_count   += len(obs_in_cluster)
                    matched = True
                    break

            if not matched:
                self._hyp_counter += 1
                hid = f"HYP-{self._hyp_counter:02d}"
                self._hypotheses[hid] = JammerHypothesis(
                    hypothesis_id = hid,
                    position_enu  = centroid,
                    confidence    = mean_confidence,
                    radius_m      = radius,
                    first_seen_s  = mission_time_s,
                    last_seen_s   = mission_time_s,
                    obs_count     = len(obs_in_cluster),
                )

    # ------------------------------------------------------------------
    # Internal — cost map
    # ------------------------------------------------------------------

    def _decay_cost_map(self, mission_time_s: float) -> None:
        """Decay all cost map cells by time elapsed since last update."""
        dt = mission_time_s - self._last_update_s
        if dt <= 0:
            return
        decay = COST_DECAY_RATE * dt
        self._cost_map = np.maximum(
            self._cost_map - decay, COST_FLOOR * (self._cost_map > 0)
        ).astype(np.float32)

    def _paint_cost_map(self) -> int:
        """
        For each active hypothesis, paint a Gaussian cost blob onto the map.
        Returns number of cells updated above COST_FLOOR.
        """
        cells_updated = 0

        for hyp in self._hypotheses.values():
            hyp_n = hyp.position_enu[0]
            hyp_e = hyp.position_enu[1]
            sigma = hyp.radius_m

            # Bounding box of influence (3σ)
            n_min = max(0, int((hyp_n - 3*sigma - GRID_NORTH_MIN) / GRID_RESOLUTION))
            n_max = min(N_NORTH, int((hyp_n + 3*sigma - GRID_NORTH_MIN) / GRID_RESOLUTION) + 1)
            e_min = max(0, int((hyp_e - 3*sigma - GRID_EAST_MIN)  / GRID_RESOLUTION))
            e_max = min(N_EAST,  int((hyp_e + 3*sigma - GRID_EAST_MIN)  / GRID_RESOLUTION) + 1)

            if n_min >= n_max or e_min >= e_max:
                continue

            # Vectorised Gaussian cost
            rows = np.arange(n_min, n_max)
            cols = np.arange(e_min, e_max)
            rr, cc = np.meshgrid(rows, cols, indexing='ij')

            cell_north = GRID_NORTH_MIN + rr * GRID_RESOLUTION
            cell_east  = GRID_EAST_MIN  + cc * GRID_RESOLUTION

            dist2 = (cell_north - hyp_n)**2 + (cell_east - hyp_e)**2
            cost  = hyp.confidence * np.exp(-dist2 / (2 * sigma**2))

            before = self._cost_map[n_min:n_max, e_min:e_max].copy()
            self._cost_map[n_min:n_max, e_min:e_max] = np.maximum(
                self._cost_map[n_min:n_max, e_min:e_max],
                cost.astype(np.float32)
            )
            cells_updated += int(np.sum(self._cost_map[n_min:n_max, e_min:e_max] > before))

        return cells_updated

    # ------------------------------------------------------------------
    # Grid coordinate utilities
    # ------------------------------------------------------------------

    @staticmethod
    def world_to_grid(north_m: float, east_m: float) -> Tuple[int, int]:
        """Convert world ENU (metres) to cost map grid indices (row, col)."""
        row = int((north_m - GRID_NORTH_MIN) / GRID_RESOLUTION)
        col = int((east_m  - GRID_EAST_MIN)  / GRID_RESOLUTION)
        row = max(0, min(N_NORTH - 1, row))
        col = max(0, min(N_EAST  - 1, col))
        return row, col

    @staticmethod
    def grid_to_world(row: int, col: int) -> Tuple[float, float]:
        """Convert grid indices to world ENU (metres)."""
        north = GRID_NORTH_MIN + row * GRID_RESOLUTION
        east  = GRID_EAST_MIN  + col * GRID_RESOLUTION
        return north, east
