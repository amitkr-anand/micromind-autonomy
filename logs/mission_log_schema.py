"""
logs/mission_log_schema.py
MicroMind / NanoCorteX — Mission Log Schema
Sprint S1 Deliverable 3 of 4

DD-02 Phase 1 minimum: mission log with learning fields.
Every field logged here feeds the post-mission cross-mission
learning pipeline (DD-02 Phase 2, deferred to post-SIL).

Log completeness target: ≥99% of fields populated per entry (NFR-013).
All timestamps sourced from SimClock.now() for monotonic ordering.

Field categories:
  - CORE:     always populated; log entry invalid if missing
  - LEARNING: DD-02 Phase 1 fields; populated when module active
  - AUDIT:    guard evaluation results; required for SIL evidence

References: DD-02, NFR-013, FR-101 (BIM), FR-103 (DMRL),
            FR-104 (ZPI), Section 1.2.2 (state transitions)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations (mirror state_machine.py — kept separate to avoid circular dep)
# ---------------------------------------------------------------------------

class LogCategory(str, Enum):
    STATE_TRANSITION  = "STATE_TRANSITION"   # NanoCorteX FSM event
    EW_OBSERVATION    = "EW_OBSERVATION"     # EW Engine output
    BIM_UPDATE        = "BIM_UPDATE"         # BIM trust score update
    NAVIGATION        = "NAVIGATION"         # Navigation engine output
    DMRL              = "DMRL"               # Decoy rejection event
    ROUTE_DEVIATION   = "ROUTE_DEVIATION"    # Corridor deviation / replan
    ZPI_BURST         = "ZPI_BURST"          # ZPI pre-terminal burst
    L10S_SE           = "L10S_SE"            # Last-10-second decision
    MISSION_START     = "MISSION_START"      # T=0 record
    MISSION_END       = "MISSION_END"        # Final state record
    SYSTEM_ALERT      = "SYSTEM_ALERT"       # Non-transition system event


class BIMState(str, Enum):
    GREEN  = "GREEN"
    AMBER  = "AMBER"
    RED    = "RED"
    UNKNOWN = "UNKNOWN"


class NavMode(str, Enum):
    GNSS_PRIMARY  = "GNSS_PRIMARY"
    VIO_TRN       = "VIO_TRN"
    IMU_VO_DEM    = "IMU_VO_DEM"
    LOITER_EGRESS = "LOITER_EGRESS"
    SUPPRESSED    = "SUPPRESSED"


# ---------------------------------------------------------------------------
# Sub-records (embedded in log entries)
# ---------------------------------------------------------------------------

@dataclass
class BIMRecord:
    """DD-02 LEARNING field: BIM trust timeseries point."""
    trust_score:    float               # 0.0–1.0
    bim_state:      BIMState            # GREEN / AMBER / RED
    pdop:           Optional[float] = None
    constellation_count: Optional[int] = None
    doppler_deviation_m: Optional[float] = None
    cn0_db:         Optional[float] = None
    hysteresis_count: int = 0           # samples in current state (3 needed for transition)
    spoof_delta_m:  Optional[float] = None  # position offset if spoof injected


@dataclass
class EWObservation:
    """DD-02 LEARNING field: EW Engine observation."""
    jammer_count:       int
    jammer_hypotheses:  List[Dict[str, Any]] = field(default_factory=list)
    # Each hypothesis: {id, bearing_deg, confidence, range_estimate_m}
    cost_map_updated:   bool  = False
    cost_map_latency_ms: Optional[float] = None
    dbscan_cluster_count: Optional[int] = None
    df_risk_level:      float = 0.0     # 0.0–1.0


@dataclass
class NavigationRecord:
    """Navigation engine state snapshot."""
    nav_mode:       NavMode
    position_enu:   List[float]         # [E, N, U] metres from origin
    velocity_enu:   List[float]         # [vE, vN, vU] m/s
    position_covariance_m2: float       # scalar trace of position covariance
    gnss_trust:     float               # 0.0–1.0 from BIM
    drift_m:        Optional[float] = None  # vs ground truth (SIL only)
    trn_correction_m: Optional[float] = None
    vio_feature_count: Optional[int] = None


@dataclass
class DMRLRecord:
    """DD-02 LEARNING field: Decoy rejection confidence."""
    eo_lock_confidence: float           # 0.0–1.0
    target_confirmed:   bool
    decoy_present:      bool
    decoy_rejected:     bool
    frame_count:        int             # frames evaluated
    thermal_dissipation_score: Optional[float] = None
    lock_latency_ms:    Optional[float] = None


@dataclass
class RouteDeviationRecord:
    """Corridor deviation / replan event."""
    deviation_trigger:      str         # e.g. "EW_COST_MAP", "CORRIDOR_VIOLATION"
    replan_executed:        bool
    replan_latency_ms:      Optional[float] = None
    avoidance_successful:   Optional[bool] = None
    corridor_violation:     bool = False
    new_waypoint_count:     Optional[int] = None


@dataclass
class ZPIBurstRecord:
    """ZPI pre-terminal EW summary burst (DD-02 Phase 1)."""
    burst_transmitted:      bool
    ew_summary_included:    bool        # DD-02 requirement
    peer_count:             int = 0
    burst_latency_ms:       Optional[float] = None
    rf_suppressed_after:    bool = False


@dataclass
class L10sSERecord:
    """Last-10-second safety envelope decision."""
    decision:               str         # "CONTINUE" or "ABORT"
    eo_lock_confidence:     float
    corridor_clear:         bool
    l10s_elapsed_s:         float
    abort_reason:           Optional[str] = None


@dataclass
class GuardEvaluation:
    """
    AUDIT field: result of a state machine guard check.
    Required for SIL evidence — every transition logged with guard result.
    (Part Two V7 Section 1.2 / TechReview R-01)
    """
    guard_name:     str
    guard_result:   bool
    guard_value:    Optional[Any] = None    # the value that was tested
    threshold:      Optional[Any] = None   # the threshold applied


# ---------------------------------------------------------------------------
# Primary log entry
# ---------------------------------------------------------------------------

@dataclass
class MissionLogEntry:
    """
    Single mission log entry.

    CORE fields must always be populated.
    LEARNING fields populated when the relevant module is active.
    AUDIT fields populated on every state transition.

    NFR-013: ≥99% field completeness per entry.
    Completeness is computed as populated_fields / total_applicable_fields.
    """

    # --- CORE (always populated) ---
    entry_id:       str     = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_s:    float   = 0.0           # SimClock.now()
    tick:           int     = 0             # SimClock.tick()
    category:       LogCategory = LogCategory.SYSTEM_ALERT
    mission_id:     str     = ""
    state:          str     = ""            # current NanoCorteX state name

    # --- STATE TRANSITION (populated on category=STATE_TRANSITION) ---
    from_state:             Optional[str]   = None
    to_state:               Optional[str]   = None
    transition_trigger:     Optional[str]   = None
    transition_latency_ms:  Optional[float] = None   # must be ≤ 2000 ms (NFR-002)
    guards:                 List[GuardEvaluation] = field(default_factory=list)

    # --- LEARNING: BIM (DD-02) ---
    bim:            Optional[BIMRecord]          = None

    # --- LEARNING: EW (DD-02) ---
    ew:             Optional[EWObservation]       = None

    # --- LEARNING: Navigation ---
    navigation:     Optional[NavigationRecord]    = None

    # --- LEARNING: DMRL (DD-02) ---
    dmrl:           Optional[DMRLRecord]          = None

    # --- LEARNING: Route ---
    route_deviation: Optional[RouteDeviationRecord] = None

    # --- LEARNING: ZPI (DD-02) ---
    zpi:            Optional[ZPIBurstRecord]      = None

    # --- L10s-SE ---
    l10s_se:        Optional[L10sSERecord]        = None

    # --- AUDIT ---
    notes:          Optional[str]   = None

    # ------------------------------------------------------------------
    # Completeness check (NFR-013)
    # ------------------------------------------------------------------

    def completeness(self) -> float:
        """
        Compute field completeness for this entry.

        CORE fields are always counted as applicable.
        Transition fields counted only for STATE_TRANSITION entries.
        Sub-record fields: only required (non-Optional) fields are counted
        against completeness — optional sensor readings are not penalised
        when the sensor is legitimately inactive.

        Returns float in [0.0, 1.0].
        """
        core_fields_check = {
            "entry_id":   lambda v: bool(v),
            "timestamp_s": lambda v: v is not None,
            "tick":        lambda v: v is not None,   # 0 is a valid tick
            "category":    lambda v: bool(v),
            "mission_id":  lambda v: bool(v),
            "state":       lambda v: bool(v),
        }
        applicable = len(core_fields_check)
        populated  = sum(
            1 for fname, check in core_fields_check.items()
            if check(getattr(self, fname))
        )

        # Transition fields — only applicable for STATE_TRANSITION entries
        if self.category == LogCategory.STATE_TRANSITION:
            applicable += 4  # from_state, to_state, trigger, latency
            populated  += sum(1 for v in [
                self.from_state, self.to_state,
                self.transition_trigger, self.transition_latency_ms
            ] if v is not None)
            applicable += 1  # guards list non-empty
            populated  += 1 if len(self.guards) > 0 else 0

        # Sub-records: only required fields counted.
        # Optional measurement fields excluded — populated only when sensor active.
        REQUIRED = {
            "BIMRecord":            ["trust_score", "bim_state", "hysteresis_count"],
            "EWObservation":        ["jammer_count", "cost_map_updated", "df_risk_level"],
            "NavigationRecord":     ["nav_mode", "position_enu", "velocity_enu",
                                     "position_covariance_m2", "gnss_trust"],
            "DMRLRecord":           ["eo_lock_confidence", "target_confirmed",
                                     "decoy_present", "decoy_rejected", "frame_count"],
            "RouteDeviationRecord": ["deviation_trigger", "replan_executed", "corridor_violation"],
            "ZPIBurstRecord":       ["burst_transmitted", "ew_summary_included",
                                     "peer_count", "rf_suppressed_after"],
            "L10sSERecord":         ["decision", "eo_lock_confidence",
                                     "corridor_clear", "l10s_elapsed_s"],
        }

        for record in [self.bim, self.ew, self.navigation,
                       self.dmrl, self.route_deviation, self.zpi, self.l10s_se]:
            if record is not None:
                required_fields = REQUIRED.get(type(record).__name__, [])
                applicable += len(required_fields)
                populated  += sum(
                    1 for fname in required_fields
                    if getattr(record, fname, None) is not None
                )

        if applicable == 0:
            return 1.0
        return populated / applicable

    def is_complete(self, threshold: float = 0.99) -> bool:
        return self.completeness() >= threshold

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, indent=2)


# ---------------------------------------------------------------------------
# Log store
# ---------------------------------------------------------------------------

class MissionLog:
    """
    In-memory mission log store.

    Accumulates MissionLogEntry objects during a simulation run.
    Provides filtering, completeness audit, and JSON export.

    Usage:
        log = MissionLog(mission_id="BCMP1-SIL-001")
        log.append(entry)
        log.export_json("mission_log.json")
        print(log.completeness_report())
    """

    def __init__(self, mission_id: str):
        self.mission_id: str = mission_id
        self._entries: List[MissionLogEntry] = []

    def append(self, entry: MissionLogEntry) -> None:
        """Append an entry. Stamps mission_id if not set."""
        if not entry.mission_id:
            entry.mission_id = self.mission_id
        self._entries.append(entry)

    def entries(self) -> List[MissionLogEntry]:
        return list(self._entries)

    def by_category(self, category: LogCategory) -> List[MissionLogEntry]:
        return [e for e in self._entries if e.category == category]

    def transitions(self) -> List[MissionLogEntry]:
        return self.by_category(LogCategory.STATE_TRANSITION)

    def count(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # NFR-013: completeness audit
    # ------------------------------------------------------------------

    def completeness_report(self) -> Dict[str, Any]:
        """
        Audit log completeness against NFR-013 (≥99%).

        Returns dict with per-entry and aggregate metrics.
        """
        if not self._entries:
            return {"entry_count": 0, "mean_completeness": 0.0, "nfr_013_pass": False}

        scores = [e.completeness() for e in self._entries]
        mean   = sum(scores) / len(scores)
        below  = [i for i, s in enumerate(scores) if s < 0.99]

        return {
            "entry_count":          len(self._entries),
            "mean_completeness":    round(mean, 4),
            "min_completeness":     round(min(scores), 4),
            "entries_below_99pct":  len(below),
            "below_threshold_idx":  below[:10],  # first 10 only
            "nfr_013_pass":         mean >= 0.99,
        }

    # ------------------------------------------------------------------
    # NFR-002: transition timing audit
    # ------------------------------------------------------------------

    def transition_timing_report(self) -> Dict[str, Any]:
        """
        Audit all state transitions against NFR-002 (≤2000 ms).
        """
        transitions = self.transitions()
        violations  = [
            {
                "entry_id":   e.entry_id,
                "timestamp_s": e.timestamp_s,
                "from_state": e.from_state,
                "to_state":   e.to_state,
                "latency_ms": e.transition_latency_ms,
            }
            for e in transitions
            if e.transition_latency_ms is not None and e.transition_latency_ms > 2000
        ]
        latencies = [
            e.transition_latency_ms for e in transitions
            if e.transition_latency_ms is not None
        ]
        return {
            "transition_count":  len(transitions),
            "nfr_002_violations": len(violations),
            "max_latency_ms":    max(latencies) if latencies else None,
            "mean_latency_ms":   round(sum(latencies)/len(latencies), 2) if latencies else None,
            "nfr_002_pass":      len(violations) == 0,
            "violations":        violations,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, path: str) -> None:
        """Export full log to JSON file."""
        payload = {
            "mission_id":   self.mission_id,
            "entry_count":  self.count(),
            "entries":      [e.to_dict() for e in self._entries],
        }
        with open(path, "w") as f:
            json.dump(payload, f, default=str, indent=2)

    def __repr__(self) -> str:
        return f"MissionLog(mission_id={self.mission_id!r}, entries={self.count()})"
