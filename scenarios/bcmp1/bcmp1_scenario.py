"""
scenarios/bcmp1/bcmp1_scenario.py
MicroMind / NanoCorteX — BCMP-1 Scenario Definition
Sprint S1 Deliverable 2 of 4

Formal definition of the Baseline Contested Mission Profile (BCMP-1).
This is the primary SIL acceptance gate scenario (Part Two V7 §5.3.3,
TechReview R-06). A successful end-to-end run against all 11 pass
criteria is the minimum required before approaching hardware partners.

Scenario parameters:
  - 100 km ingress; mountainous terrain (LAC corridor proxy)
  - GNSS denied from T+5 min
  - RF link lost at T+15 min
  - 2× jammer nodes at mid-ingress (mandatory ≥2 replans)
  - 1× hostile satellite overpass at T+20 min (terrain masking required)
  - 1× GNSS spoofer at terminal approach
  - Thermal target + 1 thermal decoy in terminal zone
  - Terminal zone: highest EW density of mission; ZPI suppressed; SHM active

11 Pass criteria (ALL must be met for BCMP-1 PASS):
  NAV-01  Drift < 2% of path length at GNSS loss check point (5 km)
  NAV-02  TRN correction error < 50 m CEP-95 during GNSS-denied segment
  EW-01   EW cost-map response ≤ 500 ms from jammer activation
  EW-02   Route replan executed ≤ 1 s; avoidance successful both replans
  EW-03   GNSS spoof rejected by BIM; trust = Red within 250 ms
  SAT-01  Terrain masking manoeuvre executed at correct satellite window
  TERM-01 Thermal target acquired; EO lock confidence ≥ 0.85
  TERM-02 Decoy rejected by DMRL; correct target engaged
  L10S-01 L10s-SE decision correct (continue or abort per envelope)
  LOG-01  All 11 BCMP-1 pass criteria measured and logged
  FSM-01  State machine traverses NOMINAL→EW_AWARE→GNSS_DENIED→
          SILENT_INGRESS→SHM_ACTIVE without error; all transitions logged

References:
  Part Two V7  §5.3.3 (BCMP-1 formal definition), §1.2 (FSM states)
  TechReview   R-06 (BCMP-1 acceptance gate — HIGH priority)
  Demand Analysis §1.3 (DISC 12 circular validation proof)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Coordinate system
# ---------------------------------------------------------------------------
# All positions in ENU (East-North-Up) metres from mission origin.
# Origin = launch point (0, 0, 0).
# 100 km ingress runs nominally along the North axis.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Terrain model (mountainous proxy)
# ---------------------------------------------------------------------------

@dataclass
class TerrainSegment:
    """Altitude profile segment along the ingress corridor."""
    start_km:   float   # distance from origin (km)
    end_km:     float
    min_alt_m:  float   # terrain floor AGL
    max_alt_m:  float   # terrain ceiling (ridge line)
    label:      str     = ""


TERRAIN_PROFILE: List[TerrainSegment] = [
    TerrainSegment(  0,  15, 1800, 2800, "Valley approach"),
    TerrainSegment( 15,  35, 2800, 4200, "First ridge — high EW shadow"),
    TerrainSegment( 35,  55, 3000, 4800, "Central plateau — jammer zone"),
    TerrainSegment( 55,  75, 3500, 5200, "Second ridge — satellite mask zone"),
    TerrainSegment( 75,  90, 2400, 3600, "Terminal valley approach"),
    TerrainSegment( 90, 100, 1500, 2200, "Terminal zone — highest EW density"),
]


# ---------------------------------------------------------------------------
# Route corridor
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    north_m:  float
    east_m:   float
    alt_m:    float
    label:    str = ""


def build_nominal_route() -> List[Waypoint]:
    """
    100 km nominal corridor waypoints (ENU, metres).
    Follows terrain profile with conservative altitude margins.
    Jammer avoidance replans will deviate east/west from this corridor.
    """
    return [
        Waypoint(      0,     0, 2400, "Launch point"),
        Waypoint(  15_000,    0, 3200, "Ridge 1 crossing"),
        Waypoint(  35_000, -500, 4000, "Plateau entry"),
        Waypoint(  45_000,    0, 4000, "Jammer zone mid (nominal)"),
        Waypoint(  55_000,  500, 4500, "Plateau exit"),
        Waypoint(  72_000,    0, 4800, "Satellite mask point (ridge 2)"),
        Waypoint(  85_000,    0, 3000, "Terminal valley entry"),
        Waypoint(  95_000,    0, 2000, "Terminal approach"),
        Waypoint( 100_000,    0, 1600, "Target vicinity"),
    ]


# ---------------------------------------------------------------------------
# Jammer nodes
# ---------------------------------------------------------------------------

class JammerType(str, Enum):
    BROADBAND_GNSS = "BROADBAND_GNSS"   # Jamming only — C/N0 degradation
    ADAPTIVE_SPOOF = "ADAPTIVE_SPOOF"   # Precision spoofing — position offset


@dataclass
class JammerNode:
    jammer_id:      str
    jammer_type:    JammerType
    position_enu:   Tuple[float, float, float]  # (N, E, U) metres
    activation_time_s: float                    # mission time T+X
    deactivation_time_s: float
    effective_radius_m: float                   # jamming influence radius
    confidence_at_50m: float = 0.95             # EW Engine detection confidence at 50m range
    spoof_offset_m: Optional[float] = None      # position offset if ADAPTIVE_SPOOF


JAMMER_NODES: List[JammerNode] = [
    JammerNode(
        jammer_id           = "JMR-01",
        jammer_type         = JammerType.BROADBAND_GNSS,
        position_enu        = (40_000, -3_000, 3_800),  # NW of corridor, plateau
        activation_time_s   = 18 * 60,                  # T+18 min
        deactivation_time_s = 40 * 60,                  # T+40 min
        effective_radius_m  = 15_000,
        confidence_at_50m   = 0.92,
    ),
    JammerNode(
        jammer_id           = "JMR-02",
        jammer_type         = JammerType.BROADBAND_GNSS,
        position_enu        = (50_000,  4_000, 4_200),  # NE of corridor, plateau
        activation_time_s   = 25 * 60,                  # T+25 min
        deactivation_time_s = 55 * 60,
        effective_radius_m  = 12_000,
        confidence_at_50m   = 0.88,
    ),
    # Terminal spoofer — activates in terminal zone
    JammerNode(
        jammer_id           = "SPF-01",
        jammer_type         = JammerType.ADAPTIVE_SPOOF,
        position_enu        = (93_000,  1_500, 1_800),
        activation_time_s   = 52 * 60,                  # T+52 min
        deactivation_time_s = 65 * 60,
        effective_radius_m  = 8_000,
        confidence_at_50m   = 0.85,
        spoof_offset_m      = 250.0,                    # 250 m position offset
    ),
]


# ---------------------------------------------------------------------------
# Satellite overpass
# ---------------------------------------------------------------------------

@dataclass
class SatelliteOverpass:
    """
    Hostile ISR satellite overpass window.
    Aircraft must use terrain masking (fly below ridge line) during window.
    Manoeuvre must begin ≥ 30 s before window opens.
    """
    satellite_id:       str
    window_open_s:      float   # mission time T+X
    window_close_s:     float
    masking_waypoint:   Waypoint  # target waypoint to achieve terrain mask
    lead_time_s:        float = 30.0  # required manoeuvre lead time


SATELLITE_OVERPASSES: List[SatelliteOverpass] = [
    SatelliteOverpass(
        satellite_id    = "SAT-OVERPASS-01",
        window_open_s   = 20 * 60,          # T+20 min
        window_close_s  = 23 * 60,
        masking_waypoint = Waypoint(
            north_m = 72_000,
            east_m  = -2_000,               # offset west into ridge shadow
            alt_m   = 4_600,                # below ridge line
            label   = "Terrain mask point"
        ),
        lead_time_s = 30.0,
    ),
]


# ---------------------------------------------------------------------------
# Target and decoy
# ---------------------------------------------------------------------------

@dataclass
class ThermalTarget:
    target_id:          str
    position_enu:       Tuple[float, float, float]
    ir_signature_w_sr:  float       # thermal emission W/sr
    is_decoy:           bool = False
    decoy_type:         Optional[str] = None  # e.g. "FLARE", "THERMAL_POT"


TERMINAL_TARGETS: List[ThermalTarget] = [
    ThermalTarget(
        target_id           = "TGT-01",
        position_enu        = (100_000, 0, 1_580),
        ir_signature_w_sr   = 42.0,             # primary radar/comms node
        is_decoy            = False,
    ),
    ThermalTarget(
        target_id           = "DCY-01",
        position_enu        = (99_600, 800, 1_590),
        ir_signature_w_sr   = 38.5,             # decoy — similar signature
        is_decoy            = True,
        decoy_type          = "THERMAL_POT",
    ),
]


# ---------------------------------------------------------------------------
# GNSS / RF timeline
# ---------------------------------------------------------------------------

@dataclass
class MissionTimeline:
    """Key mission events by simulation time (seconds from T=0)."""
    gnss_denial_start_s:    float = 5 * 60      # T+5 min
    rf_link_lost_s:         float = 15 * 60     # T+15 min
    terminal_zone_entry_s:  float = 48 * 60     # T+48 min (90 km mark)
    l10s_se_activation_s:   float = 54 * 60     # T+54 min (last 10 s of flight)
    expected_impact_s:      float = 55 * 60     # T+55 min


MISSION_TIMELINE = MissionTimeline()


# ---------------------------------------------------------------------------
# 11 Pass criteria
# ---------------------------------------------------------------------------

class PassCriteriaID(str, Enum):
    NAV_01  = "NAV-01"
    NAV_02  = "NAV-02"
    EW_01   = "EW-01"
    EW_02   = "EW-02"
    EW_03   = "EW-03"
    SAT_01  = "SAT-01"
    TERM_01 = "TERM-01"
    TERM_02 = "TERM-02"
    L10S_01 = "L10S-01"
    LOG_01  = "LOG-01"
    FSM_01  = "FSM-01"


@dataclass
class PassCriterion:
    criterion_id:   PassCriteriaID
    description:    str
    threshold:      str         # human-readable threshold
    module:         str         # responsible module
    sil_sprint:     str         # sprint where this is measured
    measured_value: Optional[float] = None
    passed:         Optional[bool]  = None
    notes:          str             = ""


def build_pass_criteria() -> Dict[PassCriteriaID, PassCriterion]:
    """Return the 11 BCMP-1 pass criteria (Part Two V7 §5.3.3)."""
    return {
        PassCriteriaID.NAV_01: PassCriterion(
            criterion_id = PassCriteriaID.NAV_01,
            description  = "Drift < 2% of total path length at GNSS loss check point",
            threshold    = "< 2% of 5 km = < 100 m",
            module       = "Navigation Engine / ESKF",
            sil_sprint   = "S3",
        ),
        PassCriteriaID.NAV_02: PassCriterion(
            criterion_id = PassCriteriaID.NAV_02,
            description  = "TRN correction error < 50 m CEP-95 during GNSS-denied segment",
            threshold    = "< 50 m CEP-95",
            module       = "TRN (Terrain Referenced Navigation)",
            sil_sprint   = "S3",
        ),
        PassCriteriaID.EW_01: PassCriterion(
            criterion_id = PassCriteriaID.EW_01,
            description  = "EW cost-map response ≤ 500 ms from jammer activation",
            threshold    = "≤ 500 ms",
            module       = "EW Engine",
            sil_sprint   = "S4",
        ),
        PassCriteriaID.EW_02: PassCriterion(
            criterion_id = PassCriteriaID.EW_02,
            description  = "Route replan executed ≤ 1 s; avoidance successful both replans",
            threshold    = "≤ 1 s; 2/2 replans successful",
            module       = "Route Planner / Hybrid A*",
            sil_sprint   = "S4",
        ),
        PassCriteriaID.EW_03: PassCriterion(
            criterion_id = PassCriteriaID.EW_03,
            description  = "GNSS spoof rejected by BIM; trust = Red within 250 ms",
            threshold    = "≤ 250 ms to RED state",
            module       = "BIM",
            sil_sprint   = "S2",
        ),
        PassCriteriaID.SAT_01: PassCriterion(
            criterion_id = PassCriteriaID.SAT_01,
            description  = "Terrain masking manoeuvre executed at correct satellite window",
            threshold    = "Manoeuvre begins ≤ 30 s before window open",
            module       = "Navigation Engine / Route Planner",
            sil_sprint   = "S4",
        ),
        PassCriteriaID.TERM_01: PassCriterion(
            criterion_id = PassCriteriaID.TERM_01,
            description  = "Thermal target acquired; EO lock confidence ≥ 0.85",
            threshold    = "≥ 0.85 lock confidence",
            module       = "Terminal Guidance / DMRL",
            sil_sprint   = "S5",
        ),
        PassCriteriaID.TERM_02: PassCriterion(
            criterion_id = PassCriteriaID.TERM_02,
            description  = "Decoy rejected by DMRL; correct target engaged",
            threshold    = "Decoy rejection rate ≥ 90% (single trial: correct target)",
            module       = "DMRL",
            sil_sprint   = "S5",
        ),
        PassCriteriaID.L10S_01: PassCriterion(
            criterion_id = PassCriteriaID.L10S_01,
            description  = "L10s-SE decision correct (continue or abort per envelope)",
            threshold    = "100% compliance with mission envelope ROE",
            module       = "L10s-SE",
            sil_sprint   = "S5",
        ),
        PassCriteriaID.LOG_01: PassCriterion(
            criterion_id = PassCriteriaID.LOG_01,
            description  = "All 11 BCMP-1 pass criteria measured and logged (NFR-013)",
            threshold    = "Log completeness ≥ 99%",
            module       = "Mission Log",
            sil_sprint   = "S1",
        ),
        PassCriteriaID.FSM_01: PassCriterion(
            criterion_id = PassCriteriaID.FSM_01,
            description  = (
                "State machine traverses NOMINAL→EW_AWARE→GNSS_DENIED→"
                "SILENT_INGRESS→SHM_ACTIVE without error; all transitions logged"
            ),
            threshold    = "All 7 states reachable; all transitions logged with guard result",
            module       = "NanoCorteX FSM",
            sil_sprint   = "S1",
        ),
    }


# ---------------------------------------------------------------------------
# Scenario bundle
# ---------------------------------------------------------------------------

@dataclass
class BCMP1Scenario:
    """
    Complete BCMP-1 scenario definition.
    Instantiate once; pass to bcmp1_runner.py in Sprint S5.
    All sub-modules (BIM, EW Engine, Nav, DMRL) read from this object.
    """
    scenario_id:        str = "BCMP1-SIL-001"
    description:        str = (
        "Baseline Contested Mission Profile 1 — "
        "100 km mountainous ingress, GNSS denied T+5 min, "
        "RF lost T+15 min, 2× jammers, 1× spoofer, "
        "1× satellite overpass, thermal target + decoy."
    )

    route:              List[Waypoint]          = field(default_factory=build_nominal_route)
    terrain:            List[TerrainSegment]    = field(default_factory=lambda: TERRAIN_PROFILE)
    jammer_nodes:       List[JammerNode]        = field(default_factory=lambda: JAMMER_NODES)
    satellite_overpasses: List[SatelliteOverpass] = field(default_factory=lambda: SATELLITE_OVERPASSES)
    targets:            List[ThermalTarget]     = field(default_factory=lambda: TERMINAL_TARGETS)
    timeline:           MissionTimeline         = field(default_factory=MissionTimeline)
    pass_criteria:      Dict[PassCriteriaID, PassCriterion] = field(
                            default_factory=build_pass_criteria)

    # Derived properties
    @property
    def total_distance_m(self) -> float:
        """Straight-line ingress distance (metres)."""
        return 100_000.0

    @property
    def jammer_count(self) -> int:
        return len(self.jammer_nodes)

    @property
    def decoy_count(self) -> int:
        return sum(1 for t in self.targets if t.is_decoy)

    @property
    def primary_target(self) -> Optional[ThermalTarget]:
        for t in self.targets:
            if not t.is_decoy:
                return t
        return None

    def summary(self) -> str:
        lines = [
            f"=== BCMP-1 Scenario: {self.scenario_id} ===",
            self.description,
            "",
            f"Route waypoints:          {len(self.route)}",
            f"Total ingress distance:   {self.total_distance_m/1000:.0f} km",
            f"Terrain segments:         {len(self.terrain)}",
            "",
            f"GNSS denial start:        T+{self.timeline.gnss_denial_start_s/60:.0f} min",
            f"RF link lost:             T+{self.timeline.rf_link_lost_s/60:.0f} min",
            f"Terminal zone entry:      T+{self.timeline.terminal_zone_entry_s/60:.0f} min",
            f"L10s-SE activation:       T+{self.timeline.l10s_se_activation_s/60:.0f} min",
            "",
            f"Jammer nodes:             {self.jammer_count}",
            f"Satellite overpasses:     {len(self.satellite_overpasses)}",
            f"Terminal targets:         {len(self.targets)} ({self.decoy_count} decoy)",
            "",
            "Pass criteria (11 total):",
        ]
        for pc in self.pass_criteria.values():
            status = (
                "✅ PASS" if pc.passed is True else
                "❌ FAIL" if pc.passed is False else
                f"⬜ PENDING [{pc.sil_sprint}]"
            )
            lines.append(f"  {pc.criterion_id.value:8s}  {status}  — {pc.description[:60]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton for import convenience
# ---------------------------------------------------------------------------

BCMP1 = BCMP1Scenario()


if __name__ == "__main__":
    print(BCMP1.summary())
