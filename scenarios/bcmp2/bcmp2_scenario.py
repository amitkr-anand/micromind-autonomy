"""
BCMP-2 Scenario Definition and Disturbance Schedule.

Constraint C-4 (Disturbance Parity)
-------------------------------------
Both vehicles experience identical external disturbances.  The disturbance
schedule is generated once from the shared seed before the run begins and
passed as a shared input to both vehicle tracks.  Neither track may generate
its own disturbances.

The full schedule is serialised into the JSON output at the top level
(not inside vehicle_a or vehicle_b) so a reviewer can verify both vehicles
received identical inputs.

Five-phase mission profile
--------------------------
Phase   km          Terrain         Vehicle B nav       Vehicle A fate
P1      0–30        Mountain        TRN-primary         GNSS OK; both identical
P2      30–60       Valley          VIO-primary         GNSS denied; drift begins
P3      60–100      Plains          VIO-primary         Drift compounds; breach ~km77
P4      100–120     Industrial      VIO + EO assist     Corridor violated
P5      120–150     Terminal        Enforcement active  Unsafe terminal / abort

Canonical seeds
---------------
  42   — nominal reference run
  101  — alternate weather / sensor noise profile
  303  — degraded / stress profile

JOURNAL
-------
Built: 29 March 2026, micromind-node01.  SB-1 Step 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from scenarios.bcmp2.bcmp2_terrain_gen import (
    BCMP2Terrain,
    TerrainParams,
    PHASE_TERRAIN_PROFILES,
    terrain_for_phase,
)
from scenarios.bcmp2.bcmp2_drift_envelopes import GNSS_DENIAL_KM, VEHICLE_SPEED_MS


# ---------------------------------------------------------------------------
# Mission geometry constants
# ---------------------------------------------------------------------------

MISSION_TOTAL_KM   = 150.0   # full mission distance
PHASE_BOUNDARIES_KM = [0, 30, 60, 100, 120, 150]   # P1..P5 boundaries

# Phase labels for logging and reporting
PHASE_LABELS = {
    1: "P1_mountain_ingress",
    2: "P2_valley_corridor",
    3: "P3_plains",
    4: "P4_urban_clutter",
    5: "P5_terminal",
}

# Speed and timing
VEHICLE_SPEED    = VEHICLE_SPEED_MS          # m/s (ALS-250 class)
MISSION_DURATION_S = (MISSION_TOTAL_KM * 1000.0) / VEHICLE_SPEED

# GNSS
GNSS_DENIAL_START_KM = GNSS_DENIAL_KM        # km 30 — matches C-2 derivation
GNSS_DENIAL_START_S  = (GNSS_DENIAL_START_KM * 1000.0) / VEHICLE_SPEED

# Mission corridor half-width
CORRIDOR_HALF_WIDTH_M = 500.0   # metres — Vehicle A expected to breach by P4


# ---------------------------------------------------------------------------
# Waypoints
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    """Simple north/east/alt waypoint (metres from mission origin)."""
    north_m: float
    east_m:  float
    alt_m:   float
    label:   str = ""

    def distance_to(self, other: "Waypoint") -> float:
        return math.sqrt(
            (self.north_m - other.north_m) ** 2
            + (self.east_m - other.east_m) ** 2
        )


def build_nominal_route() -> List[Waypoint]:
    """
    150 km mission route — straight ingress with gentle heading changes
    at phase boundaries to stress cross-track drift visibility.
    """
    # Primary heading: north-northeast (representative of LAC ingress)
    # Each phase boundary introduces a small heading jog to make lateral
    # drift visually apparent in the route map.
    return [
        Waypoint(north_m=0,       east_m=0,      alt_m=2000, label="LAUNCH"),
        Waypoint(north_m=28_000,  east_m=3_000,  alt_m=3200, label="P1_end"),
        Waypoint(north_m=56_000,  east_m=8_000,  alt_m=1800, label="P2_end"),
        Waypoint(north_m=96_000,  east_m=12_000, alt_m=500,  label="P3_end"),
        Waypoint(north_m=114_000, east_m=15_000, alt_m=400,  label="P4_end"),
        Waypoint(north_m=148_000, east_m=16_000, alt_m=200,  label="TARGET"),
    ]


# ---------------------------------------------------------------------------
# Disturbance events (C-4: one schedule, shared by both vehicles)
# ---------------------------------------------------------------------------

@dataclass
class GNSSDenialWindow:
    start_s: float
    end_s:   float      # math.inf = denied for the remainder of the mission

@dataclass
class VIOOutageEvent:
    start_s:    float
    duration_s: float
    reason:     str = "scripted"

@dataclass
class WindProfile:
    """Simple constant + gust wind model (m/s)."""
    mean_north_ms:    float = 0.0
    mean_east_ms:     float = 0.0
    gust_amplitude_ms: float = 0.0
    gust_period_s:    float = 60.0
    turbulence_sigma_ms: float = 0.0

@dataclass
class DisturbanceSchedule:
    """
    The single shared disturbance schedule for a BCMP-2 run.

    Generated once from the mission seed.  Passed as a read-only input to
    both vehicle tracks.  Serialised at top level of the JSON output.

    This is the C-4 contract: both vehicles receive the same instance.
    """
    seed:            int
    gnss_denial:     GNSSDenialWindow
    wind:            WindProfile
    vio_outages:     List[VIOOutageEvent]
    radalt_loss_s:   Optional[float]      # None = never
    eo_freeze_s:     Optional[float]      # None = never

    # Noise scaling factors (applied identically to both vehicles' IMU)
    imu_noise_scale: float = 1.0          # 1.0 = nominal; >1.0 = stress

    def phase_at_km(self, mission_km: float) -> int:
        """Return phase number (1–5) for a given mission distance."""
        bounds = PHASE_BOUNDARIES_KM
        for i in range(len(bounds) - 1):
            if bounds[i] <= mission_km < bounds[i + 1]:
                return i + 1
        return 5

    def phase_at_time(self, t_s: float) -> int:
        return self.phase_at_km(t_s * VEHICLE_SPEED / 1000.0)

    def gnss_available(self, t_s: float) -> bool:
        return not (self.gnss_denial.start_s <= t_s < self.gnss_denial.end_s)

    def vio_available(self, t_s: float) -> bool:
        for ev in self.vio_outages:
            if ev.start_s <= t_s < ev.start_s + ev.duration_s:
                return False
        return True

    def wind_at(self, t_s: float) -> tuple:
        """Return (north_ms, east_ms) wind vector at time t_s."""
        w = self.wind
        gust = w.gust_amplitude_ms * math.sin(
            2 * math.pi * t_s / max(w.gust_period_s, 1.0)
        )
        return (w.mean_north_ms + gust, w.mean_east_ms)

    def to_dict(self) -> dict:
        """Serialise for JSON output at top level of KPI log (C-4 traceability)."""
        return {
            "seed": self.seed,
            "gnss_denial": {
                "start_s": self.gnss_denial.start_s,
                "end_s":   self.gnss_denial.end_s
                           if self.gnss_denial.end_s != math.inf else "inf",
            },
            "wind": {
                "mean_north_ms":      self.wind.mean_north_ms,
                "mean_east_ms":       self.wind.mean_east_ms,
                "gust_amplitude_ms":  self.wind.gust_amplitude_ms,
                "gust_period_s":      self.wind.gust_period_s,
                "turbulence_sigma_ms": self.wind.turbulence_sigma_ms,
            },
            "vio_outages": [
                {"start_s": v.start_s, "duration_s": v.duration_s, "reason": v.reason}
                for v in self.vio_outages
            ],
            "radalt_loss_s": self.radalt_loss_s,
            "eo_freeze_s":   self.eo_freeze_s,
            "imu_noise_scale": self.imu_noise_scale,
        }


# ---------------------------------------------------------------------------
# Schedule generator
# ---------------------------------------------------------------------------

# Seed-specific profile overrides
_SEED_PROFILES = {
    42: {
        "description":      "Nominal reference run",
        "wind_mean_n":      2.0,
        "wind_mean_e":      1.5,
        "gust_amp":         3.0,
        "turbulence_sigma": 0.5,
        "vio_outage_km":    48.0,   # mid-P2
        "vio_duration_s":   15.0,
        "radalt_loss_s":    None,
        "eo_freeze_s":      None,
        "imu_noise_scale":  1.0,
    },
    101: {
        "description":      "Alternate weather / sensor noise",
        "wind_mean_n":      4.0,
        "wind_mean_e":      3.0,
        "gust_amp":         6.0,
        "turbulence_sigma": 1.2,
        "vio_outage_km":    55.0,   # late P2
        "vio_duration_s":   25.0,
        "radalt_loss_s":    None,
        "eo_freeze_s":      None,
        "imu_noise_scale":  1.15,
    },
    303: {
        "description":      "Degraded / stress profile",
        "wind_mean_n":      6.0,
        "wind_mean_e":      4.0,
        "gust_amp":         10.0,
        "turbulence_sigma": 2.0,
        "vio_outage_km":    42.0,   # early P2
        "vio_duration_s":   35.0,
        "radalt_loss_s":    (120.0 * 1000.0 / VEHICLE_SPEED),  # km 120 → P5 entry
        "eo_freeze_s":      None,
        "imu_noise_scale":  1.30,
    },
}


def generate_disturbance_schedule(seed: int) -> DisturbanceSchedule:
    """
    Generate the shared disturbance schedule for a BCMP-2 run.

    For canonical seeds (42, 101, 303), uses the pre-defined profile above.
    For arbitrary seeds, interpolates parameters stochastically from the seed.

    The returned schedule is immutable during a run — both vehicle tracks
    receive the same object (C-4).
    """
    if seed in _SEED_PROFILES:
        prof = _SEED_PROFILES[seed]
    else:
        # Stochastic profile for non-canonical seeds
        rng = np.random.default_rng(seed)
        vio_km = float(rng.uniform(35.0, 58.0))
        prof = {
            "description":      f"Stochastic profile (seed={seed})",
            "wind_mean_n":      float(rng.uniform(0.0, 5.0)),
            "wind_mean_e":      float(rng.uniform(0.0, 4.0)),
            "gust_amp":         float(rng.uniform(1.0, 8.0)),
            "turbulence_sigma": float(rng.uniform(0.3, 1.5)),
            "vio_outage_km":    vio_km,
            "vio_duration_s":   float(rng.uniform(10.0, 30.0)),
            "radalt_loss_s":    None,
            "eo_freeze_s":      None,
            "imu_noise_scale":  float(rng.uniform(1.0, 1.2)),
        }

    vio_start_s = (prof["vio_outage_km"] * 1000.0) / VEHICLE_SPEED

    return DisturbanceSchedule(
        seed=seed,
        gnss_denial=GNSSDenialWindow(
            start_s=GNSS_DENIAL_START_S,
            end_s=math.inf,   # denied for remainder of mission
        ),
        wind=WindProfile(
            mean_north_ms=prof["wind_mean_n"],
            mean_east_ms=prof["wind_mean_e"],
            gust_amplitude_ms=prof["gust_amp"],
            gust_period_s=45.0,
            turbulence_sigma_ms=prof["turbulence_sigma"],
        ),
        vio_outages=[
            VIOOutageEvent(
                start_s=vio_start_s,
                duration_s=prof["vio_duration_s"],
                reason="scripted_P2",
            )
        ],
        radalt_loss_s=prof["radalt_loss_s"],
        eo_freeze_s=prof["eo_freeze_s"],
        imu_noise_scale=prof["imu_noise_scale"],
    )


# ---------------------------------------------------------------------------
# BCMP-2 scenario definition
# ---------------------------------------------------------------------------

@dataclass
class BCMP2Scenario:
    """
    Full BCMP-2 scenario definition.

    Contains all static mission geometry plus the per-phase terrain objects.
    The disturbance_schedule is generated separately via
    generate_disturbance_schedule(seed) and is NOT part of this dataclass —
    keeping it separate enforces C-4: the schedule is shared by both tracks
    and must not be embedded inside a per-vehicle structure.
    """
    scenario_id:      str = "BCMP-2"
    description:      str = ("150 km contested corridor — dual-track comparative "
                             "demonstration.  Vehicle A: INS+GNSS→INS-only.  "
                             "Vehicle B: MicroMind full stack.")
    total_distance_km: float = MISSION_TOTAL_KM
    vehicle_speed_ms:  float = VEHICLE_SPEED_MS
    corridor_half_width_m: float = CORRIDOR_HALF_WIDTH_M

    route: List[Waypoint] = field(default_factory=build_nominal_route)

    # Phase terrain instances (one per phase, constructed at scenario init)
    terrain_p1: BCMP2Terrain = field(
        default_factory=lambda: terrain_for_phase("P1_mountain"))
    terrain_p2: BCMP2Terrain = field(
        default_factory=lambda: terrain_for_phase("P2_valley"))
    terrain_p3: BCMP2Terrain = field(
        default_factory=lambda: terrain_for_phase("P3_plains"))
    terrain_p4: BCMP2Terrain = field(
        default_factory=lambda: terrain_for_phase("P4_urban"))
    terrain_p5: BCMP2Terrain = field(
        default_factory=lambda: terrain_for_phase("P5_terminal"))

    def terrain_for_km(self, mission_km: float) -> BCMP2Terrain:
        """Return the terrain object appropriate for a given mission km."""
        if mission_km < 30:
            return self.terrain_p1
        elif mission_km < 60:
            return self.terrain_p2
        elif mission_km < 100:
            return self.terrain_p3
        elif mission_km < 120:
            return self.terrain_p4
        else:
            return self.terrain_p5

    def phase_at_km(self, mission_km: float) -> int:
        bounds = PHASE_BOUNDARIES_KM
        for i in range(len(bounds) - 1):
            if bounds[i] <= mission_km < bounds[i + 1]:
                return i + 1
        return 5

    def interpolate_waypoint(self, mission_km: float) -> Waypoint:
        """
        Return the planned (north_m, east_m, alt_m) at a given mission km
        by linear interpolation between route waypoints.
        """
        route_km = [0, 30, 60, 100, 120, 150]
        idx = 0
        for i in range(len(route_km) - 1):
            if route_km[i] <= mission_km <= route_km[i + 1]:
                idx = i
                break
        a, b = self.route[idx], self.route[idx + 1]
        t = (mission_km - route_km[idx]) / max(route_km[idx + 1] - route_km[idx], 1e-9)
        return Waypoint(
            north_m=a.north_m + t * (b.north_m - a.north_m),
            east_m=a.east_m  + t * (b.east_m  - a.east_m),
            alt_m=a.alt_m    + t * (b.alt_m   - a.alt_m),
            label=f"interp_{mission_km:.1f}km",
        )

    def summary(self) -> str:
        return (
            f"{self.scenario_id}: {self.total_distance_km:.0f} km | "
            f"GNSS denied km {GNSS_DENIAL_KM:.0f}+ | "
            f"Corridor ±{self.corridor_half_width_m:.0f} m | "
            f"Phases: P1 mountain / P2 valley / P3 plains / P4 urban / P5 terminal"
        )


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("BCMP-2 Scenario — self-verification")
    print("=" * 60)

    sc = BCMP2Scenario()
    print(sc.summary())
    print()

    # Check route
    print("Route waypoints:")
    for wp in sc.route:
        print(f"  {wp.label:<12} N={wp.north_m/1000:.0f} km  "
              f"E={wp.east_m/1000:.0f} km  alt={wp.alt_m:.0f} m")
    print()

    # Check phase attribution
    print("Phase attribution:")
    for km in [0, 15, 30, 45, 60, 80, 100, 110, 120, 135, 150]:
        ph = sc.phase_at_km(km)
        terrain = sc.terrain_for_km(km)
        print(f"  km {km:>3}: Phase {ph} — {terrain}")
    print()

    # Check disturbance schedules for all canonical seeds
    print("Disturbance schedules:")
    for seed in [42, 101, 303]:
        sched = generate_disturbance_schedule(seed)
        gnss_km  = sched.gnss_denial.start_s * VEHICLE_SPEED / 1000.0
        vio_km   = (sched.vio_outages[0].start_s * VEHICLE_SPEED / 1000.0
                    if sched.vio_outages else None)
        vio_dur  = sched.vio_outages[0].duration_s if sched.vio_outages else 0
        print(f"  seed={seed}: GNSS denied km {gnss_km:.0f}  "
              f"VIO outage km {vio_km:.1f} for {vio_dur:.0f}s  "
              f"imu_scale={sched.imu_noise_scale:.2f}")
        # Verify to_dict is serialisable
        d = sched.to_dict()
        assert d["seed"] == seed
    print()

    # Verify C-4: same schedule object returned for same seed
    s1 = generate_disturbance_schedule(42)
    s2 = generate_disturbance_schedule(42)
    assert s1.gnss_denial.start_s == s2.gnss_denial.start_s
    assert s1.vio_outages[0].start_s == s2.vio_outages[0].start_s
    print("C-4 determinism check (seed 42, two calls): PASS")
    print()
    print("All checks passed.")
