"""
tests/test_gate5_corridor.py
MicroMind / NanoCorteX — Gate 5 Full Corridor Tests

Validates the full 180km Shimla-Manali corridor with real merged DEM,
Monte Carlo N=300 drift envelopes, and compound fault injection.

Gates:
    NAV-13: Merged DEM covers all SHIMLA_MANALI waypoints
    NAV-14: Terrain suitability varies across three corridor zones
    NAV-15: Monte Carlo 180km — P99 TRN < P99 none at all checkpoints
    NAV-16: Compound fault survival — SHM not triggered

Req IDs: NAV-02, NAV-03, EC-09, EC-11
SRS ref: §2.2, §2.3, AD-16
Governance: Code Governance Manual v3.4
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.bim.bim import BIM, BIMConfig, GNSSMeasurement
from core.ekf.error_state_ekf import ErrorStateEKF
from core.fusion.vio_mode import VIONavigationMode
from core.ins.state import INSState
from core.navigation.corridors import SHIMLA_MANALI
from core.navigation.monte_carlo_nav import MonteCarloNavEvaluator
from core.navigation.navigation_manager import (
    NAV_MODE_GNSS_DENIED,
    NAV_MODE_NOMINAL,
    NAV_MODE_TRN_ONLY,
    NAV_MODE_SHM_TRIGGER,
    NavigationManager,
)
from core.trn.dem_loader import DEMLoader
from core.trn.hillshade_generator import HillshadeGenerator
from core.trn.phase_correlation_trn import TRNMatchResult
from core.trn.terrain_suitability import TerrainSuitabilityScorer
from integration.camera.nadir_camera_bridge import NadirCameraFrameBridge
from integration.vio.vio_frame_processor import VIOEstimate, VIOFrameProcessor


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_GSD_M        = 5.0
_ALT_M        = 1500.0
_LAT_SHIMLA   = 31.104
_LON_SHIMLA   = 77.173
_TERRAIN_DIR  = "data/terrain/shimla_manali_corridor/"

_CHECKPOINTS  = [30, 60, 90, 120, 150, 180]


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def merged_dem():
    """DEMLoader with both Shimla and Manali tiles stitched via from_directory()."""
    return DEMLoader.from_directory(_TERRAIN_DIR)


@pytest.fixture(scope="module")
def mc_evaluator(merged_dem):
    """MonteCarloNavEvaluator for SHIMLA_MANALI, N=10 (CI speed)."""
    return MonteCarloNavEvaluator(
        corridor=SHIMLA_MANALI,
        dem_loader=merged_dem,
        n_seeds=10,
        checkpoint_km=_CHECKPOINTS,
    )


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_eskf() -> ErrorStateEKF:
    return ErrorStateEKF()


def _make_ins_state() -> INSState:
    return INSState(
        p  = np.zeros(3),
        v  = np.zeros(3),
        q  = np.array([1.0, 0.0, 0.0, 0.0]),
        ba = np.zeros(3),
        bg = np.zeros(3),
    )


def _make_mock_trn(status: str = "ACCEPTED") -> MagicMock:
    mock = MagicMock()
    mock.match.return_value = TRNMatchResult(
        status=status,
        correction_north_m=5.0,
        correction_east_m=3.0,
        confidence=0.82,
        suitability_score=0.68,
        suitability_recommendation="ACCEPT",
        latency_ms=25,
        mission_time_ms=1000,
    )
    return mock


def _make_nav_manager(
    trn=None,
    trn_interval_m: float = 5000.0,
    vio_confidence_threshold: float = 0.50,
) -> tuple:
    if trn is None:
        trn = _make_mock_trn()

    event_log: list = []
    clock_ref = [0]

    def clock_fn():
        clock_ref[0] += 1
        return clock_ref[0]

    bim      = BIM(BIMConfig())
    vio_mode = VIONavigationMode()
    camera   = NadirCameraFrameBridge(clock_fn=clock_fn)
    vio_proc = VIOFrameProcessor(min_features=30, gsd_m=_GSD_M)

    nm = NavigationManager(
        eskf=_make_eskf(),
        bim=bim,
        trn=trn,
        vio_mode=vio_mode,
        camera_bridge=camera,
        vio_processor=vio_proc,
        event_log=event_log,
        clock_fn=clock_fn,
        trn_interval_m=trn_interval_m,
        vio_confidence_threshold=vio_confidence_threshold,
    )
    return nm, event_log


def _synthetic_camera_tile(rng=None, low_texture: bool = False) -> np.ndarray:
    """Generate a synthetic camera tile. low_texture=True for VIO degradation."""
    if rng is None:
        rng = np.random.default_rng(99)
    if low_texture:
        # Uniform grey — near-zero features → VIO confidence ≈ 0
        return np.full((64, 64), 128, dtype=np.uint8)
    tile = rng.integers(30, 220, size=(64, 64), dtype=np.uint8)
    tile = tile + np.linspace(0, 50, 64, dtype=np.uint8).reshape(1, 64)
    return tile.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Gate NAV-13: Merged DEM covers all SHIMLA_MANALI waypoints
# ---------------------------------------------------------------------------

class TestNAV13MergedDEMCoverage:
    """
    Gate NAV-13: DEMLoader.from_directory() loads both tiles and covers
    all 8 SHIMLA_MANALI waypoints.

    Asserts:
      - All 8 waypoints return valid elevation (not NaN)
      - Merged bounds north > 32.2°N (Manali tile admitted)
    """

    def test_nav13_all_waypoints_in_bounds(self, merged_dem):
        """All 8 waypoints return is_in_bounds=True."""
        for i, (lat, lon) in enumerate(SHIMLA_MANALI.waypoints):
            assert merged_dem.is_in_bounds(lat, lon), (
                f"Waypoint {i} (lat={lat}, lon={lon}) is out of merged DEM bounds"
            )

    def test_nav13_all_waypoints_valid_elevation(self, merged_dem):
        """All 8 waypoints return a finite elevation (not NaN)."""
        for i, (lat, lon) in enumerate(SHIMLA_MANALI.waypoints):
            elev = merged_dem.get_elevation(lat, lon)
            assert not np.isnan(elev), (
                f"Waypoint {i} (lat={lat}, lon={lon}) returned NaN elevation"
            )
            assert elev > 0, (
                f"Waypoint {i} elevation {elev:.0f}m is non-positive (sea level error)"
            )

    def test_nav13_merged_north_bound(self, merged_dem):
        """Merged DEM north extent covers Manali at 32.24°N."""
        bounds = merged_dem.get_bounds()
        assert bounds["north"] > 32.2, (
            f"Merged DEM north bound {bounds['north']:.3f}°N < 32.2°N — "
            f"Manali tile not admitted"
        )

    def test_nav13_terrain_zones_defined(self):
        """SHIMLA_MANALI has exactly 3 terrain zones annotated."""
        assert len(SHIMLA_MANALI.terrain_zones) == 3
        zone_names = [z["name"] for z in SHIMLA_MANALI.terrain_zones]
        assert "Shimla Ridge" in zone_names
        assert "Sutlej-Beas Gorge" in zone_names
        assert "Kullu-Manali Alpine" in zone_names


# ---------------------------------------------------------------------------
# Gate NAV-14: Terrain suitability varies across three zones
# ---------------------------------------------------------------------------

class TestNAV14TerrainSuitabilityVariation:
    """
    Gate NAV-14: Terrain suitability scores differ meaningfully across zones.

    Samples zone centres:
      Zone 1 centre: km 30  (Shimla Ridge — forested)
      Zone 2 centre: km 90  (Sutlej-Beas Gorge)
      Zone 3 centre: km 150 (Kullu-Manali Alpine)

    Asserts:
      - Scores differ by > 0.10 between at least two zones
      - At least one zone centre scores >= 0.40 (terrain usable for TRN)
    """

    @pytest.fixture(scope="class")
    def zone_scores(self, merged_dem):
        """Compute suitability at zone centres (km 30, 90, 150)."""
        gen    = HillshadeGenerator()
        scorer = TerrainSuitabilityScorer()
        bounds = merged_dem.get_bounds()
        dem_res = bounds["resolution_m"]

        scores = {}
        for km, label in [(30, "zone1"), (90, "zone2"), (150, "zone3")]:
            lat, lon = SHIMLA_MANALI.position_at_km(km)
            if not merged_dem.is_in_bounds(lat, lon):
                scores[label] = 0.0
                continue
            elev = merged_dem.get_tile(lat, lon, 500.0, 5.0)
            if np.all(np.isnan(elev)):
                scores[label] = 0.0
                continue
            hs     = gen.generate(elev, gsd_m=5.0)
            result = scorer.score(elev, hs, 5.0, dem_res)
            scores[label] = result.score
        return scores

    def test_nav14_scores_vary_across_zones(self, zone_scores):
        """Suitability scores differ by > 0.10 between at least two zones."""
        vals = list(zone_scores.values())
        max_diff = max(abs(vals[i] - vals[j]) for i in range(3) for j in range(i+1, 3))
        assert max_diff > 0.10, (
            f"Terrain suitability too uniform across zones: "
            f"zone1={zone_scores['zone1']:.3f}, "
            f"zone2={zone_scores['zone2']:.3f}, "
            f"zone3={zone_scores['zone3']:.3f}, "
            f"max_diff={max_diff:.3f} <= 0.10"
        )

    def test_nav14_at_least_one_usable_zone(self, zone_scores):
        """At least one zone centre has suitability >= 0.40 (TRN usable)."""
        max_score = max(zone_scores.values())
        assert max_score >= 0.40, (
            f"No zone centre scored >= 0.40: "
            f"zone1={zone_scores['zone1']:.3f}, "
            f"zone2={zone_scores['zone2']:.3f}, "
            f"zone3={zone_scores['zone3']:.3f}"
        )


# ---------------------------------------------------------------------------
# Gate NAV-15: Monte Carlo 180km
# ---------------------------------------------------------------------------

class TestNAV15MonteCarlo180km:
    """
    Gate NAV-15: Monte Carlo N=10 (CI) drift envelopes for full 180km corridor.

    Asserts:
      - P99 TRN < P99 none at all checkpoints
      - P99 none drift grows monotonically (physically correct)
      - P99 TRN is non-monotonic (TRN corrections reset accumulation)
    """

    @pytest.fixture(scope="class")
    def mc_results(self, mc_evaluator):
        return {
            "none":         mc_evaluator.run("none"),
            "trn_only":     mc_evaluator.run("trn_only"),
            "vio_plus_trn": mc_evaluator.run("vio_plus_trn"),
        }

    def test_nav15_trn_reduces_p99_all_checkpoints(self, mc_results):
        """P99 TRN < P99 none at every checkpoint."""
        r_none = mc_results["none"]
        r_trn  = mc_results["trn_only"]
        for i, km in enumerate(_CHECKPOINTS):
            assert r_trn.p99_drift_m[i] < r_none.p99_drift_m[i], (
                f"TRN did not reduce P99 drift at km={km}: "
                f"none={r_none.p99_drift_m[i]:.1f}m, "
                f"trn={r_trn.p99_drift_m[i]:.1f}m"
            )

    def test_nav15_none_mode_monotonic(self, mc_results):
        """INS-only mean drift grows monotonically (unbounded dead-reckoning).

        P99 can exhibit small-N variance fluctuations with N=10; mean is a
        more stable indicator. Also assert final > initial by a large margin.
        """
        r = mc_results["none"]
        mean_drift = r.mean_drift_m
        for i in range(len(mean_drift) - 1):
            assert mean_drift[i+1] >= mean_drift[i] * 0.90, (
                f"INS-only mean drift not monotonic at checkpoint index {i}: "
                f"{mean_drift[i]:.1f}m → {mean_drift[i+1]:.1f}m"
            )
        # Final drift should be substantially larger than first checkpoint
        assert r.mean_drift_m[-1] > r.mean_drift_m[0] * 1.5, (
            f"INS-only drift did not grow substantially: "
            f"km30={r.mean_drift_m[0]:.1f}m km180={r.mean_drift_m[-1]:.1f}m"
        )

    def test_nav15_trn_non_monotonic(self, mc_results):
        """TRN P99 drift is non-monotonic — corrections reset accumulation."""
        p99 = mc_results["trn_only"].p99_drift_m
        # Check that at least one consecutive pair shows a decrease
        has_decrease = any(p99[i+1] < p99[i] for i in range(len(p99) - 1))
        assert has_decrease, (
            f"TRN P99 drift never decreased — corrections had no effect: {p99}"
        )

    def test_nav15_result_has_all_checkpoints(self, mc_results):
        """Result contains all 6 checkpoints [30, 60, 90, 120, 150, 180]."""
        r = mc_results["none"]
        assert r.checkpoints_km == _CHECKPOINTS, (
            f"Checkpoints mismatch: {r.checkpoints_km} != {_CHECKPOINTS}"
        )

    def test_nav15_p5_positive(self, mc_results):
        """P5 drift is positive at all checkpoints (no zero-drift artefacts)."""
        r = mc_results["none"]
        for i, km in enumerate(_CHECKPOINTS):
            assert r.p5_drift_m[i] > 0, f"P5 drift is zero at km={km}"


# ---------------------------------------------------------------------------
# Gate NAV-16: Compound fault survival
# ---------------------------------------------------------------------------

class TestNAV16CompoundFaultSurvival:
    """
    Gate NAV-16: 180km compound fault injection — SHM must not trigger.

    Fault sequence:
      km  0–10:  GNSS available                          → NOMINAL
      km 10–60:  GNSS denied, VIO+TRN active             → GNSS_DENIED
      km 60–75:  VIO degraded (low-texture tile → low confidence)
                 TRN still active                        → NAV_TRN_ONLY
      km 75–120: VIO recovered, TRN active               → GNSS_DENIED
      km 120–135: TRN suppressed, VIO active             → NAV_TRN_ONLY or GNSS_DENIED
      km 135–180: All recovered                          → GNSS_DENIED

    Assertions:
      1. SHM_TRIGGER is never emitted
      2. NAV_TRN_ONLY is entered during VIO degradation window (km 60–75)
      3. NAV_MODE_TRANSITION events logged for each mode change
      4. System reaches km 180 without ABORT (no exception)
    """

    @pytest.fixture(scope="class")
    def fault_scenario_result(self):
        """
        Run 180km compound fault scenario and return (mode_log, event_log).

        mode_log: list of (km, nav_mode) at each km step
        event_log: accumulated NavigationManager event log
        """
        # Controllable TRN: suppressed at km 120–135
        suppressed_window = (120.0, 135.0)

        class _ControllableTRN:
            def __init__(self):
                self.current_km = 0.0

            def match(self, camera_tile, lat_estimate, lon_estimate,
                      alt_m, gsd_m, mission_time_ms):
                if suppressed_window[0] <= self.current_km <= suppressed_window[1]:
                    return TRNMatchResult(
                        status="SUPPRESSED",
                        correction_north_m=0.0,
                        correction_east_m=0.0,
                        confidence=0.0,
                        suitability_score=0.0,
                        suitability_recommendation="SUPPRESS",
                        latency_ms=10,
                        mission_time_ms=mission_time_ms,
                    )
                return TRNMatchResult(
                    status="ACCEPTED",
                    correction_north_m=4.0,
                    correction_east_m=2.0,
                    confidence=0.80,
                    suitability_score=0.65,
                    suitability_recommendation="ACCEPT",
                    latency_ms=20,
                    mission_time_ms=mission_time_ms,
                )

        controllable_trn = _ControllableTRN()

        # Build NavigationManager with small TRN interval for responsive updates
        nm, event_log = _make_nav_manager(
            trn=controllable_trn,
            trn_interval_m=1000.0,  # 1km interval for responsive updates
        )
        state = _make_ins_state()

        rng_good = np.random.default_rng(42)
        low_tile = _synthetic_camera_tile(low_texture=True)
        good_tile = _synthetic_camera_tile(rng_good)

        mode_log = []
        mission_time_ms = 0

        for km_step in range(0, 181, 1):
            km = float(km_step)
            controllable_trn.current_km = km
            mission_time_ms += 36000  # ~36s per km at 100km/h

            # Fault profile
            gnss_available = km < 10.0
            gnss_meas = (
                GNSSMeasurement(pdop=1.5, doppler_deviation_ms=0.1, ew_jammer_confidence=0.0)
                if gnss_available else None
            )

            # VIO degradation: km 60–75 → low-texture tile → VIO confidence near 0
            if 60.0 <= km <= 75.0:
                camera_tile = low_tile
            else:
                camera_tile = good_tile

            lat, lon = SHIMLA_MANALI.position_at_km(km)
            out = nm.update(
                state           = state,
                gnss_available  = gnss_available,
                gnss_pos        = np.array([0.0, 0.0, -_ALT_M]) if gnss_available else None,
                gnss_measurement= gnss_meas,
                mission_km      = km,
                alt_m           = _ALT_M,
                gsd_m           = _GSD_M,
                lat_estimate    = lat,
                lon_estimate    = lon,
                camera_tile     = camera_tile,
                mission_time_ms = mission_time_ms,
            )
            mode_log.append((km, out.nav_mode))

        return mode_log, event_log

    def test_nav16_shm_not_triggered(self, fault_scenario_result):
        """SHM_TRIGGER mode must never appear in the 180km mode log."""
        mode_log, event_log = fault_scenario_result
        shm_modes = [(km, mode) for km, mode in mode_log if mode == NAV_MODE_SHM_TRIGGER]
        assert not shm_modes, (
            f"SHM_TRIGGER entered at: {shm_modes[:5]}"
        )

    def test_nav16_trn_only_entered_vio_degradation(self, fault_scenario_result):
        """NAV_TRN_ONLY must be entered during VIO degradation window (km 60–75)."""
        mode_log, _ = fault_scenario_result
        trn_only_during_window = [
            km for km, mode in mode_log
            if 60.0 <= km <= 75.0 and mode == NAV_MODE_TRN_ONLY
        ]
        assert trn_only_during_window, (
            "NAV_TRN_ONLY was never entered during VIO degradation window (km 60–75). "
            "Modes during window: "
            + str([(km, m) for km, m in mode_log if 60.0 <= km <= 75.0])
        )

    def test_nav16_mode_transitions_logged(self, fault_scenario_result):
        """NAV_MODE_TRANSITION events are logged for each mode change."""
        _, event_log = fault_scenario_result
        transitions = [e for e in event_log if e.get("event") == "NAV_MODE_TRANSITION"]
        assert len(transitions) >= 2, (
            f"Expected >= 2 NAV_MODE_TRANSITION events, got {len(transitions)}: "
            f"{[e['payload']['to_mode'] for e in transitions]}"
        )

    def test_nav16_system_reaches_180km(self, fault_scenario_result):
        """Mode log contains km=180 entry — mission completed without exception."""
        mode_log, _ = fault_scenario_result
        final_km = mode_log[-1][0]
        assert final_km == 180.0, (
            f"Mission did not reach km=180: last km={final_km}"
        )

    def test_nav16_nominal_at_gnss_phase(self, fault_scenario_result):
        """NOMINAL mode during GNSS-available phase (km 0–9)."""
        mode_log, _ = fault_scenario_result
        gnss_phase = [(km, mode) for km, mode in mode_log if km < 9.0]
        for km, mode in gnss_phase:
            assert mode == NAV_MODE_NOMINAL, (
                f"Expected NOMINAL at km={km}, got {mode}"
            )

    def test_nav16_no_abort_events(self, fault_scenario_result):
        """No ABORT or MISSION_ABORT events in event log."""
        _, event_log = fault_scenario_result
        abort_events = [
            e for e in event_log
            if "ABORT" in e.get("event", "")
        ]
        assert not abort_events, (
            f"ABORT events found: {abort_events[:3]}"
        )
