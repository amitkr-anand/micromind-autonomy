"""
tests/test_sprint_c_om_stub.py
MicroMind / NanoCorteX — Sprint C Acceptance Gate

8 tests covering OrthophotoMatchingStub (OI-05) and terrain texture
cost term in HybridAstar (OI-08), including featureless terrain
failure mode (OI-11).

Acceptance gates:
  OM-01  High texture terrain — match applied
  OM-02  Featureless terrain — match suppressed
  OM-03  Three consecutive suppressions — count correct
  OM-04  Correction interval gate — no correction before min interval
  OM-05  OMCorrection dataclass fields all present
  OM-06  R matrix values correct (9.0**2, not old 15.0**2 = 225 m²)
  OM-07  Featureless terrain — route planner cost higher than textured
  OM-08  Featureless terrain integration — no match for 10+ km (OI-11)
"""

from __future__ import annotations

import unittest

import numpy as np

from core.ins.orthophoto_matching_stub import (
    OrthophotoMatchingStub,
    OMCorrection,
    SatelliteTileProvider,
    OM_MATCH_THRESHOLD,
    OM_CORRECTION_INTERVAL_MIN_KM,
    OM_FEATURELESS_SIGMA_THRESHOLD,
    OM_PREFERRED_SIGMA_THRESHOLD,
    OM_R_NORTH,
    OM_R_EAST,
)
from core.route_planner.hybrid_astar import (
    HybridAstar,
    TEXTURE_COST_WEIGHT,
    terrain_texture_cost,
)
from core.ew_engine.ew_engine import EWEngine


# ---------------------------------------------------------------------------
# Suite 1 — OrthophotoMatchingStub core behaviour (OM-01 through OM-06)
# ---------------------------------------------------------------------------

class TestOMStubCore(unittest.TestCase):
    """OM-01 through OM-06: stub correctness, confidence model, R matrix."""

    def _make_stub(self, seed: int = 0) -> OrthophotoMatchingStub:
        provider = SatelliteTileProvider(seed=seed)
        return OrthophotoMatchingStub(tile_provider=provider, seed=seed)

    def test_om01_high_texture_match_applied(self):
        """
        OM-01: Over high-texture terrain (sigma=45 m) the stub should
        consistently accept corrections. Run 20 trials; at least 15 must
        be accepted to account for Gaussian noise around mu=0.82 std=0.05.
        """
        accepted_count = 0
        for i in range(20):
            om = self._make_stub(seed=i)
            result = om.update(
                pos_north_m=float(i * 5_000),
                pos_east_m=0.0,
                mission_km=float(i * 10.0),
                sigma_terrain=45.0,
            )
            if result.correction_applied:
                accepted_count += 1
                self.assertGreaterEqual(
                    result.match_confidence, OM_MATCH_THRESHOLD,
                    f"Accepted fix had confidence {result.match_confidence:.3f} < threshold",
                )

        self.assertGreaterEqual(
            accepted_count, 15,
            f"Expected >= 15/20 accepted over high-texture terrain, got {accepted_count}",
        )

    def test_om02_featureless_match_suppressed(self):
        """
        OM-02: Over featureless terrain (sigma=5 m) the stub must suppress
        corrections. mu=0.25 std=0.05 → confidence almost always < 0.65.
        30 trials; at most 2 may be accepted (tail of distribution).
        """
        provider = SatelliteTileProvider(seed=7)
        om = OrthophotoMatchingStub(tile_provider=provider, seed=7)

        suppressed_count = 0
        for i in range(30):
            om.reset()
            result = om.update(
                pos_north_m=float(i * 5_000),
                pos_east_m=0.0,
                mission_km=float(i * 10.0),
                sigma_terrain=5.0,
            )
            if not result.correction_applied:
                suppressed_count += 1
                self.assertGreaterEqual(result.consecutive_suppressed_count, 1)

        self.assertGreaterEqual(
            suppressed_count, 28,
            f"Expected >= 28/30 suppressed over featureless terrain, got {suppressed_count}",
        )

    def test_om03_three_consecutive_suppressions(self):
        """
        OM-03: Three consecutive updates over featureless terrain must yield
        consecutive_suppressed_count == 3 on the third call.
        """
        provider = SatelliteTileProvider(seed=7)
        om = OrthophotoMatchingStub(tile_provider=provider, seed=7)
        sigma = 5.0  # featureless: mu=0.25, P(>0.65) negligible

        r1 = om.update(pos_north_m=0.0, pos_east_m=0.0, mission_km=0.0, sigma_terrain=sigma)
        r2 = om.update(pos_north_m=0.0, pos_east_m=0.0, mission_km=3.0, sigma_terrain=sigma)
        r3 = om.update(pos_north_m=0.0, pos_east_m=0.0, mission_km=6.0, sigma_terrain=sigma)

        self.assertFalse(r1.correction_applied, "Call 1 should be suppressed")
        self.assertFalse(r2.correction_applied, "Call 2 should be suppressed")
        self.assertFalse(r3.correction_applied, "Call 3 should be suppressed")
        self.assertEqual(
            r3.consecutive_suppressed_count, 3,
            f"Expected consecutive_suppressed_count=3, got {r3.consecutive_suppressed_count}",
        )

    def test_om04_correction_interval_gate(self):
        """
        OM-04: If mission_km delta is less than OM_CORRECTION_INTERVAL_MIN_KM
        since the last accepted fix, no correction must be applied regardless
        of terrain texture.
        """
        om = self._make_stub(seed=0)

        first = om.update(
            pos_north_m=0.0, pos_east_m=0.0,
            mission_km=10.0, sigma_terrain=45.0,
        )
        self.assertTrue(first.correction_applied,
                        "First call on high-texture terrain should be accepted")

        delta_km = OM_CORRECTION_INTERVAL_MIN_KM - 0.5
        second = om.update(
            pos_north_m=500.0, pos_east_m=0.0,
            mission_km=10.0 + delta_km, sigma_terrain=45.0,
        )
        self.assertFalse(
            second.correction_applied,
            f"Correction must be gated out when delta={delta_km} km < "
            f"OM_CORRECTION_INTERVAL_MIN_KM={OM_CORRECTION_INTERVAL_MIN_KM}",
        )

    def test_om05_omcorrection_fields_present(self):
        """
        OM-05: OMCorrection must contain all required fields; r_matrix shape (2,2).
        """
        om = self._make_stub(seed=0)
        result = om.update(
            pos_north_m=1_000.0, pos_east_m=500.0,
            mission_km=5.0, sigma_terrain=45.0,
        )

        self.assertIsInstance(result, OMCorrection)
        for field in ("timestamp_s", "correction_north_m", "correction_east_m",
                      "match_confidence", "correction_applied",
                      "consecutive_suppressed_count", "om_last_fix_km_ago",
                      "sigma_terrain", "r_matrix"):
            self.assertTrue(hasattr(result, field), f"Missing field: {field}")

        self.assertEqual(result.r_matrix.shape, (2, 2),
                         f"r_matrix shape must be (2,2), got {result.r_matrix.shape}")
        self.assertIsInstance(result.timestamp_s, float)
        self.assertIsInstance(result.correction_north_m, float)
        self.assertIsInstance(result.correction_east_m, float)
        self.assertIsInstance(result.match_confidence, float)
        self.assertIsInstance(result.correction_applied, bool)
        self.assertIsInstance(result.consecutive_suppressed_count, int)
        self.assertIsInstance(result.om_last_fix_km_ago, float)
        self.assertIsInstance(result.sigma_terrain, float)

    def test_om06_r_matrix_values(self):
        """
        OM-06 (critical QA gate): r_matrix diagonal must be [OM_R_NORTH, OM_R_EAST]
        = [81.0, 81.0] (9.0**2). Old RADALT-NCC value was 15.0**2 = 225 m².
        """
        om = self._make_stub(seed=0)
        result = om.update(
            pos_north_m=0.0, pos_east_m=0.0,
            mission_km=5.0, sigma_terrain=45.0,
        )

        self.assertEqual(OM_R_NORTH, 9.0 ** 2,
                         f"OM_R_NORTH should be 81.0, got {OM_R_NORTH}")
        self.assertEqual(OM_R_EAST, 9.0 ** 2,
                         f"OM_R_EAST should be 81.0, got {OM_R_EAST}")
        self.assertEqual(result.r_matrix[0, 0], OM_R_NORTH,
                         f"r_matrix[0,0]={result.r_matrix[0,0]:.1f} != OM_R_NORTH={OM_R_NORTH}")
        self.assertEqual(result.r_matrix[1, 1], OM_R_EAST,
                         f"r_matrix[1,1]={result.r_matrix[1,1]:.1f} != OM_R_EAST={OM_R_EAST}")

        old_r = 15.0 ** 2  # 225 m²
        self.assertNotEqual(result.r_matrix[0, 0], old_r,
                            "r_matrix[0,0] must not be old RADALT-NCC value 225 m²")
        self.assertNotEqual(result.r_matrix[1, 1], old_r,
                            "r_matrix[1,1] must not be old RADALT-NCC value 225 m²")


# ---------------------------------------------------------------------------
# Suite 2 — Route planner terrain texture cost (OM-07)
# ---------------------------------------------------------------------------

class TestOMRoutePlanner(unittest.TestCase):
    """OM-07: terrain_texture_cost and compute_cost integration."""

    def setUp(self):
        self.engine = EWEngine()
        self.planner = HybridAstar(self.engine)

    def test_om07_route_planner_texture_cost(self):
        """
        OM-07: compute_cost() with sigma=5 (featureless) must produce a higher
        cost than with sigma=40 (textured), given the same EW cost.
        Expected difference: TEXTURE_COST_WEIGHT * 1.0 = 2.0.
        """
        cost_featureless = terrain_texture_cost(5.0)
        cost_textured    = terrain_texture_cost(40.0)

        self.assertGreater(cost_featureless, cost_textured,
                           f"Featureless cost {cost_featureless} must exceed textured {cost_textured}")
        self.assertEqual(cost_textured,    0.0, "High-texture cost must be 0.0")
        self.assertEqual(cost_featureless, 1.0, "Featureless cost must be 1.0")

        class _Node:
            def __init__(self, row, col):
                self.row = row
                self.col = col

        ew_cost = 0.0
        cost_fl = self.planner.compute_cost(_Node(10, 5), ew_cost, sigma_terrain=5.0)
        cost_tx = self.planner.compute_cost(_Node(10, 5), ew_cost, sigma_terrain=40.0)

        self.assertGreater(cost_fl, cost_tx,
                           "compute_cost must produce higher cost for featureless terrain")

        expected_diff = TEXTURE_COST_WEIGHT * 1.0
        self.assertAlmostEqual(
            cost_fl - cost_tx, expected_diff, places=9,
            msg=f"Expected cost difference {expected_diff}, got {cost_fl - cost_tx:.6f}",
        )


# ---------------------------------------------------------------------------
# Suite 3 — Featureless terrain integration (OM-08 / OI-11)
# ---------------------------------------------------------------------------

class TestOMFeaturelessIntegration(unittest.TestCase):
    """OM-08: featureless terrain failure mode over 10+ km (OI-11 closure gate)."""

    def test_om08_featureless_no_fix_10km(self):
        """
        OM-08 (OI-11 closure gate): 20 consecutive updates over a purely
        featureless zone (sigma=5 m), 0.7 km per step (14 km total).
        No correction must be applied throughout; om_last_fix_km_ago > 10.0
        at the end.

        This is the first test to exercise the featureless terrain failure
        mode — structurally untestable before OI-05 was resolved.
        """
        provider = SatelliteTileProvider(seed=7)
        om = OrthophotoMatchingStub(tile_provider=provider, seed=7)

        sigma_featureless = 5.0
        step_km = 0.7
        n_steps = 20

        for i in range(n_steps):
            mission_km = i * step_km
            result = om.update(
                pos_north_m=float(i * step_km * 1_000),
                pos_east_m=0.0,
                mission_km=mission_km,
                sigma_terrain=sigma_featureless,
            )
            self.assertFalse(
                result.correction_applied,
                f"Step {i}: correction_applied=True over featureless terrain at "
                f"mission_km={mission_km:.1f}, confidence={result.match_confidence:.3f}",
            )

        final = om.update(
            pos_north_m=float(n_steps * step_km * 1_000),
            pos_east_m=0.0,
            mission_km=n_steps * step_km,
            sigma_terrain=sigma_featureless,
        )
        self.assertFalse(final.correction_applied,
                         "Final step must also be suppressed")
        self.assertGreater(
            final.om_last_fix_km_ago, 10.0,
            f"om_last_fix_km_ago={final.om_last_fix_km_ago:.1f} must be > 10.0 km "
            f"after {n_steps * step_km:.1f} km featureless zone with no fix",
        )
