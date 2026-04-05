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

import numpy as np
import pytest

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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def om_high_texture():
    """OrthophotoMatchingStub primed for high-texture terrain."""
    provider = SatelliteTileProvider(seed=0)
    return OrthophotoMatchingStub(tile_provider=provider, seed=0)


@pytest.fixture
def om_featureless():
    """OrthophotoMatchingStub primed for featureless terrain."""
    provider = SatelliteTileProvider(seed=1)
    return OrthophotoMatchingStub(tile_provider=provider, seed=1)


@pytest.fixture
def planner():
    """HybridAstar planner with a clean EW engine."""
    engine = EWEngine()
    return HybridAstar(engine)


# ---------------------------------------------------------------------------
# OM-01 — High texture terrain: match applied
# ---------------------------------------------------------------------------

def test_om01_high_texture_match_applied(om_high_texture):
    """
    OM-01: Over high-texture terrain (sigma=45 m) the stub should consistently
    accept corrections. Run 20 trials; at least 15 must be accepted to
    account for Gaussian noise around mu=0.82 with std=0.05.
    """
    om = om_high_texture
    accepted_count = 0
    for i in range(20):
        om.reset()
        result = om.update(
            pos_north_m=float(i * 5_000),
            pos_east_m=0.0,
            mission_km=float(i * 10.0),
            sigma_terrain=45.0,
        )
        if result.correction_applied:
            accepted_count += 1
            assert result.match_confidence >= OM_MATCH_THRESHOLD, (
                f"Accepted fix had confidence {result.match_confidence:.3f} < threshold"
            )

    assert accepted_count >= 15, (
        f"Expected >= 15/20 accepted over high-texture terrain, got {accepted_count}"
    )


# ---------------------------------------------------------------------------
# OM-02 — Featureless terrain: match suppressed
# ---------------------------------------------------------------------------

def test_om02_featureless_match_suppressed():
    """
    OM-02: Over featureless terrain (sigma=5 m) the stub must suppress
    corrections. mu=0.25 with std=0.05 → confidence almost always < 0.65.
    Run 30 trials; at most 2 may be accepted (tail of distribution).
    """
    # Use a fixed seed that reliably stays below threshold for sigma=5
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
            assert result.consecutive_suppressed_count >= 1

    assert suppressed_count >= 28, (
        f"Expected >= 28/30 suppressed over featureless terrain, got {suppressed_count}"
    )


# ---------------------------------------------------------------------------
# OM-03 — Three consecutive suppressions: count correct
# ---------------------------------------------------------------------------

def test_om03_three_consecutive_suppressions():
    """
    OM-03: Call update() three times in a row over featureless terrain
    (advancing mission_km by >= OM_CORRECTION_INTERVAL_MIN_KM each time).
    consecutive_suppressed_count must equal 3 on the third call.
    """
    # Use a seed reliably below threshold for sigma=5.0
    provider = SatelliteTileProvider(seed=7)
    om = OrthophotoMatchingStub(tile_provider=provider, seed=7)

    sigma = 5.0  # featureless: mu=0.25, should stay below threshold

    # Force suppression by patching the provider's rng to always return mu
    # Actually: with seed=7 and sigma=5, mu=0.25 and std=0.05, P(>0.65) < 1e-15
    # Safe to rely on this statistically.

    result1 = om.update(pos_north_m=0.0, pos_east_m=0.0, mission_km=0.0,  sigma_terrain=sigma)
    result2 = om.update(pos_north_m=0.0, pos_east_m=0.0, mission_km=3.0,  sigma_terrain=sigma)
    result3 = om.update(pos_north_m=0.0, pos_east_m=0.0, mission_km=6.0,  sigma_terrain=sigma)

    # All three should be suppressed
    assert not result1.correction_applied, "Call 1 should be suppressed"
    assert not result2.correction_applied, "Call 2 should be suppressed"
    assert not result3.correction_applied, "Call 3 should be suppressed"

    assert result3.consecutive_suppressed_count == 3, (
        f"Expected consecutive_suppressed_count=3, got {result3.consecutive_suppressed_count}"
    )


# ---------------------------------------------------------------------------
# OM-04 — Correction interval gate
# ---------------------------------------------------------------------------

def test_om04_correction_interval_gate():
    """
    OM-04: If mission_km delta is less than OM_CORRECTION_INTERVAL_MIN_KM
    since the last accepted fix, no correction must be applied regardless
    of terrain texture.
    """
    # Use high-texture terrain to ensure the first call is accepted
    provider = SatelliteTileProvider(seed=0)
    om = OrthophotoMatchingStub(tile_provider=provider, seed=0)

    # First call — no previous fix, should be accepted on high-texture
    first = om.update(
        pos_north_m=0.0,
        pos_east_m=0.0,
        mission_km=10.0,
        sigma_terrain=45.0,
    )
    assert first.correction_applied, (
        "First call on high-texture terrain should be accepted; check seed"
    )

    # Second call — delta = 0.5 km < OM_CORRECTION_INTERVAL_MIN_KM (2.0)
    delta_km = OM_CORRECTION_INTERVAL_MIN_KM - 0.5
    second = om.update(
        pos_north_m=500.0,
        pos_east_m=0.0,
        mission_km=10.0 + delta_km,
        sigma_terrain=45.0,
    )
    assert not second.correction_applied, (
        f"Correction must be gated out when delta={delta_km} km < "
        f"OM_CORRECTION_INTERVAL_MIN_KM={OM_CORRECTION_INTERVAL_MIN_KM}"
    )


# ---------------------------------------------------------------------------
# OM-05 — OMCorrection dataclass fields all present
# ---------------------------------------------------------------------------

def test_om05_omcorrection_fields_present():
    """
    OM-05: OMCorrection must contain all 9 required fields, with r_matrix
    shape (2, 2).
    """
    provider = SatelliteTileProvider(seed=0)
    om = OrthophotoMatchingStub(tile_provider=provider, seed=0)

    result = om.update(
        pos_north_m=1_000.0,
        pos_east_m=500.0,
        mission_km=5.0,
        sigma_terrain=45.0,
    )

    assert isinstance(result, OMCorrection)
    assert hasattr(result, "timestamp_s")
    assert hasattr(result, "correction_north_m")
    assert hasattr(result, "correction_east_m")
    assert hasattr(result, "match_confidence")
    assert hasattr(result, "correction_applied")
    assert hasattr(result, "consecutive_suppressed_count")
    assert hasattr(result, "om_last_fix_km_ago")
    assert hasattr(result, "sigma_terrain")
    assert hasattr(result, "r_matrix")

    assert result.r_matrix.shape == (2, 2), (
        f"r_matrix shape must be (2,2), got {result.r_matrix.shape}"
    )
    assert isinstance(result.timestamp_s, float)
    assert isinstance(result.correction_north_m, float)
    assert isinstance(result.correction_east_m, float)
    assert isinstance(result.match_confidence, float)
    assert isinstance(result.correction_applied, bool)
    assert isinstance(result.consecutive_suppressed_count, int)
    assert isinstance(result.om_last_fix_km_ago, float)
    assert isinstance(result.sigma_terrain, float)


# ---------------------------------------------------------------------------
# OM-06 — R matrix values correct (9.0**2, not old 225 m²)
# ---------------------------------------------------------------------------

def test_om06_r_matrix_values():
    """
    OM-06 (critical QA gate): r_matrix diagonal must be [OM_R_NORTH, OM_R_EAST]
    = [81.0, 81.0] (9.0**2). Old RADALT-NCC value was 15.0**2 = 225 m².
    Silently using the wrong R would underweight the OM correction.
    """
    provider = SatelliteTileProvider(seed=0)
    om = OrthophotoMatchingStub(tile_provider=provider, seed=0)

    result = om.update(
        pos_north_m=0.0,
        pos_east_m=0.0,
        mission_km=5.0,
        sigma_terrain=45.0,
    )

    # Named constants must be correct
    assert OM_R_NORTH == 9.0 ** 2, (
        f"OM_R_NORTH should be 81.0 (9²), got {OM_R_NORTH}"
    )
    assert OM_R_EAST == 9.0 ** 2, (
        f"OM_R_EAST should be 81.0 (9²), got {OM_R_EAST}"
    )

    # r_matrix diagonal must match constants
    assert result.r_matrix[0, 0] == OM_R_NORTH, (
        f"r_matrix[0,0]={result.r_matrix[0,0]:.1f} != OM_R_NORTH={OM_R_NORTH}"
    )
    assert result.r_matrix[1, 1] == OM_R_EAST, (
        f"r_matrix[1,1]={result.r_matrix[1,1]:.1f} != OM_R_EAST={OM_R_EAST}"
    )

    # Explicitly confirm not the old RADALT-NCC value
    old_r = 15.0 ** 2  # 225 m²
    assert result.r_matrix[0, 0] != old_r, "r_matrix[0,0] must not be old RADALT-NCC value 225 m²"
    assert result.r_matrix[1, 1] != old_r, "r_matrix[1,1] must not be old RADALT-NCC value 225 m²"


# ---------------------------------------------------------------------------
# OM-07 — Route planner: featureless terrain costs more than textured
# ---------------------------------------------------------------------------

def test_om07_route_planner_texture_cost(planner):
    """
    OM-07: compute_cost() with sigma=5 (featureless) must produce a higher
    cost than with sigma=40 (textured), given the same EW cost.
    """
    # We need a dummy node — compute_cost uses node only for existing logic.
    # Check the signature: it passes node to existing cost, which uses grid coords.
    # Use the same node and same ew_cost for both calls to isolate texture term.

    # Inspect the cost method by calling terrain_texture_cost directly (spec says
    # this is a module-level function) and verify TEXTURE_COST_WEIGHT is applied.
    cost_featureless = terrain_texture_cost(5.0)
    cost_textured    = terrain_texture_cost(40.0)

    assert cost_featureless > cost_textured, (
        f"Featureless cost {cost_featureless} must exceed textured cost {cost_textured}"
    )

    # Verify the expected values per spec
    assert cost_textured == 0.0,    f"High-texture cost must be 0.0, got {cost_textured}"
    assert cost_featureless == 1.0, f"Featureless cost must be 1.0, got {cost_featureless}"

    # Verify TEXTURE_COST_WEIGHT is applied in full compute_cost
    # Build a minimal test: a node with grid coords inside the grid
    from core.ew_engine.ew_engine import EWEngine, N_NORTH, N_EAST
    engine = EWEngine()
    p = HybridAstar(engine)

    class _Node:
        def __init__(self, row, col):
            self.row = row
            self.col = col

    # compute_cost(node, ew_cost, sigma_terrain)
    ew_cost = 0.0
    cost_with_featureless = p.compute_cost(_Node(10, 5), ew_cost, sigma_terrain=5.0)
    cost_with_textured    = p.compute_cost(_Node(10, 5), ew_cost, sigma_terrain=40.0)

    assert cost_with_featureless > cost_with_textured, (
        "compute_cost must produce higher cost for featureless terrain"
    )
    expected_penalty = TEXTURE_COST_WEIGHT * 1.0  # texture_cost(5.0) = 1.0
    assert abs(cost_with_featureless - cost_with_textured - expected_penalty) < 1e-9, (
        f"Expected cost difference {expected_penalty}, got "
        f"{cost_with_featureless - cost_with_textured:.6f}"
    )


# ---------------------------------------------------------------------------
# OM-08 — Featureless terrain integration, no match for 10+ km (OI-11)
# ---------------------------------------------------------------------------

def test_om08_featureless_no_fix_10km():
    """
    OM-08 (OI-11 closure gate): Run 20 consecutive updates over a purely
    featureless zone (sigma=5 m), advancing 0.7 km each step (14 km total).
    No correction must be applied throughout, and om_last_fix_km_ago
    must exceed 10.0 km by the end.

    This is the first test to exercise the featureless terrain failure mode.
    The synthetic DEM (trn_stub.py) was always textured — this scenario
    was structurally untestable before OI-05 was resolved.
    """
    provider = SatelliteTileProvider(seed=7)  # seed reliably below threshold
    om = OrthophotoMatchingStub(tile_provider=provider, seed=7)

    sigma_featureless = 5.0   # well below OM_FEATURELESS_SIGMA_THRESHOLD=10.0
    step_km = 0.7             # > OM_CORRECTION_INTERVAL_MIN_KM/3 but accumulates to 14 km
    n_steps = 20

    for i in range(n_steps):
        mission_km = i * step_km
        result = om.update(
            pos_north_m=float(i * step_km * 1_000),
            pos_east_m=0.0,
            mission_km=mission_km,
            sigma_terrain=sigma_featureless,
        )
        assert not result.correction_applied, (
            f"Step {i}: correction_applied=True over featureless terrain at "
            f"mission_km={mission_km:.1f}, confidence={result.match_confidence:.3f}"
        )

    # Final state: no fix ever achieved
    final_result = om.update(
        pos_north_m=float(n_steps * step_km * 1_000),
        pos_east_m=0.0,
        mission_km=n_steps * step_km,
        sigma_terrain=sigma_featureless,
    )

    assert not final_result.correction_applied, "Final step must also be suppressed"
    assert final_result.om_last_fix_km_ago > 10.0, (
        f"om_last_fix_km_ago={final_result.om_last_fix_km_ago:.1f} must be > 10.0 km "
        f"after {n_steps * step_km:.1f} km featureless zone with no fix"
    )
