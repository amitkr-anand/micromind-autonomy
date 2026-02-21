"""
tests/test_sprint_s3_acceptance.py
MicroMind / NanoCorteX — Sprint S3 Acceptance Gate
8 tests covering: TRN stub, nav scenario, FSM transitions, FR-107, NAV-01, dashboard

Run:
    cd micromind-autonomy
    PYTHONPATH=. python -m pytest tests/test_sprint_s3_acceptance.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make project importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ins.trn_stub import (
    DEMProvider, INSState, RadarAltimeterSim, TRNStub,
    NCC_THRESHOLD, CORRECTION_INTERVAL,
)
from sim.nav_scenario import (
    run_nav_scenario,
    GNSS_LOSS_START_M, GNSS_LOSS_END_M, FR107_DRIFT_LIMIT_M,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nav_result():
    """Run the full 50 km scenario once; reuse across tests."""
    return run_nav_scenario(seed=42, verbose=False)


@pytest.fixture(scope="module")
def trn_system():
    """Shared TRN subsystem for unit tests."""
    dem   = DEMProvider(seed=7)
    radar = RadarAltimeterSim(dem, seed=99)
    trn   = TRNStub(dem, radar)
    return dem, radar, trn


# ---------------------------------------------------------------------------
# Test 1 — TRN: NCC score above threshold for high-contrast terrain
# ---------------------------------------------------------------------------

def test_ncc_score_above_threshold(trn_system):
    """
    NCC score must be ≥ NCC_THRESHOLD (0.45) when correlating strip against
    the correct DEM location. Validates terrain matching quality.
    """
    dem, radar, trn = trn_system
    ins = INSState(north_m=500.0, east_m=300.0)
    corr = trn.update(ins,
                      true_north_m=500.0, true_east_m=300.0,
                      dt=1.0, ground_track_m=CORRECTION_INTERVAL)
    assert corr is not None, "Expected a TRN fix attempt"
    assert corr.ncc_score >= NCC_THRESHOLD, (
        f"NCC score {corr.ncc_score:.3f} < threshold {NCC_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Test 2 — TRN: Kalman correction reduces position error
# ---------------------------------------------------------------------------

def test_kalman_correction_reduces_error():
    """
    Given a 60 m INS position error, a single TRN Kalman correction shall
    reduce the error to < 10 m.
    """
    dem   = DEMProvider(seed=7)
    radar = RadarAltimeterSim(dem, seed=99)
    trn   = TRNStub(dem, radar)

    true_n, true_e = 1200.0, 800.0
    ins = INSState(north_m=true_n - 50.0, east_m=true_e + 40.0)  # 64 m initial error

    initial_error = math.hypot(ins.north_m - true_n, ins.east_m - true_e)

    corr = trn.update(ins, true_north_m=true_n, true_east_m=true_e,
                      dt=1.0, ground_track_m=CORRECTION_INTERVAL)

    assert corr is not None and corr.accepted, "TRN fix must be accepted"
    post_error = math.hypot(ins.north_m - true_n, ins.east_m - true_e)

    assert post_error < initial_error, "Kalman correction must reduce error"
    assert post_error < 10.0, (
        f"Residual error {post_error:.1f} m ≥ 10 m after correction"
    )


# ---------------------------------------------------------------------------
# Test 3 — TRN: Correction only accepted above NCC_THRESHOLD
# ---------------------------------------------------------------------------

def test_trn_rejects_low_ncc_score():
    """
    When the strip does not match terrain (flat DEM zone), NCC score should
    fall below threshold and the fix should be rejected.
    TRNStub.trn_correlation_valid must be False after rejection.
    """
    # Use a separate TRN with a patched NCC threshold set very high
    dem   = DEMProvider(seed=7)
    radar = RadarAltimeterSim(dem, seed=99)
    trn   = TRNStub(dem, radar, ncc_threshold=0.999)  # impossible threshold

    ins = INSState(north_m=100.0, east_m=100.0)
    corr = trn.update(ins, true_north_m=100.0, true_east_m=100.0,
                      dt=1.0, ground_track_m=CORRECTION_INTERVAL)

    assert corr is not None, "Update should return a correction record"
    assert not corr.accepted, "High threshold must cause rejection"
    assert not trn.trn_correlation_valid, (
        "trn_correlation_valid must be False after rejection"
    )


# ---------------------------------------------------------------------------
# Test 4 — FSM transitions: NOMINAL → EW_AWARE → GNSS_DENIED
# ---------------------------------------------------------------------------

def test_fsm_gnss_denied_transition(nav_result):
    """
    The 50 km scenario must produce the full transition chain:
    NOMINAL → EW_AWARE → GNSS_DENIED on GNSS loss event.
    S3 acceptance gate requirement.
    """
    transitions = [(t.from_state.value, t.to_state.value)
                   for t in nav_result.fsm_transitions]

    assert ("NOMINAL", "EW_AWARE") in transitions, (
        "Missing NOMINAL → EW_AWARE transition"
    )
    assert ("EW_AWARE", "GNSS_DENIED") in transitions, (
        "Missing EW_AWARE → GNSS_DENIED transition"
    )


# ---------------------------------------------------------------------------
# Test 5 — GNSS loss event detected
# ---------------------------------------------------------------------------

def test_gnss_loss_detected(nav_result):
    """
    The scenario must detect GNSS loss (gnss_loss_start_t is not None) and
    GNSS recovery (gnss_loss_end_t is not None), confirming phase execution.
    """
    assert nav_result.gnss_loss_start_t is not None, "GNSS loss not detected"
    assert nav_result.gnss_loss_end_t   is not None, "GNSS recovery not detected"
    assert nav_result.gnss_loss_start_t < nav_result.gnss_loss_end_t


# ---------------------------------------------------------------------------
# Test 6 — FR-107: Drift < 2 % over 5 km GNSS-denied segment
# ---------------------------------------------------------------------------

def test_fr107_drift_within_limit(nav_result):
    """
    FR-107: INS position error must be < 100 m (2 % of 5 000 m) at the
    end of the 5 km GNSS-denied gate segment.
    """
    assert nav_result.fr107_pass, (
        f"FR-107 FAIL: drift {nav_result.drift_at_5km_gate_m:.1f} m "
        f"> {FR107_DRIFT_LIMIT_M:.0f} m limit"
    )
    assert nav_result.drift_at_5km_gate_m <= FR107_DRIFT_LIMIT_M


# ---------------------------------------------------------------------------
# Test 7 — NAV-01: ≥ 1 TRN correction per 2 km GNSS-denied
# ---------------------------------------------------------------------------

def test_nav01_correction_frequency(nav_result):
    """
    NAV-01: TRN must deliver at least 1 accepted correction per 2 km of
    GNSS-denied flight. Over the 25 km denied segment, expect ≥ 12 corrections.
    """
    assert nav_result.nav01_pass, "NAV-01 FAIL: insufficient TRN correction frequency"

    denied_km = (GNSS_LOSS_END_M - GNSS_LOSS_START_M) / 1000.0
    min_expected = int(denied_km / 2.0)
    accepted = sum(1 for c in nav_result.trn_corrections if c.accepted)

    assert accepted >= min_expected, (
        f"NAV-01: {accepted} accepted corrections < {min_expected} required "
        f"over {denied_km:.0f} km denied segment"
    )


# ---------------------------------------------------------------------------
# Test 8 — Dashboard: scenario data fully populated
# ---------------------------------------------------------------------------

def test_dashboard_data_complete(nav_result):
    """
    Dashboard requires at minimum: position ticks, BIM trust series,
    FSM state series, and drift series — all non-empty and consistent length.
    """
    np_data = nav_result.ticks_np

    assert len(nav_result.ticks) > 0, "No simulation ticks recorded"

    n = len(np_data["time_s"])
    for key in ("ground_track_m", "bim_trust", "fsm_state", "drift_m",
                "true_north_m", "ins_north_m", "trn_ncc_score"):
        assert len(np_data[key]) == n, (
            f"Dashboard data length mismatch for '{key}': "
            f"{len(np_data[key])} ≠ {n}"
        )

    # Verify GNSS-denied ticks are present and tagged VIO_TRN
    denied_ticks = [t for t in nav_result.ticks if t.nav_mode == "VIO_TRN"]
    assert len(denied_ticks) > 0, "No VIO_TRN ticks in denied phase"

    # Verify total ground track ≥ 50 km
    assert np_data["ground_track_m"][-1] >= 50_000.0, (
        "Total ground track < 50 km"
    )
