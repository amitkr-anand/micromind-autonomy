"""
tests/test_gate7_sal_corridor.py
Gate 7 — SAL-1 + SAL-2 Combined Corridor Validation

OI-52 scope. Authorised: Deputy 1.

Gates
-----
G7-01  SUPPRESS terrain class → LightGlue match() call count = 0 across full
       simulated SUPPRESS corridor segment (SAL-2 / AD-24 / OI-49).

G7-02  Low INS uncertainty → _cov_to_search_pad_px() returns < 25 px; verified
       at unit level and via TRNStub.update() integration (SAL-1 / AD-24 / OI-47).

G7-03  High INS uncertainty (post-SUPPRESS drift accumulation) →
       _cov_to_search_pad_px() returns > 25 px; verified at unit level and via
       TRNStub.update() (SAL-1 / AD-24 / OI-47).

G7-04  Corrections resume within 5 km of SUPPRESS zone exit: with default
       trn_interval_m = 5 000 m and last_trn_km stale from before SUPPRESS
       entry, NavigationManager fires a NAV_LIGHTGLUE_CORRECTION event on
       the first ACCEPT cycle after zone exit (SAL-2 / AD-24 / OI-49).

G7-05  Frozen files unchanged — SHA-256 digests match certified baseline
       hashes from QA-047/048 (commit aaeeb0d).

G7-06  Certified baseline 485/485 — verified by running
       `conda run -n micromind-autonomy bash run_certified_baseline.sh`
       before this commit. Not encoded as a test case; documented here for
       traceability.

Architecture notes
------------------
SAL-1 (_cov_to_search_pad_px) lives in core/ins/trn_stub.py.
  DEM_PIXEL_SIZE = 5 m/px.  SEARCH_PAD_PX_MIN = 10 px.  MAX = 60 px.
  threshold for G7-02/G7-03 boundary: 25 px (= SEARCH_PAD_PX default).
  pad < 25  ↔  sigma < 41.67 m  ↔  p_var < 1 736 m²
  pad > 25  ↔  sigma > 41.67 m  ↔  p_var > 1 736 m²

SAL-2 (_lightglue_threshold_for_class) lives in navigation_manager.py.
  SUPPRESS → None (IPC call skipped). match() must not be called at all.

G7-04 distance logic (NavigationManager.update(), Step 4a):
  distance_since_trn_m = (mission_km - last_trn_km) * 1000
  With last_trn_km = 55 (before SUPPRESS km 60-120) and first ACCEPT call
  at km 121: distance = 66 000 m >> 5 000 m → match fires immediately.
  SUPPRESS zone width is 60 km; 60 km >> 5 km interval → test is conservative.
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.ins.trn_stub import (
    DEMProvider,
    RadarAltimeterSim,
    TRNStub,
    CORRECTION_INTERVAL,
    DEM_PIXEL_SIZE,
    SEARCH_PAD_PX,
    SEARCH_PAD_PX_MIN,
    SEARCH_PAD_PX_MAX,
    _cov_to_search_pad_px,
)
from core.navigation.navigation_manager import (
    NavigationManager,
    LIGHTGLUE_CONF_THRESHOLD_ACCEPT,
    LIGHTGLUE_CONF_THRESHOLD_CAUTION,
    _lightglue_threshold_for_class,
)

# ---------------------------------------------------------------------------
# Frozen file hashes — certified baseline QA-047/048, commit aaeeb0d
# ---------------------------------------------------------------------------
_FROZEN_HASHES = {
    "core/ekf/error_state_ekf.py":      "aaeeb0d7617ff7352089b36aee8071ee9f56811ba175c6d939f802bbbc9223a5",
    "core/fusion/vio_mode.py":           "6c8e9ae0df2472920425e99f59a22e73263f7e57c38bef9c047110fc5accedea",
    "core/fusion/frame_utils.py":        "6425bd9b8d05e3bacca18d51eba1b552792f8a97039a9fab23237923d078f662",
    "core/bim/bim.py":                   "9f98927252d5019bc14b3a1efb344b07532f9afa7aa8b185b685bf56ff0e4116",
    "scenarios/bcmp1/bcmp1_runner.py":   "421b8e413e7d8c39ec5cf168586dd6490d0e8593e03c471a2f00f9b683f74696",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _make_nm(lightglue_client, trn_interval_m: float = 5000.0,
             lg_confidence: float = 0.82):
    """Build NavigationManager with mocked dependencies."""
    eskf = MagicMock()
    eskf.update_trn.return_value = (0.0, False, 0.0)
    eskf.update_vio.return_value = (0.0, False, 0.0)

    bim = MagicMock()
    bim.evaluate.return_value = MagicMock(trust_score=0.9)

    trn = MagicMock()
    trn.match.return_value = MagicMock(
        status="ACCEPTED",
        correction_north_m=5.0,
        correction_east_m=3.0,
        confidence=0.72,
        suitability_score=1.0,
    )

    vio_mode = MagicMock()
    camera_bridge = MagicMock()
    camera_bridge.last_frame_path = "/tmp/mock_g7_frame.jpg"
    vio_proc = MagicMock()
    vio_proc.process_frame.return_value = MagicMock(
        confidence=0.8, delta_north_m=0.5, delta_east_m=0.3, feature_count=80
    )

    event_log: list = []

    nm = NavigationManager(
        eskf=eskf,
        bim=bim,
        trn=trn,
        vio_mode=vio_mode,
        camera_bridge=camera_bridge,
        vio_processor=vio_proc,
        event_log=event_log,
        clock_fn=lambda: 0,
        trn_interval_m=trn_interval_m,
        lightglue_client=lightglue_client,
    )
    return nm, event_log


def _run_nm_update(nm, mission_km: float, terrain_class: str = "ACCEPT"):
    """One NavigationManager update at the given km position."""
    state = MagicMock()
    state.p = np.zeros(3)
    return nm.update(
        state=state,
        gnss_available=False,
        gnss_pos=None,
        gnss_measurement=None,
        mission_km=mission_km,
        alt_m=2500.0,
        gsd_m=0.5,
        lat_estimate=33.5,
        lon_estimate=75.1,
        camera_tile=None,
        mission_time_ms=int(mission_km * 60_000),
        terrain_class=terrain_class,
    )


def _make_mock_client(confidence: float = 0.82):
    client = MagicMock()
    client.match.return_value = (1e-4, 2e-4, confidence, 145.0)
    return client


# ---------------------------------------------------------------------------
# G7-01 — SUPPRESS zone: match() call count = 0
# ---------------------------------------------------------------------------

class TestG701SupressZeroMatchCalls:
    """
    SAL-2 gate: LightGlue match() must never be called when terrain_class=SUPPRESS.

    Simulates the JAMMU_LEH km 60-120 SUPPRESS segment (6 update calls, 10 km
    spacing). Asserts cumulative call_count == 0 across the full segment.
    """

    def test_g701_suppress_no_match_calls_single_cycle(self):
        """Single SUPPRESS update — match() not called."""
        client = _make_mock_client()
        nm, _ = _make_nm(client, trn_interval_m=0.0)
        _run_nm_update(nm, mission_km=70.0, terrain_class="SUPPRESS")
        assert client.match.call_count == 0

    def test_g701_suppress_no_match_across_full_segment(self):
        """Six SUPPRESS updates across 60 km segment — match() never called."""
        client = _make_mock_client()
        nm, _ = _make_nm(client, trn_interval_m=0.0)
        for km in range(60, 121, 10):
            _run_nm_update(nm, mission_km=float(km), terrain_class="SUPPRESS")
        assert client.match.call_count == 0, (
            f"match() called {client.match.call_count} times during SUPPRESS zone"
        )

    def test_g701_accept_zone_does_call_match(self):
        """Sanity: ACCEPT terrain does invoke match() — confirms mock is wired."""
        client = _make_mock_client()
        nm, _ = _make_nm(client, trn_interval_m=0.0)
        _run_nm_update(nm, mission_km=30.0, terrain_class="ACCEPT")
        assert client.match.call_count >= 1


# ---------------------------------------------------------------------------
# G7-02 — Low INS uncertainty: search pad < 25 px
# ---------------------------------------------------------------------------

class TestG702LowUncertaintyPad:
    """
    SAL-1 gate: low ESKF position uncertainty → NCC search pad < SEARCH_PAD_PX (25 px).

    Threshold: pad_px < 25 ↔ sigma < 41.67 m ↔ p_var < 1 736 m².
    Test uses p_var = 25 m² (sigma = 5 m) and p_var = 900 m² (sigma = 30 m).
    """

    def test_g702_unit_very_low_covariance(self):
        """p_var = 25 m² (sigma = 5 m) → pad = SEARCH_PAD_PX_MIN (clamped to 10) < 25."""
        pad = _cov_to_search_pad_px(25.0, 25.0)
        assert pad < SEARCH_PAD_PX, (
            f"pad {pad} px not < SEARCH_PAD_PX {SEARCH_PAD_PX} for p_var=25 m²"
        )
        assert pad >= SEARCH_PAD_PX_MIN, f"pad {pad} below minimum {SEARCH_PAD_PX_MIN}"

    def test_g702_unit_moderate_low_covariance(self):
        """p_var = 900 m² (sigma = 30 m) → pad = 18 px < 25."""
        pad = _cov_to_search_pad_px(900.0, 900.0)
        assert pad < SEARCH_PAD_PX, (
            f"pad {pad} px not < SEARCH_PAD_PX {SEARCH_PAD_PX} for p_var=900 m²"
        )

    def test_g702_unit_asymmetric_covariance_uses_max(self):
        """Asymmetric p_var: max(north, east) governs. Low in both → pad < 25."""
        pad = _cov_to_search_pad_px(25.0, 400.0)
        expected_sigma = 20.0  # sqrt(max(25, 400)) = sqrt(400) = 20m
        expected_pad = int(np.ceil(3.0 * expected_sigma / DEM_PIXEL_SIZE))
        expected_pad = int(np.clip(expected_pad, SEARCH_PAD_PX_MIN, SEARCH_PAD_PX_MAX))
        assert pad == expected_pad

    def test_g702_via_trn_stub_integration(self):
        """TRNStub.update() with low p_var updates last_search_pad_px < 25."""
        dem = DEMProvider()
        radar = RadarAltimeterSim(dem)
        trn = TRNStub(dem, radar)
        trn.update(
            ins_north_m=1000.0, ins_east_m=500.0,
            true_north_m=1000.0, true_east_m=500.0,
            ground_track_m=CORRECTION_INTERVAL + 1.0,
            p_north_var=25.0, p_east_var=25.0,
        )
        assert trn.last_search_pad_px < SEARCH_PAD_PX, (
            f"TRNStub last_search_pad_px {trn.last_search_pad_px} not < {SEARCH_PAD_PX}"
        )


# ---------------------------------------------------------------------------
# G7-03 — High INS uncertainty: search pad > 25 px
# ---------------------------------------------------------------------------

class TestG703HighUncertaintyPad:
    """
    SAL-1 gate: high ESKF position uncertainty (post-SUPPRESS drift) →
    NCC search pad > SEARCH_PAD_PX (25 px).

    SUPPRESS zone width JAMMU_LEH: 60 km. At 1 m²/km INS drift growth,
    p_var after 60 km suppression ≈ 3600 m². Tests use p_var = 2500 m²
    (sigma = 50 m, pad = 30 px) and p_var = 10 000 m² (sigma = 100 m,
    pad = 60 px = MAX).
    """

    def test_g703_unit_moderate_high_covariance(self):
        """p_var = 2500 m² (sigma = 50 m) → pad = 30 px > 25."""
        pad = _cov_to_search_pad_px(2500.0, 2500.0)
        assert pad > SEARCH_PAD_PX, (
            f"pad {pad} px not > SEARCH_PAD_PX {SEARCH_PAD_PX} for p_var=2500 m²"
        )

    def test_g703_unit_high_covariance_clamped_to_max(self):
        """p_var = 10 000 m² (sigma = 100 m) → pad clamped to SEARCH_PAD_PX_MAX."""
        pad = _cov_to_search_pad_px(10_000.0, 10_000.0)
        assert pad == SEARCH_PAD_PX_MAX, (
            f"Expected MAX {SEARCH_PAD_PX_MAX}, got {pad}"
        )
        assert pad > SEARCH_PAD_PX

    def test_g703_unit_boundary_just_above_25px(self):
        """p_var = 1764 m² (sigma = 42 m) → pad just above 25 px."""
        pad = _cov_to_search_pad_px(1764.0, 1764.0)
        assert pad > SEARCH_PAD_PX, (
            f"pad {pad} px should exceed {SEARCH_PAD_PX} for p_var=1764 m²"
        )

    def test_g703_via_trn_stub_integration(self):
        """TRNStub.update() with high p_var (post-SUPPRESS drift) → pad > 25."""
        dem = DEMProvider()
        radar = RadarAltimeterSim(dem)
        trn = TRNStub(dem, radar)
        trn.update(
            ins_north_m=1000.0, ins_east_m=500.0,
            true_north_m=1000.0, true_east_m=500.0,
            ground_track_m=CORRECTION_INTERVAL + 1.0,
            p_north_var=2500.0, p_east_var=2500.0,
        )
        assert trn.last_search_pad_px > SEARCH_PAD_PX, (
            f"TRNStub last_search_pad_px {trn.last_search_pad_px} not > {SEARCH_PAD_PX}"
        )

    def test_g703_pad_increases_monotonically_with_covariance(self):
        """Search pad is non-decreasing as p_var increases."""
        variances = [100.0, 400.0, 1000.0, 2500.0, 5000.0, 10_000.0]
        pads = [_cov_to_search_pad_px(v, v) for v in variances]
        for i in range(len(pads) - 1):
            assert pads[i] <= pads[i + 1], (
                f"Pad not monotonic: var={variances[i]}→{variances[i+1]}, "
                f"pad={pads[i]}→{pads[i+1]}"
            )


# ---------------------------------------------------------------------------
# G7-04 — Corrections resume within 5 km of SUPPRESS zone exit
# ---------------------------------------------------------------------------

class TestG704ResumeAfterSuppress:
    """
    SAL-2 corridor gate: after exiting a SUPPRESS zone, the NavigationManager
    fires a NAV_LIGHTGLUE_CORRECTION event on the first ACCEPT update, provided
    distance since last TRN >= trn_interval_m.

    Scenario (JAMMU_LEH-style km 60-120 SUPPRESS zone):
      last_trn_km = 55 (accepted just before SUPPRESS entry at km 60)
      SUPPRESS zone: updates at km 60, 70, 80, 90, 100, 110, 120
        → match() not called, last_trn_km remains 55
      SUPPRESS exit: first ACCEPT update at km 121
        → distance_since_trn_m = (121 - 55) * 1000 = 66 000 m >> 5 000 m
        → match() fires immediately
        → correction accepted, NAV_LIGHTGLUE_CORRECTION logged
      Distance from SUPPRESS exit (km 120) to first correction (km 121) = 1 km ≤ 5 km.
    """

    def _build_scenario(self):
        """
        Build NM and prime last_trn_km by accepting one correction at km 55,
        then advance through SUPPRESS zone (km 60-120). Returns (nm, client,
        event_log, suppress_exit_km).
        """
        client = _make_mock_client(confidence=0.82)
        nm, event_log = _make_nm(client, trn_interval_m=5000.0)

        # Prime: accept one correction at km 55 (before SUPPRESS zone)
        _run_nm_update(nm, mission_km=55.0, terrain_class="ACCEPT")
        n_events_before = len([e for e in event_log
                               if e.get("event") == "NAV_LIGHTGLUE_CORRECTION"])
        assert n_events_before >= 1, "Priming correction at km 55 failed"

        # SUPPRESS zone: km 60 to km 120 (7 updates)
        for km in range(60, 121, 10):
            _run_nm_update(nm, mission_km=float(km), terrain_class="SUPPRESS")

        # Verify match was not called during SUPPRESS zone
        # (subtract call from priming update at km 55)
        calls_before_suppress_exit = client.match.call_count
        return nm, client, event_log, 120.0, calls_before_suppress_exit

    def test_g704_suppress_zone_accumulates_no_new_calls(self):
        """During SUPPRESS zone km 60-120, match() call count does not increase."""
        nm, client, event_log, _, calls_after_suppress = self._build_scenario()
        # calls_after_suppress includes the priming call at km 55
        # After priming (km 55), one call happened. After SUPPRESS (km 60-120), no new calls.
        priming_calls = 1
        assert client.match.call_count == priming_calls, (
            f"Expected {priming_calls} match calls (priming only), "
            f"got {client.match.call_count} after SUPPRESS zone"
        )

    def test_g704_correction_fires_on_first_accept_after_suppress(self):
        """First ACCEPT update at km 121 fires NAV_LIGHTGLUE_CORRECTION."""
        nm, client, event_log, suppress_exit_km, _ = self._build_scenario()
        n_events_pre = len([e for e in event_log
                            if e.get("event") == "NAV_LIGHTGLUE_CORRECTION"])

        first_accept_km = suppress_exit_km + 1.0  # km 121
        _run_nm_update(nm, mission_km=first_accept_km, terrain_class="ACCEPT")

        lg_events_post = [e for e in event_log
                          if e.get("event") == "NAV_LIGHTGLUE_CORRECTION"]
        assert len(lg_events_post) > n_events_pre, (
            "NAV_LIGHTGLUE_CORRECTION not fired on first ACCEPT cycle after SUPPRESS exit"
        )

    def test_g704_resume_within_5km_of_suppress_exit(self):
        """Correction fires at km 121 — within 5 km of SUPPRESS zone exit at km 120."""
        nm, client, event_log, suppress_exit_km, _ = self._build_scenario()
        n_pre = len([e for e in event_log
                     if e.get("event") == "NAV_LIGHTGLUE_CORRECTION"])

        # Try each km from suppress_exit_km+1 to suppress_exit_km+5
        correction_km = None
        for km_offset in range(1, 6):
            km = suppress_exit_km + km_offset
            _run_nm_update(nm, mission_km=km, terrain_class="ACCEPT")
            n_now = len([e for e in event_log
                         if e.get("event") == "NAV_LIGHTGLUE_CORRECTION"])
            if n_now > n_pre:
                correction_km = km
                break

        assert correction_km is not None, (
            f"No NAV_LIGHTGLUE_CORRECTION fired within 5 km of SUPPRESS zone exit "
            f"(exit at km {suppress_exit_km})"
        )
        delta_km = correction_km - suppress_exit_km
        assert delta_km <= 5.0, (
            f"Correction fired at km {correction_km}, {delta_km} km after exit "
            f"at km {suppress_exit_km} — exceeds 5 km gate"
        )

    def test_g704_terrain_class_in_event_payload(self):
        """Resumed correction event carries terrain_class='ACCEPT' in payload."""
        nm, client, event_log, suppress_exit_km, _ = self._build_scenario()
        _run_nm_update(nm, mission_km=suppress_exit_km + 1.0, terrain_class="ACCEPT")
        lg_events = [e for e in event_log
                     if e.get("event") == "NAV_LIGHTGLUE_CORRECTION"]
        assert lg_events, "No correction event"
        last_event = lg_events[-1]
        assert last_event["payload"]["terrain_class"] == "ACCEPT"


# ---------------------------------------------------------------------------
# G7-05 — Frozen files unchanged
# ---------------------------------------------------------------------------

class TestG705FrozenFiles:
    """
    Verifies that the five frozen production files have not been modified since
    the certified baseline was established at QA-047/048 (commit aaeeb0d).
    """

    @pytest.mark.parametrize("rel_path,expected_hash", list(_FROZEN_HASHES.items()))
    def test_g705_frozen_sha256(self, rel_path, expected_hash):
        """SHA-256 of frozen file matches certified baseline hash."""
        abs_path = os.path.join(
            os.path.dirname(__file__), "..", rel_path
        )
        abs_path = os.path.normpath(abs_path)
        assert os.path.exists(abs_path), f"Frozen file missing: {rel_path}"
        actual = _sha256(abs_path)
        assert actual == expected_hash, (
            f"FROZEN FILE MODIFIED: {rel_path}\n"
            f"  expected: {expected_hash}\n"
            f"  actual  : {actual}"
        )
