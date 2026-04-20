"""
tests/test_navigation_manager_lightglue.py
OI-50: SIL gate test for NavigationManager LightGlue integration path.

Tests the lightglue_client wiring in NavigationManager without Orin hardware.
Uses a mock lightglue_client to exercise the OI-48/49 code paths in SIL.

Gates:
  NM-LG-01: NavigationManager accepts lightglue_client=None (backward compat)
  NM-LG-02: NavigationManager accepts a mock lightglue_client without exception
  NM-LG-03: Mock client returning valid MatchResult fires NAV_LIGHTGLUE_CORRECTION
  NM-LG-04: SUPPRESS terrain class — match() never called (call_count = 0)
  NM-LG-05: CAUTION terrain class — threshold 0.40 applied (low-conf result rejected)
  NM-LG-06: _lightglue_threshold_for_class() returns correct values for all classes
"""
from __future__ import annotations

import math
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from core.navigation.navigation_manager import (
    NavigationManager,
    _lightglue_threshold_for_class,
    LIGHTGLUE_CONF_THRESHOLD_ACCEPT,
    LIGHTGLUE_CONF_THRESHOLD_CAUTION,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_nav_manager(lightglue_client=None):
    """
    Construct a NavigationManager with all dependencies mocked.
    Only lightglue_client is real (or None).
    """
    eskf       = MagicMock()
    eskf.update_gnss.return_value = None
    eskf.update_vio.return_value  = (0.5, False, 0.1)
    eskf.update_trn.return_value  = (0.5, False, 0.1)
    eskf.inject.return_value      = None

    bim        = MagicMock()
    bim.evaluate.return_value     = MagicMock(trust_score=0.9)

    trn        = MagicMock()
    trn.match.return_value        = MagicMock(
        status="ACCEPTED",
        correction_north_m=5.0,
        correction_east_m=3.0,
        confidence=0.72,
        suitability_score=1.0,
    )

    vio_mode   = MagicMock()
    camera_bridge = MagicMock()
    camera_bridge.last_frame_path = "/tmp/mock_frame.jpg"
    camera_bridge.register_consumer.return_value = None
    camera_bridge.start.return_value = None

    vio_proc   = MagicMock()
    vio_proc.process_frame.return_value = MagicMock(
        confidence=0.8,
        delta_north_m=1.0,
        delta_east_m=0.5,
        feature_count=120,
    )

    event_log  = []
    clock_fn   = lambda: 1000

    nm = NavigationManager(
        eskf=eskf,
        bim=bim,
        trn=trn,
        vio_mode=vio_mode,
        camera_bridge=camera_bridge,
        vio_processor=vio_proc,
        event_log=event_log,
        clock_fn=clock_fn,
        trn_interval_m=0.0,   # zero so TRN fires every cycle
        lightglue_client=lightglue_client,
    )
    return nm, event_log


def _make_mock_client(delta_lat=1e-4, delta_lon=2e-4,
                      confidence=0.82, latency_ms=145.0):
    """Return a mock lightglue_client whose match() returns a fixed MatchResult."""
    client = MagicMock()
    client.match.return_value = (delta_lat, delta_lon, confidence, latency_ms)
    return client


def _run_update(nm, terrain_class="ACCEPT"):
    """Run one NavigationManager update cycle with minimal state."""
    state = MagicMock()
    state.p = np.zeros(3)
    return nm.update(
        state=state,
        gnss_available=False,
        gnss_pos=None,
        gnss_measurement=None,
        mission_km=10.0,
        alt_m=540.0,
        gsd_m=0.5,
        lat_estimate=31.10,
        lon_estimate=77.17,
        camera_tile=np.zeros((64, 64), dtype=np.float32),
        mission_time_ms=1000,
        terrain_class=terrain_class,
    )


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

class TestNavigationManagerLightGlue:

    def test_nm_lg_01_none_client_backward_compat(self):
        """NM-LG-01: lightglue_client=None — no exception, PhaseCorrelationTRN used."""
        nm, event_log = _make_nav_manager(lightglue_client=None)
        out = _run_update(nm)
        lg_events = [e for e in event_log if e.get("event") == "NAV_LIGHTGLUE_CORRECTION"]
        assert len(lg_events) == 0, "No LightGlue events when client is None"

    def test_nm_lg_02_mock_client_no_exception(self):
        """NM-LG-02: mock client injected — NavigationManager constructs and runs."""
        client = _make_mock_client()
        nm, _ = _make_nav_manager(lightglue_client=client)
        out = _run_update(nm)
        assert out is not None

    def test_nm_lg_03_correction_event_fired(self):
        """NM-LG-03: valid MatchResult above threshold fires NAV_LIGHTGLUE_CORRECTION."""
        client = _make_mock_client(confidence=0.82)
        nm, event_log = _make_nav_manager(lightglue_client=client)
        _run_update(nm, terrain_class="ACCEPT")
        lg_events = [e for e in event_log if e.get("event") == "NAV_LIGHTGLUE_CORRECTION"]
        assert len(lg_events) == 1, f"Expected 1 correction event, got {len(lg_events)}"
        payload = lg_events[0]["payload"]
        assert "confidence" in payload
        assert "terrain_class" in payload
        assert payload["terrain_class"] == "ACCEPT"

    def test_nm_lg_04_suppress_no_match_call(self):
        """NM-LG-04: SUPPRESS terrain — match() never called."""
        client = _make_mock_client()
        nm, event_log = _make_nav_manager(lightglue_client=client)
        _run_update(nm, terrain_class="SUPPRESS")
        assert client.match.call_count == 0, \
            f"match() called {client.match.call_count} times on SUPPRESS terrain"

    def test_nm_lg_05_caution_threshold_applied(self):
        """NM-LG-05: CAUTION terrain — confidence 0.38 (below 0.40) is rejected."""
        client = _make_mock_client(confidence=0.38)
        nm, event_log = _make_nav_manager(lightglue_client=client)
        _run_update(nm, terrain_class="CAUTION")
        lg_events = [e for e in event_log if e.get("event") == "NAV_LIGHTGLUE_CORRECTION"]
        assert len(lg_events) == 0, \
            "Confidence 0.38 must be rejected on CAUTION terrain (threshold 0.40)"

    def test_nm_lg_06_threshold_lookup(self):
        """NM-LG-06: _lightglue_threshold_for_class() returns correct values."""
        assert _lightglue_threshold_for_class("ACCEPT")   == LIGHTGLUE_CONF_THRESHOLD_ACCEPT
        assert _lightglue_threshold_for_class("CAUTION")  == LIGHTGLUE_CONF_THRESHOLD_CAUTION
        assert _lightglue_threshold_for_class("SUPPRESS") is None
        assert _lightglue_threshold_for_class("UNKNOWN")  == LIGHTGLUE_CONF_THRESHOLD_ACCEPT
