"""
tests/test_gate3_fusion.py
MicroMind / NanoCorteX — Gate 3 Fusion Integration Tests

Validates confidence-aware fusion and degraded state handling.
Does NOT require Gazebo — camera bridge operates in inject-only mode.

Gates:
    NAV-05: TRN corrections reach ESKF via NavigationManager
    NAV-06: VIO confidence correctly weights ESKF covariance
    NAV-07: Degraded state sequence NOMINAL→GNSS_DENIED→NAV_TRN_ONLY→SHM_TRIGGER
    NAV-08: Camera bridge → VIOFrameProcessor pipeline wired correctly

Req IDs: NAV-02, NAV-03, EC-09, EC-10
SRS ref: §2.2, §2.3, §10.1, §16
Governance: Code Governance Manual v3.4
"""
from __future__ import annotations

import os
import sys
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.bim.bim import BIM, BIMConfig, GNSSMeasurement
from core.clock.sim_clock import SimClock
from core.ekf.error_state_ekf import ErrorStateEKF
from core.fusion.vio_mode import VIONavigationMode
from core.ins.state import INSState
from core.navigation.navigation_manager import (
    NavigationManager,
    NAV_MODE_NOMINAL,
    NAV_MODE_GNSS_DENIED,
    NAV_MODE_TRN_ONLY,
    NAV_MODE_SHM_TRIGGER,
)
from core.state_machine.state_machine import (
    NanoCorteXFSM,
    NCState,
    SystemInputs,
)
from core.trn.phase_correlation_trn import TRNMatchResult
from integration.camera.nadir_camera_bridge import NadirCameraFrameBridge
from integration.vio.vio_frame_processor import VIOEstimate, VIOFrameProcessor
from logs.mission_log_schema import BIMState, MissionLog


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEM_PATH = "data/terrain/shimla_corridor/SHIMLA-1_COP30.tif"
_GSD_M    = 5.0
_LAT_SHIMLA = 31.104
_LON_SHIMLA = 77.172


# ---------------------------------------------------------------------------
# Helpers
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


def _make_mock_trn(status: str = "ACCEPTED",
                   confidence: float = 0.85,
                   suitability_score: float = 0.72,
                   correction_north_m: float = 8.0,
                   correction_east_m: float = 4.0) -> MagicMock:
    """Return a mock TRN engine with controlled output."""
    mock = MagicMock()
    mock.match.return_value = TRNMatchResult(
        status=status,
        correction_north_m=correction_north_m,
        correction_east_m=correction_east_m,
        confidence=confidence,
        suitability_score=suitability_score,
        suitability_recommendation="ACCEPT",
        latency_ms=30,
        mission_time_ms=1000,
    )
    return mock


def _make_nav_manager(
    eskf=None,
    trn=None,
    trn_interval_m: float = 5000.0,
    vio_confidence_threshold: float = 0.50,
    nav_confidence_shm_threshold: float = 0.20,
) -> tuple[NavigationManager, ErrorStateEKF, list]:
    """Construct a NavigationManager with real ESKF and controlled TRN."""
    if eskf is None:
        eskf = _make_eskf()
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
        eskf=eskf,
        bim=bim,
        trn=trn,
        vio_mode=vio_mode,
        camera_bridge=camera,
        vio_processor=vio_proc,
        event_log=event_log,
        clock_fn=clock_fn,
        trn_interval_m=trn_interval_m,
        vio_confidence_threshold=vio_confidence_threshold,
        nav_confidence_shm_threshold=nav_confidence_shm_threshold,
    )
    return nm, eskf, event_log


def _healthy_gnss_measurement() -> GNSSMeasurement:
    return GNSSMeasurement(
        pdop=1.5,
        doppler_deviation_ms=0.1,
        ew_jammer_confidence=0.0,
    )


def _synthetic_camera_tile() -> np.ndarray:
    """
    Generate a synthetic 64×64 greyscale tile with structured texture.
    Used where a real camera frame is needed but DEM loading is not required.
    Not photographic quality — sufficient to test pipeline wiring.
    """
    rng = np.random.default_rng(42)
    tile = rng.integers(30, 220, size=(64, 64), dtype=np.uint8)
    # Add horizontal gradient for reliable feature detection
    tile = tile + np.linspace(0, 50, 64, dtype=np.uint8).reshape(1, 64)
    return tile.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Gate NAV-05: TRN corrections reach ESKF
# ---------------------------------------------------------------------------

class TestNAV05TRNReachesESKF:
    """
    Gate NAV-05: NavigationManager calls eskf.update_trn() at each TRN interval.
    TRN_ESKF_UPDATE event is logged with all 4 mandatory fields.
    ESKF state changes after TRN injection.
    """

    def test_nav05_update_trn_called_at_interval(self):
        """TRN fires when distance threshold reached; update_trn() logged."""
        nm, eskf, event_log = _make_nav_manager(trn_interval_m=0.0)
        state = _make_ins_state()

        nm.update(
            state           = state,
            gnss_available  = False,
            gnss_pos        = None,
            gnss_measurement= None,
            mission_km      = 1.0,
            alt_m           = 1700.0,
            gsd_m           = _GSD_M,
            lat_estimate    = _LAT_SHIMLA,
            lon_estimate    = _LON_SHIMLA,
            camera_tile     = _synthetic_camera_tile(),
            mission_time_ms = 1000,
        )

        assert hasattr(eskf, '_trn_event_log'), (
            "update_trn() was never called — _trn_event_log not created"
        )
        assert len(eskf._trn_event_log) >= 1, (
            "TRN fired but no event was logged"
        )

    def test_nav05_trn_eskf_log_mandatory_fields(self):
        """TRN_ESKF_UPDATE event contains all 4 mandatory fields."""
        nm, eskf, _ = _make_nav_manager(trn_interval_m=0.0)
        state = _make_ins_state()

        nm.update(
            state=state, gnss_available=False, gnss_pos=None,
            gnss_measurement=None, mission_km=1.0, alt_m=1700.0,
            gsd_m=_GSD_M, lat_estimate=_LAT_SHIMLA, lon_estimate=_LON_SHIMLA,
            camera_tile=_synthetic_camera_tile(), mission_time_ms=1000,
        )

        assert hasattr(eskf, '_trn_event_log') and eskf._trn_event_log
        entry = eskf._trn_event_log[-1]
        assert entry["event"] == "TRN_ESKF_UPDATE"
        assert entry["module_name"] == "ErrorStateEKF"
        assert entry["req_id"] == "NAV-02"
        assert entry["severity"] == "INFO"
        payload = entry["payload"]
        assert "nis" in payload,               "mandatory field 'nis' missing"
        assert "rejected" in payload,          "mandatory field 'rejected' missing"
        assert "confidence" in payload,        "mandatory field 'confidence' missing"
        assert "suitability_score" in payload, "mandatory field 'suitability_score' missing"
        assert "combined_weight" in payload,   "mandatory field 'combined_weight' missing"

    def test_nav05_eskf_state_changes_after_trn_injection(self):
        """
        ESKF state.p is modified after update_trn() + inject().
        Uses a 10m north / 5m east TRN correction (mock) and verifies
        the Kalman gain produces a non-zero position update.
        """
        eskf  = _make_eskf()
        state = _make_ins_state()
        p_before = state.p.copy()

        # Direct call — bypass NavigationManager to isolate ESKF behaviour
        correction = np.array([10.0, 5.0, 0.0])
        nis, rejected, innov_mag = eskf.update_trn(
            state, correction, confidence=0.85, suitability_score=0.72
        )
        eskf.inject(state)

        assert not rejected, f"update_trn rejected a valid correction: {nis=}"
        assert not np.allclose(state.p, p_before), (
            f"state.p unchanged after TRN injection — Kalman gain produced zero update.\n"
            f"  p_before={p_before}, p_after={state.p}"
        )

    def test_nav05_nav_manager_output_trn_active(self):
        """NavigationManager output reports trn_active=True after accepted correction."""
        nm, _, _ = _make_nav_manager(trn_interval_m=0.0)
        state = _make_ins_state()

        out = nm.update(
            state=state, gnss_available=False, gnss_pos=None,
            gnss_measurement=None, mission_km=1.0, alt_m=1700.0,
            gsd_m=_GSD_M, lat_estimate=_LAT_SHIMLA, lon_estimate=_LON_SHIMLA,
            camera_tile=_synthetic_camera_tile(), mission_time_ms=1000,
        )

        assert out.trn_active, (
            "NavigationManager reported trn_active=False after TRN ACCEPTED"
        )


# ---------------------------------------------------------------------------
# Gate NAV-06: VIO confidence weights ESKF covariance
# ---------------------------------------------------------------------------

class TestNAV06VIOConfidenceWeighting:
    """
    Gate NAV-06: VIO confidence correctly scales the ESKF measurement noise.
    Low confidence → larger covariance (less influence).
    Below threshold → estimate not injected.
    """

    def test_nav06_covariance_larger_at_lower_confidence(self):
        """_encode_vio_confidence(0.3) produces larger noise than (0.8)."""
        nm, _, _ = _make_nav_manager()
        cov_low  = nm._encode_vio_confidence(0.3)
        cov_high = nm._encode_vio_confidence(0.8)

        # Every diagonal element must be larger at lower confidence
        assert np.all(np.diag(cov_low) > np.diag(cov_high)), (
            f"Expected cov_low diagonal > cov_high diagonal.\n"
            f"  cov_low diag: {np.diag(cov_low)}\n"
            f"  cov_high diag: {np.diag(cov_high)}"
        )

    def test_nav06_below_threshold_not_injected(self):
        """VIO estimate with confidence < 0.50 is not passed to ESKF."""
        nm, eskf, _ = _make_nav_manager(vio_confidence_threshold=0.50)
        state = _make_ins_state()

        # Pre-load a low-confidence estimate (0.30 < threshold 0.50)
        nm._latest_vio = VIOEstimate(
            delta_north_m=5.0, delta_east_m=3.0, delta_alt_m=0.0,
            confidence=0.30, feature_count=80, timestamp_ms=500,
        )
        p_before = state.p.copy()

        out = nm.update(
            state=state, gnss_available=False, gnss_pos=None,
            gnss_measurement=None, mission_km=0.0, alt_m=1700.0,
            gsd_m=_GSD_M, lat_estimate=_LAT_SHIMLA, lon_estimate=_LON_SHIMLA,
            camera_tile=None, mission_time_ms=500,
        )

        assert not out.vio_active, (
            f"VIO was injected despite confidence={0.30} < threshold={0.50}"
        )
        assert np.allclose(state.p, p_before), (
            "state.p changed when VIO should have been suppressed"
        )

    def test_nav06_above_threshold_injected(self):
        """VIO estimate with confidence > 0.50 is passed to ESKF."""
        nm, eskf, _ = _make_nav_manager(
            trn_interval_m=999999.0,     # prevent TRN from firing
            vio_confidence_threshold=0.50,
        )
        state = _make_ins_state()

        # Pre-load a high-confidence estimate (0.75 > threshold 0.50)
        nm._latest_vio = VIOEstimate(
            delta_north_m=2.0, delta_east_m=1.0, delta_alt_m=0.0,
            confidence=0.75, feature_count=250, timestamp_ms=500,
        )

        out = nm.update(
            state=state, gnss_available=False, gnss_pos=None,
            gnss_measurement=None, mission_km=0.0, alt_m=1700.0,
            gsd_m=_GSD_M, lat_estimate=_LAT_SHIMLA, lon_estimate=_LON_SHIMLA,
            camera_tile=None, mission_time_ms=500,
        )

        assert out.vio_active, (
            f"VIO was NOT injected despite confidence={0.75} > threshold={0.50}"
        )

    def test_nav06_low_confidence_update_trn_rejected_below_weight(self):
        """update_trn() with w = conf * suit < 0.1 is rejected and logged."""
        eskf = _make_eskf()
        state = _make_ins_state()

        # w = 0.05 * 0.9 = 0.045 < 0.1 threshold
        nis, rejected, innov_mag = eskf.update_trn(
            state, np.array([5.0, 2.0, 0.0]),
            confidence=0.05, suitability_score=0.9,
        )
        assert rejected, "update_trn should reject when combined weight < 0.1"
        assert nis == 0.0
        assert innov_mag == 0.0
        # Verify rejection is logged
        assert hasattr(eskf, '_trn_event_log')
        assert eskf._trn_event_log[-1]["payload"]["rejected"] is True


# ---------------------------------------------------------------------------
# Gate NAV-07: Degraded state sequence
# ---------------------------------------------------------------------------

class TestNAV07DegradedStateSequence:
    """
    Gate NAV-07: NavigationManager mode sequence under progressive sensor failure.
    Step 1: GNSS available         → NOMINAL
    Step 2: GNSS denied, VIO OK    → GNSS_DENIED
    Step 3: VIO degraded, TRN only → NAV_TRN_ONLY
    Step 4: All sources failed     → SHM_TRIGGER (shm_trigger=True)
    FSM check: nav_confidence < threshold → SHM_ACTIVE (SHM_ENTRY_LOW_NAV_CONFIDENCE)
    """

    def _make_fsm(self) -> NanoCorteXFSM:
        clock = SimClock(dt=0.01)
        log   = MissionLog(mission_id="NAV07-TEST")
        fsm   = NanoCorteXFSM(clock=clock, log=log, mission_id="NAV07-TEST")
        clock.start()
        fsm.start()
        return fsm

    def test_nav07_step1_nominal(self):
        """GNSS available → NOMINAL mode."""
        nm, _, _ = _make_nav_manager(trn_interval_m=999999.0)
        state = _make_ins_state()

        out = nm.update(
            state=state,
            gnss_available=True,
            gnss_pos=np.array([0.0, 0.0, 1700.0]),
            gnss_measurement=_healthy_gnss_measurement(),
            mission_km=0.0, alt_m=1700.0, gsd_m=_GSD_M,
            lat_estimate=_LAT_SHIMLA, lon_estimate=_LON_SHIMLA,
            camera_tile=None, mission_time_ms=1000,
        )
        assert out.nav_mode == NAV_MODE_NOMINAL, (
            f"Expected NOMINAL, got {out.nav_mode}"
        )
        assert out.gnss_active

    def test_nav07_step2_gnss_denied_vio_trn_active(self):
        """GNSS denied, VIO confidence OK, TRN fires → GNSS_DENIED."""
        nm, _, _ = _make_nav_manager(trn_interval_m=0.0)
        state = _make_ins_state()

        # Inject a high-confidence VIO estimate
        nm._latest_vio = VIOEstimate(
            delta_north_m=1.0, delta_east_m=0.5, delta_alt_m=0.0,
            confidence=0.80, feature_count=200, timestamp_ms=2000,
        )

        out = nm.update(
            state=state, gnss_available=False, gnss_pos=None,
            gnss_measurement=None, mission_km=1.0, alt_m=1700.0,
            gsd_m=_GSD_M, lat_estimate=_LAT_SHIMLA, lon_estimate=_LON_SHIMLA,
            camera_tile=_synthetic_camera_tile(), mission_time_ms=2000,
        )
        assert out.nav_mode == NAV_MODE_GNSS_DENIED, (
            f"Expected GNSS_DENIED, got {out.nav_mode}"
        )
        assert not out.gnss_active
        assert out.vio_active or out.trn_active

    def test_nav07_step3_nav_trn_only(self):
        """VIO confidence drops below threshold → NAV_TRN_ONLY."""
        nm, _, _ = _make_nav_manager(trn_interval_m=0.0)
        state = _make_ins_state()

        # Low-confidence VIO (below 0.50 threshold)
        nm._latest_vio = VIOEstimate(
            delta_north_m=0.5, delta_east_m=0.2, delta_alt_m=0.0,
            confidence=0.10, feature_count=25, timestamp_ms=3000,
        )

        out = nm.update(
            state=state, gnss_available=False, gnss_pos=None,
            gnss_measurement=None, mission_km=1.0, alt_m=1700.0,
            gsd_m=_GSD_M, lat_estimate=_LAT_SHIMLA, lon_estimate=_LON_SHIMLA,
            camera_tile=_synthetic_camera_tile(), mission_time_ms=3000,
        )
        assert out.nav_mode == NAV_MODE_TRN_ONLY, (
            f"Expected NAV_TRN_ONLY, got {out.nav_mode}"
        )
        assert not out.vio_active
        assert out.trn_active

    def test_nav07_step4_shm_trigger(self):
        """All sources fail → nav_confidence < threshold → shm_trigger=True."""
        nm, _, event_log = _make_nav_manager(
            trn_interval_m=999999.0,          # prevent TRN
            nav_confidence_shm_threshold=0.20,
        )
        state = _make_ins_state()
        nm._latest_vio = None                 # no VIO

        out = nm.update(
            state=state, gnss_available=False, gnss_pos=None,
            gnss_measurement=None, mission_km=0.0, alt_m=1700.0,
            gsd_m=_GSD_M, lat_estimate=_LAT_SHIMLA, lon_estimate=_LON_SHIMLA,
            camera_tile=None, mission_time_ms=4000,
        )

        assert out.shm_trigger, (
            f"Expected shm_trigger=True when all sources inactive, "
            f"got nav_confidence={out.nav_confidence:.3f}"
        )
        assert out.nav_mode == NAV_MODE_SHM_TRIGGER

    def test_nav07_shm_triggered_logged(self):
        """NAV_SHM_TRIGGER event is logged when all sources fail."""
        nm, _, event_log = _make_nav_manager(
            trn_interval_m=999999.0,
            nav_confidence_shm_threshold=0.20,
        )
        state = _make_ins_state()
        nm._latest_vio = None

        nm.update(
            state=state, gnss_available=False, gnss_pos=None,
            gnss_measurement=None, mission_km=0.0, alt_m=1700.0,
            gsd_m=_GSD_M, lat_estimate=_LAT_SHIMLA, lon_estimate=_LON_SHIMLA,
            camera_tile=None, mission_time_ms=4000,
        )

        shm_events = [e for e in event_log if e["event"] == "NAV_SHM_TRIGGER"]
        assert shm_events, "NAV_SHM_TRIGGER was not logged when expected"
        payload = shm_events[0]["payload"]
        assert "nav_confidence" in payload
        assert "threshold" in payload
        assert "last_correction_source" in payload
        assert payload["last_correction_source"] == "NONE"

    def test_nav07_mode_sequence_list(self):
        """Full mode sequence is traversed in order (NOMINAL→GNSS_DENIED→TRN_ONLY→SHM)."""
        states_observed = []

        for step_id, (gnss_avail, vio_conf, with_tile, trn_int) in enumerate([
            (True,  0.0,  False, 999999.0),  # Step 1: NOMINAL
            (False, 0.80, True,  0.0),       # Step 2: GNSS_DENIED
            (False, 0.10, True,  0.0),       # Step 3: NAV_TRN_ONLY
            (False, 0.0,  False, 999999.0),  # Step 4: SHM_TRIGGER
        ]):
            nm, _, _ = _make_nav_manager(trn_interval_m=trn_int)
            state = _make_ins_state()
            if vio_conf > 0:
                nm._latest_vio = VIOEstimate(
                    delta_north_m=1.0, delta_east_m=0.5, delta_alt_m=0.0,
                    confidence=vio_conf, feature_count=200,
                    timestamp_ms=(step_id + 1) * 1000,
                )
            tile = _synthetic_camera_tile() if with_tile else None
            out = nm.update(
                state=state,
                gnss_available=gnss_avail,
                gnss_pos=np.array([0.0, 0.0, 1700.0]) if gnss_avail else None,
                gnss_measurement=_healthy_gnss_measurement() if gnss_avail else None,
                mission_km=1.0 if not gnss_avail else 0.0,
                alt_m=1700.0, gsd_m=_GSD_M,
                lat_estimate=_LAT_SHIMLA, lon_estimate=_LON_SHIMLA,
                camera_tile=tile,
                mission_time_ms=(step_id + 1) * 1000,
            )
            states_observed.append(out.nav_mode)

        expected = [
            NAV_MODE_NOMINAL,
            NAV_MODE_GNSS_DENIED,
            NAV_MODE_TRN_ONLY,
            NAV_MODE_SHM_TRIGGER,
        ]
        assert states_observed == expected, (
            f"Mode sequence mismatch.\n"
            f"  Expected: {expected}\n"
            f"  Got:      {states_observed}"
        )

    def test_nav07_fsm_confidence_shm_entry(self):
        """
        NanoCorteXFSM transitions to SHM_ACTIVE via SHM_ENTRY_LOW_NAV_CONFIDENCE
        when nav_confidence drops below NAV_CONFIDENCE_SHM_THRESHOLD in GNSS_DENIED.
        """
        fsm = self._make_fsm()

        # NOMINAL → EW_AWARE
        result = fsm.evaluate(SystemInputs(ew_jammer_confidence=0.7))
        assert result is not None and result.to_state == NCState.EW_AWARE

        # EW_AWARE → GNSS_DENIED
        result = fsm.evaluate(SystemInputs(
            bim_state=BIMState.RED,
            ew_jammer_confidence=0.7,
            vio_feature_count=50,
        ))
        assert result is not None and result.to_state == NCState.GNSS_DENIED

        # GNSS_DENIED → SHM_ACTIVE via low nav_confidence (0.10 < 0.20)
        result = fsm.evaluate(SystemInputs(
            bim_state=BIMState.RED,
            ew_jammer_confidence=0.7,
            vio_feature_count=50,
            nav_confidence=0.10,   # Gate 3 field — below threshold 0.20
        ))
        assert result is not None, (
            "Expected FSM transition from GNSS_DENIED on low nav_confidence"
        )
        assert result.to_state == NCState.SHM_ACTIVE, (
            f"Expected SHM_ACTIVE, got {result.to_state}"
        )
        assert result.trigger == "SHM_ENTRY_LOW_NAV_CONFIDENCE", (
            f"Expected trigger SHM_ENTRY_LOW_NAV_CONFIDENCE, got {result.trigger}"
        )


# ---------------------------------------------------------------------------
# Gate NAV-08: Camera bridge → VIO pipeline
# ---------------------------------------------------------------------------

class TestNAV08CameraBridgeVIOPipeline:
    """
    Gate NAV-08: NadirCameraFrameBridge delivers frames to VIOFrameProcessor
    via callback registration. VIOEstimate is produced with feature_count > 0.

    Does NOT require Gazebo — inject_frame() is used directly.
    Tests the wiring established in NavigationManager.__init__().
    """

    def test_nav08_vio_receives_frame_via_callback(self):
        """Injected frame triggers VIOFrameProcessor callback."""
        bridge   = NadirCameraFrameBridge()
        received = []

        bridge.register_consumer(
            lambda frame, ts: received.append((frame, ts))
        )
        tile = _synthetic_camera_tile()
        bridge.inject_frame(tile, timestamp_ms=100)

        assert len(received) == 1, "Camera bridge did not deliver frame to consumer"
        assert received[0][1] == 100

    def test_nav08_vio_estimate_feature_count(self):
        """
        VIOFrameProcessor returns a VIOEstimate with feature_count > 0
        after receiving two frames via bridge callback.

        Frame 1: initialises keypoints (returns None).
        Frame 2: tracks keypoints, returns VIOEstimate.

        Note: full Gazebo-rendered photographic confidence (>0.547) is
        verified in live SITL only. This gate validates wiring.
        """
        bridge   = NadirCameraFrameBridge()
        vio_proc = VIOFrameProcessor(
            min_features=10,   # relaxed — synthetic tile has limited features
            max_features=200,
            gsd_m=_GSD_M,
        )
        estimates = []

        def on_frame(frame, ts):
            est = vio_proc.process_frame(frame, ts)
            if est is not None:
                estimates.append(est)

        bridge.register_consumer(on_frame)

        tile1 = _synthetic_camera_tile()
        # Frame 2 is tile1 with small pixel offset — induces optical flow displacement
        tile2 = np.roll(tile1, shift=2, axis=0)  # 2-pixel north shift

        bridge.inject_frame(tile1, timestamp_ms=200)  # frame 1 — initialise
        bridge.inject_frame(tile2, timestamp_ms=400)  # frame 2 — should produce estimate

        assert len(estimates) >= 1, (
            "VIOFrameProcessor returned no estimate after two frames via bridge."
        )
        assert estimates[0].feature_count > 0, (
            f"VIOEstimate.feature_count == 0 (got {estimates[0].feature_count})"
        )

    def test_nav08_navigation_manager_registers_consumer(self):
        """
        NavigationManager.__init__() registers _on_camera_frame as a consumer.
        Injecting a frame via the bridge populates _latest_vio after two frames.
        """
        nm, _, _ = _make_nav_manager(trn_interval_m=999999.0)

        bridge   = nm._camera_bridge
        vio_proc = nm._vio_processor

        tile1 = _synthetic_camera_tile()
        tile2 = np.roll(tile1, shift=2, axis=0)

        # Inject two frames — first frame initialises, second produces estimate
        bridge.inject_frame(tile1, timestamp_ms=300)
        bridge.inject_frame(tile2, timestamp_ms=600)

        # Small sleep to allow any threaded delivery (should be synchronous in inject mode)
        time.sleep(0.05)

        # After two frames, _latest_vio should be populated if features were tracked
        # (outcome depends on min_features threshold vs synthetic tile quality —
        # we assert the pipeline ran without exception; feature count checked above)
        vio_log = vio_proc.event_log
        assert any(e["event"] in ("VIO_ESTIMATE_PRODUCED", "VIO_INSUFFICIENT_FEATURES")
                   for e in vio_log), (
            "VIOFrameProcessor was not called at all via bridge → NavigationManager"
        )
