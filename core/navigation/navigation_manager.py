"""
core/navigation/navigation_manager.py
MicroMind / NanoCorteX — Navigation Fusion Coordinator

NAV-02, NAV-03: Central navigation fusion coordinator.
Gate 3 deliverable — confidence-aware ESKF injection.

Receives corrections from all sources (GNSS/VIO/TRN), applies confidence
weighting, injects into ESKF, maintains unified positional confidence score,
and drives FSM SystemInputs nav health fields.

This is the class referenced in PhaseCorrelationTRN docstring as
'NavigationManager — responsible caller'. It did not exist before Gate 3.

Governance: §1.2 Three-Layer Navigation Architecture (AD-01)
  L1: VIO (relative, high-rate) — managed here
  L2: TRN orthophoto matching (absolute reset) — managed here
  L3: Baro-INS (vertical stability) — not in scope for Gate 3

Logic Box: Navigation Core Box (core/) — no MAVLink/Gazebo imports permitted.

References:
    SRS v1.3 NAV-02, NAV-03, EC-09, EC-10
    Part Two V7.2 §2.2, §2.3, §10.1, §16
    docs/interfaces/trn_contract.yaml
    docs/interfaces/eo_day_contract.yaml
    config/tunable_mission.yaml
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from core.bim.bim import BIM, GNSSMeasurement
from core.ekf.error_state_ekf import ErrorStateEKF
from core.fusion.vio_mode import VIONavigationMode
from core.trn.phase_correlation_trn import PhaseCorrelationTRN
from integration.camera.nadir_camera_bridge import NadirCameraFrameBridge
from integration.vio.vio_frame_processor import VIOFrameProcessor, VIOEstimate


# ── Navigation mode strings ───────────────────────────────────────────────────

NAV_MODE_NOMINAL    = "NOMINAL"       # GNSS available, BIM GREEN
NAV_MODE_GNSS_DENIED = "GNSS_DENIED" # GNSS unavailable, VIO or TRN active
NAV_MODE_TRN_ONLY   = "NAV_TRN_ONLY" # GNSS unavailable, VIO below threshold
NAV_MODE_INS_ONLY   = "INS_ONLY"     # GNSS + VIO unavailable, TRN gap > 15km
NAV_MODE_SHM_TRIGGER = "SHM_TRIGGER" # nav_confidence < threshold

# AD-23 validated LightGlue confidence thresholds — SAL-2 (AD-24, OI-49)
LIGHTGLUE_CONF_THRESHOLD_ACCEPT   = 0.35   # structured terrain — AD-23 validated
LIGHTGLUE_CONF_THRESHOLD_CAUTION  = 0.40   # marginal terrain — higher bar
# SUPPRESS class: match skipped entirely — no threshold needed

_LIGHTGLUE_TERRAIN_THRESHOLDS = {
    "ACCEPT":   LIGHTGLUE_CONF_THRESHOLD_ACCEPT,
    "CAUTION":  LIGHTGLUE_CONF_THRESHOLD_CAUTION,
}

# ── VIO nominal covariance (1m horizontal, 2m vertical 1-sigma) ──────────────
_R_VIO_NOMINAL = np.diag([1.0, 1.0, 2.0])  # m²


def _lightglue_threshold_for_class(terrain_class: str) -> Optional[float]:
    """
    Return the LightGlue confidence threshold for the given terrain class.

    Returns None for SUPPRESS — caller must skip the match entirely.
    Returns the float threshold for ACCEPT and CAUTION.
    Unknown classes default to ACCEPT threshold (conservative).

    SAL-2 / AD-24 / OI-49.
    """
    if terrain_class == "SUPPRESS":
        return None
    return _LIGHTGLUE_TERRAIN_THRESHOLDS.get(
        terrain_class, LIGHTGLUE_CONF_THRESHOLD_ACCEPT
    )


# ── Output dataclass ─────────────────────────────────────────────────────────

@dataclass
class NavigationManagerOutput:
    """
    Output of NavigationManager.update() — consumed by FSM SystemInputs
    and mission telemetry.
    """
    nav_mode:                   str
    nav_confidence:             float        # 0.0–1.0 unified positional confidence
    gnss_active:                bool
    vio_active:                 bool
    trn_active:                 bool
    last_trn_correction_km:     float        # mission km of last accepted TRN fix
    shm_trigger:                bool         # True when nav_confidence < threshold
    event_log_entries:          List[dict]   # structured events this update cycle


# ── Main class ────────────────────────────────────────────────────────────────

class NavigationManager:
    """
    NAV-02, NAV-03: Central navigation fusion coordinator.

    Arbitrates between GNSS, VIO, and TRN correction sources.
    Applies confidence weighting before ESKF injection.
    Maintains a unified positional confidence score.
    Triggers SHM when all correction sources fail simultaneously.

    See TECHNICAL_NOTES.md for OODA-loop rationale and design decisions.
    """

    def __init__(
        self,
        eskf:                       ErrorStateEKF,
        bim:                        BIM,
        trn:                        PhaseCorrelationTRN,
        vio_mode:                   VIONavigationMode,
        camera_bridge:              NadirCameraFrameBridge,
        vio_processor:              VIOFrameProcessor,
        event_log:                  list,
        clock_fn:                   Callable[[], int],
        trn_interval_m:             float = 5000.0,
        vio_confidence_threshold:   float = 0.50,
        nav_confidence_shm_threshold: float = 0.20,
        lightglue_client=None,
    ) -> None:
        """
        eskf                       : ErrorStateEKF instance (shared with caller)
        bim                        : BIM instance for GNSS trust evaluation
        trn                        : PhaseCorrelationTRN engine
        vio_mode                   : VIONavigationMode tracker
        camera_bridge              : NadirCameraFrameBridge (Gazebo / HIL camera)
        vio_processor              : VIOFrameProcessor (ORB+LK / OpenVINS at HIL)
        event_log                  : Shared programme event log list (appended in place)
        clock_fn                   : Callable returning mission time in ms (int).
                                     No time.time() calls — AD-11 mission clock.
        trn_interval_m             : Minimum metres between TRN correction attempts.
                                     Default 5000 m (Shimla ridge spacing).
                                     From config/tunable_mission.yaml.
        vio_confidence_threshold   : Minimum VIO confidence to inject into ESKF.
                                     0.50 — lower than Gate 2's 0.70 to allow
                                     Gazebo photographic frames to contribute.
                                     From config/tunable_mission.yaml.
        nav_confidence_shm_threshold: Unified positional confidence below which
                                     SHM is triggered. 0.20 default.
                                     From config/tunable_mission.yaml.
        lightglue_client : Optional LightGlue client module
                           (integration.lightglue_bridge.client). When None,
                           LightGlue L2 path is disabled — PhaseCorrelationTRN
                           only. Injected at construction for testability.
        """
        self._eskf              = eskf
        self._bim               = bim
        self._trn               = trn
        self._vio_mode          = vio_mode
        self._camera_bridge     = camera_bridge
        self._vio_processor     = vio_processor
        self._event_log         = event_log
        self._clock_fn          = clock_fn

        self._trn_interval_m    = trn_interval_m
        self._vio_conf_threshold = vio_confidence_threshold
        self._shm_threshold     = nav_confidence_shm_threshold
        self._lightglue_client = lightglue_client   # None disables LightGlue L2 path

        # Tracking state
        self._last_trn_km:      float                    = 0.0
        self._last_nav_mode:    str                      = NAV_MODE_NOMINAL
        self._latest_vio:       Optional[VIOEstimate]    = None

        # Step 5 — close camera loop: register VIOFrameProcessor as consumer.
        # Gazebo renders real terrain frames; bridge delivers them to VIO processor.
        # This is the Gate 2 open finding fix: VIO confidence was capped at 0.547
        # on DEM greyscale hillshades. Photographic frames yield higher contrast
        # features and push confidence above the 0.50 injection threshold.
        self._camera_bridge.register_consumer(self._on_camera_frame)
        self._camera_bridge.start()

    # ── Camera consumer (Step 5 wiring) ──────────────────────────────────────

    def _on_camera_frame(self, frame: np.ndarray, timestamp_ms: int) -> None:
        """
        Called by NadirCameraFrameBridge on each new frame.
        Passes frame to VIOFrameProcessor and stores latest estimate.
        Thread-safe: bridge delivers via callback; _latest_vio is written here
        and read in update(). Races are benign — worst case is using the
        previous estimate, which is the correct degraded behaviour.
        """
        estimate = self._vio_processor.process_frame(frame, timestamp_ms)
        if estimate is not None:
            self._latest_vio = estimate

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        state,
        gnss_available:     bool,
        gnss_pos:           Optional[np.ndarray],
        gnss_measurement:   Optional[GNSSMeasurement],
        mission_km:         float,
        alt_m:              float,
        gsd_m:              float,
        lat_estimate:       float,
        lon_estimate:       float,
        camera_tile:        Optional[np.ndarray],
        mission_time_ms:    int,
        terrain_class:      str = "ACCEPT",
    ) -> NavigationManagerOutput:
        """
        Called once per navigation cycle. Orchestrates all correction sources.

        state            : INSState (current nominal state, updated in place
                           via eskf.inject())
        gnss_available   : True if GNSS fix is present and not denied
        gnss_pos         : (3,) position in world frame — required if gnss_available
        gnss_measurement : GNSSMeasurement for BIM evaluation — None if no GNSS
        mission_km       : Current distance from launch (km)
        alt_m            : Altitude AGL (m) — for TRN
        gsd_m            : Ground sample distance (m/pixel) — for TRN + VIO
        lat_estimate     : Estimated latitude — for TRN
        lon_estimate     : Estimated longitude — for TRN
        camera_tile      : Optional direct-inject camera tile (for tests / TRN).
                           None in live operation (frames arrive via camera_bridge).
        mission_time_ms  : Mission time in ms (from injected clock)
        terrain_class    : Terrain suitability class for this corridor segment.
                           One of 'ACCEPT', 'CAUTION', 'SUPPRESS' (AD-24 SAL-2).
                           Default 'ACCEPT' — backward compatible with all existing callers.

        Returns NavigationManagerOutput.
        """
        cycle_log: List[dict] = []

        gnss_conf   = 0.0
        vio_conf    = 0.0
        trn_conf    = 0.0
        gnss_active = False
        vio_active  = False
        trn_active  = False

        # ── Step 1: BIM evaluation ────────────────────────────────────────
        trust_score = 0.0
        if gnss_available and gnss_measurement is not None:
            bim_out = self._bim.evaluate(gnss_measurement)
            trust_score = bim_out.trust_score
        elif not gnss_available:
            trust_score = 0.0

        # ── Step 2: GNSS update ───────────────────────────────────────────
        if gnss_available and gnss_pos is not None and trust_score > 0.1:
            self._eskf.update_gnss(state, gnss_pos, trust_score)
            self._eskf.inject(state)
            gnss_active = True
            gnss_conf   = float(np.clip(trust_score, 0.0, 1.0))

        # ── Step 3: VIO update ────────────────────────────────────────────
        # Use camera_tile injection for direct-inject mode (tests / TRN caller),
        # otherwise use latest estimate from camera bridge consumer callback.
        if camera_tile is not None:
            vio_est = self._vio_processor.process_frame(camera_tile, mission_time_ms)
            if vio_est is not None:
                self._latest_vio = vio_est

        vio_est = self._latest_vio
        if vio_est is not None and vio_est.confidence >= self._vio_conf_threshold:
            cov = self._encode_vio_confidence(vio_est.confidence)
            pos_ned = state.p.copy()
            pos_ned[0] += vio_est.delta_north_m
            pos_ned[1] += vio_est.delta_east_m
            # pos_ned[2] unchanged — monocular nadir cannot observe altitude
            nis, rejected, innov_mag = self._eskf.update_vio(state, pos_ned, cov)
            if not rejected:
                self._eskf.inject(state)
                vio_active = True
                vio_conf   = float(vio_est.confidence)
                self._vio_mode.on_vio_update(accepted=True, innov_mag=innov_mag)
                cycle_log.append({
                    "event":        "NAV_VIO_CONTRIBUTION",
                    "module_name":  "NavigationManager",
                    "req_id":       "NAV-03",
                    "severity":     "INFO",
                    "timestamp_ms": mission_time_ms,
                    "payload": {
                        "confidence":    round(vio_conf, 4),
                        "feature_count": vio_est.feature_count,
                        "delta_north_m": round(vio_est.delta_north_m, 3),
                        "delta_east_m":  round(vio_est.delta_east_m, 3),
                    },
                })
            else:
                self._vio_mode.on_vio_update(accepted=False, innov_mag=innov_mag)
        else:
            if vio_est is not None:
                self._vio_mode.on_vio_update(accepted=False, innov_mag=0.0)

        # Advance VIO mode clock (1 navigation cycle ≈ 0.2 s at 5 Hz)
        # Clock is approximate here — caller should pass dt if sub-Hz precise tracking needed
        self._vio_mode.tick(0.2)

        # ── Step 4a: LightGlue L2 correction (primary, AD-23 / SAL-2 AD-24) ──
        lightglue_accepted = False
        distance_since_trn_m = (mission_km - self._last_trn_km) * 1000.0
        if self._lightglue_client is not None and distance_since_trn_m >= self._trn_interval_m:
            lg_threshold = _lightglue_threshold_for_class(terrain_class)
            if lg_threshold is not None:   # None == SUPPRESS — skip IPC entirely
                lg_frame_path = getattr(self._camera_bridge, 'last_frame_path', None)
                if lg_frame_path is not None:
                    lg_result = self._lightglue_client.match(
                        uav_frame_path=lg_frame_path,
                        lat=lat_estimate,
                        lon=lon_estimate,
                        alt=alt_m,
                        heading_deg=0.0,
                    )
                    if lg_result is not None:
                        delta_lat, delta_lon, lg_conf, lg_latency_ms = lg_result
                        if lg_conf >= lg_threshold:
                            correction = np.array([
                                delta_lat * 111_320.0,
                                delta_lon * 111_320.0 * np.cos(np.radians(lat_estimate)),
                                0.0,
                            ])
                            nis, rejected, innov_mag = self._eskf.update_trn(
                                state,
                                correction,
                                lg_conf,
                                1.0,
                            )
                            if not rejected:
                                self._eskf.inject(state)
                                trn_active        = True
                                trn_conf          = lg_conf
                                self._last_trn_km = mission_km
                                lightglue_accepted = True
                                cycle_log.append({
                                    "event":        "NAV_LIGHTGLUE_CORRECTION",
                                    "module_name":  "NavigationManager",
                                    "req_id":       "NAV-02",
                                    "severity":     "INFO",
                                    "timestamp_ms": mission_time_ms,
                                    "payload": {
                                        "confidence":    round(lg_conf, 4),
                                        "latency_ms":    round(lg_latency_ms, 1),
                                        "delta_north_m": round(correction[0], 2),
                                        "delta_east_m":  round(correction[1], 2),
                                        "terrain_class": terrain_class,
                                    },
                                })

        # ── Step 4b: PhaseCorrelationTRN fallback (SIL compatibility) ────────
        # Skipped if LightGlue already accepted a correction this cycle.
        # distance_since_trn_m computed above in Step 4a.
        if not lightglue_accepted and distance_since_trn_m >= self._trn_interval_m:
            tile = camera_tile
            if tile is not None:
                trn_result = self._trn.match(
                    camera_tile=tile,
                    lat_estimate=lat_estimate,
                    lon_estimate=lon_estimate,
                    alt_m=alt_m,
                    gsd_m=gsd_m,
                    mission_time_ms=mission_time_ms,
                )
                if trn_result.status == "ACCEPTED":
                    correction = np.array([
                        trn_result.correction_north_m,
                        trn_result.correction_east_m,
                        0.0,
                    ])
                    nis, rejected, innov_mag = self._eskf.update_trn(
                        state,
                        correction,
                        trn_result.confidence,
                        trn_result.suitability_score,
                    )
                    if not rejected:
                        self._eskf.inject(state)
                        trn_active        = True
                        trn_conf          = float(trn_result.confidence)
                        self._last_trn_km = mission_km

        # ── Step 5: Unified nav confidence ───────────────────────────────
        # Weighted average of active source confidences.
        # Weights reflect measurement reliability hierarchy:
        #   GNSS absolute (1.0), VIO relative (0.7), TRN absolute (0.5)
        # Normalised by sum of active weights — only active sources contribute.
        w_gnss = 1.0 if gnss_active else 0.0
        w_vio  = 0.7 if vio_active  else 0.0
        w_trn  = 0.5 if trn_active  else 0.0
        w_sum  = w_gnss + w_vio + w_trn

        if w_sum > 0.0:
            nav_confidence = (
                w_gnss * gnss_conf +
                w_vio  * vio_conf  +
                w_trn  * trn_conf
            ) / w_sum
        else:
            nav_confidence = 0.0  # no active sources — confidence collapses

        nav_confidence = float(np.clip(nav_confidence, 0.0, 1.0))

        # ── Step 6: Determine nav mode ────────────────────────────────────
        if gnss_active:
            nav_mode = NAV_MODE_NOMINAL
        elif gnss_available is False and (vio_active or trn_active):
            if vio_conf >= self._vio_conf_threshold:
                nav_mode = NAV_MODE_GNSS_DENIED
            else:
                # VIO below threshold — check TRN recency
                km_since_trn = mission_km - self._last_trn_km
                if trn_active or km_since_trn <= 15.0:
                    nav_mode = NAV_MODE_TRN_ONLY
                else:
                    nav_mode = NAV_MODE_INS_ONLY
        else:
            km_since_trn = mission_km - self._last_trn_km
            if km_since_trn > 15.0:
                nav_mode = NAV_MODE_INS_ONLY
            else:
                nav_mode = NAV_MODE_TRN_ONLY

        shm_trigger = nav_confidence < self._shm_threshold

        if shm_trigger:
            nav_mode = NAV_MODE_SHM_TRIGGER

        # ── Step 7: Structured event logging ─────────────────────────────
        if nav_mode != self._last_nav_mode:
            cycle_log.append({
                "event":        "NAV_MODE_TRANSITION",
                "module_name":  "NavigationManager",
                "req_id":       "NAV-03",
                "severity":     "INFO",
                "timestamp_ms": mission_time_ms,
                "payload": {
                    "from_mode":     self._last_nav_mode,
                    "to_mode":       nav_mode,
                    "nav_confidence": round(nav_confidence, 4),
                    "cause":         self._mode_cause(
                        gnss_active, vio_active, trn_active, nav_confidence,
                        shm_trigger
                    ),
                },
            })
            self._last_nav_mode = nav_mode

        if nav_confidence < 0.35 and not shm_trigger:
            cycle_log.append({
                "event":        "NAV_CONFIDENCE_LOW",
                "module_name":  "NavigationManager",
                "req_id":       "NAV-02",
                "severity":     "WARNING",
                "timestamp_ms": mission_time_ms,
                "payload": {
                    "nav_confidence": round(nav_confidence, 4),
                    "gnss_active":    gnss_active,
                    "vio_active":     vio_active,
                    "trn_active":     trn_active,
                },
            })

        if shm_trigger:
            last_source = (
                "TRN" if trn_active else
                "VIO" if vio_active else
                "GNSS" if gnss_active else "NONE"
            )
            cycle_log.append({
                "event":        "NAV_SHM_TRIGGER",
                "module_name":  "NavigationManager",
                "req_id":       "NAV-02",
                "severity":     "CRITICAL",
                "timestamp_ms": mission_time_ms,
                "payload": {
                    "nav_confidence":        round(nav_confidence, 4),
                    "threshold":             self._shm_threshold,
                    "last_correction_source": last_source,
                },
            })

        # Append cycle events to shared programme log
        self._event_log.extend(cycle_log)

        return NavigationManagerOutput(
            nav_mode                = nav_mode,
            nav_confidence          = nav_confidence,
            gnss_active             = gnss_active,
            vio_active              = vio_active,
            trn_active              = trn_active,
            last_trn_correction_km  = self._last_trn_km,
            shm_trigger             = shm_trigger,
            event_log_entries       = cycle_log,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode_vio_confidence(self, confidence: float) -> np.ndarray:
        """
        Convert VIO confidence scalar to (3,3) covariance for update_vio().

        R_VIO_NOMINAL / clip(confidence, 0.1, 1.0)

        High confidence (0.9) → small noise → strong weight on VIO correction.
        Low confidence (0.1)  → 10× R_VIO_NOMINAL → weak weight.

        R_VIO_NOMINAL: [1.0, 1.0, 2.0] m² diagonal
        (1m horizontal 1-sigma, 2m vertical 1-sigma)
        """
        scale = float(np.clip(confidence, 0.1, 1.0))
        return _R_VIO_NOMINAL / scale

    def _mode_cause(
        self,
        gnss_active: bool,
        vio_active: bool,
        trn_active: bool,
        nav_confidence: float,
        shm_trigger: bool,
    ) -> str:
        """Human-readable cause string for NAV_MODE_TRANSITION log."""
        if shm_trigger:
            return f"nav_confidence={nav_confidence:.3f} below SHM threshold"
        if gnss_active:
            return "GNSS active"
        if vio_active and trn_active:
            return "GNSS denied, VIO+TRN active"
        if vio_active:
            return "GNSS denied, VIO active"
        if trn_active:
            return "GNSS denied, VIO below threshold, TRN active"
        return "all correction sources inactive"

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def last_trn_km(self) -> float:
        """Mission km of last accepted TRN correction."""
        return self._last_trn_km

    @property
    def current_nav_mode(self) -> str:
        """Current navigation mode string."""
        return self._last_nav_mode
