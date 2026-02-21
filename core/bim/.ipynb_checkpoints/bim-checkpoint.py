"""
core/bim/bim.py
MicroMind / NanoCorteX — Beacon & Integrity Monitor (FR-101)
Sprint S2 Deliverable 1 of 3

Evaluates trust in GNSS signals every call and produces:
  - Continuous trust score  0.0–1.0  (drives EKF noise scaling)
  - Discrete trust state    GREEN / AMBER / RED  (drives FSM transitions)

Architecture (Part Two V7 §1.8):
  Five scored inputs → weighted combiner → raw score → hysteresis filter
  → trust state + trust score output

Score weighting model (§1.8.3 — starting values, tune in SIL S2):
  RAIM / PDOP          0.35   — most reliable integrity indicator
  Doppler deviation    0.25   — strong spoof discriminator
  Multi-constellation  0.20   — requires ≥ 2 constellations
  Pose innovation      0.15   — detects position jumps vs inertial
  EW impact            0.05   — contextual near jammer zones

State thresholds:
  GREEN   score ≥ 0.70
  AMBER   0.40 ≤ score < 0.70
  RED     score < 0.40

Hysteresis: 3 consecutive samples at new state before transition.

Latency budget (NFR-001 ≤ 250 ms total):
  GNSS analysis   ≤ 80 ms
  Score combine   ≤ 30 ms
  Output          ≤ 20 ms
  Target total    ≤ 130 ms (with 120 ms margin)

EKF scaling: R_GNSS = R_nominal / trust_score  (capped at 10× R_nominal)

References:
  Part Two V7  §1.8 (BIM), §1.8.2 (boundary conditions), §1.8.3 (weights)
  TechReview   §2.3 (BIM proposed BCs)
  FR-101, NFR-001
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from logs.mission_log_schema import BIMState


# ---------------------------------------------------------------------------
# BIM configuration (all tunable for SIL S2)
# ---------------------------------------------------------------------------

@dataclass
class BIMConfig:
    """
    All BIM parameters in one place.
    Matches Part Two V7 §1.8.2 and §1.8.3 exactly.
    Override any field for SIL tuning runs.
    """
    # --- Score weights (must sum to 1.0) ---
    w_raim:             float = 0.35
    w_doppler:          float = 0.25
    w_constellation:    float = 0.20
    w_pose_innovation:  float = 0.15
    w_ew_impact:        float = 0.05

    # --- PDOP thresholds ---
    pdop_amber:         float = 3.0
    pdop_red:           float = 6.0

    # --- Doppler deviation thresholds (m/s) ---
    doppler_amber_ms:   float = 0.5
    doppler_red_ms:     float = 1.5

    # --- Multi-constellation position delta threshold (m) ---
    constellation_delta_red_m: float = 15.0

    # --- C/N0 drop threshold for spoof discrimination (dB) ---
    cn0_drop_spoof_db:  float = 6.0

    # --- Trust state thresholds ---
    green_threshold:    float = 0.70
    amber_threshold:    float = 0.40   # RED below this

    # --- Hysteresis: samples needed before state transition ---
    hysteresis_count:   int   = 3

    # --- EKF noise scaling cap ---
    ekf_scale_cap:      float = 10.0

    # --- Output rate (Hz) — informational; caller controls call frequency ---
    output_rate_hz:     float = 10.0

    def validate(self) -> None:
        total = self.w_raim + self.w_doppler + self.w_constellation \
                + self.w_pose_innovation + self.w_ew_impact
        assert abs(total - 1.0) < 1e-6, f"BIM weights must sum to 1.0, got {total:.6f}"
        assert self.amber_threshold < self.green_threshold
        assert self.hysteresis_count >= 1


# ---------------------------------------------------------------------------
# BIM inputs
# ---------------------------------------------------------------------------

@dataclass
class GNSSMeasurement:
    """
    Raw GNSS measurement passed to BIM each evaluation cycle.
    All fields Optional — BIM degrades gracefully on missing data.
    """
    # RAIM / signal quality
    pdop:                   Optional[float] = None   # Position Dilution of Precision
    cn0_db:                 Optional[float] = None   # Carrier-to-noise ratio (dB-Hz)
    tracked_satellites:     Optional[int]   = None

    # Multi-constellation
    gps_position_enu:       Optional[np.ndarray] = None   # shape (3,) metres
    glonass_position_enu:   Optional[np.ndarray] = None   # shape (3,) metres

    # Doppler
    doppler_deviation_ms:   Optional[float] = None   # |measured - predicted| m/s

    # Pose innovation (from EKF)
    pose_innovation_m:      Optional[float] = None   # position jump vs inertial (m)

    # EW context (from EW Engine)
    ew_jammer_confidence:   float = 0.0              # 0.0–1.0

    # Timestamp
    timestamp_s:            float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# BIM output
# ---------------------------------------------------------------------------

@dataclass
class BIMOutput:
    """
    BIM output produced each evaluation cycle.
    trust_score drives EKF noise scaling.
    trust_state drives FSM transitions.
    """
    trust_score:        float       # 0.0–1.0 continuous
    trust_state:        BIMState    # GREEN / AMBER / RED
    ekf_noise_scale:    float       # R_GNSS = R_nominal × ekf_noise_scale

    # Component scores (for logging and SIL tuning)
    score_raim:         float = 0.0
    score_doppler:      float = 0.0
    score_constellation: float = 0.0
    score_pose:         float = 0.0
    score_ew:           float = 0.0
    raw_score:          float = 0.0     # pre-hysteresis weighted sum

    # State change flag
    state_changed:      bool  = False
    prev_state:         Optional[BIMState] = None

    # Hysteresis counter (samples in candidate state)
    hysteresis_samples: int   = 0

    # Processing latency
    latency_ms:         float = 0.0

    # Spoof alert (multi-constellation + C/N0 combined criterion)
    spoof_alert:        bool  = False


# ---------------------------------------------------------------------------
# BIM — main class
# ---------------------------------------------------------------------------

class BIM:
    """
    Beacon & Integrity Monitor (FR-101).

    Stateful: maintains hysteresis buffer across calls.
    Call evaluate() at ≥ 10 Hz (set by SimClock or real sensor loop).

    Usage:
        config = BIMConfig()
        bim    = BIM(config)
        output = bim.evaluate(measurement)
        # output.trust_score → ESKF update_gnss() trust parameter
        # output.trust_state → SystemInputs.bim_state → FSM guard
    """

    def __init__(self, config: Optional[BIMConfig] = None):
        self._cfg = config or BIMConfig()
        self._cfg.validate()

        # Current confirmed state (after hysteresis)
        self._state:          BIMState = BIMState.GREEN
        # Candidate state being accumulated
        self._candidate:      BIMState = BIMState.GREEN
        # Hysteresis counter for current candidate
        self._hyst_count:     int      = 0
        # History of raw scores (for diagnostics)
        self._score_history:  Deque[float] = deque(maxlen=50)
        # Last output (for state_changed detection)
        self._last_output:    Optional[BIMOutput] = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(self, measurement: GNSSMeasurement) -> BIMOutput:
        """
        Evaluate GNSS trust from one measurement snapshot.

        Args:
            measurement: GNSSMeasurement with any subset of fields populated.

        Returns:
            BIMOutput with trust_score, trust_state, ekf_noise_scale,
            component scores, and latency.
        """
        t_start = time.monotonic()

        # --- Score each component ---
        s_raim   = self._score_raim(measurement)
        s_dopp   = self._score_doppler(measurement)
        s_const  = self._score_constellation(measurement)
        s_pose   = self._score_pose_innovation(measurement)
        s_ew     = self._score_ew_impact(measurement)

        # --- Weighted combination ---
        raw = (
            self._cfg.w_raim          * s_raim  +
            self._cfg.w_doppler       * s_dopp  +
            self._cfg.w_constellation * s_const +
            self._cfg.w_pose_innovation * s_pose +
            self._cfg.w_ew_impact     * s_ew
        )
        raw = float(np.clip(raw, 0.0, 1.0))
        self._score_history.append(raw)

        # --- Determine candidate state from raw score ---
        candidate = self._score_to_state(raw)

        # --- Spoof alert (hard criterion — bypass hysteresis) ---
        spoof = self._detect_spoof(measurement)

        # --- Apply hysteresis ---
        confirmed_state, hyst_count, state_changed = self._apply_hysteresis(
            candidate, spoof
        )

        # --- Compute trust score ---
        # Trust score is the raw weighted score, further suppressed on spoof
        trust_score = raw if not spoof else min(raw, 0.05)
        trust_score = float(np.clip(trust_score, 0.0, 1.0))

        # --- EKF noise scaling ---
        # R_GNSS = R_nominal / trust_score, capped at 10×
        if trust_score > 0.0:
            ekf_scale = min(1.0 / trust_score, self._cfg.ekf_scale_cap)
        else:
            ekf_scale = self._cfg.ekf_scale_cap

        t_end = time.monotonic()
        latency_ms = (t_end - t_start) * 1000.0

        output = BIMOutput(
            trust_score         = trust_score,
            trust_state         = confirmed_state,
            ekf_noise_scale     = ekf_scale,
            score_raim          = s_raim,
            score_doppler       = s_dopp,
            score_constellation = s_const,
            score_pose          = s_pose,
            score_ew            = s_ew,
            raw_score           = raw,
            state_changed       = state_changed,
            prev_state          = (
                self._last_output.trust_state
                if self._last_output else None
            ),
            hysteresis_samples  = hyst_count,
            latency_ms          = latency_ms,
            spoof_alert         = spoof,
        )
        self._last_output = output
        return output

    # ------------------------------------------------------------------
    # Component scorers  (each returns 0.0–1.0; 1.0 = fully healthy)
    # ------------------------------------------------------------------

    def _score_raim(self, m: GNSSMeasurement) -> float:
        """
        RAIM / PDOP score.
        PDOP ≤ 3.0 → 1.0 (Green)
        PDOP 3.0–6.0 → linear 0.4–1.0 (Amber zone)
        PDOP > 6.0 → 0.0 (Red)
        Missing data → conservative 0.5
        """
        if m.pdop is None:
            return 0.5  # conservative unknown
        if m.pdop <= self._cfg.pdop_amber:
            return 1.0
        if m.pdop >= self._cfg.pdop_red:
            return 0.0
        # Linear interpolation in Amber zone
        span = self._cfg.pdop_red - self._cfg.pdop_amber
        return 1.0 - (m.pdop - self._cfg.pdop_amber) / span

    def _score_doppler(self, m: GNSSMeasurement) -> float:
        """
        Doppler deviation score.
        |dev| ≤ 0.5 m/s → 1.0
        |dev| 0.5–1.5 m/s → linear 0.0–1.0
        |dev| > 1.5 m/s → 0.0
        Missing → 0.8 (slight penalty for unavailable data)
        """
        if m.doppler_deviation_ms is None:
            return 0.8
        dev = abs(m.doppler_deviation_ms)
        if dev <= self._cfg.doppler_amber_ms:
            return 1.0
        if dev >= self._cfg.doppler_red_ms:
            return 0.0
        span = self._cfg.doppler_red_ms - self._cfg.doppler_amber_ms
        return 1.0 - (dev - self._cfg.doppler_amber_ms) / span

    def _score_constellation(self, m: GNSSMeasurement) -> float:
        """
        Multi-constellation position consistency score.
        Requires both GPS and GLONASS positions.
        Delta > 15 m → 0.0 (spoof indicator — Red hard criterion)
        Delta 0–15 m → linear 0.0–1.0
        Only one constellation → 0.6 (degraded but not failed)
        Neither available → 0.5
        """
        if m.gps_position_enu is None or m.glonass_position_enu is None:
            # Only one constellation available
            if m.gps_position_enu is not None or m.glonass_position_enu is not None:
                return 0.6
            return 0.5
        delta = float(np.linalg.norm(
            m.gps_position_enu - m.glonass_position_enu
        ))
        if delta >= self._cfg.constellation_delta_red_m:
            return 0.0
        return 1.0 - (delta / self._cfg.constellation_delta_red_m)

    def _score_pose_innovation(self, m: GNSSMeasurement) -> float:
        """
        Pose innovation residual score.
        Detects position jumps inconsistent with inertial trajectory.
        innovation < 5 m → 1.0
        innovation 5–50 m → linear 0.0–1.0
        innovation > 50 m → 0.0
        Missing → 0.9 (inertial not yet available — mild penalty)
        """
        if m.pose_innovation_m is None:
            return 0.9
        inno = abs(m.pose_innovation_m)
        if inno < 5.0:
            return 1.0
        if inno > 50.0:
            return 0.0
        return 1.0 - (inno - 5.0) / 45.0

    def _score_ew_impact(self, m: GNSSMeasurement) -> float:
        """
        EW impact contextual score.
        High jammer confidence → lower GNSS trust (contextual amplification).
        ew_jammer_confidence 0.0 → 1.0 (no EW context)
        ew_jammer_confidence 1.0 → 0.0 (confirmed jammer zone)
        """
        return 1.0 - float(np.clip(m.ew_jammer_confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Spoof detection (hard criterion — Part Two V7 §1.8.2)
    # ------------------------------------------------------------------

    def _detect_spoof(self, m: GNSSMeasurement) -> bool:
        """
        Hard spoof criterion: multi-constellation delta > 15 m AND C/N0 drop > 6 dB.
        Either alone is suspicious; both together is high-confidence spoof.

        Returns True if spoof is detected — bypasses hysteresis,
        drives trust_score below 0.1 immediately.
        """
        # Constellation delta criterion
        const_fail = False
        if m.gps_position_enu is not None and m.glonass_position_enu is not None:
            delta = float(np.linalg.norm(m.gps_position_enu - m.glonass_position_enu))
            const_fail = delta > self._cfg.constellation_delta_red_m

        # C/N0 drop criterion (relative — needs baseline; use threshold < 30 dB-Hz as proxy)
        cn0_fail = False
        if m.cn0_db is not None:
            cn0_fail = m.cn0_db < (38.0 - self._cfg.cn0_drop_spoof_db)  # < 32 dB-Hz

        return const_fail and cn0_fail

    # ------------------------------------------------------------------
    # Hysteresis state machine
    # ------------------------------------------------------------------

    def _score_to_state(self, score: float) -> BIMState:
        if score >= self._cfg.green_threshold:
            return BIMState.GREEN
        if score >= self._cfg.amber_threshold:
            return BIMState.AMBER
        return BIMState.RED

    def _apply_hysteresis(
        self, candidate: BIMState, spoof_alert: bool
    ) -> Tuple[BIMState, int, bool]:
        """
        Apply 3-sample hysteresis before confirming a state transition.
        Spoof alert bypasses hysteresis — immediate RED.

        Returns (confirmed_state, hysteresis_count, state_changed).
        """
        if spoof_alert:
            changed = self._state != BIMState.RED
            self._state     = BIMState.RED
            self._candidate = BIMState.RED
            self._hyst_count = self._cfg.hysteresis_count
            return BIMState.RED, self._hyst_count, changed

        if candidate == self._candidate:
            self._hyst_count += 1
        else:
            # New candidate — reset counter, count this sample as first
            self._candidate  = candidate
            self._hyst_count = 1

        state_changed = False
        if self._hyst_count >= self._cfg.hysteresis_count:
            if candidate != self._state:
                state_changed = True
                self._state   = candidate

        return self._state, self._hyst_count, state_changed

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset to initial GREEN state. Use between simulation runs."""
        self._state      = BIMState.GREEN
        self._candidate  = BIMState.GREEN
        self._hyst_count = 0
        self._score_history.clear()
        self._last_output = None

    @property
    def state(self) -> BIMState:
        return self._state

    @property
    def score_history(self) -> List[float]:
        return list(self._score_history)

    def ekf_noise_multiplier(self, trust_score: float) -> float:
        """
        Standalone helper: compute EKF R scaling from a trust score.
        R_GNSS = R_nominal / trust_score, capped at 10×.
        """
        if trust_score <= 0.0:
            return self._cfg.ekf_scale_cap
        return min(1.0 / trust_score, self._cfg.ekf_scale_cap)

    def __repr__(self) -> str:
        return (
            f"BIM(state={self._state.value}, "
            f"hyst={self._hyst_count}/{self._cfg.hysteresis_count})"
        )
