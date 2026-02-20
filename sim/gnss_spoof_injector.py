"""
sim/gnss_spoof_injector.py
MicroMind / NanoCorteX — GNSS Spoof Injector
Sprint S2 Deliverable 2 of 3

Simulates GNSS attack scenarios for BIM SIL validation.
Injects controlled position offsets, C/N0 drops, and Doppler
deviations to exercise BIM detection logic.

Attack types implemented:
  POSITION_OFFSET   — gradual or step position displacement (m)
  CONSTELLATION     — GPS/GLONASS divergence injection
  CN0_DROP          — carrier-to-noise ratio degradation
  DOPPLER           — Doppler deviation injection
  COMBINED          — Position + C/N0 (high-confidence spoof)
  JAMMING           — C/N0 drop only (no position offset)

BIM acceptance gate (Sprint S2):
  Spoof injection → trust_score < 0.1 within 250 ms.
  State machine → GNSS_DENIED. Logged.

References: Part Two V7 §1.8, TechReview §2.3, BCMP-1 EW-03
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

from core.bim.bim import GNSSMeasurement


# ---------------------------------------------------------------------------
# Attack type
# ---------------------------------------------------------------------------

class AttackType(str, Enum):
    CLEAN              = "CLEAN"            # No attack — baseline
    POSITION_OFFSET    = "POSITION_OFFSET"  # GPS position displaced
    CONSTELLATION      = "CONSTELLATION"    # GPS vs GLONASS divergence
    CN0_DROP           = "CN0_DROP"         # Signal strength drop
    DOPPLER            = "DOPPLER"          # Doppler deviation
    COMBINED           = "COMBINED"         # Position + C/N0 (hard spoof)
    JAMMING            = "JAMMING"          # Broadband — C/N0 drop only


# ---------------------------------------------------------------------------
# Attack profile
# ---------------------------------------------------------------------------

@dataclass
class AttackProfile:
    """
    Defines a single attack injection event.

    activation_s:   Mission time (seconds) when attack starts.
    duration_s:     How long the attack lasts (None = indefinite).
    ramp_s:         Ramp-up time for gradual attacks (0 = step).
    """
    attack_type:        AttackType
    activation_s:       float
    duration_s:         Optional[float]  = None   # None = indefinite
    ramp_s:             float            = 0.0    # step by default

    # POSITION_OFFSET / CONSTELLATION parameters
    position_offset_m:  float = 0.0      # magnitude of GPS position offset
    constellation_delta_m: float = 0.0  # GPS vs GLONASS divergence

    # CN0_DROP parameters
    cn0_drop_db:        float = 0.0      # drop from nominal (dB)

    # DOPPLER parameters
    doppler_dev_ms:     float = 0.0      # Doppler deviation (m/s)

    label:              str   = ""


# ---------------------------------------------------------------------------
# Nominal GNSS state (healthy baseline)
# ---------------------------------------------------------------------------

@dataclass
class NominalGNSSState:
    """Represents a clean, healthy GNSS signal."""
    position_enu:       np.ndarray = field(
                            default_factory=lambda: np.array([0.0, 0.0, 3200.0])
                        )
    pdop:               float = 1.8
    cn0_db:             float = 42.0    # healthy C/N0
    doppler_deviation:  float = 0.02    # small nominal deviation
    tracked_sats:       int   = 10
    pose_innovation_m:  float = 1.5     # small healthy innovation
    ew_jammer_conf:     float = 0.0


# ---------------------------------------------------------------------------
# Spoof injector
# ---------------------------------------------------------------------------

class GNSSSpoofInjector:
    """
    Generates GNSSMeasurement objects with controlled attack injection.

    Stateful: tracks mission time and applies active attack profiles.

    Usage:
        injector = GNSSSpoofInjector()
        injector.add_attack(AttackProfile(
            attack_type       = AttackType.COMBINED,
            activation_s      = 5.0,
            duration_s        = 30.0,
            position_offset_m = 250.0,
            cn0_drop_db       = 12.0,
            label             = "BCMP-1 terminal spoofer"
        ))

        for t in timeline:
            measurement = injector.generate(t, true_position_enu)
            bim_output  = bim.evaluate(measurement)
    """

    def __init__(self, nominal: Optional[NominalGNSSState] = None):
        self._nominal  = nominal or NominalGNSSState()
        self._attacks: List[AttackProfile] = []
        self._rng      = np.random.default_rng(seed=42)  # deterministic

    def add_attack(self, profile: AttackProfile) -> None:
        self._attacks.append(profile)

    def clear_attacks(self) -> None:
        self._attacks.clear()

    # ------------------------------------------------------------------
    # Main generation method
    # ------------------------------------------------------------------

    def generate(
        self,
        mission_time_s: float,
        true_position_enu: Optional[np.ndarray] = None,
        ew_jammer_confidence: float = 0.0,
    ) -> GNSSMeasurement:
        """
        Generate a GNSSMeasurement for a given mission time.
        Applies any active attack profiles.

        Args:
            mission_time_s:     Current simulation time (seconds from T=0).
            true_position_enu:  True aircraft position (ENU metres).
                                Defaults to nominal state position.
            ew_jammer_confidence: EW Engine confidence (0.0–1.0).

        Returns:
            GNSSMeasurement ready for BIM.evaluate().
        """
        pos = (
            true_position_enu.copy()
            if true_position_enu is not None
            else self._nominal.position_enu.copy()
        )

        # Start with clean nominal values
        pdop              = self._nominal.pdop
        cn0               = self._nominal.cn0_db
        doppler_dev       = self._nominal.doppler_deviation
        pose_innovation   = self._nominal.pose_innovation_m
        gps_pos           = pos + self._noise(0.5)   # GPS position (with noise)
        glonass_pos       = pos + self._noise(0.5)   # GLONASS (independent noise)

        # Accumulate active attacks
        for attack in self._attacks:
            if not self._is_active(attack, mission_time_s):
                continue

            ramp = self._ramp_factor(attack, mission_time_s)

            if attack.attack_type == AttackType.POSITION_OFFSET:
                offset = attack.position_offset_m * ramp
                gps_pos = gps_pos + np.array([offset, 0.0, 0.0])

            elif attack.attack_type == AttackType.CONSTELLATION:
                delta = attack.constellation_delta_m * ramp
                gps_pos = gps_pos + np.array([delta, 0.0, 0.0])
                # GLONASS stays clean — creates divergence

            elif attack.attack_type == AttackType.CN0_DROP:
                cn0 = max(0.0, cn0 - attack.cn0_drop_db * ramp)

            elif attack.attack_type == AttackType.DOPPLER:
                doppler_dev = attack.doppler_dev_ms * ramp

            elif attack.attack_type == AttackType.COMBINED:
                # Position offset (GPS only) + C/N0 drop
                offset = attack.position_offset_m * ramp
                gps_pos = gps_pos + np.array([offset, 0.0, 0.0])
                cn0 = max(0.0, cn0 - attack.cn0_drop_db * ramp)
                # Doppler also deviates under combined attack
                doppler_dev = max(doppler_dev, 0.8 * ramp)

            elif attack.attack_type == AttackType.JAMMING:
                # C/N0 crushed; PDOP degrades (fewer tracked sats)
                cn0   = max(0.0, cn0 - attack.cn0_drop_db * ramp)
                pdop  = min(10.0, pdop + 5.0 * ramp)

        # Add small measurement noise to all outputs
        pdop            = max(1.0, pdop + self._noise_scalar(0.1))
        cn0             = max(0.0, cn0 + self._noise_scalar(0.5))
        doppler_dev     = abs(doppler_dev + self._noise_scalar(0.01))
        pose_innovation = max(0.0, pose_innovation + self._noise_scalar(0.3))

        return GNSSMeasurement(
            pdop                  = float(pdop),
            cn0_db                = float(cn0),
            tracked_satellites    = self._nominal.tracked_sats,
            gps_position_enu      = gps_pos,
            glonass_position_enu  = glonass_pos,
            doppler_deviation_ms  = float(doppler_dev),
            pose_innovation_m     = float(pose_innovation),
            ew_jammer_confidence  = float(np.clip(ew_jammer_confidence, 0.0, 1.0)),
            timestamp_s           = mission_time_s,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_active(self, attack: AttackProfile, t: float) -> bool:
        if t < attack.activation_s:
            return False
        if attack.duration_s is not None:
            if t > attack.activation_s + attack.duration_s:
                return False
        return True

    def _ramp_factor(self, attack: AttackProfile, t: float) -> float:
        """0.0→1.0 ramp over ramp_s seconds after activation."""
        if attack.ramp_s <= 0.0:
            return 1.0
        elapsed = t - attack.activation_s
        return float(np.clip(elapsed / attack.ramp_s, 0.0, 1.0))

    def _noise(self, sigma: float) -> np.ndarray:
        return self._rng.normal(0.0, sigma, size=3)

    def _noise_scalar(self, sigma: float) -> float:
        return float(self._rng.normal(0.0, sigma))

    def active_attacks(self, t: float) -> List[str]:
        """Return labels of currently active attacks at time t."""
        return [
            a.label or a.attack_type.value
            for a in self._attacks
            if self._is_active(a, t)
        ]

    def __repr__(self) -> str:
        return f"GNSSSpoofInjector(attacks={len(self._attacks)})"


# ---------------------------------------------------------------------------
# Pre-built BCMP-1 attack sequence (matches bcmp1_scenario.py)
# ---------------------------------------------------------------------------

def build_bcmp1_attack_sequence() -> GNSSSpoofInjector:
    """
    Return a GNSSSpoofInjector pre-loaded with the BCMP-1 attack sequence:
      - JMR-01: Broadband jamming T+18 to T+40 min
      - JMR-02: Broadband jamming T+25 to T+55 min
      - SPF-01: Combined spoof at terminal T+52 min (250 m offset)
    """
    injector = GNSSSpoofInjector()

    injector.add_attack(AttackProfile(
        attack_type   = AttackType.JAMMING,
        activation_s  = 18 * 60,
        duration_s    = 22 * 60,
        cn0_drop_db   = 18.0,
        ramp_s        = 30.0,
        label         = "JMR-01 broadband jamming",
    ))

    injector.add_attack(AttackProfile(
        attack_type   = AttackType.JAMMING,
        activation_s  = 25 * 60,
        duration_s    = 30 * 60,
        cn0_drop_db   = 15.0,
        ramp_s        = 20.0,
        label         = "JMR-02 broadband jamming",
    ))

    injector.add_attack(AttackProfile(
        attack_type           = AttackType.COMBINED,
        activation_s          = 52 * 60,
        duration_s            = 13 * 60,
        position_offset_m     = 250.0,
        cn0_drop_db           = 14.0,
        ramp_s                = 0.0,    # step — immediate
        label                 = "SPF-01 terminal precision spoof",
    ))

    return injector
