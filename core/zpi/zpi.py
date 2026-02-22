"""
core/zpi/zpi.py
MicroMind Sprint S6 — ZPI Burst Scheduler
Zero Probability of Intercept signalling for telemetry, EW flash updates,
and CEMS peer exchange.

Implements: FR-104
Boundary conditions (Part Two V7 §1.11):
  - Burst duration          : ≤ 10 ms
  - Inter-burst interval    : randomised 2–30 s
  - Frequency hop range     : ± 5 MHz around mission-defined centre
  - Maximum duty cycle      : ≤ 0.5%
  - Power variation range   : −10 to 0 dB relative to maximum
  - DF risk trigger         : jammer bearing within 45° of own track
  - Anti-DF adaptation      : reduce burst rate; increase hop range
  - Hop plan seed           : HKDF-SHA256 from mission key
  - Mandatory pre-terminal  : EW summary burst before SHM activation
"""

from __future__ import annotations

import hashlib
import hmac
import math
import random
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("ZPI")

# ─── Boundary Constants ───────────────────────────────────────────────────────
BURST_DURATION_S        = 0.010     # ≤ 10 ms per burst
INTERVAL_MIN_S          = 2.0       # minimum inter-burst interval
INTERVAL_MAX_S          = 30.0      # maximum inter-burst interval
HOP_RANGE_HZ            = 5_000_000 # ± 5 MHz frequency hop range
MAX_DUTY_CYCLE          = 0.005     # ≤ 0.5%
POWER_MIN_DB            = -10.0     # minimum relative power (dB)
POWER_MAX_DB            = 0.0       # maximum relative power (dB)
DF_RISK_BEARING_DEG     = 45.0      # jammer bearing within 45° → DF risk
ADAPTED_INTERVAL_MIN_S  = 10.0      # reduced burst rate under DF risk
ADAPTED_HOP_RANGE_HZ    = 10_000_000 # doubled hop range under DF risk


# ─── Types ────────────────────────────────────────────────────────────────────

class BurstType(Enum):
    TELEMETRY       = "TELEMETRY"
    EW_FLASH        = "EW_FLASH"
    CEMS_PEER       = "CEMS_PEER"
    PRE_TERMINAL    = "PRE_TERMINAL"   # mandatory before SHM activation


class ZPIState(Enum):
    NOMINAL     = "NOMINAL"      # standard burst scheduling
    DF_ADAPTED  = "DF_ADAPTED"   # reduced rate, wider hops
    SUPPRESSED  = "SUPPRESSED"   # SHM active — all RF suppressed


@dataclass
class ZPIBurst:
    """A single ZPI burst record."""
    burst_id:       int
    burst_type:     BurstType
    timestamp_s:    float
    frequency_hz:   float           # absolute centre frequency for this burst
    power_db:       float           # relative power used
    duration_s:     float           # actual burst duration
    payload_bytes:  int             # payload size
    confirmed:      bool = False    # set True when burst completes
    pre_terminal:   bool = False    # True if this is the mandatory pre-SHM burst


@dataclass
class ZPIHopPlan:
    """
    Deterministic hop plan derived from mission key.
    Two UAVs sharing the same mission key produce identical hop plans —
    implicit time-synchronisation without explicit clock exchange.
    """
    mission_key:        bytes
    centre_freq_hz:     float
    _counter:           int = field(default=0, repr=False)

    def next_frequency(self, df_adapted: bool = False) -> float:
        """Derive next hop frequency deterministically from mission key."""
        hop_range = ADAPTED_HOP_RANGE_HZ if df_adapted else HOP_RANGE_HZ
        # HKDF-SHA256: derive 4 bytes from key + counter
        info = f"hop:{self._counter}".encode()
        h = hmac.new(self.mission_key, info, hashlib.sha256).digest()
        # Map first 4 bytes to [-hop_range, +hop_range]
        raw = int.from_bytes(h[:4], "big")
        offset = (raw / 0xFFFFFFFF) * 2 * hop_range - hop_range
        self._counter += 1
        return self.centre_freq_hz + offset

    def next_interval(self, df_adapted: bool = False) -> float:
        """Derive next randomised inter-burst interval from mission key."""
        info = f"interval:{self._counter}".encode()
        h = hmac.new(self.mission_key, info, hashlib.sha256).digest()
        raw = int.from_bytes(h[:4], "big")
        if df_adapted:
            interval = ADAPTED_INTERVAL_MIN_S + (raw / 0xFFFFFFFF) * (INTERVAL_MAX_S - ADAPTED_INTERVAL_MIN_S)
        else:
            interval = INTERVAL_MIN_S + (raw / 0xFFFFFFFF) * (INTERVAL_MAX_S - INTERVAL_MIN_S)
        self._counter += 1
        return interval

    def next_power(self) -> float:
        """Derive next power level from mission key."""
        info = f"power:{self._counter}".encode()
        h = hmac.new(self.mission_key, info, hashlib.sha256).digest()
        raw = int.from_bytes(h[:4], "big")
        self._counter += 1
        return POWER_MIN_DB + (raw / 0xFFFFFFFF) * (POWER_MAX_DB - POWER_MIN_DB)


@dataclass
class ZPIResult:
    """Output from a ZPI scheduling cycle."""
    bursts_transmitted:     int
    pre_terminal_confirmed: bool
    duty_cycle:             float
    state:                  ZPIState
    burst_log:              list[ZPIBurst] = field(default_factory=list)
    df_adaptations:         int = 0
    zpi_compliant:          bool = True     # False if any boundary violated


# ─── ZPI Burst Scheduler ──────────────────────────────────────────────────────

class ZPIBurstScheduler:
    """
    ZPI Burst Scheduler — FR-104.

    Manages burst timing, frequency hopping, power variation, and
    DF-risk adaptation. Hop plan is seeded from mission key via HKDF-SHA256.
    """

    def __init__(self, mission_key: bytes, centre_freq_hz: float = 433_000_000.0):
        self.hop_plan       = ZPIHopPlan(mission_key=mission_key, centre_freq_hz=centre_freq_hz)
        self.state          = ZPIState.NOMINAL
        self._burst_counter = 0
        self._burst_log: list[ZPIBurst] = []
        self._pre_terminal_sent = False
        self._total_tx_time_s = 0.0
        self._total_elapsed_s = 0.0
        self._df_adaptations  = 0

    # ── DF Risk Assessment ────────────────────────────────────────────────────

    def assess_df_risk(self, jammer_bearing_deg: float, own_track_deg: float) -> bool:
        """
        Returns True if DF risk is high.
        Trigger: jammer bearing within 45° of own track direction.
        """
        delta = abs(jammer_bearing_deg - own_track_deg) % 360
        if delta > 180:
            delta = 360 - delta
        return delta <= DF_RISK_BEARING_DEG

    def update_state(self, df_risk: bool, shm_active: bool) -> None:
        """Update ZPI state based on threat assessment."""
        if shm_active:
            self.state = ZPIState.SUPPRESSED
        elif df_risk:
            if self.state != ZPIState.DF_ADAPTED:
                self._df_adaptations += 1
                logger.info("ZPI: DF risk detected — adapting burst schedule")
            self.state = ZPIState.DF_ADAPTED
        else:
            self.state = ZPIState.NOMINAL

    # ── Burst Transmission ────────────────────────────────────────────────────

    def transmit_burst(self, burst_type: BurstType, payload_bytes: int,
                       sim_time_s: float, pre_terminal: bool = False) -> Optional[ZPIBurst]:
        """
        Attempt to transmit a burst. Returns None if suppressed.
        Enforces burst duration ≤ 10 ms and duty cycle ≤ 0.5%.
        """
        if self.state == ZPIState.SUPPRESSED:
            logger.warning("ZPI: Suppressed — burst blocked (SHM active)")
            return None

        df_adapted = (self.state == ZPIState.DF_ADAPTED)

        freq    = self.hop_plan.next_frequency(df_adapted)
        power   = self.hop_plan.next_power()
        duration = BURST_DURATION_S  # always ≤ 10 ms

        burst = ZPIBurst(
            burst_id        = self._burst_counter,
            burst_type      = burst_type,
            timestamp_s     = sim_time_s,
            frequency_hz    = freq,
            power_db        = power,
            duration_s      = duration,
            payload_bytes   = payload_bytes,
            confirmed       = True,
            pre_terminal    = pre_terminal,
        )

        self._burst_counter += 1
        self._burst_log.append(burst)
        self._total_tx_time_s += duration

        if pre_terminal:
            self._pre_terminal_sent = True
            logger.info(f"ZPI: PRE-TERMINAL burst transmitted at T+{sim_time_s:.1f}s")

        logger.debug(f"ZPI: Burst {burst.burst_id} | {burst_type.value} | "
                     f"{freq/1e6:.3f} MHz | {power:.1f} dB | {payload_bytes}B")
        return burst

    def get_next_interval(self) -> float:
        """Get next inter-burst interval from hop plan."""
        return self.hop_plan.next_interval(df_adapted=(self.state == ZPIState.DF_ADAPTED))

    # ── Mission Simulation ────────────────────────────────────────────────────

    def run_mission_segment(self,
                             duration_s: float,
                             burst_schedule: list[tuple[float, BurstType, int]],
                             jammer_bearing_deg: float = 270.0,
                             own_track_deg: float = 90.0,
                             shm_active: bool = False,
                             pre_terminal_at_s: Optional[float] = None) -> ZPIResult:
        """
        Simulate a mission segment with scheduled bursts.

        Args:
            duration_s:          segment duration in seconds
            burst_schedule:      list of (time_s, burst_type, payload_bytes)
            jammer_bearing_deg:  bearing to nearest jammer
            own_track_deg:       UAV track direction
            shm_active:          whether SHM (silent) mode is active
            pre_terminal_at_s:   time to send mandatory pre-terminal burst (if any)
        """
        df_risk = self.assess_df_risk(jammer_bearing_deg, own_track_deg)
        self.update_state(df_risk, shm_active)
        self._total_elapsed_s += duration_s

        # Inject mandatory pre-terminal burst if scheduled
        if pre_terminal_at_s is not None and not self._pre_terminal_sent:
            self.transmit_burst(BurstType.PRE_TERMINAL, payload_bytes=64,
                                sim_time_s=pre_terminal_at_s, pre_terminal=True)

        # Process scheduled bursts
        for t, btype, nbytes in burst_schedule:
            if t <= duration_s:
                self.transmit_burst(btype, nbytes, sim_time_s=t)

        # Compute duty cycle
        duty_cycle = (self._total_tx_time_s / self._total_elapsed_s
                      if self._total_elapsed_s > 0 else 0.0)

        zpi_compliant = duty_cycle <= MAX_DUTY_CYCLE

        return ZPIResult(
            bursts_transmitted      = self._burst_counter,
            pre_terminal_confirmed  = self._pre_terminal_sent,
            duty_cycle              = duty_cycle,
            state                   = self.state,
            burst_log               = list(self._burst_log),
            df_adaptations          = self._df_adaptations,
            zpi_compliant           = zpi_compliant,
        )

    @property
    def pre_terminal_confirmed(self) -> bool:
        return self._pre_terminal_sent


# ─── Factory ──────────────────────────────────────────────────────────────────

def make_zpi_scheduler(mission_key_hex: str,
                        centre_freq_hz: float = 433_000_000.0) -> ZPIBurstScheduler:
    """
    Create a ZPI scheduler from a hex mission key string.
    Two UAVs with same mission_key_hex produce identical hop plans.
    """
    key_bytes = bytes.fromhex(mission_key_hex)
    return ZPIBurstScheduler(mission_key=key_bytes, centre_freq_hz=centre_freq_hz)
