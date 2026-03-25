"""
core/fusion/vio_mode.py
=======================
VIO Navigation Mode tracker — S-NEP-08 (L-07, L-08, L-05).

Implements the three-state navigation mode doctrine defined in
S-NEP-07 Rev 3, Section 4D:

    NOMINAL     — VIO active and providing accepted updates
    OUTAGE      — VIO absent (dt_since_last_vio > threshold)
    RESUMPTION  — VIO has returned; stabilisation in progress

ARCHITECTURE CONTRACT (non-negotiable):
    - This module is fusion-layer only.
    - It produces VIONavigationMode as an output signal.
    - The mission FSM (core/state_machine/state_machine.py) is NOT
      modified. It consumes .current_mode as an external input signal.
    - No import of ESKF internals. No estimator behaviour changes.

CONSTANTS:
    All four constants are module-level and configurable at construction.
    They are NOT in core/constants.py — these are fusion-layer parameters,
    not frozen estimator constants.

    VIO_OUTAGE_THRESHOLD_S
        Default 2.0 s. Source: S-NEP-07 L-07.
        Time since last accepted VIO update before OUTAGE is declared.

    VIO_INNOVATION_SPIKE_THRESHOLD_M
        Default 1.0 m. Source: S-NEP-07 L-08.
        Innovation magnitude at first post-outage update that triggers
        an innovation_spike_alert.

    VIO_DRIFT_RATE_CONSERVATIVE_M_S
        Default 0.800 m/s. Source: S-NEP-06 C-07 (MH_03 straight-line
        outage, slope +800 mm/s, R²=0.941 — highest measured drift rate
        across all controlled runs). Used as conservative over-estimate
        for drift_envelope_m during OUTAGE.
        IMPORTANT: drift_envelope_m is a confidence degradation signal,
        NOT a guaranteed position error bound. See S-NEP-07 L-05.

    VIO_RESUMPTION_CYCLES
        Default 1. Source: S-NEP-07 Section 4D doctrine — "at least one
        additional VIO update cycle before full trust is restored."
        Data does not support a higher number. Configurable if programme
        decision changes.
"""

from enum import Enum, auto


# ── Module-level configurable constants ──────────────────────────────────────

VIO_OUTAGE_THRESHOLD_S: float = 2.0
"""Seconds without an accepted VIO update before OUTAGE is declared.
Source: S-NEP-07 L-07. Configurable per mission regime."""

VIO_INNOVATION_SPIKE_THRESHOLD_M: float = 1.0
"""Innovation magnitude (metres) at first post-outage VIO update that
triggers innovation_spike_alert=True. Source: S-NEP-07 L-08."""

VIO_DRIFT_RATE_CONSERVATIVE_M_S: float = 0.800
"""Conservative drift rate estimate used to compute drift_envelope_m
during OUTAGE (m/s). Equals the highest measured drift slope from
S-NEP-06 controlled experiments (C-07: MH_03, +800 mm/s, R²=0.941).
This is an over-estimate by design (S-NEP-07 L-05 conservatism
requirement). drift_envelope_m is a confidence degradation signal,
NOT a guaranteed maximum position error."""

VIO_RESUMPTION_CYCLES: int = 1
"""Minimum number of accepted VIO update cycles required in RESUMPTION
state before transitioning to NOMINAL. Default=1 implements the
doctrine minimum from S-NEP-07 Section 4D. Data does not support
a higher number. Configurable if programme decision changes."""


# ── Mode enumeration ──────────────────────────────────────────────────────────

class VIOMode(Enum):
    """Three navigation modes defined in S-NEP-07 Section 4D."""
    NOMINAL    = auto()    # VIO active, position strongly observed
    OUTAGE     = auto()    # VIO absent, position confidence degrading
    RESUMPTION = auto()    # VIO returned, stabilisation in progress


# ── Main class ────────────────────────────────────────────────────────────────

class VIONavigationMode:
    """
    Fusion-layer VIO navigation mode tracker.

    Tracks the current navigation mode (NOMINAL / OUTAGE / RESUMPTION)
    based on VIO update availability. Computes drift envelope and emits
    innovation spike alerts per S-NEP-07 decisions D-03 and D-04.

    Usage (in fusion runner):
        vio_nav = VIONavigationMode()

        # On each IMU propagation step:
        vio_nav.tick(dt)

        # On each VIO update attempt:
        mode_changed, spike_alert = vio_nav.on_vio_update(
            accepted=not rejected,
            innov_mag=innov_mag
        )

        # Read state for logging and mission layer:
        mode      = vio_nav.current_mode        # VIOMode enum
        mode_str  = vio_nav.current_mode.name   # "NOMINAL" | "OUTAGE" | "RESUMPTION"
        dt_vio    = vio_nav.dt_since_vio        # float seconds
        envelope  = vio_nav.drift_envelope_m    # float | None
    """

    def __init__(
        self,
        outage_threshold_s: float = VIO_OUTAGE_THRESHOLD_S,
        spike_threshold_m: float  = VIO_INNOVATION_SPIKE_THRESHOLD_M,
        drift_rate_m_s: float     = VIO_DRIFT_RATE_CONSERVATIVE_M_S,
        resumption_cycles: int    = VIO_RESUMPTION_CYCLES,
    ) -> None:
        self._outage_threshold_s  = outage_threshold_s
        self._spike_threshold_m   = spike_threshold_m
        self._drift_rate_m_s      = drift_rate_m_s
        self._resumption_cycles   = resumption_cycles

        # Internal state
        self._mode: VIOMode       = VIOMode.NOMINAL
        self._dt_since_vio: float = 0.0          # seconds since last accepted update
        self._resumption_count: int = 0          # accepted updates since entering RESUMPTION
        self._in_outage: bool     = False         # True while mode is OUTAGE or RESUMPTION

        # Counters for summary logging
        self._n_outage_events: int = 0
        self._n_spike_alerts: int  = 0
        self._max_dt_since_vio: float       = 0.0
        self._max_drift_envelope_m: float   = 0.0

    # ── Public interface ──────────────────────────────────────────────────────

    def tick(self, dt: float) -> None:
        """
        Advance internal clock by dt seconds (called once per IMU step).
        Triggers NOMINAL→OUTAGE transition if threshold exceeded.

        Args:
            dt: elapsed time since last call (seconds, must be > 0)
        """
        if dt <= 0.0:
            return

        self._dt_since_vio += dt

        # Update peak tracker
        if self._dt_since_vio > self._max_dt_since_vio:
            self._max_dt_since_vio = self._dt_since_vio

        # NOMINAL → OUTAGE transition
        if (self._mode is VIOMode.NOMINAL
                and self._dt_since_vio >= self._outage_threshold_s):
            self._mode = VIOMode.OUTAGE
            self._in_outage = True
            self._n_outage_events += 1

        # Update drift envelope peak tracker (OUTAGE only)
        if self._mode is VIOMode.OUTAGE:
            env = self._drift_rate_m_s * self._dt_since_vio
            if env > self._max_drift_envelope_m:
                self._max_drift_envelope_m = env

    def on_vio_update(
        self,
        accepted: bool,
        innov_mag: float,
    ) -> tuple[bool, bool]:
        """
        Called on every VIO update attempt, whether accepted or rejected.

        Handles:
          - OUTAGE → RESUMPTION on first accepted update after outage
          - RESUMPTION → NOMINAL after VIO_RESUMPTION_CYCLES accepted updates
          - Innovation spike detection on first post-outage accepted update

        Args:
            accepted:  True if the VIO update was accepted by the ESKF
                       (i.e. not rejected by gating)
            innov_mag: innovation magnitude in metres (‖z − Hx‖)

        Returns:
            (mode_changed, spike_alert)
            mode_changed: True if a mode transition occurred this call
            spike_alert:  True if this is the first post-outage accepted
                         update AND innov_mag > spike threshold.
                         False in all other cases.
        """
        if not accepted:
            return False, False

        spike_alert   = False
        mode_changed  = False

        if self._mode is VIOMode.OUTAGE:
            # First accepted update after outage → RESUMPTION
            # Check for innovation spike
            if innov_mag > self._spike_threshold_m:
                spike_alert = True
                self._n_spike_alerts += 1

            self._mode = VIOMode.RESUMPTION
            self._resumption_count = 1
            self._dt_since_vio = 0.0
            mode_changed = True

        elif self._mode is VIOMode.RESUMPTION:
            self._resumption_count += 1
            self._dt_since_vio = 0.0

            # RESUMPTION → NOMINAL after required cycles
            if self._resumption_count >= self._resumption_cycles:
                self._mode = VIOMode.NOMINAL
                self._resumption_count = 0
                self._in_outage = False
                mode_changed = True

        else:
            # NOMINAL — normal accepted update, reset clock
            self._dt_since_vio = 0.0

        return mode_changed, spike_alert

    # ── Properties (read-only, consumed by fusion_logger and mission layer) ───

    @property
    def current_mode(self) -> VIOMode:
        """Current navigation mode. Read by fusion_logger and mission layer."""
        return self._mode

    @property
    def dt_since_vio(self) -> float:
        """Seconds since last accepted VIO update. 0.0 in NOMINAL."""
        return self._dt_since_vio

    @property
    def drift_envelope_m(self) -> float | None:
        """
        Conservative drift uncertainty estimate (metres). None outside OUTAGE.

        Computed as: VIO_DRIFT_RATE_CONSERVATIVE_M_S × dt_since_vio

        IMPORTANT: This is a confidence degradation signal only.
        It is NOT a guaranteed position error bound.
        Downstream consumers must treat it as such.
        Source: S-NEP-07 L-05, S-NEP-06 C-07.
        """
        if self._mode is not VIOMode.OUTAGE:
            return None
        return self._drift_rate_m_s * self._dt_since_vio

    @property
    def in_outage(self) -> bool:
        """True during OUTAGE or RESUMPTION (position-dependent functions
        should be suppressed while this is True)."""
        return self._in_outage

    # ── Summary statistics (for run-level log fields) ─────────────────────────

    @property
    def n_outage_events(self) -> int:
        """Count of NOMINAL→OUTAGE transitions since construction."""
        return self._n_outage_events

    @property
    def n_spike_alerts(self) -> int:
        """Count of innovation_spike_alert=True events since construction."""
        return self._n_spike_alerts

    @property
    def max_dt_since_vio(self) -> float:
        """Maximum observed outage duration (seconds) since construction."""
        return self._max_dt_since_vio

    @property
    def max_drift_envelope_m(self) -> float:
        """Maximum drift_envelope_m value reached since construction."""
        return self._max_drift_envelope_m
