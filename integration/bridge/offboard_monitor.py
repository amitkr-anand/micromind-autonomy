"""
integration/bridge/offboard_monitor.py
MicroMind Pre-HIL — PX4-01 OFFBOARD Continuity Monitor

Instruments OFFBOARD continuity and setpoint dispatch rate per SRS §6.1
and IT-PX4-01.  Pure Python — no pymavlink dependency.  Instantiated by
MAVLinkBridge; also imported directly by tests.

Responsibilities (PX4-01, EC-01):
  - Track total OFFBOARD loss time over mission duration.
  - Compute offboard_continuity_percent on demand.
  - Count OFFBOARD_LOSS events.
  - Measure setpoint dispatch rate over a 1 s rolling window.
  - Log SETPOINT_RATE_LOW (WARNING) if rate < 20 Hz.
  - On link recovery (gap resolved): log OFFBOARD_RESTORED with
    gap_duration_ms and stale_setpoints_discarded=True (SRS §6.1 §3.4).

All timestamps via clock_fn only (§1.4 — no time.time()).
Must NOT read raw sensor data or modify navigation state vectors (§1.3).

References:
    SRS §6.1        IT-PX4-01 — OFFBOARD continuity ≥ 99.5%, rate ≥ 20 Hz
    SRS §3.4        Stale setpoint discard on link recovery
    Code Governance Manual v3.2  §1.3 (PX4Bridge — detection/logging only)
    Code Governance Manual v3.2  §1.4 (no time.time(); clock_fn only)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class PX4ContinuityMonitor:
    """
    OFFBOARD continuity tracker and setpoint rate measurer (PX4-01, SRS §6.1).

    Usage:
        event_log = []
        monitor = PX4ContinuityMonitor(
            event_log=event_log,
            clock_fn=lambda: int(time_ref.time_boot_ms()),
        )

        # On OFFBOARD loss:
        monitor.record_offboard_loss()
        # On OFFBOARD recovery:
        monitor.record_offboard_restored()
        # Each setpoint dispatch in T-SP:
        monitor.record_setpoint()
        # At 1 Hz: log rate
        monitor.log_setpoint_rate()
        # At mission end:
        continuity = monitor.compute_continuity(total_mission_ms)
    """

    RATE_WINDOW_MS  = 1000      # rolling window for setpoint rate measurement
    MIN_SETPOINT_HZ = 20.0      # SRS §6.1 minimum required dispatch rate

    def __init__(
        self,
        event_log: List[Dict[str, Any]],
        clock_fn:  Callable[[], int],
    ) -> None:
        """
        Args:
            event_log: Shared list receiving OFFBOARD_LOSS, OFFBOARD_RESTORED,
                       SETPOINT_RATE_LOG, and SETPOINT_RATE_LOW log dicts.
            clock_fn:  Zero-argument callable returning current time in ms
                       (mission_clock/time_ref only — §1.4).
        """
        self._event_log: List[Dict[str, Any]] = event_log
        self._clock_fn:  Callable[[], int]    = clock_fn

        # Continuity tracking
        self._total_offboard_loss_ms: int          = 0
        self._offboard_loss_count:    int          = 0
        self._offboard_loss_start_ms: Optional[int] = None
        self._offboard_active:        bool         = False

        # Setpoint rate tracking (sorted list of ms timestamps)
        self._sp_timestamps: List[int] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def offboard_loss_count(self) -> int:
        """Number of OFFBOARD loss events recorded."""
        return self._offboard_loss_count

    @property
    def total_offboard_loss_ms(self) -> int:
        """Cumulative OFFBOARD loss time in milliseconds."""
        return self._total_offboard_loss_ms

    # ------------------------------------------------------------------
    # Continuity accounting
    # ------------------------------------------------------------------

    def record_offboard_loss(self, ts_ms: Optional[int] = None) -> None:
        """
        Record the start of an OFFBOARD loss event.

        Logs OFFBOARD_LOSS at WARNING level.  Increments offboard_loss_count.

        Args:
            ts_ms: Loss event timestamp in ms.  Defaults to clock_fn().
        """
        ts = ts_ms if ts_ms is not None else self._clock_fn()
        self._offboard_loss_start_ms = ts
        self._offboard_loss_count   += 1
        self._offboard_active        = False
        self._event_log.append({
            "event":        "OFFBOARD_LOSS",
            "req_id":       "PX4-01",
            "severity":     "WARNING",
            "module_name":  "PX4Bridge",
            "timestamp_ms": ts,
            "payload":      {"loss_count": self._offboard_loss_count},
        })

    def record_offboard_restored(self, ts_ms: Optional[int] = None) -> None:
        """
        Record OFFBOARD restoration and discard stale setpoints (SRS §3.4).

        Accumulates gap duration into total_offboard_loss_ms.
        Clears the setpoint rate buffer (stale setpoints discarded).
        Logs OFFBOARD_RESTORED at INFO level with gap_duration_ms and
        stale_setpoints_discarded=True.

        Args:
            ts_ms: Restoration timestamp in ms.  Defaults to clock_fn().
        """
        ts      = ts_ms if ts_ms is not None else self._clock_fn()
        gap_ms  = 0
        if self._offboard_loss_start_ms is not None:
            gap_ms                       = ts - self._offboard_loss_start_ms
            self._total_offboard_loss_ms += gap_ms
            self._offboard_loss_start_ms = None

        self._offboard_active = True

        # Discard stale setpoints from rate buffer on link recovery (SRS §3.4)
        self._sp_timestamps.clear()

        self._event_log.append({
            "event":        "OFFBOARD_RESTORED",
            "req_id":       "PX4-01",
            "severity":     "INFO",
            "module_name":  "PX4Bridge",
            "timestamp_ms": ts,
            "payload":      {
                "gap_duration_ms":          gap_ms,
                "stale_setpoints_discarded": True,
            },
        })

    def compute_continuity(self, total_mission_ms: int) -> float:
        """
        Compute offboard_continuity_percent over total_mission_ms.

        offboard_continuity_percent =
            (total_mission_ms - total_offboard_loss_ms)
            / total_mission_ms * 100

        SRS §6.1 threshold: ≥ 99.5 % (≤ 9 s loss per 30-minute mission).

        Args:
            total_mission_ms: Total elapsed mission time in ms.

        Returns:
            Continuity percentage in [0.0, 100.0].
        """
        if total_mission_ms <= 0:
            return 100.0
        loss_ms = self._total_offboard_loss_ms
        return (total_mission_ms - loss_ms) / total_mission_ms * 100.0

    # ------------------------------------------------------------------
    # Setpoint rate measurement
    # ------------------------------------------------------------------

    def record_setpoint(self, ts_ms: Optional[int] = None) -> None:
        """
        Record a single setpoint dispatch event (called from T-SP per iteration).

        Args:
            ts_ms: Dispatch timestamp in ms.  Defaults to clock_fn().
        """
        ts = ts_ms if ts_ms is not None else self._clock_fn()
        self._sp_timestamps.append(ts)

    def measure_rate_hz(self, ts_ms: Optional[int] = None) -> float:
        """
        Measure setpoint dispatch rate over the last RATE_WINDOW_MS milliseconds.

        Prunes timestamps outside the window as a side-effect.

        Args:
            ts_ms: Current timestamp in ms.  Defaults to clock_fn().

        Returns:
            Setpoints per second (Hz) in the rolling window.
        """
        ts     = ts_ms if ts_ms is not None else self._clock_fn()
        cutoff = ts - self.RATE_WINDOW_MS
        self._sp_timestamps = [t for t in self._sp_timestamps if t >= cutoff]
        return len(self._sp_timestamps) / (self.RATE_WINDOW_MS / 1000.0)

    def log_setpoint_rate(self, ts_ms: Optional[int] = None) -> None:
        """
        Log the current setpoint rate at DEBUG level (intended for 1 Hz heartbeat).

        Appends SETPOINT_RATE_LOG with rate, offboard_active, and stale age.
        Appends SETPOINT_RATE_LOW (WARNING) if rate < MIN_SETPOINT_HZ.

        Args:
            ts_ms: Current timestamp in ms.  Defaults to clock_fn().
        """
        ts       = ts_ms if ts_ms is not None else self._clock_fn()
        rate_hz  = self.measure_rate_hz(ts_ms=ts)
        stale_ms = 0
        if self._sp_timestamps:
            stale_ms = ts - max(self._sp_timestamps)

        self._event_log.append({
            "event":        "SETPOINT_RATE_LOG",
            "req_id":       "PX4-01",
            "severity":     "DEBUG",
            "module_name":  "PX4Bridge",
            "timestamp_ms": ts,
            "payload":      {
                "setpoint_rate_hz":    rate_hz,
                "offboard_active":     self._offboard_active,
                "stale_setpoint_age_ms": stale_ms,
            },
        })

        if rate_hz < self.MIN_SETPOINT_HZ:
            self._event_log.append({
                "event":        "SETPOINT_RATE_LOW",
                "req_id":       "PX4-01",
                "severity":     "WARNING",
                "module_name":  "PX4Bridge",
                "timestamp_ms": ts,
                "payload":      {
                    "setpoint_rate_hz": rate_hz,
                    "required_hz":      self.MIN_SETPOINT_HZ,
                },
            })
