"""
integration/bridge/time_reference.py
MicroMind Pre-HIL — Phase 1.5 MAVLink Bridge

TimeReference: MicroMind monotonic clock with PX4 boot offset.

ADR-0 v1.1 D-2: MicroMind owns the internal monotonic mission clock.
All outgoing MAVLink timestamps derive from this clock. SYSTEM_TIME
sent at connect as courtesy sync only. No dependency on PX4 adopting it.

All outgoing MAVLink messages carry time_boot_ms derived here.
"""

from __future__ import annotations

import time


class TimeReference:
    """MicroMind monotonic clock with PX4 boot-time offset.

    MicroMind runs its own monotonic clock from process start.
    On connecting to PX4, we receive a SYSTEM_TIME message and
    compute the offset between PX4 boot time and our monotonic clock.
    All outgoing MAVLink messages use time_boot_ms derived from this offset.

    Usage:
        ref = TimeReference()
        ref.sync_from_px4(px4_time_boot_ms=12345)  # on SYSTEM_TIME receipt
        ms = ref.time_boot_ms()   # use in every outgoing MAVLink message
    """

    def __init__(self) -> None:
        self._t0_monotonic: float = time.monotonic()
        self._px4_offset_ms: float = 0.0
        self._synced: bool = False

    def sync_from_px4(self, px4_time_boot_ms: int) -> None:
        """Compute offset from received PX4 SYSTEM_TIME.

        Called once on connection when SYSTEM_TIME is received.
        Records the difference between PX4 boot milliseconds and
        our local elapsed milliseconds at the moment of receipt.

        Args:
            px4_time_boot_ms: time_boot_ms from PX4 SYSTEM_TIME message.
        """
        local_elapsed_ms = (time.monotonic() - self._t0_monotonic) * 1000.0
        self._px4_offset_ms = px4_time_boot_ms - local_elapsed_ms
        self._synced = True

    def time_boot_ms(self) -> int:
        """Return current time_boot_ms for outgoing MAVLink messages.

        If sync_from_px4() has not been called, returns elapsed milliseconds
        from process start (still consistent; just not PX4-relative).

        Returns:
            int: milliseconds suitable for MAVLink time_boot_ms field.
        """
        elapsed_ms = (time.monotonic() - self._t0_monotonic) * 1000.0
        return int(elapsed_ms + self._px4_offset_ms)

    def monotonic_s(self) -> float:
        """Return current MicroMind monotonic time in seconds.

        This is the authoritative clock for all internal timestamps
        (BridgeLogger t_monotonic, Setpoint.t, health snapshots).

        Returns:
            float: monotonic seconds from process start.
        """
        return time.monotonic() - self._t0_monotonic

    @property
    def is_synced(self) -> bool:
        """True if sync_from_px4() has been called at least once."""
        return self._synced

    @property
    def px4_offset_ms(self) -> float:
        """The computed PX4 boot offset in milliseconds. 0.0 if not synced."""
        return self._px4_offset_ms
