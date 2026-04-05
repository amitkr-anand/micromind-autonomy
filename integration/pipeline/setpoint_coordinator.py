"""
integration/pipeline/setpoint_coordinator.py
MicroMind Pre-HIL — SetpointCoordinator

Closes RC-11c wiring gap (Sprint D code review 4972110).
Does not modify LivePipeline or MAVLinkBridge.
External coordinator pattern.

Wires LivePipeline.setpoint_queue → MAVLinkBridge.update_setpoint().

LivePipeline (T-NAV) enqueues Setpoint objects at 200 Hz.
MAVLinkBridge (T-SP) reads _setpoint_x_m/_y_m/_z_m at 20 Hz.
These were previously unconnected. SetpointCoordinator bridges them
without touching either class.

Design:
    Runs a daemon thread at poll_hz (default 50 Hz).
    Each poll cycle drains the entire setpoint queue, keeps only the
    most recent entry (discards stale intermediate setpoints), and
    calls bridge.update_setpoint(). This matches T-SP's 20 Hz
    consumption rate — the bridge never receives a stale position.

Threading contract:
    - Coordinator thread is a daemon — stops automatically when main
      thread exits.
    - setpoints_forwarded / setpoints_dropped are protected by a lock
      and may be read from any thread.
    - Does not call any method on LivePipeline other than reading
      .setpoint_queue (a public property).
    - Does not call any method on MAVLinkBridge other than
      .update_setpoint() (a public method).
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from integration.pipeline.live_pipeline import LivePipeline
    from integration.bridge.mavlink_bridge import MAVLinkBridge


class SetpointCoordinator:
    """
    External coordinator wiring LivePipeline.setpoint_queue to
    MAVLinkBridge.update_setpoint().

    Args:
        pipeline:  LivePipeline instance (must be started before coordinator).
        bridge:    MAVLinkBridge instance (or any object with update_setpoint()).
        poll_hz:   Coordinator polling rate in Hz (default 50 Hz).
    """

    def __init__(
        self,
        pipeline: 'LivePipeline',
        bridge:   'MAVLinkBridge',
        poll_hz:  float = 50.0,
    ) -> None:
        self._pipeline         = pipeline
        self._bridge           = bridge
        self._poll_interval_s  = 1.0 / poll_hz

        self._thread:      threading.Thread | None = None
        self._stop_event   = threading.Event()

        self._forwarded: int = 0
        self._dropped:   int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start coordinator daemon thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="T-COORD",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Signal coordinator to stop and wait for thread exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    # ------------------------------------------------------------------
    # Metrics (thread-safe)
    # ------------------------------------------------------------------

    @property
    def setpoints_forwarded(self) -> int:
        """Number of setpoints successfully forwarded to bridge."""
        with self._lock:
            return self._forwarded

    @property
    def setpoints_dropped(self) -> int:
        """Number of stale intermediate setpoints discarded (kept only most recent)."""
        with self._lock:
            return self._dropped

    # ------------------------------------------------------------------
    # Coordinator loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """T-COORD: drain queue at poll_hz, forward most recent to bridge."""
        q = self._pipeline.setpoint_queue

        while not self._stop_event.is_set():
            sp = None
            dropped_this_cycle = 0

            # Drain all pending entries — keep only the most recent
            while True:
                try:
                    candidate = q.get_nowait()
                    if sp is not None:
                        # Previous candidate is now stale
                        dropped_this_cycle += 1
                    sp = candidate
                except queue.Empty:
                    break

            if sp is not None:
                try:
                    self._bridge.update_setpoint(sp.x_m, sp.y_m, sp.z_m)
                    with self._lock:
                        self._forwarded += 1
                        self._dropped   += dropped_this_cycle
                except Exception:
                    pass   # Never crash T-COORD

            self._stop_event.wait(timeout=self._poll_interval_s)
