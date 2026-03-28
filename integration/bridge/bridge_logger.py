"""
integration/bridge/bridge_logger.py
MicroMind Pre-HIL — Phase 1.5 MAVLink Bridge

BridgeLogger: TX/RX event logger for the MAVLink bridge.

Implements the logging contract from ADR-0 v1.1 Section 5.
All log entries are JSON-lines written by T-LOG (async queue consumer).
The logger is the primary diagnostic tool for Phase 1.5 failures.

Thread safety: log() is called from multiple threads (T-HB, T-SP, T-MON).
It is non-blocking — events are enqueued, never written inline.
T-LOG consumes the queue and writes to disk independently.

ADR-0 v1.1 D-2: t_monotonic uses TimeReference.monotonic_s().
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Log entry dataclasses — one per event type
# ---------------------------------------------------------------------------

@dataclass
class BridgeLogEntry:
    """Base log entry. All events carry these five fields."""
    t_monotonic: float
    msg_type:    str
    direction:   str          # 'TX' or 'RX'
    seq:         Optional[int]
    source_type: str          # 'sim' or 'real'
    extra:       dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BridgeLogger
# ---------------------------------------------------------------------------

class BridgeLogger:
    """Non-blocking bridge event logger writing JSON-lines via T-LOG.

    Usage:
        logger = BridgeLogger(log_path="bridge_run.jsonl", source_type="sim")
        logger.start()
        logger.log_heartbeat_rx(base_mode=29, custom_mode=0, ...)
        logger.log_setpoint_tx(x_m=0., y_m=0., z_m=-5., ...)
        logger.stop()

    Args:
        log_path:    path to JSON-lines output file.
        source_type: 'sim' or 'real' — annotates all log entries.
        queue_maxsize: T-LOG queue bound (default 10000).
        time_ref:    TimeReference instance. If None, uses time.monotonic().
    """

    def __init__(
        self,
        log_path: str,
        source_type: str = 'sim',
        queue_maxsize: int = 10_000,
        time_ref=None,
    ) -> None:
        self._log_path    = log_path
        self._source_type = source_type
        self._time_ref    = time_ref
        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._thread: Optional[threading.Thread] = None
        self._stop_event  = threading.Event()
        self._drop_count: int = 0
        self._write_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start T-LOG consumer thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._log_loop,
            name="T-LOG",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        """Flush remaining entries and stop T-LOG thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    # ------------------------------------------------------------------
    # Public log methods — called from T-HB, T-SP, T-MON (non-blocking)
    # ------------------------------------------------------------------

    def log(self, msg_type: str, direction: str,
            seq: Optional[int] = None, **extra) -> None:
        """Enqueue a generic log entry. Non-blocking — drops if queue full.

        Args:
            msg_type:  MAVLink message type string.
            direction: 'TX' or 'RX'.
            seq:       MAVLink sequence number or None.
            **extra:   event-specific fields (see ADR-0 v1.1 Section 5.2).
        """
        t = self._now()
        entry = BridgeLogEntry(
            t_monotonic=t,
            msg_type=msg_type,
            direction=direction,
            seq=seq,
            source_type=self._source_type,
            extra=extra,
        )
        try:
            self._queue.put_nowait(entry)
        except queue.Full:
            self._drop_count += 1

    def log_heartbeat_rx(self, base_mode: int, custom_mode: int,
                          system_status: int, mavlink_version: int,
                          target_system: int, target_component: int,
                          seq: Optional[int] = None) -> None:
        """Log HEARTBEAT received from PX4 (FM-6: sysid/compid source)."""
        self.log("HEARTBEAT", "RX", seq=seq,
                 base_mode=base_mode, custom_mode=custom_mode,
                 system_status=system_status, mavlink_version=mavlink_version,
                 derived_target_system=target_system,
                 derived_target_component=target_component)

    def log_setpoint_tx(self, x_m: float, y_m: float, z_m: float,
                         setpoint_hz: float, coordinate_frame: int = 1,
                         seq: Optional[int] = None) -> None:
        """Log outgoing NED setpoint. coordinate_frame must be 1."""
        self.log("SET_POSITION_TARGET_LOCAL_NED", "TX", seq=seq,
                 x_m=x_m, y_m=y_m, z_m=z_m,
                 setpoint_hz=setpoint_hz,
                 coordinate_frame=coordinate_frame)

    def log_offboard_request(self, target_system: int, target_component: int,
                              requested_mode: int, seq: Optional[int] = None) -> None:
        """Log outgoing OFFBOARD mode request."""
        self.log("CMD_DO_SET_MODE_OFFBOARD", "TX", seq=seq,
                 target_system=target_system,
                 target_component=target_component,
                 requested_mode=requested_mode)

    def log_arm_command(self, target_system: int, target_component: int,
                         arm_state_before: bool, seq: Optional[int] = None) -> None:
        """Log outgoing ARM command."""
        self.log("CMD_COMPONENT_ARM_DISARM", "TX", seq=seq,
                 target_system=target_system,
                 target_component=target_component,
                 arm_state_before=arm_state_before)

    def log_command_ack(self, command_id: int, result_code: int,
                         result_str: str, latency_ms: float,
                         seq: Optional[int] = None) -> None:
        """Log received COMMAND_ACK."""
        self.log("COMMAND_ACK", "RX", seq=seq,
                 command_id=command_id,
                 result_code=result_code,
                 result_str=result_str,
                 latency_ms=latency_ms)

    def log_mode_transition(self, old_mode: int, new_mode: int,
                             expected: bool, seq: Optional[int] = None) -> None:
        """Log PX4 mode transition observed by T-MON."""
        direction_str = "expected" if expected else "unexpected"
        self.log("MODE_TRANSITION", "RX", seq=seq,
                 old_mode=old_mode, new_mode=new_mode,
                 direction=direction_str)

    def log_staleness_alert(self, msg_type_watched: str,
                             last_seen_t: float, elapsed_ms: float,
                             threshold_ms: float) -> None:
        """Log T-MON staleness alert for a watched message."""
        self.log("STALENESS_ALERT", "RX",
                 msg_type_watched=msg_type_watched,
                 last_seen_t=last_seen_t,
                 elapsed_ms=elapsed_ms,
                 threshold_ms=threshold_ms)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def drop_count(self) -> int:
        """Number of log entries dropped due to full queue."""
        return self._drop_count

    @property
    def write_count(self) -> int:
        """Number of log entries successfully written to disk."""
        return self._write_count

    # ------------------------------------------------------------------
    # T-LOG consumer thread
    # ------------------------------------------------------------------

    def _log_loop(self) -> None:
        """T-LOG: consume queue and write JSON-lines to disk."""
        with open(self._log_path, 'w') as fh:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    entry = self._queue.get(timeout=0.05)
                    d = asdict(entry)
                    d.update(d.pop('extra', {}))
                    fh.write(json.dumps(d) + '\n')
                    self._write_count += 1
                except queue.Empty:
                    continue
                except Exception:
                    continue

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _now(self) -> float:
        if self._time_ref is not None:
            return self._time_ref.monotonic_s()
        return time.monotonic()
