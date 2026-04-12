"""
integration/camera/nadir_camera_bridge.py
MicroMind / NanoCorteX — Nadir Camera Frame Bridge

NAV-03: Camera frame ingestion pipeline.
Subscribes to Gazebo gz.transport camera topic and delivers frames to
registered consumers as numpy uint8 arrays.

Sensor substitution contract:
    Current:     Gazebo gz.transport camera topic (/nadir_camera/image)
    HIL:         ROS2 sensor_msgs/Image or direct V4L2/MIPI camera driver
    Interface:   identical — numpy uint8 array delivered via callback
    Breaking change at HIL: NO
    Contract:    docs/interfaces/eo_day_contract.yaml

References:
    SRS v1.3 NAV-03
    docs/interfaces/eo_day_contract.yaml
"""
from __future__ import annotations

import collections
import threading
from typing import Callable, List, Optional, Tuple

import numpy as np


class NadirCameraFrameBridge:
    """
    NAV-03: Camera frame ingestion bridge.

    Subscribes to Gazebo camera topic and delivers frames to registered
    consumers. Designed for gz.transport (Gazebo Harmonic) but the
    consumer interface is identical at HIL — only the subscriber backend
    changes.

    At HIL: replace _start_gazebo_subscriber() with real camera driver
    (ROS2 subscriber or V4L2 reader). Consumer callbacks unchanged.

    Thread safety: all public methods are thread-safe.
    """

    _RATE_WINDOW_S: float = 5.0   # window for rolling rate estimate
    _LOG_EVERY_N_FRAMES: int = 50  # log CAMERA_FRAME_RECEIVED every N frames

    def __init__(
        self,
        topic:             str   = '/nadir_camera/image',
        expected_rate_hz:  float = 5.0,
        clock_fn:          Optional[Callable[[], int]] = None,
    ) -> None:
        """
        topic            : Gazebo camera topic name
        expected_rate_hz : nominal frame delivery rate (for rate degradation alert)
        clock_fn         : callable returning mission time in ms (int).
                           No time.time() permitted per governance §7.2.
        """
        self._topic           = topic
        self._expected_rate   = expected_rate_hz
        self._clock_fn        = clock_fn

        self._consumers: List[Callable] = []
        self._lock        = threading.Lock()

        self._frame_count: int                            = 0
        self._latest_frame: Optional[Tuple[np.ndarray, int]] = None
        self._running: bool                               = False
        self._sub_thread: Optional[threading.Thread]     = None

        # Rolling timestamp buffer for rate estimation
        self._ts_buffer: collections.deque = collections.deque(maxlen=int(expected_rate_hz * self._RATE_WINDOW_S) + 1)

        # Structured event log
        self._event_log: list = []

    # ── Public API ────────────────────────────────────────────────────────────

    def register_consumer(self, consumer: Callable) -> None:
        """
        Register a callback that receives (frame: np.ndarray, timestamp_ms: int)
        when a new frame arrives. Thread-safe.
        """
        with self._lock:
            self._consumers.append(consumer)

    def start(self) -> None:
        """
        Begin subscribing. Non-blocking. Frames delivered to registered consumers
        via callback. Attempts gz.transport subscription; falls back to no-op if
        Gazebo is not running (allows unit tests without Gazebo).
        """
        with self._lock:
            if self._running:
                return
            self._running = True

        self._sub_thread = threading.Thread(
            target=self._subscriber_loop,
            daemon=True,
            name="NadirCamBridge",
        )
        self._sub_thread.start()

    def stop(self) -> None:
        """Unsubscribe and release resources."""
        with self._lock:
            self._running = False

    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, int]]:
        """
        Return (frame: np.ndarray, timestamp_ms: int) of most recent frame,
        or None if no frame has been received yet.
        """
        with self._lock:
            return self._latest_frame

    def get_frame_rate(self) -> float:
        """Return measured frame delivery rate in Hz over last 5 seconds."""
        with self._lock:
            buf = list(self._ts_buffer)
        if len(buf) < 2:
            return 0.0
        span_ms = buf[-1] - buf[0]
        if span_ms <= 0:
            return 0.0
        return (len(buf) - 1) / (span_ms / 1000.0)

    def inject_frame(self, frame: np.ndarray, timestamp_ms: int) -> None:
        """
        Inject a frame directly (for testing without Gazebo, or for
        alternative camera driver backends).

        This is the HIL camera driver insertion point.
        """
        self._on_frame_received(frame, timestamp_ms)

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def event_log(self) -> list:
        return list(self._event_log)

    # ── Internal: Gazebo subscriber loop ─────────────────────────────────────

    def _subscriber_loop(self) -> None:
        """
        Attempt to connect to Gazebo gz.transport and subscribe to camera topic.
        If gz.transport bindings are not available, thread exits quietly.
        Frames are decoded and forwarded via _on_frame_received().
        """
        try:
            import gz.transport13 as gz_transport  # type: ignore
            import gz.msgs10.image_pb2 as gz_image  # type: ignore
        except ImportError:
            # gz.transport Python bindings not installed — no Gazebo available.
            # Bridge runs in inject-only mode (unit test / HIL driver mode).
            return

        node = gz_transport.Node()

        def _gz_callback(msg):
            if not self._running:
                return
            try:
                frame = _decode_gz_image(msg)
                ts_ms = self._clock_ms()
                self._on_frame_received(frame, ts_ms)
            except Exception:
                pass  # never crash the callback thread

        node.subscribe(gz_image.Image, self._topic, _gz_callback)

        # Keep thread alive while running
        import time
        while self._running:
            time.sleep(0.1)

    def _on_frame_received(self, frame: np.ndarray, timestamp_ms: int) -> None:
        """Process an incoming frame: update state, call consumers, log."""
        with self._lock:
            self._frame_count += 1
            self._latest_frame = (frame, timestamp_ms)
            self._ts_buffer.append(timestamp_ms)
            count = self._frame_count
            consumers = list(self._consumers)

        # Deliver to consumers outside lock
        for cb in consumers:
            try:
                cb(frame, timestamp_ms)
            except Exception:
                pass

        # Periodic log every 50 frames
        if count % self._LOG_EVERY_N_FRAMES == 0:
            rate = self.get_frame_rate()
            self._event_log.append({
                "event":        "CAMERA_FRAME_RECEIVED",
                "module_name":  "NadirCameraFrameBridge",
                "req_id":       "NAV-03",
                "severity":     "INFO",
                "timestamp_ms": timestamp_ms,
                "payload": {
                    "frame_count": count,
                    "rate_hz":     round(rate, 2),
                    "timestamp_ms": timestamp_ms,
                },
            })

        # Rate degradation check
        rate = self.get_frame_rate()
        if count > 10 and rate < self._expected_rate * 0.5:
            self._event_log.append({
                "event":        "CAMERA_RATE_DEGRADED",
                "module_name":  "NadirCameraFrameBridge",
                "req_id":       "NAV-03",
                "severity":     "WARNING",
                "timestamp_ms": timestamp_ms,
                "payload": {
                    "measured_rate_hz":  round(rate, 2),
                    "expected_rate_hz":  self._expected_rate,
                    "frame_count":       count,
                },
            })

    def _clock_ms(self) -> int:
        if self._clock_fn is not None:
            return int(self._clock_fn())
        return 0


def _decode_gz_image(msg) -> np.ndarray:
    """
    Decode a Gazebo gz.msgs.Image protobuf message into a numpy uint8 array.
    Handles R8G8B8 (3-channel) and L_INT8 (grayscale) formats.
    """
    data = bytes(msg.data)
    h, w = msg.height, msg.width
    if msg.pixel_format_type == 3:   # R8G8B8
        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
    else:
        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
    return arr
