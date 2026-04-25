"""
integration/bridge/reboot_detector.py
MicroMind / NanoCorteX — PX4 Reboot Detector (SRS IT-PX4-02, PX4-04)

Detects PX4 reboot by monitoring MAVLink HEARTBEAT sequence-number resets.

PX4 increments its outgoing sequence number (0–255 rolling) with every
MAVLink message.  On a genuine reboot the counter resets to zero, producing
a large backward jump in sequence space.  This differs from the normal
0 → 255 → 0 rollover because:

  Rollover:  last_seq ≈ 255, new_seq ≈ 0
             forward_dist = (new – last) % 256  → small (≈ 1–4)
             backward_dist = (last – new) % 256 → large (≈ 252–255)
             → NOT detected (backward > threshold BUT forward ≤ threshold)

  Reboot:    last_seq arbitrary, new_seq ≈ 0
             backward_dist = (last – new) % 256 → large
             forward_dist  = (new – last) % 256 → also large (≈ 200+)
             → DETECTED (both backward > threshold AND forward > threshold)

Design:
    - Pure Python; no pymavlink dependency — safe to import in SIL environment.
    - MAVLinkBridge instantiates RebootDetector and calls feed() from T-MON on
      each received HEARTBEAT.  Detection logic is here; bridge does I/O only.
    - Logging authority boundary (§1.3): this module logs the raw detection event.
      Mission recovery decisions (D8a gate) are made by MissionManager.

Log event emitted on detection (appended to event_log):
    {
        "event":        "PX4_REBOOT_DETECTED",
        "req_id":       "PX4-04",
        "severity":     "WARNING",
        "module_name":  "MAVLinkBridge",
        "timestamp_ms": <int>,
        "payload":      {"elapsed_detection_ms": <int>},
    }

References:
    SRS IT-PX4-02  PX4-04
    Code Governance Manual v3.2 §1.3 (no mission logic in PX4 Bridge)
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


class RebootDetector:
    """
    Stateful MAVLink sequence-number reset detector.

    Maintains the last observed HEARTBEAT sequence number and wall-clock
    timestamp.  On each call to feed(), checks whether the new sequence
    number represents a genuine reboot (large backward jump in sequence space,
    not the normal 255→0 rollover).

    Detection criterion:
        backward_dist = (last_seq – new_seq) % 256  > threshold  (default 5)
        forward_dist  = (new_seq – last_seq) % 256  > threshold

        Both conditions must hold.  The rollover case (last≈255, new≈0) has
        forward_dist ≈ 1–4, which fails the second condition.

    Usage (from MAVLinkBridge T-MON):
        detector = RebootDetector(event_log=shared_list,
                                  clock_fn=time_ref.time_boot_ms)
        # on each HEARTBEAT receipt:
        detected = detector.feed(seq=msg.get_seq())
        if detected:
            <notify MissionManager>

    Usage in SIL tests (no pymavlink):
        log = []
        det = RebootDetector(event_log=log)
        det.feed(seq=50)         # establish baseline
        det.feed(seq=40)         # backward jump of 10 → DETECTED
        assert any(e["event"] == "PX4_REBOOT_DETECTED" for e in log)
    """

    _SEQ_MAX = 256

    def __init__(
        self,
        event_log:     Optional[List[Dict[str, Any]]] = None,
        clock_fn:      Optional[Callable[[], int]]    = None,
        seq_threshold: int = 5,
    ):
        """
        Args:
            event_log:      External list to receive PX4_REBOOT_DETECTED events.
                            If None an internal list is created.
            clock_fn:       Zero-argument callable returning current time as
                            integer milliseconds (e.g. TimeReference.time_boot_ms).
                            If None, wall-clock milliseconds are used.
            seq_threshold:  Minimum backward distance to trigger detection (default 5).
                            Both backward_dist and forward_dist must exceed this value.
        """
        self._event_log     = event_log if event_log is not None else []
        self._clock_fn      = clock_fn  or (lambda: int(time.monotonic() * 1000))
        self._seq_threshold = seq_threshold

        # Per-instance state (§1.3: no globals)
        self._last_rx_seq:      int   = -1      # -1 = not yet seen
        self._last_hb_wall_t:   float = 0.0     # wall-clock time of last feed()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def event_log(self) -> List[Dict[str, Any]]:
        """Snapshot of all PX4_REBOOT_DETECTED events logged by this detector."""
        return list(self._event_log)

    @property
    def last_rx_seq(self) -> int:
        """Last observed sequence number (-1 if no heartbeat seen yet)."""
        return self._last_rx_seq

    def feed(self, seq: int, wall_t: Optional[float] = None) -> bool:
        """
        Process a received HEARTBEAT sequence number.

        On first call: records seq as baseline, returns False.
        On subsequent calls: checks for sequence-number reset.

        Args:
            seq:    MAVLink sequence number of the received HEARTBEAT (0–255).
            wall_t: Wall-clock time of receipt (time.monotonic()).
                    Defaults to time.monotonic() if not provided.

        Returns:
            True  — PX4_REBOOT_DETECTED event was appended to event_log.
            False — Normal heartbeat (or first call).
        """
        if wall_t is None:
            wall_t = time.monotonic()

        detected = False

        if self._last_rx_seq >= 0:
            backward_dist = (self._last_rx_seq - seq) % self._SEQ_MAX
            forward_dist  = (seq - self._last_rx_seq) % self._SEQ_MAX

            if backward_dist > self._seq_threshold and forward_dist > self._seq_threshold:
                # Genuine reboot: large backward jump, NOT the normal rollover
                elapsed_ms = int((wall_t - self._last_hb_wall_t) * 1000)
                recovery_start_ms = self._clock_fn()
                self._event_log.append({
                    "event":              "PX4_REBOOT_DETECTED",
                    "req_id":             "PX4-04",
                    "severity":           "WARNING",
                    "module_name":        "MAVLinkBridge",
                    "timestamp_ms":       recovery_start_ms,
                    "seq_reset_value":    seq,
                    "recovery_start_ms":  recovery_start_ms,
                    "payload":            {"elapsed_detection_ms": elapsed_ms},
                })
                detected = True

        # Update state regardless of detection
        self._last_rx_seq    = seq
        self._last_hb_wall_t = wall_t
        return detected

    def reset(self) -> None:
        """Reset detector state (e.g. on deliberate planned reboot)."""
        self._last_rx_seq    = -1
        self._last_hb_wall_t = 0.0
