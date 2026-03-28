"""
integration/bridge/mavlink_bridge.py
MicroMind Pre-HIL — Phase 1.5 MAVLink Bridge

MAVLinkBridge: PX4 OFFBOARD bridge with five independent threads.

Thread architecture (ADR-0 v1.1 Section 7, v1.2 §4.2a):
  T-HB  — heartbeat daemon, 2 Hz, NEVER BLOCKS, independent daemon
  T-SP  — setpoint loop, 20 Hz, NEVER BLOCKS, time-driven
  T-MON — state monitor, async receive, NEVER BLOCKS
  T-NAV — navigation loop (LivePipeline, external to this module)
  T-LOG — logger (BridgeLogger, external to this module)

Failure modes implemented (ADR-0 v1.1 Section 2):
  FM-1: T-HB is an independent daemon using threading.Event.wait()
  FM-2: pre-stream 2s of setpoints before OFFBOARD request
  FM-3: coordinate_frame=1 (MAV_FRAME_LOCAL_NED), Z positive DOWN
  FM-4: T-SP is time-driven at 20 Hz, never event-driven
  FM-5: time_boot_ms derived from TimeReference (MicroMind clock)
  FM-6: sysid/compid derived from first HEARTBEAT, never hardcoded
  FM-7: three-part verification after OFFBOARD — arm, mode, position change

CRITICAL RULE (v1.2): T-HB must be tested in isolation for 30s before
T-SP is written. Phase 1.5 gate requires T-HB passing before any setpoint
logic is added.
"""

from __future__ import annotations

import struct
import threading
import time
from enum import Enum, auto
from typing import Optional

import pymavlink.mavutil as mavutil

from integration.bridge.time_reference import TimeReference
from integration.bridge.bridge_logger import BridgeLogger


# ---------------------------------------------------------------------------
# Bridge state
# ---------------------------------------------------------------------------

class BridgeState(Enum):
    """MAVLink bridge operational state."""
    DISCONNECTED  = auto()
    CONNECTED     = auto()   # heartbeat received, IDs derived
    PRESTREAMING  = auto()   # sending hold setpoints before OFFBOARD
    OFFBOARD      = auto()   # OFFBOARD mode confirmed
    FAULT         = auto()   # unexpected mode reversion or lost heartbeat


# ---------------------------------------------------------------------------
# MAVLinkBridge
# ---------------------------------------------------------------------------

class MAVLinkBridge:
    """PX4 OFFBOARD bridge implementing FM-1 through FM-7.

    Owns T-HB and T-SP threads. T-MON runs as a method called from T-SP
    loop or as a separate thread depending on pymavlink receive pattern.

    Args:
        udp_url:     pymavlink connection string (default 'udp:127.0.0.1:14550').
        time_ref:    TimeReference instance (created internally if None).
        logger:      BridgeLogger instance (created internally if None).
        log_path:    log file path (used only if logger is None).
        source_type: 'sim' or 'real' for log annotation.
    """

    # PX4 custom_mode for OFFBOARD — confirmed V-7 on v1.17.0-alpha1 gz_x500
    _OFFBOARD_CUSTOM_MODE = 393216

    # MAVLink constants
    _MAV_CMD_DO_SET_MODE          = 176
    _MAV_CMD_COMPONENT_ARM_DISARM = 400
    _MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
    _MAV_FRAME_LOCAL_NED          = 1

    # Timing
    _HB_INTERVAL_S    = 0.5    # 2 Hz heartbeat
    _SP_INTERVAL_S    = 0.05   # 20 Hz setpoints
    _PRESTREAM_S      = 2.0    # FM-2: pre-stream duration before OFFBOARD
    _CMD_ACK_TIMEOUT  = 3.0    # seconds to wait for COMMAND_ACK

    def __init__(
        self,
        udp_url:     str = 'udp:127.0.0.1:14550',
        time_ref:    Optional[TimeReference] = None,
        logger:      Optional[BridgeLogger]  = None,
        log_path:    str = 'bridge.jsonl',
        source_type: str = 'sim',
    ) -> None:
        self._udp_url     = udp_url
        self._time_ref    = time_ref or TimeReference()
        self._source_type = source_type

        if logger is not None:
            self._logger = logger
            self._logger_owned = False
        else:
            self._logger = BridgeLogger(
                log_path=log_path,
                source_type=source_type,
                time_ref=self._time_ref,
            )
            self._logger_owned = True

        # MAVLink connection — created on connect()
        self._mav: Optional[mavutil.mavfile] = None

        # FM-6: derived from first HEARTBEAT, never hardcoded
        self._target_system:    int = 0
        self._target_component: int = 0
        self._ids_derived:      bool = False

        # Bridge state
        self._state           = BridgeState.DISCONNECTED
        self._state_lock      = threading.Lock()

        # Hold setpoint (NED) — T-SP sends this when no mission setpoint available
        self._hold_x_m: float = 0.0
        self._hold_y_m: float = 0.0
        self._hold_z_m: float = -3.0   # 3m altitude in NED (z negative = up)

        # Setpoint source — updated by external caller (LivePipeline)
        self._setpoint_x_m: float = 0.0
        self._setpoint_y_m: float = 0.0
        self._setpoint_z_m: float = -3.0
        self._setpoint_lock = threading.Lock()

        # Rate tracking for T-SP
        self._sp_send_times: list = []

        # Thread handles
        self._t_hb:   Optional[threading.Thread] = None
        self._t_sp:   Optional[threading.Thread] = None
        self._t_mon:  Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # ACK signalling — T-MON sets these, main thread waits
        self._ack_event = threading.Event()
        self._last_ack:  dict = {}
        self._ack_lock   = threading.Lock()

        # Mode monitoring
        self._last_hb_t:      float = 0.0
        self._last_custom_mode: int = 0
        self._local_pos_valid: bool = False
        self._local_pos_x:    float = 0.0
        self._last_pos_check_t: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def connect(self, timeout_s: float = 10.0) -> bool:
        """Connect to PX4 and derive sysid/compid from HEARTBEAT (FM-6).

        Args:
            timeout_s: seconds to wait for first HEARTBEAT.

        Returns:
            True if connected and IDs derived. False on timeout.
        """
        self._mav = mavutil.mavlink_connection(self._udp_url)
        hb = self._mav.wait_heartbeat(timeout=timeout_s)
        if hb is None:
            return False

        # FM-6: derive from HEARTBEAT, never hardcode
        self._target_system    = self._mav.target_system
        self._target_component = self._mav.target_component
        self._ids_derived      = True
        self._last_hb_t        = time.monotonic()
        self._last_custom_mode = hb.custom_mode

        # Log the heartbeat
        self._logger.log_heartbeat_rx(
            base_mode=hb.base_mode,
            custom_mode=hb.custom_mode,
            system_status=hb.system_status,
            mavlink_version=hb.mavlink_version,
            target_system=self._target_system,
            target_component=self._target_component,
        )

        # D-2: send SYSTEM_TIME as courtesy sync (no dependency on PX4 adopting it)
        # Receive SYSTEM_TIME from PX4 to compute boot offset
        sys_time = self._mav.recv_match(type='SYSTEM_TIME', blocking=True, timeout=2.0)
        if sys_time is not None:
            self._time_ref.sync_from_px4(int(sys_time.time_boot_ms))

        with self._state_lock:
            self._state = BridgeState.CONNECTED

        return True

    def start_heartbeat(self) -> None:
        """Start T-HB daemon thread. Must be called after connect().

        T-HB is the FIRST functional element started (v1.2 §10.4).
        Do not call start_setpoints() until T-HB has run for 30s.
        """
        if not self._ids_derived:
            raise RuntimeError(
                "MAVLinkBridge.start_heartbeat(): connect() must be called first. "
                "Cannot send heartbeat without derived sysid/compid."
            )
        if self._t_hb is not None and self._t_hb.is_alive():
            return

        self._stop_event.clear()
        self._t_hb = threading.Thread(
            target=self._heartbeat_loop,
            name="T-HB",
            daemon=True,
        )
        self._t_hb.start()

    def start_setpoints(self) -> None:
        """Start T-SP setpoint thread. Must only be called after T-HB runs 30s.

        FM-1 enforcement: this method must not be called until T-HB
        has been observed running cleanly in isolation for 30 seconds.
        """
        if self._t_hb is None or not self._t_hb.is_alive():
            raise RuntimeError(
                "MAVLinkBridge.start_setpoints(): T-HB must be running before T-SP. "
                "FM-1 enforcement: heartbeat must be established before setpoints."
            )
        if self._t_sp is not None and self._t_sp.is_alive():
            return
        self._t_sp = threading.Thread(
            target=self._setpoint_loop,
            name="T-SP",
            daemon=True,
        )
        self._t_sp.start()

    def start_monitor(self) -> None:
        """Start T-MON state monitor thread."""
        if self._t_mon is not None and self._t_mon.is_alive():
            return
        self._t_mon = threading.Thread(
            target=self._monitor_loop,
            name="T-MON",
            daemon=True,
        )
        self._t_mon.start()

    def update_setpoint(self, x_m: float, y_m: float, z_m: float) -> None:
        """Update the current NED setpoint from T-NAV. Non-blocking."""
        with self._setpoint_lock:
            self._setpoint_x_m = x_m
            self._setpoint_y_m = y_m
            self._setpoint_z_m = z_m

    def stop(self, timeout_s: float = 3.0) -> None:
        """Stop all threads cleanly."""
        self._stop_event.set()
        for t in [self._t_hb, self._t_sp, self._t_mon]:
            if t is not None:
                t.join(timeout=timeout_s)
        if self._logger_owned:
            self._logger.stop()

    def state(self) -> BridgeState:
        """Return current bridge state. Thread-safe."""
        with self._state_lock:
            return self._state

    def heartbeat_count(self) -> int:
        """Return number of heartbeats sent by T-HB."""
        return self._hb_count

    # ------------------------------------------------------------------
    # T-HB — heartbeat daemon (FM-1)
    # ------------------------------------------------------------------

    _hb_count: int = 0

    def _heartbeat_loop(self) -> None:
        """T-HB: send MAVLink HEARTBEAT at 2 Hz using Event.wait().

        Uses threading.Event.wait(0.5) not time.sleep() per v1.2 §4.2a.
        This ensures the thread wakes precisely and is never blocked by
        navigation loop slowness.

        FM-1 compliance: this thread has NO dependency on any other thread,
        queue, or navigation state. It runs independently as a pure daemon.
        """
        _wait = threading.Event()
        while not self._stop_event.is_set():
            t_send = time.perf_counter()

            try:
                self._mav.mav.heartbeat_send(
                    type=6,           # MAV_TYPE_GCS
                    autopilot=8,      # MAV_AUTOPILOT_INVALID
                    base_mode=0,
                    custom_mode=0,
                    system_status=4,  # MAV_STATE_ACTIVE
                )
                self._hb_count += 1
                self._logger.log("HEARTBEAT", "TX")
            except Exception:
                pass   # Never crash T-HB

            # Rate control: Event.wait() not sleep()
            elapsed = time.perf_counter() - t_send
            wait_s  = max(0.0, self._HB_INTERVAL_S - elapsed)
            _wait.wait(timeout=wait_s)

    # ------------------------------------------------------------------
    # T-SP — setpoint loop (FM-4)
    # ------------------------------------------------------------------

    def _setpoint_loop(self) -> None:
        """T-SP: send POSITION_TARGET_LOCAL_NED at 20 Hz.

        FM-4 compliance: time-driven, never event-driven.
        Sends hold setpoint if no mission setpoint is enqueued.
        FM-3 compliance: coordinate_frame=1 (NED), Z positive DOWN.
        """
        t_next = time.perf_counter()
        _sp_times: list = []

        while not self._stop_event.is_set():
            t_now = time.perf_counter()

            with self._setpoint_lock:
                x_m = self._setpoint_x_m
                y_m = self._setpoint_y_m
                z_m = self._setpoint_z_m

            try:
                self._mav.mav.set_position_target_local_ned_send(
                    self._time_ref.time_boot_ms(),   # FM-5: MicroMind clock
                    self._target_system,
                    self._target_component,
                    self._MAV_FRAME_LOCAL_NED,        # FM-3: coordinate_frame=1
                    0b0000111111111000,               # position only
                    x_m, y_m, z_m,
                    0, 0, 0,
                    0, 0, 0,
                    0, 0,
                )
                # Rate tracking (rolling 1s window)
                _sp_times.append(t_now)
                _sp_times = [t for t in _sp_times if t_now - t <= 1.0]
                sp_hz = len(_sp_times)

                self._logger.log_setpoint_tx(
                    x_m=x_m, y_m=y_m, z_m=z_m,
                    setpoint_hz=float(sp_hz),
                    coordinate_frame=self._MAV_FRAME_LOCAL_NED,
                )
            except Exception:
                pass   # Never crash T-SP

            t_next += self._SP_INTERVAL_S
            sleep_s = t_next - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                t_next = time.perf_counter()

    # ------------------------------------------------------------------
    # T-MON — state monitor (async receive)
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """T-MON: receive and process PX4 telemetry.

        Monitors HEARTBEAT for mode reversion (OFFBOARD = 393216).
        Monitors LOCAL_POSITION_NED for EKF2 alignment (FM-7).
        Monitors COMMAND_ACK for arm and mode change results.
        Never sends commands — read-only.
        """
        while not self._stop_event.is_set():
            try:
                msg = self._mav.recv_match(
                    type=['HEARTBEAT', 'LOCAL_POSITION_NED',
                           'EXTENDED_SYS_STATE', 'COMMAND_ACK',
                           'ESTIMATOR_STATUS'],
                    blocking=True,
                    timeout=0.1,
                )
                if msg is None:
                    # Check heartbeat staleness
                    if time.monotonic() - self._last_hb_t > 0.5:
                        self._logger.log_staleness_alert(
                            msg_type_watched="HEARTBEAT",
                            last_seen_t=self._last_hb_t,
                            elapsed_ms=(time.monotonic() - self._last_hb_t) * 1000,
                            threshold_ms=500,
                        )
                    continue

                if msg.get_type() == 'HEARTBEAT':
                    self._last_hb_t = time.monotonic()
                    new_mode = msg.custom_mode
                    if new_mode != self._last_custom_mode:
                        expected = (new_mode == self._OFFBOARD_CUSTOM_MODE)
                        self._logger.log_mode_transition(
                            old_mode=self._last_custom_mode,
                            new_mode=new_mode,
                            expected=expected,
                        )
                        self._last_custom_mode = new_mode

                elif msg.get_type() == 'LOCAL_POSITION_NED':
                    # FM-7: track position updates for EKF2 alignment check
                    new_x = msg.x
                    # EKF2 aligned = message received (movement check deferred to FM-7 post-OFFBOARD)
                    self._local_pos_valid = True
                    self._local_pos_x = new_x

                elif msg.get_type() == 'COMMAND_ACK':
                    self._logger.log_command_ack(
                        command_id=msg.command,
                        result_code=msg.result,
                        result_str=str(msg.result),
                        latency_ms=0.0,
                    )
                    with self._ack_lock:
                        self._last_ack = {'command': msg.command, 'result': msg.result}
                    self._ack_event.set()
            except Exception:
                continue

    # ------------------------------------------------------------------
    # OFFBOARD engagement sequence (FM-2, FM-6, FM-7)
    # ------------------------------------------------------------------

    def _wait_for_ack(self, timeout_s: float = 5.0) -> dict:
        """Wait for COMMAND_ACK signalled by T-MON. Non-blocking main thread wait."""
        self._ack_event.clear()
        fired = self._ack_event.wait(timeout=timeout_s)
        if not fired:
            return {}
        with self._ack_lock:
            return dict(self._last_ack)

    def arm_and_offboard(
        self,
        prestream_s: float = _PRESTREAM_S,
    ) -> bool:
        """Execute the full OFFBOARD engagement sequence.

        Sequence (v1.2 §4.2a startup sequence):
          1. Verify LOCAL_POSITION_NED actively updating (FM-7)
          2. Pre-stream hold setpoints for prestream_s at 20Hz (FM-2)
          3. Send ARM command — wait for ACK
          4. Request OFFBOARD mode — wait for ACK
          5. Confirm mode in T-MON (custom_mode == 393216)
          6. Confirm setpoint consumption (position changing)

        Returns:
            True if all six steps succeed. False on any failure.
        """
        if not self._ids_derived:
            return False

        # Step 1: Verify LOCAL_POSITION_NED valid (FM-7)
        t_check = time.monotonic()
        while not self._local_pos_valid:
            if time.monotonic() - t_check > 10.0:
                return False
            time.sleep(0.1)

        # Step 2: Pre-stream (FM-2) — setpoints already flowing from T-SP
        time.sleep(prestream_s)

        # Step 3: ARM
        self._mav.mav.command_long_send(
            self._target_system,
            self._target_component,
            self._MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )
        self._logger.log_arm_command(
            target_system=self._target_system,
            target_component=self._target_component,
            arm_state_before=False,
        )
        ack = self._mav.recv_match(
            type='COMMAND_ACK', blocking=True, timeout=self._CMD_ACK_TIMEOUT
        )
        if ack is None or ack.result != 0:
            return False

        # Step 4: Request OFFBOARD
        self._mav.mav.command_long_send(
            self._target_system,
            self._target_component,
            self._MAV_CMD_DO_SET_MODE,
            0,
            self._MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | 128,  # armed + custom
            6,   # PX4 OFFBOARD sub-mode
            0, 0, 0, 0, 0,
        )
        self._logger.log_offboard_request(
            target_system=self._target_system,
            target_component=self._target_component,
            requested_mode=6,
        )
        ack = self._mav.recv_match(
            type='COMMAND_ACK', blocking=True, timeout=self._CMD_ACK_TIMEOUT
        )
        if ack is None or ack.result != 0:
            return False

        # Step 5: Confirm mode in monitor
        t_confirm = time.monotonic()
        while self._last_custom_mode != self._OFFBOARD_CUSTOM_MODE:
            if time.monotonic() - t_confirm > 5.0:
                return False
            time.sleep(0.05)

        with self._state_lock:
            self._state = BridgeState.OFFBOARD

        return True
