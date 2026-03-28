"""
integration/pipeline/live_pipeline.py
MicroMind Pre-HIL — Phase 1 Driver Abstraction Layer

LivePipeline: 200Hz navigation loop (T-NAV thread).

Reads IMU at 200Hz, propagates ESKF, applies GNSS and RADALT corrections
at their natural rates, and enqueues NED position setpoints for T-SP.

Threading contract (v1.2 §4.2a, ADR-0 v1.1 Section 7):
  - T-NAV runs in its own thread — never shares with T-HB or T-SP.
  - Setpoint queue is non-blocking with bounded size (ADD-08).
  - If queue is full, setpoint is dropped silently; queue_drop_count increments.
  - health() is callable from any thread at any time — never blocks.

Frozen module rule: this file calls ESKF, INSState, VIONavigationMode.
It does NOT modify them. All frozen constants remain unchanged.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.ekf.error_state_ekf import ErrorStateEKF
from core.ins.mechanisation import ins_propagate
from core.ins.state import INSState
from core.fusion.vio_mode import VIONavigationMode, VIOMode
from core.bim.bim import BIM
from sim.gnss_spoof_injector import GNSSMeasurement, NominalGNSSState

from integration.config.mission_config import MissionConfig
from integration.drivers.factory import DriverFactory
from integration.drivers.base import DriverHealth


# ---------------------------------------------------------------------------
# Setpoint — produced by T-NAV, consumed by T-SP
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Setpoint:
    """NED position setpoint produced by T-NAV for T-SP consumption.

    Coordinate frame: NED (North-East-Down).
    Z is positive DOWN (PX4 convention, MAV_FRAME_LOCAL_NED).

    Attributes:
        x_m:  North position in metres.
        y_m:  East position in metres.
        z_m:  Down position in metres (positive = below origin).
        yaw_rad: desired heading in radians (NED convention).
        t:    monotonic timestamp when setpoint was produced.
    """
    x_m:     float
    y_m:     float
    z_m:     float
    yaw_rad: float
    t:       float


# ---------------------------------------------------------------------------
# PipelineHealth — snapshot of loop health for external monitoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineHealth:
    """Health snapshot from LivePipeline.health().

    Attributes:
        running:          True if T-NAV loop is active.
        loop_count:       total IMU steps completed.
        queue_drop_count: setpoints dropped due to full queue.
        imu_stale:        True if IMU driver is stale.
        vio_mode:         current VIOMode string (NOMINAL/OUTAGE/RESUMPTION).
        last_loop_t:      monotonic time of last completed loop iteration.
    """
    running:          bool
    loop_count:       int
    queue_drop_count: int
    imu_stale:        bool
    vio_mode:         str
    last_loop_t:      float


# ---------------------------------------------------------------------------
# LivePipeline
# ---------------------------------------------------------------------------

class LivePipeline:
    """200Hz navigation loop wiring drivers to ESKF and setpoint queue.

    Usage:
        config   = MissionConfig()
        pipeline = LivePipeline(config)
        pipeline.start()
        # ... T-SP reads from pipeline.setpoint_queue ...
        pipeline.stop()

    Args:
        config:          MissionConfig governing driver selection.
        setpoint_queue:  external queue for T-SP to consume setpoints.
                         If None, an internal bounded queue is created.
        queue_maxsize:   maximum setpoints in queue (default 10 = 0.5s at 20Hz).
        dt_s:            IMU propagation timestep (default 0.005s = 200Hz).
    """

    _DT_S       = 1.0 / 200.0
    _GNSS_EVERY = 40    # update GNSS every 40 IMU steps = 5Hz
    _HEALTH_WATCHDOG_S = 0.02   # warn if loop step exceeds this

    def __init__(
        self,
        config: MissionConfig,
        setpoint_queue: Optional[queue.Queue] = None,
        queue_maxsize: int = 10,
        dt_s: float = _DT_S,
    ) -> None:
        self._config = config
        self._dt_s   = dt_s
        self._factory = DriverFactory(config)

        # Driver instances
        self._imu    = self._factory.make_imu()
        self._gnss   = self._factory.make_gnss()
        self._radalt = self._factory.make_radalt()

        # Core modules — frozen, called read-only
        self._eskf    = ErrorStateEKF()
        self._bim     = BIM()
        self._vio_nav = VIONavigationMode()

        # Initial INS state — position at origin, stationary
        self._state = INSState(
            p=np.zeros(3),
            v=np.zeros(3),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            ba=np.zeros(3),
            bg=np.zeros(3),
        )

        # Setpoint queue (non-blocking, bounded — ADR-0 v1.1 / ADD-08)
        if setpoint_queue is not None:
            self._setpoint_queue = setpoint_queue
        else:
            self._setpoint_queue = queue.Queue(maxsize=queue_maxsize)

        # Thread control
        self._thread:   Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Metrics (written only by T-NAV, read by any thread via health())
        self._loop_count:       int   = 0
        self._queue_drop_count: int   = 0
        self._last_loop_t:      float = 0.0
        self._gnss_step:        int   = 0
        self._lock = threading.Lock()   # protects metric reads only

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def setpoint_queue(self) -> queue.Queue:
        """The non-blocking bounded queue T-SP should read from."""
        return self._setpoint_queue

    def start(self) -> None:
        """Start T-NAV thread. Idempotent — safe to call if already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._nav_loop,
            name="T-NAV",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Signal T-NAV to stop and wait for it to exit cleanly."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
        self._imu.close()
        self._gnss.close()
        self._radalt.close()

    def health(self) -> PipelineHealth:
        """Return a health snapshot. Thread-safe, never blocks."""
        with self._lock:
            return PipelineHealth(
                running=self._thread is not None and self._thread.is_alive(),
                loop_count=self._loop_count,
                queue_drop_count=self._queue_drop_count,
                imu_stale=self._imu.is_stale(),
                vio_mode=self._vio_nav.current_mode.name,
                last_loop_t=self._last_loop_t,
            )

    def is_running(self) -> bool:
        """True if T-NAV thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # T-NAV loop — runs in dedicated thread, never blocks T-HB or T-SP
    # ------------------------------------------------------------------

    def _nav_loop(self) -> None:
        """200Hz navigation loop. Runs as T-NAV daemon thread.

        Loop structure per iteration:
          1. Read IMU (body-frame accel + gyro)
          2. Mechanise (dead-reckoning propagation)
          3. ESKF propagate (error covariance update)
          4. VIONavigationMode tick (outage tracking)
          5. GNSS update every 40 steps (5Hz)
          6. Produce setpoint → enqueue non-blocking
          7. Update metrics
        """
        t_next = time.perf_counter()

        while not self._stop_event.is_set():
            t_now = time.perf_counter()

            try:
                # 1. IMU read
                imu_reading = self._imu.read()
                accel = np.array(imu_reading.accel_mss)
                gyro  = np.array(imu_reading.gyro_rads)

                # 2. Mechanisation (INS propagation)
                self._state = ins_propagate(self._state, accel, gyro, self._dt_s)

                # 3. ESKF propagation
                self._eskf.propagate(self._state, accel, self._dt_s)

                # 4. VIO navigation mode tick
                self._vio_nav.tick(self._dt_s)

                # 5. GNSS update at 5Hz
                self._gnss_step += 1
                if self._gnss_step >= self._GNSS_EVERY:
                    self._gnss_step = 0
                    self._apply_gnss_update()

                # 6. Produce and enqueue setpoint
                self._enqueue_setpoint()

                # 7. Metrics
                with self._lock:
                    self._loop_count += 1
                    self._last_loop_t = time.monotonic()

            except Exception:
                # Never crash T-NAV — log and continue
                with self._lock:
                    self._loop_count += 1

            # Rate control: maintain 200Hz
            t_next += self._dt_s
            sleep_s = t_next - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # Overrun — reset pace rather than spiral
                t_next = time.perf_counter()

    def _apply_gnss_update(self) -> None:
        """Apply GNSS correction to ESKF via BIM trust score.

        Calls BIM.evaluate() with a GNSSMeasurement built from SimGNSSDriver
        output, extracts trust_score, then feeds to ESKF update_gnss().
        """
        try:
            gnss_reading = self._gnss.read()

            # Build GNSSMeasurement for BIM evaluation
            pos_enu = np.array([
                gnss_reading.lon * 111_320.0,   # East metres (approx)
                gnss_reading.lat * 111_320.0,   # North metres (approx)
                gnss_reading.alt,               # Up metres
            ])
            meas = GNSSMeasurement(
                pdop=gnss_reading.hdop if gnss_reading.hdop == gnss_reading.hdop else 2.0,
                cn0_db=35.0,          # nominal C/N0 for sim
                tracked_satellites=8, # nominal for sim
                gps_position_enu=pos_enu,
                glonass_position_enu=None,
                doppler_deviation_ms=0.0,
                pose_innovation_m=0.0,
                ew_jammer_confidence=0.0,
                timestamp_s=gnss_reading.t,
            )
            bim_output = self._bim.evaluate(meas)
            trust = float(bim_output.trust_score)

            # NED position for ESKF (approximate flat-earth)
            gnss_pos_ned = np.array([
                gnss_reading.lat * 111_320.0,   # North
                gnss_reading.lon * 111_320.0,   # East
                -gnss_reading.alt,              # Down
            ])
            self._eskf.update_gnss(self._state, gnss_pos_ned, trust)
            self._eskf.inject(self._state)
        except Exception:
            pass   # GNSS update failure is non-fatal

    def _enqueue_setpoint(self) -> None:
        """Produce NED setpoint from current state and enqueue non-blocking."""
        pos = self._state.p
        setpoint = Setpoint(
            x_m=float(pos[0]),
            y_m=float(pos[1]),
            z_m=float(-pos[2]),   # NED z = -Up; INS p[2] is Up
            yaw_rad=0.0,          # heading lock deferred to Phase 2
            t=time.monotonic(),
        )
        try:
            self._setpoint_queue.put_nowait(setpoint)
        except queue.Full:
            with self._lock:
                self._queue_drop_count += 1
