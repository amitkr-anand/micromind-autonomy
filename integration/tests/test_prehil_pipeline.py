"""
integration/tests/test_prehil_pipeline.py
MicroMind Pre-HIL — Phase 1 LivePipeline Tests

Gates:
  G-PIPE-01: LivePipeline instantiates with default MissionConfig
  G-PIPE-02: Setpoint is a frozen dataclass with correct fields
  G-PIPE-03: PipelineHealth is a frozen dataclass with correct fields
  G-PIPE-04: health() returns not-running before start()
  G-PIPE-05: start() launches T-NAV thread named "T-NAV"
  G-PIPE-06: T-NAV is a daemon thread
  G-PIPE-07: loop_count increments after start
  G-PIPE-08: setpoint_queue receives Setpoint objects after start
  G-PIPE-09: Setpoint fields are finite floats
  G-PIPE-10: stop() terminates T-NAV within timeout
  G-PIPE-11: health() returns not-running after stop()
  G-PIPE-12: queue_drop_count increments when queue is full
  G-PIPE-13: start() is idempotent — safe to call twice
  G-PIPE-14: stop() is safe to call before start()
  G-PIPE-15: 332 SIL gates unaffected (run in separate suite)
"""

import sys, os, time, queue, threading, math
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from integration.pipeline.live_pipeline import LivePipeline, Setpoint, PipelineHealth
from integration.config.mission_config import MissionConfig


def _make_pipeline(queue_maxsize=5):
    """Helper: create a LivePipeline with small queue for fast tests."""
    return LivePipeline(MissionConfig(), queue_maxsize=queue_maxsize)


class TestSetpointDataclass:
    def test_G_PIPE_02_setpoint_frozen(self):
        """G-PIPE-02: Setpoint is a frozen dataclass."""
        s = Setpoint(x_m=1.0, y_m=2.0, z_m=-5.0, yaw_rad=0.0, t=1.0)
        with pytest.raises((AttributeError, TypeError)):
            s.x_m = 99.0

    def test_setpoint_fields(self):
        s = Setpoint(x_m=1.0, y_m=2.0, z_m=-5.0, yaw_rad=0.1, t=3.0)
        assert s.x_m == 1.0
        assert s.y_m == 2.0
        assert s.z_m == -5.0
        assert s.yaw_rad == 0.1
        assert s.t == 3.0


class TestPipelineHealthDataclass:
    def test_G_PIPE_03_pipeline_health_frozen(self):
        """G-PIPE-03: PipelineHealth is a frozen dataclass."""
        h = PipelineHealth(running=False, loop_count=0, queue_drop_count=0,
                           imu_stale=True, vio_mode='NOMINAL', last_loop_t=0.0)
        with pytest.raises((AttributeError, TypeError)):
            h.running = True

    def test_health_fields(self):
        h = PipelineHealth(running=True, loop_count=42, queue_drop_count=1,
                           imu_stale=False, vio_mode='NOMINAL', last_loop_t=1.5)
        assert h.running is True
        assert h.loop_count == 42
        assert h.queue_drop_count == 1
        assert h.imu_stale is False
        assert h.vio_mode == 'NOMINAL'
        assert h.last_loop_t == 1.5


class TestLivePipelineLifecycle:
    def test_G_PIPE_01_instantiates(self):
        """G-PIPE-01: LivePipeline instantiates with default MissionConfig."""
        assert _make_pipeline() is not None

    def test_G_PIPE_04_not_running_before_start(self):
        """G-PIPE-04: health() reports not running before start()."""
        p = _make_pipeline()
        assert p.health().running is False
        assert p.is_running() is False

    def test_G_PIPE_14_stop_safe_before_start(self):
        """G-PIPE-14: stop() is safe to call before start()."""
        p = _make_pipeline()
        p.stop()   # must not raise

    def test_G_PIPE_05_start_launches_t_nav_thread(self):
        """G-PIPE-05: start() launches a thread named T-NAV."""
        p = _make_pipeline()
        try:
            p.start()
            time.sleep(0.05)
            names = [t.name for t in threading.enumerate()]
            assert 'T-NAV' in names
        finally:
            p.stop()

    def test_G_PIPE_06_t_nav_is_daemon(self):
        """G-PIPE-06: T-NAV thread is a daemon thread."""
        p = _make_pipeline()
        try:
            p.start()
            time.sleep(0.05)
            t_nav = next((t for t in threading.enumerate() if t.name == 'T-NAV'), None)
            assert t_nav is not None
            assert t_nav.daemon is True
        finally:
            p.stop()

    def test_G_PIPE_07_loop_count_increments(self):
        """G-PIPE-07: loop_count increments after start()."""
        p = _make_pipeline()
        try:
            p.start()
            time.sleep(0.1)
            h = p.health()
            assert h.loop_count > 0
        finally:
            p.stop()

    def test_G_PIPE_10_stop_terminates_thread(self):
        """G-PIPE-10: stop() terminates T-NAV within 2s."""
        p = _make_pipeline()
        p.start()
        time.sleep(0.05)
        assert p.is_running() is True
        p.stop(timeout_s=2.0)
        assert p.is_running() is False

    def test_G_PIPE_11_not_running_after_stop(self):
        """G-PIPE-11: health() reports not running after stop()."""
        p = _make_pipeline()
        p.start()
        time.sleep(0.05)
        p.stop()
        assert p.health().running is False

    def test_G_PIPE_13_start_idempotent(self):
        """G-PIPE-13: start() is idempotent — safe to call twice."""
        p = _make_pipeline()
        try:
            p.start()
            p.start()   # must not raise or create second thread
            time.sleep(0.05)
            t_nav_threads = [t for t in threading.enumerate() if t.name == 'T-NAV']
            assert len(t_nav_threads) == 1
        finally:
            p.stop()


class TestSetpointProduction:
    def test_G_PIPE_08_queue_receives_setpoints(self):
        """G-PIPE-08: setpoint_queue receives Setpoint objects after start."""
        p = _make_pipeline(queue_maxsize=20)
        try:
            p.start()
            time.sleep(0.15)   # allow ~30 setpoints at 200Hz
            assert not p.setpoint_queue.empty()
            sp = p.setpoint_queue.get_nowait()
            assert isinstance(sp, Setpoint)
        finally:
            p.stop()

    def test_G_PIPE_09_setpoint_fields_are_finite(self):
        """G-PIPE-09: Setpoint fields are finite floats."""
        p = _make_pipeline(queue_maxsize=20)
        try:
            p.start()
            time.sleep(0.15)
            sp = p.setpoint_queue.get(timeout=1.0)
            assert math.isfinite(sp.x_m)
            assert math.isfinite(sp.y_m)
            assert math.isfinite(sp.z_m)
            assert math.isfinite(sp.yaw_rad)
            assert math.isfinite(sp.t)
        finally:
            p.stop()

    def test_G_PIPE_12_queue_drop_count_increments_when_full(self):
        """G-PIPE-12: queue_drop_count increments when queue is full."""
        # Use queue_maxsize=1 so it fills immediately
        p = _make_pipeline(queue_maxsize=1)
        try:
            p.start()
            time.sleep(0.15)   # let queue fill and overflow
            h = p.health()
            assert h.queue_drop_count > 0
        finally:
            p.stop()
