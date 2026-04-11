"""
BCMP-2 SB-2 Fault Injection Validation Tests.

Validates that the three proxy layers (fault_manager, sensor_fault_proxy,
nav_source_proxy) are correctly wired and produce expected behaviour for
the initial scripted fault set: FI-01, FI-02, FI-05.

Test philosophy
---------------
These tests validate the fault injection *infrastructure* — not the full
BCMP-2 mission run.  Full mission fault runs are AT-3 through AT-5 (SB-3).

What is verified here:
  1. Fault activates → proxy intercepts → downstream sees degraded data
  2. Fault clears   → proxy transparent → downstream sees real data
  3. Multiple simultaneous faults work correctly (multi-fault proxy)
  4. Event log records all activations and clears
  5. No frozen core module is modified by any proxy operation

FI-01 — GNSS denied entire mission
  Proxy: sensor_fault_proxy.gnss() returns denied measurement
  Expected: BIM evaluates denied measurement and moves toward RED

FI-02 — 10s VIO outage during forward motion
  Proxy: sensor_fault_proxy.vio_update() returns (False, 0.0)
  Expected: VIOMode transitions NOMINAL -> OUTAGE after threshold

FI-05 — EO feed frozen
  Proxy: sensor_fault_proxy.eo_frame() returns stale cached frame
  Expected: DMRL receives stale frame (frame_id does not advance)
"""

import threading
import time
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fault_injection.fault_manager import (
    FaultManager,
    FI_GNSS_LOSS,
    FI_VIO_LOSS,
    FI_RADALT_LOSS,
    FI_EO_FREEZE,
    FI_TERRAIN_CONF_DROP,
    FI_IMU_JITTER,
    PRESET_VIO_GNSS,
)
from fault_injection.sensor_fault_proxy import SensorFaultProxy
from fault_injection.nav_source_proxy import NavSourceProxy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_fault_manager():
    """Give each test its own clean FaultManager (not the singleton)."""
    return FaultManager()


@pytest.fixture()
def fm(fresh_fault_manager):
    return fresh_fault_manager


@pytest.fixture()
def sensor_proxy(fm):
    return SensorFaultProxy(fault_manager=fm)


@pytest.fixture()
def nav_proxy(fm):
    return NavSourceProxy(fault_manager=fm, seed=42)


# ---------------------------------------------------------------------------
# FI-01 — GNSS denied entire mission
# ---------------------------------------------------------------------------

class TestFI01GNSSLoss:

    def test_gnss_pass_through_when_no_fault(self, sensor_proxy):
        class _Real:
            pdop = 1.2
        real = _Real()
        assert sensor_proxy.gnss(real) is real

    def test_gnss_denied_when_fault_active(self, sensor_proxy, fm):
        class _Real:
            pdop = 1.2
        real = _Real()
        fm.activate(FI_GNSS_LOSS, duration_s=0.0, source="scripted")
        result = sensor_proxy.gnss(real)
        assert result is not real, "proxy must return denied measurement, not real"

    def test_gnss_availability_false_when_fault_active(self, sensor_proxy, fm):
        fm.activate(FI_GNSS_LOSS)
        assert sensor_proxy.gnss_available(True) is False
        assert sensor_proxy.gnss_available(False) is False

    def test_gnss_restored_after_clear(self, sensor_proxy, fm):
        class _Real:
            pdop = 1.2
        real = _Real()
        fm.activate(FI_GNSS_LOSS)
        assert sensor_proxy.gnss(real) is not real
        fm.clear(FI_GNSS_LOSS)
        assert sensor_proxy.gnss(real) is real

    def test_bim_moves_toward_red_on_denied_gnss(self, sensor_proxy, fm):
        """BIM evaluates denied measurements and trust_score drops toward RED."""
        from core.bim.bim import BIM
        b = BIM()
        fm.activate(FI_GNSS_LOSS)
        initial_score = None
        for i in range(15):
            denied_meas = sensor_proxy.gnss(None)   # real=None, proxy returns denied
            out = b.evaluate(denied_meas)
            if initial_score is None:
                initial_score = out.trust_score
        # After 15 denied evaluations, score should have dropped
        assert out.trust_score <= initial_score, \
            f"trust_score should drop: initial={initial_score:.2f} final={out.trust_score:.2f}"

    def test_event_log_records_fi01_activation(self, sensor_proxy, fm):
        fm.activate(FI_GNSS_LOSS, duration_s=0.0, source="scripted")
        log = fm.event_log()
        assert len(log) == 1
        assert log[0].fault_id == FI_GNSS_LOSS
        assert log[0].action == "activated"
        assert log[0].source == "scripted"

    def test_fi01_indefinite_does_not_expire(self, sensor_proxy, fm):
        """duration_s=0 means indefinite — must not auto-expire."""
        fm.activate(FI_GNSS_LOSS, duration_s=0.0)
        time.sleep(0.05)
        fm.update_mission_km(1.0)   # trigger expiry check
        assert fm.is_active(FI_GNSS_LOSS), "indefinite fault must not expire"


# ---------------------------------------------------------------------------
# FI-02 — 10s VIO outage during forward motion
# ---------------------------------------------------------------------------

class TestFI02VIOLoss:

    def test_vio_update_pass_through_when_no_fault(self, sensor_proxy):
        acc, innov = sensor_proxy.vio_update(True, 0.45)
        assert acc is True and innov == 0.45

    def test_vio_update_suppressed_when_fault_active(self, sensor_proxy, fm):
        fm.activate(FI_VIO_LOSS, duration_s=10.0, source="scripted")
        acc, innov = sensor_proxy.vio_update(True, 0.45)
        assert acc is False
        assert innov == 0.0

    def test_vio_frame_unavailable_when_fault_active(self, sensor_proxy, fm):
        fm.activate(FI_VIO_LOSS)
        assert sensor_proxy.vio_frame_available(True) is False

    def test_vio_mode_transitions_to_outage(self, sensor_proxy, fm):
        """
        VIOMode transitions to OUTAGE when it stops receiving accepted updates.
        Simulates FI-02: VIO outage during forward motion.
        """
        from core.fusion.vio_mode import VIONavigationMode, VIOMode

        dt = 0.005   # 200 Hz
        vio = VIONavigationMode()
        assert vio.current_mode == VIOMode.NOMINAL

        fm.activate(FI_VIO_LOSS, duration_s=0.0)

        # Drive VIOMode with no accepted updates until OUTAGE
        mode_sequence = []
        for _ in range(600):   # 3 seconds at 200 Hz
            acc, innov = sensor_proxy.vio_update(True, 0.4)
            vio.on_vio_update(acc, innov)
            vio.tick(dt)
            mode_sequence.append(vio.current_mode)

        # Must have transitioned to OUTAGE at some point
        assert VIOMode.OUTAGE in mode_sequence, \
            "VIOMode must reach OUTAGE during FI-02"

    def test_vio_restored_after_fault_clear(self, sensor_proxy, fm):
        fm.activate(FI_VIO_LOSS)
        acc, _ = sensor_proxy.vio_update(True, 0.5)
        assert acc is False
        fm.clear(FI_VIO_LOSS)
        acc, innov = sensor_proxy.vio_update(True, 0.5)
        assert acc is True and innov == 0.5

    def test_vio_auto_expires_after_duration(self, sensor_proxy, fm):
        fm.activate(FI_VIO_LOSS, duration_s=0.05)
        assert sensor_proxy.vio_frame_available(True) is False
        time.sleep(0.06)
        fm.update_mission_km(1.0)
        assert sensor_proxy.vio_frame_available(True) is True

    def test_event_log_records_fi02_with_duration(self, fm):
        fm.activate(FI_VIO_LOSS, duration_s=10.0, source="scripted")
        log = fm.event_log()
        assert log[-1].fault_id == FI_VIO_LOSS
        assert log[-1].duration_s == 10.0

    def test_nav_proxy_vio_source_unavailable_during_fault(self, nav_proxy, fm):
        assert nav_proxy.vio_source_available(True) is True
        fm.activate(FI_VIO_LOSS)
        assert nav_proxy.vio_source_available(True) is False
        fm.clear(FI_VIO_LOSS)
        assert nav_proxy.vio_source_available(True) is True


# ---------------------------------------------------------------------------
# FI-05 — EO feed frozen
# ---------------------------------------------------------------------------

class TestFI05EOFreeze:

    def test_eo_pass_through_when_no_fault(self, sensor_proxy):
        class _Frame:
            frame_id = 1
        f = _Frame()
        assert sensor_proxy.eo_frame(f) is f

    def test_eo_returns_stale_when_fault_active(self, sensor_proxy, fm):
        class _Frame:
            pass
        frame1 = _Frame(); frame1.frame_id = 10
        frame2 = _Frame(); frame2.frame_id = 11

        # Receive frame1 while no fault
        sensor_proxy.eo_frame(frame1)

        # Activate freeze — frame2 arrives but proxy returns stale frame1
        fm.activate(FI_EO_FREEZE, duration_s=0.0, source="scripted")
        result = sensor_proxy.eo_frame(frame2)
        assert result is frame1, "proxy must return last cached (stale) frame"
        assert result.frame_id == 10

    def test_eo_returns_none_when_frozen_before_any_frame(self, sensor_proxy, fm):
        """If frozen before any frame received, returns None."""
        fm.activate(FI_EO_FREEZE)
        class _Frame:
            frame_id = 1
        result = sensor_proxy.eo_frame(_Frame())
        assert result is None

    def test_eo_resumes_after_clear(self, sensor_proxy, fm):
        class _Frame:
            pass
        f1 = _Frame(); f1.frame_id = 1
        f2 = _Frame(); f2.frame_id = 2
        sensor_proxy.eo_frame(f1)
        fm.activate(FI_EO_FREEZE)
        sensor_proxy.eo_frame(f2)   # stale
        fm.clear(FI_EO_FREEZE)
        f3 = _Frame(); f3.frame_id = 3
        result = sensor_proxy.eo_frame(f3)
        assert result is f3, "proxy must return real frame after clear"

    def test_eo_availability_unchanged_during_freeze(self, sensor_proxy, fm):
        """EO freeze does not remove the feed — frame is present but stale."""
        fm.activate(FI_EO_FREEZE)
        assert sensor_proxy.eo_available(True) is True
        assert sensor_proxy.eo_available(False) is False


# ---------------------------------------------------------------------------
# FaultInjectionProxy infrastructure classes
# ---------------------------------------------------------------------------

class FI06IOStarve:
    """
    FI-NEW-01: Checkpoint I/O starvation.
    Saturates /tmp disk I/O to simulate
    heavy checkpoint write contention.
    Req: RS-04, EC-05
    """
    def __init__(self, target_dir="/tmp",
                 duration_s=5.0):
        self._target = target_dir
        self._duration = duration_s
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._spam_io,
            daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _spam_io(self):
        import os, tempfile
        deadline = time.monotonic() + \
                   self._duration
        while not self._stop.is_set() and \
              time.monotonic() < deadline:
            try:
                fd, path = tempfile.mkstemp(
                    dir=self._target,
                    prefix="fi06_"
                )
                os.write(fd, b"x" * 65536)
                os.fsync(fd)
                os.close(fd)
                os.unlink(path)
            except OSError:
                pass


class FI07EWMaze:
    """
    FI-NEW-02: RoutePlanner infinite cost maze.
    Generates a synthetic EW costmap that
    fully encloses the vehicle in
    infinite-cost nodes, forcing A* to exhaust
    the search space before R-06 timeout fires.
    Req: PLN-02, R-06
    """
    def __init__(self, map_size=100,
                 center=(50, 50),
                 wall_cost=1e9):
        self._size = map_size
        self._center = center
        self._wall_cost = wall_cost

    def generate_maze_costmap(self):
        """
        Returns a 2D cost grid with infinite-
        cost walls surrounding the center cell.
        All escape paths are blocked.
        """
        import numpy as np
        grid = np.zeros(
            (self._size, self._size),
            dtype=float
        )
        cx, cy = self._center
        # Surround center with infinite walls
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) == 2 or \
                   abs(dy) == 2:
                    x = min(max(cx+dx, 0),
                            self._size-1)
                    y = min(max(cy+dy, 0),
                            self._size-1)
                    grid[x][y] = \
                        self._wall_cost
        return grid

    def get_costmap_dict(self):
        """
        Returns costmap in the EW schema
        format expected by RoutePlanner.
        """
        grid = self.generate_maze_costmap()
        return {
            "costmap": grid.tolist(),
            "resolution_m": 10.0,
            "origin_north_m": 0.0,
            "origin_east_m": 0.0,
            "last_updated_s": 0.0,
            "source": "FI07EWMaze"
        }


# ---------------------------------------------------------------------------
# Multi-fault and proxy interaction tests
# ---------------------------------------------------------------------------

class TestMultiFaultProxy:

    def test_gnss_and_vio_simultaneous(self, sensor_proxy, fm):
        """FI-09: VIO + GNSS simultaneous — both intercepts active."""
        class _Real:
            pdop = 1.2
        real = _Real()
        fm.activate(FI_GNSS_LOSS)
        fm.activate(FI_VIO_LOSS)
        assert sensor_proxy.gnss(real) is not real
        acc, innov = sensor_proxy.vio_update(True, 0.5)
        assert acc is False

    def test_preset_vio_gnss_activates_both(self, fm, sensor_proxy):
        fm.activate_preset(PRESET_VIO_GNSS, duration_s=0.0)
        assert fm.is_active(FI_VIO_LOSS)
        assert fm.is_active(FI_GNSS_LOSS)
        assert len(fm.active_fault_ids()) == 2

    def test_clear_all_restores_all_proxies(self, sensor_proxy, nav_proxy, fm):
        fm.activate(FI_GNSS_LOSS)
        fm.activate(FI_VIO_LOSS)
        fm.activate(FI_TERRAIN_CONF_DROP)
        fm.clear_all()
        class _R:
            pdop = 1.0
        assert sensor_proxy.gnss(_R()) is not None
        assert sensor_proxy.vio_frame_available(True) is True
        assert nav_proxy.vio_source_available(True) is True

    def test_event_log_captures_all_events(self, fm):
        fm.activate(FI_GNSS_LOSS, source="scripted")
        fm.activate(FI_VIO_LOSS, source="scripted")
        fm.clear(FI_GNSS_LOSS)
        fm.clear(FI_VIO_LOSS)
        log = fm.event_log()
        assert len(log) == 4
        actions = [e.action for e in log]
        assert actions.count("activated") == 2
        assert actions.count("cleared") == 2

    def test_no_frozen_core_module_modified(self, fm, sensor_proxy, nav_proxy):
        """
        Activate all SB-2 faults and verify frozen core modules remain importable
        and their module-level constants are unchanged.
        """
        from core.ekf.error_state_ekf import ErrorStateEKF
        from core.fusion.vio_mode import VIONavigationMode
        from core.bim.bim import BIM

        fm.activate(FI_GNSS_LOSS)
        fm.activate(FI_VIO_LOSS)
        fm.activate(FI_TERRAIN_CONF_DROP)
        fm.activate(FI_IMU_JITTER)

        # Frozen constants are class-level attributes — verify via instance
        eskf = ErrorStateEKF()
        assert eskf._ACC_BIAS_RW == 9.81e-7,  "ESKF ACC_BIAS_RW modified"
        assert eskf._GYRO_BIAS_RW == 4.04e-8, "ESKF GYRO_BIAS_RW modified"

        # Frozen modules must still instantiate normally
        eskf = ErrorStateEKF()
        assert eskf is not None
        bim  = BIM()
        assert bim is not None
        vio  = VIONavigationMode()
        assert vio is not None

        fm.clear_all()
