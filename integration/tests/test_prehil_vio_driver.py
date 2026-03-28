"""
integration/tests/test_prehil_vio_driver.py
MicroMind Pre-HIL — Phase 3 VIO Driver Tests

Gates:
  G-VIO-01: OfflineVIODriver instantiates from ndarray
  G-VIO-02: OfflineVIODriver instantiates from .npy file
  G-VIO-03: read() returns VIOReading
  G-VIO-04: health() is DEGRADED before first read, OK after
  G-VIO-05: pos_ned_m is finite (3,) ndarray
  G-VIO-06: cov_ned is (3,3) positive-definite diagonal
  G-VIO-07: source_type() returns 'sim'
  G-VIO-08: frame_index advances on each read
  G-VIO-09: loop=True wraps frame_index to 0 at end of array
  G-VIO-10: loop=False raises DriverReadError when exhausted
  G-VIO-11: ENU→NED rotation correct (East→East, North↔East swap)
  G-VIO-12: IFM-01 gate rejects non-monotonic timestamp (valid=False)
  G-VIO-13: IFM-01 records event_id, t_prev, t_current, delta
  G-VIO-14: IFM-01 violation_count increments correctly
  G-VIO-15: Valid frames after violation continue normally
  G-VIO-16: read() after close() raises DriverReadError
  G-VIO-17: close() is idempotent
  G-VIO-18: OfflineVIODriver is SensorDriver subclass
  G-VIO-19: LiveVIODriver raises DriverReadError with interface path
  G-VIO-20: LiveVIODriver source_type() returns 'real'
  G-VIO-21: LiveVIODriver health() returns FAILED
  G-VIO-22: S-NEP-04 interface: update_vio accepts OfflineVIODriver output
"""

import sys, os, math, tempfile
import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from integration.drivers.vio_driver import (
    OfflineVIODriver, LiveVIODriver, VIOReading, _MonotonicityGuard
)
from integration.drivers.base import SensorDriver, DriverHealth, DriverReadError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pos_enu(n=50, dx=0.5):
    """Simple straight-line ENU trajectory."""
    pos = np.zeros((n, 3), dtype=np.float64)
    pos[:, 0] = np.arange(n) * dx   # East
    pos[:, 1] = 0.0                  # North
    pos[:, 2] = 100.0                # Up (constant altitude)
    return pos


def _make_driver(n=50, loop=True, sigma=0.1, dt=0.04):
    return OfflineVIODriver(_make_pos_enu(n), sigma_pos_m=sigma,
                            dt_s=dt, loop=loop)


# ---------------------------------------------------------------------------
# OfflineVIODriver conformance
# ---------------------------------------------------------------------------

class TestOfflineVIODriver:
    def test_G_VIO_01_instantiates_from_ndarray(self):
        """G-VIO-01: OfflineVIODriver instantiates from ndarray."""
        assert _make_driver() is not None

    def test_G_VIO_02_instantiates_from_npy_file(self):
        """G-VIO-02: OfflineVIODriver instantiates from .npy file."""
        pos = _make_pos_enu(20)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            np.save(f.name, pos)
            path = f.name
        d = OfflineVIODriver(path)
        assert d.n_frames == 20
        os.unlink(path)

    def test_G_VIO_03_read_returns_vio_reading(self):
        """G-VIO-03: read() returns VIOReading instance."""
        d = _make_driver()
        assert isinstance(d.read(), VIOReading)

    def test_G_VIO_04_health_transitions(self):
        """G-VIO-04: health() DEGRADED before first read, OK after."""
        d = _make_driver()
        assert d.health() == DriverHealth.DEGRADED
        d.read()
        assert d.health() == DriverHealth.OK

    def test_G_VIO_05_pos_ned_is_finite_3vec(self):
        """G-VIO-05: pos_ned_m is finite (3,) ndarray."""
        d = _make_driver()
        r = d.read()
        assert r.pos_ned_m.shape == (3,)
        assert all(math.isfinite(x) for x in r.pos_ned_m)

    def test_G_VIO_06_cov_ned_shape_and_positive_diagonal(self):
        """G-VIO-06: cov_ned is (3,3) with positive diagonal."""
        d = _make_driver(sigma=0.1)
        r = d.read()
        assert r.cov_ned.shape == (3, 3)
        assert all(r.cov_ned[i, i] > 0 for i in range(3))

    def test_G_VIO_07_source_type_is_sim(self):
        """G-VIO-07: source_type() returns 'sim'."""
        assert _make_driver().source_type() == 'sim'

    def test_G_VIO_08_frame_index_advances(self):
        """G-VIO-08: frame_index advances on each read."""
        d = _make_driver()
        assert d.frame_index == 0
        d.read()
        assert d.frame_index == 1
        d.read()
        assert d.frame_index == 2

    def test_G_VIO_09_loop_wraps_to_zero(self):
        """G-VIO-09: loop=True wraps frame_index to 0 at end of array."""
        d = _make_driver(n=3, loop=True)
        d.read(); d.read(); d.read()   # exhaust
        assert d.frame_index == 0      # wrapped
        r = d.read()                   # must not raise
        assert isinstance(r, VIOReading)

    def test_G_VIO_10_no_loop_raises_when_exhausted(self):
        """G-VIO-10: loop=False raises DriverReadError when exhausted."""
        d = _make_driver(n=2, loop=False)
        d.read(); d.read()
        with pytest.raises(DriverReadError, match="exhausted"):
            d.read()

    def test_G_VIO_11_enu_to_ned_rotation(self):
        """G-VIO-11: ENU East → NED [0, East_val, 0] (East unchanged)."""
        # Position: 10m East in ENU → NED should be [0, 10, -alt]
        pos_enu = np.array([[10.0, 0.0, 0.0]])
        d = OfflineVIODriver(pos_enu, dt_s=0.04)
        r = d.read()
        assert abs(r.pos_ned_m[0]) < 1e-9   # North ~ 0
        assert abs(r.pos_ned_m[1] - 10.0) < 1e-9  # East = 10

    def test_G_VIO_12_ifm01_rejects_non_monotonic(self):
        """G-VIO-12: IFM-01 gate rejects non-monotonic timestamp."""
        d = _make_driver()
        d._guard._last_t = 999.0   # force violation
        r = d.read()
        assert r.valid is False

    def test_G_VIO_13_ifm01_records_event_fields(self):
        """G-VIO-13: IFM-01 records event_id, t_prev, t_current, delta."""
        d = _make_driver()
        d._guard._last_t = 999.0
        d.read()
        ev = d.monotonicity_guard.last_violation()
        assert ev is not None
        assert 'event_id' in ev
        assert 't_prev' in ev
        assert 't_current' in ev
        assert 'delta' in ev
        assert ev['event_id'].startswith('IFM01-')
        assert ev['t_prev'] == 999.0

    def test_G_VIO_14_ifm01_violation_count(self):
        """G-VIO-14: IFM-01 violation_count increments correctly."""
        d = _make_driver()
        assert d.monotonicity_guard.violation_count == 0
        d._guard._last_t = 999.0
        d.read()
        assert d.monotonicity_guard.violation_count == 1
        d._guard._last_t = 999.0
        d.read()
        assert d.monotonicity_guard.violation_count == 2

    def test_G_VIO_15_valid_frames_continue_after_violation(self):
        """G-VIO-15: valid frames after a violation resume normally."""
        d = _make_driver(n=10)
        d._guard._last_t = 999.0   # inject violation on frame 0
        r0 = d.read()
        assert r0.valid is False
        # Reset guard to allow normal timestamps
        d._guard._last_t = -1.0
        r1 = d.read()
        assert r1.valid is True

    def test_G_VIO_16_read_after_close_raises(self):
        """G-VIO-16: read() after close() raises DriverReadError."""
        d = _make_driver()
        d.close()
        with pytest.raises(DriverReadError):
            d.read()

    def test_G_VIO_17_close_idempotent(self):
        """G-VIO-17: close() can be called multiple times."""
        d = _make_driver()
        d.close()
        d.close()

    def test_G_VIO_18_is_sensor_driver_subclass(self):
        """G-VIO-18: OfflineVIODriver is a SensorDriver subclass."""
        assert issubclass(OfflineVIODriver, SensorDriver)

    def test_G_VIO_22_update_vio_accepts_output(self):
        """G-VIO-22: ESKF.update_vio() accepts OfflineVIODriver output."""
        from core.ekf.error_state_ekf import ErrorStateEKF
        from core.ins.state import INSState
        eskf  = ErrorStateEKF()
        state = INSState(
            p=np.zeros(3), v=np.zeros(3),
            q=np.array([1., 0., 0., 0.]),
            ba=np.zeros(3), bg=np.zeros(3),
        )
        d = _make_driver()
        r = d.read()
        assert r.valid is True
        nis, rejected, innov = eskf.update_vio(state, r.pos_ned_m, r.cov_ned)
        assert not rejected, "ESKF rejected valid OfflineVIODriver output"
        assert math.isfinite(nis)


# ---------------------------------------------------------------------------
# LiveVIODriver conformance
# ---------------------------------------------------------------------------

class TestLiveVIODriver:
    def test_G_VIO_19_raises_with_interface_path(self):
        """G-VIO-19: LiveVIODriver raises DriverReadError with interface info."""
        lv = LiveVIODriver()
        with pytest.raises(DriverReadError) as exc_info:
            lv.read()
        assert 'VIO' in str(exc_info.value)
        assert 'rotate_pos_enu_to_ned' in str(exc_info.value)

    def test_G_VIO_20_source_type_is_real(self):
        """G-VIO-20: LiveVIODriver source_type() returns 'real'."""
        assert LiveVIODriver().source_type() == 'real'

    def test_G_VIO_21_health_is_failed(self):
        """G-VIO-21: LiveVIODriver health() returns FAILED."""
        assert LiveVIODriver().health() == DriverHealth.FAILED
