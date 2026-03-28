"""
integration/tests/test_prehil_drivers.py
MicroMind Pre-HIL — Phase 1 Driver ABC Conformance Tests

Gates:
  G-BASE-01: SensorDriver is abstract — cannot be instantiated directly
  G-BASE-02: DriverHealth enum has exactly OK, DEGRADED, FAILED
  G-BASE-03: DriverReadError is a RuntimeError subclass
  G-BASE-04: Concrete subclass must implement all six abstract methods
  G-BASE-05: Partial implementation is rejected at instantiation
  G-BASE-06: _record_successful_read() updates last_update_time
  G-BASE-07: _default_is_stale() returns True when never read
  G-BASE-08: _default_is_stale() returns False when recently read
  G-BASE-09: _default_is_stale() returns True when threshold exceeded
  G-BASE-10: source_type() contract — must return 'sim' or 'real'
"""

import sys, os, time
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from integration.drivers.base import SensorDriver, DriverHealth, DriverReadError


class _ConcreteDriver(SensorDriver):
    def __init__(self, stale_threshold_s=1.0, source='sim'):
        super().__init__(stale_threshold_s)
        self._source = source
        self._health = DriverHealth.OK
        self._closed = False

    def health(self): return self._health
    def last_update_time(self): return self._last_update_time
    def is_stale(self): return self._default_is_stale()
    def source_type(self): return self._source
    def read(self):
        self._record_successful_read()
        return {'value': 42.0}
    def close(self): self._closed = True


class _PartialDriver(SensorDriver):
    def health(self): return DriverHealth.OK
    def last_update_time(self): return 0.0
    def is_stale(self): return True
    def source_type(self): return 'sim'


class TestDriverHealthEnum:
    def test_G_BASE_02_health_enum_members(self):
        assert {m.name for m in DriverHealth} == {'OK', 'DEGRADED', 'FAILED'}

    def test_health_values_are_distinct(self):
        assert DriverHealth.OK != DriverHealth.DEGRADED
        assert DriverHealth.DEGRADED != DriverHealth.FAILED
        assert DriverHealth.OK != DriverHealth.FAILED


class TestDriverReadError:
    def test_G_BASE_03_is_runtime_error_subclass(self):
        err = DriverReadError("RealIMUDriver: device not connected. Check SPI /dev/spidev0.0.")
        assert isinstance(err, RuntimeError)

    def test_message_preserved(self):
        msg = "RealGNSSDriver: NMEA port not open. Check /dev/ttyUSB0."
        assert msg in str(DriverReadError(msg))


class TestSensorDriverABC:
    def test_G_BASE_01_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            SensorDriver(stale_threshold_s=1.0)

    def test_G_BASE_04_concrete_subclass_instantiates(self):
        assert _ConcreteDriver() is not None

    def test_G_BASE_05_partial_implementation_rejected(self):
        with pytest.raises(TypeError):
            _PartialDriver(stale_threshold_s=1.0)

    def test_all_six_methods_present(self):
        d = _ConcreteDriver()
        for method in ['health', 'last_update_time', 'is_stale',
                        'source_type', 'read', 'close']:
            assert callable(getattr(d, method))


class TestRecordSuccessfulRead:
    def test_G_BASE_06_updates_last_update_time(self):
        d = _ConcreteDriver()
        assert d._last_update_time == 0.0
        before = time.monotonic()
        d._record_successful_read()
        after = time.monotonic()
        assert before <= d._last_update_time <= after

    def test_read_calls_record(self):
        d = _ConcreteDriver()
        d.read()
        assert d.last_update_time() > 0.0


class TestIsStale:
    def test_G_BASE_07_stale_when_never_read(self):
        assert _ConcreteDriver(stale_threshold_s=10.0).is_stale() is True

    def test_G_BASE_08_not_stale_immediately_after_read(self):
        d = _ConcreteDriver(stale_threshold_s=5.0)
        d.read()
        assert d.is_stale() is False

    def test_G_BASE_09_stale_after_threshold_exceeded(self):
        d = _ConcreteDriver(stale_threshold_s=0.05)
        d.read()
        assert d.is_stale() is False
        time.sleep(0.1)
        assert d.is_stale() is True


class TestSourceType:
    def test_G_BASE_10_sim_source_type(self):
        assert _ConcreteDriver(source='sim').source_type() == 'sim'

    def test_real_source_type(self):
        assert _ConcreteDriver(source='real').source_type() == 'real'

    def test_source_type_is_string(self):
        assert isinstance(_ConcreteDriver().source_type(), str)


class TestClose:
    def test_close_is_idempotent(self):
        d = _ConcreteDriver()
        d.close(); d.close()
        assert d._closed is True

    def test_health_after_close_does_not_raise(self):
        d = _ConcreteDriver()
        d.close()
        assert isinstance(d.health(), DriverHealth)


# ---------------------------------------------------------------------------
# IMUDriver ABC conformance — appended to test_prehil_drivers.py
# ---------------------------------------------------------------------------

import math
from integration.drivers.imu import IMUDriver, IMUReading


class _ConcreteIMUDriver(IMUDriver):
    def health(self): return DriverHealth.OK
    def last_update_time(self): return self._last_update_time
    def is_stale(self): return self._default_is_stale()
    def source_type(self): return 'sim'
    def read(self) -> IMUReading:
        self._record_successful_read()
        return IMUReading(
            accel_mss=(0.0, 0.0, -9.81),
            gyro_rads=(0.0, 0.0, 0.0),
            temp_c=25.0,
            t=self._last_update_time,
        )
    def close(self): pass


class _PartialIMUDriver(IMUDriver):
    def health(self): return DriverHealth.OK
    def last_update_time(self): return 0.0
    def is_stale(self): return True
    def source_type(self): return 'sim'
    # read() and close() not implemented


class TestIMUReading:
    def test_G_IMU_01_is_frozen_dataclass(self):
        """G-IMU-01: IMUReading is immutable."""
        r = IMUReading(accel_mss=(0.,0.,-9.81), gyro_rads=(0.,0.,0.), temp_c=25.0, t=1.0)
        import pytest
        with pytest.raises((AttributeError, TypeError)):
            r.temp_c = 30.0

    def test_G_IMU_02_fields_accessible(self):
        """G-IMU-02: all four fields accessible by name."""
        r = IMUReading(accel_mss=(1.,2.,3.), gyro_rads=(0.1,0.2,0.3), temp_c=20.0, t=5.0)
        assert r.accel_mss == (1., 2., 3.)
        assert r.gyro_rads == (0.1, 0.2, 0.3)
        assert r.temp_c == 20.0
        assert r.t == 5.0

    def test_G_IMU_03_nan_temp_allowed(self):
        """G-IMU-03: temp_c=nan is valid (source does not provide temperature)."""
        r = IMUReading(accel_mss=(0.,0.,0.), gyro_rads=(0.,0.,0.), temp_c=float('nan'), t=0.0)
        assert math.isnan(r.temp_c)


class TestIMUDriverABC:
    def test_G_IMU_04_cannot_instantiate_directly(self):
        """G-IMU-04: IMUDriver is abstract."""
        import pytest
        with pytest.raises(TypeError):
            IMUDriver(stale_threshold_s=0.01)

    def test_G_IMU_05_partial_rejected(self):
        """G-IMU-05: partial IMUDriver implementation rejected."""
        import pytest
        with pytest.raises(TypeError):
            _PartialIMUDriver(stale_threshold_s=0.01)

    def test_G_IMU_06_concrete_instantiates(self):
        """G-IMU-06: concrete IMUDriver instantiates cleanly."""
        assert _ConcreteIMUDriver(stale_threshold_s=0.01) is not None

    def test_G_IMU_07_read_returns_imu_reading(self):
        """G-IMU-07: read() returns IMUReading instance."""
        d = _ConcreteIMUDriver(stale_threshold_s=0.01)
        result = d.read()
        assert isinstance(result, IMUReading)

    def test_G_IMU_08_read_updates_timestamp(self):
        """G-IMU-08: read() updates last_update_time."""
        d = _ConcreteIMUDriver(stale_threshold_s=0.01)
        assert d.last_update_time() == 0.0
        d.read()
        assert d.last_update_time() > 0.0

    def test_G_IMU_09_reading_timestamp_matches_driver(self):
        """G-IMU-09: IMUReading.t matches driver last_update_time after read."""
        d = _ConcreteIMUDriver(stale_threshold_s=0.01)
        r = d.read()
        assert r.t == d.last_update_time()

    def test_G_IMU_10_is_sensor_driver_subclass(self):
        """G-IMU-10: IMUDriver is a SensorDriver subclass."""
        from integration.drivers.base import SensorDriver
        assert issubclass(IMUDriver, SensorDriver)
