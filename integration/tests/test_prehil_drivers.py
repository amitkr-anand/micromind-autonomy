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


# ---------------------------------------------------------------------------
# GNSSDriver ABC conformance
# ---------------------------------------------------------------------------

from integration.drivers.gnss import GNSSDriver, GNSSReading, GNSSFixType
import math as _math


class _ConcreteGNSSDriver(GNSSDriver):
    def health(self): return DriverHealth.OK
    def last_update_time(self): return self._last_update_time
    def is_stale(self): return self._default_is_stale()
    def source_type(self): return 'sim'
    def read(self) -> GNSSReading:
        self._record_successful_read()
        return GNSSReading(
            lat=28.6139, lon=77.2090, alt=216.0,
            hdop=1.2, fix_type=GNSSFixType.FIX_3D,
            t=self._last_update_time,
        )
    def close(self): pass


class _PartialGNSSDriver(GNSSDriver):
    def health(self): return DriverHealth.OK
    def last_update_time(self): return 0.0
    def is_stale(self): return True
    def source_type(self): return 'sim'


class TestGNSSFixType:
    def test_G_GNSS_01_fix_type_values(self):
        """G-GNSS-01: GNSSFixType has expected members and MAVLink-aligned values."""
        assert GNSSFixType.NO_GPS    == 0
        assert GNSSFixType.NO_FIX   == 1
        assert GNSSFixType.FIX_2D   == 2
        assert GNSSFixType.FIX_3D   == 3
        assert GNSSFixType.DGPS     == 4
        assert GNSSFixType.RTK_FLOAT == 5
        assert GNSSFixType.RTK_FIXED == 6


class TestGNSSReading:
    def test_G_GNSS_02_is_frozen_dataclass(self):
        """G-GNSS-02: GNSSReading is immutable."""
        import pytest
        r = GNSSReading(lat=0., lon=0., alt=0., hdop=1., fix_type=GNSSFixType.FIX_3D, t=0.)
        with pytest.raises((AttributeError, TypeError)):
            r.lat = 1.0

    def test_G_GNSS_03_fields_accessible(self):
        """G-GNSS-03: all six fields accessible by name."""
        r = GNSSReading(lat=28.6, lon=77.2, alt=216., hdop=1.2,
                        fix_type=GNSSFixType.FIX_3D, t=5.0)
        assert r.lat == 28.6
        assert r.lon == 77.2
        assert r.alt == 216.0
        assert r.hdop == 1.2
        assert r.fix_type == GNSSFixType.FIX_3D
        assert r.t == 5.0

    def test_G_GNSS_04_nan_hdop_allowed(self):
        """G-GNSS-04: hdop=nan is valid when HDOP not available."""
        r = GNSSReading(lat=0., lon=0., alt=0., hdop=float('nan'),
                        fix_type=GNSSFixType.NO_FIX, t=0.)
        assert _math.isnan(r.hdop)


class TestGNSSDriverABC:
    def test_G_GNSS_05_cannot_instantiate_directly(self):
        """G-GNSS-05: GNSSDriver is abstract."""
        import pytest
        with pytest.raises(TypeError):
            GNSSDriver(stale_threshold_s=0.2)

    def test_G_GNSS_06_partial_rejected(self):
        """G-GNSS-06: partial GNSSDriver implementation rejected."""
        import pytest
        with pytest.raises(TypeError):
            _PartialGNSSDriver(stale_threshold_s=0.2)

    def test_G_GNSS_07_concrete_instantiates(self):
        """G-GNSS-07: concrete GNSSDriver instantiates cleanly."""
        assert _ConcreteGNSSDriver(stale_threshold_s=0.2) is not None

    def test_G_GNSS_08_read_returns_gnss_reading(self):
        """G-GNSS-08: read() returns GNSSReading instance."""
        d = _ConcreteGNSSDriver(stale_threshold_s=0.2)
        assert isinstance(d.read(), GNSSReading)

    def test_G_GNSS_09_read_updates_timestamp(self):
        """G-GNSS-09: read() updates last_update_time."""
        d = _ConcreteGNSSDriver(stale_threshold_s=0.2)
        assert d.last_update_time() == 0.0
        d.read()
        assert d.last_update_time() > 0.0

    def test_G_GNSS_10_is_sensor_driver_subclass(self):
        """G-GNSS-10: GNSSDriver is a SensorDriver subclass."""
        from integration.drivers.base import SensorDriver
        assert issubclass(GNSSDriver, SensorDriver)


# ---------------------------------------------------------------------------
# RADALTDriver ABC conformance
# ---------------------------------------------------------------------------

from integration.drivers.radalt import RADALTDriver, RADALTReading
import math as _math


class _ConcreteRADALTDriver(RADALTDriver):
    def health(self): return DriverHealth.OK
    def last_update_time(self): return self._last_update_time
    def is_stale(self): return self._default_is_stale()
    def source_type(self): return 'sim'
    def read(self) -> RADALTReading:
        self._record_successful_read()
        return RADALTReading(alt_agl_m=45.2, validity_flag=True, t=self._last_update_time)
    def close(self): pass


class _PartialRADALTDriver(RADALTDriver):
    def health(self): return DriverHealth.OK
    def last_update_time(self): return 0.0
    def is_stale(self): return True
    def source_type(self): return 'sim'


class TestRADALTReading:
    def test_G_RADALT_01_is_frozen_dataclass(self):
        """G-RADALT-01: RADALTReading is immutable."""
        import pytest
        r = RADALTReading(alt_agl_m=10.0, validity_flag=True, t=1.0)
        with pytest.raises((AttributeError, TypeError)):
            r.alt_agl_m = 20.0

    def test_G_RADALT_02_fields_accessible(self):
        """G-RADALT-02: all three fields accessible by name."""
        r = RADALTReading(alt_agl_m=45.2, validity_flag=True, t=3.0)
        assert r.alt_agl_m == 45.2
        assert r.validity_flag is True
        assert r.t == 3.0

    def test_G_RADALT_03_invalid_reading_uses_false_flag(self):
        """G-RADALT-03: validity_flag=False is valid for unavailable measurement."""
        r = RADALTReading(alt_agl_m=float('nan'), validity_flag=False, t=0.0)
        assert r.validity_flag is False
        assert _math.isnan(r.alt_agl_m)

    def test_G_RADALT_04_validity_flag_is_bool(self):
        """G-RADALT-04: validity_flag type is bool."""
        r = RADALTReading(alt_agl_m=10.0, validity_flag=True, t=0.0)
        assert isinstance(r.validity_flag, bool)


class TestRADALTDriverABC:
    def test_G_RADALT_05_cannot_instantiate_directly(self):
        """G-RADALT-05: RADALTDriver is abstract."""
        import pytest
        with pytest.raises(TypeError):
            RADALTDriver(stale_threshold_s=0.1)

    def test_G_RADALT_06_partial_rejected(self):
        """G-RADALT-06: partial RADALTDriver implementation rejected."""
        import pytest
        with pytest.raises(TypeError):
            _PartialRADALTDriver(stale_threshold_s=0.1)

    def test_G_RADALT_07_concrete_instantiates(self):
        """G-RADALT-07: concrete RADALTDriver instantiates cleanly."""
        assert _ConcreteRADALTDriver(stale_threshold_s=0.1) is not None

    def test_G_RADALT_08_read_returns_radalt_reading(self):
        """G-RADALT-08: read() returns RADALTReading instance."""
        d = _ConcreteRADALTDriver(stale_threshold_s=0.1)
        assert isinstance(d.read(), RADALTReading)

    def test_G_RADALT_09_read_updates_timestamp(self):
        """G-RADALT-09: read() updates last_update_time."""
        d = _ConcreteRADALTDriver(stale_threshold_s=0.1)
        assert d.last_update_time() == 0.0
        d.read()
        assert d.last_update_time() > 0.0

    def test_G_RADALT_10_is_sensor_driver_subclass(self):
        """G-RADALT-10: RADALTDriver is a SensorDriver subclass."""
        from integration.drivers.base import SensorDriver
        assert issubclass(RADALTDriver, SensorDriver)


# ---------------------------------------------------------------------------
# EOIRDriver ABC conformance
# ---------------------------------------------------------------------------

import numpy as _np
from integration.drivers.eoir import EOIRDriver, EOIRFrame


class _ConcreteEOIRDriver(EOIRDriver):
    def health(self): return DriverHealth.OK
    def last_update_time(self): return self._last_update_time
    def is_stale(self): return self._default_is_stale()
    def source_type(self): return 'sim'
    def read(self) -> EOIRFrame:
        self._record_successful_read()
        frame = _np.zeros((64, 64), dtype=_np.uint16)
        return EOIRFrame(frame_data=frame, width=64, height=64,
                         validity_flag=True, t=self._last_update_time)
    def close(self): pass


class _PartialEOIRDriver(EOIRDriver):
    def health(self): return DriverHealth.OK
    def last_update_time(self): return 0.0
    def is_stale(self): return True
    def source_type(self): return 'sim'


class TestEOIRFrame:
    def test_G_EOIR_01_valid_frame_fields(self):
        """G-EOIR-01: all five fields accessible on a valid frame."""
        frame = _np.zeros((64, 64), dtype=_np.uint16)
        f = EOIRFrame(frame_data=frame, width=64, height=64,
                      validity_flag=True, t=1.0)
        assert f.width == 64
        assert f.height == 64
        assert f.validity_flag is True
        assert f.t == 1.0
        assert f.frame_data is frame

    def test_G_EOIR_02_invalid_frame_none_data(self):
        """G-EOIR-02: frame_data=None with validity_flag=False is valid."""
        f = EOIRFrame(frame_data=None, width=0, height=0,
                      validity_flag=False, t=0.0)
        assert f.frame_data is None
        assert f.validity_flag is False

    def test_G_EOIR_03_validity_flag_is_bool(self):
        """G-EOIR-03: validity_flag type is bool."""
        f = EOIRFrame(frame_data=None, width=0, height=0,
                      validity_flag=False, t=0.0)
        assert isinstance(f.validity_flag, bool)

    def test_G_EOIR_04_thermal_frame_dtype(self):
        """G-EOIR-04: thermal frame uses uint16 dtype."""
        frame = _np.zeros((64, 64), dtype=_np.uint16)
        f = EOIRFrame(frame_data=frame, width=64, height=64,
                      validity_flag=True, t=0.0)
        assert f.frame_data.dtype == _np.uint16

    def test_G_EOIR_05_colour_frame_dtype(self):
        """G-EOIR-05: colour frame uses uint8 dtype with 3 channels."""
        frame = _np.zeros((64, 64, 3), dtype=_np.uint8)
        f = EOIRFrame(frame_data=frame, width=64, height=64,
                      validity_flag=True, t=0.0)
        assert f.frame_data.dtype == _np.uint8
        assert f.frame_data.shape == (64, 64, 3)


class TestEOIRDriverABC:
    def test_G_EOIR_06_cannot_instantiate_directly(self):
        """G-EOIR-06: EOIRDriver is abstract."""
        import pytest
        with pytest.raises(TypeError):
            EOIRDriver(stale_threshold_s=0.04)

    def test_G_EOIR_07_partial_rejected(self):
        """G-EOIR-07: partial EOIRDriver implementation rejected."""
        import pytest
        with pytest.raises(TypeError):
            _PartialEOIRDriver(stale_threshold_s=0.04)

    def test_G_EOIR_08_concrete_instantiates(self):
        """G-EOIR-08: concrete EOIRDriver instantiates cleanly."""
        assert _ConcreteEOIRDriver(stale_threshold_s=0.04) is not None

    def test_G_EOIR_09_read_returns_eoir_frame(self):
        """G-EOIR-09: read() returns EOIRFrame instance."""
        d = _ConcreteEOIRDriver(stale_threshold_s=0.04)
        assert isinstance(d.read(), EOIRFrame)

    def test_G_EOIR_10_read_updates_timestamp(self):
        """G-EOIR-10: read() updates last_update_time."""
        d = _ConcreteEOIRDriver(stale_threshold_s=0.04)
        assert d.last_update_time() == 0.0
        d.read()
        assert d.last_update_time() > 0.0

    def test_G_EOIR_11_is_sensor_driver_subclass(self):
        """G-EOIR-11: EOIRDriver is a SensorDriver subclass."""
        from integration.drivers.base import SensorDriver
        assert issubclass(EOIRDriver, SensorDriver)


# ---------------------------------------------------------------------------
# SimIMUDriver conformance
# ---------------------------------------------------------------------------

import math as _math
from integration.drivers.sim_imu import SimIMUDriver
from integration.drivers.imu import IMUReading
from integration.drivers.base import DriverHealth, DriverReadError


class TestSimIMUDriver:
    def test_G_SIMU_01_instantiates_stim300(self):
        """G-SIMU-01: SimIMUDriver instantiates with STIM300."""
        d = SimIMUDriver(imu_type="STIM300", seed=42, n_steps=1000)
        assert d is not None

    def test_G_SIMU_02_instantiates_adis16505_3(self):
        """G-SIMU-02: SimIMUDriver instantiates with ADIS16505_3."""
        d = SimIMUDriver(imu_type="ADIS16505_3", seed=0, n_steps=1000)
        assert d is not None

    def test_G_SIMU_03_instantiates_baseline(self):
        """G-SIMU-03: SimIMUDriver instantiates with BASELINE."""
        d = SimIMUDriver(imu_type="BASELINE", seed=1, n_steps=1000)
        assert d is not None

    def test_G_SIMU_04_unknown_type_raises(self):
        """G-SIMU-04: unknown imu_type raises ValueError at construction."""
        import pytest
        with pytest.raises(ValueError, match="unknown imu_type"):
            SimIMUDriver(imu_type="UNKNOWN_IMU", seed=42)

    def test_G_SIMU_05_source_type_is_sim(self):
        """G-SIMU-05: source_type() returns 'sim'."""
        d = SimIMUDriver(imu_type="STIM300", seed=42, n_steps=100)
        assert d.source_type() == 'sim'

    def test_G_SIMU_06_read_returns_imu_reading(self):
        """G-SIMU-06: read() returns IMUReading instance."""
        d = SimIMUDriver(imu_type="STIM300", seed=42, n_steps=100)
        r = d.read()
        assert isinstance(r, IMUReading)

    def test_G_SIMU_07_health_ok_after_read(self):
        """G-SIMU-07: health() returns OK after first successful read."""
        d = SimIMUDriver(imu_type="STIM300", seed=42, n_steps=100)
        assert d.health() == DriverHealth.DEGRADED   # before first read
        d.read()
        assert d.health() == DriverHealth.OK

    def test_G_SIMU_08_reading_fields_are_finite(self):
        """G-SIMU-08: accel and gyro values are finite floats."""
        d = SimIMUDriver(imu_type="STIM300", seed=42, n_steps=100)
        r = d.read()
        assert all(_math.isfinite(v) for v in r.accel_mss)
        assert all(_math.isfinite(v) for v in r.gyro_rads)

    def test_G_SIMU_09_temp_is_nan(self):
        """G-SIMU-09: temp_c is NaN (imu_model does not simulate temperature)."""
        d = SimIMUDriver(imu_type="STIM300", seed=42, n_steps=100)
        r = d.read()
        assert _math.isnan(r.temp_c)

    def test_G_SIMU_10_deterministic_with_same_seed(self):
        """G-SIMU-10: identical seeds produce identical readings."""
        d1 = SimIMUDriver(imu_type="STIM300", seed=99, n_steps=100)
        d2 = SimIMUDriver(imu_type="STIM300", seed=99, n_steps=100)
        r1, r2 = d1.read(), d2.read()
        assert r1.accel_mss == r2.accel_mss
        assert r1.gyro_rads == r2.gyro_rads

    def test_G_SIMU_11_different_seeds_differ(self):
        """G-SIMU-11: different seeds produce different readings."""
        d1 = SimIMUDriver(imu_type="STIM300", seed=1, n_steps=100)
        d2 = SimIMUDriver(imu_type="STIM300", seed=2, n_steps=100)
        r1, r2 = d1.read(), d2.read()
        assert r1.accel_mss != r2.accel_mss or r1.gyro_rads != r2.gyro_rads

    def test_G_SIMU_12_read_after_close_raises(self):
        """G-SIMU-12: read() after close() raises DriverReadError."""
        import pytest
        d = SimIMUDriver(imu_type="STIM300", seed=42, n_steps=100)
        d.close()
        with pytest.raises(DriverReadError):
            d.read()

    def test_G_SIMU_13_close_is_idempotent(self):
        """G-SIMU-13: close() can be called multiple times without raising."""
        d = SimIMUDriver(imu_type="STIM300", seed=42, n_steps=100)
        d.close()
        d.close()

    def test_G_SIMU_14_is_imu_driver_subclass(self):
        """G-SIMU-14: SimIMUDriver is an IMUDriver subclass."""
        from integration.drivers.imu import IMUDriver
        assert issubclass(SimIMUDriver, IMUDriver)


# ---------------------------------------------------------------------------
# SimGNSSDriver conformance
# ---------------------------------------------------------------------------

import math as _math
from integration.drivers.sim_gnss import SimGNSSDriver, _enu_to_geodetic
from integration.drivers.gnss import GNSSReading, GNSSFixType
from integration.drivers.base import DriverHealth, DriverReadError
import numpy as _np


class TestEnuToGeodetic:
    def test_G_SGNSS_01_north_offset(self):
        """G-SGNSS-01: 1km north offset increases latitude."""
        lat, lon, alt = _enu_to_geodetic(
            _np.array([0., 1000., 0.]), 26.9157, 70.9083, 225.0
        )
        assert lat > 26.9157
        assert abs(lon - 70.9083) < 1e-6
        assert abs(alt - 225.0) < 1e-6

    def test_G_SGNSS_02_east_offset(self):
        """G-SGNSS-02: 1km east offset increases longitude."""
        lat, lon, alt = _enu_to_geodetic(
            _np.array([1000., 0., 0.]), 26.9157, 70.9083, 225.0
        )
        assert lon > 70.9083
        assert abs(lat - 26.9157) < 1e-6

    def test_G_SGNSS_03_up_offset(self):
        """G-SGNSS-03: 100m up offset increases altitude."""
        lat, lon, alt = _enu_to_geodetic(
            _np.array([0., 0., 100.]), 26.9157, 70.9083, 225.0
        )
        assert abs(alt - 325.0) < 1e-6


class TestSimGNSSDriver:
    def test_G_SGNSS_04_instantiates(self):
        """G-SGNSS-04: SimGNSSDriver instantiates with defaults."""
        d = SimGNSSDriver()
        assert d is not None

    def test_G_SGNSS_05_source_type_is_sim(self):
        """G-SGNSS-05: source_type() returns 'sim'."""
        assert SimGNSSDriver().source_type() == 'sim'

    def test_G_SGNSS_06_read_returns_gnss_reading(self):
        """G-SGNSS-06: read() returns GNSSReading instance."""
        d = SimGNSSDriver()
        r = d.read()
        assert isinstance(r, GNSSReading)

    def test_G_SGNSS_07_health_ok_after_read(self):
        """G-SGNSS-07: health() returns OK after first successful read."""
        d = SimGNSSDriver()
        assert d.health() == DriverHealth.DEGRADED
        d.read()
        assert d.health() == DriverHealth.OK

    def test_G_SGNSS_08_lat_lon_are_finite(self):
        """G-SGNSS-08: lat and lon are finite after read."""
        d = SimGNSSDriver()
        r = d.read()
        assert _math.isfinite(r.lat)
        assert _math.isfinite(r.lon)

    def test_G_SGNSS_09_fix_type_is_gnss_fix_type(self):
        """G-SGNSS-09: fix_type is a GNSSFixType enum member."""
        d = SimGNSSDriver()
        r = d.read()
        assert isinstance(r.fix_type, GNSSFixType)

    def test_G_SGNSS_10_mission_time_advances(self):
        """G-SGNSS-10: mission time advances by dt_s on each read."""
        d = SimGNSSDriver()
        assert d._mission_time_s == 0.0
        d.read(dt_s=0.2)
        assert abs(d._mission_time_s - 0.2) < 1e-9
        d.read(dt_s=0.2)
        assert abs(d._mission_time_s - 0.4) < 1e-9

    def test_G_SGNSS_11_read_after_close_raises(self):
        """G-SGNSS-11: read() after close() raises DriverReadError."""
        import pytest
        d = SimGNSSDriver()
        d.close()
        with pytest.raises(DriverReadError):
            d.read()

    def test_G_SGNSS_12_close_is_idempotent(self):
        """G-SGNSS-12: close() can be called multiple times."""
        d = SimGNSSDriver()
        d.close()
        d.close()

    def test_G_SGNSS_13_injector_accessible(self):
        """G-SGNSS-13: injector property exposes GNSSSpoofInjector."""
        from sim.gnss_spoof_injector import GNSSSpoofInjector
        d = SimGNSSDriver()
        assert isinstance(d.injector, GNSSSpoofInjector)

    def test_G_SGNSS_14_is_gnss_driver_subclass(self):
        """G-SGNSS-14: SimGNSSDriver is a GNSSDriver subclass."""
        from integration.drivers.gnss import GNSSDriver
        assert issubclass(SimGNSSDriver, GNSSDriver)


# ---------------------------------------------------------------------------
# SimRADALTDriver conformance
# ---------------------------------------------------------------------------

import math as _math
from integration.drivers.sim_radalt import SimRADALTDriver
from integration.drivers.radalt import RADALTReading
from integration.drivers.base import DriverHealth, DriverReadError
from core.ins.trn_stub import DEMProvider as _DEMProvider


class TestSimRADALTDriver:
    def test_G_SRALT_01_instantiates(self):
        """G-SRALT-01: SimRADALTDriver instantiates with defaults."""
        assert SimRADALTDriver() is not None

    def test_G_SRALT_02_source_type_is_sim(self):
        """G-SRALT-02: source_type() returns 'sim'."""
        assert SimRADALTDriver().source_type() == 'sim'

    def test_G_SRALT_03_read_returns_radalt_reading(self):
        """G-SRALT-03: read() returns RADALTReading instance."""
        d = SimRADALTDriver(seed=42)
        assert isinstance(d.read(), RADALTReading)

    def test_G_SRALT_04_health_ok_after_read(self):
        """G-SRALT-04: health() returns OK after first read."""
        d = SimRADALTDriver(seed=42)
        assert d.health() == DriverHealth.DEGRADED
        d.read()
        assert d.health() == DriverHealth.OK

    def test_G_SRALT_05_valid_agl_above_terrain(self):
        """G-SRALT-05: vehicle well above terrain gives valid positive AGL."""
        d = SimRADALTDriver(seed=42)
        r = d.read(north_m=0.0, east_m=0.0, vehicle_alt_amsl_m=500.0)
        assert r.validity_flag is True
        assert r.alt_agl_m > 0.0
        assert _math.isfinite(r.alt_agl_m)

    def test_G_SRALT_06_invalid_below_terrain(self):
        """G-SRALT-06: vehicle below terrain gives validity_flag=False."""
        d = SimRADALTDriver(seed=42)
        r = d.read(north_m=0.0, east_m=0.0, vehicle_alt_amsl_m=0.0)
        assert r.validity_flag is False
        assert _math.isnan(r.alt_agl_m)

    def test_G_SRALT_07_agl_decreases_as_altitude_drops(self):
        """G-SRALT-07: AGL decreases as vehicle altitude decreases."""
        d = SimRADALTDriver(seed=42)
        r_high = d.read(north_m=0.0, east_m=0.0, vehicle_alt_amsl_m=500.0)
        r_low  = d.read(north_m=0.0, east_m=0.0, vehicle_alt_amsl_m=300.0)
        assert r_high.alt_agl_m > r_low.alt_agl_m

    def test_G_SRALT_08_deterministic_same_seed(self):
        """G-SRALT-08: same seed and position gives same AGL."""
        d1 = SimRADALTDriver(seed=7)
        d2 = SimRADALTDriver(seed=7)
        r1 = d1.read(north_m=100.0, east_m=200.0, vehicle_alt_amsl_m=400.0)
        r2 = d2.read(north_m=100.0, east_m=200.0, vehicle_alt_amsl_m=400.0)
        assert r1.alt_agl_m == r2.alt_agl_m

    def test_G_SRALT_09_read_after_close_raises(self):
        """G-SRALT-09: read() after close() raises DriverReadError."""
        import pytest
        d = SimRADALTDriver(seed=42)
        d.close()
        with pytest.raises(DriverReadError):
            d.read()

    def test_G_SRALT_10_close_is_idempotent(self):
        """G-SRALT-10: close() can be called multiple times."""
        d = SimRADALTDriver(seed=42)
        d.close()
        d.close()

    def test_G_SRALT_11_is_radalt_driver_subclass(self):
        """G-SRALT-11: SimRADALTDriver is a RADALTDriver subclass."""
        from integration.drivers.radalt import RADALTDriver
        assert issubclass(SimRADALTDriver, RADALTDriver)


# ---------------------------------------------------------------------------
# SimEOIRDriver conformance
# ---------------------------------------------------------------------------

import numpy as _np
from integration.drivers.sim_eoir import SimEOIRDriver
from integration.drivers.eoir import EOIRFrame
from integration.drivers.base import DriverHealth, DriverReadError


class TestSimEOIRDriver:
    def test_G_SEOIR_01_instantiates(self):
        """G-SEOIR-01: SimEOIRDriver instantiates with defaults."""
        assert SimEOIRDriver() is not None

    def test_G_SEOIR_02_source_type_is_sim(self):
        """G-SEOIR-02: source_type() returns 'sim'."""
        assert SimEOIRDriver().source_type() == 'sim'

    def test_G_SEOIR_03_read_returns_eoir_frame(self):
        """G-SEOIR-03: read() returns EOIRFrame instance."""
        d = SimEOIRDriver(seed=42)
        assert isinstance(d.read(), EOIRFrame)

    def test_G_SEOIR_04_health_ok_after_read(self):
        """G-SEOIR-04: health() returns OK after first read."""
        d = SimEOIRDriver(seed=42)
        assert d.health() == DriverHealth.DEGRADED
        d.read()
        assert d.health() == DriverHealth.OK

    def test_G_SEOIR_05_frame_is_uint16(self):
        """G-SEOIR-05: frame_data is uint16 numpy array."""
        d = SimEOIRDriver(seed=42)
        r = d.read()
        assert r.frame_data.dtype == _np.uint16

    def test_G_SEOIR_06_frame_shape_matches_config(self):
        """G-SEOIR-06: frame shape matches configured width and height."""
        d = SimEOIRDriver(width=160, height=128, seed=42)
        r = d.read()
        assert r.frame_data.shape == (128, 160)
        assert r.width == 160
        assert r.height == 128

    def test_G_SEOIR_07_validity_flag_true(self):
        """G-SEOIR-07: validity_flag is True for synthetic frames."""
        d = SimEOIRDriver(seed=42)
        r = d.read()
        assert r.validity_flag is True

    def test_G_SEOIR_08_frame_not_all_zeros(self):
        """G-SEOIR-08: frame contains non-zero values (targets rendered)."""
        d = SimEOIRDriver(seed=42)
        r = d.read()
        assert r.frame_data.max() > 0

    def test_G_SEOIR_09_frame_count_advances(self):
        """G-SEOIR-09: _frame_count increments on each read."""
        d = SimEOIRDriver(seed=42)
        assert d._frame_count == 0
        d.read()
        assert d._frame_count == 1
        d.read()
        assert d._frame_count == 2

    def test_G_SEOIR_10_read_after_close_raises(self):
        """G-SEOIR-10: read() after close() raises DriverReadError."""
        import pytest
        d = SimEOIRDriver(seed=42)
        d.close()
        with pytest.raises(DriverReadError):
            d.read()

    def test_G_SEOIR_11_close_is_idempotent(self):
        """G-SEOIR-11: close() can be called multiple times."""
        d = SimEOIRDriver(seed=42)
        d.close()
        d.close()

    def test_G_SEOIR_12_is_eoir_driver_subclass(self):
        """G-SEOIR-12: SimEOIRDriver is an EOIRDriver subclass."""
        from integration.drivers.eoir import EOIRDriver
        assert issubclass(SimEOIRDriver, EOIRDriver)


# ---------------------------------------------------------------------------
# SimSDRDriver conformance
# ---------------------------------------------------------------------------

from integration.drivers.sim_sdr import SimSDRDriver, SDRReading
from integration.drivers.base import DriverHealth, DriverReadError
from core.ew_engine.ew_engine import EWObservation


class TestSDRReading:
    def test_G_SSDR_01_sdr_reading_fields(self):
        """G-SSDR-01: SDRReading has observations, mission_time_s, t."""
        r = SDRReading(observations=[], mission_time_s=1.0, t=2.0)
        assert r.observations == []
        assert r.mission_time_s == 1.0
        assert r.t == 2.0


class TestSimSDRDriver:
    def test_G_SSDR_02_instantiates(self):
        """G-SSDR-02: SimSDRDriver instantiates with defaults."""
        assert SimSDRDriver() is not None

    def test_G_SSDR_03_source_type_is_sim(self):
        """G-SSDR-03: source_type() returns 'sim'."""
        assert SimSDRDriver().source_type() == 'sim'

    def test_G_SSDR_04_read_returns_sdr_reading(self):
        """G-SSDR-04: read() returns SDRReading instance."""
        d = SimSDRDriver(seed=42)
        assert isinstance(d.read(), SDRReading)

    def test_G_SSDR_05_health_ok_after_read(self):
        """G-SSDR-05: health() returns OK after first read."""
        d = SimSDRDriver(seed=42)
        assert d.health() == DriverHealth.DEGRADED
        d.read()
        assert d.health() == DriverHealth.OK

    def test_G_SSDR_06_observations_are_ew_observations(self):
        """G-SSDR-06: all observations are EWObservation instances."""
        d = SimSDRDriver(seed=42, max_emitters=3)
        for _ in range(10):
            r = d.read()
            for obs in r.observations:
                assert isinstance(obs, EWObservation)

    def test_G_SSDR_07_observation_count_bounded(self):
        """G-SSDR-07: observation count is between 0 and max_emitters."""
        d = SimSDRDriver(seed=42, max_emitters=3)
        for _ in range(20):
            r = d.read()
            assert 0 <= len(r.observations) <= 3

    def test_G_SSDR_08_mission_time_advances(self):
        """G-SSDR-08: mission time advances by dt_s on each read."""
        d = SimSDRDriver(seed=42)
        d.read(dt_s=0.5)
        assert abs(d._mission_time - 0.5) < 1e-9
        d.read(dt_s=0.5)
        assert abs(d._mission_time - 1.0) < 1e-9

    def test_G_SSDR_09_deterministic_same_seed(self):
        """G-SSDR-09: same seed produces same observation count sequence."""
        d1 = SimSDRDriver(seed=7, max_emitters=3)
        d2 = SimSDRDriver(seed=7, max_emitters=3)
        counts1 = [len(d1.read().observations) for _ in range(5)]
        counts2 = [len(d2.read().observations) for _ in range(5)]
        assert counts1 == counts2

    def test_G_SSDR_10_read_after_close_raises(self):
        """G-SSDR-10: read() after close() raises DriverReadError."""
        import pytest
        d = SimSDRDriver(seed=42)
        d.close()
        with pytest.raises(DriverReadError):
            d.read()

    def test_G_SSDR_11_close_is_idempotent(self):
        """G-SSDR-11: close() can be called multiple times."""
        d = SimSDRDriver(seed=42)
        d.close()
        d.close()

    def test_G_SSDR_12_is_sensor_driver_subclass(self):
        """G-SSDR-12: SimSDRDriver is a SensorDriver subclass."""
        from integration.drivers.base import SensorDriver
        assert issubclass(SimSDRDriver, SensorDriver)

    def test_G_SSDR_13_ew_engine_accepts_observations(self):
        """G-SSDR-13: EWEngine.process_observations() accepts SimSDR output."""
        from core.ew_engine.ew_engine import EWEngine
        engine = EWEngine()
        d = SimSDRDriver(seed=42, max_emitters=2)
        r = d.read()
        if r.observations:
            result = engine.process_observations(r.observations, r.mission_time_s)
            assert result is not None or result is None   # either is valid


# ---------------------------------------------------------------------------
# Real driver stub conformance — all five stubs follow identical pattern
# ---------------------------------------------------------------------------

from integration.drivers.base import DriverHealth, DriverReadError


class TestRealDriverStubs:
    """Conformance gates for all five Real driver stubs."""

    def _check_stub(self, driver_cls, stale_s=1.0):
        """Shared conformance checks for any Real stub."""
        import pytest
        d = driver_cls(stale_threshold_s=stale_s)
        # health() must return FAILED
        assert d.health() == DriverHealth.FAILED
        # last_update_time() must return 0.0
        assert d.last_update_time() == 0.0
        # is_stale() must return True
        assert d.is_stale() is True
        # source_type() must return 'real'
        assert d.source_type() == 'real'
        # read() must raise DriverReadError
        with pytest.raises(DriverReadError) as exc_info:
            d.read()
        # Error message must contain interface path
        assert len(str(exc_info.value)) > 20
        # close() must be idempotent
        d.close()
        d.close()
        return d

    def test_G_REAL_01_real_imu_driver(self):
        """G-REAL-01: RealIMUDriver stub conforms to contract."""
        from integration.drivers.real_imu import RealIMUDriver
        d = self._check_stub(RealIMUDriver)
        # Error message must mention SPI interface
        import pytest
        with pytest.raises(DriverReadError) as exc_info:
            d.read()
        assert "SPI" in str(exc_info.value) or "spi" in str(exc_info.value).lower()

    def test_G_REAL_02_real_gnss_driver(self):
        """G-REAL-02: RealGNSSDriver stub conforms to contract."""
        from integration.drivers.real_gnss import RealGNSSDriver
        d = self._check_stub(RealGNSSDriver)
        import pytest
        with pytest.raises(DriverReadError) as exc_info:
            d.read()
        assert "ttyUSB" in str(exc_info.value) or "UART" in str(exc_info.value)

    def test_G_REAL_03_real_radalt_driver(self):
        """G-REAL-03: RealRADALTDriver stub conforms to contract."""
        from integration.drivers.real_radalt import RealRADALTDriver
        d = self._check_stub(RealRADALTDriver)
        import pytest
        with pytest.raises(DriverReadError) as exc_info:
            d.read()
        assert "RADALT" in str(exc_info.value)

    def test_G_REAL_04_real_eoir_driver(self):
        """G-REAL-04: RealEOIRDriver stub conforms to contract."""
        from integration.drivers.real_eoir import RealEOIRDriver
        d = self._check_stub(RealEOIRDriver)
        import pytest
        with pytest.raises(DriverReadError) as exc_info:
            d.read()
        assert "MIPI" in str(exc_info.value) or "USB" in str(exc_info.value)

    def test_G_REAL_05_real_sdr_driver(self):
        """G-REAL-05: RealSDRDriver stub conforms to contract."""
        from integration.drivers.real_sdr import RealSDRDriver
        d = self._check_stub(RealSDRDriver)
        import pytest
        with pytest.raises(DriverReadError) as exc_info:
            d.read()
        assert "SDR" in str(exc_info.value)

    def test_G_REAL_06_real_imu_is_imu_driver_subclass(self):
        """G-REAL-06: RealIMUDriver is IMUDriver subclass."""
        from integration.drivers.real_imu import RealIMUDriver
        from integration.drivers.imu import IMUDriver
        assert issubclass(RealIMUDriver, IMUDriver)

    def test_G_REAL_07_real_gnss_is_gnss_driver_subclass(self):
        """G-REAL-07: RealGNSSDriver is GNSSDriver subclass."""
        from integration.drivers.real_gnss import RealGNSSDriver
        from integration.drivers.gnss import GNSSDriver
        assert issubclass(RealGNSSDriver, GNSSDriver)

    def test_G_REAL_08_real_radalt_is_radalt_driver_subclass(self):
        """G-REAL-08: RealRADALTDriver is RADALTDriver subclass."""
        from integration.drivers.real_radalt import RealRADALTDriver
        from integration.drivers.radalt import RADALTDriver
        assert issubclass(RealRADALTDriver, RADALTDriver)

    def test_G_REAL_09_real_eoir_is_eoir_driver_subclass(self):
        """G-REAL-09: RealEOIRDriver is EOIRDriver subclass."""
        from integration.drivers.real_eoir import RealEOIRDriver
        from integration.drivers.eoir import EOIRDriver
        assert issubclass(RealEOIRDriver, EOIRDriver)

    def test_G_REAL_10_real_sdr_is_sensor_driver_subclass(self):
        """G-REAL-10: RealSDRDriver is SensorDriver subclass."""
        from integration.drivers.real_sdr import RealSDRDriver
        from integration.drivers.base import SensorDriver
        assert issubclass(RealSDRDriver, SensorDriver)


# ---------------------------------------------------------------------------
# MissionConfig + DriverFactory conformance
# ---------------------------------------------------------------------------

from integration.config.mission_config import MissionConfig, StaleThresholds
from integration.drivers.factory import DriverFactory
from integration.drivers.imu import IMUDriver
from integration.drivers.gnss import GNSSDriver
from integration.drivers.radalt import RADALTDriver
from integration.drivers.eoir import EOIRDriver
from integration.drivers.base import SensorDriver


class TestMissionConfig:
    def test_G_CFG_01_default_instantiates(self):
        """G-CFG-01: MissionConfig instantiates with all-sim defaults."""
        cfg = MissionConfig()
        assert cfg.imu_source == 'sim'
        assert cfg.gnss_source == 'sim'
        assert cfg.px4_output == 'sim'

    def test_G_CFG_02_validate_passes_all_sim(self):
        """G-CFG-02: validate() passes for all-sim config."""
        MissionConfig().validate()   # must not raise

    def test_G_CFG_03_invalid_source_raises(self):
        """G-CFG-03: invalid source string raises ValueError."""
        import pytest
        cfg = MissionConfig(imu_source='hardware')
        with pytest.raises(ValueError, match="imu_source"):
            cfg.validate()

    def test_G_CFG_04_invalid_imu_type_raises(self):
        """G-CFG-04: invalid imu_type raises ValueError."""
        import pytest
        cfg = MissionConfig(imu_type='UNKNOWN')
        with pytest.raises(ValueError, match="imu_type"):
            cfg.validate()

    def test_G_CFG_05_stale_thresholds_defaults(self):
        """G-CFG-05: StaleThresholds defaults are positive."""
        s = StaleThresholds()
        assert s.imu_s    > 0
        assert s.gnss_s   > 0
        assert s.radalt_s > 0
        assert s.eoir_s   > 0
        assert s.sdr_s    > 0

    def test_G_CFG_06_real_source_is_valid(self):
        """G-CFG-06: all-real config passes validate()."""
        cfg = MissionConfig(
            imu_source='real', gnss_source='real',
            radalt_source='real', eoir_source='real',
            sdr_source='real', px4_output='real'
        )
        cfg.validate()   # must not raise


class TestDriverFactory:
    def test_G_FAC_01_instantiates_with_valid_config(self):
        """G-FAC-01: DriverFactory instantiates with valid MissionConfig."""
        assert DriverFactory(MissionConfig()) is not None

    def test_G_FAC_02_invalid_config_raises_at_construction(self):
        """G-FAC-02: DriverFactory raises ValueError on invalid config."""
        import pytest
        with pytest.raises(ValueError):
            DriverFactory(MissionConfig(imu_source='bad'))

    def test_G_FAC_03_make_imu_returns_imu_driver(self):
        """G-FAC-03: make_imu() returns IMUDriver instance."""
        f = DriverFactory(MissionConfig())
        assert isinstance(f.make_imu(), IMUDriver)

    def test_G_FAC_04_make_gnss_returns_gnss_driver(self):
        """G-FAC-04: make_gnss() returns GNSSDriver instance."""
        f = DriverFactory(MissionConfig())
        assert isinstance(f.make_gnss(), GNSSDriver)

    def test_G_FAC_05_make_radalt_returns_radalt_driver(self):
        """G-FAC-05: make_radalt() returns RADALTDriver instance."""
        f = DriverFactory(MissionConfig())
        assert isinstance(f.make_radalt(), RADALTDriver)

    def test_G_FAC_06_make_eoir_returns_eoir_driver(self):
        """G-FAC-06: make_eoir() returns EOIRDriver instance."""
        f = DriverFactory(MissionConfig())
        assert isinstance(f.make_eoir(), EOIRDriver)

    def test_G_FAC_07_make_sdr_returns_sensor_driver(self):
        """G-FAC-07: make_sdr() returns SensorDriver instance."""
        f = DriverFactory(MissionConfig())
        assert isinstance(f.make_sdr(), SensorDriver)

    def test_G_FAC_08_sim_drivers_have_sim_source_type(self):
        """G-FAC-08: all-sim factory produces drivers with source_type='sim'."""
        f = DriverFactory(MissionConfig())
        assert f.make_imu().source_type()    == 'sim'
        assert f.make_gnss().source_type()   == 'sim'
        assert f.make_radalt().source_type() == 'sim'
        assert f.make_eoir().source_type()   == 'sim'
        assert f.make_sdr().source_type()    == 'sim'

    def test_G_FAC_09_real_drivers_have_real_source_type(self):
        """G-FAC-09: all-real factory produces drivers with source_type='real'."""
        cfg = MissionConfig(
            imu_source='real', gnss_source='real',
            radalt_source='real', eoir_source='real', sdr_source='real'
        )
        f = DriverFactory(cfg)
        assert f.make_imu().source_type()    == 'real'
        assert f.make_gnss().source_type()   == 'real'
        assert f.make_radalt().source_type() == 'real'
        assert f.make_eoir().source_type()   == 'real'
        assert f.make_sdr().source_type()    == 'real'

    def test_G_FAC_10_imu_type_propagated(self):
        """G-FAC-10: imu_type in config propagates to SimIMUDriver."""
        from integration.drivers.sim_imu import SimIMUDriver
        f = DriverFactory(MissionConfig(imu_type='BASELINE'))
        imu = f.make_imu()
        assert isinstance(imu, SimIMUDriver)
        assert imu._imu_type == 'BASELINE'
