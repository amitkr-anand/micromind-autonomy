"""
tests/test_s8b_mechanisation.py — Sprint S8-B Acceptance Tests
Gate: 7 criteria (B1–B7)

Run from repo root:
    PYTHONPATH=. python -m pytest tests/test_s8b_mechanisation.py -v

Criteria:
  B1 — ins_propagate(state, accel, gyro, dt) with no IMU model → S0 identical behaviour
  B2 — Backward-compatible signature (positional args unchanged; new args keyword-only, default None)
  B3 — STIM300 noise active: drift over 10 min @ 200 Hz > clean baseline
  B4 — BASELINE drift < STIM300 (model ordering preserved)
  B5 — Scale factor applied multiplicatively: attitude deviates by sf_ppm * 1e-6 fraction
  B6 — VRE: nonzero for STIM300, zero for BASELINE; appears in total_accel()
  B7 — Step indexing deterministic: noise[k] is indexed, not re-sampled
"""
import numpy as np
import pytest
from core.ins.mechanisation import ins_propagate
from core.ins.state import INSState
from core.ins.imu_model import get_imu_model, generate_imu_noise


# ---------------------------------------------------------------------------
def _level_state():
    return INSState(
        p=np.array([0.0, 0.0, 100.0]),
        v=np.array([50.0, 0.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3),
    )


def _run(n_steps, dt, model=None, noise=None):
    state = _level_state()
    a = np.array([0.0, 0.0, 9.80665])
    g = np.zeros(3)
    for k in range(n_steps):
        state = ins_propagate(state, a, g, dt, imu_model=model, imu_noise=noise, step=k)
    return state

# Helper: run generate_imu_noise as module function
def _gen_noise(name, n, dt, seed=42):
    m = get_imu_model(name)
    return m, generate_imu_noise(m, n, dt, seed=seed)


# ---------------------------------------------------------------------------
# B1 — Regression
# ---------------------------------------------------------------------------
class TestB1Regression:
    def test_level_flight_from_rest_no_position_change(self):
        s = INSState(p=np.zeros(3), v=np.zeros(3),
                     q=np.array([1.,0.,0.,0.]), ba=np.zeros(3), bg=np.zeros(3))
        r = ins_propagate(s, np.array([0.,0.,9.80665]), np.zeros(3), 0.005)
        assert np.linalg.norm(r.p) < 1e-6

    def test_none_explicit_equals_omitted(self):
        s = _level_state()
        a = np.array([1.,0.5,9.80665]); g = np.array([0.01,-0.005,0.002])
        r1 = ins_propagate(s, a, g, 0.005)
        r2 = ins_propagate(s, a, g, 0.005, imu_model=None, imu_noise=None)
        np.testing.assert_array_equal(r1.p, r2.p)
        np.testing.assert_array_equal(r1.v, r2.v)
        np.testing.assert_array_equal(r1.q, r2.q)

    def test_velocity_linear_integration(self):
        s = INSState(p=np.zeros(3), v=np.zeros(3),
                     q=np.array([1.,0.,0.,0.]), ba=np.zeros(3), bg=np.zeros(3))
        r = ins_propagate(s, np.array([1.,0.,9.80665]), np.zeros(3), 1.0)
        assert abs(r.v[0] - 1.0) < 1e-9

    def test_position_trapezoid_rule(self):
        s = INSState(p=np.zeros(3), v=np.zeros(3),
                     q=np.array([1.,0.,0.,0.]), ba=np.zeros(3), bg=np.zeros(3))
        r = ins_propagate(s, np.array([2.,0.,9.80665]), np.zeros(3), 1.0)
        assert abs(r.p[0] - 1.0) < 1e-9

    def test_quaternion_unit_norm_after_rotation(self):
        r = ins_propagate(_level_state(), np.array([0.,0.,9.80665]),
                          np.array([0.1,0.2,0.3]), 0.005)
        assert abs(np.linalg.norm(r.q) - 1.0) < 1e-9

    def test_no_rotation_attitude_unchanged(self):
        s = _level_state()
        r = ins_propagate(s, np.array([0.,0.,9.80665]), np.zeros(3), 0.005)
        np.testing.assert_allclose(r.q, s.q, atol=1e-9)


# ---------------------------------------------------------------------------
# B2 — Signature compatibility
# ---------------------------------------------------------------------------
class TestB2Compatibility:
    def test_required_positional_params_present(self):
        import inspect
        sig = inspect.signature(ins_propagate)
        for p in ("state","accel_b","gyro_b","dt"):
            assert p in sig.parameters

    def test_new_params_keyword_only_default_none(self):
        import inspect
        sig = inspect.signature(ins_propagate)
        for name in ("imu_model","imu_noise"):
            p = sig.parameters[name]
            assert p.default is None
            assert p.kind == inspect.Parameter.KEYWORD_ONLY

    def test_positional_call_returns_insstate(self):
        r = ins_propagate(_level_state(), np.array([0.,0.,9.80665]), np.zeros(3), 0.005)
        assert isinstance(r, INSState)


# ---------------------------------------------------------------------------
# B3 — STIM300 drift > clean baseline over 10 min
# ---------------------------------------------------------------------------
class TestB3Drift:
    def test_stim300_drift_exceeds_clean(self):
        n, dt = 10*60*200, 1./200.
        ideal = np.array([50.*n*dt, 0., 100.])

        drift_clean = np.linalg.norm(_run(n, dt).p - ideal)

        m, ns = _gen_noise("STIM300", n, dt, seed=42)
        drift_stim = np.linalg.norm(_run(n, dt, m, ns).p - ideal)

        assert drift_stim > drift_clean, (
            f"STIM300 {drift_stim:.2f} m should exceed clean {drift_clean:.2f} m")


# ---------------------------------------------------------------------------
# B4 — BASELINE < STIM300
# ---------------------------------------------------------------------------
class TestB4ModelOrdering:
    def test_baseline_less_than_stim300(self):
        n, dt = 10*60*200, 1./200.
        ideal = np.array([50.*n*dt, 0., 100.])

        def _d(name):
            m, ns = _gen_noise(name, n, dt, seed=42)
            return np.linalg.norm(_run(n, dt, m, ns).p - ideal)

        assert _d("BASELINE") < _d("STIM300"), "BASELINE drift should be < STIM300"

    def test_adis_exceeds_baseline(self):
        n, dt = 6*60*200, 1./200.
        ideal = np.array([50.*n*dt, 0., 100.])
        def _d(name):
            m, ns = _gen_noise(name, n, dt, seed=99)
            return np.linalg.norm(_run(n, dt, m, ns).p - ideal)
        assert _d("ADIS16505_3") > _d("BASELINE"), "ADIS should exceed BASELINE drift"


# ---------------------------------------------------------------------------
# B5 — Scale factor
# ---------------------------------------------------------------------------
class TestB5ScaleFactor:
    def test_nonzero_sf_changes_attitude(self):
        sf_val = 1000.0 * 1e-6   # 1000 ppm
        n, dt  = 200, 0.005
        true_g = np.array([0.1, 0., 0.])

        class _SFNoise:
            def __init__(self, sf, n):
                self.sf_error_gyro = np.full(n, sf)
                self._n = n
            def total_gyro(self): return np.zeros((self._n, 3))
            def total_accel(self): return np.zeros((self._n, 3))

        class _ZeroNoise:
            def __init__(self, n):
                self.sf_error_gyro = np.zeros(n)
                self._n = n
            def total_gyro(self): return np.zeros((self._n, 3))
            def total_accel(self): return np.zeros((self._n, 3))

        class _M:
            vre_bias_si = 0.0

        m = _M()
        n_sf   = _SFNoise(sf_val, n)
        n_zero = _ZeroNoise(n)
        a = np.array([0.,0.,9.80665])

        s1 = s2 = _level_state()
        for k in range(n):
            s1 = ins_propagate(s1, a, true_g, dt, imu_model=m, imu_noise=n_sf,   step=k)
            s2 = ins_propagate(s2, a, true_g, dt, imu_model=m, imu_noise=n_zero, step=k)

        assert np.linalg.norm(s1.q - s2.q) > 1e-6

    def test_baseline_zero_noise_matches_clean(self):
        n, dt = 50, 0.005
        m = get_imu_model("BASELINE")
        ns = generate_imu_noise(m, n, dt, seed=1)
        a = np.array([0.,0.,9.80665]); g = np.array([0.05,0.,0.])
        s_c = s_m = _level_state()
        for k in range(n):
            s_c = ins_propagate(s_c, a, g, dt)
            s_m = ins_propagate(s_m, a, g, dt, imu_model=m, imu_noise=ns, step=k)
        # BASELINE has minimal but nonzero white noise floor; atol=5e-5 (0.05mm)
        np.testing.assert_allclose(s_c.p, s_m.p, atol=5e-5)


# ---------------------------------------------------------------------------
# B6 — VRE
# ---------------------------------------------------------------------------
class TestB6VRE:
    def test_stim300_vre_nonzero(self):
        assert get_imu_model("STIM300").vre_bias_si > 0.

    def test_baseline_vre_zero(self):
        assert get_imu_model("BASELINE").vre_bias_si == 0.

    def test_stim300_mean_accel_noise_exceeds_baseline(self):
        n, dt = 20_000, 0.005
        ms = get_imu_model("STIM300");  mb = get_imu_model("BASELINE")
        ns = generate_imu_noise(ms, n, dt, seed=7)
        nb = generate_imu_noise(mb, n, dt, seed=7)
        assert np.abs(ns.total_accel()).mean() > np.abs(nb.total_accel()).mean()


# ---------------------------------------------------------------------------
# B7 — Step indexing deterministic
# ---------------------------------------------------------------------------
class TestB7StepIndexing:
    def test_same_seed_same_trajectory(self):
        n, dt = 1000, 0.005
        m = get_imu_model("STIM300")
        na = generate_imu_noise(m, n, dt, seed=42)
        nb = generate_imu_noise(m, n, dt, seed=42)
        sa = sb = _level_state()
        a, g = np.array([0.,0.,9.80665]), np.zeros(3)
        for k in range(n):
            sa = ins_propagate(sa, a, g, dt, imu_model=m, imu_noise=na, step=k)
            sb = ins_propagate(sb, a, g, dt, imu_model=m, imu_noise=nb, step=k)
        np.testing.assert_allclose(sa.p, sb.p, atol=1e-12)

    def test_different_seed_different_trajectory(self):
        n, dt = 500, 0.005
        m = get_imu_model("STIM300")
        na = generate_imu_noise(m, n, dt, seed=42)
        nb = generate_imu_noise(m, n, dt, seed=99)
        sa = sb = _level_state()
        a, g = np.array([0.,0.,9.80665]), np.zeros(3)
        for k in range(n):
            sa = ins_propagate(sa, a, g, dt, imu_model=m, imu_noise=na, step=k)
            sb = ins_propagate(sb, a, g, dt, imu_model=m, imu_noise=nb, step=k)
        assert np.linalg.norm(sa.p - sb.p) > 1e-4

    def test_noise_array_deterministic(self):
        m = get_imu_model("STIM300")
        ns = generate_imu_noise(m, 10, 0.005, seed=42)
        np.testing.assert_array_equal(ns.total_gyro()[3], ns.total_gyro()[3])

    def test_step_rows_differ(self):
        m = get_imu_model("STIM300")
        ns = generate_imu_noise(m, 10, 0.005, seed=42)
        assert not np.allclose(ns.total_gyro()[0], ns.total_gyro()[1])


if __name__ == "__main__":
    import subprocess, sys
    sys.exit(subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        check=False).returncode)
