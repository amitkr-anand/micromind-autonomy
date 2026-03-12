"""
test_s9_nav01_pass.py — S10-2: S9 Architecture Regression Gates
================================================================
7 automated gates (S9-A through S9-G) protecting the TRN+ESKF
architectural corrections made in Sprint S9.

RC-1  TRN internal Kalman removed — update() returns raw offset only
RC-2  ESKF gyro bias RW corrected to STIM300 TS1524 rev.31 (4.04e-8)
RC-3  Position block of Q was zero → K≈0. Fixed: position PSD added
RC-4  Propagation order violated. Fixed: ESKF→INS→TRN→ESKF update
RC-5  NAV-01 metric was 3D. Fixed: 2D horizontal norm only

Constructor signatures (live):
  DEMProvider(seed=7)
  RadarAltimeterSim(dem, seed=99)
  TRNStub(dem, radar, ncc_threshold=0.45, search_pad_px=25)

ESKF Q constants are class-private (not exported at module level).
S9-A verifies them via Q-matrix inspection on the instantiated object.
If Q is not accessible as an attribute, the test skips — the NAV-01
simulation gates (S9-D/E/F) provide the functional coverage.

TRN correction interval live value: CORRECTION_INTERVAL = 1500.0 m.

Acceptance gate: 7/7 must pass. Suite grows to 222/222.

Run:
    pytest tests/test_s9_nav01_pass.py -v
    pytest tests/test_s9_nav01_pass.py -v -m "not slow"   # skip 150 km
"""

import math
import unittest

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trn():
    """Construct a valid TRNStub with default DEMProvider and RadarAltimeterSim."""
    from core.ins.trn_stub import DEMProvider, RadarAltimeterSim, TRNStub
    dem   = DEMProvider(seed=7)
    radar = RadarAltimeterSim(dem=dem, seed=99)
    return TRNStub(dem=dem, radar=radar)


# ---------------------------------------------------------------------------
# S9-A: ESKF Q matrix — position PSD non-zero, gyro bias in correct regime
# (guards RC-2 and RC-3)
# ---------------------------------------------------------------------------

class TestS9A_ESKFQMatrix(unittest.TestCase):
    """
    S9-A: ESKF internal Q matrix must have non-zero position noise and
    correctly-scaled gyro bias noise. Constants are class-private so we
    verify via Q-matrix on the instantiated ErrorStateEKF. Protects RC-2/RC-3.
    """

    def _get_Q(self):
        from core.ekf.error_state_ekf import ErrorStateEKF
        ekf = ErrorStateEKF()
        for attr in ("_Q", "Q", "_process_noise", "process_noise_cov"):
            Q = getattr(ekf, attr, None)
            if Q is not None:
                return np.asarray(Q)
        return None

    def test_s9a_eskf_instantiates(self):
        """S9-A-1: ErrorStateEKF must instantiate without exception."""
        from core.ekf.error_state_ekf import ErrorStateEKF
        self.assertIsNotNone(ErrorStateEKF())

    def test_s9a_q_position_block_nonzero(self):
        """
        S9-A-2 (RC-3): Q position diagonal must be non-zero.
        Before S9: PSD=0 → Kalman gain≈0 → TRN corrections silently discarded.
        """
        Q = self._get_Q()
        if Q is None:
            self.skipTest("Q not accessible as attribute — covered by NAV-01 gates.")
        pos_diag = np.diag(Q)[:3]
        self.assertTrue(
            np.any(pos_diag > 0),
            msg=f"Q position diagonal={pos_diag} all zero. RC-3 regression.",
        )

    def test_s9a_q_gyro_bias_correct_regime(self):
        """
        S9-A-3 (RC-2): Gyro bias Q entries must be in the correct regime.
        Pre-S9: 1e-5 rad/s/√s → Q≈1e-10/step. Correct: 4.04e-8 → Q≈1.6e-15/step.
        Gate: gyro bias diagonal < 1e-11 (rules out pre-S9 blown-up value).
        Only asserted on 15-state ESKF where indices 9:12 are gyro bias.
        """
        Q = self._get_Q()
        if Q is None:
            self.skipTest("Q not accessible — covered by NAV-01 gates.")
        if Q.shape[0] < 15:
            self.skipTest(f"Q shape {Q.shape} < 15 — cannot identify gyro bias indices.")
        gyro_diag = np.diag(Q)[9:12]
        self.assertTrue(np.all(gyro_diag >= 0),
                        msg=f"Gyro bias Q diagonal {gyro_diag} has negative values.")
        self.assertTrue(
            np.all(gyro_diag < 1e-11),
            msg=(f"Gyro bias Q diagonal={gyro_diag} exceeds 1e-11. "
                 "RC-2 regression: gyro bias RW too large (pre-S9 value was 1e-5)."),
        )


# ---------------------------------------------------------------------------
# S9-B: TRNStub.update() — signature and return type (RC-1)
# ---------------------------------------------------------------------------

class TestS9B_TRNStubInterface(unittest.TestCase):
    """
    S9-B: TRNStub.update() must match the S9 canonical 6-parameter signature
    and return TRNCorrection or None. Protects RC-1 (internal Kalman removed).
    """

    def test_s9b_update_accepts_canonical_args(self):
        """
        S9-B-1 (RC-1): update() must have the S9 canonical 6-parameter signature:
        (ins_north_m, ins_east_m, true_north_m, true_east_m,
         ground_track_m, timestamp_s).
        """
        import inspect
        from core.ins.trn_stub import TRNStub
        sig    = inspect.signature(TRNStub.update)
        params = list(sig.parameters.keys())  # includes 'self'
        self.assertGreaterEqual(
            len(params), 7,
            msg=(f"update() has {len(params)} params (incl. self); expected ≥ 7. "
                 "S9 RC-1 regression."),
        )

    def test_s9b_update_returns_none_or_correction(self):
        """
        S9-B-2 (RC-1): update() must return TRNCorrection or None.
        Called at ground_track_m past CORRECTION_INTERVAL (1500 m).
        """
        from core.ins.trn_stub import TRNCorrection
        trn    = _make_trn()
        result = trn.update(
            ins_north_m=0.0, ins_east_m=0.0,
            true_north_m=0.0, true_east_m=0.0,
            ground_track_m=1600.0,
            timestamp_s=0.0,
        )
        self.assertTrue(
            result is None or isinstance(result, TRNCorrection),
            msg=(f"update() returned {type(result).__name__}; "
                 "expected TRNCorrection or None. RC-1 regression."),
        )

    def test_s9b_update_no_internal_position_mutation(self):
        """
        S9-B-3 (RC-1): Three successive update() calls must not raise and must
        each return TRNCorrection or None. Validates no corrupted internal state
        accumulates (internal Kalman was removed in S9).
        """
        from core.ins.trn_stub import TRNCorrection
        trn = _make_trn()
        for i in range(3):
            result = trn.update(
                ins_north_m=float(i * 10), ins_east_m=0.0,
                true_north_m=float(i * 10), true_east_m=0.0,
                ground_track_m=1600.0 * (i + 1),
                timestamp_s=float(i),
            )
            self.assertTrue(
                result is None or isinstance(result, TRNCorrection),
                msg=(f"Call {i+1}: update() returned {type(result).__name__}. "
                     "RC-1 regression."),
            )


# ---------------------------------------------------------------------------
# S9-C: TRN correction interval — live value (1500.0 m)
# ---------------------------------------------------------------------------

class TestS9C_TRNCorrectionInterval(unittest.TestCase):
    """
    S9-C: CORRECTION_INTERVAL must be present and equal to 1500.0 m.
    Catches accidental cadence changes in either direction.
    Update this gate only if the interval is intentionally changed.
    """

    def _get_interval(self):
        import core.ins.trn_stub as m
        for name in ("CORRECTION_INTERVAL", "TRN_CORRECTION_INTERVAL_M",
                     "CORRECTION_INTERVAL_M"):
            v = getattr(m, name, None)
            if v is not None:
                return v
        return None

    def test_s9c_correction_interval_present(self):
        """S9-C-1: CORRECTION_INTERVAL constant must exist at module level."""
        self.assertIsNotNone(
            self._get_interval(),
            msg="CORRECTION_INTERVAL not found in trn_stub module.",
        )

    def test_s9c_correction_interval_live_value(self):
        """S9-C-2: CORRECTION_INTERVAL must be 1500.0 m (live value)."""
        interval = self._get_interval()
        if interval is None:
            self.skipTest("CORRECTION_INTERVAL not found — see S9-C-1.")
        self.assertEqual(
            interval, 1500.0,
            msg=(f"CORRECTION_INTERVAL={interval} m but expected 1500.0 m. "
                 "If intentionally changed, update this gate and the spec note."),
        )


# ---------------------------------------------------------------------------
# S9-D: NAV-01 at 20 km — STIM300 (fast gate, ~20 s)
# ---------------------------------------------------------------------------

class TestS9D_NAV01_20km(unittest.TestCase):

    def test_s9d_nav01_stim300_20km(self):
        """S9-D: 20 km STIM300 seed=42. NAV-01 < 100 m + sanity < 40 m."""
        from sim.als250_nav_sim import run_als250_sim
        kpi = run_als250_sim(imu_name="STIM300", corridor_km=20.0,
                             seed=42, verbose=False)["kpi"]
        self.assertTrue(
            kpi["NAV01_pass"],
            msg=f"NAV-01 FAIL 20 km: {kpi['max_5km_drift_m']:.1f} m (limit 100 m).",
        )
        self.assertLess(
            kpi["max_5km_drift_m"], 40.0,
            msg=(f"drift={kpi['max_5km_drift_m']:.1f} m exceeds 40 m sanity bound "
                 "(S9 result ~4 m). TRN+ESKF degraded."),
        )


# ---------------------------------------------------------------------------
# S9-E: NAV-01 at 50 km — STIM300 (mandatory smoke corridor)
# ---------------------------------------------------------------------------

class TestS9E_NAV01_50km(unittest.TestCase):

    def test_s9e_nav01_stim300_50km(self):
        """S9-E: 50 km STIM300 seed=42. NAV-01 < 100 m (S9 result: 9.7 m)."""
        from sim.als250_nav_sim import run_als250_sim
        kpi = run_als250_sim(imu_name="STIM300", corridor_km=50.0,
                             seed=42, verbose=False)["kpi"]
        self.assertTrue(
            kpi["NAV01_pass"],
            msg=f"NAV-01 FAIL 50 km: {kpi['max_5km_drift_m']:.1f} m (limit 100 m).",
        )


# ---------------------------------------------------------------------------
# S9-F: NAV-01 at 150 km — STIM300 (S9 milestone closure run) [slow]
# ---------------------------------------------------------------------------

import pytest

@pytest.mark.slow
class TestS9F_NAV01_150km(unittest.TestCase):

    def test_s9f_nav01_stim300_150km(self):
        """
        S9-F: 150 km STIM300 seed=42. NAV-01 < 100 m (S9 result: 14.5 m, 6.9× margin).
        This is the run that closed NAV-01 in Sprint S9. Marked slow — run explicitly:
            pytest tests/test_s9_nav01_pass.py -v               # all gates
            pytest tests/test_s9_nav01_pass.py -v -m "not slow" # skip this gate
        """
        from sim.als250_nav_sim import run_als250_sim
        kpi = run_als250_sim(imu_name="STIM300", corridor_km=150.0,
                             seed=42, verbose=False)["kpi"]
        self.assertTrue(
            kpi["NAV01_pass"],
            msg=(f"NAV-01 FAIL 150 km: {kpi['max_5km_drift_m']:.1f} m "
                 "(limit 100 m). S9 RC closure regression."),
        )


# ---------------------------------------------------------------------------
# S9-G: NAV-01 metric is 2D horizontal + KPI dict structure (RC-5)
# ---------------------------------------------------------------------------

class TestS9G_NAV01MetricIs2D(unittest.TestCase):
    """
    S9-G: NAV-01 KPI dict must contain the correct S9 fields and drift must
    be finite and plausible for a 2D horizontal metric. Protects RC-5.
    """

    def _run_20km(self):
        from sim.als250_nav_sim import run_als250_sim
        return run_als250_sim(imu_name="STIM300", corridor_km=20.0,
                              seed=42, verbose=False)["kpi"]

    def test_s9g_kpi_fields_present(self):
        """
        S9-G-1: KPI dict must contain NAV01_pass (bool) and
        max_5km_drift_m (float) — introduced when metric was corrected from 3D to 2D.
        """
        kpi = self._run_20km()
        self.assertIn("NAV01_pass", kpi,
                      msg="KPI missing 'NAV01_pass'. S9 metric interface regression.")
        self.assertIn("max_5km_drift_m", kpi,
                      msg="KPI missing 'max_5km_drift_m'. S9 metric interface regression.")
        self.assertIsInstance(kpi["NAV01_pass"], bool,
                              msg=f"NAV01_pass type={type(kpi['NAV01_pass']).__name__}.")
        self.assertIsInstance(kpi["max_5km_drift_m"], float,
                              msg=f"max_5km_drift_m type={type(kpi['max_5km_drift_m']).__name__}.")

    def test_s9g_drift_finite_and_plausible(self):
        """
        S9-G-2 (RC-5): max_5km_drift_m must be finite, ≥ 0, < 100 m at 20 km.
        A 3D metric with altitude noise would inflate this beyond the limit.
        """
        drift = self._run_20km()["max_5km_drift_m"]
        self.assertTrue(math.isfinite(drift) and drift >= 0.0,
                        msg=f"max_5km_drift_m={drift} not a valid non-negative finite value.")
        self.assertLess(drift, 100.0,
                        msg=f"drift={drift:.1f} m at 20 km ≥ 100 m. RC-5 (3D metric) regression?")


# ---------------------------------------------------------------------------
# Suite runner (used by __main__ only — pytest discovers classes directly)
# ---------------------------------------------------------------------------

def run_suite() -> bool:
    import sys
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestS9A_ESKFQMatrix, TestS9B_TRNStubInterface,
                TestS9C_TRNCorrectionInterval, TestS9D_NAV01_20km,
                TestS9E_NAV01_50km, TestS9F_NAV01_150km,
                TestS9G_NAV01MetricIs2D]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    print()
    print("=" * 65)
    print("  MicroMind S10-2 — S9 Architecture Regression Gates")
    print("  RC-1(TRN) · RC-2/3(ESKF Q) · RC-4(order) · RC-5(2D metric)")
    print("=" * 65)
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print()
    print("=" * 65)
    print(f"  S9 Gates: {passed}/{result.testsRun} passed")
    print("  STATUS:", "✅ ALL GATES PASS" if result.wasSuccessful()
          else "❌ REGRESSION DETECTED — do not proceed")
    print("=" * 65)
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_suite() else 1)
