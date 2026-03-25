# tests/test_s_nep_04a_interface.py
# MicroMind — S-NEP-04 Step 04-A gate tests
#
# T-01  Frame Sanity Check          — HARD GATE (zero tolerance)
# T-02  Covariance Extraction Check — HARD GATE (zero tolerance)
#
# Both must pass before any replay run (04-B) is attempted.
# Run:  python3 -m pytest tests/test_s_nep_04a_interface.py -v

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports — fusion layer
# ---------------------------------------------------------------------------
from core.fusion.frame_utils import (
    R_ENU_TO_NED,
    extract_vio_position_cov,
    rotate_cov_enu_to_ned,
    rotate_pos_enu_to_ned,
)
from core.fusion.fusion_logger import FusionLogger
from core.fusion.vio_covariance_error import VIOCovarianceError

# ---------------------------------------------------------------------------
# Imports — ESKF (must not be modified; update_vio is additive)
# ---------------------------------------------------------------------------
from core.ekf.error_state_ekf import ErrorStateEKF
from core.ins.state import INSState


# ===========================================================================
# Helpers
# ===========================================================================

def _make_zeroed_state() -> INSState:
    """INSState with all fields at zero — clean baseline for T-01."""
    return INSState(
        p=np.zeros(3),
        v=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3),
    )


def _make_eskf() -> ErrorStateEKF:
    """Fresh ESKF with default initialisation."""
    eskf = ErrorStateEKF()
    return eskf


def _synthetic_ros_cov_36(sigma_pos: float, sigma_rot: float = 0.01) -> np.ndarray:
    """
    Build a synthetic 36-element ROS2 flat covariance array.

    Position block diagonals = sigma_pos² (m²).
    Rotation block diagonals = sigma_rot² (rad²).
    All off-diagonals = 0.
    """
    cov_6x6 = np.zeros((6, 6))
    cov_6x6[0, 0] = sigma_pos ** 2
    cov_6x6[1, 1] = sigma_pos ** 2
    cov_6x6[2, 2] = sigma_pos ** 2
    cov_6x6[3, 3] = sigma_rot ** 2
    cov_6x6[4, 4] = sigma_rot ** 2
    cov_6x6[5, 5] = sigma_rot ** 2
    return cov_6x6.flatten()


# ===========================================================================
# T-01 — Frame Sanity Check (HARD GATE)
# ===========================================================================

class TestT01FrameSanity:
    """
    T-01: Known ENU pose → correct NED fused state.

    Input:  [1.0, 0.0, 0.0] East in ENU (1 m East)
    After rotation: [0.0, 1.0, 0.0] in NED (1 m North)
    Zero tolerance: [1.0, 0.0, 0.0] as fused NED position is a FAIL.

    The ESKF starts at origin.  One VIO update with a low-noise covariance
    should pull the estimated position toward the measurement.
    North component must dominate; East component must be near-zero.
    """

    def test_rotation_matrix_is_correct(self):
        """R_ENU_TO_NED has the exact values specified in the interface contract."""
        expected = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float64)
        np.testing.assert_array_equal(
            R_ENU_TO_NED, expected,
            err_msg="R_ENU_TO_NED does not match interface contract [[0,1,0],[1,0,0],[0,0,-1]]",
        )

    def test_east_vector_rotates_to_north(self):
        """
        1 m East in ENU → 0 m North, 1 m East, 0 m Down in NED.

        ENU [1, 0, 0] = East
        NED result:
          NED_x (North) = ENU_y = 0
          NED_y (East)  = ENU_x = 1
          NED_z (Down)  = -ENU_z = 0
        """
        pos_enu = np.array([1.0, 0.0, 0.0])
        pos_ned = rotate_pos_enu_to_ned(pos_enu)
        expected_ned = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(
            pos_ned, expected_ned, atol=1e-12,
            err_msg=(
                f"1 m East (ENU) should rotate to [0, 1, 0] NED, "
                f"got {pos_ned}. This is axis swap IFM-02."
            ),
        )

    def test_north_vector_rotates_to_north(self):
        """1 m North in ENU → 1 m North in NED."""
        pos_enu = np.array([0.0, 1.0, 0.0])
        pos_ned = rotate_pos_enu_to_ned(pos_enu)
        expected_ned = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(pos_ned, expected_ned, atol=1e-12)

    def test_up_rotates_to_down_negated(self):
        """1 m Up in ENU → -1 m Down in NED (NED_z is Down, so +Up = -Down)."""
        pos_enu = np.array([0.0, 0.0, 1.0])
        pos_ned = rotate_pos_enu_to_ned(pos_enu)
        expected_ned = np.array([0.0, 0.0, -1.0])
        np.testing.assert_allclose(pos_ned, expected_ned, atol=1e-12)

    def test_vio_update_fuses_toward_correct_ned_position(self):
        """
        Core T-01 gate: ESKF fused state converges toward [0, 1, 0] NED,
        NOT toward [1, 0, 0] NED.

        Protocol:
        - ESKF starts at origin (p = [0, 0, 0]).
        - VIO measurement = 1 m East in ENU → rotate to NED.
        - Low-noise covariance (σ = 0.01 m) forces strong correction.
        - After one update: fused p_north ≈ 1.0, p_east ≈ 0.0.
        """
        state = _make_zeroed_state()
        eskf = _make_eskf()

        # 1 m East in ENU
        pos_enu = np.array([1.0, 0.0, 0.0])
        pos_ned = rotate_pos_enu_to_ned(pos_enu)  # → [0, 1, 0]

        # Low-noise diagonal covariance (σ = 0.01 m → σ² = 1e-4 m²)
        sigma = 0.01
        cov_enu = np.diag([sigma**2, sigma**2, sigma**2])
        cov_ned = rotate_cov_enu_to_ned(cov_enu)

        nis, rejected, innov_mag = eskf.update_vio(state, pos_ned, cov_ned)

        assert not rejected, "update_vio unexpectedly rejected a valid low-noise measurement"

        # Apply correction to state
        eskf.inject(state)

        # T-01 hard assertions — zero tolerance on axis assignment
        # ENU [1,0,0] = 1m East → NED [0,1,0]: p_east (NED_y) ≈ 1.0, p_north (NED_x) ≈ 0.0
        assert state.p[1] > 0.5, (
            f"p_east (NED_y) should be ~1.0 after fusing 1m-East ENU, "
            f"got {state.p[1]:.4f}. Axis swap detected (IFM-02)."
        )
        assert abs(state.p[0]) < 0.5, (
            f"p_north (NED_x) should be ~0.0 after fusing 1m-East ENU, "
            f"got {state.p[0]:.4f}. Axis swap detected (IFM-02)."
        )

        # NIS should be finite and positive for a valid update
        assert np.isfinite(nis), f"NIS must be finite, got {nis}"
        assert nis >= 0.0, f"NIS must be non-negative, got {nis}"

    def test_covariance_rotation_preserves_positive_definite(self):
        """Rotated covariance must remain positive definite."""
        cov_enu = np.diag([0.01, 0.02, 0.03])
        cov_ned = rotate_cov_enu_to_ned(cov_enu)
        eigenvalues = np.linalg.eigvalsh(cov_ned)
        assert np.all(eigenvalues > 0), (
            f"Rotated covariance lost positive-definiteness: eigenvalues = {eigenvalues}"
        )

    def test_covariance_rotation_formula(self):
        """
        R @ C_enu @ R.T must equal rotate_cov_enu_to_ned(C_enu).
        Verifies the similarity transform is applied correctly.
        """
        R = R_ENU_TO_NED
        cov_enu = np.array([
            [0.04, 0.001, 0.0],
            [0.001, 0.02, 0.0],
            [0.0, 0.0, 0.015],
        ])
        expected = R @ cov_enu @ R.T
        actual = rotate_cov_enu_to_ned(cov_enu)
        np.testing.assert_allclose(actual, expected, atol=1e-15)

    def test_rejected_on_zero_covariance_diagonal(self):
        """update_vio must return (0.0, True) if any diagonal of cov_pos_ned ≤ 0."""
        state = _make_zeroed_state()
        eskf = _make_eskf()
        pos_ned = np.array([0.0, 1.0, 0.0])
        bad_cov = np.diag([0.01, 0.0, 0.01])   # zero on axis 1
        nis, rejected, innov_mag = eskf.update_vio(state, pos_ned, bad_cov)
        assert rejected is True, "update_vio must reject zero-diagonal covariance"
        assert nis == 0.0, "NIS must be 0.0 on rejection"

    def test_update_vio_does_not_modify_state_on_rejection(self):
        """ESKF state must be unchanged after a rejection."""
        state = _make_zeroed_state()
        eskf = _make_eskf()
        P_before = eskf.P.copy()
        x_before = eskf.x.copy()

        bad_cov = np.diag([0.0, 0.01, 0.01])
        eskf.update_vio(state, np.array([0.0, 1.0, 0.0]), bad_cov)

        np.testing.assert_array_equal(eskf.P, P_before)
        np.testing.assert_array_equal(eskf.x, x_before)


# ===========================================================================
# T-02 — Covariance Extraction Check (HARD GATE)
# ===========================================================================

class TestT02CovarianceExtraction:
    """
    T-02: Covariance extraction and trace bounds consistent with Stage-2.

    Stage-2 σ_position range: [0.09, 0.22] m → diag ∈ [0.0081, 0.0484] m²
    For a 3-axis diagonal covariance: trace ∈ [3×0.0081, 3×0.0484]
                                               = [0.0243, 0.1452] m²
    The test spec uses [0.01, 0.20] as the assertion range (slight margin).

    30 synthetic arrays are generated and checked.
    Zero-covariance guard is verified with a deliberately bad input.
    """

    # Stage-2 σ range from OpenVINS endurance validation
    _SIGMA_MIN = 0.09   # m
    _SIGMA_MAX = 0.22   # m

    # Assertion range for trace(R) — slight margin on stage-2 range
    _TRACE_MIN = 0.01   # m²
    _TRACE_MAX = 0.20   # m²

    def _make_stage2_cov_36(self, rng: np.random.Generator) -> np.ndarray:
        """
        Build a synthetic 36-element ROS2 covariance with diagonals
        sampled from the Stage-2 σ range.
        """
        sigmas = rng.uniform(self._SIGMA_MIN, self._SIGMA_MAX, size=3)
        cov_6x6 = np.zeros((6, 6))
        cov_6x6[0, 0] = sigmas[0] ** 2
        cov_6x6[1, 1] = sigmas[1] ** 2
        cov_6x6[2, 2] = sigmas[2] ** 2
        cov_6x6[3, 3] = 0.001  # rotation — irrelevant but non-zero
        cov_6x6[4, 4] = 0.001
        cov_6x6[5, 5] = 0.001
        return cov_6x6.flatten()

    def test_30_stage2_samples_all_diagonal_positive(self):
        """
        All 30 Stage-2 samples: every diagonal element of the extracted
        3×3 block must be > 0 after rotation.
        """
        rng = np.random.default_rng(seed=42)
        failures = []
        for i in range(30):
            cov_36 = self._make_stage2_cov_36(rng)
            cov_pos_enu = extract_vio_position_cov(cov_36)
            cov_pos_ned = rotate_cov_enu_to_ned(cov_pos_enu)
            diag = np.diag(cov_pos_ned)
            if np.any(diag <= 0.0):
                failures.append((i, diag.tolist()))

        assert not failures, (
            f"Samples with non-positive diagonal after rotation: {failures}"
        )

    def test_30_stage2_samples_trace_in_range(self):
        """
        All 30 Stage-2 samples: trace(R_ned) must be in [0.01, 0.20] m².
        """
        rng = np.random.default_rng(seed=42)
        out_of_range = []
        for i in range(30):
            cov_36 = self._make_stage2_cov_36(rng)
            cov_pos_enu = extract_vio_position_cov(cov_36)
            cov_pos_ned = rotate_cov_enu_to_ned(cov_pos_enu)
            trace = float(np.trace(cov_pos_ned))
            if not (self._TRACE_MIN <= trace <= self._TRACE_MAX):
                out_of_range.append((i, trace))

        assert not out_of_range, (
            f"Samples with trace(R) outside [{self._TRACE_MIN}, {self._TRACE_MAX}] m²: "
            f"{out_of_range}"
        )

    def test_30_stage2_samples_consistent_with_sigma_range(self):
        """
        Each diagonal of the extracted position covariance must be
        consistent with the Stage-2 σ range: diag ∈ [σ_min², σ_max²].
        """
        diag_min = self._SIGMA_MIN ** 2   # 0.0081
        diag_max = self._SIGMA_MAX ** 2   # 0.0484

        rng = np.random.default_rng(seed=42)
        failures = []
        for i in range(30):
            cov_36 = self._make_stage2_cov_36(rng)
            cov_pos_enu = extract_vio_position_cov(cov_36)
            # Check ENU block before rotation (rotation preserves magnitude)
            diag = np.diag(cov_pos_enu)
            bad = [
                (ax, float(d))
                for ax, d in enumerate(diag)
                if not (diag_min <= d <= diag_max)
            ]
            if bad:
                failures.append((i, bad))

        assert not failures, (
            f"Samples with diagonal outside Stage-2 σ² range "
            f"[{diag_min:.4f}, {diag_max:.4f}]: {failures}"
        )

    def test_position_block_indices_are_correct(self):
        """
        extract_vio_position_cov must extract [0:3, 0:3] — not [3:6, 3:6].
        Confirmed by placing a known value at position (0,0) only.
        """
        cov_6x6 = np.eye(6) * 0.001
        cov_6x6[0, 0] = 0.05   # distinctive value in position block
        cov_6x6[3, 3] = 0.99   # distinctive value in rotation block
        cov_36 = cov_6x6.flatten()

        extracted = extract_vio_position_cov(cov_36)
        assert abs(extracted[0, 0] - 0.05) < 1e-12, (
            f"Position block [0,0] should be 0.05, got {extracted[0,0]:.6f}. "
            f"Block indices may be wrong."
        )
        assert abs(extracted[0, 0] - 0.99) > 0.5, (
            "Position block extracted rotation block diagonal — wrong indices."
        )

    def test_zero_covariance_guard_raises(self):
        """
        Input with diagonal element = 0 must raise VIOCovarianceError.
        Confirms IFM-04 detection in the extraction path.
        """
        cov_6x6 = np.eye(6) * 0.01
        cov_6x6[1, 1] = 0.0   # zero diagonal at position axis 1 (y/North)
        cov_36 = cov_6x6.flatten()

        with pytest.raises(VIOCovarianceError) as exc_info:
            extract_vio_position_cov(cov_36)

        assert "IFM-04" in str(exc_info.value), (
            f"VIOCovarianceError should mention IFM-04, got: {exc_info.value}"
        )

    def test_negative_diagonal_raises(self):
        """Negative diagonal must also raise VIOCovarianceError."""
        cov_6x6 = np.eye(6) * 0.01
        cov_6x6[2, 2] = -0.001
        cov_36 = cov_6x6.flatten()

        with pytest.raises(VIOCovarianceError):
            extract_vio_position_cov(cov_36)

    def test_wrong_length_raises_value_error(self):
        """Input with != 36 elements must raise ValueError."""
        with pytest.raises(ValueError):
            extract_vio_position_cov(np.ones(35))

    def test_rotation_does_not_inflate_trace_beyond_bound(self):
        """
        After ENU→NED rotation, trace(R) must not exceed Stage-2 bound.
        R_ENU_TO_NED is orthogonal — rotation preserves trace for diagonal
        input (trace is invariant under similarity transform for orthogonal R).
        """
        sigma = 0.15   # mid-range Stage-2 value
        cov_enu = np.diag([sigma**2, sigma**2, sigma**2])
        cov_ned = rotate_cov_enu_to_ned(cov_enu)

        trace_enu = np.trace(cov_enu)
        trace_ned = np.trace(cov_ned)

        np.testing.assert_allclose(
            trace_ned, trace_enu, rtol=1e-10,
            err_msg="trace(R) must be invariant under orthogonal rotation for isotropic cov",
        )
        assert trace_ned <= self._TRACE_MAX, (
            f"trace(R_ned) = {trace_ned:.4f} exceeds bound {self._TRACE_MAX}"
        )


# ===========================================================================
# Bonus: FusionLogger smoke test (not a hard gate — confirms JSON output)
# ===========================================================================

class TestFusionLoggerSmoke:
    """
    Smoke test: FusionLogger writes valid JSON with all 8 O-signals present.
    Not a hard gate for 04-A, but confirms the logging wrapper is wired correctly.
    """

    def test_close_writes_valid_json_with_all_signals(self, tmp_path):
        """
        Smoke test updated for schema 08.1 (S-NEP-08).
        Original O-01..O-08 signals replaced by schema 08.1 fields.
        """
        out_path = tmp_path / "fusion_test_log.json"
        logger = FusionLogger(log_path=out_path, label="smoke_01")

        # Log propagation entries
        for i in range(5):
            logger.log_propagate(
                t=float(i) * 0.1, trace_P=0.05,
                vio_mode="NOMINAL", dt_since_vio=0.0
            )

        # Log VIO update entries
        for i in range(3):
            logger.log_vio_update(
                t=float(i) * 0.2, nis=0.8 + i * 0.1,
                innov_mag=0.08 + i * 0.01, trace_P=0.04,
                vio_mode="NOMINAL", dt_since_vio=0.0,
                drift_envelope_m=None, innovation_spike_alert=False,
                error_m=0.09, ba_est=[0.0, 0.0, 0.0]
            )

        # Log one rejection
        logger.log_rejection(t=0.5, nis=12.5, innov_mag=2.1)

        logger.close()

        assert out_path.exists(), "FusionLogger.close() did not write output file"

        with open(out_path) as f:
            doc = json.load(f)

        assert doc["summary"]["schema"] == "08.1"
        assert doc["summary"]["n_vio_updates"] == 3
        assert doc["summary"]["n_rejections"] == 1
        assert doc["summary"]["n_propagations"] == 5
        assert "vel_err_note" in doc["summary"]
        assert "drift_envelope_note" in doc["summary"]

        # Verify entry types are present
        types = {e["type"] for e in doc["time_series"]}
        assert "PROPAGATE" in types
        assert "VIO_UPDATE" in types
        assert "REJECTION" in types

    def test_invalid_update_type_raises(self):
        """Schema 08.1: unsupported log call raises AttributeError (no log_update method)."""
        logger = FusionLogger()
        with pytest.raises(AttributeError):
            logger.log_update(0.0, "GNSS_UPDATE")  # old API — must not exist
