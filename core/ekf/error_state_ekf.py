# core/ekf/error_state_ekf.py
"""
MicroMind — Error-State Extended Kalman Filter (ESKF)
======================================================
15-state formulation:
    [0:3]   δp   — position error (m)
    [3:6]   δv   — velocity error (m/s)
    [6:9]   δθ   — attitude error (rad, small-angle)
    [9:12]  δba  — accelerometer bias error (m/s²)
    [12:15] δbg  — gyroscope bias error (rad/s)

V2 changes from V1:
  - Full F matrix with velocity-attitude and bias coupling terms
  - Structured Q matrix (per-state noise, not scalar * I)
  - GNSS position measurement update (update_gnss)
  - BIM trust score hook: R_eff = R_nominal / trust_score
  - Full inject(): corrects p, v, q (attitude), ba, bg
  - position_std property for dashboard display

References:
    Sola (2017) — Quaternion kinematics for ESKF
    Titterton & Weston — Strapdown Inertial Navigation Technology
"""
import numpy as np


def skew(v):
    """3×3 skew-symmetric (cross-product) matrix."""
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])


class ErrorStateEKF:
    """
    Full 15-state Error-State EKF with BIM trust score integration.

    Typical usage per timestep:
        acc_body = acc_meas - state.ba          # bias-corrected
        ekf.propagate(state, acc_body, dt)      # covariance propagation
        state = ins_propagate(...)              # nominal state forward
        if gnss_available:
            ekf.update_gnss(state, gnss_pos, trust_score)
            ekf.inject(state)                   # feed corrections back
    """

    # ── Noise parameters (tune in SIL Sprint S2) ────────────────────────
    _ACC_BIAS_RW    = 1e-4    # m/s²/√s  — accelerometer bias random walk
    _GYRO_BIAS_RW   = 1e-5    # rad/s/√s — gyroscope bias random walk
    _ACC_NOISE_PSD  = 0.04    # m/s²/√Hz — velocity noise
    _GYRO_NOISE_PSD = 1e-3    # rad/s/√Hz — attitude noise

    # ── GNSS measurement noise (BIM-scaled) ─────────────────────────────
    _GNSS_R_NOMINAL = np.diag([2.5**2, 2.5**2, 5.0**2])   # m²

    def __init__(self):
        self.x = np.zeros(15)
        self.P = np.zeros((15, 15))
        # State-specific initial covariance
        self.P[0:3,   0:3]  = np.eye(3) * 1.0          # position  (m²)
        self.P[3:6,   3:6]  = np.eye(3) * 0.1          # velocity  (m/s)²
        self.P[6:9,   6:9]  = np.eye(3) * (1e-3)**2    # attitude  (rad²)
        self.P[9:12,  9:12] = np.eye(3) * (0.1)**2     # acc bias  (m/s²)²
        self.P[12:15,12:15] = np.eye(3) * (0.01)**2    # gyro bias (rad/s)²

    # ── Internal builders ────────────────────────────────────────────────
    def _build_Q(self, dt):
        Q = np.zeros((15, 15))
        Q[3:6,   3:6]  = np.eye(3) * self._ACC_NOISE_PSD**2  * dt
        Q[6:9,   6:9]  = np.eye(3) * self._GYRO_NOISE_PSD**2 * dt
        Q[9:12,  9:12] = np.eye(3) * self._ACC_BIAS_RW**2    * dt
        Q[12:15,12:15] = np.eye(3) * self._GYRO_BIAS_RW**2   * dt
        return Q

    def _build_F(self, state, acc_body, dt):
        """
        Linearised error-state transition Jacobian.
        acc_body: specific force in body frame (after bias subtraction).
        """
        from core.math.quaternion import quat_rotate
        f_n = quat_rotate(state.q, acc_body)   # specific force in world frame
        F = np.eye(15)
        F[0:3,  3:6]  =  np.eye(3) * dt        # δp ← δv
        F[3:6,  6:9]  = -skew(f_n) * dt        # δv ← δθ (Coriolis-like)
        F[3:6,  9:12] = -np.eye(3) * dt        # δv ← δba
        F[6:9, 12:15] = -np.eye(3) * dt        # δθ ← δbg
        return F

    # ── Public API ───────────────────────────────────────────────────────
    def propagate(self, state, acc_body, dt):
        """
        Propagate error-state covariance for one IMU step.
        Call BEFORE ins_propagate() in the main loop.

        state    : current nominal INSState (read-only here)
        acc_body : acc_meas - state.ba  (bias-corrected, body frame)
        dt       : timestep (s)
        """
        F = self._build_F(state, acc_body, dt)
        Q = self._build_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update_gnss(self, state, gnss_pos, trust_score):
        """
        GNSS position measurement update.

        gnss_pos    : (3,) position measurement in world frame (m)
        trust_score : float [0, 1] from BIM module.
                      1.0 → nominal GNSS noise (R_nominal)
                      0.4 → Amber — R scaled up (less trusted)
                     <0.1 → Red   — update skipped entirely

        This is the BIM integration hook. As BIM trust degrades,
        GNSS measurements are automatically down-weighted in the EKF.
        """
        if trust_score < 0.1:
            return  # GNSS Red — do not corrupt the filter

        z = gnss_pos - state.p          # innovation
        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3)        # position observation

        # BIM-scaled measurement noise
        R_eff = self._GNSS_R_NOMINAL / np.clip(trust_score, 0.1, 1.0)

        S = H @ self.P @ H.T + R_eff
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(15) - K @ H) @ self.P

    def inject(self, state):
        """
        Inject accumulated error corrections into nominal INSState.
        Call after update_gnss(). Resets error state to zero.

        Corrects: position, velocity, attitude (via small-angle quat),
                  accelerometer bias, gyroscope bias.
        """
        state.p  += self.x[0:3]
        state.v  += self.x[3:6]

        # Attitude correction: small-angle quaternion multiplication
        dtheta = self.x[6:9]
        dq = np.array([1.0, *0.5*dtheta])
        dq /= np.linalg.norm(dq)
        from core.math.quaternion import quat_multiply, quat_normalize
        state.q = quat_normalize(quat_multiply(state.q, dq))

        # Bias corrections
        state.ba += self.x[9:12]
        state.bg += self.x[12:15]

        # Reset error state
        self.x[:] = 0.0

    # ── Diagnostics (for dashboard) ──────────────────────────────────────
    @property
    def position_std(self):
        """1-sigma position uncertainty (m) — feed to dashboard."""
        return np.sqrt(np.diag(self.P[0:3, 0:3]))

    @property
    def velocity_std(self):
        return np.sqrt(np.diag(self.P[3:6, 3:6]))

    @property
    def bias_acc_est(self):
        """Current accelerometer bias estimate from error state."""
        return self.x[9:12].copy()

    @property
    def bias_gyro_est(self):
        return self.x[12:15].copy()
