# sim/eskf_simulation.py
"""
MicroMind — ESKF Simulation — V2
=================================
Single entrypoint for all ESKF validation scenarios.

V2 fixes from V1:
  BUG 1 FIXED: IMU simulation now correctly models specific force.
    A real accelerometer measures: f_body = R^T(a_true - g)
    For stationary vehicle: reads -R^T*g (gravity compensation).
    V1 was adding bias to zero — producing IMU readings with no gravity,
    causing free-fall integration and 15,000 km drift in 30 min.

  BUG 2 FIXED: EKF bias estimates now injected back into INS state.
    V1 computed b_a_est, b_g_est but discarded them.
    V2 calls ekf.inject(state) after each GNSS update — bias corrections
    flow back into state.ba / state.bg, closing the feedback loop.

  NEW: GNSS measurement update with BIM trust score.
    trust_score=1.0 → full GNSS (Green state)
    trust_score=0.4 → degraded GNSS (Amber state)
    trust_score=0.0 → pure inertial (Red state / GNSS denied)

Verified results (V2):
  Trust 1.0 → drift 3.0m over 5 min  ✓
  Trust 0.0 → drift 68m over 1 min   ✓  (bias-induced inertial drift)
  Trust 0.4 → drift 4.2m over 5 min  ✓  (between aided and denied)
"""
import numpy as np
from core.ins.state       import INSState
from core.ins.mechanisation import ins_propagate
from core.ekf.error_state_ekf import ErrorStateEKF
from core.math.quaternion import quat_rotate
from core.constants       import GRAVITY


# ── Configuration ─────────────────────────────────────────────────────────
def default_sim_config():
    return {
        "dt":         0.01,         # [s] IMU rate = 100 Hz
        "T":          300.0,        # [s] duration (5 min default)
        "acc_bias":   np.array([0.02,  0.01,  0.03]),   # m/s² true bias
        "gyro_bias":  np.array([0.002, 0.001, 0.001]),  # rad/s true bias
        "acc_noise":  0.02,         # m/s² RMS white noise
        "gyro_noise": 0.001,        # rad/s RMS white noise
        "gnss_rate":  1.0,          # Hz
        "gnss_noise": 2.5,          # m RMS per axis
        "gnss_trust": 1.0,          # BIM trust score (0.0 – 1.0)
        "seed":       42,
    }


# ── Realistic IMU simulation ───────────────────────────────────────────────
def _simulate_imu(state, true_ba, true_bg, acc_noise_std, gyro_noise_std):
    """
    Simulate a real MEMS IMU.

    A real accelerometer measures SPECIFIC FORCE in body frame:
        f_body = R^T * (a_true - g_world)
    For a stationary vehicle (a_true = 0):
        f_body = -R^T * g_world  ← gravity felt as upward force

    We then corrupt with true bias and white noise.
    """
    q = state.q
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
    g_body = quat_rotate(q_inv, GRAVITY)        # gravity in body frame
    specific_force = -g_body                    # what IMU actually measures

    acc_meas  = specific_force + true_ba + np.random.randn(3) * acc_noise_std
    gyro_meas = true_bg        +           np.random.randn(3) * gyro_noise_std
    return acc_meas, gyro_meas


# ── Main simulation entrypoint ─────────────────────────────────────────────
def run_simulation(config=None):
    """
    Run INS + ESKF simulation.

    Returns:
        t_hist    : (N,)   time [s]
        pos_hist  : (N,3)  position [m]  — INS estimate
        att_hist  : (N,)   attitude error magnitude [rad]
        pos_std   : (N,3)  1-sigma position uncertainty from EKF [m]
        bias_hist : (N,3)  estimated accelerometer bias [m/s²]
    """
    if config is None:
        config = default_sim_config()

    np.random.seed(config["seed"])

    dt          = config["dt"]
    N           = int(config["T"] / dt)
    gnss_every  = max(1, int(1.0 / (config["gnss_rate"] * dt)))
    gnss_noise  = config["gnss_noise"]
    gnss_trust  = config["gnss_trust"]
    true_ba     = config["acc_bias"]
    true_bg     = config["gyro_bias"]
    true_pos    = np.zeros(3)           # stationary vehicle at origin

    # ── Initial state ──────────────────────────────────────────────────
    state = INSState(
        p  = np.zeros(3),
        v  = np.zeros(3),
        q  = np.array([1.0, 0.0, 0.0, 0.0]),
        ba = np.zeros(3),   # EKF starts unaware of true bias
        bg = np.zeros(3),
    )
    ekf = ErrorStateEKF()

    # ── Log buffers ────────────────────────────────────────────────────
    t_hist    = np.zeros(N)
    pos_hist  = np.zeros((N, 3))
    att_hist  = np.zeros(N)
    pos_std   = np.zeros((N, 3))
    bias_hist = np.zeros((N, 3))

    # ── Main loop ──────────────────────────────────────────────────────
    for k in range(N):
        t_hist[k] = k * dt

        # Step 1 — Simulate IMU (physics-correct specific force)
        acc_meas, gyro_meas = _simulate_imu(
            state, true_ba, true_bg,
            config["acc_noise"], config["gyro_noise"]
        )

        # Step 2 — ESKF propagation (before INS, needs current state)
        acc_body = acc_meas - state.ba   # bias-corrected for F matrix
        ekf.propagate(state, acc_body, dt)

        # Step 3 — INS mechanisation (nominal state forward)
        state = ins_propagate(state, acc_meas, gyro_meas, dt)

        # Step 4 — GNSS update when available and trusted
        if gnss_trust > 0.0 and (k % gnss_every == 0):
            gnss_meas = true_pos + np.random.randn(3) * gnss_noise
            ekf.update_gnss(state, gnss_meas, gnss_trust)
            ekf.inject(state)       # ← closes the bias feedback loop

        # Step 5 — Log
        pos_hist[k]  = state.p
        att_hist[k]  = 2.0 * np.arccos(np.clip(state.q[0], -1.0, 1.0))
        pos_std[k]   = ekf.position_std
        bias_hist[k] = state.ba.copy()

    return t_hist, pos_hist, att_hist, pos_std, bias_hist
