# sim/eskf_simulation.py
"""
MicroMind Canonical Error-State EKF Simulation
----------------------------------------------

Purpose:
- Validate INS mechanisation + ESKF bias propagation
- GNSS-denied baseline (no aiding)
- Deterministic, autonomy-first sandbox

This file is the SINGLE source of truth for autonomy simulation.
"""

import numpy as np

from core.ins.state import INSState
from core.ins.mechanisation import ins_propagate
from core.ekf.error_state_ekf import ErrorStateEKF
from core.constants import GRAVITY


# ===============================
# Simulation configuration
# ===============================

def default_sim_config():
    return {
        "dt": 0.01,            # [s]
        "T": 1800.0,           # [s] → 30 min GNSS-denied
        "acc_bias": np.array([0.02, 0.01, 0.03]),     # m/s^2
        "gyro_bias": np.array([0.002, 0.001, 0.001]), # rad/s
        "acc_noise": 0.02,     # m/s^2 RMS
        "gyro_noise": 0.001,   # rad/s RMS
        "seed": 42
    }


# ===============================
# Main simulation entrypoint
# ===============================

def run_simulation(config=None):
    """
    Runs a deterministic INS + Error-State EKF propagation.

    Returns:
        t       : (N,) time vector [s]
        pos     : (N,3) position history [m]
        att_err : (N,) attitude error magnitude [rad]
    """

    if config is None:
        config = default_sim_config()

    np.random.seed(config["seed"])

    dt = config["dt"]
    N  = int(config["T"] / dt)

    # -------------------------------
    # Initial INS state
    # -------------------------------
    state = INSState(
        p=np.zeros(3),
        v=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3)
    )

    # -------------------------------
    # Error-State EKF
    # -------------------------------
    ekf = ErrorStateEKF()

    # -------------------------------
    # Logging buffers
    # -------------------------------
    t_hist   = np.zeros(N)
    pos_hist = np.zeros((N, 3))
    att_err  = np.zeros(N)

    # -------------------------------
    # Simulation loop
    # -------------------------------
    for k in range(N):
        t = k * dt
        t_hist[k] = t

        # Simulated IMU measurements
        acc_meas = (
            np.array([0.0, 0.0, 0.0])
            + config["acc_bias"]
            + np.random.randn(3) * config["acc_noise"]
        )

        gyro_meas = (
            np.array([0.0, 0.0, 0.0])
            + config["gyro_bias"]
            + np.random.randn(3) * config["gyro_noise"]
        )

        # --- ESKF propagation (bias only, no updates)
        ekf.propagate(dt)

        b_a_est = ekf.x[9:12]
        b_g_est = ekf.x[12:15]

        # --- INS propagation
        # Bias injection already done via ESKF
        state = ins_propagate(
            state,
            acc_meas,
            gyro_meas,
            dt
        )

    
        # --- Log outputs
        pos_hist[k] = state.p
        att_err[k]  = 2.0 * np.arccos(np.clip(state.q[0], -1.0, 1.0))

    return t_hist, pos_hist, att_err
