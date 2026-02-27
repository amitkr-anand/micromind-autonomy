"""
core/ins/mechanisation.py — INS Strapdown Mechanisation
Sprint S0: original ins_propagate (ENU frame, quaternion attitude)
Sprint S8-B: optional imu_model noise injection (backward-compatible)

Coordinate frame: ENU (East-North-Up)
Attitude representation: unit quaternion q = [w, x, y, z]

S8-B extension rules:
  - imu_model=None  → zero regression on all 147 existing tests
  - imu_model given → caller pre-generates IMUNoiseOutput and passes it;
                       noise arrays are indexed per step (no generation inside loop)
  - Scale factor: applied as true_rate * (1 + sf_ppm * 1e-6) BEFORE noise addition
  - Temperature bias: static offset per mission, already encoded in IMUNoiseOutput
  - VRE: nonzero constant offset on accel channel for sensors that have it
"""

from __future__ import annotations

import numpy as np
from typing import Optional, TYPE_CHECKING

from core.ins.state import INSState
from core.math.quaternion import quat_multiply, quat_rotate, quat_from_gyro, quat_normalize
from core.constants import GRAVITY

if TYPE_CHECKING:
    # Avoid circular import at runtime; imu_model imported only for type hints
    from core.ins.imu_model import IMUModel, IMUNoiseOutput


def ins_propagate(
    state: INSState,
    accel_b: np.ndarray,
    gyro_b: np.ndarray,
    dt: float,
    *,
    imu_model: Optional["IMUModel"] = None,
    imu_noise: Optional["IMUNoiseOutput"] = None,
    step: int = 0,
) -> INSState:
    """
    Propagate INS state by one timestep dt.

    Parameters
    ----------
    state   : INSState — current navigation state
    accel_b : (3,) ndarray — true specific force in body frame [m/s²]
    gyro_b  : (3,) ndarray — true angular rate in body frame [rad/s]
    dt      : float — timestep [s]

    Keyword-only (S8-B extension):
    imu_model : IMUModel | None
        When provided, `imu_noise` must also be provided.  The model is used
        only to retrieve scale-factor and VRE metadata; the actual noise arrays
        come from `imu_noise` (pre-generated outside the propagation loop).
    imu_noise : IMUNoiseOutput | None
        Pre-generated noise output from imu_model.generate_imu_noise().
        Step index is taken from `step`.
    step : int
        Current propagation step index into imu_noise arrays.

    Returns
    -------
    INSState — updated navigation state

    Notes
    -----
    When imu_model is None (default) the function is identical to the S0
    implementation — no branching, no overhead, zero regression risk.

    When imu_model + imu_noise are provided:
      1. Scale factor applied: gyro_eff = gyro_b * (1 + sf_error[step])
      2. Total gyro noise (white noise + Gauss-Markov bias + temp bias)
         indexed from imu_noise.total_gyro()[step]
      3. Total accel noise (white noise + VRE + temp bias) indexed from
         imu_noise.total_accel()[step]
      The noised measurements are then fed into the standard mechanisation.
    """
    # ------------------------------------------------------------------ #
    # S8-B: Apply sensor noise when model is provided                      #
    # ------------------------------------------------------------------ #
    if imu_model is not None and imu_noise is not None:
        # --- gyro: scale factor first, then additive noise ---
        sf = imu_noise.sf_error_gyro[step] if hasattr(imu_noise, "sf_error_gyro") else 0.0
        gyro_effective = gyro_b * (1.0 + sf) + imu_noise.total_gyro()[step]

        # --- accel: additive noise (VRE already embedded in total_accel) ---
        accel_effective = accel_b + imu_noise.total_accel()[step]
    else:
        # S0 baseline — no noise, no modification
        gyro_effective = gyro_b
        accel_effective = accel_b

    # ------------------------------------------------------------------ #
    # Standard strapdown mechanisation (unchanged from S0)                 #
    # ------------------------------------------------------------------ #

    # 1. Attitude update via quaternion integration
    q = np.array(state.q, dtype=float)
    dq = quat_from_gyro(gyro_effective, dt)
    q_new = quat_normalize(quat_multiply(q, dq))

    # 2. Transform specific force from body to navigation (ENU) frame
    #    using the mean attitude (trapezoidal — simple mid-point here)
    f_nav = quat_rotate(q_new, accel_effective)

    # 3. Remove gravity (ENU: gravity acts in -Up direction = -Z)
    # GRAVITY = [0, 0, -9.80665] already in ENU convention (constants.py)
    accel_nav = f_nav + GRAVITY

    # 4. Velocity update (Euler integration)
    vel = np.array(state.v, dtype=float)
    vel_new = vel + accel_nav * dt

    # 5. Position update (trapezoid)
    pos = np.array(state.p, dtype=float)
    pos_new = pos + 0.5 * (vel + vel_new) * dt

    # 6. Build and return new state
    return INSState(
        p=pos_new,
        v=vel_new,
        q=q_new,
        ba=state.ba.copy(),
        bg=state.bg.copy(),
    )
