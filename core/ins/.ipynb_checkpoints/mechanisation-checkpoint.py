import numpy as np

from core.math.quaternion import (
    quat_from_gyro,
    quat_multiply,
    quat_normalize,
    quat_rotate
)

from core.constants import GRAVITY

def ins_propagate(state, acc_meas, gyro_meas, dt):
    acc  = acc_meas  - state.ba
    gyro = gyro_meas - state.bg

    dq = quat_from_gyro(gyro, dt)
    q_new = quat_normalize(quat_multiply(state.q, dq))

    acc_world = quat_rotate(q_new, acc) + GRAVITY

    state.v = state.v + acc_world * dt
    state.p = state.p + state.v * dt + 0.5 * acc_world * dt**2
    state.q = q_new

    return state

"""

def ins_propagate(state, acc_meas, gyro_meas, dt):
    
   # Nominal INS mechanisation using biases stored in state
    
    acc  = acc_meas  - state.ba
    gyro = gyro_meas - state.bg

    dq = quat_from_gyro(gyro, dt)
    q_new = quat_normalize(quat_multiply(state.q, dq))

    acc_world = quat_rotate(q_new, acc) + GRAVITY

    v_new = state.v + acc_world * dt
    p_new = state.p + state.v * dt + 0.5 * acc_world * dt**2

    state.q = q_new
    state.v = v_new
    state.p = p_new

    return state
"""