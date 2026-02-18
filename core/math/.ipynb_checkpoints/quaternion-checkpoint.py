import numpy as np

def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

"""def quat_from_gyro(omega, dt):
    angle = np.linalg.norm(omega) * dt
    if angle < 1e-8:
        return np.array([1, 0, 0, 0])
    axis = omega / np.linalg.norm(omega)
    half = angle / 2
    return np.hstack([np.cos(half), axis*np.sin(half)])
"""

def quat_from_gyro(omega, dt):
    """
    Small-angle quaternion from angular rate
    omega: rad/s (3,)
    """
    theta = np.linalg.norm(omega) * dt
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = omega / np.linalg.norm(omega)
    half = 0.5 * theta

    return np.array([
        np.cos(half),
        *(axis * np.sin(half))
    ])


def quat_rotate(q, v):
    qv = np.hstack([0, v])
    return quat_multiply(
        quat_multiply(q, qv),
        np.array([q[0], -q[1], -q[2], -q[3]])
    )[1:]


__all__ = [
    "quat_from_gyro",
    "quat_multiply",
    "quat_normalize",
    "quat_rotate"
]
