# core/fusion/frame_utils.py
# MicroMind — S-NEP-04 Step 04-A
#
# ENU → NED frame rotation and ROS2 covariance extraction.
#
# Interface contract (non-negotiable):
#   R_ENU_TO_NED = [[0,1,0],[1,0,0],[0,0,-1]]
#   Covariance rotation: R_ned = R @ C_enu @ R.T
#   Zero-covariance guard: any diagonal element ≤ 0 → VIOCovarianceError

import numpy as np
from core.fusion.vio_covariance_error import VIOCovarianceError

# ---------------------------------------------------------------------------
# ENU → NED rotation matrix (constant, frozen for the programme)
# ---------------------------------------------------------------------------
# ENU axes: x=East, y=North, z=Up
# NED axes: x=North, y=East, z=Down
#
# Mapping:  North = ENU_y  → NED_x = ENU_y  → row 0 = [0, 1, 0]
#           East  = ENU_x  → NED_y = ENU_x  → row 1 = [1, 0, 0]
#           Down  = -ENU_z → NED_z = -ENU_z → row 2 = [0, 0, -1]
R_ENU_TO_NED: np.ndarray = np.array(
    [[0, 1, 0],
     [1, 0, 0],
     [0, 0, -1]],
    dtype=np.float64,
)


def rotate_pos_enu_to_ned(pos_enu: np.ndarray) -> np.ndarray:
    """
    Rotate a 3-vector position from ENU frame to NED frame.

    Parameters
    ----------
    pos_enu : (3,) array-like — position in ENU frame [East, North, Up] (m)

    Returns
    -------
    pos_ned : (3,) np.ndarray — position in NED frame [North, East, Down] (m)
    """
    pos_enu = np.asarray(pos_enu, dtype=np.float64).reshape(3)
    return R_ENU_TO_NED @ pos_enu


def rotate_cov_enu_to_ned(cov_enu: np.ndarray) -> np.ndarray:
    """
    Rotate a 3×3 position covariance matrix from ENU frame to NED frame.

    The similarity transform R @ C @ R.T preserves the Gaussian structure
    of the measurement noise distribution.

    Parameters
    ----------
    cov_enu : (3, 3) array-like — position covariance in ENU frame (m²)

    Returns
    -------
    cov_ned : (3, 3) np.ndarray — position covariance in NED frame (m²)
    """
    cov_enu = np.asarray(cov_enu, dtype=np.float64).reshape(3, 3)
    return R_ENU_TO_NED @ cov_enu @ R_ENU_TO_NED.T


def extract_vio_position_cov(ros_covariance_36) -> np.ndarray:
    """
    Extract the 3×3 position covariance block from a ROS2 flat covariance array.

    ROS2 geometry_msgs/PoseWithCovarianceStamped stores pose.covariance as a
    row-major 6×6 matrix with state order [x, y, z, roll, pitch, yaw].
    The position block occupies rows 0–2, columns 0–2 → indices [0:3, 0:3].

    Parameters
    ----------
    ros_covariance_36 : array-like of length 36 — flat row-major 6×6 covariance

    Returns
    -------
    cov_pos : (3, 3) np.ndarray — position covariance block (m²)

    Raises
    ------
    VIOCovarianceError
        If any diagonal element of the extracted 3×3 block is ≤ 0.
        A zero or negative diagonal means the filter was not properly
        initialised or the VIO system is reporting a degenerate state.
        Corresponds to IFM-04.
    """
    cov_flat = np.asarray(ros_covariance_36, dtype=np.float64)
    if cov_flat.shape != (36,):
        raise ValueError(
            f"ros_covariance_36 must have 36 elements, got {cov_flat.shape}"
        )

    cov_6x6 = cov_flat.reshape(6, 6)
    cov_pos = cov_6x6[0:3, 0:3].copy()

    # Zero-covariance guard — IFM-04
    diag = np.diag(cov_pos)
    if np.any(diag <= 0.0):
        bad = np.where(diag <= 0.0)[0].tolist()
        raise VIOCovarianceError(
            f"VIO position covariance has degenerate diagonal element(s) at "
            f"index {bad}: diag = {diag.tolist()}. IFM-04 triggered."
        )

    return cov_pos
