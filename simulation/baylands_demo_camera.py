#!/usr/bin/env python3.12
"""
simulation/baylands_demo_camera.py
Sets the Gazebo GUI camera to a fixed top-down isometric position covering
the full Baylands terrain (899m x 587m).

  Camera position: X=450, Y=295, Z=800 (above terrain centre)
  Pitch: -89 degrees (near top-down)
  FOV: sufficient to cover full terrain from Z=800

Called once after GAZEBO_READY is confirmed in run_demo.sh.
Non-fatal: if the Gazebo GUI is not running, prints a warning and exits 0.

Service pattern mirrors launch_two_vehicle_sitl.sh (OI-20 verified pattern).

Req: VIZ-02  SRS: §9.3
"""

import json
import subprocess
import sys

# ---------------------------------------------------------------------------
# Camera pose — top-down over Baylands terrain centre
#
# Baylands terrain: 899m × 587m.  Centre ≈ (450, 295) in world ENU.
# Z=800 m gives ~60° half-angle FOV coverage of 1000m+ diagonal at Z=800.
# Orientation: rotate ~89° around X axis → camera looks nearly straight down.
#   Quaternion for -89° pitch (rotation around X axis):
#   angle = -89° = -1.5533 rad  →  half = -0.7767 rad
#   w = cos(-0.7767) ≈ 0.7133,  x = sin(-0.7767) ≈ -0.7009
# ---------------------------------------------------------------------------

_CAMERA_REQ = json.dumps({
    "pose": {
        "position": {"x": 450.0, "y": 295.0, "z": 800.0},
        "orientation": {"x": -0.7009, "y": 0.0, "z": 0.0, "w": 0.7133},
    }
})


def set_camera_pose() -> bool:
    """
    Send camera pose via gz service.
    Returns True on success, False on failure (non-fatal).
    """
    result = subprocess.run(
        [
            "gz", "service",
            "-s", "/gui/move_to_pose",
            "--reqtype", "gz.msgs.GUICamera",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "2000",
            "--req", _CAMERA_REQ,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("[CAMERA] Top-down camera set: X=450 Y=295 Z=800 pitch=-89°")
        return True

    # Fallback: try topic publish (Gazebo Harmonic also accepts topic)
    result2 = subprocess.run(
        [
            "gz", "topic",
            "-t", "/gui/camera/pose",
            "-m", "gz.msgs.Pose",
            "-p", (
                "position:{x:450.0 y:295.0 z:800.0} "
                "orientation:{x:-0.7009 y:0.0 z:0.0 w:0.7133}"
            ),
        ],
        capture_output=True,
        text=True,
    )

    if result2.returncode == 0:
        print("[CAMERA] Top-down camera set via topic: X=450 Y=295 Z=800")
        return True

    # Non-fatal — GUI may not be running (HEADLESS mode)
    print(
        f"[CAMERA] Warning: camera pose not set "
        f"(service: {result.stderr.strip()!r}; "
        f"topic: {result2.stderr.strip()!r}). "
        "Continuing — operators may adjust camera manually."
    )
    return False


if __name__ == "__main__":
    set_camera_pose()
    sys.exit(0)
