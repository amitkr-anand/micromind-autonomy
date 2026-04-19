"""
LightGlue IPC bridge configuration.

Server runs in hil-h3 (Python 3.10) environment.
Client runs in micromind-autonomy (Python 3.11) environment.
Both run on the same host (Jetson Orin or dev machine).
"""
import os
from pathlib import Path

# Unix domain socket path — same machine, both sides
SOCKET_PATH = os.environ.get(
    "LIGHTGLUE_SOCKET",
    "/tmp/micromind_lightglue.sock",
)

# Root of terrain tile data (absolute, resolved from repo root)
_REPO_ROOT = Path(__file__).parents[2]
TILE_DIR = os.environ.get(
    "LIGHTGLUE_TILE_DIR",
    str(_REPO_ROOT / "data" / "terrain"),
)

# Path to the hil-h3 Python 3.10 interpreter that has LightGlue installed.
# On Orin: /home/mmuser/miniforge3/envs/hil-h3/bin/python3
# Override via env var for custom installs.
LIGHTGLUE_PYTHON = os.environ.get(
    "LIGHTGLUE_PYTHON",
    "/home/mmuser/miniforge3/envs/hil-h3/bin/python3",
)

# How long the client waits for the server to become ready after spawning it
SERVER_START_TIMEOUT_S = float(os.environ.get("LIGHTGLUE_START_TIMEOUT", "15.0"))

# Per-request timeout on the client side (seconds)
REQUEST_TIMEOUT_S = float(os.environ.get("LIGHTGLUE_REQUEST_TIMEOUT", "10.0"))

# LightGlue model parameters
MAX_FEATURES = int(os.environ.get("LIGHTGLUE_MAX_FEATURES", "2048"))
MATCH_THRESHOLD = float(os.environ.get("LIGHTGLUE_MATCH_THRESHOLD", "0.2"))

# Minimum inlier count required to accept a match result
MIN_INLIERS = int(os.environ.get("LIGHTGLUE_MIN_INLIERS", "20"))

# Tile patch extraction radius in metres around the GPS prior
TILE_PATCH_RADIUS_M = float(os.environ.get("LIGHTGLUE_PATCH_RADIUS", "500.0"))

# Satellite tile GSD in metres/pixel (GLO-30 COP30 nominal)
TILE_GSD_M = float(os.environ.get("LIGHTGLUE_TILE_GSD", "30.0"))
