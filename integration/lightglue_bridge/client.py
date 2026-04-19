"""
LightGlue IPC client — runs in micromind-autonomy (Python 3.11) environment.

Public API:
    correction = lightglue_client.match(uav_frame_path, lat, lon, alt, heading_deg)
    # Returns (delta_lat, delta_lon, confidence, latency_ms) or None

The server is started automatically on first use if not already running.
Thread-safe: one outstanding request at a time (single L2 caller thread).
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
from typing import Optional, Tuple

from .config import (
    SOCKET_PATH,
    LIGHTGLUE_PYTHON,
    LIGHTGLUE_LD_LIBRARY_PATH_EXTRA,
    SERVER_START_TIMEOUT_S,
    REQUEST_TIMEOUT_S,
)

log = logging.getLogger(__name__)

# Return type: (delta_lat_deg, delta_lon_deg, confidence_0_to_1, latency_ms)
MatchResult = Tuple[float, float, float, float]

_server_proc: Optional[subprocess.Popen] = None  # owned by this module


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _send_recv(request: dict, timeout_s: float = REQUEST_TIMEOUT_S) -> dict:
    """Open a connection, send one request, return the response."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect(SOCKET_PATH)
        payload = json.dumps(request).encode("utf-8") + b"\n"
        sock.sendall(payload)

        # Receive response (newline-terminated)
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break

        line = data.split(b"\n", 1)[0]
        return json.loads(line.decode("utf-8"))
    finally:
        sock.close()


def _server_alive() -> bool:
    """Return True if the server responds to a ping within 1 second."""
    try:
        resp = _send_recv({"cmd": "ping"}, timeout_s=1.0)
        return resp.get("status") == "pong"
    except (OSError, json.JSONDecodeError):
        return False


def _start_server() -> None:
    """Launch the server subprocess using the hil-h3 Python interpreter."""
    global _server_proc

    # Resolve server.py path relative to this file
    server_script = os.path.join(os.path.dirname(__file__), "server.py")

    # Determine which Python to use.  Prefer configured hil-h3 interpreter;
    # fall back to the current interpreter so tests work on dev machine.
    python_exe = LIGHTGLUE_PYTHON
    if not os.path.isfile(python_exe):
        log.warning(
            "hil-h3 interpreter not found at %s — falling back to %s",
            python_exe,
            sys.executable,
        )
        python_exe = sys.executable

    # Build environment: prepend Jetson-specific CUDA library paths so that
    # libcusparseLt.so.0 (shipped inside the nvidia/cusparselt conda package)
    # is found at import time.  No-op on dev machine where the path is absent.
    env = os.environ.copy()
    if LIGHTGLUE_LD_LIBRARY_PATH_EXTRA:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{LIGHTGLUE_LD_LIBRARY_PATH_EXTRA}:{existing}" if existing
            else LIGHTGLUE_LD_LIBRARY_PATH_EXTRA
        )

    log.info("Starting LightGlue server: %s %s", python_exe, server_script)
    _server_proc = subprocess.Popen(
        [python_exe, server_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    # Wait for the server to become ready
    deadline = time.monotonic() + SERVER_START_TIMEOUT_S
    while time.monotonic() < deadline:
        if _server_alive():
            log.info("LightGlue server ready (pid=%d)", _server_proc.pid)
            return
        time.sleep(0.1)

    _server_proc.terminate()
    _server_proc = None
    raise RuntimeError(
        f"LightGlue server did not become ready within {SERVER_START_TIMEOUT_S}s"
    )


def _ensure_server() -> None:
    """Ensure the server is running; start it if not."""
    if _server_alive():
        return
    log.info("LightGlue server not responding — starting …")
    _start_server()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ping() -> dict:
    """Ping the server (starts it if not running).  Returns server info dict."""
    _ensure_server()
    return _send_recv({"cmd": "ping"})


def match(
    uav_frame_path: str,
    lat: float,
    lon: float,
    alt: float,
    heading_deg: float,
) -> Optional[MatchResult]:
    """
    Request a position correction from the LightGlue server.

    Parameters
    ----------
    uav_frame_path : str
        Absolute path to the UAV nadir frame (PNG/JPEG, 640×640 nominal).
    lat : float
        GPS prior latitude in decimal degrees (WGS-84).
    lon : float
        GPS prior longitude in decimal degrees (WGS-84).
    alt : float
        Altitude above sea level in metres.
    heading_deg : float
        UAV heading in degrees (0 = north, 90 = east).

    Returns
    -------
    (delta_lat, delta_lon, confidence, latency_ms) or None
        delta_lat  : correction to add to lat (degrees north, positive = north)
        delta_lon  : correction to add to lon (degrees east, positive = east)
        confidence : match confidence 0.0–1.0
        latency_ms : total round-trip time including IPC overhead (ms)
    None if no match found or invalid coordinates.
    """
    t0 = time.monotonic()
    try:
        _ensure_server()
    except RuntimeError as exc:
        log.error("Cannot reach LightGlue server: %s", exc)
        return None

    request = {
        "cmd": "match",
        "frame_path": uav_frame_path,
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "heading_deg": heading_deg,
    }

    try:
        response = _send_recv(request, timeout_s=REQUEST_TIMEOUT_S)
    except socket.timeout:
        log.warning("LightGlue request timed out after %.1fs", REQUEST_TIMEOUT_S)
        return None
    except OSError as exc:
        log.warning("LightGlue IPC error: %s", exc)
        return None

    latency_ms = (time.monotonic() - t0) * 1000.0

    status = response.get("status")
    if status == "ok":
        return (
            float(response["delta_lat"]),
            float(response["delta_lon"]),
            float(response["confidence"]),
            latency_ms,
        )
    elif status in ("no_match", "error"):
        log.debug("LightGlue server: %s — %s",
                  status, response.get("reason") or response.get("message"))
        return None
    else:
        log.warning("Unexpected LightGlue response: %s", response)
        return None


def shutdown() -> None:
    """Terminate the server subprocess if this client started it."""
    global _server_proc
    if _server_proc is not None and _server_proc.poll() is None:
        log.info("Terminating LightGlue server pid=%d", _server_proc.pid)
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
    _server_proc = None
