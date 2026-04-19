"""
HIL H-4: LightGlue IPC bridge tests.

T1 — Server starts and responds to ping
T2 — match() returns valid result on Site 04 frame (shimla km=020, lat=31.157°N)
T3 — match() returns None on invalid coordinates (lat=999)

Tests run against the server in stub mode on the dev machine (hil-h3
interpreter absent).  The same tests pass on Orin with real LightGlue.
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
import socket
import json
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Resolve paths relative to repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parents[1]
SERVER_SCRIPT = REPO_ROOT / "integration" / "lightglue_bridge" / "server.py"
SITE04_FRAME = (
    REPO_ROOT / "data" / "synthetic_imagery" / "shimla_corridor_300m" / "frame_km020.png"
)

# Site 04 GPS prior: shimla corridor km=020 (interpolated)
SITE04_LAT = 31.157
SITE04_LON = 77.256
SITE04_ALT = 300.0
SITE04_HEADING = 45.0

# Socket path used during tests (override to avoid collisions)
TEST_SOCKET = "/tmp/micromind_lightglue_test.sock"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def server_process():
    """Start the LightGlue server subprocess for the test session."""
    env = os.environ.copy()
    env["LIGHTGLUE_SOCKET"] = TEST_SOCKET

    proc = subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait up to 10 s for the socket to appear and become responsive
    deadline = time.monotonic() + 10.0
    ready = False
    while time.monotonic() < deadline:
        if os.path.exists(TEST_SOCKET):
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                sock.connect(TEST_SOCKET)
                sock.sendall(b'{"cmd": "ping"}\n')
                data = sock.recv(4096)
                sock.close()
                if b"pong" in data:
                    ready = True
                    break
            except OSError:
                pass
        time.sleep(0.1)

    if not ready:
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=5)
        pytest.fail(
            f"Server did not start within 10 s.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    try:
        os.unlink(TEST_SOCKET)
    except FileNotFoundError:
        pass


def _call(request: dict, timeout: float = 8.0) -> dict:
    """Send one request to the test server and return the parsed response."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(TEST_SOCKET)
        sock.sendall(json.dumps(request).encode() + b"\n")
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        return json.loads(data.split(b"\n", 1)[0])
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# T1 — Server starts and responds to ping
# ---------------------------------------------------------------------------

def test_T1_server_ping(server_process):
    """T1: server must start successfully and respond to a ping command."""
    resp = _call({"cmd": "ping"})
    assert resp["status"] == "pong", f"Expected pong, got: {resp}"
    assert "version" in resp
    assert "python" in resp
    assert "lightglue_available" in resp
    print(f"\n  T1 server info: version={resp['version']}, "
          f"lightglue_available={resp['lightglue_available']}, "
          f"python={resp['python'][:20]}")


# ---------------------------------------------------------------------------
# T2 — Valid match on Site 04 frame
# ---------------------------------------------------------------------------

def test_T2_valid_match_site04(server_process):
    """T2: match on shimla km=020 frame must return ok with valid fields."""
    assert SITE04_FRAME.exists(), f"Site 04 frame not found: {SITE04_FRAME}"

    t0 = time.monotonic()
    resp = _call({
        "cmd": "match",
        "frame_path": str(SITE04_FRAME),
        "lat": SITE04_LAT,
        "lon": SITE04_LON,
        "alt": SITE04_ALT,
        "heading_deg": SITE04_HEADING,
    }, timeout=15.0)
    round_trip_ms = (time.monotonic() - t0) * 1000.0

    assert resp["status"] == "ok", (
        f"Expected ok, got {resp.get('status')!r} — reason: {resp.get('reason') or resp.get('message')}"
    )

    delta_lat = resp["delta_lat"]
    delta_lon = resp["delta_lon"]
    confidence = resp["confidence"]
    match_ms = resp["match_ms"]

    # Sanity bounds: corrections must be sub-metre in degrees
    assert abs(delta_lat) < 0.01, f"delta_lat out of bounds: {delta_lat}"
    assert abs(delta_lon) < 0.01, f"delta_lon out of bounds: {delta_lon}"
    assert 0.0 <= confidence <= 1.0, f"confidence out of range: {confidence}"
    assert match_ms > 0, f"match_ms must be positive: {match_ms}"

    print(f"\n  T2 Site 04 result: dlat={delta_lat:.6f}° dlon={delta_lon:.6f}° "
          f"conf={confidence:.3f} match_ms={match_ms:.1f} "
          f"round_trip_ms={round_trip_ms:.1f}")


# ---------------------------------------------------------------------------
# T3 — Invalid coordinates return None (no_match)
# ---------------------------------------------------------------------------

def test_T3_invalid_coords_returns_no_match(server_process):
    """T3: coordinates outside [-90,90]/[-180,180] must return no_match."""
    resp = _call({
        "cmd": "match",
        "frame_path": str(SITE04_FRAME),
        "lat": 999.0,    # invalid
        "lon": 77.256,
        "alt": 300.0,
        "heading_deg": 0.0,
    })
    assert resp["status"] == "no_match", (
        f"Expected no_match for invalid coordinates, got {resp.get('status')!r}"
    )
    assert resp.get("reason") == "invalid_coordinates", (
        f"Expected reason=invalid_coordinates, got {resp.get('reason')!r}"
    )
    print(f"\n  T3 invalid coords: status={resp['status']}, reason={resp['reason']}")


# ---------------------------------------------------------------------------
# Latency benchmark (informational, not a gate)
# ---------------------------------------------------------------------------

def test_latency_benchmark_5frames(server_process):
    """Informational: measure round-trip latency over 5 consecutive requests."""
    assert SITE04_FRAME.exists(), "Site 04 frame not found"

    results = []
    for i in range(5):
        t0 = time.monotonic()
        resp = _call({
            "cmd": "match",
            "frame_path": str(SITE04_FRAME),
            "lat": SITE04_LAT + i * 1e-5,
            "lon": SITE04_LON,
            "alt": SITE04_ALT,
            "heading_deg": SITE04_HEADING,
        }, timeout=15.0)
        total_ms = (time.monotonic() - t0) * 1000.0
        match_ms = resp.get("match_ms", 0.0) if resp.get("status") == "ok" else 0.0
        ipc_ms = total_ms - match_ms
        results.append((match_ms, ipc_ms, total_ms))

    mean_total = sum(r[2] for r in results) / len(results)
    mean_ipc = sum(r[1] for r in results) / len(results)

    print("\n  Latency benchmark (stub mode — IPC overhead only):")
    print(f"  {'Frame':<6} {'match_ms':>10} {'ipc_ms':>10} {'total_ms':>10}")
    for idx, (m, ipc, total) in enumerate(results):
        print(f"  {idx:<6} {m:>10.1f} {ipc:>10.2f} {total:>10.1f}")
    print(f"  Mean total: {mean_total:.1f} ms  IPC overhead: {mean_ipc:.2f} ms")

    # IPC overhead must be below 50 ms (Unix socket should be < 5 ms)
    assert mean_ipc < 50.0, f"IPC overhead too high: {mean_ipc:.1f} ms"
