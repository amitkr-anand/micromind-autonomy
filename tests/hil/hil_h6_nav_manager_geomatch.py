"""
HIL H-6 — NavigationManager + LightGlue Geographically Matched Test
=====================================================================
OI-51  |  Gate: H-6  |  Authorised: Deputy 1, 21 Apr 2026

NOTE: HIL-ONLY — requires Orin hardware (mmuser-orin@192.168.1.53), hil-h3
conda environment (Python 3.10, CUDA 12.6), and benchmark assets in
~/hil_benchmark/ (satellite04.tif, site04_frames/). This script is NOT
part of the SIL certified baseline (run_certified_baseline.sh) and must
not be added to it.

Sections
--------
1. Cold-start / server readiness
2. Warm-path match calls with latency measurement
3. NavigationManager NAV_LIGHTGLUE_CORRECTION event (AC-4)
4. Frozen file SHA-256 hashes (AC-5)

Architecture notes
------------------
TILE RESOLVER:
  Site 04 (Jiangsu, China) is not a built-in region in tile_resolver.py.
  Registration is done via LIGHTGLUE_EXTRA_TILES before any client import
  so the spawned server subprocess inherits it. Any resident server that
  was started without the env var will not have site04 registered — this
  script kills any existing server and starts fresh.

ALT PARAMETER:
  client.match() docstring states alt = "Altitude above sea level in
  metres". CSV height field is MSL. Value passed: 545.22 (row 1).
  Finding: server._run_lightglue() does not use alt in match logic
  (only lat/lon for tile selection and fixed TILE_PATCH_RADIUS_M for
  crop). Alt is a forward-reserved parameter in this server version.

HEADING PARAMETER:
  Derived from GPS track vector frames 1–5: delta_north ≈ +438.8 m,
  delta_east ≈ −77.8 m → atan2(−77.8, 438.8) ≈ −10.1° → 350.0°.
  Finding: server._run_lightglue() does not use heading_deg. Also
  a forward-reserved parameter in this server version.

REPORT LABELS:
  Script prints measured values. Pass/fail ruling is Deputy 1's
  responsibility. Labels used: MEASURED / WITHIN EXPECTATIONS /
  FLAG FOR DEPUTY 1.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

# ── Must be set before any client import so spawned server inherits them ──
SITE04_TILE_JSON = json.dumps([{
    "name": "site04",
    "lat_min": 32.15,
    "lat_max": 32.26,
    "lon_min": 119.90,
    "lon_max": 119.96,
    "tile_path": "/home/mmuser-orin/hil_benchmark/satellite04.tif",
}])
os.environ["LIGHTGLUE_EXTRA_TILES"] = SITE04_TILE_JSON
os.environ["LIGHTGLUE_START_TIMEOUT"] = "60"   # H-5 cold-start was 23.6s; 15s default too short

REPO_ROOT = "/home/mmuser-orin/micromind-autonomy"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Imports after env vars and sys.path are set
from unittest.mock import MagicMock

import numpy as np

from integration.lightglue_bridge.config import SOCKET_PATH
import integration.lightglue_bridge.client as _lg_module
from integration.lightglue_bridge.client import ping, match
from core.navigation.navigation_manager import NavigationManager

# ── Test parameters (verbatim from 04.csv row 1) ─────────────────────────
FRAME_PATH  = "/home/mmuser-orin/hil_benchmark/site04_frames/04_0001.JPG"
LAT         = 32.1555603
LON         = 119.9289015
ALT         = 545.22    # MSL; forward-reserved in server (see header)
HEADING     = 350.0     # deg; forward-reserved in server (see header)
CONF_ACCEPT = 0.35      # LIGHTGLUE_CONF_THRESHOLD_ACCEPT (AD-23)

FROZEN_FILES = [
    "core/ekf/error_state_ekf.py",
    "core/fusion/vio_mode.py",
    "core/fusion/frame_utils.py",
    "core/bim/bim.py",
    "scenarios/bcmp1/bcmp1_runner.py",
]

SEP = "=" * 70


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 1 — Cold-start / server readiness
# ─────────────────────────────────────────────────────────────────────────
print(SEP)
print("SECTION 1 — SERVER READINESS")
print(SEP)

print("[S1] Killing any resident LightGlue server (stale tile resolver risk) …")
subprocess.run(["pkill", "-f", "lightglue_bridge/server.py"], capture_output=True)
time.sleep(0.5)

if os.path.exists(SOCKET_PATH):
    os.unlink(SOCKET_PATH)
    print(f"[S1] Removed stale socket: {SOCKET_PATH}")
else:
    print(f"[S1] No stale socket at {SOCKET_PATH}")

print(f"[S1] LIGHTGLUE_EXTRA_TILES = {SITE04_TILE_JSON}")
print(f"[S1] LIGHTGLUE_START_TIMEOUT = {os.environ['LIGHTGLUE_START_TIMEOUT']} s")
print(f"[S1] Frame : {FRAME_PATH}")
print(f"[S1] Coords: lat={LAT}  lon={LON}  alt={ALT} m MSL  heading={HEADING}°")
print("[S1] Pinging server (will spawn fresh) …")

t0_ping = time.perf_counter()
try:
    ping_resp = ping()
except RuntimeError as exc:
    print(f"[S1] SERVER FAILED TO START: {exc}")
    sys.exit(1)
cold_ms = (time.perf_counter() - t0_ping) * 1000.0

print(f"[S1] Server response  : {ping_resp}")
print(f"[S1] Cold-start time  : {cold_ms:.1f} ms")


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2 — Warm-path match calls
# ─────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("SECTION 2 — WARM-PATH MATCH CALLS")
print(SEP)

print("[S2] Call 1 (first match — GPU may still be warming) …")
t0_c1 = time.perf_counter()
result1 = match(FRAME_PATH, LAT, LON, ALT, HEADING)
wall_ms_1 = (time.perf_counter() - t0_c1) * 1000.0

if result1 is None:
    print(f"[S2] Call 1 returned None (wall={wall_ms_1:.1f} ms)")
    print("[S2] Tile resolver or match failure — cannot satisfy AC-1. FLAG FOR DEPUTY 1")
    sys.exit(1)

d_lat1, d_lon1, conf1, int_ms1 = result1
print(f"[S2] Call 1  conf={conf1:.4f}  Δlat={d_lat1:+.6f}°  Δlon={d_lon1:+.6f}°")
print(f"[S2]         wall={wall_ms_1:.1f} ms  internal={int_ms1:.1f} ms")

print("[S2] Call 2 (warm path) …")
t0_c2 = time.perf_counter()
result2 = match(FRAME_PATH, LAT, LON, ALT, HEADING)
wall_ms_2 = (time.perf_counter() - t0_c2) * 1000.0

if result2 is None:
    print(f"[S2] Call 2 returned None (wall={wall_ms_2:.1f} ms). FLAG FOR DEPUTY 1")
    sys.exit(1)

d_lat2, d_lon2, conf2, int_ms2 = result2
ipc_overhead_ms = wall_ms_2 - int_ms2

print(f"[S2] Call 2  conf={conf2:.4f}  Δlat={d_lat2:+.6f}°  Δlon={d_lon2:+.6f}°")
print(f"[S2]         wall={wall_ms_2:.1f} ms  internal={int_ms2:.1f} ms")
print(f"[S2]         IPC overhead (wall − internal): {ipc_overhead_ms:.1f} ms")

# AC-1
print()
print("── H6-AC-1  match() non-None for geographically matched frame ───────")
print(f"  MEASURED  : both calls returned non-None MatchResult")
print(f"  ASSESSMENT: WITHIN EXPECTATIONS")

# AC-2
print()
print("── H6-AC-2  confidence is float in [0.0, 1.0] ───────────────────────")
in_range = isinstance(conf2, float) and 0.0 <= conf2 <= 1.0
label_ac2 = "WITHIN EXPECTATIONS" if in_range else "FLAG FOR DEPUTY 1"
print(f"  MEASURED  : conf2 = {conf2:.4f}  type={type(conf2).__name__}")
print(f"  ASSESSMENT: {label_ac2}")
if conf2 > 0.97:
    print(f"  NOTE      : conf > 0.97 — possible identity-match artefact. FLAG FOR DEPUTY 1")
if conf2 < CONF_ACCEPT:
    print(f"  NOTE      : conf < {CONF_ACCEPT} — frame in reject minority (Phase D-1 ~7%). FLAG FOR DEPUTY 1")

# AC-3
print()
print("── H6-AC-3  warm-path latency reported in ms ────────────────────────")
print(f"  MEASURED  : wall={wall_ms_2:.1f} ms  internal={int_ms2:.1f} ms  IPC overhead={ipc_overhead_ms:.1f} ms")
print(f"  REFERENCE : H-3 warm median = 628 ms")
# Flag if wall > 5000ms (would suggest re-cold-start)
label_ac3 = "FLAG FOR DEPUTY 1 (possible re-cold-start)" if wall_ms_2 > 5000 else "WITHIN EXPECTATIONS"
print(f"  ASSESSMENT: {label_ac3}")


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3 — NavigationManager AC-4
# ─────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("SECTION 3 — NavigationManager NAV_LIGHTGLUE_CORRECTION (AC-4)")
print(SEP)
print("[S3] Constructing minimal mock stack …")

mock_eskf         = MagicMock()
mock_eskf.update_trn.return_value = (0.0, False, 0.0)  # (nis, rejected=False, innov_mag)
# rejected=False is critical: bare MagicMock() is truthy → would block event

mock_bim          = MagicMock()
mock_trn          = MagicMock()
mock_vio_mode     = MagicMock()
mock_vio_proc     = MagicMock()

mock_camera_bridge = MagicMock()
mock_camera_bridge.last_frame_path = FRAME_PATH  # Step 4a reads via getattr()

event_log: list = []

nm = NavigationManager(
    eskf=mock_eskf,
    bim=mock_bim,
    trn=mock_trn,
    vio_mode=mock_vio_mode,
    camera_bridge=mock_camera_bridge,
    vio_processor=mock_vio_proc,
    event_log=event_log,
    clock_fn=lambda: 0,
    trn_interval_m=5000.0,   # default; mission_km=6.0 → 6000m > 5000m → Step 4a entered
    lightglue_client=_lg_module,
)

mock_state = MagicMock()

print(f"[S3] NavigationManager constructed. lightglue_client={_lg_module.__name__}")
print(f"[S3] Calling update(): gnss_available=False  mission_km=6.0  terrain_class=ACCEPT")

t0_nm = time.perf_counter()
output = nm.update(
    state=mock_state,
    gnss_available=False,
    gnss_pos=None,
    gnss_measurement=None,
    mission_km=6.0,
    alt_m=ALT,
    gsd_m=0.3,
    lat_estimate=LAT,
    lon_estimate=LON,
    camera_tile=None,
    mission_time_ms=0,
    terrain_class="ACCEPT",
)
nm_wall_ms = (time.perf_counter() - t0_nm) * 1000.0

all_events = [e["event"] for e in output.event_log_entries]
lg_events  = [e for e in output.event_log_entries if e["event"] == "NAV_LIGHTGLUE_CORRECTION"]

print(f"[S3] update() wall time    : {nm_wall_ms:.1f} ms")
print(f"[S3] cycle_log events      : {all_events}")
print(f"[S3] NAV_LIGHTGLUE_CORRECTION count: {len(lg_events)}")

if lg_events:
    payload = lg_events[0]["payload"]
    print(f"[S3] Event payload:")
    for k, v in payload.items():
        print(f"       {k}: {v}")

print()
print("── H6-AC-4  NAV_LIGHTGLUE_CORRECTION in cycle_log ──────────────────")
if lg_events:
    payload = lg_events[0]["payload"]
    label_ac4 = "WITHIN EXPECTATIONS"
    print(f"  MEASURED  : event present  confidence={payload['confidence']}  "
          f"terrain_class={payload['terrain_class']}")
else:
    label_ac4 = "FLAG FOR DEPUTY 1"
    print(f"  MEASURED  : event NOT present in cycle_log")
    print(f"  DIAG      : mock_eskf.update_trn called = {mock_eskf.update_trn.called}")
    print(f"  DIAG      : output.nav_mode = {output.nav_mode}")
print(f"  ASSESSMENT: {label_ac4}")


# ─────────────────────────────────────────────────────────────────────────
# SECTION 4 — Frozen file hashes (AC-5)
# ─────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("SECTION 4 — FROZEN FILE SHA-256 HASHES (AC-5)")
print(SEP)

hash_ok = True
for rel in FROZEN_FILES:
    abs_path = os.path.join(REPO_ROOT, rel)
    try:
        h = _sha256(abs_path)
        print(f"  {rel}")
        print(f"    {h}")
    except FileNotFoundError:
        print(f"  {rel}: FILE NOT FOUND — FLAG FOR DEPUTY 1")
        hash_ok = False

print()
print("── H6-AC-5  frozen file hashes ──────────────────────────────────────")
print(f"  MEASURED  : hashes printed above (Orin HEAD 793eb73)")
print(f"  NOTE      : Deputy 1 to compare against certified dev baseline")
print(f"  ASSESSMENT: reported for Deputy 1 ruling")


# ─────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("H-6 SCRIPT COMPLETE")
print(SEP)
