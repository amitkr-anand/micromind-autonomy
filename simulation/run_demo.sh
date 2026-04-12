#!/usr/bin/env bash
# simulation/run_demo.sh — MicroMind OEM Demo, single-command launcher
# OI-30: Two-vehicle Baylands SITL + GPS denial mission
#
# Usage:
#   ./simulation/run_demo.sh
#   LOOPS=1 GPS_DENIAL_TIME=30 ./simulation/run_demo.sh
#   HEADLESS=1 ./simulation/run_demo.sh          # skip Gazebo GUI
#   VIO_OUTAGE=1 ./simulation/run_demo.sh        # inject 10s VIO outage on Veh B
#
# Configurable via environment variables (all have safe defaults):
#   LOOPS            — ellipse laps per vehicle       (default: 2)
#   GPS_DENIAL_TIME  — seconds before GPS denial      (default: 60)
#   VIO_OUTAGE       — set to 1 to enable VIO outage  (default: 0)
#   HEADLESS         — set to 1 to skip Gazebo GUI    (default: 0)
#
# Requirement references:
#   OI-30: single command, no developer knowledge required
#   OI-33: Baylands world, spawn ≥ 50 m AGL (tree collision mesh)
#   OI-20: NVIDIA EGL + OGRE1 fix for RTX 5060 Ti on micromind-node01
#   OI-35: Vehicle A OFFBOARD fix (setpoint stream thread, run_mission.py)

set -uo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$HOME/PX4-Autopilot/build/px4_sitl_default"
PX4_BIN="$BUILD/bin/px4"
PX4_ETC="$BUILD/etc"
PX4_MODELS="$HOME/PX4-Autopilot/Tools/simulation/gz/models"
BAYLANDS_WORLD="$HOME/PX4-Autopilot/Tools/simulation/gz/worlds/baylands.sdf"

# PID tracking — initialised empty; checked before use in cleanup
GZ_SERVER_PID=""
GUI_PID=""
PX4_0_PID=""
PX4_1_PID=""
OVERLAY_PID=""

# ---------------------------------------------------------------------------
# Exit trap — fires on EXIT, INT, TERM
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "[SHUTDOWN] Stopping demo..."
    [ -n "$OVERLAY_PID" ]   && kill "$OVERLAY_PID"   2>/dev/null || true
    [ -n "$GUI_PID" ]       && kill "$GUI_PID"       2>/dev/null || true
    [ -n "$PX4_1_PID" ]     && kill "$PX4_1_PID"     2>/dev/null || true
    [ -n "$PX4_0_PID" ]     && kill "$PX4_0_PID"     2>/dev/null || true
    [ -n "$GZ_SERVER_PID" ] && kill "$GZ_SERVER_PID" 2>/dev/null || true
    pkill -f "demo_overlay.py"       2>/dev/null || true
    pkill -f "baylands_demo_camera"  2>/dev/null || true
    pkill -f "run_mission.py" 2>/dev/null || true
    pkill -f "gz sim"          2>/dev/null || true
    pkill -f "px4"             2>/dev/null || true
    echo "[SHUTDOWN] Done."
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  MicroMind OEM Demo — Baylands Two-Vehicle SITL"
echo "  Loops: ${LOOPS:-2}   GPS denial at: ${GPS_DENIAL_TIME:-60}s"
[ "${HEADLESS:-0}" = "1" ] && echo "  Mode: HEADLESS (no Gazebo GUI)"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------
if [ ! -f "$PX4_BIN" ]; then
    echo "[ERROR] PX4 binary not found: $PX4_BIN"
    echo "        Ensure PX4-Autopilot is built at ~/PX4-Autopilot"
    exit 1
fi
if [ ! -f "$BAYLANDS_WORLD" ]; then
    echo "[ERROR] Baylands world not found: $BAYLANDS_WORLD"
    echo "        Ensure PX4-Autopilot is cloned at ~/PX4-Autopilot"
    exit 1
fi
if [ ! -f "$REPO_DIR/simulation/run_mission.py" ]; then
    echo "[ERROR] run_mission.py not found: $REPO_DIR/simulation/run_mission.py"
    exit 1
fi
if ! python3.12 -c "from pymavlink import mavutil" 2>/dev/null; then
    echo "[ERROR] pymavlink unavailable in python3.12 — activate the conda env first"
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 0 — Stale process cleanup
# Lesson from OI-20: stale PX4 SITL pollutes GZ transport; recurring hazard.
# ---------------------------------------------------------------------------
echo "[PHASE 0] Killing stale gz / px4 / mission processes..."
pkill -f "gz sim"       2>/dev/null || true
pkill -f "gzserver"     2>/dev/null || true
pkill -f " gz "         2>/dev/null || true
pkill -f "gz$"          2>/dev/null || true
pkill -f "px4"          2>/dev/null || true
pkill -f "gz_bridge"    2>/dev/null || true
pkill -f "run_mission"  2>/dev/null || true
pkill -f "mavlink"      2>/dev/null || true
sleep 3

# Fail-safe: ensure no gz sim PID survives before proceeding
STALE_GZ=$(pgrep -f "gz sim" 2>/dev/null || true)
if [ -n "$STALE_GZ" ]; then
    echo "[ERROR] Stale gz sim still running (PID $STALE_GZ) after cleanup."
    echo "        Run: kill -9 $STALE_GZ  then retry."
    exit 1
fi
echo "[PHASE 0] Clean."

# ---------------------------------------------------------------------------
# Phase 1 — Environment + Gazebo headless server (Baylands)
#
# Env var block reused from simulation/launch_two_vehicle_sitl.sh exactly.
# OI-20 fix: GZ_ENGINE_NAME=ogre (OGRE1) avoids OGRE2 crash on RTX 5060 Ti.
# EGL fix:   __EGL_VENDOR_LIBRARY_FILENAMES + LD_PRELOAD + XDG_RUNTIME_DIR.
# ---------------------------------------------------------------------------
echo "[PHASE 1] Configuring environment..."

# --- env var block from launch_two_vehicle_sitl.sh ---
export GZ_SIM_RESOURCE_PATH="$PX4_MODELS:$HOME/PX4-Autopilot/Tools/simulation/gz/worlds"
export GZ_IP=127.0.0.1
export DISPLAY="${DISPLAY:-:1}"
# --- end env var block ---

echo "[PHASE 1] Launching Baylands world (headless server)..."

GZ_ENGINE_NAME=ogre \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0 \
XDG_RUNTIME_DIR=/run/user/1000 \
GZ_IP=127.0.0.1 \
GZ_SIM_RESOURCE_PATH="$GZ_SIM_RESOURCE_PATH" \
gz sim -r -s --headless-rendering "$BAYLANDS_WORLD" > /tmp/gz_demo.log 2>&1 &
GZ_SERVER_PID=$!
echo "[PHASE 1] Gazebo server PID $GZ_SERVER_PID"

echo "[PHASE 1] Waiting for Baylands world (max 15s)..."
WORLD_READY=0
for i in $(seq 1 15); do
    sleep 1
    if GZ_IP=127.0.0.1 gz service -s /world/baylands/scene/info \
            --reqtype gz.msgs.Empty --reptype gz.msgs.Scene \
            --timeout 2000 --req "" > /dev/null 2>&1; then
        WORLD_READY=1
        echo "[PHASE 1] World ready at ${i}s."
        break
    fi
    echo "[PHASE 1] ... ${i}/15"
done

if [ "$WORLD_READY" -eq 0 ]; then
    echo "[ERROR] Gazebo failed to start within 15s. See /tmp/gz_demo.log"
    exit 1
fi

# Gazebo GUI — skip in HEADLESS mode or if DISPLAY is unset
if [ "${HEADLESS:-0}" != "1" ]; then
    echo "[PHASE 1] Launching Gazebo GUI (NVIDIA EGL fix applied)..."
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0 \
    XDG_RUNTIME_DIR=/run/user/1000 \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    GZ_IP=127.0.0.1 \
    gz sim -g > /tmp/gz_gui.log 2>&1 &
    GUI_PID=$!
    echo "[PHASE 1] GUI PID $GUI_PID"
    sleep 3
fi

# ---------------------------------------------------------------------------
# Phase 2 — PX4 SITL instances
# Instance 0: Vehicle B (MicroMind, spawn [0,0,0.5], MAVLink port 14540)
# Instance 1: Vehicle A (INS-only,   spawn [0,5,0.5], MAVLink port 14541)
# Each runs in a subshell so cd does not affect the parent; exec replaces
# the subshell so the captured PID is the PX4 process itself.
# ---------------------------------------------------------------------------
echo "[PHASE 2] Launching PX4 instance 0 (Vehicle B, port 14540)..."
mkdir -p /tmp/px4_demo_0 /tmp/px4_demo_1

(
    cd /tmp/px4_demo_0
    GZ_IP=127.0.0.1 \
    GZ_SIM_RESOURCE_PATH="$PX4_MODELS" \
    PX4_SYS_AUTOSTART=4001 \
    PX4_GZ_WORLD=baylands \
    PX4_GZ_STANDALONE=1 \
    PX4_GZ_MODEL_POSE="0,0,0.5,0,0,0" \
    exec "$PX4_BIN" -i 0 "$PX4_ETC"
) > /tmp/px4_0.log 2>&1 &
PX4_0_PID=$!
echo "[PHASE 2] PX4 instance 0 PID $PX4_0_PID"

echo "[PHASE 2] Launching PX4 instance 1 (Vehicle A, port 14541)..."
(
    cd /tmp/px4_demo_1
    GZ_IP=127.0.0.1 \
    GZ_SIM_RESOURCE_PATH="$PX4_MODELS" \
    PX4_SYS_AUTOSTART=4001 \
    PX4_GZ_WORLD=baylands \
    PX4_GZ_STANDALONE=1 \
    PX4_GZ_MODEL_POSE="0,5,0.5,0,0,0" \
    exec "$PX4_BIN" -i 1 "$PX4_ETC"
) > /tmp/px4_1.log 2>&1 &
PX4_1_PID=$!
echo "[PHASE 2] PX4 instance 1 PID $PX4_1_PID"

echo "[PHASE 2] Waiting 20s for PX4 init + gz_bridge + EKF2 pre-alignment..."
sleep 20
echo "[PHASE 2] Init wait complete."

# ---------------------------------------------------------------------------
# Phase 3 — EKF2 alignment gate
# Polls LOCAL_POSITION_NED on both MAVLink ports.
# Exit 1 with clear diagnostics if either vehicle fails to align.
# ---------------------------------------------------------------------------
echo "[PHASE 3] Checking EKF2 alignment on both vehicles..."

python3.12 - <<'PYEOF'
import sys
import time
from pymavlink import mavutil

PORTS = [(14540, "VEH_B"), (14541, "VEH_A")]
failures = []

for port, label in PORTS:
    try:
        mav = mavutil.mavlink_connection(f"udp:127.0.0.1:{port}")
        hb = mav.wait_heartbeat(timeout=10)
        if hb is None:
            failures.append(f"{label}: no heartbeat on port {port}")
            continue
        # Send GCS heartbeats to register GCS and trigger telemetry stream
        for _ in range(6):
            mav.mav.heartbeat_send(6, 8, 0, 0, 4)
            time.sleep(0.5)
        # Poll LOCAL_POSITION_NED — published once EKF2 has aligned
        deadline = time.time() + 15
        aligned = False
        while time.time() < deadline:
            mav.mav.heartbeat_send(6, 8, 0, 0, 4)
            msg = mav.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=1.0)
            if msg is not None:
                print(f"[PHASE 3] {label} EKF2 aligned: x={msg.x:.3f}m", flush=True)
                aligned = True
                break
        if not aligned:
            failures.append(f"{label}: EKF2 alignment timeout (port {port})")
    except Exception as exc:
        failures.append(f"{label}: {exc}")

if failures:
    for fail in failures:
        print(f"[PHASE 3] FAIL — {fail}", file=sys.stderr, flush=True)
    sys.exit(1)

sys.exit(0)
PYEOF

EKF2_RESULT=$?
if [ "$EKF2_RESULT" -ne 0 ]; then
    echo "[ERROR] EKF2 alignment gate FAILED."
    echo "        Logs: /tmp/px4_0.log  /tmp/px4_1.log"
    exit 1
fi
echo "[PHASE 3] EKF2 gate PASS — both vehicles aligned."

# ---------------------------------------------------------------------------
# Phase 3b — VIZ-02 Run 1 Overlay
# baylands_demo_camera.py: set Gazebo GUI camera to top-down position once.
# demo_overlay.py: launch real-time matplotlib MAVLink overlay in background.
# Both run as background processes; OVERLAY_PID tracks demo_overlay.py.
# ---------------------------------------------------------------------------
echo "[PHASE 3b] Setting Gazebo camera (top-down Baylands)..."
python3.12 "$REPO_DIR/simulation/baylands_demo_camera.py" &
CAMERA_PID=$!
# Non-blocking: camera script exits after one service call
sleep 1

if [ "${HEADLESS:-0}" != "1" ]; then
    echo "[PHASE 3b] Launching live mission overlay (demo_overlay.py)..."
    python3.12 "$REPO_DIR/simulation/demo_overlay.py" &
    OVERLAY_PID=$!
    echo "[PHASE 3b] Overlay PID $OVERLAY_PID"
else
    echo "[PHASE 3b] HEADLESS mode — skipping matplotlib overlay."
fi

# ---------------------------------------------------------------------------
# Phase 4 — Mission
# run_mission.py is exec'd with tee so output goes to terminal and log.
# set -o pipefail is suspended so the mission exit code is captured cleanly
# via PIPESTATUS[0] before being re-raised.
# ---------------------------------------------------------------------------
EXTRA_ARGS=""
[ "${VIO_OUTAGE:-0}" = "1" ] && EXTRA_ARGS="--vio-outage"

echo ""
echo "[PHASE 4] Starting GPS denial mission..."
echo "          Loops=${LOOPS:-2}  GPS-denial=${GPS_DENIAL_TIME:-60}s  VIO-outage=${VIO_OUTAGE:-0}"
echo ""

set +o pipefail
python3.12 "$REPO_DIR/simulation/run_mission.py" \
    --loops "${LOOPS:-2}" \
    --gps-denial-time "${GPS_DENIAL_TIME:-60}" \
    ${EXTRA_ARGS:+$EXTRA_ARGS} \
    2>&1 | tee /tmp/demo_mission.log
MISSION_EXIT=${PIPESTATUS[0]}
set -o pipefail

echo ""
if [ "$MISSION_EXIT" -eq 0 ]; then
    echo "[DEMO] PASS — two-vehicle GPS denial demo complete."
else
    echo "[DEMO] FAIL — mission exit code $MISSION_EXIT. See /tmp/demo_mission.log"
fi

# Phase 5 (cleanup) is handled by the EXIT trap above.
exit "$MISSION_EXIT"
