#!/usr/bin/env bash
# run_demo.sh
# MicroMind OEM Demonstration — Two-Vehicle GPS Denial Demo
# v3.0  OI-30 Phase A + Phase B + Phase C (full integration)
#
# Usage:
#   ./run_demo.sh
#
# What it does:
#   Phase 0 — Kill any existing PX4/Gazebo processes; clean stale instance dirs
#   Phase A — Start Gazebo server (baylands, OGRE1); wait for world ready; open GUI
#   Phase B — Start PX4 SITL instance 0 (Vehicle B) and instance 1 (Vehicle A);
#              wait for EKF2 alignment on both (MAVLink LOCAL_POSITION_NED, 60s timeout each)
#   Phase C — exec run_mission.py (two-vehicle GPS denial demo mission)
#
# Req IDs : PX4-01, VIZ-02
# Open item: OI-30 (CLOSED this commit)
#
# Exit codes:
#   0 — Mission complete (run_mission.py PASS)
#   1 — Startup failure or EKF2 timeout

set -e

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="$HOME/PX4-Autopilot"
PX4_BIN="$PX4_DIR/build/px4_sitl_default/bin/px4"
PX4_ETC="$PX4_DIR/build/px4_sitl_default/etc"
PX4_GZ_MODELS="$PX4_DIR/Tools/simulation/gz/models"
PX4_GZ_WORLDS="$PX4_DIR/Tools/simulation/gz/worlds"
PX4_GZ_PLUGINS="$PX4_DIR/build/px4_sitl_default/src/modules/simulation/gz_plugins"
BAYLANDS_SDF="$PX4_GZ_WORLDS/baylands.sdf"

# ---------------------------------------------------------------------------
# Trap — clean shutdown of PX4 instances on INT/TERM/EXIT.
# Registered before any process is launched so it covers all phases.
# PX4_INST0_PID and PX4_INST1_PID are captured after each background launch;
# until set they are empty, so kill fails silently (2>/dev/null).
# Note: after 'exec python3.12 run_mission.py' (Phase C) the shell is
# replaced; this trap protects Phases 0-B only.
# ---------------------------------------------------------------------------

# shellcheck disable=SC2064
trap 'kill ${PX4_INST0_PID} ${PX4_INST1_PID} 2>/dev/null; pkill -f "bin/px4" 2>/dev/null; exit' INT TERM EXIT

# ---------------------------------------------------------------------------
# Environment — NVIDIA EGL fix (proven on micromind-node01, RTX 5060 Ti)
# Exact pattern from simulation/launch_two_vehicle_sitl.sh (OI-20 fix)
# ---------------------------------------------------------------------------

export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0
export XDG_RUNTIME_DIR=/run/user/1000
export DISPLAY="${DISPLAY:-:1}"

# GZ resource paths — from build/px4_sitl_default/rootfs/gz_env.sh
export GZ_SIM_RESOURCE_PATH="$PX4_GZ_MODELS:$PX4_GZ_WORLDS"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$PX4_GZ_PLUGINS"
export GZ_IP=127.0.0.1

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  MicroMind — Two-Vehicle GPS Denial Demo"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Basis: micromind-autonomy @ $(git -C "$REPO_DIR" rev-parse --short HEAD)"
echo "  OI-30 Phase A + B + C (full integration)"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Phase 0: Cleanup
# Kill any stale PX4/Gazebo processes and remove previous instance dirs.
# Pattern proven in launch_two_vehicle_sitl.sh and QA-012 (stale instance risk).
# ---------------------------------------------------------------------------

echo "[run_demo] Phase 0: Cleaning up stale processes..."
pkill -9 -f "bin/px4"        2>/dev/null || true
pkill -9 -f "gz sim"         2>/dev/null || true
pkill -9 -f "ruby.*gz"       2>/dev/null || true
rm -rf /tmp/px4_inst0 /tmp/px4_inst1
sleep 2
echo "[run_demo] Cleanup done."
echo ""

# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------

if [ ! -f "$PX4_BIN" ]; then
    echo "[run_demo] ERROR: PX4 binary not found at $PX4_BIN"
    echo "           Run: cd $PX4_DIR && make px4_sitl_default"
    exit 1
fi

if [ ! -f "$BAYLANDS_SDF" ]; then
    echo "[run_demo] ERROR: Baylands world not found at $BAYLANDS_SDF"
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase A: Gazebo server — baylands world, OGRE1 render engine
# Pattern: GZ_ENGINE_NAME=ogre (from px4-rc.gzsim line 50 — OGRE1 avoids
# OGRE2 crash on RTX 5060 Ti). --headless-rendering keeps server stable
# when no display-attached viewer is open yet.
# ---------------------------------------------------------------------------

echo "[run_demo] Phase A: Starting Gazebo server (baylands, OGRE1)..."
GZ_ENGINE_NAME=ogre \
gz sim -r -s --headless-rendering \
    "$BAYLANDS_SDF" > /tmp/gz_server.log 2>&1 &
GZ_SRV_PID=$!
echo "[run_demo] Gazebo server PID: $GZ_SRV_PID"

# Wait for the world to be ready — poll scene/info service (same pattern as
# launch_two_vehicle_sitl.sh scene verification block)
echo "[run_demo] Waiting for Gazebo world to initialise..."
GZ_READY=0
for attempt in $(seq 1 30); do
    SERVICE_INFO=$(gz service -i --service "/world/baylands/scene/info" 2>&1 || true)
    if echo "$SERVICE_INFO" | grep -q "Service providers"; then
        GZ_READY=1
        break
    fi
    sleep 1
done

if [ "$GZ_READY" -eq 0 ]; then
    echo "[run_demo] ERROR: Timed out waiting for Gazebo world (30s)"
    echo "           Check /tmp/gz_server.log for details"
    kill "$GZ_SRV_PID" 2>/dev/null || true
    exit 1
fi
echo "[run_demo] GAZEBO_READY world=baylands"

# Gazebo GUI — NVIDIA EGL fix applied (same env as launch_two_vehicle_sitl.sh)
echo "[run_demo] Starting Gazebo GUI (NVIDIA EGL fix applied)..."
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0 \
XDG_RUNTIME_DIR=/run/user/1000 \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
GZ_IP=127.0.0.1 \
gz sim -g > /tmp/gz_gui.log 2>&1 &
GZ_GUI_PID=$!
echo "[run_demo] Gazebo GUI PID: $GZ_GUI_PID"
sleep 2
echo ""

# ---------------------------------------------------------------------------
# Phase B: PX4 SITL instances
#
# Instance 0 — Vehicle B (MicroMind, sysid=1, MAVLink port 14540)
#   Spawn: 0,0,0.5  (matches SPAWN_B_ENU in run_mission.py)
#   No PX4_GZ_STANDALONE — px4-rc.gzsim auto-detects the running world via
#   "gz topic -l | grep /world/.*/clock" and skips launching Gazebo.
#
# Instance 1 — Vehicle A (INS-only, sysid=2, MAVLink port 14541)
#   Spawn: 0,5,0.5  (matches SPAWN_A_ENU in run_mission.py)
#   PX4_GZ_STANDALONE=1 — bypasses the world launch block entirely, so
#   instance 1 does not race against instance 0 on Gazebo init.
#
# Both run from separate working directories (sitl_multiple_run.sh pattern).
# GZ_ENGINE_NAME=ogre — consistent with server render engine (px4-rc.gzsim).
# GZ_SIM_RESOURCE_PATH already exported above.
# ---------------------------------------------------------------------------

echo "[run_demo] Phase B: Starting PX4 SITL instances..."

# --- Instance 0: Vehicle B ---
mkdir -p /tmp/px4_inst0
cd /tmp/px4_inst0
echo "[run_demo] Starting PX4 instance 0 (Vehicle B, port 14540)..."
PX4_SYS_AUTOSTART=4001 \
PX4_SIM_MODEL=gz_x500 \
PX4_GZ_MODEL_POSE="0,0,0.5" \
GZ_ENGINE_NAME=ogre \
GZ_SIM_RESOURCE_PATH="$PX4_GZ_MODELS:$PX4_GZ_WORLDS" \
"$PX4_BIN" -i 0 -d "$PX4_ETC" > /tmp/px4_inst0.log 2>&1 &
PX4_INST0_PID=$!
echo "[run_demo] PX4 instance 0 PID: $PX4_INST0_PID"
cd "$REPO_DIR"

# Brief stagger — let instance 0 attach gz_bridge and spawn x500_0 before
# instance 1 starts. px4-rc.gzsim waits ~30s for scene/info; 4s is enough
# for the Gazebo service registration to stabilise.
sleep 4

# --- Instance 1: Vehicle A ---
mkdir -p /tmp/px4_inst1
cd /tmp/px4_inst1
echo "[run_demo] Starting PX4 instance 1 (Vehicle A, port 14541)..."
PX4_SYS_AUTOSTART=4001 \
PX4_SIM_MODEL=gz_x500 \
PX4_GZ_STANDALONE=1 \
PX4_GZ_WORLD=baylands \
PX4_GZ_MODEL_POSE="0,5,0.5" \
GZ_ENGINE_NAME=ogre \
GZ_SIM_RESOURCE_PATH="$PX4_GZ_MODELS:$PX4_GZ_WORLDS" \
"$PX4_BIN" -i 1 -d "$PX4_ETC" > /tmp/px4_inst1.log 2>&1 &
PX4_INST1_PID=$!
echo "[run_demo] PX4 instance 1 PID: $PX4_INST1_PID"
cd "$REPO_DIR"
echo ""

# ---------------------------------------------------------------------------
# EKF2 alignment wait — MAVLink LOCAL_POSITION_NED per instance
#
# Phase B (afdde74) used gz topic polling for /fmu/out/vehicle_local_position.
# That topic is published by uxrce_dds_client over UDP to the ROS2 DDS realm,
# NOT to Gazebo transport — gz topic cannot see it. Fixed here to use the
# proven MAVLink approach from run_mission.py:wait_ekf2_ready() and the
# original run_demo.sh v1.2 (both use LOCAL_POSITION_NED on the per-instance
# MAVLink UDP port: 14540 for instance 0, 14541 for instance 1).
#
# Port mapping (from px4-rc.mavlink):
#   udp_offboard_port_remote = 14540 + px4_instance
# ---------------------------------------------------------------------------

wait_ekf2_aligned() {
    local instance="$1"
    local port="$2"
    echo "[run_demo] Waiting for EKF2 alignment on instance $instance (MAVLink port $port, timeout: 60s)..."
    python3.12 - "$instance" "$port" <<'PYEOF'
import sys, time
from pymavlink import mavutil
instance, port = sys.argv[1], int(sys.argv[2])
m = mavutil.mavlink_connection(f'udp:127.0.0.1:{port}')
hb = m.wait_heartbeat(timeout=30)
if hb is None:
    print(f'EKF2_ALIGNMENT_TIMEOUT instance={instance}')
    sys.exit(1)
t = time.monotonic()
while time.monotonic() - t < 60.0:
    msg = m.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=1.0)
    if msg is not None:
        print(f'EKF2_ALIGNED instance={instance}')
        sys.exit(0)
print(f'EKF2_ALIGNMENT_TIMEOUT instance={instance}')
sys.exit(1)
PYEOF
}

# Allow PX4 instances time to start gz_bridge and begin sensor fusion before
# the first poll attempt. Typical EKF2 alignment: 10–20s after sensor start.
echo "[run_demo] Waiting 10s for PX4 instances to initialise sensor fusion..."
sleep 10

wait_ekf2_aligned 0 14540 || { echo "[run_demo] FAIL: EKF2 timeout on instance 0 — check /tmp/px4_inst0.log"; exit 1; }
wait_ekf2_aligned 1 14541 || { echo "[run_demo] FAIL: EKF2 timeout on instance 1 — check /tmp/px4_inst1.log"; exit 1; }

# ---------------------------------------------------------------------------
# Phase C: run_mission.py
# OI-30 CLOSED — full integration verified on micromind-node01
# Date: 10 April 2026
# Verified: Gazebo ✅ | PX4 x2 ✅ | EKF2 x2 ✅ | mission ✅
# Vehicle A: altitude 95 m ✅ | lap 1 complete ✅
# Commit: 97b2f5a
# ---------------------------------------------------------------------------

echo ""
echo "[run_demo] Phase C: Launching run_mission.py..."
echo ""
exec python3.12 "$REPO_DIR/simulation/run_mission.py" "$@"
