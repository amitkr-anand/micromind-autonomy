#!/usr/bin/env bash
# run_demo.sh
# MicroMind OEM Demonstration — Two-Vehicle GPS Denial Demo
# v2.0  OI-30 Phase A + Phase B
#
# Usage:
#   ./run_demo.sh
#
# What it does (this version):
#   Phase 0 — Kill any existing PX4/Gazebo processes; clean stale instance dirs
#   Phase A — Start Gazebo server (baylands, OGRE1); wait for world ready; open GUI
#   Phase B — Start PX4 SITL instance 0 (Vehicle B) and instance 1 (Vehicle A);
#              wait for EKF2 alignment on both (gz topic poll, 60s timeout each)
#   [Prompt 3] — exec run_mission.py (NOT YET WIRED — OI-30 Phase C)
#
# Req IDs : PX4-01, VIZ-02 (infrastructure dependency)
# Open item: OI-30
# Commit   : feat(demo): OI-30 Phase B — PX4 SITL dual-instance launch with EKF2 alignment wait
#
# Exit codes:
#   0 — Phase B complete; both instances EKF2-aligned
#   1 — Timeout or startup failure

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
echo "  OI-30 Phase A + Phase B"
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
PX4_0_PID=$!
echo "[run_demo] PX4 instance 0 PID: $PX4_0_PID"
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
PX4_GZ_MODEL_POSE="0,5,0.5" \
GZ_ENGINE_NAME=ogre \
GZ_SIM_RESOURCE_PATH="$PX4_GZ_MODELS:$PX4_GZ_WORLDS" \
"$PX4_BIN" -i 1 -d "$PX4_ETC" > /tmp/px4_inst1.log 2>&1 &
PX4_1_PID=$!
echo "[run_demo] PX4 instance 1 PID: $PX4_1_PID"
cd "$REPO_DIR"
echo ""

# ---------------------------------------------------------------------------
# EKF2 alignment wait — poll gz topic for each instance
#
# Topic paths (from build/px4_sitl_default/etc/init.d-posix/rcS line 296-299):
#   instance 0: /fmu/out/vehicle_local_position  (no namespace prefix)
#   instance 1: /px4_1/fmu/out/vehicle_local_position  (ns prefix px4_<N>)
#
# Each poll issues "gz topic -e -n 1 <topic>" with a 2s per-attempt timeout.
# Loop runs for up to 60s per instance. xy_valid: true confirms EKF2 aligned.
# ---------------------------------------------------------------------------

wait_ekf2_aligned() {
    local instance=$1
    local topic=$2
    local timeout_s=60
    local start_t
    start_t=$(date +%s)

    echo "[run_demo] Waiting for EKF2 alignment on instance $instance (topic: $topic, timeout: ${timeout_s}s)..."

    while true; do
        local now elapsed
        now=$(date +%s)
        elapsed=$(( now - start_t ))

        if [ "$elapsed" -ge "$timeout_s" ]; then
            echo "EKF2_ALIGNMENT_TIMEOUT instance=$instance"
            return 1
        fi

        # Receive one message with a 2s hard deadline; ignore errors if topic
        # is not yet publishing (PX4 still initialising).
        local msg
        msg=$(timeout 2 gz topic -e -n 1 "$topic" 2>/dev/null) || true

        if echo "$msg" | grep -q "xy_valid: true"; then
            echo "EKF2_ALIGNED instance=$instance"
            return 0
        fi

        sleep 1
    done
}

# Allow PX4 instances time to start gz_bridge and begin sensor fusion before
# the first poll attempt. Typical EKF2 alignment: 10–20s after sensor start.
echo "[run_demo] Waiting 10s for PX4 instances to initialise sensor fusion..."
sleep 10

wait_ekf2_aligned 0 "/fmu/out/vehicle_local_position"        || { echo "[run_demo] FAIL: EKF2 timeout on instance 0 — check /tmp/px4_inst0.log"; exit 1; }
wait_ekf2_aligned 1 "/px4_1/fmu/out/vehicle_local_position"  || { echo "[run_demo] FAIL: EKF2 timeout on instance 1 — check /tmp/px4_inst1.log"; exit 1; }

# ---------------------------------------------------------------------------
# Phase B complete
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  PHASE B COMPLETE"
echo "  Vehicle B (instance 0) — EKF2 aligned, port 14540"
echo "  Vehicle A (instance 1) — EKF2 aligned, port 14541"
echo "  Next: Prompt 3 will wire run_mission.py (OI-30 Phase C)"
echo "  Logs: /tmp/px4_inst0.log  /tmp/px4_inst1.log"
echo "        /tmp/gz_server.log  /tmp/gz_gui.log"
echo "============================================================"
echo ""

# Keep script alive so Gazebo/PX4 stay up; exit on Ctrl-C
echo "[run_demo] Ctrl-C to stop."
wait "$GZ_SRV_PID" 2>/dev/null || true
