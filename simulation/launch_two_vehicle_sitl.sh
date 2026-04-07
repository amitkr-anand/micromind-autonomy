#!/usr/bin/env bash
# launch_two_vehicle_sitl.sh
# MicroMind OI-20 — Two-vehicle Gazebo SITL rendering verification.
#
# Launches Gazebo Sim with two x500 UAV models visible simultaneously.
# Does NOT require PX4 or MicroMind stack — rendering verification only.
#
# RENDERING FIX (OI-20):
#   Root cause: RTX 5060 Ti (micromind-node01) requires NVIDIA EGL.
#               Mesa's gallium driver crashes on this GPU with OGRE2.
#   Fix 1 (server):  GZ_ENGINE_NAME=ogre  → forces OGRE1, no OGRE2 crash
#   Fix 2 (GUI):     __EGL_VENDOR_LIBRARY_FILENAMES=10_nvidia.json
#                    + LD_PRELOAD=libpthread.so.0
#                    + XDG_RUNTIME_DIR=/run/user/1000
#
# Verified on micromind-node01:
#   GPU:    NVIDIA GeForce RTX 5060 Ti, driver 580.126.09
#   OS:     Ubuntu 24.04.4
#   Gazebo: Harmonic 8.11.0
#   Result: x500_0 and x500_1 both in scene, RTF ~1.0, stable 35+ s
#
# Usage:
#   ./simulation/launch_two_vehicle_sitl.sh
#   # Opens Gazebo GUI — both drones visible, no blank screen.
#
#   HEADLESS=1 ./simulation/launch_two_vehicle_sitl.sh
#   # Runs server only (no GUI), for CI/headless verification.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORLD_FILE="$REPO_DIR/simulation/worlds/two_vehicle_sitl.sdf"
PX4_MODELS="$HOME/PX4-Autopilot/Tools/simulation/gz/models"

# Validate world file exists
if [ ! -f "$WORLD_FILE" ]; then
    echo "ERROR: World file not found: $WORLD_FILE"
    exit 1
fi

# Validate PX4 model path
if [ ! -d "$PX4_MODELS" ]; then
    echo "ERROR: PX4 gz models not found at: $PX4_MODELS"
    echo "       Ensure PX4-Autopilot is cloned at ~/PX4-Autopilot"
    exit 1
fi

echo ""
echo "============================================================"
echo "  MicroMind — Two-Vehicle SITL Rendering Test (OI-20)"
echo "  World: $WORLD_FILE"
echo "  Vehicles: x500_0 @ [0,0,0.5]  |  x500_1 @ [0,5,0.5]"
echo "============================================================"
echo ""

export GZ_SIM_RESOURCE_PATH="$PX4_MODELS"
export GZ_IP=127.0.0.1
export DISPLAY="${DISPLAY:-:1}"

# Server (headless physics engine)
echo "[sitl] Starting Gazebo server (OGRE1 render engine)..."
GZ_ENGINE_NAME=ogre gz sim -r -s --headless-rendering \
    "$WORLD_FILE" 2>&1 &
SRV_PID=$!

echo "[sitl] Server PID: $SRV_PID"
echo "[sitl] Waiting for world to initialise..."
sleep 8

# Confirm both models are in the scene
echo "[sitl] Querying scene..."
SCENE=$(GZ_IP=127.0.0.1 gz service -s /world/two_vehicle_sitl/scene/info \
    --reqtype gz.msgs.Empty --reptype gz.msgs.Scene \
    --timeout 5000 --req "" 2>/dev/null)

X500_0_OK=$(echo "$SCENE" | grep -c '"x500_0"' || true)
X500_1_OK=$(echo "$SCENE" | grep -c '"x500_1"' || true)

if [ "$X500_0_OK" -gt 0 ] && [ "$X500_1_OK" -gt 0 ]; then
    echo "[sitl] ✅ Both vehicles in scene: x500_0 and x500_1"
else
    echo "[sitl] ❌ Scene check FAILED:"
    echo "$SCENE" | grep "name:"
    kill $SRV_PID 2>/dev/null
    exit 1
fi

if [ -n "${HEADLESS}" ]; then
    echo "[sitl] HEADLESS mode — no GUI. Server running. Press Ctrl-C to stop."
    wait $SRV_PID
    exit 0
fi

# GUI (NVIDIA EGL fix required for RTX 5060 Ti)
echo "[sitl] Starting Gazebo GUI (NVIDIA EGL fix applied)..."
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0 \
XDG_RUNTIME_DIR=/run/user/1000 \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
GZ_IP=127.0.0.1 \
gz sim -g 2>&1 &
GUI_PID=$!
echo "[sitl] GUI PID: $GUI_PID"

echo ""
echo "[sitl] Gazebo running. Both x500 drones should be visible."
echo "[sitl] Close the Gazebo window or press Ctrl-C to stop."
echo ""

# Wait for GUI to exit
wait $GUI_PID
kill $SRV_PID 2>/dev/null
wait $SRV_PID 2>/dev/null
echo "[sitl] Done."
