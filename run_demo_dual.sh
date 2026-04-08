#!/usr/bin/env bash
# run_demo_dual.sh
# MicroMind Pre-HIL — Two-Vehicle GPS Denial Demonstration
# v1.0
#
# Usage:
#   ./run_demo_dual.sh [--gps-denial-time N] [--loops N] [--vio-outage] [--headless]
#
# What it does:
#   Phase 0: Kill stale processes
#   Phase 1: Validate environment
#   Phase 2: Launch PX4 instance 0 (Vehicle B — MicroMind, launches Gazebo)
#   Phase 3: Launch PX4 instance 1 (Vehicle A — INS-only, standalone)
#   Phase 4: Wait for both MAVLink heartbeats
#   Phase 5: Launch Gazebo GUI (skipped if --headless)
#   Phase 6: Spawn overlay panels for both vehicles
#   Phase 7: Run mission — wait for exit, then cleanup
#
# Arguments:
#   --gps-denial-time N   Seconds after takeoff before GPS denied (default: 60)
#   --loops N             Number of ellipse loops per vehicle (default: 2)
#   --vio-outage          Inject 10s VIO outage on Vehicle B mid-mission
#   --headless            Skip Gazebo GUI (CI / remote sessions)
#
# Exit codes:
#   0 — mission completed, all pass criteria met
#   1 — failure (see output above)

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="$HOME/PX4-Autopilot"
PX4_BIN="$PX4_DIR/build/px4_sitl_default/bin/px4"
PX4_ETC="$PX4_DIR/build/px4_sitl_default/etc"
GZ_ENV="$PX4_DIR/build/px4_sitl_default/rootfs/gz_env.sh"
INST0_DIR="/tmp/px4_inst0"
INST1_DIR="/tmp/px4_inst1"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
GPS_DENIAL_TIME=60
LOOPS=2
VIO_OUTAGE=0
HEADLESS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gps-denial-time)
            GPS_DENIAL_TIME="$2"; shift 2 ;;
        --loops)
            LOOPS="$2"; shift 2 ;;
        --vio-outage)
            VIO_OUTAGE=1; shift ;;
        --headless)
            HEADLESS=1; shift ;;
        *)
            echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# PID tracking for cleanup
# ---------------------------------------------------------------------------
INST0_PID=""
INST1_PID=""
GUI_PID=""
OVERLAY_A_PID=""
OVERLAY_B_PID=""

cleanup() {
    echo ""
    echo "[SHUTDOWN] Stopping demo..."
    [ -n "$OVERLAY_B_PID" ] && kill "$OVERLAY_B_PID" 2>/dev/null || true
    [ -n "$OVERLAY_A_PID" ] && kill "$OVERLAY_A_PID" 2>/dev/null || true
    [ -n "$GUI_PID" ]       && kill "$GUI_PID"       2>/dev/null || true
    [ -n "$INST1_PID" ]     && kill "$INST1_PID"     2>/dev/null || true
    [ -n "$INST0_PID" ]     && kill "$INST0_PID"     2>/dev/null || true
    pkill -f "gz sim"  2>/dev/null || true
    pkill -f "gzserver" 2>/dev/null || true
    pkill -f "bin/px4"  2>/dev/null || true
    echo "[SHUTDOWN] Demo stopped."
}

trap cleanup EXIT SIGINT SIGTERM

# ---------------------------------------------------------------------------
# Phase 0 — Stale process cleanup
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  MicroMind — Two-Vehicle GPS Denial Demo"
echo "  GPS denial at T+${GPS_DENIAL_TIME}s  |  Loops: ${LOOPS}  |  VIO outage: ${VIO_OUTAGE}"
echo "  Basis: micromind-autonomy @ $(git -C "$REPO_DIR" rev-parse --short HEAD)"
echo "============================================================"
echo ""
echo "[PHASE 0] Killing stale processes..."

(
    pkill -f "bin/px4"  2>/dev/null || true
    pkill -f "gz sim"   2>/dev/null || true
    pkill -f "gzserver" 2>/dev/null || true
    pkill -f " gz "     2>/dev/null || true
    pkill -f "gz$"      2>/dev/null || true
    pkill -f "gz_bridge" 2>/dev/null || true
    pkill -f "ros2.*gz" 2>/dev/null || true
    pkill -f "run_mission" 2>/dev/null || true
    pkill -f "demo_overlay_dual" 2>/dev/null || true
) || true

sleep 2
echo "[PHASE 0] Stale processes cleared."

# ---------------------------------------------------------------------------
# Phase 1 — Environment validation
# ---------------------------------------------------------------------------
echo "[PHASE 1] Validating environment..."

if [ ! -f "$PX4_BIN" ]; then
    echo "[ERROR] PX4 binary not found: $PX4_BIN"
    echo "        Build PX4 first: make -C $PX4_DIR px4_sitl"
    exit 1
fi

if [ ! -f "$GZ_ENV" ]; then
    echo "[ERROR] gz_env.sh not found: $GZ_ENV"
    exit 1
fi

if ! python3.12 -c "import pymavlink" 2>/dev/null; then
    echo "[ERROR] pymavlink not found for python3.12"
    echo "        Install: pip3.12 install pymavlink"
    exit 1
fi

if [ ! -f "$REPO_DIR/simulation/run_mission.py" ]; then
    echo "[ERROR] simulation/run_mission.py not found"
    exit 1
fi

export GZ_ENGINE_NAME=ogre
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0
export XDG_RUNTIME_DIR=/run/user/1000
export GZ_IP=127.0.0.1
export DISPLAY="${DISPLAY:-:1}"

echo "[PHASE 1] Environment validated."

# ---------------------------------------------------------------------------
# Phase 2 — Launch PX4 instance 0 (Vehicle B — MicroMind, launches Gazebo)
# ---------------------------------------------------------------------------
echo "[PHASE 2] Launching PX4 instance 0 (Vehicle B — MicroMind)..."

mkdir -p "$INST0_DIR"
cp "$GZ_ENV" "$INST0_DIR/gz_env.sh"

(
    cd "$INST0_DIR"
    GZ_ENGINE_NAME=ogre \
    GZ_IP=127.0.0.1 \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0 \
    XDG_RUNTIME_DIR=/run/user/1000 \
    PX4_SYS_AUTOSTART=4001 \
    PX4_SIM_MODEL=gz_x500 \
    PX4_GZ_WORLD=baylands \
    px4_instance=0 \
    "$PX4_BIN" -i 0 -d "$PX4_ETC" \
    > /tmp/px4_inst0.log 2>&1
) &
INST0_PID=$!

echo "[PHASE 2] PX4 instance 0 PID $INST0_PID — waiting 20s for Gazebo + EKF2..."
sleep 20

if ! kill -0 "$INST0_PID" 2>/dev/null; then
    echo "[ERROR] PX4 instance 0 exited unexpectedly."
    echo "        Log: /tmp/px4_inst0.log"
    exit 1
fi
echo "[PHASE 2] Instance 0 running."

# ---------------------------------------------------------------------------
# Phase 3 — Launch PX4 instance 1 (Vehicle A — INS-only, standalone)
# ---------------------------------------------------------------------------
echo "[PHASE 3] Launching PX4 instance 1 (Vehicle A — INS-only)..."

mkdir -p "$INST1_DIR"
cp "$GZ_ENV" "$INST1_DIR/gz_env.sh"

(
    cd "$INST1_DIR"
    GZ_ENGINE_NAME=ogre \
    GZ_IP=127.0.0.1 \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0 \
    XDG_RUNTIME_DIR=/run/user/1000 \
    PX4_SYS_AUTOSTART=4001 \
    PX4_SIM_MODEL=gz_x500 \
    PX4_GZ_WORLD=baylands \
    PX4_GZ_MODEL_POSE="0,5,0.5,0,0,0" \
    PX4_GZ_STANDALONE=1 \
    px4_instance=1 \
    "$PX4_BIN" -i 1 -d "$PX4_ETC" \
    > /tmp/px4_inst1.log 2>&1
) &
INST1_PID=$!

echo "[PHASE 3] PX4 instance 1 PID $INST1_PID — waiting 20s for EKF2..."
sleep 20

if ! kill -0 "$INST1_PID" 2>/dev/null; then
    echo "[ERROR] PX4 instance 1 exited unexpectedly."
    echo "        Log: /tmp/px4_inst1.log"
    exit 1
fi
echo "[PHASE 3] Instance 1 running."

# ---------------------------------------------------------------------------
# Phase 4 — Verify both MAVLink heartbeats
# ---------------------------------------------------------------------------
echo "[PHASE 4] Verifying MAVLink heartbeats on ports 14540 and 14541..."

python3.12 << 'PYEOF'
import sys, time
sys.path.insert(0, '.')
import pymavlink.mavutil as mavutil

ok = True
for port, label in [(14540, "Vehicle B (inst0)"), (14541, "Vehicle A (inst1)")]:
    try:
        m = mavutil.mavlink_connection(f'udp:127.0.0.1:{port}')
        hb = m.wait_heartbeat(timeout=15)
        if hb is None:
            print(f"[PHASE 4] ERROR: No heartbeat from {label} (port {port})")
            ok = False
        else:
            print(f"[PHASE 4] Heartbeat OK: {label}  sysid={hb.get_srcSystem()}  port={port}")
    except Exception as e:
        print(f"[PHASE 4] ERROR connecting to {label} port {port}: {e}")
        ok = False

sys.exit(0 if ok else 1)
PYEOF

if [ $? -ne 0 ]; then
    echo "[ERROR] MAVLink verification failed — check PX4 logs above."
    exit 1
fi
echo "[PHASE 4] Both vehicles reachable."

# ---------------------------------------------------------------------------
# Phase 5 — Gazebo GUI  (skipped if --headless)
# ---------------------------------------------------------------------------
if [ "$HEADLESS" -eq 0 ]; then
    echo "[PHASE 5] Launching Gazebo GUI..."
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0 \
    XDG_RUNTIME_DIR=/run/user/1000 \
    GZ_IP=127.0.0.1 \
    gz sim -g &
    GUI_PID=$!
    sleep 5
    echo "[PHASE 5] Gazebo GUI launched (PID $GUI_PID)."
else
    echo "[PHASE 5] Skipped — headless mode."
fi

# ---------------------------------------------------------------------------
# Phase 6 — Overlay panels
# ---------------------------------------------------------------------------
echo "[PHASE 6] Launching overlay panels..."

if command -v gnome-terminal &>/dev/null && [ "$HEADLESS" -eq 0 ]; then
    gnome-terminal --title="MicroMind Overlay — Vehicle B" -- \
        bash -c "cd $REPO_DIR && python3.12 simulation/demo_overlay_dual.py --vehicle B; exec bash" &
    OVERLAY_B_PID=$!

    gnome-terminal --title="MicroMind Overlay — Vehicle A" -- \
        bash -c "cd $REPO_DIR && python3.12 simulation/demo_overlay_dual.py --vehicle A; exec bash" &
    OVERLAY_A_PID=$!

    sleep 2
    echo "[PHASE 6] Overlay panels launched."
else
    echo "[PHASE 6] Skipped — no display or headless mode."
fi

# ---------------------------------------------------------------------------
# Phase 7 — Run mission
# ---------------------------------------------------------------------------
echo ""
echo "[PHASE 7] Starting two-vehicle mission..."
echo "          GPS denial at T+${GPS_DENIAL_TIME}s"
echo "          Loops: ${LOOPS}"
echo "          VIO outage: ${VIO_OUTAGE}"
echo ""

cd "$REPO_DIR"

MISSION_ARGS=(
    "--gps-denial-time" "$GPS_DENIAL_TIME"
    "--loops"           "$LOOPS"
)
[ "$VIO_OUTAGE" -eq 1 ] && MISSION_ARGS+=("--vio-outage")

python3.12 simulation/run_mission.py "${MISSION_ARGS[@]}"
MISSION_EXIT=$?

echo ""
if [ $MISSION_EXIT -eq 0 ]; then
    echo "============================================================"
    echo "  DEMO PASS"
    echo "  GPS denial at T+${GPS_DENIAL_TIME}s, ${LOOPS} loops completed."
    echo "============================================================"
else
    echo "============================================================"
    echo "  DEMO FAIL — check output above"
    echo "  PX4 inst0 log: /tmp/px4_inst0.log"
    echo "  PX4 inst1 log: /tmp/px4_inst1.log"
    echo "============================================================"
fi

exit $MISSION_EXIT
