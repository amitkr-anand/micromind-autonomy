#!/usr/bin/env bash
# run_demo.sh
# MicroMind Pre-HIL — OEM Demonstration Script
# v1.2 §Part 9 — one command, reproducible 3x
#
# Usage:
#   ./run_demo.sh
#
# What it does:
#   1. Kills any existing PX4/Gazebo instances
#   2. Starts PX4 SITL with Gazebo (vehicle visible)
#   3. Waits for EKF2 alignment
#   4. Runs inject_outage.py (full §Part 9 flight sequence)
#   5. Generates HTML report in dashboard/
#   6. Prints report path on exit
#
# Exit codes:
#   0 — demo completed successfully
#   1 — demo failed

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="$HOME/PX4-Autopilot"
DASHBOARD_DIR="$REPO_DIR/dashboard"

# Gazebo NVIDIA EGL fix
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0
export XDG_RUNTIME_DIR=/run/user/1000
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

echo ""
echo "============================================================"
echo "  MicroMind — OEM Demonstration"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Basis: micromind-autonomy @ $(git -C "$REPO_DIR" rev-parse --short HEAD)"
echo "============================================================"
echo ""

# Step 1: Kill any existing instances
echo "[run_demo] Clearing previous SITL instances..."
pkill -9 -f "bin/px4" 2>/dev/null || true
pkill -9 -f "gz sim"  2>/dev/null || true
sleep 3

# Step 2: Start PX4 SITL + Gazebo in background
echo "[run_demo] Starting PX4 SITL + Gazebo..."
make -C "$PX4_DIR" px4_sitl gz_x500 > /tmp/px4_sitl_demo.log 2>&1 &
PX4_PID=$!

# Step 3: Wait for MAVLink to be available (EKF2 alignment)
echo "[run_demo] Waiting for PX4 EKF2 alignment (up to 30s)..."
python3 << 'PYEOF'
import sys, time
sys.path.insert(0, '.')
import pymavlink.mavutil as mavutil
m = mavutil.mavlink_connection('udp:127.0.0.1:14550')
hb = m.wait_heartbeat(timeout=15)
if hb is None:
    print("[run_demo] ERROR: No heartbeat from PX4")
    sys.exit(1)

t_wait = time.monotonic()
while time.monotonic() - t_wait < 20.0:
    msg = m.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=1.0)
    if msg:
        print(f"[run_demo] EKF2 aligned: x={msg.x:.3f}m")
        sys.exit(0)
print("[run_demo] WARNING: LOCAL_POSITION_NED not received — proceeding anyway")
sys.exit(0)
PYEOF

echo "[run_demo] Starting demonstration flight..."
echo ""

# Step 4: Run the demo flight
cd "$REPO_DIR"
python3 inject_outage.py
DEMO_EXIT=$?

# Step 5: Report result
echo ""
if [ $DEMO_EXIT -eq 0 ]; then
    LATEST_REPORT=$(ls -t "$DASHBOARD_DIR"/demo_report_*.html 2>/dev/null | head -1)
    echo "============================================================"
    echo "  DEMO PASS"
    echo "  Report: $LATEST_REPORT"
    echo "============================================================"
else
    echo "============================================================"
    echo "  DEMO FAIL — check output above"
    echo "  PX4 log: /tmp/px4_sitl_demo.log"
    echo "============================================================"
fi

# Cleanup
unset LD_PRELOAD
exit $DEMO_EXIT
