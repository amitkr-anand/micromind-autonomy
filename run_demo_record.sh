#!/bin/bash
# run_demo_record.sh
# MicroMind Pre-HIL — Recorded Demo
# Starts recording, runs ./run_demo.sh, stops recording, re-encodes to yuv420p

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RAW_MP4="$REPO_DIR/dashboard/micromind_demo_raw_${TIMESTAMP}.mp4"
FINAL_MP4="$REPO_DIR/dashboard/micromind_demo_recording_${TIMESTAMP}.mp4"

echo ""
echo "============================================================"
echo "  MicroMind — Recorded OEM Demonstration"
echo "  $(date '+%Y-%m-%d %H:%M:%S IST')"
echo "============================================================"
echo ""

# Start recording
echo "[recorder] Starting screen capture..."
nohup ffmpeg -f x11grab -r 25 -s 1920x1080 -i :1 \
  -vcodec libx264 -preset ultrafast -crf 28 \
  "$RAW_MP4" -y \
  < /dev/null > /tmp/ffmpeg_demo_record.log 2>&1 &
FFMPEG_PID=$!
disown $FFMPEG_PID
sleep 3

# Verify recording started
if ! ps -p $FFMPEG_PID > /dev/null 2>&1; then
    echo "[recorder] ERROR: ffmpeg failed to start"
    cat /tmp/ffmpeg_demo_record.log | tail -5
    exit 1
fi
echo "[recorder] Recording active (PID=$FFMPEG_PID)"
echo ""

# Run the demo
cd "$REPO_DIR"
./run_demo.sh
DEMO_EXIT=$?

# Stop recording
echo ""
echo "[recorder] Stopping recording..."
sleep 2
kill -INT $FFMPEG_PID 2>/dev/null
wait $FFMPEG_PID 2>/dev/null || true
sleep 2

# Re-encode to yuv420p for VLC compatibility
echo "[recorder] Re-encoding to yuv420p..."
ffmpeg -i "$RAW_MP4" \
  -vcodec libx264 -preset fast -crf 23 \
  -pix_fmt yuv420p \
  "$FINAL_MP4" -y 2>/dev/null

# Remove raw file
rm -f "$RAW_MP4"

echo "[recorder] Recording saved: $FINAL_MP4"
ls -lh "$FINAL_MP4"
echo ""

if [ $DEMO_EXIT -eq 0 ]; then
    echo "DEMO + RECORDING: PASS"
else
    echo "DEMO FAIL — check output above"
fi

exit $DEMO_EXIT
