#!/usr/bin/env python3.12
"""
simulation/run_mission.py
MicroMind Two-Vehicle GPS Denial Demo — Phase B Mission Script

Command line interface:
    python3.12 simulation/run_mission.py \
      [--gps-denial-time 60] \
      [--loops 2] \
      [--vio-outage]

Vehicle B = MicroMind equipped  (instance 0, MAVLink port 14540, sysid 1)
Vehicle A = INS-only plain UAV  (instance 1, MAVLink port 14541, sysid 2)
Both vehicles fly offset ellipses in Baylands world.
"""

import argparse
import json
import math
import subprocess
import sys
import threading
import time

from pymavlink import mavutil

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELLIPSE_CX, ELLIPSE_CY = 0.0, 20.0      # world frame centre (ENU metres); near default Baylands spawn
ELLIPSE_A  = 150.0    # semi-axis X (metres)
ELLIPSE_B  = 80.0     # semi-axis Y (metres)
ALTITUDE_M = 100.0    # AGL (metres)
SPEED_MPS  = 13.89    # 50 km/h

VEH_A_CY_OFFSET = 17.5   # Vehicle A at Y=37.5 (20 + 17.5); positive, north of B

STIM300_ARW  = 0.15   # deg/sqrt(hr) — STIM300 spec
DRIFT_SCALE  = 5.0    # demonstration multiplier (5x for visual clarity)

PORT_VEH_B = 14540    # Vehicle B (MicroMind, instance 0)
PORT_VEH_A = 14541    # Vehicle A (INS-only, instance 1)

OVERLAY_FILE_A = '/tmp/mm_overlay_a.json'
OVERLAY_FILE_B = '/tmp/mm_overlay_b.json'
MARKER_TRAIL_LIFETIME_S = 5.0

# ---------------------------------------------------------------------------
# Threading primitives
# ---------------------------------------------------------------------------

barrier          = threading.Barrier(2)
start_time       = None
_start_time_lock = threading.Lock()
gps_denial_event = threading.Event()
mission_complete = threading.Event()
abort_event      = threading.Event()
_overlay_lock    = threading.Lock()

# VIO outage state for Vehicle B
_vio_paused      = False
_vio_paused_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def ellipse_waypoints(cx, cy, a, b, n=8):
    pts = []
    for i in range(n):
        theta = 2 * math.pi * i / n  # counter-clockwise
        x = cx + a * math.cos(theta)
        y = cy + b * math.sin(theta)
        pts.append((x, y))
    return pts


def enu_to_ned_setpoint(enu_x, enu_y, spawn_enu_x, spawn_enu_y, alt_m):
    ned_n = enu_x - spawn_enu_x   # ENU X = NED North
    ned_e = enu_y - spawn_enu_y   # ENU Y = NED East
    ned_d = -alt_m                # ENU Z up = NED D down (negative)
    return ned_n, ned_e, ned_d


def compute_drift_offset(t_since_denial):
    arw = STIM300_ARW * DRIFT_SCALE * math.pi / 180.0 / 60.0  # rad/sqrt(s)
    drift_m = SPEED_MPS * arw * (t_since_denial ** 1.5) / math.sqrt(3)
    return drift_m

# ---------------------------------------------------------------------------
# Gazebo trail markers
# ---------------------------------------------------------------------------

def publish_trail_marker(marker_id, ned_n, ned_e, ned_d, colour):
    # colour: "red" for Vehicle A, "green" for Vehicle B
    r, g, b = (1, 0, 0) if colour == "red" else (0, 1, 0)
    req = json.dumps({
        "id": marker_id,
        "action": 0,   # ADD_MODIFY
        "type": 4,     # LINE_STRIP (or SPHERE = 1 for point markers)
        "lifetime": {"sec": int(MARKER_TRAIL_LIFETIME_S), "nsec": 0},
        "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
        "material": {"ambient": {"r": r, "g": g, "b": b, "a": 1.0}},
        "point": [{"x": ned_e, "y": ned_n, "z": -ned_d}]  # Gazebo ENU from NED
    })
    subprocess.run(
        ["gz", "service", "-s", "/marker",
         "--reqtype", "gz.msgs.Marker",
         "--reptype", "gz.msgs.Boolean",
         "--timeout", "500", "--req", req],
        capture_output=True
    )

# ---------------------------------------------------------------------------
# MAVLink helpers
# ---------------------------------------------------------------------------

def _send_position_setpoint(mav, target_system, target_component,
                             ned_n, ned_e, ned_d):
    mav.mav.set_position_target_local_ned_send(
        int(time.monotonic() * 1000) & 0xFFFFFFFF,
        target_system, target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,  # type_mask — position only
        ned_n, ned_e, ned_d,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0
    )


def _start_heartbeat(mav, stop_event):
    """Send GCS heartbeat at 1 Hz so PX4 clears 'No connection to the GCS' check."""
    while not stop_event.is_set():
        mav.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0,
            mavutil.mavlink.MAV_STATE_ACTIVE,
        )
        stop_event.wait(1.0)


def _start_setpoint_stream(conn, target_pos, stop_event, rate_hz=20):
    """
    Stream SET_POSITION_TARGET_LOCAL_NED at `rate_hz` until stop_event is set.
    Holds the vehicle at `target_pos` (x, y, z) during ARM/OFFBOARD ACK waits.
    Called as a daemon thread — exits immediately when stop_event fires.
    OI-35 fix: prevents PX4 timing out the OFFBOARD setpoint stream during
    the ~10s ARM + OFFBOARD ACK blocking wait in _arm_and_offboard().
    """
    def _loop():
        interval = 1.0 / rate_hz
        while not stop_event.is_set():
            conn.mav.set_position_target_local_ned_send(
                0,                              # time_boot_ms (ignored by PX4)
                conn.target_system,
                conn.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111111000,             # position only, ignore vel/acc/yaw_rate
                target_pos[0],                  # x (North, m)
                target_pos[1],                  # y (East, m)
                target_pos[2],                  # z (Down, m — negative = up)
                0, 0, 0,                        # vx, vy, vz
                0, 0, 0,                        # afx, afy, afz
                0, 0                            # yaw, yaw_rate
            )
            time.sleep(interval)
    t = threading.Thread(target=_loop, daemon=True, name="setpoint_stream_a")
    t.start()
    return t


def _arm_and_offboard(mav, target_system, target_component, label):
    # ARM
    mav.mav.command_long_send(
        target_system, target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    ack = mav.recv_match(type='COMMAND_ACK', blocking=True, timeout=5.0)
    if not ack or ack.result != 0:
        result_str = str(ack.result) if ack else 'TIMEOUT'
        print(f"[{label}] ARM rejected — result={result_str}")
        return False
    print(f"[{label}] ARMED")

    # OFFBOARD (mode 6, base_mode=209)
    mav.mav.command_long_send(
        target_system, target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0, 209, 6, 0, 0, 0, 0, 0
    )
    ack = mav.recv_match(type='COMMAND_ACK', blocking=True, timeout=5.0)
    if not ack or ack.result != 0:
        result_str = str(ack.result) if ack else 'TIMEOUT'
        print(f"[{label}] OFFBOARD rejected — result={result_str}")
        return False
    print(f"[{label}] OFFBOARD ENGAGED")
    return True


def _climb_to_altitude(mav, target_system, target_component,
                        spawn_ned_n, spawn_ned_e, alt_m, label,
                        timeout_s=90.0):
    """Stream NED z=-alt_m setpoint at 20 Hz until 95% altitude reached."""
    t_start = time.monotonic()
    dt = 0.05  # 20 Hz
    while time.monotonic() - t_start < timeout_s:
        _send_position_setpoint(mav, target_system, target_component,
                                spawn_ned_n, spawn_ned_e, -alt_m)
        msg = mav.recv_match(type='LOCAL_POSITION_NED',
                             blocking=True, timeout=dt)
        if msg and -msg.z >= alt_m * 0.95:
            print(f"[{label}] Altitude {-msg.z:.1f} m reached")
            return True
    print(f"[{label}] CLIMB TIMEOUT after {timeout_s:.0f}s")
    return False


def _read_local_pos(mav, timeout=0.05):
    return mav.recv_match(type='LOCAL_POSITION_NED',
                          blocking=True, timeout=timeout)


def wait_ekf2_ready(conn, label, timeout=60):
    """Wait for EKF2 to align by polling LOCAL_POSITION_NED."""
    print(f"[{label}] Waiting for EKF2 alignment (up to {timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = conn.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=1.0)
        if msg is not None:
            print(f"[{label}] EKF2 aligned: x={msg.x:.3f}m")
            return True
    print(f"[{label}] EKF2 alignment timeout — aborting")
    return False

# ---------------------------------------------------------------------------
# Spawn points in ENU world frame (θ=0° of each ellipse)
# ---------------------------------------------------------------------------

SPAWN_B_ENU = (0.0, 0.0)    # Vehicle B default PX4 spawn ≈ world origin
SPAWN_A_ENU = (0.0, 5.0)    # Vehicle A spawn at PX4_GZ_MODEL_POSE="0,5,0.5"

# ---------------------------------------------------------------------------
# Vehicle B — MicroMind equipped (instance 0, port 14540)
# ---------------------------------------------------------------------------

def mission_vehicle_b(args):
    global start_time, _vio_paused

    label     = "VEH B"
    waypoints = ellipse_waypoints(ELLIPSE_CX, ELLIPSE_CY, ELLIPSE_A, ELLIPSE_B)

    # 1. Connect
    print(f"[{label}] Connecting to udp:127.0.0.1:{PORT_VEH_B}...")
    mav = mavutil.mavlink_connection(f'udp:127.0.0.1:{PORT_VEH_B}')
    hb  = mav.wait_heartbeat(timeout=30)
    if hb is None:
        print(f"[{label}] ABORT — no heartbeat within 30s")
        return
    target_system    = mav.target_system
    target_component = mav.target_component
    print(f"[{label}] Heartbeat sysid={target_system}")

    # 1b. Start GCS heartbeat thread — required to clear PX4 'No connection to GCS' pre-arm check
    _hb_stop = threading.Event()
    _hb_thread = threading.Thread(target=_start_heartbeat, args=(mav, _hb_stop), daemon=True)
    _hb_thread.start()

    # 2. Pre-stream setpoints at spawn position to bootstrap EKF2 — 5.0s at 20 Hz
    for _ in range(100):
        _send_position_setpoint(mav, target_system, target_component,
                                0.0, 0.0, 0.0)
        time.sleep(0.05)

    # 3. Wait for EKF2 alignment (aligns quickly once setpoint stream is active)
    if not wait_ekf2_ready(mav, label, timeout=30):
        _hb_stop.set()
        abort_event.set()
        return

    # 3. Barrier — synchronise start with Vehicle A
    barrier.wait()
    with _start_time_lock:
        if start_time is None:
            start_time = time.monotonic()
    t0 = start_time

    # 4. Pre-stream setpoints before engaging OFFBOARD — 2.0s at 20 Hz (matches inject_outage.py)
    for _ in range(40):
        _send_position_setpoint(mav, target_system, target_component,
                                0.0, 0.0, -ALTITUDE_M)
        time.sleep(0.05)

    # 5. ARM → OFFBOARD → climb
    if not _arm_and_offboard(mav, target_system, target_component, label):
        _hb_stop.set()
        abort_event.set()
        return

    if not _climb_to_altitude(mav, target_system, target_component,
                               0.0, 0.0, ALTITUDE_M, label):
        _hb_stop.set()
        abort_event.set()
        return

    # 6. Fly ellipse laps
    wp_idx              = 0
    lap_count           = 0
    marker_id           = 200
    t_last_marker       = 0.0
    events              = []
    vio_outage_started  = False
    vio_outage_end_t    = None
    dt                  = 0.05  # 20 Hz

    while lap_count < args.loops:
        t_now   = time.monotonic()
        elapsed = t_now - t0

        curr_wp = waypoints[wp_idx]
        next_wp = waypoints[(wp_idx + 1) % len(waypoints)]  # noqa: F841

        ned_n, ned_e, ned_d = enu_to_ned_setpoint(
            curr_wp[0], curr_wp[1],
            SPAWN_B_ENU[0], SPAWN_B_ENU[1], ALTITUDE_M
        )

        _send_position_setpoint(mav, target_system, target_component,
                                ned_n, ned_e, ned_d)

        # 7. Optional VIO outage at T + gps_denial_time + 30s
        # Implemented via MAVLink PARAM_SET to PX4 instance 0 (Vehicle B)
        # EKF2_EV_CTRL = 0 disables all external vision fusion (equivalent to VIO denial)
        if args.vio_outage and not vio_outage_started:
            if elapsed >= args.gps_denial_time + 30.0:
                # Disable EV (external vision) fusion in PX4 EKF2
                mav.mav.param_set_send(
                    target_system,
                    target_component,
                    b'EKF2_EV_CTRL\x00\x00\x00',   # 16 bytes, null-padded
                    0.0,                            # value 0 = disable all EV
                    mavutil.mavlink.MAV_PARAM_TYPE_INT32
                )
                with _vio_paused_lock:
                    _vio_paused = True
                vio_outage_started = True
                vio_outage_end_t   = t_now + 10.0
                events.append(f"T+{elapsed:.0f}s VIO_OUTAGE_START")
                print(f"[{label}] VIO outage injected at T+{elapsed:.1f}s — EKF2_EV_CTRL=0 sent")

        if vio_outage_started and vio_outage_end_t is not None:
            if t_now >= vio_outage_end_t:
                # Re-enable EV fusion
                mav.mav.param_set_send(
                    target_system,
                    target_component,
                    b'EKF2_EV_CTRL\x00\x00\x00',
                    15.0,    # restore default: bits 0-3 all set (pos+vel+hgt+yaw)
                    mavutil.mavlink.MAV_PARAM_TYPE_INT32
                )
                with _vio_paused_lock:
                    _vio_paused = False
                vio_outage_end_t = None
                events.append(f"T+{elapsed:.0f}s VIO_OUTAGE_END")
                print(f"[{label}] VIO outage ended — EKF2_EV_CTRL=15 restored")

        with _vio_paused_lock:
            vio_mode_str = "OUTAGE" if _vio_paused else "NOMINAL"

        # Actual position from LOCAL_POSITION_NED
        msg  = _read_local_pos(mav, timeout=dt)
        ac_n = msg.x if msg else ned_n
        ac_e = msg.y if msg else ned_e
        ac_d = msg.z if msg else ned_d

        # Waypoint advance criterion: distance to current NED setpoint < 15m
        dist = math.sqrt((ac_n - ned_n) ** 2 + (ac_e - ned_e) ** 2)
        if dist < 15.0:
            wp_idx += 1
            if wp_idx >= len(waypoints):
                wp_idx = 0
                lap_count += 1
                events.append(f"T+{elapsed:.0f}s LAP_{lap_count}_COMPLETE")
                print(f"[{label}] Lap {lap_count} complete at T+{elapsed:.1f}s")

        # 6. Overlay JSON at every 20Hz iteration
        overlay = {
            "vehicle":   "B",
            "label":     "MICROMIND",
            "vio_mode":  vio_mode_str,
            "gps_denied": False,
            "drift_m":   0.0,
            "sp_n": round(ned_n, 2), "sp_e": round(ned_e, 2),
            "sp_d": round(ned_d, 2),
            "ac_n": round(ac_n, 2),  "ac_e": round(ac_e, 2),
            "ac_d": round(ac_d, 2),
            "sp_hz":     20.0,
            "events":    events[-6:],
            "t_elapsed": round(elapsed, 1),
        }
        with _overlay_lock:
            with open(OVERLAY_FILE_B, 'w') as fh:
                json.dump(overlay, fh)

        # Trail marker (green) at 0.5s interval
        if t_now - t_last_marker >= 0.5:
            publish_trail_marker(marker_id, ned_n, ned_e, ned_d, "green")
            marker_id += 1
            if marker_id > 250:
                marker_id = 200
            t_last_marker = t_now

        time.sleep(dt)

    # 8. Mission complete — land
    mission_complete.set()
    print(f"[{label}] Mission complete — landing")
    mav.mav.command_long_send(
        target_system, target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0
    )

# ---------------------------------------------------------------------------
# Vehicle A — INS-only plain UAV (instance 1, port 14541)
# ---------------------------------------------------------------------------

def mission_vehicle_a(args):
    global start_time

    label     = "VEH A"
    waypoints = ellipse_waypoints(
        ELLIPSE_CX, ELLIPSE_CY + VEH_A_CY_OFFSET, ELLIPSE_A, ELLIPSE_B
    )

    # 1. Connect
    print(f"[{label}] Connecting to udp:127.0.0.1:{PORT_VEH_A}...")
    mav = mavutil.mavlink_connection(f'udp:127.0.0.1:{PORT_VEH_A}')
    hb  = mav.wait_heartbeat(timeout=30)
    if hb is None:
        print(f"[{label}] ABORT — no heartbeat within 30s")
        return
    target_system    = mav.target_system
    target_component = mav.target_component
    print(f"[{label}] Heartbeat sysid={target_system}")

    # 1b. Start GCS heartbeat thread — required to clear PX4 'No connection to GCS' pre-arm check
    _hb_stop = threading.Event()
    _hb_thread = threading.Thread(target=_start_heartbeat, args=(mav, _hb_stop), daemon=True)
    _hb_thread.start()

    # 2. Pre-stream setpoints at spawn position to bootstrap EKF2 — 5.0s at 20 Hz
    for _ in range(100):
        _send_position_setpoint(mav, target_system, target_component,
                                0.0, 0.0, 0.0)
        time.sleep(0.05)

    # 3. Wait for EKF2 alignment (aligns quickly once setpoint stream is active)
    if not wait_ekf2_ready(mav, label, timeout=30):
        _hb_stop.set()
        abort_event.set()
        return

    # 3. Barrier — synchronise start with Vehicle B
    barrier.wait()
    with _start_time_lock:
        if start_time is None:
            start_time = time.monotonic()
    t0 = start_time

    # 4. Pre-stream setpoints before engaging OFFBOARD — 2.0s at 20 Hz (matches inject_outage.py)
    for _ in range(40):
        _send_position_setpoint(mav, target_system, target_component,
                                0.0, 0.0, -ALTITUDE_M)
        time.sleep(0.05)

    # 5. ARM → OFFBOARD → climb
    # OI-35 fix: continuous setpoint stream during ARM/OFFBOARD ACK waits (~10s)
    # _arm_and_offboard() blocks on two recv_match() calls (5s each); without this
    # thread PX4 times out the OFFBOARD setpoint stream and drops OFFBOARD mode.
    _sp_stop = threading.Event()
    _sp_thread = _start_setpoint_stream(
        mav,
        target_pos=[0.0, 0.0, -ALTITUDE_M],   # matches pre-arm setpoints at step 4
        stop_event=_sp_stop
    )
    if not _arm_and_offboard(mav, target_system, target_component, label):
        _sp_stop.set()
        _sp_thread.join(timeout=1.0)
        _hb_stop.set()
        abort_event.set()
        return
    _sp_stop.set()
    _sp_thread.join(timeout=1.0)   # confirm clean exit before _climb_to_altitude takes over

    if not _climb_to_altitude(mav, target_system, target_component,
                               0.0, 0.0, ALTITUDE_M, label):
        _hb_stop.set()
        abort_event.set()
        return

    # 6. Fly ellipse laps
    wp_idx         = 0
    lap_count      = 0
    marker_id      = 100
    t_last_overlay = 0.0
    events         = []
    gps_denied     = False
    drift_m        = 0.0
    t_denial       = None
    dt             = 0.05  # 20 Hz

    while lap_count < args.loops:
        t_now   = time.monotonic()
        elapsed = t_now - t0

        curr_wp = waypoints[wp_idx]
        next_wp = waypoints[(wp_idx + 1) % len(waypoints)]

        # 6. GPS denial trigger at T + gps_denial_time
        if not gps_denied and elapsed >= args.gps_denial_time:
            gps_denial_event.set()
            gps_denied = True
            t_denial   = t_now
            events.append(f"T+{elapsed:.0f}s GPS_DENIAL_START")
            print(f"[VEH A] GPS denial active at T+{elapsed:.1f}s")

            # Disable GPS on instance 1 via PARAM_SET EKF2_GPS_CTRL = 0
            param_id = b'EKF2_GPS_CTRL\x00\x00\x00'  # padded to 16 bytes
            mav.mav.param_set_send(
                target_system, target_component,
                param_id,
                0.0,
                mavutil.mavlink.MAV_PARAM_TYPE_INT32
            )

        # 7. Compute INS drift offset after GPS denial
        drift_n = 0.0
        drift_e = 0.0
        if gps_denied and t_denial is not None:
            t_since = t_now - t_denial
            drift_m = compute_drift_offset(t_since)
            heading = math.atan2(next_wp[1] - curr_wp[1],
                                 next_wp[0] - curr_wp[0])
            perp    = heading + math.pi / 2   # perpendicular = 90 degrees left
            drift_n = drift_m * math.cos(perp)
            drift_e = drift_m * math.sin(perp)

        # NED setpoint: base waypoint + drift
        base_n, base_e, ned_d = enu_to_ned_setpoint(
            curr_wp[0], curr_wp[1],
            SPAWN_A_ENU[0], SPAWN_A_ENU[1], ALTITUDE_M
        )
        ned_n = base_n + drift_n
        ned_e = base_e + drift_e

        _send_position_setpoint(mav, target_system, target_component,
                                ned_n, ned_e, ned_d)

        # Actual position
        msg  = _read_local_pos(mav, timeout=dt)
        ac_n = msg.x if msg else ned_n
        ac_e = msg.y if msg else ned_e
        ac_d = msg.z if msg else ned_d

        # Waypoint advance criterion against undrifted base setpoint
        dist = math.sqrt((ac_n - base_n) ** 2 + (ac_e - base_e) ** 2)
        if dist < 15.0:
            wp_idx += 1
            if wp_idx >= len(waypoints):
                wp_idx = 0
                lap_count += 1
                events.append(f"T+{elapsed:.0f}s LAP_{lap_count}_COMPLETE")
                print(f"[{label}] Lap {lap_count} complete at T+{elapsed:.1f}s")

        # 8. Overlay + trail marker at 0.5s interval
        if t_now - t_last_overlay >= 0.5:
            overlay = {
                "vehicle":    "A",
                "label":      "INS ONLY",
                "vio_mode":   "N/A",
                "gps_denied": gps_denied,
                "drift_m":    round(drift_m, 3),
                "sp_n": round(ned_n, 2), "sp_e": round(ned_e, 2),
                "sp_d": round(ned_d, 2),
                "ac_n": round(ac_n, 2),  "ac_e": round(ac_e, 2),
                "ac_d": round(ac_d, 2),
                "sp_hz":      20.0,
                "events":     events[-6:],
                "t_elapsed":  round(elapsed, 1),
            }
            with _overlay_lock:
                with open(OVERLAY_FILE_A, 'w') as fh:
                    json.dump(overlay, fh)

            # 9. Trail marker (red)
            publish_trail_marker(marker_id, ned_n, ned_e, ned_d, "red")
            marker_id += 1
            if marker_id > 150:
                marker_id = 100
            t_last_overlay = t_now

        time.sleep(dt)

    print(f"[{label}] Mission complete")

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='MicroMind two-vehicle GPS denial demo mission'
    )
    p.add_argument(
        '--gps-denial-time', type=float, default=60.0,
        dest='gps_denial_time',
        help='Seconds after mission start to trigger GPS denial on Vehicle A (default: 60)',
    )
    p.add_argument(
        '--loops', type=int, default=2,
        help='Number of ellipse laps per vehicle (default: 2)',
    )
    p.add_argument(
        '--vio-outage', action='store_true', dest='vio_outage',
        help='Inject 10s VIO outage on Vehicle B at T+gps_denial_time+30s',
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    t_a = threading.Thread(target=mission_vehicle_a, args=(args,), daemon=True)
    t_b = threading.Thread(target=mission_vehicle_b, args=(args,), daemon=True)

    t_b.start()
    t_a.start()

    t_a.join()
    t_b.join()

    if abort_event.is_set():
        print("[MISSION] ABORT — EKF2/arming failure on one or both vehicles.")
        sys.exit(1)
    elif mission_complete.is_set():
        print("[MISSION] PASS — two-vehicle GPS denial demo complete.")
        sys.exit(0)
    else:
        print("[MISSION] FAIL — mission did not complete cleanly.")
        sys.exit(1)


if __name__ == '__main__':
    main()
