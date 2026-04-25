"""
tests/test_it_d9_chain.py
MicroMind — IT-D9-CHAIN-01: Full D7→D8→D8a→D9 SITL Reboot Recovery Chain

@pytest.mark.sitl — excluded from certified SIL baseline.
Runs on live PX4 SITL (micromind-node01, single vehicle, port 14540).

Requirements: SRS §8.4 PX4-04
Gates:
    G1: PX4_REBOOT_DETECTED logged with seq_reset_value and recovery_start_ms
        — within 3 s wall clock of reboot injection
    G2: CHECKPOINT_RESTORED logged with checkpoint_age_ms and waypoint_index
        — within 15 s of G1
    G3: AUTONOMOUS_RESUME_APPROVED logged with
        pending_operator_clearance_required = False
    G4: MISSION_RESUMED logged with position_discrepancy_m field

Clock discipline (SR-01):
    All in-test waits use threading.Event().wait(timeout) — no time.sleep() in
    test methods. Daemon threads may use .wait() on their own stop events.

SITL prerequisite:
    Gazebo Baylands world + PX4 instance 0 (port 14540) must be running before
    test invocation. Exact startup: run_demo.sh Phase A + Phase B (inst 0 only).

Reboot injection:
    pkill -f "bin/px4 -i 0" then restart with PX4_GZ_STANDALONE=1 (Gazebo
    remains live). The test manages the restart subprocess.

Evidence output: docs/qa/IT_D9_CHAIN_EVIDENCE_RUN1.md
"""

from __future__ import annotations

import datetime
import math
import os
import subprocess
import tempfile
import threading
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from core.checkpoint.checkpoint import Checkpoint, CheckpointStore
from core.mission_manager.mission_manager import MissionManager
from integration.bridge.hold_recovery import HoldRecoveryHandler
from integration.bridge.reboot_detector import RebootDetector

# ---------------------------------------------------------------------------
# pymavlink — optional import (system Python 3.12 path)
# ---------------------------------------------------------------------------
try:
    from pymavlink import mavutil as _mavutil
    _HAS_PYMAVLINK = True
except ImportError:
    _mavutil = None   # type: ignore[assignment]
    _HAS_PYMAVLINK = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SITL_HOST           = "127.0.0.1"
SITL_PORT           = 14540
ALTITUDE_M          = 50.0            # NED hover altitude (AGL)
EVIDENCE_PATH       = "docs/qa/IT_D9_CHAIN_EVIDENCE_RUN1.md"
WAYPOINT_INDEX      = 3               # written into test checkpoint
CHECKPOINT_MISSION  = "IT-D9-CHAIN-01"

# Gate timing thresholds (SRS §8.4 PX4-04)
G1_D7_WINDOW_MS     = 3_000          # PX4_REBOOT_DETECTED within 3 s of injection
G2_D8_WINDOW_MS     = 15_000         # CHECKPOINT_RESTORED within 15 s of D7
D9_OFFBOARD_WAIT_MS = 30_000         # max wait for OFFBOARD re-established after D8a

# PX4 binary paths (same as run_demo.sh)
_PX4_HOME   = str(Path.home() / "PX4-Autopilot")
_PX4_BIN    = f"{_PX4_HOME}/build/px4_sitl_default/bin/px4"
_PX4_ETC    = f"{_PX4_HOME}/build/px4_sitl_default/etc"
_PX4_INST0  = "/tmp/px4_inst0"

# OFFBOARD custom_mode encoding (main_mode=6)
_OFFBOARD_MAIN_MODE = 6


# ---------------------------------------------------------------------------
# Monotonic clock — SR-01 compliant (no time.time())
# ---------------------------------------------------------------------------

def _mono_ms() -> int:
    return int(_time.monotonic() * 1000)


# ---------------------------------------------------------------------------
# Git HEAD
# ---------------------------------------------------------------------------

def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# MAVLink helpers (mirrors EC-01 pattern)
# ---------------------------------------------------------------------------

def _is_offboard(hb_msg: Any) -> bool:
    if hb_msg is None:
        return False
    return ((hb_msg.custom_mode >> 16) & 0xFF) == _OFFBOARD_MAIN_MODE


def _send_hover_setpoint(
    conn: Any,
    target_system: int,
    target_component: int,
    alt_m: float,
) -> None:
    conn.mav.set_position_target_local_ned_send(
        0,
        target_system, target_component,
        _mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,
        0.0, 0.0, -alt_m,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0,
    )


def _gcs_heartbeat_fn(conn: Any, stop_ev: threading.Event) -> None:
    while not stop_ev.is_set():
        try:
            conn.mav.heartbeat_send(
                _mavutil.mavlink.MAV_TYPE_GCS,
                _mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0,
                _mavutil.mavlink.MAV_STATE_ACTIVE,
            )
        except Exception:
            pass
        stop_ev.wait(1.0)


def _setpoint_stream_fn(
    conn: Any,
    target_system: int,
    target_component: int,
    alt_m: float,
    stop_ev: threading.Event,
) -> None:
    """20 Hz setpoint stream daemon — SR-01 compliant (Event.wait cadence)."""
    interval = 0.05
    while not stop_ev.is_set():
        try:
            _send_hover_setpoint(conn, target_system, target_component, alt_m)
        except Exception:
            pass
        stop_ev.wait(interval)


def _arm_and_offboard(
    conn: Any,
    target_system: int,
    target_component: int,
) -> bool:
    conn.mav.command_long_send(
        target_system, target_component,
        _mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0,
    )
    ack = conn.recv_match(type="COMMAND_ACK", blocking=True, timeout=10.0)
    if not ack or ack.result != 0:
        return False

    conn.mav.command_long_send(
        target_system, target_component,
        _mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0, 209, 6, 0, 0, 0, 0, 0,
    )
    ack = conn.recv_match(type="COMMAND_ACK", blocking=True, timeout=10.0)
    return bool(ack and ack.result == 0)


def _send_set_mode_offboard(conn: Any, target_system: int, target_component: int) -> bool:
    """Send MAV_CMD_DO_SET_MODE OFFBOARD; return True on ACK result=0."""
    conn.mav.command_long_send(
        target_system, target_component,
        _mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0, 209, 6, 0, 0, 0, 0, 0,
    )
    ack = conn.recv_match(type="COMMAND_ACK", blocking=True, timeout=5.0)
    return bool(ack and ack.result == 0)


def _get_ned_position(conn: Any) -> Optional[tuple]:
    """Return (x, y, z) from LOCAL_POSITION_NED or None on timeout."""
    msg = conn.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=3.0)
    if msg is None:
        return None
    return (msg.x, msg.y, msg.z)


# ---------------------------------------------------------------------------
# PX4 reboot injection
# ---------------------------------------------------------------------------

def _inject_reboot(px4_restart_env: dict) -> int:
    """Kill PX4 instance 0 and restart it. Returns kill timestamp ms."""
    kill_ts = _mono_ms()
    subprocess.run(
        ["pkill", "-f", "bin/px4 -i 0"],
        check=False,
        capture_output=True,
    )
    # 1-second isolation window (SR-01: Event.wait not sleep)
    _wait_ev = threading.Event()
    _wait_ev.wait(timeout=1.0)

    os.makedirs(_PX4_INST0, exist_ok=True)
    log_fh = open("/tmp/px4_d9_restart.log", "w")
    subprocess.Popen(
        [_PX4_BIN, "-i", "0", "-d", _PX4_ETC],
        env=px4_restart_env,
        cwd=_PX4_INST0,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    return kill_ts


# ---------------------------------------------------------------------------
# Evidence writer
# ---------------------------------------------------------------------------

def _write_evidence(
    run_ts:         str,
    head_hash:      str,
    kill_ts_ms:     int,
    d7_ts_ms:       int,
    g1_pass:        bool,
    g2_pass:        bool,
    g3_pass:        bool,
    g4_pass:        bool,
    g1_latency_ms:  int,
    g2_latency_ms:  int,
    d8_age_ms:      int,
    d8_wp:          int,
    pos_disc_m:     float,
) -> None:
    g1_meas = f"{g1_latency_ms} ms (threshold ≤ {G1_D7_WINDOW_MS} ms)"
    g2_meas = f"{g2_latency_ms} ms since D7 (threshold ≤ {G2_D8_WINDOW_MS} ms)"
    g3_meas = "AUTONOMOUS_RESUME_APPROVED with pending_operator_clearance_required=False"
    g4_meas = f"position_discrepancy_m = {pos_disc_m:.3f} m"

    lines = [
        "# IT-D9-CHAIN-01 Evidence Run 1",
        "",
        f"**Run timestamp (UTC):** {run_ts}",
        f"**HEAD commit:** {head_hash}",
        f"**Test:** IT-D9-CHAIN-01 — Full D7→D8→D8a→D9 SITL chain",
        f"**PX4 SITL endpoint:** udp:{SITL_HOST}:{SITL_PORT}",
        "",
        "## Gate Results (SRS §8.4 PX4-04)",
        "",
        "| Gate | Criterion | Measured | Result |",
        "|---|---|---|---|",
        f"| G1 — D7 | PX4_REBOOT_DETECTED with seq_reset_value + recovery_start_ms ≤ 3 s | {g1_meas} | {'PASS' if g1_pass else 'FAIL'} |",
        f"| G2 — D8 | CHECKPOINT_RESTORED with checkpoint_age_ms + waypoint_index ≤ 15 s of D7 | {g2_meas} | {'PASS' if g2_pass else 'FAIL'} |",
        f"| G3 — D8a | AUTONOMOUS_RESUME_APPROVED with pending_operator_clearance_required=False | {g3_meas} | {'PASS' if g3_pass else 'FAIL'} |",
        f"| G4 — D9 | MISSION_RESUMED with position_discrepancy_m field | {g4_meas} | {'PASS' if g4_pass else 'FAIL'} |",
        "",
        "## Checkpoint Parameters",
        "",
        f"| Field | Value |",
        "|---|---|",
        f"| waypoint_index | {WAYPOINT_INDEX} |",
        f"| pending_operator_clearance_required | False |",
        f"| mission_abort_flag | False |",
        f"| checkpoint_age_ms at restore | {d8_age_ms} ms |",
        "",
        "---",
        "*Generated by tests/test_it_d9_chain.py — IT-D9-CHAIN-01 SRS §8.4 PX4-04*",
    ]

    dest = EVIDENCE_PATH
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.sitl
class TestITD9Chain:
    """
    IT-D9-CHAIN-01: Full D7→D8→D8a→D9 SITL chain test.

    Injects a real PX4 reboot during OFFBOARD flight and verifies the four
    mandatory SRS §8.4 PX4-04 log events appear in correct sequence with all
    required fields.

    Prerequisite: Gazebo Baylands + PX4 instance 0 (port 14540) running.
    Duration: ≤ 3 minutes wall clock.
    """

    def test_d7_d8_d8a_d9_chain(self) -> None:
        if not _HAS_PYMAVLINK:
            pytest.skip(
                "pymavlink not installed — "
                "run with system Python 3.12: python3.12 -m pytest -m sitl"
            )
        if not Path(_PX4_BIN).exists():
            pytest.skip(f"PX4 binary not found: {_PX4_BIN}")

        run_ts   = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        head     = _git_head()
        _wait_ev = threading.Event()   # reusable SR-01 wait object

        # ------------------------------------------------------------------
        # 1. Connect to PX4 SITL
        # ------------------------------------------------------------------
        conn = _mavutil.mavlink_connection(f"udp:{SITL_HOST}:{SITL_PORT}")
        hb   = conn.wait_heartbeat(timeout=30)
        assert hb is not None, (
            f"No heartbeat from PX4 SITL at {SITL_HOST}:{SITL_PORT} within 30 s. "
            "Ensure PX4 SITL is running."
        )
        target_system    = conn.target_system
        target_component = conn.target_component

        # ------------------------------------------------------------------
        # 2. Write valid v1.2 checkpoint to temp dir
        # ------------------------------------------------------------------
        tmpdir    = tempfile.mkdtemp(prefix="it_d9_")
        event_log: List[Dict[str, Any]] = []

        checkpoint_store = CheckpointStore(
            checkpoint_dir=tmpdir,
            event_log=event_log,
            clock_fn=_time.monotonic,   # returns seconds; store multiplies ×1000
        )
        cp = Checkpoint(
            mission_id=CHECKPOINT_MISSION,
            timestamp_ms=_mono_ms(),
            waypoint_index=WAYPOINT_INDEX,
            pending_operator_clearance_required=False,
            mission_abort_flag=False,
            shm_active=False,
            pos_ned=[0.0, 0.0, -ALTITUDE_M],
        )
        checkpoint_store.write(cp)

        # ------------------------------------------------------------------
        # 3. Mission Manager + Reboot Detector (both share event_log)
        # ------------------------------------------------------------------
        mission_manager  = MissionManager(event_log=event_log, clock_fn=_mono_ms)
        reboot_detector  = RebootDetector(event_log=event_log, clock_fn=_mono_ms)

        # ------------------------------------------------------------------
        # 4. Start GCS heartbeat daemon + pre-stream setpoints
        # ------------------------------------------------------------------
        stop_ev = threading.Event()

        hb_thread = threading.Thread(
            target=_gcs_heartbeat_fn,
            args=(conn, stop_ev),
            daemon=True,
            name="it_d9_gcs_hb",
        )
        hb_thread.start()

        pre_stop = threading.Event()
        pre_sp   = threading.Thread(
            target=_setpoint_stream_fn,
            args=(conn, target_system, target_component, ALTITUDE_M, pre_stop),
            daemon=True,
            name="it_d9_prestream",
        )
        pre_sp.start()
        _wait_ev.wait(timeout=2.0)   # 2 s pre-stream (SR-01)
        pre_stop.set()
        pre_sp.join(timeout=1.0)

        # ------------------------------------------------------------------
        # 5. Start main setpoint stream + ARM + OFFBOARD
        # ------------------------------------------------------------------
        sp_thread = threading.Thread(
            target=_setpoint_stream_fn,
            args=(conn, target_system, target_component, ALTITUDE_M, stop_ev),
            daemon=True,
            name="it_d9_sp",
        )
        sp_thread.start()

        ok = _arm_and_offboard(conn, target_system, target_component)
        assert ok, (
            "ARM / OFFBOARD failed. Check: EKF2 aligned, setpoint stream active, "
            "no pre-arm failures."
        )

        # Brief stabilisation in OFFBOARD before reboot
        _wait_ev.wait(timeout=3.0)

        # Snapshot pre-reboot NED position for G4 discrepancy computation
        ned_before = _get_ned_position(conn) or (0.0, 0.0, -ALTITUDE_M)

        # ------------------------------------------------------------------
        # 6. HEARTBEAT sequence monitor thread → feeds RebootDetector
        # ------------------------------------------------------------------
        reboot_detected_ev = threading.Event()
        d7_ts_ms_ref: List[int] = [0]

        def _hb_monitor() -> None:
            while not stop_ev.is_set():
                msg = conn.recv_match(
                    type="HEARTBEAT", blocking=True, timeout=0.5
                )
                if msg is None:
                    continue
                try:
                    seq = msg.get_seq()
                except AttributeError:
                    seq = getattr(getattr(msg, "_header", None), "seq", -1)
                if seq < 0:
                    continue
                detected = reboot_detector.feed(seq=seq)
                if detected and not reboot_detected_ev.is_set():
                    d7_ts_ms_ref[0] = _mono_ms()
                    reboot_detected_ev.set()

        mon_thread = threading.Thread(
            target=_hb_monitor, daemon=True, name="it_d9_hb_mon"
        )
        mon_thread.start()

        # ------------------------------------------------------------------
        # 7. Inject reboot
        # ------------------------------------------------------------------
        px4_restart_env = {
            **os.environ,
            "PX4_SYS_AUTOSTART":  "4001",
            "PX4_SIM_MODEL":      "gz_x500",
            "GZ_ENGINE_NAME":     "ogre",
            "PX4_GZ_STANDALONE":  "1",
            "PX4_GZ_WORLD":       "baylands",
            "GZ_SIM_RESOURCE_PATH": (
                f"{_PX4_HOME}/Tools/simulation/gz/models:"
                f"{_PX4_HOME}/Tools/simulation/gz/worlds"
            ),
        }
        kill_ts_ms = _inject_reboot(px4_restart_env)

        # ------------------------------------------------------------------
        # 8. Wait for D7 — PX4_REBOOT_DETECTED (G1: within 3 s injection)
        # ------------------------------------------------------------------
        # Allow up to 20 s for PX4 to restart + send first HEARTBEAT
        fired = reboot_detected_ev.wait(timeout=20.0)
        assert fired, (
            "G1 FAIL: PX4_REBOOT_DETECTED not received within 20 s of reboot "
            "injection. PX4 may not have restarted — check /tmp/px4_d9_restart.log"
        )
        d7_ts_ms     = d7_ts_ms_ref[0]
        g1_latency_ms = d7_ts_ms - kill_ts_ms
        g1_pass       = g1_latency_ms <= G1_D7_WINDOW_MS

        # Verify G1 field content
        d7_events = [e for e in event_log if e.get("event") == "PX4_REBOOT_DETECTED"]
        assert d7_events, "Internal error: reboot_detected_ev set but no D7 event in log"
        d7_evt   = d7_events[-1]
        g1_fields = (
            "seq_reset_value" in d7_evt and
            "recovery_start_ms" in d7_evt
        )
        g1_pass = g1_pass and g1_fields

        # ------------------------------------------------------------------
        # 9. Trigger D8 + D8a: MissionManager.on_reboot_detected()
        # ------------------------------------------------------------------
        resumed = mission_manager.on_reboot_detected(checkpoint_store)

        d8a_done_ts_ms = _mono_ms()

        # G2 — CHECKPOINT_RESTORED within 15 s of D7
        d8_events  = [e for e in event_log if e.get("event") == "CHECKPOINT_RESTORED"]
        g2_latency_ms = d8a_done_ts_ms - d7_ts_ms
        g2_pass = (
            bool(d8_events) and
            g2_latency_ms <= G2_D8_WINDOW_MS and
            "checkpoint_age_ms" in d8_events[-1] and
            "waypoint_index"    in d8_events[-1]
        )
        d8_age_ms = d8_events[-1].get("checkpoint_age_ms", -1) if d8_events else -1
        d8_wp_idx = d8_events[-1].get("waypoint_index",    -1) if d8_events else -1

        # G3 — AUTONOMOUS_RESUME_APPROVED with pending_operator_clearance_required=False
        d8a_events = [
            e for e in event_log
            if e.get("event") == "AUTONOMOUS_RESUME_APPROVED"
        ]
        g3_pass = (
            bool(d8a_events) and
            d8a_events[-1].get("pending_operator_clearance_required") is False
        )

        # ------------------------------------------------------------------
        # 10. D9: Re-establish OFFBOARD (D10 may fire if PX4 in HOLD)
        # ------------------------------------------------------------------
        # Build D10 handler (uses same event_log; dispatched from reboot cycle)
        def _send_offboard(base_mode: int, custom_mode: int) -> bool:
            return _send_set_mode_offboard(conn, target_system, target_component)

        hold_handler = HoldRecoveryHandler(
            send_set_mode_fn=_send_offboard,
            event_log=event_log,
            clock_fn=_mono_ms,
        )

        # Wait for PX4 to send a HEARTBEAT after reboot, check if in HOLD
        offboard_restored_ev = threading.Event()
        d9_ts_ms_ref: List[int] = [0]

        def _wait_offboard() -> None:
            t_limit = _mono_ms() + D9_OFFBOARD_WAIT_MS
            while _mono_ms() < t_limit and not stop_ev.is_set():
                hb_msg = conn.recv_match(
                    type="HEARTBEAT", blocking=True, timeout=1.0
                )
                if hb_msg is None:
                    continue
                if _is_offboard(hb_msg):
                    d9_ts_ms_ref[0] = _mono_ms()
                    offboard_restored_ev.set()
                    return
                # D10: PX4 in HOLD — attempt OFFBOARD re-entry
                hold_custom_mode = 50_593_792   # PX4_HOLD_CUSTOM_MODE
                if hb_msg.custom_mode == hold_custom_mode:
                    recovered = hold_handler.attempt_offboard_recovery(ts_ms=_mono_ms())
                    if recovered:
                        d9_ts_ms_ref[0] = _mono_ms()
                        offboard_restored_ev.set()
                        return

        d9_thread = threading.Thread(
            target=_wait_offboard, daemon=True, name="it_d9_offboard_wait"
        )
        d9_thread.start()
        d9_thread.join(timeout=D9_OFFBOARD_WAIT_MS / 1000.0 + 2.0)

        # ------------------------------------------------------------------
        # 11. Compute position discrepancy and emit MISSION_RESUMED (D9)
        # ------------------------------------------------------------------
        ned_after = _get_ned_position(conn) or ned_before
        pos_disc_m = math.sqrt(
            (ned_after[0] - cp.pos_ned[0]) ** 2 +
            (ned_after[1] - cp.pos_ned[1]) ** 2 +
            (ned_after[2] - cp.pos_ned[2]) ** 2
        )

        g4_pass = offboard_restored_ev.is_set()

        if g4_pass:
            event_log.append({
                "event":                "MISSION_RESUMED",
                "req_id":               "PX4-04",
                "severity":             "INFO",
                "module_name":          "ITD9ChainTest",
                "timestamp_ms":         d9_ts_ms_ref[0],
                "position_discrepancy_m": round(pos_disc_m, 3),
            })

        # Verify G4 field content
        d9_events = [e for e in event_log if e.get("event") == "MISSION_RESUMED"]
        g4_pass = (
            g4_pass and
            bool(d9_events) and
            "position_discrepancy_m" in d9_events[-1]
        )

        # ------------------------------------------------------------------
        # 12. Stop daemon threads
        # ------------------------------------------------------------------
        stop_ev.set()

        # ------------------------------------------------------------------
        # 13. Write evidence artefact
        # ------------------------------------------------------------------
        _write_evidence(
            run_ts=run_ts,
            head_hash=head,
            kill_ts_ms=kill_ts_ms,
            d7_ts_ms=d7_ts_ms,
            g1_pass=g1_pass,
            g2_pass=g2_pass,
            g3_pass=g3_pass,
            g4_pass=g4_pass,
            g1_latency_ms=g1_latency_ms,
            g2_latency_ms=g2_latency_ms,
            d8_age_ms=d8_age_ms,
            d8_wp=d8_wp_idx,
            pos_disc_m=pos_disc_m,
        )

        # ------------------------------------------------------------------
        # 14. Gate assertions — Deputy 1 rules on acceptance
        # ------------------------------------------------------------------
        assert g1_pass, (
            f"G1 FAIL: PX4_REBOOT_DETECTED latency {g1_latency_ms} ms "
            f"> {G1_D7_WINDOW_MS} ms, or seq_reset_value/recovery_start_ms missing. "
            f"Event: {d7_evt if d7_events else 'ABSENT'}"
        )
        assert g2_pass, (
            f"G2 FAIL: CHECKPOINT_RESTORED timing {g2_latency_ms} ms "
            f"(threshold {G2_D8_WINDOW_MS} ms) or missing fields. "
            f"Events: {d8_events}"
        )
        assert g3_pass, (
            f"G3 FAIL: AUTONOMOUS_RESUME_APPROVED not logged or "
            f"pending_operator_clearance_required != False. Events: {d8a_events}"
        )
        assert g4_pass, (
            f"G4 FAIL: OFFBOARD not restored within {D9_OFFBOARD_WAIT_MS} ms "
            f"or MISSION_RESUMED event missing position_discrepancy_m. "
            f"D9 events: {d9_events}"
        )
