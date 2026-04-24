"""
tests/test_ec01_offboard_endurance.py
MicroMind — EC-01 OFFBOARD 30-Minute Endurance Gate

@pytest.mark.sitl — excluded from certified SIL baseline.
Runs on live PX4 SITL (micromind-node01, single vehicle, port 14540).

Requirements: SRS §6.1 PX4-01, EC-01
Gates:
    EC01-G1: offboard_continuity_percent >= 99.5 %
    EC01-G2: offboard_loss_count <= 1
    EC01-G3: setpoint_rate_hz >= 20.0 Hz at every 1 Hz measurement point
    EC01-G4: stale_setpoints_discarded_on_recovery = True on every recovery event

Clock discipline (SR-01):
    Mission timing uses time.monotonic() (monotonic wall-clock, not time.time()).
    Per-tick cadence uses threading.Event.wait(timeout) — no time.sleep() calls
    in any test method.  Setpoint stream daemon thread uses stop_ev.wait(interval)
    for its delivery cadence.  PX4ContinuityMonitor receives clock_fn=_mono_ms
    wrapping time.monotonic().

Evidence output: docs/qa/EC01_EVIDENCE_RUN1.md
"""

from __future__ import annotations

import datetime
import os
import subprocess
import threading
import time as _time
from typing import Any, Dict, List, Optional

import pytest

from integration.bridge.offboard_monitor import PX4ContinuityMonitor

# ---------------------------------------------------------------------------
# pymavlink — optional import (available in system Python 3.12, not conda env)
# Module-level try/except avoids ament-plugin collection interference.
# The test skips at runtime via pytest.skip() if not importable.
# ---------------------------------------------------------------------------
try:
    from pymavlink import mavutil as _mavutil
    _HAS_PYMAVLINK = True
except ImportError:
    _mavutil = None  # type: ignore[assignment]
    _HAS_PYMAVLINK = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SITL_HOST          = "127.0.0.1"
SITL_PORT          = 14540
MISSION_DURATION_S = 1800          # 30 minutes
SETPOINT_RATE_HZ   = 20.0
ALTITUDE_M         = 50.0          # NED hover altitude (metres AGL)
EVIDENCE_PATH      = "docs/qa/EC01_EVIDENCE_RUN1.md"

# SRS §6.1 PX4-01 gate thresholds
EC01_G1_MIN_CONTINUITY_PCT = 99.5
EC01_G2_MAX_LOSS_COUNT     = 1
EC01_G3_MIN_RATE_HZ        = 20.0

# PX4 custom-mode encoding (px4_custom_mode.h)
# HEARTBEAT.custom_mode bits [16-23] = main_mode; OFFBOARD = 6
_PX4_OFFBOARD_MAIN_MODE: int = 6


# ---------------------------------------------------------------------------
# Monotonic clock (no time.time())
# ---------------------------------------------------------------------------

def _mono_ms() -> int:
    """Return monotonic wall-clock time in milliseconds."""
    return int(_time.monotonic() * 1000)


# ---------------------------------------------------------------------------
# SITL helpers
# ---------------------------------------------------------------------------

def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _is_offboard(hb_msg: Any) -> bool:
    """True if PX4 HEARTBEAT reports OFFBOARD as the active main flight mode."""
    if hb_msg is None:
        return False
    return ((hb_msg.custom_mode >> 16) & 0xFF) == _PX4_OFFBOARD_MAIN_MODE


def _send_hover_setpoint(
    conn: Any,
    target_system: int,
    target_component: int,
    alt_m: float,
) -> None:
    """Send a single SET_POSITION_TARGET_LOCAL_NED holding position at alt_m AGL."""
    conn.mav.set_position_target_local_ned_send(
        0,                              # time_boot_ms (ignored by PX4)
        target_system, target_component,
        _mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,             # position only; ignore vel/acc/yaw_rate
        0.0, 0.0, -alt_m,              # NED: x=N, y=E, z=down (negative = up)
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0,
    )


def _setpoint_stream_fn(
    conn: Any,
    target_system: int,
    target_component: int,
    alt_m: float,
    stop_ev: threading.Event,
    monitor: PX4ContinuityMonitor,
    rate_hz: float = SETPOINT_RATE_HZ,
) -> None:
    """
    Daemon thread: stream position setpoints at rate_hz and notify monitor of
    each dispatch.  Uses stop_ev.wait(interval) for delivery cadence (SR-01
    compliant — no time.sleep(); stop_ev allows clean abort).
    """
    interval = 1.0 / rate_hz
    while not stop_ev.is_set():
        ts_ms = _mono_ms()
        _send_hover_setpoint(conn, target_system, target_component, alt_m)
        monitor.record_setpoint(ts_ms=ts_ms)
        stop_ev.wait(interval)


def _gcs_heartbeat_fn(conn: Any, stop_ev: threading.Event) -> None:
    """Daemon thread: send GCS heartbeat at 1 Hz (PX4 pre-arm check)."""
    while not stop_ev.is_set():
        conn.mav.heartbeat_send(
            _mavutil.mavlink.MAV_TYPE_GCS,
            _mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0,
            _mavutil.mavlink.MAV_STATE_ACTIVE,
        )
        stop_ev.wait(1.0)


def _arm_and_offboard(
    conn: Any,
    target_system: int,
    target_component: int,
) -> bool:
    """
    Send ARM then MAV_CMD_DO_SET_MODE OFFBOARD.
    Returns True on both ACKs; False on any failure or timeout.
    """
    conn.mav.command_long_send(
        target_system, target_component,
        _mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0,
    )
    ack = conn.recv_match(type='COMMAND_ACK', blocking=True, timeout=10.0)
    if not ack or ack.result != 0:
        return False

    conn.mav.command_long_send(
        target_system, target_component,
        _mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0, 209, 6, 0, 0, 0, 0, 0,  # base_mode=209, custom_mode=6 (OFFBOARD)
    )
    ack = conn.recv_match(type='COMMAND_ACK', blocking=True, timeout=10.0)
    return bool(ack and ack.result == 0)


# ---------------------------------------------------------------------------
# Evidence writer
# ---------------------------------------------------------------------------

def _write_evidence(
    run_ts: str,
    head_hash: str,
    continuity_pct: float,
    loss_count: int,
    rate_min: float,
    rate_mean: float,
    rate_max: float,
    g1_pass: bool,
    g2_pass: bool,
    g3_pass: bool,
    g3_fail_ticks: int,
    g4_pass: bool,
    loss_detail: List[Dict[str, Any]],
) -> None:
    g4_label = "PASS" if g4_pass else "FAIL"
    if not loss_detail:
        g4_label = "N/A — zero recovery events"

    lines = [
        "# EC-01 Evidence Run 1",
        "",
        f"**Run timestamp (UTC):** {run_ts}",
        f"**HEAD commit:** {head_hash}",
        f"**Mission duration:** {MISSION_DURATION_S} s (30 minutes)",
        f"**PX4 SITL endpoint:** udp:{SITL_HOST}:{SITL_PORT}",
        f"**Setpoint rate target:** {SETPOINT_RATE_HZ:.0f} Hz",
        "",
        "## Measured Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| offboard_continuity_percent | {continuity_pct:.4f} % |",
        f"| offboard_loss_count | {loss_count} |",
        f"| setpoint_rate_hz (min) | {rate_min:.2f} Hz |",
        f"| setpoint_rate_hz (mean) | {rate_mean:.2f} Hz |",
        f"| setpoint_rate_hz (max) | {rate_max:.2f} Hz |",
        "",
        "## Gate Results (SRS §6.1 PX4-01)",
        "",
        "| Gate | Criterion | Measured | Result |",
        "|---|---|---|---|",
        (
            f"| EC01-G1 | offboard_continuity_percent ≥ {EC01_G1_MIN_CONTINUITY_PCT} % "
            f"| {continuity_pct:.4f} % | {'PASS' if g1_pass else 'FAIL'} |"
        ),
        (
            f"| EC01-G2 | offboard_loss_count ≤ {EC01_G2_MAX_LOSS_COUNT} "
            f"| {loss_count} | {'PASS' if g2_pass else 'FAIL'} |"
        ),
        (
            f"| EC01-G3 | setpoint_rate_hz ≥ {EC01_G3_MIN_RATE_HZ:.0f} Hz at every tick "
            f"| min={rate_min:.2f} Hz, {g3_fail_ticks} tick(s) below threshold "
            f"| {'PASS' if g3_pass else 'FAIL'} |"
        ),
        (
            f"| EC01-G4 | stale_setpoints_discarded_on_recovery = True "
            f"| {len(loss_detail)} recovery event(s) "
            f"| {g4_label} |"
        ),
        "",
    ]

    if loss_detail:
        lines += [
            "## OFFBOARD_LOSS / Recovery Events",
            "",
            "| # | loss_ts_ms | gap_ms | recovery_time_ms | stale_setpoints_discarded |",
            "|---|---|---|---|---|",
        ]
        for i, ev in enumerate(loss_detail, 1):
            lines.append(
                f"| {i} "
                f"| {ev.get('loss_ts_ms', 'N/A')} "
                f"| {ev.get('gap_ms', 'N/A')} "
                f"| {ev.get('recovery_time_ms', 'N/A')} "
                f"| {ev.get('stale_setpoints_discarded', 'N/A')} |"
            )
        lines.append("")
    else:
        lines += [
            "## OFFBOARD_LOSS / Recovery Events",
            "",
            "None — zero loss events recorded during run.",
            "",
        ]

    lines += [
        "---",
        "*Generated by tests/test_ec01_offboard_endurance.py — EC-01 SRS §6.1 PX4-01*",
    ]

    dest = EVIDENCE_PATH
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.sitl
class TestEC01OffboardEndurance:
    """
    EC-01 OFFBOARD 30-minute endurance gate — live PX4 SITL.

    Requires:
        PX4 SITL running on 127.0.0.1:14540 before the test is invoked.
        Single vehicle in Gazebo Baylands world; EKF2 aligned.

    Run time: 1800 s real wall-clock + setup (~60 s).
    """

    def test_ec01_offboard_endurance(self) -> None:
        if not _HAS_PYMAVLINK:
            pytest.skip(
                "pymavlink not installed in this environment — "
                "run with system Python 3.12: python3.12 -m pytest -m sitl"
            )

        run_ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        head   = _git_head()

        # ----------------------------------------------------------------
        # 1. Connect to PX4 SITL
        # ----------------------------------------------------------------
        conn = _mavutil.mavlink_connection(f"udp:{SITL_HOST}:{SITL_PORT}")
        hb   = conn.wait_heartbeat(timeout=30)
        assert hb is not None, (
            f"No heartbeat from PX4 SITL at {SITL_HOST}:{SITL_PORT} within 30 s. "
            "Ensure PX4 SITL is running before invoking this test."
        )
        target_system    = conn.target_system
        target_component = conn.target_component

        # ----------------------------------------------------------------
        # 2. Instantiate PX4ContinuityMonitor with monotonic clock
        # ----------------------------------------------------------------
        event_log: List[Dict[str, Any]] = []
        monitor = PX4ContinuityMonitor(
            event_log=event_log,
            clock_fn=_mono_ms,
        )

        # ----------------------------------------------------------------
        # 3. Start daemon threads: GCS heartbeat + pre-stream setpoints
        # ----------------------------------------------------------------
        stop_ev = threading.Event()

        hb_thread = threading.Thread(
            target=_gcs_heartbeat_fn,
            args=(conn, stop_ev),
            daemon=True,
            name="ec01_gcs_hb",
        )
        hb_thread.start()

        # Pre-stream 2 s at 20 Hz to satisfy PX4 OFFBOARD precondition
        _pre_stop = threading.Event()
        _pre = threading.Thread(
            target=_setpoint_stream_fn,
            args=(conn, target_system, target_component,
                  ALTITUDE_M, _pre_stop, monitor),
            daemon=True,
            name="ec01_prestream",
        )
        _pre.start()
        _pre_stop.wait(2.0)
        _pre_stop.set()
        _pre.join(timeout=1.0)

        # ----------------------------------------------------------------
        # 4. ARM + OFFBOARD (main setpoint stream covers ACK wait windows)
        # ----------------------------------------------------------------
        sp_thread = threading.Thread(
            target=_setpoint_stream_fn,
            args=(conn, target_system, target_component,
                  ALTITUDE_M, stop_ev, monitor),
            daemon=True,
            name="ec01_sp_stream",
        )
        sp_thread.start()

        ok = _arm_and_offboard(conn, target_system, target_component)
        assert ok, (
            "ARM / OFFBOARD engage failed. Check: vehicle armed, EKF2 aligned, "
            "setpoint stream active, no pre-arm failures."
        )

        # ----------------------------------------------------------------
        # 5. 30-minute measurement loop at 1 Hz
        # ----------------------------------------------------------------
        t_start       = _time.monotonic()
        t_mission_end = t_start + MISSION_DURATION_S
        t_next_tick   = t_start + 1.0        # first measurement at T+1 s

        measurements: List[Dict[str, Any]] = []   # 1800 per-tick records
        loss_detail:  List[Dict[str, Any]] = []   # per-event evidence rows

        last_offboard: bool           = True   # assumed engaged after step 4
        current_loss_ts_ms: Optional[int] = None
        tick_ev = threading.Event()            # never externally set; interruptible wait

        while _time.monotonic() < t_mission_end:
            # Interruptible wait until next 1 Hz tick
            remaining = t_next_tick - _time.monotonic()
            if remaining > 0:
                tick_ev.wait(timeout=remaining)

            ts_ms = _mono_ms()
            t_next_tick += 1.0

            # Receive latest HEARTBEAT from PX4 (short window)
            hb_msg = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=0.08)
            is_off = _is_offboard(hb_msg) if hb_msg is not None else last_offboard

            # ----------------------------------------------------------
            # Detect OFFBOARD_LOSS / OFFBOARD_RESTORED transitions
            # ----------------------------------------------------------
            if last_offboard and not is_off:
                monitor.record_offboard_loss(ts_ms=ts_ms)
                current_loss_ts_ms = ts_ms

            elif not last_offboard and is_off:
                monitor.record_offboard_restored(ts_ms=ts_ms)
                restored = [
                    e for e in event_log
                    if e.get("event") == "OFFBOARD_RESTORED"
                ]
                if restored:
                    payload = restored[-1]["payload"]
                    loss_detail.append({
                        "loss_ts_ms":              current_loss_ts_ms,
                        "gap_ms":                  payload["gap_duration_ms"],
                        "recovery_time_ms":        ts_ms - (current_loss_ts_ms or ts_ms),
                        "stale_setpoints_discarded": payload["stale_setpoints_discarded"],
                    })
                current_loss_ts_ms = None

            last_offboard = is_off

            # 1 Hz rate measurement and log
            rate_hz = monitor.measure_rate_hz(ts_ms=ts_ms)
            monitor.log_setpoint_rate(ts_ms=ts_ms)

            measurements.append({
                "ts_ms":            ts_ms,
                "setpoint_rate_hz": rate_hz,
                "offboard_active":  is_off,
            })

        # ----------------------------------------------------------------
        # 6. Stop daemon threads cleanly
        # ----------------------------------------------------------------
        stop_ev.set()

        # ----------------------------------------------------------------
        # 7. Compute final metrics
        # ----------------------------------------------------------------
        total_mission_ms = MISSION_DURATION_S * 1000
        continuity_pct   = monitor.compute_continuity(
            total_mission_ms=total_mission_ms
        )
        loss_count = monitor.offboard_loss_count

        rates         = [m["setpoint_rate_hz"] for m in measurements]
        rate_min      = min(rates)            if rates else 0.0
        rate_mean     = sum(rates) / len(rates) if rates else 0.0
        rate_max      = max(rates)            if rates else 0.0
        g3_fail_ticks = sum(1 for r in rates if r < EC01_G3_MIN_RATE_HZ)

        g1_pass = continuity_pct >= EC01_G1_MIN_CONTINUITY_PCT
        g2_pass = loss_count     <= EC01_G2_MAX_LOSS_COUNT
        g3_pass = g3_fail_ticks  == 0
        g4_pass = (
            all(ev.get("stale_setpoints_discarded") is True for ev in loss_detail)
            if loss_detail else True
        )

        # ----------------------------------------------------------------
        # 8. Write evidence artefact
        # ----------------------------------------------------------------
        _write_evidence(
            run_ts=run_ts,
            head_hash=head,
            continuity_pct=continuity_pct,
            loss_count=loss_count,
            rate_min=rate_min,
            rate_mean=rate_mean,
            rate_max=rate_max,
            g1_pass=g1_pass,
            g2_pass=g2_pass,
            g3_pass=g3_pass,
            g3_fail_ticks=g3_fail_ticks,
            g4_pass=g4_pass,
            loss_detail=loss_detail,
        )

        # ----------------------------------------------------------------
        # 9. Gate assertions (EC01-G1 through EC01-G4)
        # ----------------------------------------------------------------
        assert g1_pass, (
            f"EC01-G1 FAIL: offboard_continuity_percent {continuity_pct:.4f} % "
            f"< SRS §6.1 threshold {EC01_G1_MIN_CONTINUITY_PCT} %"
        )
        assert g2_pass, (
            f"EC01-G2 FAIL: offboard_loss_count {loss_count} "
            f"> limit {EC01_G2_MAX_LOSS_COUNT}"
        )
        assert g3_pass, (
            f"EC01-G3 FAIL: {g3_fail_ticks} measurement tick(s) with "
            f"setpoint_rate_hz < {EC01_G3_MIN_RATE_HZ} Hz "
            f"(min measured = {rate_min:.2f} Hz)"
        )
        assert g4_pass, (
            f"EC01-G4 FAIL: not all recovery events had "
            f"stale_setpoints_discarded = True — detail: {loss_detail}"
        )
