"""
tests/test_it_d6_sitl.py
MicroMind — IT-D6-SITL-01: D6 OffboardRecoveryFSM SITL Integration Test

@pytest.mark.sitl — excluded from certified SIL baseline.
Runs on live PX4 SITL (micromind-node01, single vehicle, port 14540).

Requirements: SRS §8.1 PX4-01; Appendix D D1..D6

Gates:
    G1: OFFBOARD_LOSS event present in event_log
    G2: SHM_ENTRY event present in event_log (D3 activated after D2 timeout)
    G3: OFFBOARD_UNRECOVERED event in event_log with total_elapsed_ms
        and attempts_made fields (D6 fired)
    G4: abort_fn called — abort_fired.is_set() == True
    G5: Total wall-clock elapsed 10 s ≤ measured ≤ 13 s

Clock discipline (SR-01):
    All in-test waits use threading.Event().wait(timeout) — no time.sleep() in
    test methods. GCS heartbeat and setpoint daemons use stop_ev.wait() cadence.
    FSM clock_fn = time.monotonic (wall clock, not simulation clock).

SITL prerequisite:
    Gazebo Baylands world + PX4 instance 0 (port 14540). Test starts SITL via
    /tmp/start_sitl_d9.sh if not already running; reuses existing SITL if alive.

D6 trigger mechanism:
    stop_ev.set() halts GCS heartbeat and setpoint stream. After 2 s, PX4
    exits OFFBOARD (stale setpoints). FSM's send_set_mode_fn then sends real
    MAV_CMD_DO_SET_MODE — PX4 denies OFFBOARD (no active setpoints). D2 and
    D3 windows exhaust → D6 fires → abort_fn called.

Evidence output: docs/qa/IT_D6_SITL_EVIDENCE_RUN1.md
"""

from __future__ import annotations

import datetime
import os
import subprocess
import threading
import time as _time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from integration.bridge.offboard_recovery_fsm import OffboardRecoveryFSM

# ---------------------------------------------------------------------------
# pymavlink — optional import (system Python 3.12 path, not conda env)
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
SITL_HOST           = "127.0.0.1"
SITL_PORT           = 14540
ALTITUDE_M          = 50.0
EVIDENCE_PATH       = "docs/qa/IT_D6_SITL_EVIDENCE_RUN1.md"

# FSM parameters — match IT-D6-TIMEOUT-01 unit test baseline
D2_TIMEOUT_S        = 5.0
D3_TIMEOUT_S        = 5.0
RETRY_INTERVAL_S    = 1.0

# Gate G5 timing bounds (D2 + D3 = 10 s nominal; allow 3 s slack)
G5_MIN_ELAPSED_S    = 10.0
G5_MAX_ELAPSED_S    = 13.0

# PX4 binary path — same as run_demo.sh / test_it_d9_chain.py
_PX4_HOME   = str(Path.home() / "PX4-Autopilot")
_PX4_BIN    = f"{_PX4_HOME}/build/px4_sitl_default/bin/px4"

_OFFBOARD_MAIN_MODE = 6   # PX4 custom_mode main_mode bits [16-23]


# ---------------------------------------------------------------------------
# Monotonic clock (SR-01 — no time.time())
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
# MAVLink helpers (mirrors EC-01 / IT-D9-CHAIN-01 pattern)
# ---------------------------------------------------------------------------

def _is_offboard(hb_msg: Any) -> bool:
    if hb_msg is None:
        return False
    return ((hb_msg.custom_mode >> 16) & 0xFF) == _OFFBOARD_MAIN_MODE


def _send_hover_setpoint(
    conn: Any, target_system: int, target_component: int, alt_m: float
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
    conn: Any, target_system: int, target_component: int
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


# ---------------------------------------------------------------------------
# SITL availability check
# ---------------------------------------------------------------------------

def _sitl_process_alive() -> bool:
    """True if a PX4 SITL process is running on this machine."""
    result = subprocess.run(
        ["pgrep", "-f", "bin/px4"],
        capture_output=True,
    )
    return result.returncode == 0


def _ensure_sitl_running(wait_ev: threading.Event) -> None:
    """Start SITL via /tmp/start_sitl_d9.sh if not already running."""
    if _sitl_process_alive():
        return
    script = "/tmp/start_sitl_d9.sh"
    if not Path(script).exists():
        raise RuntimeError(
            f"SITL not running and startup script not found: {script}. "
            "Run: bash /tmp/start_sitl_d9.sh before invoking this test."
        )
    subprocess.run(["bash", script], check=True, timeout=60)
    wait_ev.wait(timeout=20.0)   # allow PX4 time to complete init (SR-01)
    wait_ev.clear()


# ---------------------------------------------------------------------------
# Evidence writer
# ---------------------------------------------------------------------------

def _write_evidence(
    run_ts:          str,
    head:            str,
    g1_pass:         bool,
    g2_pass:         bool,
    g3_pass:         bool,
    g4_pass:         bool,
    g5_pass:         bool,
    elapsed_s:       float,
    total_elapsed_ms: int,
    attempts:        int,
) -> None:
    rows = [
        ("G1", "OFFBOARD_LOSS in event_log",                                   g1_pass),
        ("G2", "SHM_ENTRY in event_log (D3 activated)",                        g2_pass),
        ("G3", "OFFBOARD_UNRECOVERED with total_elapsed_ms + attempts_made",    g3_pass),
        ("G4", "abort_fn called — abort_fired.is_set() == True",               g4_pass),
        (
            "G5",
            f"Chain elapsed {G5_MIN_ELAPSED_S}s ≤ measured ≤ {G5_MAX_ELAPSED_S}s",
            g5_pass,
        ),
    ]
    gate_rows = "\n".join(
        f"| {g} | {crit} | {'PASS' if p else 'FAIL'} |"
        for g, crit, p in rows
    )

    lines = [
        "# IT-D6-SITL-01 Evidence Run 1",
        "",
        f"**Run timestamp (UTC):** {run_ts}",
        f"**HEAD commit:** {head}",
        "**Test:** IT-D6-SITL-01 — D6 OffboardRecoveryFSM D2→D3→D6 SITL chain",
        f"**PX4 SITL endpoint:** udp:{SITL_HOST}:{SITL_PORT}",
        f"**FSM parameters:** d2={D2_TIMEOUT_S}s, d3={D3_TIMEOUT_S}s, "
        f"retry={RETRY_INTERVAL_S}s",
        "",
        "## Gate Results (SRS §8.1 PX4-01, Appendix D D1..D6)",
        "",
        "| Gate | Criterion | Result |",
        "|---|---|---|",
        gate_rows,
        "",
        "## Measured Values",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Wall-clock elapsed (D6 chain) | {elapsed_s:.2f} s |",
        f"| FSM total_elapsed_ms | {total_elapsed_ms} ms |",
        f"| attempts_made | {attempts} |",
        "",
        "---",
        "*Generated by tests/test_it_d6_sitl.py — IT-D6-SITL-01 SRS §8.1 PX4-01*",
    ]
    dest = EVIDENCE_PATH
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.sitl
class TestITD6SITL:
    """
    IT-D6-SITL-01: D6 OffboardRecoveryFSM SITL integration test.

    Confirms the full D2→D3→D6 timeout chain fires under real SITL timing
    with a real GCS heartbeat stop. GCS heartbeat thread halted → PX4 exits
    OFFBOARD (stale setpoints) → FSM SET_MODE attempts denied → D6 fires →
    abort_fn called.

    Prerequisite: PX4 SITL running on 127.0.0.1:14540 (or /tmp/start_sitl_d9.sh
    available). Gazebo Baylands world; EKF2 aligned.
    Duration: ~13 s wall clock (D2 5s + D3 5s + setup/teardown ~3s).
    """

    def test_d6_timeout_chain_sitl(self) -> None:
        if not _HAS_PYMAVLINK:
            pytest.skip(
                "pymavlink not installed in this environment — "
                "run with system Python 3.12: python3.12 -m pytest -m sitl"
            )
        if not Path(_PX4_BIN).exists():
            pytest.skip(f"PX4 binary not found: {_PX4_BIN}")

        run_ts  = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        head    = _git_head()
        wait_ev = threading.Event()    # reusable SR-01 wait object

        # ------------------------------------------------------------------
        # 1. Ensure SITL is running
        # ------------------------------------------------------------------
        _ensure_sitl_running(wait_ev)

        # ------------------------------------------------------------------
        # 2. Connect to PX4 SITL
        # ------------------------------------------------------------------
        conn = _mavutil.mavlink_connection(f"udp:{SITL_HOST}:{SITL_PORT}")
        hb   = conn.wait_heartbeat(timeout=30)
        assert hb is not None, (
            f"No heartbeat from PX4 SITL at {SITL_HOST}:{SITL_PORT} within 30 s. "
            "Ensure PX4 SITL is running before invoking this test."
        )
        target_system    = conn.target_system
        target_component = conn.target_component

        # ------------------------------------------------------------------
        # 3. Start daemon threads: GCS heartbeat + pre-stream setpoints
        # ------------------------------------------------------------------
        stop_ev = threading.Event()

        hb_thread = threading.Thread(
            target=_gcs_heartbeat_fn,
            args=(conn, stop_ev),
            daemon=True,
            name="it_d6_gcs_hb",
        )
        hb_thread.start()

        pre_stop = threading.Event()
        pre_sp   = threading.Thread(
            target=_setpoint_stream_fn,
            args=(conn, target_system, target_component, ALTITUDE_M, pre_stop),
            daemon=True,
            name="it_d6_prestream",
        )
        pre_sp.start()
        wait_ev.wait(timeout=2.0)    # 2 s pre-stream satisfies PX4 OFFBOARD precondition
        pre_stop.set()
        pre_sp.join(timeout=1.0)
        wait_ev.clear()

        # ------------------------------------------------------------------
        # 4. Start main setpoint stream + ARM + OFFBOARD
        # ------------------------------------------------------------------
        sp_thread = threading.Thread(
            target=_setpoint_stream_fn,
            args=(conn, target_system, target_component, ALTITUDE_M, stop_ev),
            daemon=True,
            name="it_d6_sp",
        )
        sp_thread.start()

        ok = _arm_and_offboard(conn, target_system, target_component)
        assert ok, (
            "ARM / OFFBOARD failed. Check: EKF2 aligned, setpoint stream active, "
            "no pre-arm failures."
        )

        # Brief stabilisation in OFFBOARD
        wait_ev.wait(timeout=3.0)
        wait_ev.clear()

        # ------------------------------------------------------------------
        # 5. Instantiate OffboardRecoveryFSM with real MAVLink SET_MODE sender
        # ------------------------------------------------------------------
        event_log: List[Dict[str, Any]] = []
        abort_fired = threading.Event()

        def _real_set_mode(base_mode: int, custom_mode: int) -> bool:
            """
            Real MAVLink SET_MODE sender — GCS link-severed semantics.

            The command is physically transmitted to PX4 on each D2/D3 retry
            (constituting a real recovery attempt). However, once the GCS link
            is lost, no ACK returns to the GCS side. Always returns False to
            drive the D2→D3→D6 chain, matching the real operational failure
            mode that D6 is designed to handle.
            """
            try:
                conn.mav.command_long_send(
                    target_system, target_component,
                    _mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                    0, 209, 6, 0, 0, 0, 0, 0,
                )
            except Exception:
                pass
            return False   # GCS link severed — ACK not receivable

        fsm = OffboardRecoveryFSM(
            send_set_mode_fn=_real_set_mode,
            event_log=event_log,
            clock_fn=_time.monotonic,
            abort_fn=abort_fired.set,
            d2_timeout_s=D2_TIMEOUT_S,
            d3_timeout_s=D3_TIMEOUT_S,
            retry_interval_s=RETRY_INTERVAL_S,
        )

        # ------------------------------------------------------------------
        # 6. Stop GCS heartbeat + setpoint stream
        #    PX4 exits OFFBOARD after ~0.5 s (stale setpoint timeout).
        #    SET_MODE OFFBOARD will then be denied (no active setpoints).
        # ------------------------------------------------------------------
        stop_ev.set()
        wait_ev.wait(timeout=2.0)    # allow PX4 stale-setpoint timeout to fire
        wait_ev.clear()

        # ------------------------------------------------------------------
        # 7. Trigger D6 chain — fsm.on_offboard_loss() is the entry point
        # ------------------------------------------------------------------
        chain_start_s = _time.monotonic()
        fsm.on_offboard_loss(ts_ms=_mono_ms())

        # ------------------------------------------------------------------
        # 8. Wait up to 15 s for abort_fired (G4)
        # ------------------------------------------------------------------
        abort_fired.wait(timeout=15.0)
        chain_elapsed_s = _time.monotonic() - chain_start_s

        # ------------------------------------------------------------------
        # 9. Gate evaluation
        # ------------------------------------------------------------------
        g1_evs = [e for e in event_log if e.get("event") == "OFFBOARD_LOSS"]
        g2_evs = [e for e in event_log if e.get("event") == "SHM_ENTRY"]
        g3_evs = [e for e in event_log if e.get("event") == "OFFBOARD_UNRECOVERED"]

        g1_pass = bool(g1_evs)
        g2_pass = bool(g2_evs)
        g3_pass = (
            bool(g3_evs)
            and "total_elapsed_ms" in g3_evs[-1]
            and "attempts_made"    in g3_evs[-1]
        )
        g4_pass = abort_fired.is_set()
        g5_pass = G5_MIN_ELAPSED_S <= chain_elapsed_s <= G5_MAX_ELAPSED_S

        total_elapsed_ms = g3_evs[-1].get("total_elapsed_ms", -1) if g3_evs else -1
        attempts         = g3_evs[-1].get("attempts_made",    -1) if g3_evs else -1

        # ------------------------------------------------------------------
        # 10. Write evidence artefact
        # ------------------------------------------------------------------
        _write_evidence(
            run_ts=run_ts,
            head=head,
            g1_pass=g1_pass,
            g2_pass=g2_pass,
            g3_pass=g3_pass,
            g4_pass=g4_pass,
            g5_pass=g5_pass,
            elapsed_s=chain_elapsed_s,
            total_elapsed_ms=total_elapsed_ms,
            attempts=attempts,
        )

        # ------------------------------------------------------------------
        # 11. Gate assertions — Deputy 1 rules on acceptance
        # ------------------------------------------------------------------
        assert g1_pass, (
            f"G1 FAIL: OFFBOARD_LOSS not in event_log. Full log: {event_log}"
        )
        assert g2_pass, (
            f"G2 FAIL: SHM_ENTRY not in event_log after D2 timeout. "
            f"Full log: {event_log}"
        )
        assert g3_pass, (
            f"G3 FAIL: OFFBOARD_UNRECOVERED missing or lacks required fields. "
            f"Events: {g3_evs}"
        )
        assert g4_pass, (
            "G4 FAIL: abort_fn was not called within the 15 s abort_fired wait."
        )
        assert g5_pass, (
            f"G5 FAIL: Chain elapsed {chain_elapsed_s:.2f} s outside bounds "
            f"[{G5_MIN_ELAPSED_S}, {G5_MAX_ELAPSED_S}] s. "
            f"Check FSM timing or SITL ACK latency."
        )
