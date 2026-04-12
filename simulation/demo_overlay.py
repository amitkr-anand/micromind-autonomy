#!/usr/bin/env python3.12
"""
simulation/demo_overlay.py
MicroMind VIZ-02 — Run 1 Live Mission Tracker

Standalone overlay that runs alongside Gazebo during run_demo.sh.
Subscribes to MAVLink LOCAL_POSITION_NED from both vehicles at 10 Hz,
displays a real-time matplotlib figure with:

  - Full Baylands terrain boundary (899m × 587m, grey outline)
  - Planned waypoints: blue markers
  - Vehicle A actual track: red line, 'A' label at current position
  - Vehicle B actual track: green line, 'B' label at current position
  - Mission phase annotation: top-left
  - Altitude bar: right panel showing current altitude of each vehicle

Window title: "MicroMind — Live Mission Tracker | Baylands SITL"
Update rate: 100 ms (10 Hz) via FuncAnimation.

Constraints
-----------
  - Does NOT import from core/.  Reads MAVLink only.  Orchestration Box only.
  - Does NOT modify run_mission.py or any core/ file.
  - Runs in a background thread from run_demo.sh so it does not block the mission.

Planned waypoints are read from simulation/run_mission.py constants at startup
(imported module — no hardcoding).

Req: VIZ-02  SRS: §9.3  V-01 clock rule  V-02 overlay rule
"""

import math
import sys
import threading
import time

import matplotlib
matplotlib.use("TkAgg")   # non-blocking backend; falls back gracefully
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from pymavlink import mavutil

# ---------------------------------------------------------------------------
# Planned waypoint source — read from run_mission.py constants at startup.
# Import is safe: run_mission.py is in simulation/ (not core/).
# ---------------------------------------------------------------------------
try:
    from simulation.run_mission import (
        ellipse_waypoints,
        ELLIPSE_CX, ELLIPSE_CY, ELLIPSE_A, ELLIPSE_B,
        VEH_A_CY_OFFSET,
        PORT_VEH_B, PORT_VEH_A,
    )
except ImportError:
    # Fallback for direct execution from repo root
    import importlib.util, os
    _spec = importlib.util.spec_from_file_location(
        "run_mission",
        os.path.join(os.path.dirname(__file__), "run_mission.py"),
    )
    _rm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_rm)
    ellipse_waypoints = _rm.ellipse_waypoints
    ELLIPSE_CX        = _rm.ELLIPSE_CX
    ELLIPSE_CY        = _rm.ELLIPSE_CY
    ELLIPSE_A         = _rm.ELLIPSE_A
    ELLIPSE_B         = _rm.ELLIPSE_B
    VEH_A_CY_OFFSET   = _rm.VEH_A_CY_OFFSET
    PORT_VEH_B        = _rm.PORT_VEH_B
    PORT_VEH_A        = _rm.PORT_VEH_A

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TERRAIN_WIDTH_M  = 899.0   # Baylands terrain X extent
TERRAIN_HEIGHT_M = 587.0   # Baylands terrain Y extent

UPDATE_INTERVAL_MS = 100   # 10 Hz animation

MAVLINK_TIMEOUT_S = 0.05   # non-blocking poll at each animation tick

MAX_TRAIL_POINTS = 1000    # keep last N track points per vehicle

# Phase labels keyed by elapsed time thresholds (seconds) — best-effort
# Phase boundaries based on GPS denial at T+60s default
_PHASE_THRESHOLDS = [
    (0,   "PHASE: Pre-Denial (GPS OK)"),
    (60,  "PHASE: GPS Denied — INS Only (Veh A) / VIO (Veh B)"),
    (120, "PHASE: Extended INS Propagation"),
    (300, "PHASE: Long-Range INS — Drift Accumulating"),
]

# ---------------------------------------------------------------------------
# Planned waypoints (NED, relative to Vehicle B spawn)
# ---------------------------------------------------------------------------

def _get_planned_waypoints_ned():
    """
    Build planned NED waypoint list for both vehicles at startup.
    Returns list of (ned_n, ned_e) tuples.
    """
    # Vehicle B ellipse (spawn at SPAWN_B_ENU = (0, 0))
    pts_b = ellipse_waypoints(ELLIPSE_CX, ELLIPSE_CY, ELLIPSE_A, ELLIPSE_B)
    ned_b = [(x, y) for (x, y) in pts_b]   # ENU X = NED North, ENU Y = NED East

    # Vehicle A ellipse (spawn at SPAWN_A_ENU = (0, 5), offset by VEH_A_CY_OFFSET)
    pts_a = ellipse_waypoints(ELLIPSE_CX, ELLIPSE_CY + VEH_A_CY_OFFSET,
                               ELLIPSE_A, ELLIPSE_B)
    ned_a = [(x, y - 5.0) for (x, y) in pts_a]  # subtract spawn_a_enu[1]=5

    return ned_b, ned_a


# ---------------------------------------------------------------------------
# Shared state — updated by MAVLink threads, read by animation callback
# ---------------------------------------------------------------------------

class _VehicleState:
    def __init__(self):
        self.north_m  = 0.0
        self.east_m   = 0.0
        self.alt_m    = 0.0
        self.elapsed  = 0.0
        self.connected = False
        self.trail_n: list = []
        self.trail_e: list = []
        self._lock = threading.Lock()

    def update(self, north, east, alt, elapsed=None):
        with self._lock:
            self.north_m  = north
            self.east_m   = east
            self.alt_m    = alt
            if elapsed is not None:
                self.elapsed = elapsed
            self.connected = True
            self.trail_n.append(north)
            self.trail_e.append(east)
            if len(self.trail_n) > MAX_TRAIL_POINTS:
                self.trail_n = self.trail_n[-MAX_TRAIL_POINTS:]
                self.trail_e = self.trail_e[-MAX_TRAIL_POINTS:]

    def snapshot(self):
        with self._lock:
            return (
                self.north_m, self.east_m, self.alt_m, self.elapsed,
                self.connected,
                list(self.trail_n), list(self.trail_e),
            )


_state_a = _VehicleState()
_state_b = _VehicleState()
_t_mission_start = None
_t_lock = threading.Lock()
_stop_event = threading.Event()


# ---------------------------------------------------------------------------
# MAVLink reader threads
# ---------------------------------------------------------------------------

def _mavlink_reader(port: int, state: _VehicleState, label: str):
    """
    Connect to MAVLink on given UDP port, poll LOCAL_POSITION_NED at ~10 Hz.
    Updates shared _VehicleState.  Runs until _stop_event is set.
    """
    global _t_mission_start

    print(f"[OVERLAY/{label}] Connecting to udp:127.0.0.1:{port}...")
    try:
        mav = mavutil.mavlink_connection(f"udp:127.0.0.1:{port}")
    except Exception as exc:
        print(f"[OVERLAY/{label}] Connection error: {exc}")
        return

    # Wait for heartbeat (non-blocking with 2s timeout, retry loop)
    while not _stop_event.is_set():
        hb = mav.wait_heartbeat(timeout=2.0)
        if hb is not None:
            print(f"[OVERLAY/{label}] Heartbeat sysid={mav.target_system}")
            break
        print(f"[OVERLAY/{label}] Waiting for heartbeat...")

    if _stop_event.is_set():
        return

    # Send GCS heartbeats to trigger telemetry
    def _hb_loop():
        while not _stop_event.is_set():
            try:
                mav.mav.heartbeat_send(6, 8, 0, 0, 4)
            except Exception:
                pass
            _stop_event.wait(1.0)

    hb_thread = threading.Thread(target=_hb_loop, daemon=True)
    hb_thread.start()

    while not _stop_event.is_set():
        msg = mav.recv_match(type="LOCAL_POSITION_NED",
                              blocking=True, timeout=MAVLINK_TIMEOUT_S)
        if msg is None:
            continue

        north = msg.x
        east  = msg.y
        alt   = -msg.z  # NED z is negative up

        with _t_lock:
            if _t_mission_start is None:
                _t_mission_start = time.monotonic()
            elapsed = time.monotonic() - _t_mission_start

        state.update(north, east, alt, elapsed)


# ---------------------------------------------------------------------------
# Phase label helper
# ---------------------------------------------------------------------------

def _phase_label(elapsed_s: float) -> str:
    label = _PHASE_THRESHOLDS[0][1]
    for thresh, lbl in _PHASE_THRESHOLDS:
        if elapsed_s >= thresh:
            label = lbl
    return label


# ---------------------------------------------------------------------------
# Overlay figure builder
# ---------------------------------------------------------------------------

def _build_figure(wp_b_ned, wp_a_ned):
    """Create and return (fig, ax_main, ax_alt, artists_dict)."""
    fig = plt.figure(figsize=(14, 8), facecolor="#1a1a2e")
    fig.canvas.manager.set_window_title(
        "MicroMind — Live Mission Tracker | Baylands SITL"
    )

    # Grid: main map (left, wide) + altitude bar (right, narrow)
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
    ax  = fig.add_subplot(gs[0, 0])
    ax_alt = fig.add_subplot(gs[0, 1])

    # ── Main map styling ──
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(colors="grey")
    ax.spines[:].set_color("#333355")
    ax.set_xlabel("East (m)", color="grey", fontsize=9)
    ax.set_ylabel("North (m)", color="grey", fontsize=9)

    # Terrain boundary (899m × 587m, grey outline)
    terrain_rect = mpatches.Rectangle(
        (0, 0), TERRAIN_HEIGHT_M, TERRAIN_WIDTH_M,
        linewidth=1.5, edgecolor="#556677", facecolor="none",
        linestyle="--", label="Baylands boundary",
    )
    ax.add_patch(terrain_rect)

    # Set axis limits with margin
    margin = 30.0
    ax.set_xlim(-margin, TERRAIN_HEIGHT_M + margin)
    ax.set_ylim(-margin, TERRAIN_WIDTH_M + margin)
    ax.set_aspect("equal")

    # Planned waypoints
    wp_b_e = [wp[1] for wp in wp_b_ned]
    wp_b_n = [wp[0] for wp in wp_b_ned]
    wp_a_e = [wp[1] for wp in wp_a_ned]
    wp_a_n = [wp[0] for wp in wp_a_ned]

    ax.plot(wp_b_e + [wp_b_e[0]], wp_b_n + [wp_b_n[0]],
            "b--", alpha=0.4, linewidth=0.8, label="Planned route B")
    ax.plot(wp_a_e + [wp_a_e[0]], wp_a_n + [wp_a_n[0]],
            "b--", alpha=0.4, linewidth=0.8, label="Planned route A")
    ax.scatter(wp_b_e, wp_b_n, c="dodgerblue", s=25, zorder=4)
    ax.scatter(wp_a_e, wp_a_n, c="cornflowerblue", s=25, zorder=4)

    # Dynamic artists — initialised empty
    trail_b,  = ax.plot([], [], "g-",  linewidth=1.2, alpha=0.85, label="Veh B (MicroMind)")
    trail_a,  = ax.plot([], [], "r-",  linewidth=1.2, alpha=0.85, label="Veh A (INS Only)")
    dot_b,    = ax.plot([], [], "go",  markersize=9,  zorder=5)
    dot_a,    = ax.plot([], [], "ro",  markersize=9,  zorder=5)

    lbl_b = ax.text(0, 0, "B", color="lime",   fontsize=10, fontweight="bold",
                    ha="left", va="bottom", zorder=6)
    lbl_a = ax.text(0, 0, "A", color="tomato", fontsize=10, fontweight="bold",
                    ha="left", va="bottom", zorder=6)

    phase_txt = ax.text(
        0.01, 0.97, "PHASE: Waiting for telemetry...",
        transform=ax.transAxes, color="yellow", fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#0d0d1a",
              "edgecolor": "#556677", "alpha": 0.8},
    )

    ax.legend(loc="lower right", fontsize=7, facecolor="#0d0d1a",
              labelcolor="grey", edgecolor="#333355")

    # ── Altitude bar ──
    ax_alt.set_facecolor("#0d0d1a")
    ax_alt.spines[:].set_color("#333355")
    ax_alt.tick_params(colors="grey", labelsize=7)
    ax_alt.set_title("Alt (m)", color="grey", fontsize=8, pad=4)
    ax_alt.set_xlim(0, 2)
    ax_alt.set_ylim(0, 150)
    ax_alt.set_xticks([])
    ax_alt.yaxis.set_label_position("right")
    ax_alt.yaxis.tick_right()

    bar_b = ax_alt.bar([0.5], [0], width=0.6, color="limegreen",  alpha=0.8, label="B")
    bar_a = ax_alt.bar([1.5], [0], width=0.6, color="tomato",     alpha=0.8, label="A")
    alt_lbl_b = ax_alt.text(0.5, 2, "0m", color="limegreen", fontsize=7,
                             ha="center", va="bottom")
    alt_lbl_a = ax_alt.text(1.5, 2, "0m", color="tomato",    fontsize=7,
                             ha="center", va="bottom")
    ax_alt.legend(loc="upper center", fontsize=6, facecolor="#0d0d1a",
                  labelcolor="grey", edgecolor="#333355", ncol=2)

    artists = {
        "trail_b":   trail_b,
        "trail_a":   trail_a,
        "dot_b":     dot_b,
        "dot_a":     dot_a,
        "lbl_b":     lbl_b,
        "lbl_a":     lbl_a,
        "phase_txt": phase_txt,
        "bar_b":     bar_b,
        "bar_a":     bar_a,
        "alt_lbl_b": alt_lbl_b,
        "alt_lbl_a": alt_lbl_a,
    }

    return fig, ax, ax_alt, artists


# ---------------------------------------------------------------------------
# Animation update callback
# ---------------------------------------------------------------------------

def _make_update(artists):
    def update(frame):
        nb, eb, ab, el_b, conn_b, tn_b, te_b = _state_b.snapshot()
        na, ea, aa, el_a, conn_a, tn_a, te_a = _state_a.snapshot()

        # Trails
        if len(tn_b) > 1:
            artists["trail_b"].set_data(te_b, tn_b)
        if len(tn_a) > 1:
            artists["trail_a"].set_data(te_a, tn_a)

        # Current position dots + labels
        if conn_b:
            artists["dot_b"].set_data([eb], [nb])
            artists["lbl_b"].set_position((eb + 5, nb + 5))
        if conn_a:
            artists["dot_a"].set_data([ea], [na])
            artists["lbl_a"].set_position((ea + 5, na + 5))

        # Phase annotation
        elapsed = max(el_a, el_b)
        if conn_a or conn_b:
            artists["phase_txt"].set_text(_phase_label(elapsed))

        # Altitude bars
        for bar_container, alt_val in [
            (artists["bar_b"], ab),
            (artists["bar_a"], aa),
        ]:
            for rect in bar_container:
                rect.set_height(min(alt_val, 150.0))

        artists["alt_lbl_b"].set_position((0.5, min(ab + 2, 145)))
        artists["alt_lbl_b"].set_text(f"{ab:.0f}m")
        artists["alt_lbl_a"].set_position((1.5, min(aa + 2, 145)))
        artists["alt_lbl_a"].set_text(f"{aa:.0f}m")

        return list(artists.values())

    return update


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_overlay():
    """
    Entry point.  Starts MAVLink reader threads then launches matplotlib
    FuncAnimation.  Blocks until the window is closed or _stop_event is set.
    """
    wp_b_ned, wp_a_ned = _get_planned_waypoints_ned()

    # Start MAVLink reader threads
    t_b = threading.Thread(
        target=_mavlink_reader,
        args=(PORT_VEH_B, _state_b, "VEH_B"),
        daemon=True, name="overlay_mav_b",
    )
    t_a = threading.Thread(
        target=_mavlink_reader,
        args=(PORT_VEH_A, _state_a, "VEH_A"),
        daemon=True, name="overlay_mav_a",
    )
    t_b.start()
    t_a.start()

    fig, ax, ax_alt, artists = _build_figure(wp_b_ned, wp_a_ned)
    update_fn = _make_update(artists)

    ani = animation.FuncAnimation(
        fig,
        update_fn,
        interval=UPDATE_INTERVAL_MS,
        blit=False,
        cache_frame_data=False,
    )

    try:
        plt.show()
    finally:
        _stop_event.set()
        print("[OVERLAY] Closed.")


def run_overlay_background():
    """
    Launch overlay in a background daemon thread.
    Returns the thread object (daemon — exits with process).
    """
    t = threading.Thread(target=run_overlay, daemon=True, name="demo_overlay")
    t.start()
    return t


if __name__ == "__main__":
    print("[OVERLAY] Starting MicroMind Live Mission Tracker...")
    print(f"[OVERLAY] Vehicle B: udp:127.0.0.1:{PORT_VEH_B}")
    print(f"[OVERLAY] Vehicle A: udp:127.0.0.1:{PORT_VEH_A}")
    run_overlay()
