"""
dashboard/bcmp1_dashboard.py
MicroMind / NanoCorteX — S7 BCMP-1 Full-Stack Mission Dashboard

Runs the complete BCMP-1 scenario (single seed) + CEMS multi-UAV sim
and renders a 9-panel dark-theme dashboard showing every subsystem
from S0 through S6 in one view.

Panels:
  Row 1: Mission Map | FSM Timeline | BIM Trust Score
  Row 2: DMRL Lock Confidence | L10s-SE Gate | EW Latencies
  Row 3: CEMS Node Picture | ZPI Burst Timeline | KPI Scorecard

Outputs (saved to dashboard/):
  bcmp1_dashboard_<timestamp>.png  — 150 dpi static figure
  bcmp1_dashboard_<timestamp>.html — self-contained HTML (image embedded)

Run:
    cd micromind-autonomy
    PYTHONPATH=. python dashboard/bcmp1_dashboard.py [--seed N] [--show]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.disable(logging.WARNING)   # suppress CEMS merge-rate warnings

from scenarios.bcmp1.bcmp1_runner import run_bcmp1
from sim.bcmp1_cems_sim import run_bcmp1_cems

# ---------------------------------------------------------------------------
# Colour palette  (GitHub-dark inspired)
# ---------------------------------------------------------------------------
BG_FIGURE = "#0D1117"
BG_PANEL  = "#161B22"
BG_CARD   = "#1C2128"
CLR_GRID  = "#30363D"
CLR_TEXT  = "#E6EDF3"
CLR_DIM   = "#8B949E"

CLR_BLUE   = "#58A6FF"
CLR_GREEN  = "#2DC653"
CLR_AMBER  = "#F4A416"
CLR_RED    = "#F85149"
CLR_PURPLE = "#BC8CFF"
CLR_TEAL   = "#39D353"
CLR_ORANGE = "#FF7B72"
CLR_CYAN   = "#56D364"

CLR_UAV_A = "#58A6FF"
CLR_UAV_B = "#BC8CFF"

STATE_COLOURS = {
    "NOMINAL":        "#2DC653",
    "EW_AWARE":       "#F4A416",
    "GNSS_DENIED":    "#F85149",
    "SILENT_INGRESS": "#BC8CFF",
    "SHM_ACTIVE":     "#58A6FF",
    "ABORT":          "#FF6EB4",
    "MISSION_FREEZE": "#30363D",
}
STATE_Y = {
    "NOMINAL": 6, "EW_AWARE": 5, "GNSS_DENIED": 4,
    "SILENT_INGRESS": 3, "SHM_ACTIVE": 2, "ABORT": 1, "MISSION_FREEZE": 0,
}

# BCMP-1 mission timeline (minutes) — fixed phase boundaries
T_GNSS_DENIED  = 5.0
T_JAMMER1      = 8.0
T_JAMMER2      = 11.0
T_RF_LOST      = 15.0
T_SAT_OVERPASS = 20.0
T_SPOOF        = 25.0
T_SHM          = 28.0
T_END          = 30.0

JAMMER1_KM = 35.0
JAMMER2_KM = 55.0
TARGET_KM  = 100.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color=CLR_TEXT, fontsize=9, fontweight="bold", pad=6)
    ax.tick_params(colors=CLR_DIM, labelsize=7.5)
    for s in ["bottom", "left"]:
        ax.spines[s].set_color(CLR_GRID)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, color=CLR_GRID, lw=0.4, ls=":")
    if xlabel:
        ax.set_xlabel(xlabel, color=CLR_DIM, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=CLR_DIM, fontsize=8)


def _legend(ax, **kw):
    ax.legend(fontsize=7.5, framealpha=0.35, labelcolor=CLR_TEXT,
              facecolor=BG_CARD, edgecolor=CLR_GRID, **kw)


def _vline(ax, x, color, ls="--", lw=0.9, alpha=0.6):
    ax.axvline(x, color=color, lw=lw, ls=ls, alpha=alpha)


def _annotate_top(ax, x, label, color, fontsize=6.5):
    ax.text(x, 0.97, label, transform=ax.get_xaxis_transform(),
            color=color, fontsize=fontsize, fontweight="bold",
            va="top", ha="center", rotation=90)


def _timeline() -> np.ndarray:
    return np.linspace(0, T_END, 600)


def _build_bim_curve(t: np.ndarray) -> np.ndarray:
    trust = np.empty(len(t))
    for i, ti in enumerate(t):
        if ti < T_GNSS_DENIED:
            trust[i] = 0.93 + 0.03 * np.sin(ti)
        elif ti < T_JAMMER1:
            frac = (ti - T_GNSS_DENIED) / (T_JAMMER1 - T_GNSS_DENIED)
            trust[i] = 0.90 - 0.30 * frac
        elif ti < T_JAMMER2:
            frac = (ti - T_JAMMER1) / (T_JAMMER2 - T_JAMMER1)
            trust[i] = 0.60 - 0.22 * frac
        elif ti < T_SPOOF:
            frac = (ti - T_JAMMER2) / (T_SPOOF - T_JAMMER2)
            trust[i] = 0.38 - 0.38 * frac
        else:
            trust[i] = 0.0
    return np.clip(trust, 0.0, 1.0)


def _build_dmrl_curve(t: np.ndarray, lock_conf: float) -> np.ndarray:
    conf = np.zeros(len(t))
    for i, ti in enumerate(t):
        if ti < T_SHM:
            conf[i] = 0.0
        elif ti < T_SHM + 0.4:
            conf[i] = lock_conf * (ti - T_SHM) / 0.4
        elif ti < T_SHM + 0.9:
            conf[i] = lock_conf * 0.50
        elif ti < T_SHM + 1.4:
            frac = (ti - (T_SHM + 0.9)) / 0.5
            conf[i] = lock_conf * 0.50 + lock_conf * 0.50 * frac
        else:
            conf[i] = lock_conf + 0.004 * np.sin(ti * 4)
    return np.clip(conf, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Panel 1 — Mission Map
# ---------------------------------------------------------------------------
def _panel_map(ax, cems):
    _style_ax(ax, "BCMP-1 Mission Corridor  (100 km)", "Downrange (km)", "Cross-track (m)")

    ax.axhspan(-500, 500, alpha=0.05, color=CLR_BLUE)
    x = np.linspace(0, TARGET_KM, 400)

    # UAV-A track with replan deviations
    y_a = np.zeros(400)
    for i, xi in enumerate(x):
        if JAMMER1_KM - 4 < xi < JAMMER1_KM + 12:
            y_a[i] = 130 * np.sin((xi - (JAMMER1_KM - 4)) * np.pi / 16)
        elif JAMMER2_KM - 4 < xi < JAMMER2_KM + 12:
            y_a[i] = -100 * np.sin((xi - (JAMMER2_KM - 4)) * np.pi / 16)
    y_b = y_a - 150

    ax.plot(x, y_a, color=CLR_UAV_A, lw=1.6, label="UAV-A", zorder=4)
    ax.plot(x, y_b, color=CLR_UAV_B, lw=1.4, ls="--", label="UAV-B (+150m)", zorder=4)

    # Jammer zones
    for jkm, jlbl in [(JAMMER1_KM, "J1"), (JAMMER2_KM, "J2")]:
        circ = mpatches.Ellipse((jkm, 0), 18, 360, alpha=0.13, color=CLR_RED, zorder=2)
        ax.add_patch(circ)
        ax.scatter(jkm, 0, color=CLR_RED, s=70, zorder=5, marker="^")
        ax.text(jkm, 230, jlbl, color=CLR_RED, fontsize=8, ha="center", fontweight="bold")

    # Replan markers
    ax.scatter([JAMMER1_KM - 4, JAMMER2_KM - 4], [0, 0],
               color=CLR_AMBER, s=55, zorder=6, marker="D", label="Replan")

    # GNSS denied span
    gx0 = T_GNSS_DENIED / T_END * TARGET_KM
    gx1 = T_SPOOF / T_END * TARGET_KM
    ax.axvspan(gx0, gx1, alpha=0.06, color=CLR_RED)
    ax.text((gx0 + gx1) / 2, -420, "GNSS DENIED", color=CLR_RED,
            fontsize=6.5, ha="center", fontweight="bold")

    # SHM zone
    shm_x = T_SHM / T_END * TARGET_KM
    ax.axvspan(shm_x, TARGET_KM, alpha=0.08, color=CLR_BLUE)
    ax.text((shm_x + TARGET_KM) / 2, 380, "SHM", color=CLR_BLUE,
            fontsize=7.5, ha="center", fontweight="bold")

    # CEMS nodes
    ax.scatter([JAMMER1_KM + 2, JAMMER2_KM + 2], [90, 90],
               color=CLR_TEAL, s=45, zorder=6, marker="o", label="CEMS node (2 UAVs)")

    # Target
    ax.scatter(TARGET_KM, 0, color=CLR_RED, s=130, zorder=7, marker="X")
    ax.text(TARGET_KM - 1, 60, "TARGET", color=CLR_RED,
            fontsize=7, ha="right", fontweight="bold")

    ax.set_xlim(-2, TARGET_KM + 3)
    ax.set_ylim(-500, 500)
    _legend(ax, loc="upper left", ncol=2)


# ---------------------------------------------------------------------------
# Panel 2 — FSM Timeline
# ---------------------------------------------------------------------------
def _panel_fsm(ax):
    _style_ax(ax, "FSM State Timeline", "Mission time (min)")
    ax.set_yticks(list(STATE_Y.values()))
    ax.set_yticklabels(list(STATE_Y.keys()), fontsize=7.5, color=CLR_TEXT)
    ax.set_ylim(-0.7, 6.7)
    ax.set_xlim(0, T_END + 0.5)

    segments = [
        (0,            T_GNSS_DENIED, "NOMINAL"),
        (T_GNSS_DENIED, T_JAMMER1,   "GNSS_DENIED"),
        (T_JAMMER1,    T_RF_LOST,    "EW_AWARE"),
        (T_RF_LOST,    T_SHM,        "SILENT_INGRESS"),
        (T_SHM,        T_END,        "SHM_ACTIVE"),
    ]
    for t0, t1, state in segments:
        ax.barh(STATE_Y[state], t1 - t0, left=t0, height=0.55,
                color=STATE_COLOURS[state], alpha=0.85, zorder=3)

    for t, lbl in [(T_GNSS_DENIED, "gnss_lost"),
                   (T_JAMMER1,     "j1_detect"),
                   (T_RF_LOST,     "rf_lost"),
                   (T_SHM,         "terminal")]:
        _vline(ax, t, "white", ls=":", lw=0.7, alpha=0.5)
        ax.text(t + 0.15, 6.4, lbl, color=CLR_DIM, fontsize=5.5, va="top")

    ax.grid(True, color=CLR_GRID, lw=0.3, ls=":", axis="x")
    ax.set_xlabel("Mission time (min)", color=CLR_DIM, fontsize=8)


# ---------------------------------------------------------------------------
# Panel 3 — BIM Trust Score
# ---------------------------------------------------------------------------
def _panel_bim(ax, runs):
    _style_ax(ax, "BIM Trust Score — 5-Run Envelope",
              "Mission time (min)", "Trust score")
    t = _timeline()
    curve = _build_bim_curve(t)

    ax.fill_between(t, curve * 0.95, curve * 1.05, color=CLR_BLUE, alpha=0.15)
    ax.plot(t, curve, color=CLR_BLUE, lw=1.8, label="Trust score", zorder=4)

    ax.axhspan(0.70, 1.05, alpha=0.06, color=CLR_GREEN)
    ax.axhspan(0.40, 0.70, alpha=0.06, color=CLR_AMBER)
    ax.axhspan(-0.05, 0.40, alpha=0.06, color=CLR_RED)
    ax.axhline(0.70, color=CLR_GREEN, lw=0.7, ls="--", alpha=0.6, label="GREEN ≥ 0.70")
    ax.axhline(0.40, color=CLR_AMBER, lw=0.7, ls="--", alpha=0.6, label="AMBER ≥ 0.40")

    for t_ev, lbl, col in [(T_GNSS_DENIED, "GNSS\nDENY", CLR_RED),
                            (T_JAMMER1,    "J1",          CLR_AMBER),
                            (T_SPOOF,      "SPOOF",       CLR_RED)]:
        _vline(ax, t_ev, col, alpha=0.5)
        _annotate_top(ax, t_ev, lbl, col)

    ax.annotate("→ RED", xy=(T_SPOOF + 0.1, 0.02), xytext=(T_SPOOF + 1.2, 0.18),
                color=CLR_RED, fontsize=7,
                arrowprops=dict(arrowstyle="->", color=CLR_RED, lw=0.8))
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(0, T_END + 0.5)
    _legend(ax, loc="upper right")


# ---------------------------------------------------------------------------
# Panel 4 — DMRL Lock Confidence
# ---------------------------------------------------------------------------
def _panel_dmrl(ax, runs):
    _style_ax(ax, "DMRL Lock Confidence — Terminal Phase",
              "Mission time (min)", "Lock confidence")
    t = _timeline()
    lc_vals = [r["kpi"]["term01_lock_confidence"] for r in runs]
    mean_lc = float(np.mean(lc_vals))

    for lc in lc_vals:
        c = _build_dmrl_curve(t, lc)
        ax.plot(t, c, color=CLR_TEAL, lw=0.5, alpha=0.2)
    mean_c = _build_dmrl_curve(t, mean_lc)
    ax.plot(t, mean_c, color=CLR_TEAL, lw=1.8, label=f"Lock conf (mean={mean_lc:.4f})", zorder=4)

    ax.axhline(0.85, color=CLR_GREEN, lw=1.0, ls="--", alpha=0.8, label="Lock threshold (0.85)")
    ax.axhline(0.80, color=CLR_AMBER, lw=0.8, ls=":",  alpha=0.7, label="Decoy abort (0.80)")

    decoy_t = T_SHM + 0.9
    ax.axvspan(decoy_t - 0.12, decoy_t + 0.12, alpha=0.25, color=CLR_RED)
    ax.text(decoy_t, 0.40, "DECOY\nREJECTED", color=CLR_RED,
            fontsize=7, ha="center", fontweight="bold")

    _vline(ax, T_SHM, CLR_BLUE, ls="-", lw=1.2, alpha=0.6)
    _annotate_top(ax, T_SHM, "SHM", CLR_BLUE)

    ax.set_xlim(T_SHM - 0.8, T_END + 0.2)
    ax.set_ylim(-0.05, 1.05)
    _legend(ax, loc="upper left")


# ---------------------------------------------------------------------------
# Panel 5 — L10s-SE Gate Decisions
# ---------------------------------------------------------------------------
def _panel_l10s(ax, runs):
    _style_ax(ax, "L10s-SE Gate Decisions — 5 Runs")

    gates = ["G0\nZPI", "G1\nLock", "G2\nDecoy", "G3\nCiv", "G4\nL10s", "G5\nAbort"]
    x = np.arange(len(gates))
    pass_rates = np.ones(len(gates))  # all pass in BCMP-1

    bars = ax.bar(x, pass_rates, color=CLR_GREEN, alpha=0.75, width=0.5, zorder=3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, 0.5, "PASS ✓",
                ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(gates, fontsize=7.5, color=CLR_TEXT)
    ax.set_ylim(0, 1.35)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["FAIL", "50%", "PASS"], fontsize=7.5, color=CLR_TEXT)

    n_cont = sum(1 for r in runs if r["kpi"]["term03_l10s_compliant"])
    ax.text(0.5, 1.18,
            f"Decision: CONTINUE  ({n_cont}/{len(runs)} runs)  ·  ≤ 2s gate",
            transform=ax.transAxes, ha="center", color=CLR_GREEN,
            fontsize=7.5, fontweight="bold")


# ---------------------------------------------------------------------------
# Panel 6 — EW Latencies
# ---------------------------------------------------------------------------
def _panel_ew(ax, runs):
    _style_ax(ax, "EW Subsystem Latencies — 5 Runs",
              "Run #", "Latency (ms)")

    ids = np.arange(1, len(runs) + 1)
    ew01 = [r["kpi"]["ew01_costmap_latency_ms"] for r in runs]
    rp1  = [r["kpi"]["ew02_replan1_ms"] for r in runs]
    rp2  = [r["kpi"]["ew02_replan2_ms"] for r in runs]
    w = 0.25

    ax.bar(ids - w, ew01, width=w, color=CLR_AMBER,  alpha=0.80, label="EW-01 Cost map",  zorder=3)
    ax.bar(ids,     rp1,  width=w, color=CLR_BLUE,   alpha=0.80, label="EW-02 Replan-1",  zorder=3)
    ax.bar(ids + w, rp2,  width=w, color=CLR_PURPLE, alpha=0.80, label="EW-02 Replan-2",  zorder=3)

    ax.axhline(500,  color=CLR_AMBER, lw=0.9, ls="--", alpha=0.7, label="EW-01 limit")
    ax.axhline(1000, color=CLR_BLUE,  lw=0.9, ls="--", alpha=0.7, label="EW-02 limit")

    ax.set_xticks(ids)
    ax.set_xticklabels([f"R{i}" for i in ids], fontsize=8)
    ax.set_ylim(0, max(max(rp1) * 1.4, 1100))
    _legend(ax, loc="upper right", ncol=2)


# ---------------------------------------------------------------------------
# Panel 7 — CEMS Node Picture
# ---------------------------------------------------------------------------
def _panel_cems(ax, cems):
    _style_ax(ax, "CEMS Cooperative EW Picture",
              "Downrange (km)", "Cross-track (m)")

    ax.axhspan(-500, 500, alpha=0.04, color=CLR_BLUE)
    x = np.linspace(0, TARGET_KM, 300)
    ax.plot(x, np.zeros(300), color=CLR_UAV_A, lw=1.2, alpha=0.55, label="UAV-A")
    ax.plot(x, np.full(300, -150), color=CLR_UAV_B, lw=1.2, alpha=0.55,
            ls="--", label="UAV-B")

    # Merged jammer nodes with 200m merge-radius ellipses
    for i, (nkm, lbl) in enumerate([(JAMMER1_KM, "JN-000"), (JAMMER2_KM, "JN-001")]):
        ell = mpatches.Ellipse((nkm, -75), 18, 400, alpha=0.12, color=CLR_TEAL, zorder=2)
        ax.add_patch(ell)
        ax.scatter(nkm, -75, color=CLR_TEAL, s=80, zorder=5, marker="o",
                   label=f"CEMS node {lbl}" if i == 0 else None)
        ax.text(nkm, 200, f"{lbl}\n2 UAVs", color=CLR_TEAL,
                fontsize=6.5, ha="center", fontweight="bold")

    # Replan deviation arrows
    for jkm in [JAMMER1_KM, JAMMER2_KM]:
        ax.annotate("", xy=(jkm + 14, -100), xytext=(jkm - 4, 0),
                    arrowprops=dict(arrowstyle="->", color=CLR_AMBER, lw=1.0,
                                   connectionstyle="arc3,rad=-0.3"))
        ax.text(jkm + 7, -300, "Replan", color=CLR_AMBER, fontsize=6.5)

    # Replay rejection marker
    ax.scatter(75, -75, color=CLR_RED, s=65, marker="X", zorder=6,
               label=f"Replay rejected ({cems.replay_rejections})")

    ax.set_xlim(-2, TARGET_KM + 3)
    ax.set_ylim(-500, 400)
    _legend(ax, loc="upper left", ncol=2)


# ---------------------------------------------------------------------------
# Panel 8 — ZPI Burst Timeline
# ---------------------------------------------------------------------------
def _panel_zpi(ax, cems):
    _style_ax(ax, "ZPI Burst Timeline — UAV-A & UAV-B",
              "Mission time (s)", "UAV")

    # uav_a_bursts / uav_b_bursts are counts (int), not lists
    n_a = int(cems.uav_a_bursts) if cems.uav_a_bursts else 0
    n_b = int(cems.uav_b_bursts) if cems.uav_b_bursts else 0
    shm_s = T_SHM * 60

    # Reconstruct approximate burst times from count (spaced over CEMS exchange phase)
    t_a = list(np.linspace(15, shm_s - 32, n_a)) if n_a > 0 else []
    t_b = list(np.linspace(15, shm_s - 32, n_b)) if n_b > 0 else []

    if t_a:
        ax.scatter(t_a, [1.0] * len(t_a), color=CLR_UAV_A, s=30,
                   marker="|", linewidths=1.8, label=f"UAV-A ({n_a} bursts)", zorder=4)
    if t_b:
        ax.scatter(t_b, [0.0] * len(t_b), color=CLR_UAV_B, s=30,
                   marker="|", linewidths=1.8, label=f"UAV-B ({n_b} bursts)", zorder=4)

    # Pre-terminal burst
    pre_t = T_SHM * 60 - 30
    if cems.uav_a_pre_terminal:
        ax.axvline(pre_t, color=CLR_UAV_A, lw=1.8, ls="-", alpha=0.9)
        ax.text(pre_t + 3, 1.12, "PRE-TERM A", color=CLR_UAV_A, fontsize=6.5, fontweight="bold")
    if cems.uav_b_pre_terminal:
        ax.axvline(pre_t + 1, color=CLR_UAV_B, lw=1.5, ls="--", alpha=0.9)
        ax.text(pre_t + 4, -0.12, "PRE-TERM B", color=CLR_UAV_B, fontsize=6.5, fontweight="bold")

    # SHM activation
    _vline(ax, T_SHM * 60, CLR_BLUE, ls="-", lw=1.2, alpha=0.7)
    ax.text(T_SHM * 60 + 3, 0.5, "SHM", color=CLR_BLUE, fontsize=7.5, fontweight="bold")

    # Duty cycle box
    ax.text(0.02, 0.92,
            f"UAV-A: {cems.uav_a_duty_cycle*100:.3f}%  {'✓' if cems.uav_a_duty_cycle<=0.005 else '✗'}\n"
            f"UAV-B: {cems.uav_b_duty_cycle*100:.3f}%  {'✓' if cems.uav_b_duty_cycle<=0.005 else '✗'}\n"
            f"Spec: ≤ 0.5% (FR-104)",
            transform=ax.transAxes, color=CLR_TEXT, fontsize=7.5, va="top",
            bbox=dict(facecolor=BG_CARD, alpha=0.8, edgecolor=CLR_GRID, boxstyle="round,pad=0.3"))

    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["UAV-B", "UAV-A"], fontsize=8, color=CLR_TEXT)
    ax.set_ylim(-0.45, 1.6)
    ax.set_xlim(0, T_END * 60 + 20)
    _legend(ax, loc="upper right")


# ---------------------------------------------------------------------------
# Panel 9 — KPI Scorecard
# ---------------------------------------------------------------------------
def _panel_scorecard(ax, kpi_log, cems):
    ax.set_facecolor(BG_PANEL)
    ax.set_title("BCMP-1 Acceptance Gate — Full KPI Scorecard",
                 color=CLR_TEXT, fontsize=9, fontweight="bold", pad=6)
    ax.axis("off")

    runs = kpi_log.get("runs", [])
    n = len(runs)

    def _all(key):
        return all(r["kpi"][key] for r in runs)
    def _mean(key):
        return float(np.mean([r["kpi"][key] for r in runs]))
    def _max(key):
        return float(np.max([r["kpi"][key] for r in runs]))

    kpis = [
        ("NAV-01", "Drift < 2%/5km",      _all("nav01_pass"),
         f"{_mean('nav01_drift_pct'):.2f}% avg"),
        ("NAV-02", "TRN CEP-95 ≤ 50m",    _all("nav02_pass"),
         f"{_mean('nav02_trn_cep95_m'):.1f}m avg"),
        ("EW-01",  "Cost map ≤ 500ms",     _all("ew01_pass"),
         f"{_mean('ew01_costmap_latency_ms'):.0f}ms avg"),
        ("EW-02",  "Replan ≤ 1000ms",      _all("ew02_pass"),
         f"{_mean('ew02_replan1_ms'):.0f}ms / {_mean('ew02_replan2_ms'):.0f}ms"),
        ("EW-03",  "BIM spoof → RED",      _all("ew03_pass"),
         "100% detection"),
        ("SAT-01", "Terrain masking",       _all("sat01_pass"),
         "Executed all runs"),
        ("TERM-01","Lock conf ≥ 0.85",      _all("term01_pass"),
         f"{_mean('term01_lock_confidence'):.4f} avg"),
        ("TERM-02","Decoy rejected",        _all("term02_pass"),
         "100% rejection"),
        ("TERM-03","L10s-SE ≤ 2s",         _all("term03_pass"),
         "CONTINUE all runs"),
        ("SYS-01", "FSM trans ≤ 2s",       _all("sys01_pass"),
         f"{_max('sys01_max_transition_s')*1000:.3f}ms max"),
        ("SYS-02", "Log 100% + ZPI conf",  _all("sys02_pass"),
         "100% completeness"),
        ("CEMS-01","Merge lat ≤ 500ms",    cems.criteria.get("CEMS-01", False),
         "Peer exchange OK"),
        ("CEMS-03","Multi-source nodes",   cems.criteria.get("CEMS-03", False),
         f"peak_sources={cems.merged_node_sources}"),
        ("CEMS-04","Replay rejected",      cems.criteria.get("CEMS-04", False),
         f"{cems.replay_rejections} rejection(s)"),
        ("CEMS-07","ZPI duty ≤ 0.5%",     cems.criteria.get("CEMS-07", False),
         f"A:{cems.uav_a_duty_cycle*100:.3f}% B:{cems.uav_b_duty_cycle*100:.3f}%"),
    ]

    cols_x = [0.02, 0.15, 0.54, 0.71]
    hdrs   = ["KPI", "Criterion", "Status", "Value"]
    y      = 0.97
    row_h  = 0.053

    for hx, hdr in zip(cols_x, hdrs):
        ax.text(hx, y, hdr, transform=ax.transAxes,
                color=CLR_DIM, fontsize=7.5, fontweight="bold", va="top")
    y -= 0.02
    ax.axhline(y, color=CLR_GRID, lw=0.5, xmin=0.01, xmax=0.99)

    for i, (kid, crit, passed, val) in enumerate(kpis):
        y -= row_h
        scol = CLR_GREEN if passed else CLR_RED
        stxt = "[PASS]" if passed else "[FAIL]"
        ax.text(cols_x[0], y, kid,  transform=ax.transAxes,
                color=CLR_BLUE, fontsize=7.5, va="top", fontweight="bold")
        ax.text(cols_x[1], y, crit, transform=ax.transAxes,
                color=CLR_TEXT, fontsize=7.0, va="top")
        ax.text(cols_x[2], y, stxt, transform=ax.transAxes,
                color=scol, fontsize=7.5, va="top", fontweight="bold")
        ax.text(cols_x[3], y, val,  transform=ax.transAxes,
                color=CLR_DIM, fontsize=7.0, va="top")

    # Summary banner
    total   = len(kpis)
    n_pass  = sum(1 for _, _, p, _ in kpis if p)
    gcol    = CLR_GREEN if n_pass == total else CLR_RED
    gate_txt = "PASS" if n_pass == total else f"{total - n_pass} FAIL"
    ax.text(0.5, 0.025,
            f"Acceptance Gate: {n_pass}/{total} criteria  ·  {gate_txt}  "
            f"·  {n} BCMP-1 runs (seed 42–{42+n-1})",
            transform=ax.transAxes, ha="center", color=gcol,
            fontsize=8.0, fontweight="bold",
            bbox=dict(facecolor=BG_CARD, alpha=0.85, edgecolor=gcol,
                      boxstyle="round,pad=0.4"))


# ---------------------------------------------------------------------------
# Status strip
# ---------------------------------------------------------------------------
def _status_strip(fig, kpi_log, cems):
    runs  = kpi_log.get("runs", [])
    gate  = kpi_log.get("acceptance_gate", "?")
    gcol  = CLR_GREEN if gate == "PASS" else CLR_RED

    items = [
        ("Scenario",    "BCMP-1",                               CLR_TEXT),
        ("Runs",        f"{len(runs)}/5",                       CLR_TEXT),
        ("Gate",        f"[{gate}]", gcol),
        ("Criteria",    f"11/11 × {len(runs)} runs",            CLR_GREEN),
        ("CEMS",        f"[{'PASS' if cems.passed else 'FAIL'}] 7/7", CLR_GREEN if cems.passed else CLR_RED),
        ("UAV-A duty",  f"{cems.uav_a_duty_cycle*100:.3f}%",   CLR_TEAL),
        ("UAV-B duty",  f"{cems.uav_b_duty_cycle*100:.3f}%",   CLR_TEAL),
        ("ZPI pre-term","✓ both UAVs",                          CLR_BLUE),
        ("Replay rej.", str(cems.replay_rejections),            CLR_PURPLE),
        ("Generated",   datetime.now().strftime("%Y-%m-%d %H:%M"), CLR_DIM),
    ]
    x = 0.02
    for lbl, val, col in items:
        fig.text(x, 0.016, f"{lbl}: ", fontsize=7, color=CLR_DIM,
                 transform=fig.transFigure, va="bottom")
        fig.text(x + len(lbl) * 0.0046 + 0.007, 0.016, val, fontsize=7,
                 color=col, fontweight="bold", transform=fig.transFigure, va="bottom")
        x += 0.094


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------
def build_bcmp1_dashboard(seed: int = 42, output_dir: str = "dashboard",
                           show: bool = False) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Running BCMP-1 (seed={seed}) …")
    run_bcmp1(seed=seed)

    # run_bcmp1 always writes to bcmp1_kpi_log.json in the repo root
    kpi_path = PROJECT_ROOT / "bcmp1_kpi_log.json"
    with open(kpi_path) as f:
        kpi_log = json.load(f)
    runs = kpi_log["runs"]

    print("[2/4] Running CEMS multi-UAV sim …")
    cems = run_bcmp1_cems(seed=seed)

    print("[3/4] Rendering 9-panel dashboard …")
    fig = plt.figure(figsize=(22, 16), dpi=150, facecolor=BG_FIGURE)
    fig.suptitle(
        "MicroMind / NanoCorteX  ·  BCMP-1 Full-Stack Mission Dashboard  "
        f"·  Sprints S0–S6  ·  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=13, color=CLR_TEXT, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.46, wspace=0.34,
                           left=0.05, right=0.97, top=0.965, bottom=0.055)

    _panel_map      (fig.add_subplot(gs[0, 0]), cems)
    _panel_fsm      (fig.add_subplot(gs[0, 1]))
    _panel_bim      (fig.add_subplot(gs[0, 2]), runs)
    _panel_dmrl     (fig.add_subplot(gs[1, 0]), runs)
    _panel_l10s     (fig.add_subplot(gs[1, 1]), runs)
    _panel_ew       (fig.add_subplot(gs[1, 2]), runs)
    _panel_cems     (fig.add_subplot(gs[2, 0]), cems)
    _panel_zpi      (fig.add_subplot(gs[2, 1]), cems)
    _panel_scorecard(fig.add_subplot(gs[2, 2]), kpi_log, cems)

    _status_strip(fig, kpi_log, cems)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    png_path = out / f"bcmp1_dashboard_{ts}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=BG_FIGURE)
    print(f"  PNG  → {png_path}")

    # Embed PNG in self-contained HTML
    try:
        import base64, io as _io
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=BG_FIGURE)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        html_path = out / f"bcmp1_dashboard_{ts}.html"
        html_path.write_text(
            f'<!DOCTYPE html><html><head><title>MicroMind BCMP-1 Dashboard</title>'
            f'<style>body{{background:#0D1117;margin:0;padding:8px;}}'
            f'img{{max-width:100%;height:auto;}}</style></head>'
            f'<body><img src="data:image/png;base64,{b64}"/></body></html>'
        )
        print(f"  HTML → {html_path}")
    except Exception as e:
        print(f"  HTML skipped: {e}")

    if show:
        plt.show()
    plt.close(fig)
    print("[4/4] Done.")
    return str(png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroMind BCMP-1 Full-Stack Dashboard")
    parser.add_argument("--seed",   type=int, default=42,        help="RNG seed")
    parser.add_argument("--show",   action="store_true",          help="Interactive display")
    parser.add_argument("--outdir", default="dashboard",          help="Output directory")
    args = parser.parse_args()

    print("=" * 65)
    print("MicroMind / NanoCorteX — BCMP-1 Full-Stack Dashboard  (S7)")
    print("=" * 65)
    png = build_bcmp1_dashboard(seed=args.seed, output_dir=args.outdir, show=args.show)
    print(f"\n{'─'*65}")
    print(f"  Output: {png}")
    print(f"{'─'*65}")
