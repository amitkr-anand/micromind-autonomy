"""
dashboard/bcmp2_dashboard.py
MicroMind / NanoCorteX — BCMP-2 Dual-Track Comparative Dashboard

Runs a full BCMP-2 dual-track mission (both vehicles) and renders a
7-panel dark-theme dashboard showing the comparative outcome.

Panels:
  Row 1: [P1] Dual-Track Route Map   [P2] Vehicle A Drift Profile
                                     [P3] Vehicle B Nav Timeline
  Row 2: [P4] BIM Trust Proxy        [P5] Disturbance Schedule
                                     [P6] C-2 Envelope Gates
  Row 3: [P7] Outcome Summary  ← full width, always visible

Panel 7 (Outcome Summary) is the programme-visible verdict panel.
It mirrors the business comparison block from bcmp2_report.py.

Outputs (saved to dashboard/):
  bcmp2_dashboard_seed<N>_<timestamp>.png  — 150 dpi static figure
  bcmp2_dashboard_seed<N>_<timestamp>.html — self-contained HTML

Run:
    cd micromind-autonomy
    PYTHONPATH=. python3 dashboard/bcmp2_dashboard.py
    PYTHONPATH=. python3 dashboard/bcmp2_dashboard.py --seed 101
    PYTHONPATH=. python3 dashboard/bcmp2_dashboard.py --seed 42 --km 150

JOURNAL
-------
Built: 31 March 2026, micromind-node01.  SB-4 Step 1.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scenarios.bcmp2.bcmp2_runner import run_bcmp2, BCMP2RunConfig
from scenarios.bcmp2.bcmp2_drift_envelopes import (
    PHASE_ENVELOPES, VEHICLE_SPEED_MS,
)

# ---------------------------------------------------------------------------
# Colour palette  (matches bcmp1_dashboard — GitHub-dark)
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
CLR_ORANGE = "#FF7B72"

CLR_VA = CLR_RED    # Vehicle A — baseline (degraded)
CLR_VB = CLR_GREEN  # Vehicle B — MicroMind (bounded)


def _style(ax, title: str = "") -> None:
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=CLR_DIM, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(CLR_GRID)
    ax.xaxis.label.set_color(CLR_DIM)
    ax.yaxis.label.set_color(CLR_DIM)
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold",
                     color=CLR_TEXT, pad=5)
    ax.grid(True, color=CLR_GRID, linewidth=0.4, alpha=0.7)


def _km_axis(ax, max_km: float = 150.0) -> None:
    ax.set_xlim(0, max_km)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.set_xlabel("Mission distance (km)", fontsize=7)


def _denial_km(sched: dict) -> float:
    return sched["gnss_denial"]["start_s"] * VEHICLE_SPEED_MS / 1000.0


# ---------------------------------------------------------------------------
# Panel 1 — Dual-Track Route Map
# ---------------------------------------------------------------------------

def _p1_route(ax, run_out: dict) -> None:
    _style(ax, "P1 — Dual-Track Route Map")
    va     = run_out["vehicle_a"]
    sched  = run_out["disturbance_schedule"]
    states = va.get("states", [])
    dk     = _denial_km(sched)

    if states:
        tn = [s["true_north_m"] / 1000 for s in states]
        te = [s["true_east_m"]  / 1000 for s in states]
        an = [s.get("north_m", s["true_north_m"]) / 1000 for s in states]
        ae = [s.get("east_m",  s["true_east_m"])  / 1000 for s in states]

        ax.plot(te, tn, color=CLR_GRID, lw=0.8, ls="--",
                label="Planned route")
        ax.plot(te, tn, color=CLR_VB, lw=1.5, ls="-.",
                label="Vehicle B (MicroMind)", alpha=0.9)
        ax.plot(ae, an, color=CLR_VA, lw=1.0,
                label="Vehicle A (baseline)", alpha=0.8)

        # GNSS denial start marker on truth track
        di = min(range(len(states)),
                 key=lambda i: abs(states[i]["mission_km"] - dk))
        ax.scatter([te[di]], [tn[di]], color=CLR_AMBER, s=55,
                   zorder=5, label=f"GNSS denial km {dk:.0f}")

    ax.set_xlabel("East (km)", fontsize=7)
    ax.set_ylabel("North (km)", fontsize=7)
    ax.legend(loc="upper left", fontsize=6, facecolor=BG_CARD,
              labelcolor=CLR_TEXT, edgecolor=CLR_GRID, framealpha=0.9)


# ---------------------------------------------------------------------------
# Panel 2 — Vehicle A Drift Profile vs C-2 Envelope
# ---------------------------------------------------------------------------

def _p2_drift(ax, run_out: dict) -> None:
    _style(ax, "P2 — Vehicle A Drift vs C-2 Envelope")
    va     = run_out["vehicle_a"]
    sched  = run_out["disturbance_schedule"]
    states = va.get("states", [])
    dk     = _denial_km(sched)
    max_km = run_out.get("max_km", 150.0)

    if states:
        kms = [s["mission_km"] for s in states]
        # cross_track_m is the signed lateral error from baseline_nav_sim
        drifts = [abs(s.get("cross_track_m",
                       s.get("lateral_error_m", 0))) for s in states]
        ax.plot(kms, drifts, color=CLR_VA, lw=1.2, label="Vehicle A drift")

    # Shade C-2 envelope bands
    prev = dk
    for gkm in sorted(PHASE_ENVELOPES.keys()):
        env = PHASE_ENVELOPES[gkm]
        xs  = [prev, gkm, gkm, prev]
        ax.fill_between([prev, gkm],
                        [env["floor"],   env["floor"]],
                        [env["ceiling"], env["ceiling"]],
                        color=CLR_GREEN, alpha=0.07)
        ax.plot([prev, gkm], [env["ceiling"]] * 2,
                color=CLR_RED, lw=0.7, ls="--", alpha=0.6)
        ax.plot([prev, gkm], [env["floor"]] * 2,
                color=CLR_GREEN, lw=0.7, ls="--", alpha=0.6)
        ax.axvline(gkm, color=CLR_GRID, lw=0.5, ls=":")
        prev = gkm

    ax.axvline(dk, color=CLR_AMBER, lw=1.0, ls="--", alpha=0.8)
    ax.text(dk + 1, 10, "GNSS\ndenied", fontsize=6, color=CLR_AMBER)

    # Breach marker
    breach = va.get("first_corridor_violation_km")
    if breach:
        ax.axvline(breach, color=CLR_RED, lw=1.2, alpha=0.9)
        ax.text(breach + 1, 5, f"Breach\nkm {breach:.0f}",
                fontsize=6, color=CLR_RED)

    # Gate point labels
    for attr, gkm in [("drift_at_km60_m", 60),
                      ("drift_at_km100_m", 100),
                      ("drift_at_km120_m", 120)]:
        val = va.get(attr)
        if val:
            env = PHASE_ENVELOPES[gkm]
            col = CLR_GREEN if env["floor"] <= val <= env["ceiling"] else CLR_RED
            ax.scatter([gkm], [val], color=col, s=35, zorder=5)
            ax.text(gkm + 1, val, f"{val:.0f}m", fontsize=6,
                    color=col, va="center")

    _km_axis(ax, max_km)
    ax.set_ylabel("Lateral drift (m)", fontsize=7)
    ax.legend(loc="upper left", fontsize=6, facecolor=BG_CARD,
              labelcolor=CLR_TEXT, edgecolor=CLR_GRID, framealpha=0.9)


# ---------------------------------------------------------------------------
# Panel 3 — Vehicle B Navigation Source Timeline
# ---------------------------------------------------------------------------

def _p3_nav_timeline(ax, run_out: dict) -> None:
    _style(ax, "P3 — Vehicle B Nav Source Timeline")
    sched  = run_out["disturbance_schedule"]
    vb     = run_out["vehicle_b"]
    max_km = run_out.get("max_km", 150.0)
    dk     = _denial_km(sched)

    # Source bands (stacked swimlanes)
    ax.barh(0.75, dk,        height=0.22, left=0,
            color=CLR_GREEN, alpha=0.75, label="GNSS + fusion")
    ax.barh(0.75, max_km-dk, height=0.22, left=dk,
            color=CLR_AMBER, alpha=0.75, label="TRN / VIO only")

    for i, outage in enumerate(sched.get("vio_outages", [])):
        skm  = outage["start_s"]    * VEHICLE_SPEED_MS / 1000.0
        dkm2 = outage["duration_s"] * VEHICLE_SPEED_MS / 1000.0
        ax.barh(0.75, dkm2, height=0.22, left=skm,
                color=CLR_RED, alpha=0.85,
                label="VIO outage" if i == 0 else "")

    # RADALT / EO events
    for key, col, label in [("radalt_loss_s", CLR_PURPLE, "RADALT loss"),
                              ("eo_freeze_s",  CLR_ORANGE, "EO freeze")]:
        t = sched.get(key)
        if t:
            km = t * VEHICLE_SPEED_MS / 1000.0
            ax.axvline(km, color=col, lw=1.0, ls=":", alpha=0.8)
            ax.text(km + 1, 0.5, label, fontsize=5.5, color=col, va="center")

    # KPI summary text
    kpi_items = [(k, v) for k, v in vb.items() if v is not None]
    y = 0.40
    for k, v in kpi_items[:5]:
        lbl = k.replace("_", " ").title()
        if isinstance(v, bool):
            sym, col = ("✓", CLR_GREEN) if v else ("✗", CLR_RED)
            ax.text(2, y, f"{sym} {lbl}", fontsize=6, color=col,
                    transform=ax.get_xaxis_transform())
        else:
            ax.text(2, y, f"  {lbl}: {v}", fontsize=6, color=CLR_BLUE,
                    transform=ax.get_xaxis_transform())
        y -= 0.065

    _km_axis(ax, max_km)
    ax.set_ylim(0.1, 1.0)
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=6, facecolor=BG_CARD,
              labelcolor=CLR_TEXT, edgecolor=CLR_GRID, framealpha=0.9)


# ---------------------------------------------------------------------------
# Panel 4 — BIM Trust Score (proxy curve)
# ---------------------------------------------------------------------------

def _p4_bim(ax, run_out: dict) -> None:
    _style(ax, "P4 — BIM Trust Score (SIL proxy)")
    sched  = run_out["disturbance_schedule"]
    max_km = run_out.get("max_km", 150.0)
    dk     = _denial_km(sched)
    d_dur  = sched["gnss_denial"].get("duration_s", 9999)
    d_end  = min(dk + d_dur * VEHICLE_SPEED_MS / 1000.0, max_km)

    km    = np.linspace(0, max_km, 600)
    trust = np.ones(600)

    denied = (km >= dk) & (km <= d_end)
    t_loc  = km[denied] - dk
    trust[denied] = np.clip(1.0 - 0.88 * (1 - np.exp(-t_loc / 6)), 0.05, 1.0)
    if d_end < max_km:
        recovery = km > d_end
        trust[recovery] = np.clip(
            0.1 + 0.7 * (km[recovery] - d_end) / 20, 0.1, 0.8)

    for outage in sched.get("vio_outages", []):
        skm  = outage["start_s"]    * VEHICLE_SPEED_MS / 1000.0
        ekm  = skm + outage["duration_s"] * VEHICLE_SPEED_MS / 1000.0
        mask = (km >= skm) & (km <= ekm)
        trust[mask] *= 0.65

    ax.fill_between(km, trust, alpha=0.20, color=CLR_BLUE)
    ax.plot(km, trust, color=CLR_BLUE, lw=1.2)
    ax.axhline(0.10, color=CLR_RED,   lw=0.9, ls="--", alpha=0.7, label="RED threshold")
    ax.axhline(0.50, color=CLR_AMBER, lw=0.9, ls="--", alpha=0.7, label="AMBER threshold")
    ax.axvline(dk,   color=CLR_AMBER, lw=1.0, ls="--", alpha=0.8)
    ax.text(dk + 1, 0.92, "denial", fontsize=6, color=CLR_AMBER)
    ax.set_ylim(0, 1.05)
    _km_axis(ax, max_km)
    ax.set_ylabel("Trust score (0–1)", fontsize=7)
    ax.legend(loc="upper right", fontsize=6, facecolor=BG_CARD,
              labelcolor=CLR_TEXT, edgecolor=CLR_GRID, framealpha=0.9)
    ax.text(4, 0.02,
            "Synthetic proxy — actual BIM requires SITL sensor bus",
            fontsize=5.5, color=CLR_DIM, style="italic")


# ---------------------------------------------------------------------------
# Panel 5 — Disturbance Schedule (C-4)
# ---------------------------------------------------------------------------

def _p5_disturbance(ax, run_out: dict) -> None:
    _style(ax, "P5 — Disturbance Schedule (C-4)")
    sched  = run_out["disturbance_schedule"]
    max_km = run_out.get("max_km", 150.0)

    rows    = ["EO freeze", "RADALT loss", "VIO outage", "GNSS denial"]
    colours = [CLR_ORANGE, CLR_PURPLE, CLR_BLUE, CLR_AMBER]
    y_vals  = np.arange(len(rows))

    # GNSS
    d    = sched["gnss_denial"]
    dkm  = d["start_s"] * VEHICLE_SPEED_MS / 1000.0
    ddur = d.get("duration_s", 9999) * VEHICLE_SPEED_MS / 1000.0
    ax.barh(3, min(ddur, max_km - dkm), height=0.35, left=dkm,
            color=colours[3], alpha=0.85)
    ax.text(dkm + 0.5, 3, f"km {dkm:.0f}→", fontsize=6,
            color=colours[3], va="center")

    # VIO outages (all)
    for i, o in enumerate(sched.get("vio_outages", [])):
        skm  = o["start_s"]    * VEHICLE_SPEED_MS / 1000.0
        dkm2 = o["duration_s"] * VEHICLE_SPEED_MS / 1000.0
        ax.barh(2, dkm2, height=0.35, left=skm, color=colours[2], alpha=0.85)

    # RADALT
    if sched.get("radalt_loss_s"):
        km = sched["radalt_loss_s"] * VEHICLE_SPEED_MS / 1000.0
        ax.barh(1, max_km - km, height=0.35, left=km,
                color=colours[1], alpha=0.85)

    # EO freeze
    if sched.get("eo_freeze_s"):
        km = sched["eo_freeze_s"] * VEHICLE_SPEED_MS / 1000.0
        ax.barh(0, max_km - km, height=0.35, left=km,
                color=colours[0], alpha=0.85)

    _km_axis(ax, max_km)
    ax.set_yticks(y_vals)
    ax.set_yticklabels(rows, fontsize=7, color=CLR_TEXT)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.grid(axis="x", color=CLR_GRID, linewidth=0.4, alpha=0.7)
    ax.grid(axis="y", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Panel 6 — C-2 Envelope Gate Results
# ---------------------------------------------------------------------------

def _p6_c2gates(ax, run_out: dict) -> None:
    _style(ax, "P6 — C-2 Gates (Vehicle A)")
    va    = run_out["vehicle_a"]
    gates = va.get("c2_gates", {})

    sorted_kms = sorted(gates.keys())
    xp         = np.arange(len(sorted_kms))
    max_ceil   = max((PHASE_ENVELOPES[k]["ceiling"] for k in sorted_kms), default=1000)

    for i, km in enumerate(sorted_kms):
        env  = PHASE_ENVELOPES[km]
        gate = gates[km]
        obs  = gate.get("observed_m") or 0
        ok   = gate.get("passed", False)
        col  = CLR_GREEN if ok else (CLR_DIM if not obs else CLR_RED)

        # Envelope range shading
        ax.barh(xp[i], env["ceiling"] - env["floor"],
                height=0.35, left=env["floor"],
                color=CLR_GREEN, alpha=0.12)
        ax.plot([env["floor"]] * 2,
                [xp[i] - 0.2, xp[i] + 0.2], color=CLR_GREEN, lw=1.0)
        ax.plot([env["ceiling"]] * 2,
                [xp[i] - 0.2, xp[i] + 0.2], color=CLR_RED, lw=1.0, ls="--")

        if obs:
            ax.scatter([obs], [xp[i]], color=col, s=60, zorder=5)
            ax.text(obs + max_ceil * 0.02, xp[i],
                    f"{obs:.0f} m", fontsize=7, color=col, va="center")

        label = "PASS" if ok else ("N/A" if not obs else "FAIL")
        ax.text(max_ceil * 1.22, xp[i], label, fontsize=7,
                fontweight="bold", color=col, va="center", ha="right")

    ax.set_yticks(xp)
    ax.set_yticklabels([f"km {km}" for km in sorted_kms],
                       fontsize=8, color=CLR_TEXT)
    ax.set_xlabel("Lateral drift (m)", fontsize=7)
    ax.set_xlim(0, max_ceil * 1.3)
    ax.grid(axis="x", color=CLR_GRID, linewidth=0.4, alpha=0.7)
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------------------
# Panel 7 — Outcome Summary (full-width, programme-visible verdict)
# ---------------------------------------------------------------------------

def _p7_outcome(ax, run_out: dict) -> None:
    ax.set_facecolor(BG_CARD)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("P7 — Mission Outcome Summary",
                 fontsize=9, fontweight="bold", color=CLR_TEXT, pad=6)

    comp    = run_out.get("comparison", {})
    va_res  = comp.get("vehicle_a_mission_result", "UNKNOWN")
    vb_res  = comp.get("vehicle_b_mission_result", "UNKNOWN")
    seed    = run_out.get("seed", "?")
    denial  = comp.get("gnss_denial_km", 30)
    a_drift = comp.get("vehicle_a_drift_km120_m")
    b_drift = comp.get("vehicle_b_max_5km_drift_m")
    breach  = comp.get("vehicle_a_first_corridor_violation_km")
    a_chain = comp.get("vehicle_a_causal_chain", [])
    b_chain = comp.get("vehicle_b_causal_chain", [])

    va_col  = CLR_RED    if "FAIL" in va_res else CLR_AMBER
    vb_col  = CLR_GREEN  if "SUCC" in vb_res else CLR_AMBER

    # Left column — Vehicle A
    ax.text(0.01, 0.95, "WITHOUT MicroMind (Vehicle A):",
            transform=ax.transAxes, fontsize=8, fontweight="bold",
            color=CLR_TEXT, va="top")
    ax.text(0.01, 0.82, f"  {va_res}",
            transform=ax.transAxes, fontsize=13, fontweight="bold",
            color=va_col, va="top")
    y = 0.68
    for item in a_chain[:4]:
        ax.text(0.01, y, f"  → {item}", transform=ax.transAxes,
                fontsize=6.5, color=va_col, va="top", alpha=0.9)
        y -= 0.14
    if a_drift:
        ax.text(0.01, 0.08,
                f"  Drift @ km 120: {a_drift:.0f} m"
                + (f"  ·  Corridor breach: km {breach:.0f}" if breach else "  ·  No corridor breach"),
                transform=ax.transAxes, fontsize=7, color=va_col, va="bottom")

    # Divider
    ax.plot([0.5, 0.5], [0.05, 0.95], color=CLR_GRID, lw=1.0, transform=ax.transAxes)



    # Right column — Vehicle B
    ax.text(0.52, 0.95, "WITH MicroMind (Vehicle B):",
            transform=ax.transAxes, fontsize=8, fontweight="bold",
            color=CLR_TEXT, va="top")
    ax.text(0.52, 0.82, f"  {vb_res}",
            transform=ax.transAxes, fontsize=13, fontweight="bold",
            color=vb_col, va="top")
    y = 0.68
    for item in b_chain[:4]:
        ax.text(0.52, y, f"  → {item}", transform=ax.transAxes,
                fontsize=6.5, color=vb_col, va="top", alpha=0.9)
        y -= 0.14
    if b_drift:
        ax.text(0.52, 0.08, f"  Max 5km drift bounded: {b_drift:.1f} m",
                transform=ax.transAxes, fontsize=7, color=vb_col, va="bottom")

    # Border colour reflects outcome
    border_col = CLR_RED if "FAIL" in va_res else CLR_AMBER
    for spine in ax.spines.values():
        spine.set_edgecolor(border_col)
        spine.set_linewidth(2.5)


# ---------------------------------------------------------------------------
# Status strip
# ---------------------------------------------------------------------------

def _status_strip(fig, run_out: dict) -> None:
    comp = run_out.get("comparison", {})
    ts   = datetime.now().strftime("%d %b %Y %H:%M IST")
    fig.text(
        0.01, 0.993,
        f"MicroMind / NanoCorteX — BCMP-2 Dashboard  |  "
        f"Seed {run_out.get('seed','?')}  |  "
        f"{run_out.get('hardware_source','simulated')}  |  "
        f"IMU: {run_out.get('imu_model','STIM300')}  |  "
        f"A: {comp.get('vehicle_a_mission_result','?')}   "
        f"B: {comp.get('vehicle_b_mission_result','?')}  |  "
        f"Run: {run_out.get('run_duration_s',0):.1f}s  |  {ts}",
        fontsize=6.5, color=CLR_DIM, va="top",
        transform=fig.transFigure,
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_bcmp2_dashboard(
    seed:       int   = 42,
    max_km:     float = 150.0,
    output_dir: str   = "dashboard",
    show:       bool  = False,
    verbose:    bool  = True,
) -> tuple[str, str]:
    """
    Run BCMP-2 dual-track and render 7-panel comparative dashboard.

    Returns (png_path, html_path).
    """
    if verbose:
        print(f"[1/4] Running BCMP-2 dual-track  "
              f"seed={seed}  max_km={max_km} ...")

    config  = BCMP2RunConfig(seed=seed, max_km=max_km, verbose=False)
    run_out = run_bcmp2(config)

    if verbose:
        comp = run_out["comparison"]
        print(f"      Vehicle A: {comp['vehicle_a_mission_result']}")
        print(f"      Vehicle B: {comp['vehicle_b_mission_result']}")
        print("[2/4] Building 7 panels ...")

    fig = plt.figure(figsize=(18, 12), facecolor=BG_FIGURE)
    gs  = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.40, wspace=0.28,
        left=0.05, right=0.97, top=0.963, bottom=0.055,
    )

    _p1_route        (fig.add_subplot(gs[0, 0]), run_out)
    _p2_drift        (fig.add_subplot(gs[0, 1]), run_out)
    _p3_nav_timeline (fig.add_subplot(gs[0, 2]), run_out)
    _p4_bim          (fig.add_subplot(gs[1, 0]), run_out)
    _p5_disturbance  (fig.add_subplot(gs[1, 1]), run_out)
    _p6_c2gates      (fig.add_subplot(gs[1, 2]), run_out)
    _p7_outcome      (fig.add_subplot(gs[2, :]), run_out)   # full width

    _status_strip(fig, run_out)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M")
    png_path = out / f"bcmp2_dashboard_seed{seed}_{ts}.png"

    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=BG_FIGURE)
    if verbose:
        print(f"[3/4] PNG  → {png_path}")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=BG_FIGURE)
    buf.seek(0)
    b64       = base64.b64encode(buf.read()).decode()
    html_path = out / f"bcmp2_dashboard_seed{seed}_{ts}.html"
    html_path.write_text(
        f'<!DOCTYPE html><html><head>'
        f'<title>MicroMind BCMP-2 Dashboard — Seed {seed}</title>'
        f'<style>body{{background:#0D1117;margin:0;padding:8px;}}'
        f'img{{max-width:100%;height:auto;}}</style></head>'
        f'<body><img src="data:image/png;base64,{b64}"/></body></html>'
    )
    if verbose:
        print(f"[4/4] HTML → {html_path}")

    if show:
        matplotlib.use("TkAgg")
        plt.show()
    plt.close(fig)
    return str(png_path), str(html_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MicroMind BCMP-2 Dual-Track Comparative Dashboard"
    )
    parser.add_argument("--seed",   type=int,   default=42,
                        help="Seed: 42 (nominal) / 101 (stressed) / 303")
    parser.add_argument("--km",     type=float, default=150.0,
                        help="Mission distance km (default 150)")
    parser.add_argument("--show",   action="store_true",
                        help="Display figure interactively")
    parser.add_argument("--outdir", default="dashboard",
                        help="Output directory (default: dashboard/)")
    args = parser.parse_args()

    print("=" * 65)
    print("MicroMind / NanoCorteX — BCMP-2 Dashboard  (SB-4)")
    print("=" * 65)
    png, html = build_bcmp2_dashboard(
        seed=args.seed, max_km=args.km,
        output_dir=args.outdir, show=args.show,
    )
    print()
    print("─" * 65)
    print(f"  PNG  → {png}")
    print(f"  HTML → {html}")
    print("─" * 65)
