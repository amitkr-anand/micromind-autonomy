"""
dashboard/bcmp2_replay.py
MicroMind / NanoCorteX — BCMP-2 Replay Driver

Four replay modes:
  executive  — 3-panel exec summary, annotated drift, 2-3 min briefing
  technical  — full 7-panel, all details, 8-10 min engineering review
  hifi       — frame-by-frame (1 frame per 5 km), slideshow ready
  overnight  — static summary + full report, batch/cron use

Usage:
    cd micromind-autonomy
    PYTHONPATH=. python3 dashboard/bcmp2_replay.py --mode executive --seed 42
    PYTHONPATH=. python3 dashboard/bcmp2_replay.py --mode technical --seed 101
    PYTHONPATH=. python3 dashboard/bcmp2_replay.py --mode hifi --seed 42 --km 50
    PYTHONPATH=. python3 dashboard/bcmp2_replay.py --mode overnight --seed 303
    PYTHONPATH=. python3 dashboard/bcmp2_replay.py --mode overnight \\
        --json dashboard/bcmp2_run_seed42_20260331_1200.json

JOURNAL
-------
Built: 31 March 2026, micromind-node01.  SB-4 Step 2.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scenarios.bcmp2.bcmp2_runner import run_bcmp2, BCMP2RunConfig
from scenarios.bcmp2.bcmp2_report import BCMPReport
from scenarios.bcmp2.bcmp2_drift_envelopes import VEHICLE_SPEED_MS
from dashboard.bcmp2_dashboard import (
    build_bcmp2_dashboard,
    BG_FIGURE, CLR_DIM, CLR_TEXT, CLR_AMBER, CLR_RED, CLR_VB,
    BG_CARD, CLR_GRID,
    _p1_route, _p2_drift, _p3_nav_timeline, _p4_bim,
    _p5_disturbance, _p6_c2gates, _p7_outcome, _status_strip,
    _denial_km,
)

MODES = ("executive", "technical", "hifi", "overnight")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, out: Path, stem: str, verbose: bool = True) -> tuple:
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{stem}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor=BG_FIGURE)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=BG_FIGURE)
    buf.seek(0)
    b64  = base64.b64encode(buf.read()).decode()
    html = out / f"{stem}.html"
    html.write_text(
        f'<!DOCTYPE html><html><head>'
        f'<title>MicroMind BCMP-2 — {stem}</title>'
        f'<style>body{{background:#0D1117;margin:0;padding:8px;}}'
        f'img{{max-width:100%;height:auto;}}</style></head>'
        f'<body><img src="data:image/png;base64,{b64}"/></body></html>'
    )
    if verbose:
        print(f"  PNG  -> {png}")
        print(f"  HTML -> {html}")
    plt.close(fig)
    return str(png), str(html)


def _load_or_run(seed, max_km, json_path, verbose) -> dict:
    if json_path and Path(json_path).exists():
        if verbose:
            print(f"  Loading: {json_path}")
        with open(json_path) as f:
            return json.load(f)
    if verbose:
        print(f"  Running seed={seed} km={max_km} ...")
    return run_bcmp2(BCMP2RunConfig(seed=seed, max_km=max_km, verbose=False))


def _write_json(run_out, out, stem):
    p = out / f"{stem}.json"
    with open(p, "w") as f:
        json.dump(run_out, f, indent=2, default=str)
    return str(p)


# ---------------------------------------------------------------------------
# Executive mode
# ---------------------------------------------------------------------------

def run_executive(seed, max_km, json_path, output_dir, verbose):
    """3-panel exec summary: drift + disturbance + outcome."""
    if verbose:
        print("\n-- EXECUTIVE MODE --")
    run_out = _load_or_run(seed, max_km, json_path, verbose)
    ts      = datetime.now().strftime("%Y%m%d_%H%M")
    stem    = f"bcmp2_executive_seed{seed}_{ts}"

    fig = plt.figure(figsize=(16, 10), facecolor=BG_FIGURE)
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.35, wspace=0.22,
                            left=0.06, right=0.97, top=0.96, bottom=0.06)

    ax_drift = fig.add_subplot(gs[0, 0])
    ax_sched = fig.add_subplot(gs[0, 1])
    ax_out   = fig.add_subplot(gs[1, :])

    _p2_drift(ax_drift, run_out)
    # Executive annotations on drift panel
    va     = run_out["vehicle_a"]
    sched  = run_out["disturbance_schedule"]
    dk     = _denial_km(sched)
    breach = va.get("first_corridor_violation_km")
    ax_drift.annotate(
        f"GNSS denied\n(km {dk:.0f})",
        xy=(dk, 5), xytext=(dk + 8, 40),
        fontsize=7, color=CLR_AMBER,
        arrowprops=dict(arrowstyle="->", color=CLR_AMBER, lw=1.0),
    )
    if breach:
        ax_drift.annotate(
            f"Corridor breach\n(km {breach:.0f})",
            xy=(breach, 500), xytext=(max(5, breach - 25), 300),
            fontsize=7, color=CLR_RED,
            arrowprops=dict(arrowstyle="->", color=CLR_RED, lw=1.0),
        )
    ax_drift.text(0.97, 0.93,
                  "MicroMind\nprevents this",
                  transform=ax_drift.transAxes,
                  fontsize=7, color=CLR_VB, ha="right", va="top",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_CARD,
                            edgecolor=CLR_VB, alpha=0.9))

    _p5_disturbance(ax_sched, run_out)
    _p7_outcome(ax_out, run_out)
    _status_strip(fig, run_out)
    fig.text(0.50, 0.975,
             "EXECUTIVE SUMMARY — GNSS-Denied Mission Comparison",
             fontsize=10, fontweight="bold", color=CLR_TEXT,
             ha="center", va="top", transform=fig.transFigure)

    return _save(fig, Path(output_dir), stem, verbose)


# ---------------------------------------------------------------------------
# Technical mode — delegates to full dashboard
# ---------------------------------------------------------------------------

def run_technical(seed, max_km, json_path, output_dir, verbose):
    """Full 7-panel dashboard — delegates to build_bcmp2_dashboard."""
    if verbose:
        print("\n-- TECHNICAL MODE --")
    return build_bcmp2_dashboard(
        seed=seed, max_km=max_km,
        output_dir=output_dir,
        show=False, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# High-fidelity mode — frame per 5 km
# ---------------------------------------------------------------------------

def run_hifi(seed, max_km, json_path, output_dir, verbose):
    """One dashboard frame per 5 km — produces slideshow-ready PNGs."""
    if verbose:
        print("\n-- HIGH-FIDELITY MODE --")
    run_out  = _load_or_run(seed, max_km, json_path, verbose)
    states   = run_out["vehicle_a"].get("states", [])
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M")
    checkpts = list(range(5, int(max_km) + 1, 5))

    if verbose:
        print(f"  {len(checkpts)} frames ...")

    for fi, ckm in enumerate(checkpts):
        sub     = dict(run_out)
        sub["vehicle_a"] = dict(run_out["vehicle_a"])
        sub["vehicle_a"]["states"] = [s for s in states
                                       if s["mission_km"] <= ckm]
        sub["max_km"] = float(ckm)

        fig = plt.figure(figsize=(18, 12), facecolor=BG_FIGURE)
        gs  = gridspec.GridSpec(3, 3, figure=fig,
                                hspace=0.40, wspace=0.28,
                                left=0.05, right=0.97,
                                top=0.963, bottom=0.055)
        _p1_route       (fig.add_subplot(gs[0, 0]), sub)
        _p2_drift       (fig.add_subplot(gs[0, 1]), sub)
        _p3_nav_timeline(fig.add_subplot(gs[0, 2]), sub)
        _p4_bim         (fig.add_subplot(gs[1, 0]), sub)
        _p5_disturbance (fig.add_subplot(gs[1, 1]), sub)
        _p6_c2gates     (fig.add_subplot(gs[1, 2]), sub)
        _p7_outcome     (fig.add_subplot(gs[2, :]), run_out)
        _status_strip(fig, run_out)
        fig.text(0.5, 0.001,
                 f"Frame {fi+1}/{len(checkpts)}  km {ckm:.0f}/{max_km:.0f}",
                 fontsize=6, color=CLR_DIM, ha="center",
                 transform=fig.transFigure)

        fp = out_dir / f"bcmp2_hifi_seed{seed}_{ts}_f{fi+1:03d}.png"
        fig.savefig(fp, dpi=100, bbox_inches="tight", facecolor=BG_FIGURE)
        plt.close(fig)
        if verbose and (fi % 5 == 0 or fi == len(checkpts) - 1):
            print(f"  Frame {fi+1:3d}/{len(checkpts)}  km={ckm:.0f}  {fp.name}")

    # Final summary frame
    stem = f"bcmp2_hifi_seed{seed}_{ts}_final"
    fig  = plt.figure(figsize=(18, 12), facecolor=BG_FIGURE)
    gs   = gridspec.GridSpec(3, 3, figure=fig,
                             hspace=0.40, wspace=0.28,
                             left=0.05, right=0.97,
                             top=0.963, bottom=0.055)
    _p1_route       (fig.add_subplot(gs[0, 0]), run_out)
    _p2_drift       (fig.add_subplot(gs[0, 1]), run_out)
    _p3_nav_timeline(fig.add_subplot(gs[0, 2]), run_out)
    _p4_bim         (fig.add_subplot(gs[1, 0]), run_out)
    _p5_disturbance (fig.add_subplot(gs[1, 1]), run_out)
    _p6_c2gates     (fig.add_subplot(gs[1, 2]), run_out)
    _p7_outcome     (fig.add_subplot(gs[2, :]), run_out)
    _status_strip(fig, run_out)
    return _save(fig, out_dir, stem, verbose)


# ---------------------------------------------------------------------------
# Overnight mode — static summary + full report
# ---------------------------------------------------------------------------

def run_overnight(seed, max_km, json_path, output_dir, verbose):
    """Batch: run + save JSON + dashboard PNG + full report HTML."""
    if verbose:
        print("\n-- OVERNIGHT MODE --")
    run_out = _load_or_run(seed, max_km, json_path, verbose)
    ts      = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(output_dir)

    jp = _write_json(run_out, out_dir, f"bcmp2_run_seed{seed}_{ts}")
    if verbose:
        print(f"  Run JSON -> {jp}")

    png, _ = build_bcmp2_dashboard(
        seed=seed, max_km=max_km,
        output_dir=output_dir,
        show=False, verbose=verbose,
    )

    report      = BCMPReport(run_out, run_date=datetime.now().strftime("%Y-%m-%d"))
    report_html = out_dir / f"bcmp2_report_seed{seed}_{ts}.html"
    report.write_html(str(report_html))
    if verbose:
        print(f"  Report  -> {report_html}")

    return png, str(report_html)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MicroMind BCMP-2 Replay Driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  executive  3-panel annotated summary — TASL BD briefing (2-3 min)
  technical  Full 7-panel dashboard  — engineering review (8-10 min)
  hifi       Frame per 5 km — slideshow (30 frames for 150 km)
  overnight  Static summary + full report — batch/cron

Examples:
  python3 dashboard/bcmp2_replay.py --mode executive --seed 101
  python3 dashboard/bcmp2_replay.py --mode hifi --seed 42 --km 50
  python3 dashboard/bcmp2_replay.py --mode overnight --seed 303
""",
    )
    parser.add_argument("--mode",   choices=MODES, default="executive")
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--km",     type=float, default=150.0)
    parser.add_argument("--json",   default=None,
                        help="Existing run JSON (skips re-running)")
    parser.add_argument("--outdir", default="dashboard")
    parser.add_argument("--quiet",  action="store_true")
    args    = parser.parse_args()
    verbose = not args.quiet

    print("=" * 65)
    print(f"MicroMind BCMP-2 Replay  |  mode={args.mode}  seed={args.seed}")
    print("=" * 65)

    fns = {"executive": run_executive, "technical": run_technical,
           "hifi": run_hifi, "overnight": run_overnight}
    png, html = fns[args.mode](
        seed=args.seed, max_km=args.km,
        json_path=args.json,
        output_dir=args.outdir,
        verbose=verbose,
    )
    print()
    print("─" * 65)
    print(f"  Primary PNG  -> {png}")
    print(f"  Primary HTML -> {html}")
    print("─" * 65)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    main()
