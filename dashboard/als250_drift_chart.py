"""
dashboard/als250_drift_chart.py — ALS-250 IMU Drift Chart (Sprint S8-D)
Three-curve position drift comparison: STIM300 vs ADIS16505-3 vs BASELINE
for the TASL engineering presentation.

Usage:
    PYTHONPATH=. python dashboard/als250_drift_chart.py
    PYTHONPATH=. python dashboard/als250_drift_chart.py --seed 7 --duration 600 --show
    PYTHONPATH=. python dashboard/als250_drift_chart.py --from-npy sim/  # load pre-saved arrays

Outputs (in dashboard/):
    als250_drift_chart_<YYYYMMDD_HHMM>.png  — 150 dpi, TASL-ready static PNG
    als250_drift_chart_<YYYYMMDD_HHMM>.html — self-contained HTML (PNG embedded)

The chart shows:
  - X axis: distance along corridor [km]
  - Y axis: CEP position drift [m]
  - Three curves: STIM300 (red), ADIS16505-3 (orange), BASELINE (green)
  - NAV-01 limit line: 100 m per 5 km dashed red
  - Annotation: final drift + spec compliance per model
  - Inset table: ARW / bias instability / VRE / SF per sensor
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Core imports
from core.ins.imu_model import get_imu_model, IMU_REGISTRY

# S8-C simulation (run inline if no pre-saved data)
from sim.als250_nav_sim import run_als250_sim, SEGMENT_M, DRIFT_LIMIT_M, CORRIDOR_DURATION_S

# ---------------------------------------------------------------------------
# Visual identity (consistent with S7 bcmp1_dashboard.py)
# ---------------------------------------------------------------------------

COLOURS = {
    "bg":        "#0d1117",
    "panel":     "#161b22",
    "border":    "#30363d",
    "text":      "#e6edf3",
    "subtext":   "#8b949e",
    "green":     "#3fb950",
    "amber":     "#d29922",
    "red":       "#f85149",
    "blue":      "#58a6ff",
    "purple":    "#bc8cff",
    "STIM300":   "#f85149",
    "ADIS16505_3": "#d29922",
    "BASELINE":  "#3fb950",
    "limit":     "#8b949e",
}

MODEL_LABELS = {
    "STIM300":     "Safran STIM300",
    "ADIS16505_3": "ADI ADIS16505-3",
    "BASELINE":    "BASELINE (ideal)",
}

CHART_MODELS = ["STIM300", "ADIS16505_3", "BASELINE"]

# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

def _load_or_run(
    models: list[str],
    duration_s: float,
    seed: int,
    npy_dir: Optional[pathlib.Path],
    verbose: bool,
) -> dict[str, dict]:
    """
    For each model: try loading pre-saved .npy files, otherwise run simulation.
    Returns {model_name: run_als250_sim result}.
    """
    results = {}
    for name in models:
        loaded = False
        if npy_dir:
            pos_path   = npy_dir / f"als250_nav_{name}_{seed}_position.npy"
            drift_path = npy_dir / f"als250_nav_{name}_{seed}_drift.npy"
            meta_path  = npy_dir / f"als250_nav_{name}_{seed}_meta.json"
            if pos_path.exists() and drift_path.exists():
                pos   = np.load(pos_path)
                drift = np.load(drift_path)
                kpi   = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                results[name] = {
                    "position":      pos,
                    "true_position": pos,   # not needed for chart
                    "drift_per_seg": drift,
                    "imu_name":      name,
                    "kpi":           kpi,
                }
                if verbose:
                    print(f"  [chart] Loaded pre-saved data: {pos_path.name}")
                loaded = True

        if not loaded:
            if verbose:
                print(f"  [chart] Running simulation for {name} ...")
            results[name] = run_als250_sim(
                imu_name=name,
                duration_s=duration_s,
                seed=seed,
                verbose=verbose,
            )
    return results


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------

def build_drift_chart(
    results: dict[str, dict],
    duration_s: float,
    seed: int,
) -> plt.Figure:
    """
    Build the three-curve drift chart.
    """
    from core.constants import GRAVITY  # noqa — just ensure env is consistent

    fig = plt.figure(figsize=(14, 8), facecolor=COLOURS["bg"])
    gs  = GridSpec(1, 1, figure=fig, left=0.08, right=0.72, top=0.88, bottom=0.12)
    ax  = fig.add_subplot(gs[0, 0])

    ax.set_facecolor(COLOURS["panel"])
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOURS["border"])

    cruise_speed = 55.0   # m/s

    legend_handles = []

    for name in CHART_MODELS:
        if name not in results:
            continue
        r      = results[name]
        drift  = r["drift_per_seg"]
        n_segs = len(drift)

        # X axis: distance in km at each segment boundary
        seg_dist_km = np.arange(1, n_segs + 1) * (SEGMENT_M / 1_000.0)

        colour = COLOURS[name]
        lw     = 2.0 if name != "BASELINE" else 1.5
        ls     = "-" if name != "ADIS16505_3" else "--"

        ax.plot(seg_dist_km, drift,
                color=colour, linewidth=lw, linestyle=ls,
                label=MODEL_LABELS[name])

        # Mark final drift value
        final = float(drift[-1]) if len(drift) > 0 else 0.0
        nav01 = final < DRIFT_LIMIT_M
        mark  = "[PASS]" if nav01 else "[FAIL]"
        ax.annotate(
            f"{final:.0f} m {mark}",
            xy=(seg_dist_km[-1], final),
            xytext=(seg_dist_km[-1] + 2, final + (8 if name != "BASELINE" else -12)),
            color=colour, fontsize=7.5,
            arrowprops=dict(arrowstyle="-", color=colour, lw=0.8),
        )

        handle = mpatches.Patch(color=colour, label=f"{MODEL_LABELS[name]} — final {final:.0f} m")
        legend_handles.append(handle)

    # NAV-01 limit line
    x_max = (duration_s * cruise_speed) / 1_000.0
    ax.axhline(DRIFT_LIMIT_M, color=COLOURS["limit"], linewidth=1.2,
               linestyle=":", alpha=0.8, label=f"NAV-01 limit ({DRIFT_LIMIT_M:.0f} m / 5 km)")
    limit_handle = mpatches.Patch(
        color=COLOURS["limit"],
        label=f"NAV-01 limit ({DRIFT_LIMIT_M:.0f} m per 5 km)"
    )
    legend_handles.append(limit_handle)

    # Axis labels
    ax.set_xlabel("Distance along corridor [km]",
                  color=COLOURS["text"], fontsize=10)
    ax.set_ylabel("CEP position drift [m]",
                  color=COLOURS["text"], fontsize=10)
    ax.tick_params(colors=COLOURS["subtext"], labelsize=8)
    ax.xaxis.label.set_color(COLOURS["text"])
    ax.yaxis.label.set_color(COLOURS["text"])
    ax.grid(True, color=COLOURS["border"], linewidth=0.5, alpha=0.6)

    # Title
    fig.suptitle(
        f"ALS-250 INS Drift Comparison — {x_max:.0f} km GNSS-Denied Corridor",
        color=COLOURS["text"], fontsize=12, fontweight="bold", y=0.96,
    )
    ax.set_title(
        f"200 Hz IMU | Seed {seed} | NAV-01 limit {DRIFT_LIMIT_M:.0f} m / 5 km",
        color=COLOURS["subtext"], fontsize=8.5,
    )

    # Legend (inside chart area)
    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=8,
        facecolor=COLOURS["panel"],
        edgecolor=COLOURS["border"],
        labelcolor=COLOURS["text"],
        framealpha=0.9,
    )

    # ---- Sensor spec table (right panel) ----
    table_ax = fig.add_axes([0.74, 0.15, 0.25, 0.68])
    table_ax.set_facecolor(COLOURS["panel"])
    table_ax.axis("off")
    for spine in table_ax.spines.values():
        spine.set_visible(False)

    table_ax.text(0.5, 1.00, "Sensor Specifications",
                  color=COLOURS["text"], fontsize=9, fontweight="bold",
                  ha="center", va="top", transform=table_ax.transAxes)

    col_headers = ["Sensor", "ARW\n[deg/√hr]", "BI\n[deg/hr]", "VRE\n[mg]", "SF\n[ppm]"]
    col_x = [0.0, 0.28, 0.48, 0.68, 0.88]
    row_y_start = 0.88
    row_dy      = 0.14

    # Header row
    for hdr, cx in zip(col_headers, col_x):
        table_ax.text(cx, row_y_start, hdr,
                      color=COLOURS["subtext"], fontsize=7, fontweight="bold",
                      ha="left", va="top", transform=table_ax.transAxes)

    # Data rows
    for i, name in enumerate(CHART_MODELS):
        try:
            m = get_imu_model(name)
        except Exception:
            continue
        vre_mg = m.vre_accel_ms2 / 9.80665 * 1000.0
        row_data = [
            MODEL_LABELS[name].replace(" ", "\n"),
            f"{m.arw_deg_rthz:.3f}",
            f"{m.bias_instability_deg_hr:.2f}",
            f"{vre_mg:.3f}" if vre_mg > 0 else "0",
            f"{m.sf_ppm:.0f}",
        ]
        y = row_y_start - (i + 1) * row_dy
        colour = COLOURS[name]
        for val, cx in zip(row_data, col_x):
            table_ax.text(cx, y, val,
                          color=colour, fontsize=6.5, ha="left", va="top",
                          transform=table_ax.transAxes)

    # V7 spec note
    table_ax.text(0.0, row_y_start - 4 * row_dy - 0.08,
                  "V7 spec floor (strict):\nARW <= 0.1 deg/sqrt(hr)\n"
                  "Note: STIM300 ARW 0.15 >\nV7 floor; update to 0.2\nbefore TASL",
                  color=COLOURS["amber"], fontsize=6.2,
                  ha="left", va="top", transform=table_ax.transAxes)

    return fig


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_chart(fig: plt.Figure, out_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Save PNG and self-contained HTML. Returns (png_path, html_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M")
    png_path = out_dir / f"als250_drift_chart_{ts}.png"
    html_path = out_dir / f"als250_drift_chart_{ts}.html"

    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  [chart] PNG  → {png_path}")

    # Embed PNG in self-contained HTML
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    html = (
        "<!DOCTYPE html><html><head>"
        "<meta charset='utf-8'>"
        "<title>ALS-250 IMU Drift Chart — MicroMind S8-D</title>"
        "<style>body{margin:0;background:#0d1117;display:flex;"
        "justify-content:center;align-items:center;min-height:100vh;}"
        "img{max-width:100%;height:auto;}</style>"
        "</head><body>"
        f"<img src='data:image/png;base64,{b64}' alt='ALS-250 Drift Chart'>"
        "</body></html>"
    )
    html_path.write_text(html)
    print(f"  [chart] HTML → {html_path}")
    return png_path, html_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="ALS-250 IMU drift chart — three-curve TASL presentation (S8-D)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--duration", type=float, default=None,
                   help="Simulation duration in seconds (default: full 250 km)")
    p.add_argument("--from-npy", type=str,   default=None,
                   help="Load pre-saved .npy files from this directory instead of running sim")
    p.add_argument("--out",      type=str,   default="dashboard",
                   help="Output directory for PNG/HTML")
    p.add_argument("--show",     action="store_true", help="Display chart interactively")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    duration_s = args.duration if args.duration else CORRIDOR_DURATION_S
    npy_dir    = pathlib.Path(args.from_npy) if args.from_npy else None
    out_dir    = pathlib.Path(args.out)

    print(f"\n{'='*60}")
    print(f"  ALS-250 DRIFT CHART — Sprint S8-D")
    print(f"{'='*60}")

    results = _load_or_run(
        models    = CHART_MODELS,
        duration_s= duration_s,
        seed      = args.seed,
        npy_dir   = npy_dir,
        verbose   = True,
    )

    fig = build_drift_chart(results, duration_s=duration_s, seed=args.seed)
    png_path, html_path = save_chart(fig, out_dir)

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()
    plt.close(fig)

    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
