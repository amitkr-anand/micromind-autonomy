"""
BCMP-2 Report Generator.

Produces JSON and HTML reports from a dual-track run output dict.

Report structure (HTML)
-----------------------
1. Business comparison block  ← ALWAYS FIRST. No technical metric appears above it.
2. Route map                  ← static dual-path Plotly figure (inline SVG)
3. Drift chart per phase      ← Vehicle A and B on shared axis
4. Navigation source timeline ← Vehicle B mode transitions
5. Event log                  ← timestamped, including fault injection events
6. Technical metrics tables   ← all AT criteria evidence

The business comparison block format follows §8.3 of the BCMP-2
architecture document v1.1 exactly.

Usage
-----
    from scenarios.bcmp2.bcmp2_report import BCMPReport

    report = BCMPReport(run_output)        # run_output from bcmp2_runner.run_bcmp2()
    report.write_json("bcmp2_run42.json")
    report.write_html("bcmp2_run42.html")

JOURNAL
-------
Built: 30 March 2026, micromind-node01.  SB-3 Step 1.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Colour palette (colourblind-safe, high contrast)
# ---------------------------------------------------------------------------

_RED    = "#C0392B"
_GREEN  = "#27AE60"
_AMBER  = "#E67E22"
_BLUE   = "#2980B9"
_GREY   = "#7F8C8D"
_DARK   = "#1A1A2E"
_LIGHT  = "#ECF0F1"
_WHITE  = "#FFFFFF"


def _result_colour(result: str) -> str:
    if "SUCCEEDED" in result or "PASS" in result:
        return _GREEN
    if "FAILED" in result or "FAIL" in result:
        return _RED
    return _AMBER


# ---------------------------------------------------------------------------
# Report class
# ---------------------------------------------------------------------------

class BCMPReport:
    """
    Generates JSON and HTML reports from a bcmp2_runner output dict.

    Parameters
    ----------
    run_output : dict returned by run_bcmp2()
    run_date   : optional ISO date string; defaults to today
    """

    def __init__(self, run_output: dict, run_date: Optional[str] = None):
        self._data     = run_output
        self._run_date = run_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._seed     = run_output.get("seed", "?")
        self._comp     = run_output.get("comparison", {})
        self._va       = run_output.get("vehicle_a", {})
        self._vb       = run_output.get("vehicle_b", {})
        self._sched    = run_output.get("disturbance_schedule", {})

    # ── JSON output ────────────────────────────────────────────────────────

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self._data, indent=indent, default=str)

    def write_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())

    # ── HTML output ────────────────────────────────────────────────────────

    def to_html(self) -> str:
        sections = [
            self._html_head(),
            self._html_business_block(),   # §1 — always first
            self._html_drift_chart(),       # §2 — Vehicle A vs B drift
            self._html_nav_timeline(),      # §3 — Vehicle B nav source
            self._html_event_log(),         # §4 — event log
            self._html_metrics_tables(),    # §5 — technical evidence
            self._html_foot(),
        ]
        return "\n".join(sections)

    def write_html(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_html())

    # ── §0 — HTML head ─────────────────────────────────────────────────────

    def _html_head(self) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BCMP-2 Run Report — Seed {self._seed} — {self._run_date}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: {_DARK};
          color: {_LIGHT}; padding: 24px; }}
  h1   {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 6px; }}
  h2   {{ font-size: 1.15rem; font-weight: 600; margin: 28px 0 10px; color: #BDC3C7; }}
  h3   {{ font-size: 0.95rem; font-weight: 600; margin: 14px 0 6px; color: #95A5A6; }}
  .card {{ background: #16213E; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
  .outcome-row {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }}
  .outcome-box {{ flex: 1; min-width: 260px; border-radius: 8px; padding: 18px 20px; }}
  .outcome-label {{ font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
                    letter-spacing: 1px; opacity: 0.8; margin-bottom: 8px; }}
  .outcome-result {{ font-size: 1.4rem; font-weight: 800; margin-bottom: 12px; }}
  .chain-item {{ font-size: 0.85rem; line-height: 1.6; padding: 2px 0; opacity: 0.9; }}
  .chain-item::before {{ content: "→ "; opacity: 0.5; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
                  gap: 12px; margin: 12px 0; }}
  .metric-cell {{ background: #0F3460; border-radius: 6px; padding: 12px; text-align: center; }}
  .metric-val  {{ font-size: 1.2rem; font-weight: 700; }}
  .metric-lbl  {{ font-size: 0.72rem; opacity: 0.7; margin-top: 4px; }}
  .pass {{ color: {_GREEN}; }} .fail {{ color: {_RED}; }} .warn {{ color: {_AMBER}; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ background: #0F3460; padding: 8px 10px; text-align: left; font-weight: 600; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #1E3A5F; }}
  tr:hover td {{ background: #1A2F50; }}
  .ev-tag {{ display: inline-block; border-radius: 4px; padding: 1px 6px; font-size: 0.72rem;
             font-weight: 700; }}
  .ev-gnss {{ background: #7D3C98; }} .ev-vio {{ background: {_BLUE}; }}
  .ev-trn  {{ background: #117A65; }} .ev-eo  {{ background: #A93226; }}
  .ev-info {{ background: #2C3E50; }}
  .divider {{ border: none; border-top: 1px solid #2C3E50; margin: 18px 0; }}
  svg {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<h1>BCMP-2 Mission Report</h1>
<p style="opacity:0.6; font-size:0.82rem; margin-bottom:20px;">
  Seed {self._seed} &nbsp;·&nbsp; {self._run_date} &nbsp;·&nbsp;
  {self._data.get('hardware_source','simulated')} &nbsp;·&nbsp;
  IMU: {self._data.get('imu_model','STIM300')} &nbsp;·&nbsp;
  Duration: {self._data.get('run_duration_s',0):.1f}s
</p>"""

    # ── §1 — Business comparison block (architecture doc §8.3) ────────────

    def _html_business_block(self) -> str:
        comp     = self._comp
        va_result = comp.get("vehicle_a_mission_result", "UNKNOWN")
        vb_result = comp.get("vehicle_b_mission_result", "UNKNOWN")
        va_colour = _result_colour(va_result)
        vb_colour = _result_colour(vb_result)
        denial_km = comp.get("gnss_denial_km", 30)

        a_chain = comp.get("vehicle_a_causal_chain", [])
        b_chain = comp.get("vehicle_b_causal_chain", [])

        def chain_html(items):
            return "".join(f'<div class="chain-item">{i}</div>' for i in items)

        # Key metric boxes
        a_drift = self._va.get("drift_at_km120_m")
        b_drift = self._vb.get("max_5km_drift_m")
        a_breach = comp.get("vehicle_a_first_corridor_violation_km")
        b_trn    = self._vb.get("trn_corrections")

        a_drift_s  = f"{a_drift:.0f} m" if a_drift  else "N/A"
        b_drift_s  = f"{b_drift:.1f} m" if b_drift  else "N/A"
        a_breach_s = f"km {a_breach:.0f}" if a_breach else "None"
        b_trn_s    = str(b_trn) if b_trn is not None else "N/A"

        return f"""
<div class="card">
  <h2 style="margin-top:0; color:{_LIGHT}; font-size:1.05rem;">
    Mission Outcome — GNSS Denied at km {denial_km:.0f}
  </h2>
  <div class="outcome-row">

    <div class="outcome-box" style="background:{va_colour}22; border:2px solid {va_colour};">
      <div class="outcome-label">Without MicroMind (Baseline)</div>
      <div class="outcome-result" style="color:{va_colour};">&#9888; {va_result}</div>
      {chain_html(a_chain)}
      <hr class="divider">
      <div class="metric-grid">
        <div class="metric-cell">
          <div class="metric-val" style="color:{va_colour};">{a_drift_s}</div>
          <div class="metric-lbl">Lateral drift @ km 120</div>
        </div>
        <div class="metric-cell">
          <div class="metric-val" style="color:{va_colour};">{a_breach_s}</div>
          <div class="metric-lbl">First corridor breach</div>
        </div>
      </div>
    </div>

    <div class="outcome-box" style="background:{vb_colour}22; border:2px solid {vb_colour};">
      <div class="outcome-label">With MicroMind</div>
      <div class="outcome-result" style="color:{vb_colour};">&#10003; {vb_result}</div>
      {chain_html(b_chain)}
      <hr class="divider">
      <div class="metric-grid">
        <div class="metric-cell">
          <div class="metric-val" style="color:{vb_colour};">{b_drift_s}</div>
          <div class="metric-lbl">Max 5km drift bounded</div>
        </div>
        <div class="metric-cell">
          <div class="metric-val" style="color:{vb_colour};">{b_trn_s}</div>
          <div class="metric-lbl">TRN corrections applied</div>
        </div>
      </div>
    </div>

  </div>
</div>"""

    # ── §2 — Drift chart (SVG inline, no external dependencies) ───────────

    def _html_drift_chart(self) -> str:
        # Build a simple SVG bar chart of drift at each phase boundary
        va = self._va
        gates = [
            ("km 60",  va.get("drift_at_km60_m"),  5,   80),
            ("km 100", va.get("drift_at_km100_m"), 12,  350),
            ("km 120", va.get("drift_at_km120_m"), 15,  650),
        ]

        w, h = 560, 180
        pad_l, pad_r, pad_t, pad_b = 60, 20, 20, 40
        chart_w = w - pad_l - pad_r
        chart_h = h - pad_t - pad_b
        bar_w   = chart_w // (len(gates) * 3 + 1)

        max_val = max((g[1] or 0) for g in gates) * 1.25 or 1000

        def y_of(val):
            if val is None:
                return pad_t + chart_h
            return pad_t + chart_h - int((val / max_val) * chart_h)

        bars = ""
        labels = ""
        for i, (lbl, obs, fl, ce) in enumerate(gates):
            x_base = pad_l + i * (bar_w * 3 + bar_w // 2) + bar_w // 2
            # Floor line
            y_fl = y_of(fl)
            y_ce = y_of(ce)
            bars += (f'<rect x="{x_base}" y="{y_fl}" width="{bar_w*2}" '
                     f'height="{y_of(0)-y_fl}" fill="#2ecc7122" stroke="#2ecc71" '
                     f'stroke-width="1" stroke-dasharray="3,2"/>')
            # Ceiling line
            bars += (f'<line x1="{x_base}" y1="{y_ce}" x2="{x_base+bar_w*2}" y2="{y_ce}" '
                     f'stroke="{_RED}" stroke-width="1" stroke-dasharray="4,2"/>')
            # Observed bar
            if obs is not None:
                colour = _GREEN if fl <= obs <= ce else _RED
                y_obs = y_of(obs)
                bar_h = y_of(0) - y_obs
                bars += (f'<rect x="{x_base+bar_w//4}" y="{y_obs}" '
                         f'width="{bar_w}" height="{bar_h}" '
                         f'fill="{colour}" rx="2"/>')
                bars += (f'<text x="{x_base+bar_w//4+bar_w//2}" y="{y_obs-4}" '
                         f'fill="{colour}" font-size="10" text-anchor="middle">'
                         f'{obs:.0f}m</text>')
            labels += (f'<text x="{x_base+bar_w}" y="{h-8}" '
                       f'fill="{_GREY}" font-size="11" text-anchor="middle">{lbl}</text>')

        # Y-axis labels
        y_axis = ""
        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            val = int(max_val * tick)
            y   = y_of(val)
            y_axis += (f'<line x1="{pad_l-4}" y1="{y}" x2="{w-pad_r}" y2="{y}" '
                       f'stroke="#2C3E50" stroke-width="1"/>')
            y_axis += (f'<text x="{pad_l-6}" y="{y+4}" fill="{_GREY}" '
                       f'font-size="9" text-anchor="end">{val}</text>')

        svg = (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
               f'style="background:#16213E;border-radius:6px;">'
               f'{y_axis}{bars}{labels}'
               f'<text x="{pad_l//2}" y="{pad_t+chart_h//2}" fill="{_GREY}" '
               f'font-size="9" text-anchor="middle" '
               f'transform="rotate(-90,{pad_l//2},{pad_t+chart_h//2})">Lateral drift (m)</text>'
               f'</svg>')

        legend = (f'<p style="font-size:0.75rem;opacity:0.6;margin-top:6px;">'
                  f'<span style="color:{_GREEN}">■</span> Observed drift (within envelope) &nbsp; '
                  f'<span style="color:{_RED}">■</span> Observed drift (out of envelope) &nbsp; '
                  f'<span style="color:{_RED}">— —</span> Ceiling &nbsp; '
                  f'<span style="color:#2ecc71">▒</span> Valid envelope band'
                  f'</p>')

        return f'<div class="card"><h2>Vehicle A Drift vs C-2 Envelopes</h2>{svg}{legend}</div>'

    # ── §3 — Navigation source timeline (Vehicle B) ────────────────────────

    def _html_nav_timeline(self) -> str:
        vb = self._vb
        rows = []
        for k, v in vb.items():
            if v is not None:
                label = k.replace("_", " ").title()
                cls   = "pass" if v is True else ("fail" if v is False else "")
                val_s = "✓ PASS" if v is True else ("✗ FAIL" if v is False else str(v))
                rows.append(f'<tr><td>{label}</td><td class="{cls}">{val_s}</td></tr>')

        table = (f'<table><thead><tr><th>KPI</th><th>Value</th></tr></thead>'
                 f'<tbody>{"".join(rows)}</tbody></table>')

        return f'<div class="card"><h2>Vehicle B Navigation KPIs (MicroMind)</h2>{table}</div>'

    # ── §4 — Event log ─────────────────────────────────────────────────────

    def _html_event_log(self) -> str:
        sched = self._sched
        events = []

        denial = sched.get("gnss_denial", {})
        if denial.get("start_s"):
            km = denial["start_s"] * 55 / 1000
            events.append(("gnss", f"km {km:.0f}", "GNSS Denial begins",
                            f"Duration: {denial.get('duration_s','?')}s"))

        for i, outage in enumerate(sched.get("vio_outages", [])):
            km = outage.get("start_s", 0) * 55 / 1000
            events.append(("vio", f"km {km:.0f}", f"VIO Outage #{i+1}",
                            f"Duration: {outage.get('duration_s','?')}s"))

        if sched.get("radalt_loss_s"):
            km = sched["radalt_loss_s"] * 55 / 1000
            events.append(("trn", f"km {km:.0f}", "RADALT loss", "NavSource: TRN suppressed"))

        if sched.get("eo_freeze_s"):
            km = sched["eo_freeze_s"] * 55 / 1000
            events.append(("eo", f"km {km:.0f}", "EO feed freeze", "DMRL stale-frame detection"))

        if not events:
            events.append(("info", "—", "No disturbance events in this run", ""))

        rows = ""
        for tag, km, desc, detail in events:
            rows += (f'<tr><td><span class="ev-tag ev-{tag}">{tag.upper()}</span></td>'
                     f'<td>{km}</td><td>{desc}</td><td style="opacity:0.7">{detail}</td></tr>')

        table = (f'<table><thead><tr><th>Type</th><th>Position</th>'
                 f'<th>Event</th><th>Detail</th></tr></thead>'
                 f'<tbody>{rows}</tbody></table>')

        return f'<div class="card"><h2>Disturbance Event Log</h2>{table}</div>'

    # ── §5 — Technical metrics tables ─────────────────────────────────────

    def _html_metrics_tables(self) -> str:
        va    = self._va
        comp  = self._comp

        # C-2 gate table
        c2_rows = ""
        for km, gate in va.get("c2_gates", {}).items():
            obs = gate.get("observed_m")
            obs_s = f"{obs:.0f} m" if obs is not None else "Not reached"
            passed = gate.get("passed", False)
            cls  = "pass" if passed else ("warn" if obs is None else "fail")
            status = "PASS" if passed else ("N/A" if obs is None else "FAIL")
            c2_rows += (f'<tr><td>km {km}</td><td>{obs_s}</td>'
                        f'<td class="{cls}">{status}</td></tr>')

        c2_table = (f'<h3>C-2 Drift Gate Results (Vehicle A)</h3>'
                    f'<table><thead><tr><th>Phase boundary</th>'
                    f'<th>Observed drift</th><th>Gate</th></tr></thead>'
                    f'<tbody>{c2_rows}</tbody></table>')

        # Run metadata table
        meta_rows = ""
        for k, v in [
            ("Seed", self._seed),
            ("Max km", self._data.get("max_km")),
            ("IMU model", self._data.get("imu_model")),
            ("Hardware source", self._data.get("hardware_source")),
            ("Run duration (s)", self._data.get("run_duration_s")),
            ("Vehicle A time (s)", self._data.get("vehicle_a_time_s")),
            ("Vehicle B time (s)", self._data.get("vehicle_b_time_s")),
            ("BCMP-2 version", self._data.get("bcmp2_version")),
        ]:
            meta_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

        meta_table = (f'<h3>Run Metadata</h3>'
                      f'<table><thead><tr><th>Field</th><th>Value</th></tr></thead>'
                      f'<tbody>{meta_rows}</tbody></table>')

        return (f'<div class="card">'
                f'<h2>Technical Evidence</h2>'
                f'{c2_table}<br>{meta_table}'
                f'</div>')

    # ── §6 — HTML foot ─────────────────────────────────────────────────────

    def _html_foot(self) -> str:
        return (f'<p style="font-size:0.72rem;opacity:0.4;margin-top:20px;text-align:center;">'
                f'MicroMind / NanoCorteX — BCMP-2 Report — '
                f'Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")} UTC — '
                f'Seed {self._seed}'
                f'</p></body></html>')


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def generate_report(
    run_output: dict,
    output_dir: str = ".",
    prefix:     str = "bcmp2_report",
    run_date:   Optional[str] = None,
) -> tuple[str, str]:
    """
    Generate both JSON and HTML reports from a run_bcmp2() output dict.

    Returns
    -------
    (json_path, html_path)
    """
    seed = run_output.get("seed", "unknown")
    date = run_date or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"{prefix}_seed{seed}_{date}"

    json_path = os.path.join(output_dir, f"{stem}.json")
    html_path = os.path.join(output_dir, f"{stem}.html")

    report = BCMPReport(run_output, run_date=run_date)
    report.write_json(json_path)
    report.write_html(html_path)

    return json_path, html_path


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, tempfile
    sys.path.insert(0, ".")

    from scenarios.bcmp2.bcmp2_runner import run_bcmp2, BCMP2RunConfig

    print("BCMPReport self-verification")
    print("=" * 45)

    config = BCMP2RunConfig(seed=42, max_km=5.0, verbose=False)
    out    = run_bcmp2(config)

    report = BCMPReport(out, run_date="2026-03-30")

    # JSON output
    j = report.to_json()
    import json
    parsed = json.loads(j)
    assert "comparison" in parsed
    assert "vehicle_a" in parsed
    print("  JSON output valid:         PASS")

    # HTML output — business block is first substantive section
    h = report.to_html()
    assert "Mission Outcome" in h, "business block missing"
    assert "Without MicroMind" in h
    assert "With MicroMind" in h
    # Business block must appear before technical tables
    business_pos = h.index("Mission Outcome")
    technical_pos = h.index("Technical Evidence")
    assert business_pos < technical_pos, "business block must precede technical tables"
    print("  HTML business block first: PASS")

    # HTML contains all required sections
    for section in ["Vehicle A Drift", "Vehicle B Navigation", "Disturbance Event Log"]:
        assert section in h, f"Section missing: {section}"
    print("  HTML all sections present: PASS")

    # Write to temp dir
    with tempfile.TemporaryDirectory() as tmp:
        jp, hp = generate_report(out, output_dir=tmp, run_date="20260330")
        assert os.path.exists(jp) and os.path.getsize(jp) > 100
        assert os.path.exists(hp) and os.path.getsize(hp) > 1000
        print(f"  Files written ({os.path.getsize(hp)//1024} KB HTML): PASS")

    print()
    print("All checks passed.")
