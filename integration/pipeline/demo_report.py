"""
integration/pipeline/demo_report.py
MicroMind Pre-HIL — Phase 2 Demo Report Generator

Generates a self-contained HTML report from a latency JSON file.
One command produces the OEM-ready demo artefact.

Usage:
    from integration.pipeline.demo_report import generate_report
    generate_report(
        latency_json="cp2_latency.json",
        output_html="micromind_demo_report.html",
        run_label="BCMP-1 SITL Demo Run",
    )
"""

from __future__ import annotations

import json
import os
from datetime import datetime


def generate_report(
    latency_json: str,
    output_html: str,
    run_label: str = "MicroMind Pre-HIL Demo Run",
) -> str:
    """Generate self-contained HTML demo report from latency JSON.

    Args:
        latency_json: path to JSON file from LatencyMonitor.export_json()
        output_html:  output path for HTML report
        run_label:    title label for the run

    Returns:
        path to generated HTML file
    """
    with open(latency_json) as f:
        d = json.load(f)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def gate_badge(passed: bool) -> str:
        if passed:
            return '<span style="background:#16a34a;color:#fff;padding:2px 10px;border-radius:4px;font-weight:700;font-size:13px">PASS</span>'
        return '<span style="background:#dc2626;color:#fff;padding:2px 10px;border-radius:4px;font-weight:700;font-size:13px">FAIL</span>'

    all_pass = all([
        d.get("eskf_gate_pass"), d.get("e2e_gate_pass"),
        d.get("rate_gate_pass"), d.get("cpu_gate_pass"),
        d.get("memory_gate_pass"),
    ])

    overall_badge = gate_badge(all_pass)
    overall_color = "#16a34a" if all_pass else "#dc2626"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>MicroMind — Demo Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:Arial,sans-serif;background:#0f172a;color:#e2e8f0;line-height:1.5}}
.container{{max-width:960px;margin:32px auto;padding:0 24px}}
.header{{background:linear-gradient(135deg,#1e3a5f,#0f1f3d);border-radius:12px;padding:32px;margin-bottom:24px;border:1px solid #2563eb}}
.header h1{{font-size:32px;font-weight:900;color:#fff;margin-bottom:4px}}
.header .sub{{color:#93c5fd;font-size:14px}}
.header .ts{{color:#64748b;font-size:12px;margin-top:8px}}
.overall{{background:#1e293b;border-radius:12px;padding:20px 28px;margin-bottom:24px;display:flex;align-items:center;gap:16px;border:2px solid {overall_color}}}
.overall .label{{font-size:18px;font-weight:700;color:#fff}}
.card{{background:#1e293b;border-radius:10px;padding:20px 24px;margin-bottom:16px}}
.card h3{{font-size:13px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px;margin-bottom:14px}}
.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.metric{{background:#0f172a;border-radius:8px;padding:14px;border:1px solid #1e293b}}
.metric .value{{font-size:26px;font-weight:900;font-family:monospace;color:#fff}}
.metric .label{{font-size:11px;color:#64748b;margin-top:4px}}
.metric .gate{{margin-top:6px}}
.tbl{{width:100%;border-collapse:collapse;font-size:13px}}
.tbl th{{background:#0f172a;color:#94a3b8;padding:8px 12px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.4px}}
.tbl td{{padding:10px 12px;border-bottom:1px solid #1e293b;color:#e2e8f0}}
.tbl tr:last-child td{{border-bottom:none}}
.footer{{text-align:center;color:#475569;font-size:11px;margin-top:32px;padding-bottom:32px}}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>MicroMind</h1>
  <div class="sub">Pre-HIL Integration Report &mdash; {run_label}</div>
  <div class="ts">Generated: {ts} &nbsp;&bull;&nbsp; Basis: micromind-autonomy Pre-HIL &nbsp;&bull;&nbsp; RESTRICTED</div>
</div>

<div class="overall">
  <div class="label">CP-2 Gate Result</div>
  {overall_badge}
  <div style="color:#64748b;font-size:13px;margin-left:auto">{d.get("n_steps",0):,} steps &bull; {d.get("run_duration_s",0):.1f}s run</div>
</div>

<div class="card">
  <h3>Latency Performance</h3>
  <div class="metric-grid">
    <div class="metric">
      <div class="value">{d.get("eskf_p95_ms",0):.3f}ms</div>
      <div class="label">ESKF P95 latency</div>
      <div class="gate">{gate_badge(d.get("eskf_gate_pass",False))} <span style="color:#64748b;font-size:11px">gate &lt;10ms</span></div>
    </div>
    <div class="metric">
      <div class="value">{d.get("e2e_p95_ms",0):.3f}ms</div>
      <div class="label">End-to-end P95 latency</div>
      <div class="gate">{gate_badge(d.get("e2e_gate_pass",False))} <span style="color:#64748b;font-size:11px">gate &lt;50ms</span></div>
    </div>
    <div class="metric">
      <div class="value">{d.get("eskf_max_ms",0):.3f}ms</div>
      <div class="label">ESKF max latency</div>
    </div>
    <div class="metric">
      <div class="value">{d.get("e2e_max_ms",0):.3f}ms</div>
      <div class="label">E2E max latency</div>
    </div>
  </div>
</div>

<div class="card">
  <h3>Control Loop</h3>
  <div class="metric-grid">
    <div class="metric">
      <div class="value">{d.get("setpoint_rate_hz",0):.1f}Hz</div>
      <div class="label">Setpoint rate (mean)</div>
      <div class="gate">{gate_badge(d.get("rate_gate_pass",False))} <span style="color:#64748b;font-size:11px">gate 20&plusmn;2Hz</span></div>
    </div>
    <div class="metric">
      <div class="value">{d.get("setpoint_min_hz",0):.1f}Hz</div>
      <div class="label">Setpoint rate (min)</div>
    </div>
    <div class="metric">
      <div class="value">{d.get("n_steps",0):,}</div>
      <div class="label">Nav loop steps</div>
    </div>
    <div class="metric">
      <div class="value">{d.get("run_duration_s",0):.1f}s</div>
      <div class="label">Run duration</div>
    </div>
  </div>
</div>

<div class="card">
  <h3>Compute Profile</h3>
  <div class="metric-grid">
    <div class="metric">
      <div class="value">{d.get("cpu_mean_pct",0):.1f}%</div>
      <div class="label">CPU mean</div>
      <div class="gate">{gate_badge(d.get("cpu_gate_pass",False))} <span style="color:#64748b;font-size:11px">gate &lt;60%</span></div>
    </div>
    <div class="metric">
      <div class="value">{d.get("cpu_peak_pct",0):.1f}%</div>
      <div class="label">CPU peak</div>
    </div>
    <div class="metric">
      <div class="value">{d.get("memory_peak_mb",0):.1f}MB</div>
      <div class="label">Memory peak RSS</div>
      <div class="gate">{gate_badge(d.get("memory_gate_pass",False))} <span style="color:#64748b;font-size:11px">gate &lt;500MB</span></div>
    </div>
  </div>
</div>

<div class="card">
  <h3>Gate Summary</h3>
  <table class="tbl">
    <tr><th>Gate</th><th>Metric</th><th>Threshold</th><th>Actual</th><th>Result</th></tr>
    <tr><td>ESKF latency</td><td>P95 ESKF step</td><td>&lt;10ms</td><td>{d.get("eskf_p95_ms",0):.3f}ms</td><td>{gate_badge(d.get("eskf_gate_pass",False))}</td></tr>
    <tr><td>E2E latency</td><td>P95 end-to-end</td><td>&lt;50ms</td><td>{d.get("e2e_p95_ms",0):.3f}ms</td><td>{gate_badge(d.get("e2e_gate_pass",False))}</td></tr>
    <tr><td>Setpoint rate</td><td>Min window Hz</td><td>&ge;18Hz</td><td>{d.get("setpoint_min_hz",0):.1f}Hz</td><td>{gate_badge(d.get("rate_gate_pass",False))}</td></tr>
    <tr><td>CPU usage</td><td>Mean CPU %</td><td>&lt;60%</td><td>{d.get("cpu_mean_pct",0):.1f}%</td><td>{gate_badge(d.get("cpu_gate_pass",False))}</td></tr>
    <tr><td>Memory</td><td>Peak RSS</td><td>&lt;500MB</td><td>{d.get("memory_peak_mb",0):.1f}MB</td><td>{gate_badge(d.get("memory_gate_pass",False))}</td></tr>
  </table>
</div>

<div class="footer">
  MicroMind Pre-HIL Integration Report &bull; India-origin autonomous navigation &bull; No ITAR restriction &bull; RESTRICTED
</div>

</div>
</body>
</html>"""

    with open(output_html, "w") as f:
        f.write(html)
    return output_html
