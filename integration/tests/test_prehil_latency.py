"""
integration/tests/test_prehil_latency.py
MicroMind Pre-HIL — Phase 2 Latency Monitor Tests

Gates:
  G-LAT-01: LatencyMonitor instantiates
  G-LAT-02: begin_step/mark_eskf/mark_decision records a LatencyRecord
  G-LAT-03: ESKF latency is positive
  G-LAT-04: mark_send updates t_send_ns on most recent record
  G-LAT-05: export_csv produces valid CSV with required columns
  G-LAT-06: export_json produces valid JSON with summary fields
  G-LAT-07: LatencySummary all_gates_pass reflects individual gates
  G-LAT-08: step_count increments correctly
  G-LAT-09: stop() returns LatencySummary
  G-LAT-10: system metrics sampled (cpu/memory > 0)
"""

import sys, os, time, csv, json, tempfile
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from integration.pipeline.latency_monitor import LatencyMonitor, LatencyRecord, LatencySummary


def _run_steps(monitor, n=100):
    """Simulate n nav loop steps."""
    for i in range(n):
        monitor.begin_step(step=i)
        time.sleep(0.0001)   # ~0.1ms simulated sensor read
        monitor.mark_eskf()
        time.sleep(0.0001)   # ~0.1ms simulated ESKF
        monitor.mark_decision()
        monitor.mark_send()


class TestLatencyMonitor:
    def test_G_LAT_01_instantiates(self):
        """G-LAT-01: LatencyMonitor instantiates."""
        assert LatencyMonitor() is not None

    def test_G_LAT_02_records_latency_record(self):
        """G-LAT-02: begin_step/mark_eskf/mark_decision records a LatencyRecord."""
        m = LatencyMonitor()
        m.start()
        m.begin_step(0)
        m.mark_eskf()
        m.mark_decision()
        assert len(m._records) == 1
        assert isinstance(m._records[0], LatencyRecord)
        m.stop()

    def test_G_LAT_03_eskf_latency_positive(self):
        """G-LAT-03: ESKF latency (t_eskf - t_sensor) is positive."""
        m = LatencyMonitor()
        m.start()
        m.begin_step(0)
        time.sleep(0.001)
        m.mark_eskf()
        m.mark_decision()
        r = m._records[0]
        assert r.t_eskf_ns > r.t_sensor_ns
        m.stop()

    def test_G_LAT_04_mark_send_updates_record(self):
        """G-LAT-04: mark_send() updates t_send_ns on most recent record."""
        m = LatencyMonitor()
        m.start()
        m.begin_step(0)
        m.mark_eskf()
        m.mark_decision()
        assert m._records[0].t_send_ns == 0
        m.mark_send()
        assert m._records[0].t_send_ns > 0
        m.stop()

    def test_G_LAT_05_export_csv(self):
        """G-LAT-05: export_csv produces valid CSV with required columns."""
        m = LatencyMonitor()
        m.start()
        _run_steps(m, 10)
        summary = m.stop()
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            path = f.name
        m.export_csv(path)
        rows = list(csv.DictReader(open(path)))
        assert len(rows) == 10
        assert 'step' in rows[0]
        assert 'eskf_ms' in rows[0]
        assert 'e2e_ms' in rows[0]
        os.unlink(path)

    def test_G_LAT_06_export_json(self):
        """G-LAT-06: export_json produces valid JSON with summary fields."""
        m = LatencyMonitor()
        m.start()
        _run_steps(m, 10)
        m.stop()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        m.export_json(path)
        data = json.load(open(path))
        for key in ['n_steps', 'eskf_p95_ms', 'e2e_p95_ms',
                    'setpoint_rate_hz', 'cpu_mean_pct', 'memory_peak_mb']:
            assert key in data, f"Missing key: {key}"
        os.unlink(path)

    def test_G_LAT_07_all_gates_pass_reflects_gates(self):
        """G-LAT-07: all_gates_pass is True only when all individual gates pass."""
        s = LatencySummary(
            n_steps=100, run_duration_s=1.0,
            eskf_p50_ms=0.5, eskf_p95_ms=2.0, eskf_p99_ms=3.0, eskf_max_ms=5.0,
            e2e_p50_ms=5.0, e2e_p95_ms=15.0, e2e_p99_ms=20.0, e2e_max_ms=25.0,
            setpoint_rate_hz=20.0, setpoint_min_hz=19.5, setpoint_drops=0,
            cpu_mean_pct=30.0, cpu_peak_pct=45.0, memory_peak_mb=200.0,
            eskf_gate_pass=True, e2e_gate_pass=True, rate_gate_pass=True,
            cpu_gate_pass=True, memory_gate_pass=True,
        )
        assert s.all_gates_pass is True

        s_fail = LatencySummary(
            n_steps=100, run_duration_s=1.0,
            eskf_p50_ms=0.5, eskf_p95_ms=15.0, eskf_p99_ms=20.0, eskf_max_ms=30.0,
            e2e_p50_ms=5.0, e2e_p95_ms=60.0, e2e_p99_ms=70.0, e2e_max_ms=80.0,
            setpoint_rate_hz=20.0, setpoint_min_hz=19.5, setpoint_drops=0,
            cpu_mean_pct=30.0, cpu_peak_pct=45.0, memory_peak_mb=200.0,
            eskf_gate_pass=False, e2e_gate_pass=False, rate_gate_pass=True,
            cpu_gate_pass=True, memory_gate_pass=True,
        )
        assert s_fail.all_gates_pass is False

    def test_G_LAT_08_step_count_increments(self):
        """G-LAT-08: step_count increments correctly."""
        m = LatencyMonitor()
        m.start()
        _run_steps(m, 50)
        assert m.step_count == 50
        m.stop()

    def test_G_LAT_09_stop_returns_summary(self):
        """G-LAT-09: stop() returns LatencySummary."""
        m = LatencyMonitor()
        m.start()
        _run_steps(m, 20)
        summary = m.stop()
        assert isinstance(summary, LatencySummary)
        assert summary.n_steps == 20

    def test_G_LAT_10_system_metrics_sampled(self):
        """G-LAT-10: CPU and memory samples collected after 2s run."""
        m = LatencyMonitor()
        m.start()
        time.sleep(2.1)   # let _sys_loop sample at least twice
        _run_steps(m, 10)
        m.stop()
        assert len(m._cpu_samples) >= 1
        assert len(m._mem_samples) >= 1
        assert m._mem_samples[0] > 0   # process RSS > 0
