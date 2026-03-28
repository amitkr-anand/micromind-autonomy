"""
integration/pipeline/latency_monitor.py
MicroMind Pre-HIL — Phase 2 Output Pipeline + Timing

LatencyMonitor: instruments each pipeline stage and computes P95 metrics.

Measurement plan (v1.2 §5.1):
  Stage 1: sensor read (driver.read() overhead)
  Stage 2: ESKF propagation (ins_propagate + eskf.propagate)
  Stage 3: decision → setpoint enqueue
  Stage 4: setpoint → MAVLink send (measured in T-SP)
  End-to-end: Stage 1 start → Stage 4 complete

Thresholds (v1.2 §5.2):
  ESKF update P95:    < 10ms
  End-to-end P95:     < 50ms
  Setpoint rate:      20 ± 2 Hz
  CPU average:        < 60%
  Memory peak:        < 500MB
  Heartbeat jitter:   ± 100ms on 2Hz send

ADR-0 v1.1 D-2: all timestamps use time.perf_counter_ns() for ns precision.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

import psutil


# ---------------------------------------------------------------------------
# Per-step latency record
# ---------------------------------------------------------------------------

@dataclass
class LatencyRecord:
    """Timing record for one 200Hz navigation loop iteration."""
    step:          int
    t_sensor_ns:   int    # perf_counter_ns at start of driver.read()
    t_eskf_ns:     int    # perf_counter_ns after ESKF propagation
    t_decision_ns: int    # perf_counter_ns after setpoint enqueue
    t_send_ns:     int    # perf_counter_ns after MAVLink send (0 if not sent)
    imu_stale:     bool
    vio_mode:      str


# ---------------------------------------------------------------------------
# LatencySummary — produced at end of run
# ---------------------------------------------------------------------------

@dataclass
class LatencySummary:
    """Statistical summary of latency over a full run."""
    n_steps:           int
    run_duration_s:    float

    # ESKF stage (ns → ms)
    eskf_p50_ms:       float
    eskf_p95_ms:       float
    eskf_p99_ms:       float
    eskf_max_ms:       float

    # End-to-end (ns → ms)
    e2e_p50_ms:        float
    e2e_p95_ms:        float
    e2e_p99_ms:        float
    e2e_max_ms:        float

    # Setpoint rate
    setpoint_rate_hz:  float
    setpoint_min_hz:   float   # minimum over 5s windows
    setpoint_drops:    int

    # System
    cpu_mean_pct:      float
    cpu_peak_pct:      float
    memory_peak_mb:    float

    # Gate results
    eskf_gate_pass:    bool    # eskf_p95_ms < 10ms
    e2e_gate_pass:     bool    # e2e_p95_ms < 50ms
    rate_gate_pass:    bool    # setpoint_min_hz >= 18Hz
    cpu_gate_pass:     bool    # cpu_mean_pct < 60%
    memory_gate_pass:  bool    # memory_peak_mb < 500MB

    @property
    def all_gates_pass(self) -> bool:
        return all([self.eskf_gate_pass, self.e2e_gate_pass,
                    self.rate_gate_pass, self.cpu_gate_pass,
                    self.memory_gate_pass])


# ---------------------------------------------------------------------------
# LatencyMonitor
# ---------------------------------------------------------------------------

class LatencyMonitor:
    """Instruments the LivePipeline navigation loop for latency measurement.

    Usage:
        monitor = LatencyMonitor()
        monitor.start()

        # Inside nav loop:
        with monitor.step() as s:
            s.mark_sensor()   # after driver.read()
            s.mark_eskf()     # after ESKF propagation
            s.mark_decision() # after setpoint enqueue

        monitor.mark_send()   # called from T-SP after MAVLink send

        summary = monitor.stop()
        monitor.export_csv("latency.csv")
        monitor.export_json("latency.json")

    Args:
        max_records:  maximum records to keep in memory (default 100000 = ~500s)
        window_s:     setpoint rate window in seconds (default 5.0)
    """

    def __init__(self, max_records: int = 100_000, window_s: float = 5.0) -> None:
        self._records:    list[LatencyRecord] = []
        self._lock        = threading.Lock()
        self._max_records = max_records
        self._window_s    = window_s
        self._step_count  = 0
        self._send_times: deque = deque()

        # System monitoring
        self._cpu_samples:  list[float] = []
        self._mem_samples:  list[float] = []
        self._process       = psutil.Process()
        self._sys_thread:   Optional[threading.Thread] = None
        self._stop_event    = threading.Event()

        # Timing
        self._t_start:  float = 0.0
        self._t_stop:   float = 0.0

        # Current step context (written by nav loop, read by mark_send)
        self._current_step: int = 0
        self._t_sensor_ns:  int = 0
        self._t_eskf_ns:    int = 0
        self._t_decision_ns:int = 0

        # Setpoint rate tracking (5s windows)
        self._sp_window_rates: list[float] = []
        self._sp_window_start: float = 0.0
        self._sp_window_count: int   = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start monitoring. Launch system metrics thread."""
        self._t_start = time.monotonic()
        self._sp_window_start = self._t_start
        self._stop_event.clear()
        self._sys_thread = threading.Thread(
            target=self._sys_loop, name="T-METRICS", daemon=True
        )
        self._sys_thread.start()

    def stop(self) -> LatencySummary:
        """Stop monitoring and compute summary statistics."""
        self._t_stop = time.monotonic()
        self._stop_event.set()
        if self._sys_thread:
            self._sys_thread.join(timeout=3.0)
        return self._compute_summary()

    # ------------------------------------------------------------------
    # Step instrumentation — called from T-NAV
    # ------------------------------------------------------------------

    def begin_step(self, step: int, imu_stale: bool = False,
                   vio_mode: str = "NOMINAL") -> None:
        """Mark start of step. Call immediately before driver.read()."""
        self._current_step  = step
        self._t_sensor_ns   = time.perf_counter_ns()
        self._current_imu_stale = imu_stale
        self._current_vio_mode  = vio_mode

    def mark_eskf(self) -> None:
        """Mark ESKF propagation complete."""
        self._t_eskf_ns = time.perf_counter_ns()

    def mark_decision(self) -> None:
        """Mark setpoint enqueue complete."""
        self._t_decision_ns = time.perf_counter_ns()
        # Record without send time (will be updated by mark_send)
        record = LatencyRecord(
            step=self._current_step,
            t_sensor_ns=self._t_sensor_ns,
            t_eskf_ns=self._t_eskf_ns,
            t_decision_ns=self._t_decision_ns,
            t_send_ns=0,
            imu_stale=getattr(self, "_current_imu_stale", False),
            vio_mode=getattr(self, "_current_vio_mode", "NOMINAL"),
        )
        with self._lock:
            if len(self._records) < self._max_records:
                self._records.append(record)
            self._step_count += 1

    # ------------------------------------------------------------------
    # Send instrumentation — called from T-SP
    # ------------------------------------------------------------------

    def mark_send(self) -> None:
        """Mark MAVLink setpoint send complete. Called from T-SP."""
        t_send = time.perf_counter_ns()
        now_m  = time.monotonic()

        # Update send time on most recent record
        with self._lock:
            if self._records:
                rec = self._records[-1]
                # Create new frozen record with send time
                self._records[-1] = LatencyRecord(
                    step=rec.step,
                    t_sensor_ns=rec.t_sensor_ns,
                    t_eskf_ns=rec.t_eskf_ns,
                    t_decision_ns=rec.t_decision_ns,
                    t_send_ns=t_send,
                    imu_stale=rec.imu_stale,
                    vio_mode=rec.vio_mode,
                )

        # Setpoint rate tracking
        self._send_times.append(now_m)
        # Trim to window
        while self._send_times and now_m - self._send_times[0] > self._window_s:
            self._send_times.popleft()

        self._sp_window_count += 1
        elapsed_window = now_m - self._sp_window_start
        if elapsed_window >= self._window_s:
            rate_hz = self._sp_window_count / elapsed_window
            self._sp_window_rates.append(rate_hz)
            self._sp_window_start = now_m
            self._sp_window_count = 0

    # ------------------------------------------------------------------
    # Properties for live monitoring
    # ------------------------------------------------------------------

    @property
    def current_setpoint_hz(self) -> float:
        """Current setpoint rate over last window_s seconds."""
        if not self._send_times:
            return 0.0
        return len(self._send_times) / self._window_s

    @property
    def step_count(self) -> int:
        return self._step_count

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, path: str) -> None:
        """Export per-step latency records to CSV."""
        with self._lock:
            records = list(self._records)

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "eskf_ms", "e2e_ms", "imu_stale", "vio_mode"])
            for r in records:
                eskf_ms = (r.t_eskf_ns - r.t_sensor_ns) / 1e6
                e2e_ms  = ((r.t_send_ns - r.t_sensor_ns) / 1e6
                           if r.t_send_ns > 0 else 0.0)
                w.writerow([r.step, f"{eskf_ms:.3f}", f"{e2e_ms:.3f}",
                            r.imu_stale, r.vio_mode])

    def export_json(self, path: str) -> None:
        """Export summary statistics to JSON."""
        summary = self._compute_summary()
        with open(path, "w") as f:
            json.dump(asdict(summary), f, indent=2)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_summary(self) -> LatencySummary:
        with self._lock:
            records = list(self._records)

        n = len(records)
        if n == 0:
            return LatencySummary(
                n_steps=0, run_duration_s=0,
                eskf_p50_ms=0, eskf_p95_ms=0, eskf_p99_ms=0, eskf_max_ms=0,
                e2e_p50_ms=0, e2e_p95_ms=0, e2e_p99_ms=0, e2e_max_ms=0,
                setpoint_rate_hz=0, setpoint_min_hz=0, setpoint_drops=0,
                cpu_mean_pct=0, cpu_peak_pct=0, memory_peak_mb=0,
                eskf_gate_pass=False, e2e_gate_pass=False,
                rate_gate_pass=False, cpu_gate_pass=False,
                memory_gate_pass=False,
            )

        eskf_ms = [(r.t_eskf_ns - r.t_sensor_ns) / 1e6 for r in records]
        e2e_ms  = [(r.t_send_ns - r.t_sensor_ns) / 1e6
                   for r in records if r.t_send_ns > 0]

        def pct(data, p):
            if not data: return 0.0
            idx = max(0, int(len(data) * p / 100) - 1)
            return sorted(data)[idx]

        dur = self._t_stop - self._t_start if self._t_stop > self._t_start else 1.0
        sp_rates = self._sp_window_rates or [self.current_setpoint_hz]
        sp_mean  = statistics.mean(sp_rates) if sp_rates else 0.0
        sp_min   = min(sp_rates) if sp_rates else 0.0

        cpu_mean = statistics.mean(self._cpu_samples) if self._cpu_samples else 0.0
        cpu_peak = max(self._cpu_samples) if self._cpu_samples else 0.0
        mem_peak = max(self._mem_samples) if self._mem_samples else 0.0

        ep95 = pct(eskf_ms, 95)
        xp95 = pct(e2e_ms,  95)

        return LatencySummary(
            n_steps=n,
            run_duration_s=dur,
            eskf_p50_ms=pct(eskf_ms, 50),
            eskf_p95_ms=ep95,
            eskf_p99_ms=pct(eskf_ms, 99),
            eskf_max_ms=max(eskf_ms),
            e2e_p50_ms=pct(e2e_ms, 50),
            e2e_p95_ms=xp95,
            e2e_p99_ms=pct(e2e_ms,  99),
            e2e_max_ms=max(e2e_ms) if e2e_ms else 0.0,
            setpoint_rate_hz=sp_mean,
            setpoint_min_hz=sp_min,
            setpoint_drops=0,
            cpu_mean_pct=cpu_mean,
            cpu_peak_pct=cpu_peak,
            memory_peak_mb=mem_peak,
            eskf_gate_pass=ep95 < 10.0,
            e2e_gate_pass=xp95 < 50.0,
            rate_gate_pass=sp_min >= 18.0,
            cpu_gate_pass=cpu_mean < 60.0,
            memory_gate_pass=mem_peak < 500.0,
        )

    def _sys_loop(self) -> None:
        """Sample CPU and memory at 1Hz in background thread."""
        while not self._stop_event.is_set():
            try:
                cpu = psutil.cpu_percent(interval=1.0)
                mem = self._process.memory_info().rss / (1024 * 1024)
                self._cpu_samples.append(cpu)
                self._mem_samples.append(mem)
            except Exception:
                time.sleep(1.0)
