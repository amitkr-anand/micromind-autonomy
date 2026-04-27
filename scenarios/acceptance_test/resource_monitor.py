"""
Resource monitor for Internal Acceptance Test.
Runs as daemon thread. Samples CPU/memory/GPU at 1Hz.
Writes to shared log list passed by reference.
"""

import threading
import time
import json
from pathlib import Path


def _sample_cpu_mem():
    """Sample CPU and memory. Returns dict."""
    result = {}
    try:
        import psutil
        result['cpu_pct_per_core'] = psutil.cpu_percent(
            interval=None, percpu=True)
        vm = psutil.virtual_memory()
        result['mem_used_mb'] = vm.used / 1024 / 1024
        result['mem_total_mb'] = vm.total / 1024 / 1024
        result['mem_pct'] = vm.percent
    except ImportError:
        # Fallback: read /proc/stat
        try:
            with open('/proc/meminfo') as f:
                lines = {l.split(':')[0]: l.split(':')[1].strip()
                         for l in f if ':' in l}
            mem_total = int(
                lines.get('MemTotal', '0 kB').split()[0])
            mem_avail = int(
                lines.get('MemAvailable', '0 kB').split()[0])
            result['mem_used_mb'] = (
                mem_total - mem_avail) / 1024
            result['mem_total_mb'] = mem_total / 1024
            result['mem_pct'] = round(
                (mem_total - mem_avail) / mem_total * 100, 1)
            result['cpu_pct_per_core'] = []
        except Exception as e:
            result['error'] = str(e)
    return result


def _sample_gpu():
    """Sample GPU if available. Returns dict."""
    result = {}
    try:
        import subprocess
        out = subprocess.run(
            ['tegrastats', '--interval', '1'],
            capture_output=True, text=True, timeout=2)
        if out.stdout:
            result['tegrastats_line'] = out.stdout.strip()
    except Exception:
        pass
    return result


class ResourceMonitor:
    def __init__(self, log_records, t0_ns,
                 interval_s=1.0):
        self.log_records = log_records
        self.t0_ns = t0_ns
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            t_ns = time.monotonic_ns() - self.t0_ns
            payload = _sample_cpu_mem()
            payload.update(_sample_gpu())
            self.log_records.append({
                't_ns': t_ns,
                'record_type': 'SYSTEM_PERF',
                'payload': payload
            })
            time.sleep(self.interval_s)
