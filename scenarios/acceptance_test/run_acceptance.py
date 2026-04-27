"""
Internal Acceptance Test Runner — Gate 5 Extension
Corridor: Shimla-Manali 180km
Platform: Jetson Orin Nano Super
Date: 27 April 2026

Calls Gate 5 test suite via subprocess and monitors
resource usage simultaneously. Logs all output.
Does NOT modify any production file.
Consumer of core/ public APIs only.
"""

import subprocess
import sys
import time
import json
import threading
import os
from pathlib import Path
from datetime import datetime

# Import resource monitor from same package
sys.path.insert(0, str(Path(__file__).parent))
from resource_monitor import ResourceMonitor

REPO_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = Path(__file__).parent / 'output'
LOG_DIR = Path(__file__).parent / 'logs'


def run_acceptance():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = LOG_DIR / f'acceptance_run_{run_id}.jsonl'
    raw_log_path = LOG_DIR / f'gate5_raw_{run_id}.txt'

    print(f"\n{'='*60}")
    print(f"INTERNAL ACCEPTANCE TEST — GATE 5 EXTENSION")
    print(f"Run ID: {run_id}")
    print(f"Corridor: Shimla-Manali 180km")
    print(f"{'='*60}\n")

    log_records = []
    t0_ns = time.monotonic_ns()

    # Write run metadata
    log_records.append({
        't_ns': 0,
        'record_type': 'RUN_META',
        'payload': {
            'run_id': run_id,
            'corridor': 'shimla_manali_180km',
            'test_suite': 'gate5',
            'lightglue': 'natural_state_no_server',
            'trn_path': 'PhaseCorrelationTRN_fallback',
            'start_utc': datetime.utcnow().isoformat(),
            'head': '7e475da',
        }
    })

    # Start resource monitor
    monitor = ResourceMonitor(log_records, t0_ns)
    monitor.start()
    print("[MONITOR] Resource monitoring started (1Hz)")

    # Run Gate 5 test suite
    print("[RUN] Starting Gate 5 test suite...")
    print("[RUN] test_gate5_corridor.py")
    print("[RUN] This exercises 180km Shimla-Manali "
          "corridor with compound fault injection\n")

    gate5_cmd = [
        sys.executable, '-m', 'pytest',
        'tests/test_gate5_corridor.py',
        '-v', '--tb=short',
        '--log-cli-level=INFO',
        '-p', 'no:timeout',
    ]

    start_time = time.time()

    with open(raw_log_path, 'w') as raw_log:
        proc = subprocess.Popen(
            gate5_cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream and log output
        test_results = []
        for line in proc.stdout:
            line = line.rstrip()
            raw_log.write(line + '\n')
            raw_log.flush()

            # Print live
            print(line)

            # Parse test results
            if ' PASSED' in line or ' FAILED' in line \
               or ' ERROR' in line:
                t_ns = time.monotonic_ns() - t0_ns
                status = ('PASSED' if 'PASSED' in line
                          else 'FAILED' if 'FAILED' in line
                          else 'ERROR')
                test_name = line.split('::')[-1].split()[0] \
                    if '::' in line else line
                log_records.append({
                    't_ns': t_ns,
                    'record_type': 'TEST_RESULT',
                    'payload': {
                        'test': test_name,
                        'status': status,
                        'elapsed_s': round(
                            time.time() - start_time, 1)
                    }
                })

        proc.wait()

    elapsed = time.time() - start_time
    exit_code = proc.returncode

    # Stop resource monitor
    monitor.stop()
    print(f"\n[MONITOR] Resource monitoring stopped")

    # Write completion record
    log_records.append({
        't_ns': time.monotonic_ns() - t0_ns,
        'record_type': 'RUN_META',
        'payload': {
            'event': 'RUN_COMPLETE',
            'exit_code': exit_code,
            'elapsed_s': round(elapsed, 1),
            'end_utc': datetime.utcnow().isoformat(),
        }
    })

    # Write JSONL log
    with open(log_path, 'w') as f:
        for record in log_records:
            f.write(json.dumps(record) + '\n')

    print(f"\n{'='*60}")
    print(f"ACCEPTANCE RUN COMPLETE")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Exit code: {exit_code}")
    print(f"Gate 5 result: "
          f"{'PASS' if exit_code == 0 else 'FAIL'}")
    print(f"Log: {log_path}")
    print(f"Raw: {raw_log_path}")
    print(f"{'='*60}\n")

    return exit_code, log_path, raw_log_path


if __name__ == '__main__':
    exit_code, log_path, raw_log = run_acceptance()
    sys.exit(exit_code)
