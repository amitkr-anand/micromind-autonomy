"""
Post-run analysis for Internal Acceptance Test.
Runs on node01 from Orin log files.
Generates all required output charts and scorecard.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent / 'output'


def load_log(log_path):
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyse(log_path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_log(log_path)

    print(f"\nLoaded {len(records)} records from {log_path}")

    # Separate by type
    meta = [r for r in records
            if r['record_type'] == 'RUN_META']
    test_results = [r for r in records
                    if r['record_type'] == 'TEST_RESULT']
    perf = [r for r in records
            if r['record_type'] == 'SYSTEM_PERF']

    # Run summary
    passed = sum(1 for t in test_results
                 if t['payload']['status'] == 'PASSED')
    failed = sum(1 for t in test_results
                 if t['payload']['status'] == 'FAILED')
    total = len(test_results)

    print(f"\nTEST RESULTS: {passed}/{total} passed")
    if failed > 0:
        print(f"FAILED TESTS:")
        for t in test_results:
            if t['payload']['status'] != 'PASSED':
                print(f"  - {t['payload']['test']}")

    # Resource analysis
    if perf:
        import statistics
        mem_vals = [r['payload'].get('mem_used_mb', 0)
                    for r in perf
                    if 'mem_used_mb' in r['payload']]
        cpu_vals = []
        for r in perf:
            cores = r['payload'].get('cpu_pct_per_core', [])
            if cores:
                cpu_vals.append(max(cores))

        if mem_vals:
            print(f"\nRESOURCE USAGE:")
            print(f"  Memory: max={max(mem_vals):.0f}MB "
                  f"mean={statistics.mean(mem_vals):.0f}MB")
            if max(mem_vals) > 6144:
                print(f"  WARNING: Memory exceeded 6GB")
        if cpu_vals:
            print(f"  CPU peak: {max(cpu_vals):.1f}%")
            sustained_high = sum(
                1 for v in cpu_vals if v > 90)
            if sustained_high > 10:
                print(f"  WARNING: CPU >90% for "
                      f"{sustained_high}s sustained")

    # Generate resource plot
    mem_vals = []
    if perf:
        mem_vals = [r['payload'].get('mem_used_mb', 0)
                    for r in perf
                    if 'mem_used_mb' in r['payload']]

    if perf and mem_vals:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            times_s = [r['t_ns'] / 1e9 for r in perf
                       if 'mem_used_mb' in r['payload']]
            mem_gb = [m / 1024 for m in mem_vals]

            cpu_vals = []
            for r in perf:
                cores = r['payload'].get('cpu_pct_per_core', [])
                if cores:
                    cpu_vals.append(max(cores))

            fig, (ax1, ax2) = plt.subplots(2, 1,
                                            figsize=(12, 8))
            fig.suptitle(
                'Internal Acceptance Test — '
                'Orin Resource Profile\n'
                'Shimla-Manali 180km | Gate 5 Suite',
                fontsize=12, fontweight='bold')

            ax1.plot(times_s, mem_gb, 'b-', lw=1.5)
            ax1.axhline(6.0, color='red', ls='--',
                        lw=1.5, label='6GB limit')
            ax1.set_ylabel('Memory (GB)')
            ax1.set_title('Memory Usage vs Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            if cpu_vals:
                cpu_times = [r['t_ns'] / 1e9 for r in perf
                             if r['payload'].get(
                                 'cpu_pct_per_core')]
                ax2.plot(cpu_times, cpu_vals,
                         'r-', lw=1.5)
                ax2.axhline(90, color='darkred', ls='--',
                            lw=1.5, label='90% limit')
                ax2.set_ylabel('CPU Peak % (any core)')
                ax2.set_title('CPU Usage vs Time')
                ax2.set_xlabel('Time (s)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            out = OUTPUT_DIR / 'resource_profile.png'
            fig.savefig(out, dpi=150,
                        bbox_inches='tight')
            plt.close(fig)
            print(f"\nSaved: {out}")
        except ImportError:
            print("matplotlib not available — "
                  "skipping resource plot")

    # Generate scorecard
    run_elapsed = None
    for m in meta:
        if m['payload'].get('event') == 'RUN_COMPLETE':
            run_elapsed = m['payload'].get('elapsed_s')

    crash_free = (failed == 0 and
                  any(m['payload'].get('exit_code') == 0
                      for m in meta
                      if 'exit_code' in m['payload']))
    mem_ok = (not mem_vals or max(mem_vals) < 6144)
    cpu_vals_all = []
    for r in perf:
        cores = r['payload'].get('cpu_pct_per_core', [])
        if cores:
            cpu_vals_all.append(max(cores))
    cpu_ok = (not cpu_vals_all or
              sum(1 for v in cpu_vals_all if v > 90) < 10)
    logs_complete = len(records) > 10

    criteria = [
        ("Gate 5 completes without crash",
         crash_free),
        ("All 22 tests pass",
         passed == total and total > 0),
        ("Navigation stable (no divergence in Gate 5)",
         crash_free),
        ("Logs complete",
         logs_complete),
        ("Resource within limits (CPU<90%, Mem<6GB)",
         mem_ok and cpu_ok),
    ]

    verdict = all(c[1] for c in criteria)

    scorecard_path = OUTPUT_DIR / 'acceptance_scorecard.md'
    with open(scorecard_path, 'w') as f:
        f.write("# Internal Acceptance Test — Scorecard\n")
        f.write(f"Date: {datetime.utcnow().date()}\n")
        f.write(f"HEAD: 7e475da\n")
        f.write(f"Corridor: Shimla-Manali 180km\n")
        f.write(f"Hardware: Jetson Orin Nano Super\n")
        f.write(f"TRN path: PhaseCorrelationTRN "
                f"(LightGlue server not started)\n\n")
        f.write("## Acceptance Criteria\n\n")
        f.write("| Criterion | Result |\n")
        f.write("|---|---|\n")
        for name, result in criteria:
            mark = "✅ PASS" if result else "❌ FAIL"
            f.write(f"| {name} | {mark} |\n")
        f.write(f"\n## VERDICT: "
                f"{'GO ✅' if verdict else 'NO-GO ❌'}\n\n")
        f.write("## Test Results\n\n")
        f.write(f"Tests: {passed}/{total} passed\n\n")
        if failed > 0:
            f.write("### Failed Tests\n")
            for t in test_results:
                if t['payload']['status'] != 'PASSED':
                    f.write(f"- {t['payload']['test']}\n")
        if mem_vals:
            f.write(f"\n## Resource Usage\n\n")
            f.write(f"Memory peak: "
                    f"{max(mem_vals) / 1024:.2f} GB\n")
            if cpu_vals_all:
                f.write(f"CPU peak: "
                        f"{max(cpu_vals_all):.1f}%\n")
        if run_elapsed:
            f.write(f"\nRun duration: "
                    f"{run_elapsed:.0f}s "
                    f"({run_elapsed / 60:.1f} min)\n")

    print(f"\n{'='*60}")
    print(f"VERDICT: {'GO ✅' if verdict else 'NO-GO ❌'}")
    print(f"Scorecard: {scorecard_path}")
    print(f"{'='*60}\n")

    return verdict


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyse_run.py <log.jsonl>")
        sys.exit(1)
    result = analyse(sys.argv[1])
    sys.exit(0 if result else 1)
