"""
tests/test_bcmp2_at6.py
BCMP-2 Acceptance Test AT-6 — Three-Seed Repeatability and Endurance.

Governs SB-5 closure.  17 gates across 4 groups.
All 17 gates must be in PASS state for SB-5 to be declared closed.
No partial pass is defined.

Gate groups
-----------
  Group 1 (G-01–G-09): C-2 drift envelope compliance — 3 seeds × 3 boundaries
  Group 2 (G-10–G-12): FSM phase transition chain consistency across seeds
  Group 3 (G-13–G-15): Endurance stability — crash, memory, log completeness
  Group 4 (G-16–G-17): Report generation and BCMP-2 Closure Report

Endurance duration
------------------
  Group 3 gates are marked @pytest.mark.endurance.
  Duration is controlled by the AT6_ENDURANCE_HOURS environment variable:

    Default (CI, 2 minutes):
        pytest tests/test_bcmp2_at6.py -m endurance

    Full overnight (4 hours):
        AT6_ENDURANCE_HOURS=4 pytest tests/test_bcmp2_at6.py -m endurance

  All endurance gates are genuinely runnable in both modes.
  They are not skipped in CI — only shortened.

Retry policy
------------
  Groups 1 and 2: no retry (per AT6_Acceptance_Criteria.md §6).
  Group 3: no retry for crashes/OOM; investigate before re-run.
  Group 4: report generation may be fixed and re-run without re-running missions.

SIL caveats (mandatory — appear in BCMP-2 Closure Report G-17)
---------------------------------------------------------------
  1. C-2 envelopes calibrated on BASELINE IMU (ARW 0.05 °/√hr). STIM300
     typical ARW is 0.15 °/√hr. Re-calibration pending (OI-03).
  2. Navigation correction is RADALT-NCC stub (orthophoto AD-01 not yet
     integrated). AT-6 results reflect RADALT-NCC mechanism.
  3. DMRL is a rule-based stub. No CNN-based decoy rejection tested.
  4. Vehicle A drift is illustrative, not precision mechanisation (AD-15).
  5. OpenVINS validation is indoor EuRoC only (≤ 130 m). Km-scale and
     outdoor validation pending (OI-07).

Canonical references (locked — do not modify without PD approval)
-----------------------------------------------------------------
  Phase chain reference:  ['NOMINAL', 'EW_AWARE']  — seed 42, SB-3 baseline
  C-2 gate key type:      int  (60, 100, 120)
  c2_gates_all_passed:    result['comparison']['vehicle_a_c2_gates_all_passed']
  phase_sequence:         result['vehicle_b_phase_sequence']  (top-level key)
  Envelope source:        PHASE_ENVELOPES from bcmp2_drift_envelopes.py

Seeds
-----
  42  — nominal (SB-3 baseline)
  101 — alternate weather stress (used in AT-2)
  303 — stress, virgin (not used in AT-1 through AT-5 or N=300 Monte Carlo)
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scenarios.bcmp2.bcmp2_runner import run_bcmp2, BCMP2RunConfig
from scenarios.bcmp2.bcmp2_report import generate_report
from scenarios.bcmp2.bcmp2_drift_envelopes import PHASE_ENVELOPES


# ---------------------------------------------------------------------------
# Canonical constants
# ---------------------------------------------------------------------------

# G-10/G-11/G-12: FSM phase chain established from seed 42 full-mission run
# at SB-3 closure.  Do not modify without PD approval and re-running AT-6.
SB3_REFERENCE_CHAIN: list[str] = ["NOMINAL", "EW_AWARE"]

# Endurance duration: AT6_ENDURANCE_HOURS env var, default 2/60 hr = 2 min
_ENDURANCE_HOURS:   float = float(os.environ.get("AT6_ENDURANCE_HOURS",
                                                   str(2.0 / 60.0)))
_ENDURANCE_SECONDS: float = _ENDURANCE_HOURS * 3600.0

# RSS sampling interval: fine-grained in CI, 60 s overnight
_RSS_INTERVAL_S: float = 10.0 if _ENDURANCE_SECONDS < 300.0 else 60.0

# G-14 pass/warn thresholds
_RSS_SLOPE_PASS_MB_HR: float = 25.0

# G-15 log completeness threshold (mission success rate over endurance loop)
_LOG_COMPLETENESS_MIN: float = 0.99

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedup_chain(seq: list[str]) -> list[str]:
    """Collapse consecutive identical states → compact transition chain."""
    if not seq:
        return []
    out = [seq[0]]
    for s in seq[1:]:
        if s != out[-1]:
            out.append(s)
    return out


def _rss_mb() -> float:
    """
    Current process RSS in MB.  Reads /proc/self/status (Linux, no psutil).
    Returns 0.0 if unavailable (non-Linux or permission error).
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0  # kB → MB
    except OSError:
        pass
    return 0.0


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of ys vs xs.  Returns 0.0 for < 2 points."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    var = sum((xs[i] - mean_x) ** 2 for i in range(n))
    return (cov / var) if var > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Fixtures — scope="module": one 150 km run per seed, shared across all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run_seed42():
    """Seed 42 — nominal.  SB-3 baseline."""
    return run_bcmp2(BCMP2RunConfig(seed=42, max_km=150.0, verbose=False))


@pytest.fixture(scope="module")
def run_seed101():
    """Seed 101 — alternate weather stress.  Used in AT-2."""
    return run_bcmp2(BCMP2RunConfig(seed=101, max_km=150.0, verbose=False))


@pytest.fixture(scope="module")
def run_seed303():
    """Seed 303 — stress, virgin.  Not used in AT-1 through AT-5."""
    return run_bcmp2(BCMP2RunConfig(seed=303, max_km=150.0, verbose=False))


# ---------------------------------------------------------------------------
# Group 1 — Drift Envelope Gates  G-01 through G-09
# ---------------------------------------------------------------------------

class TestAT6Group1DriftEnvelope:
    """
    G-01–G-09: Vehicle A cross-track error at km 60, 100, 120 must fall within
    the C-2 Monte Carlo envelope (P5 floor, P99 ceiling) for all three seeds.

    Envelope values are read from PHASE_ENVELOPES (not hardcoded here):
      km 60:  floor=5 m,  ceiling=80 m
      km 100: floor=12 m, ceiling=350 m
      km 120: floor=15 m, ceiling=650 m

    Vehicle A breach of the corridor (for seeds 101 and 303) is a PASS signal
    for the dual-track comparison and does not fail any envelope gate.

    Retry policy: none.  Failing result requires root-cause investigation.
    """

    # ── Seed 42 ──────────────────────────────────────────────────────────────

    def test_G01_seed42_km60(self, run_seed42):
        """G-01: seed 42 — Vehicle A drift at km 60 within C-2 envelope."""
        env = PHASE_ENVELOPES[60]
        obs = run_seed42["vehicle_a"]["drift_at_km60_m"]
        assert env["floor"] <= obs <= env["ceiling"], (
            f"G-01 FAIL seed42 km60: {obs:.1f} m not in "
            f"[{env['floor']}, {env['ceiling']}] m"
        )

    def test_G02_seed42_km100(self, run_seed42):
        """G-02: seed 42 — Vehicle A drift at km 100 within C-2 envelope."""
        env = PHASE_ENVELOPES[100]
        obs = run_seed42["vehicle_a"]["drift_at_km100_m"]
        assert env["floor"] <= obs <= env["ceiling"], (
            f"G-02 FAIL seed42 km100: {obs:.1f} m not in "
            f"[{env['floor']}, {env['ceiling']}] m"
        )

    def test_G03_seed42_km120(self, run_seed42):
        """G-03: seed 42 — Vehicle A drift at km 120 within C-2 envelope."""
        env = PHASE_ENVELOPES[120]
        obs = run_seed42["vehicle_a"]["drift_at_km120_m"]
        assert env["floor"] <= obs <= env["ceiling"], (
            f"G-03 FAIL seed42 km120: {obs:.1f} m not in "
            f"[{env['floor']}, {env['ceiling']}] m"
        )

    # ── Seed 101 ─────────────────────────────────────────────────────────────

    def test_G04_seed101_km60(self, run_seed101):
        """G-04: seed 101 — Vehicle A drift at km 60 within C-2 envelope."""
        env = PHASE_ENVELOPES[60]
        obs = run_seed101["vehicle_a"]["drift_at_km60_m"]
        assert env["floor"] <= obs <= env["ceiling"], (
            f"G-04 FAIL seed101 km60: {obs:.1f} m not in "
            f"[{env['floor']}, {env['ceiling']}] m"
        )

    def test_G05_seed101_km100(self, run_seed101):
        """G-05: seed 101 — Vehicle A drift at km 100 within C-2 envelope."""
        env = PHASE_ENVELOPES[100]
        obs = run_seed101["vehicle_a"]["drift_at_km100_m"]
        assert env["floor"] <= obs <= env["ceiling"], (
            f"G-05 FAIL seed101 km100: {obs:.1f} m not in "
            f"[{env['floor']}, {env['ceiling']}] m"
        )

    def test_G06_seed101_km120(self, run_seed101):
        """G-06: seed 101 — Vehicle A drift at km 120 within C-2 envelope."""
        env = PHASE_ENVELOPES[120]
        obs = run_seed101["vehicle_a"]["drift_at_km120_m"]
        assert env["floor"] <= obs <= env["ceiling"], (
            f"G-06 FAIL seed101 km120: {obs:.1f} m not in "
            f"[{env['floor']}, {env['ceiling']}] m"
        )

    # ── Seed 303 ─────────────────────────────────────────────────────────────

    def test_G07_seed303_km60(self, run_seed303):
        """G-07: seed 303 — Vehicle A drift at km 60 within C-2 envelope."""
        env = PHASE_ENVELOPES[60]
        obs = run_seed303["vehicle_a"]["drift_at_km60_m"]
        assert env["floor"] <= obs <= env["ceiling"], (
            f"G-07 FAIL seed303 km60: {obs:.1f} m not in "
            f"[{env['floor']}, {env['ceiling']}] m"
        )

    def test_G08_seed303_km100(self, run_seed303):
        """G-08: seed 303 — Vehicle A drift at km 100 within C-2 envelope."""
        env = PHASE_ENVELOPES[100]
        obs = run_seed303["vehicle_a"]["drift_at_km100_m"]
        assert env["floor"] <= obs <= env["ceiling"], (
            f"G-08 FAIL seed303 km100: {obs:.1f} m not in "
            f"[{env['floor']}, {env['ceiling']}] m"
        )

    def test_G09_seed303_km120(self, run_seed303):
        """G-09: seed 303 — Vehicle A drift at km 120 within C-2 envelope."""
        env = PHASE_ENVELOPES[120]
        obs = run_seed303["vehicle_a"]["drift_at_km120_m"]
        assert env["floor"] <= obs <= env["ceiling"], (
            f"G-09 FAIL seed303 km120: {obs:.1f} m not in "
            f"[{env['floor']}, {env['ceiling']}] m"
        )


# ---------------------------------------------------------------------------
# Group 2 — Phase Transition Consistency Gates  G-10 through G-12
# ---------------------------------------------------------------------------

class TestAT6Group2PhaseChain:
    """
    G-10–G-12: Vehicle B FSM phase transition chain must be identical across
    all three seeds.

    "Identical" means same ordered sequence of state IDs after deduplication
    of consecutive repeats.  Timing differences are permitted.

    Reference chain: SB3_REFERENCE_CHAIN = ['NOMINAL', 'EW_AWARE']
    Established from seed 42 150 km full-mission run at SB-3 closure.

    G-10: seed 42 matches SB-3 reference (identity check against locked constant)
    G-11: seed 101 matches seed 42 chain from this run
    G-12: seed 303 matches seed 42 chain from this run

    If any seed produces a different chain, the gate fails without retry.
    Investigation must precede re-run.
    """

    def test_G10_seed42_chain_matches_sb3_reference(self, run_seed42):
        """G-10: seed 42 FSM chain matches SB-3 locked reference."""
        seq   = run_seed42.get("vehicle_b_phase_sequence", [])
        chain = _dedup_chain(seq)
        assert chain == SB3_REFERENCE_CHAIN, (
            f"G-10 FAIL: seed 42 chain {chain} != "
            f"SB-3 reference {SB3_REFERENCE_CHAIN}. "
            "If the runner has changed, update SB3_REFERENCE_CHAIN with PD approval."
        )

    def test_G11_seed101_chain_matches_seed42(self, run_seed42, run_seed101):
        """G-11: seed 101 FSM chain identical to seed 42 chain (this run)."""
        ref_chain = _dedup_chain(run_seed42.get("vehicle_b_phase_sequence", []))
        chain_101 = _dedup_chain(run_seed101.get("vehicle_b_phase_sequence", []))
        assert chain_101 == ref_chain, (
            f"G-11 FAIL: seed 101 chain {chain_101} != seed 42 chain {ref_chain}"
        )

    def test_G12_seed303_chain_matches_seed42(self, run_seed42, run_seed303):
        """G-12: seed 303 FSM chain identical to seed 42 chain (this run)."""
        ref_chain = _dedup_chain(run_seed42.get("vehicle_b_phase_sequence", []))
        chain_303 = _dedup_chain(run_seed303.get("vehicle_b_phase_sequence", []))
        assert chain_303 == ref_chain, (
            f"G-12 FAIL: seed 303 chain {chain_303} != seed 42 chain {ref_chain}"
        )


# ---------------------------------------------------------------------------
# Group 3 — Endurance Stability Gates  G-13 through G-15
# ---------------------------------------------------------------------------

class TestAT6Group3Endurance:
    """
    G-13–G-15: Process stability, memory growth, and mission completion rate
    over a sustained endurance run on seed 42.

    The endurance loop calls run_bcmp2() repeatedly until the wall-clock
    duration expires.  RSS is sampled at _RSS_INTERVAL_S intervals.  A
    mission that raises an exception counts as a crash (G-13) and as a
    failed attempt in the completion rate (G-15).

    Duration control (AT6_ENDURANCE_HOURS env var):
      CI default — 2/60 hours = 2 minutes
      Full overnight — AT6_ENDURANCE_HOURS=4

    All three gates call _run_endurance() which caches its result at the
    class level.  The endurance loop runs once regardless of which gates
    are collected.

    Pass criteria:
      G-13: crash_count == 0
      G-14: RSS linear regression slope ≤ 25.0 MB/hour
      G-15: missions_completed / missions_attempted ≥ 0.99

    Register the mark to suppress pytest warnings:
      Add "endurance" to markers in pytest.ini or conftest.py.
    """

    _result: dict | None = None  # class-level cache — populated on first call

    @classmethod
    def _run_endurance(cls) -> dict:
        """
        Execute the endurance loop and cache the result.

        Safe to call from multiple test methods — runs once, returns cached
        result on subsequent calls.
        """
        if cls._result is not None:
            return cls._result

        config       = BCMP2RunConfig(seed=42, max_km=150.0, verbose=False)
        duration_s   = _ENDURANCE_SECONDS
        rss_interval = _RSS_INTERVAL_S

        rss_trace: list[tuple[float, float]] = []  # (elapsed_s, rss_mb)
        crash_count  = 0
        missions_run = 0
        t_start      = time.monotonic()

        while True:
            elapsed = time.monotonic() - t_start
            if elapsed >= duration_s:
                break

            try:
                run_bcmp2(config)
                missions_run += 1
            except Exception as exc:  # noqa: BLE001
                crash_count += 1
                # Record crash but continue — G-13 counts total crashes.
                print(f"\n[endurance] crash #{crash_count}: {exc}")

            # Sample RSS after each mission, at interval.
            # Sampling post-mission (not pre) excludes the cold-start module-load
            # spike from the slope regression.  The first measurement reflects
            # warm steady-state RSS, not Python import overhead.
            elapsed = time.monotonic() - t_start
            since_last_sample = (
                elapsed - rss_trace[-1][0] if rss_trace else rss_interval
            )
            if since_last_sample >= rss_interval:
                rss_trace.append((elapsed, _rss_mb()))

        # Final RSS sample at end of run
        rss_trace.append((time.monotonic() - t_start, _rss_mb()))

        # RSS slope (MB/hour) via linear regression over the trace
        ts = [t for t, _ in rss_trace]
        rs = [r for _, r in rss_trace]
        slope_per_s = _linear_slope(ts, rs)
        rss_slope_mb_hr = slope_per_s * 3600.0

        # Mission completion rate (analogue of log_completeness)
        total_attempts  = missions_run + crash_count
        log_completeness = (
            missions_run / total_attempts if total_attempts > 0 else 1.0
        )

        cls._result = {
            "crash_count":       crash_count,
            "missions_run":      missions_run,
            "total_attempts":    total_attempts,
            "rss_slope_mb_hr":   rss_slope_mb_hr,
            "rss_trace":         rss_trace,
            "log_completeness":  log_completeness,
            "duration_s":        time.monotonic() - t_start,
            "endurance_hours":   _ENDURANCE_HOURS,
        }
        return cls._result

    @pytest.mark.endurance
    def test_G13_process_stability_zero_crashes(self):
        """
        G-13: Zero crashes and zero OOM kills across the endurance run.

        A crash is any exception raised by run_bcmp2().  The loop continues
        after each crash so the full duration is exercised.
        Fail condition: crash_count > 0.
        """
        r = self._run_endurance()
        print(
            f"\nG-13 endurance: missions={r['missions_run']}, "
            f"crashes={r['crash_count']}, "
            f"duration={r['duration_s']:.0f}s "
            f"({r['endurance_hours']:.3f}h)"
        )
        assert r["crash_count"] == 0, (
            f"G-13 FAIL: {r['crash_count']} crash(es) in "
            f"{r['total_attempts']} missions over {r['duration_s']:.0f}s. "
            "Investigate root cause before re-run."
        )

    @pytest.mark.endurance
    def test_G14_memory_growth_slope(self):
        """
        G-14: RSS linear regression slope ≤ 25.0 MB/hour over the endurance run.

        Fail condition: slope > 25.0 MB/hr.
        Hard fail condition: > 50 MB/hr (likely active leak).
        RSS is sampled from /proc/self/status at _RSS_INTERVAL_S intervals.
        If RSS is unavailable (non-Linux), slope will be 0.0 — gate passes
        but provides no evidence.
        """
        r = self._run_endurance()
        slope = r["rss_slope_mb_hr"]
        samples = len(r["rss_trace"])
        print(
            f"\nG-14 RSS slope: {slope:.3f} MB/hr over {samples} samples, "
            f"duration={r['duration_s']:.0f}s"
        )
        if samples < 3:
            pytest.xfail(
                f"G-14: only {samples} RSS samples collected — "
                "run was too short for reliable slope measurement. "
                "Re-run with AT6_ENDURANCE_HOURS ≥ 1 for a valid result."
            )
        assert slope <= _RSS_SLOPE_PASS_MB_HR, (
            f"G-14 FAIL: RSS slope {slope:.2f} MB/hr exceeds "
            f"{_RSS_SLOPE_PASS_MB_HR} MB/hr limit. "
            f"Trace ({samples} points): first={r['rss_trace'][0]}, "
            f"last={r['rss_trace'][-1]}"
        )

    @pytest.mark.endurance
    def test_G15_log_completeness(self):
        """
        G-15: Mission completion rate (missions_completed / missions_attempted)
        ≥ 99% across the full endurance run.

        This is the process-level analogue of the RC-8 FusionLogger
        log_completeness gate (SD-07).  A mission counts as failed if
        run_bcmp2() raises an exception.

        Fail condition: log_completeness < 0.99.
        """
        r = self._run_endurance()
        lc = r["log_completeness"]
        print(
            f"\nG-15 log_completeness: {lc:.4f} "
            f"({r['missions_run']}/{r['total_attempts']} missions)"
        )
        assert lc >= _LOG_COMPLETENESS_MIN, (
            f"G-15 FAIL: log_completeness {lc:.4f} < {_LOG_COMPLETENESS_MIN} "
            f"({r['crash_count']} failure(s) in {r['total_attempts']} attempts)"
        )


# ---------------------------------------------------------------------------
# Group 4 — Report and Closure Gates  G-16 through G-17
# ---------------------------------------------------------------------------

class TestAT6Group4ReportAndClosure:
    """
    G-16: HTML report generated for all three seeds.
          - Business comparison block appears before technical tables (§8.3)
          - File size > 2 kB and < 5 MB
          - No external CDN dependencies (must be air-gap safe)

    G-17: BCMP-2 Closure Report exists at artifacts/BCMP2_ClosureReport.md
          and contains all required sections and all five mandatory SIL caveats.

          This gate fails until the closure report is authored and committed.
          Fix: write the report, commit it, then re-run Group 4 only — no
          need to re-run missions (AT6_Acceptance_Criteria.md §6 retry policy).
    """

    def test_G16_html_reports_all_three_seeds(
        self, run_seed42, run_seed101, run_seed303
    ):
        """
        G-16: HTML report must generate for all three seeds; business block
        before technical tables; < 5 MB; no external CDN dependencies.
        """
        with tempfile.TemporaryDirectory() as tmp:
            for seed, result in [
                (42,  run_seed42),
                (101, run_seed101),
                (303, run_seed303),
            ]:
                _, hp = generate_report(
                    result,
                    output_dir=tmp,
                    run_date=f"at6_seed{seed}",
                )
                assert os.path.exists(hp), (
                    f"G-16 FAIL seed {seed}: HTML report not written at {hp}"
                )

                size_bytes = os.path.getsize(hp)
                assert size_bytes > 2_000, (
                    f"G-16 FAIL seed {seed}: report too small "
                    f"({size_bytes} bytes) — likely empty or truncated"
                )
                assert size_bytes < 5 * 1024 * 1024, (
                    f"G-16 FAIL seed {seed}: report too large "
                    f"({size_bytes / 1e6:.1f} MB > 5 MB limit)"
                )

                with open(hp) as fh:
                    html = fh.read()

                assert "Mission Outcome" in html, (
                    f"G-16 FAIL seed {seed}: 'Mission Outcome' not in report"
                )
                assert "Technical Evidence" in html, (
                    f"G-16 FAIL seed {seed}: 'Technical Evidence' not in report"
                )

                business_pos  = html.index("Mission Outcome")
                technical_pos = html.index("Technical Evidence")
                assert business_pos < technical_pos, (
                    f"G-16 FAIL seed {seed}: business block (pos {business_pos}) "
                    f"not before technical tables (pos {technical_pos}) — "
                    "violates §8.3 ordering requirement"
                )

                # Air-gap safe: no external CDN dependencies
                for cdn_marker in ("cdn.", "googleapis", "jsdelivr", "unpkg"):
                    assert cdn_marker not in html, (
                        f"G-16 FAIL seed {seed}: report references external "
                        f"CDN '{cdn_marker}' — not air-gap safe"
                    )

    def test_G17_closure_report_exists_and_complete(self):
        """
        G-17: BCMP-2 Closure Report must exist at artifacts/BCMP2_ClosureReport.md
        and contain all required sections and all five mandatory SIL caveats.

        Required sections (case-insensitive substring match):
          'executive summary', 'dual-track', 'C-2', 'SIL', 'programme state'

        Mandatory SIL caveats (exact token match — per §8 of AT-6 spec):
          'BASELINE'  — IMU calibration caveat (OI-03)
          'RADALT'    — navigation correction caveat (OI-05)
          'DMRL'      — decoy rejection caveat (OI-06)
          'AD-15'     — Vehicle A illustrative model caveat
          'EuRoC'     — OpenVINS validation scope caveat (OI-07)

        This gate will FAIL until the closure report is authored and committed.
        Re-run Group 4 only after committing the report (no mission re-run needed).
        """
        repo_root    = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        closure_path = os.path.join(repo_root, "artifacts", "BCMP2_ClosureReport.md")

        assert os.path.exists(closure_path), (
            f"G-17 FAIL: BCMP2_ClosureReport.md not found at:\n"
            f"  {closure_path}\n"
            "Author and commit the closure report, then re-run Group 4."
        )

        with open(closure_path) as fh:
            content = fh.read()

        required_sections = [
            "executive summary",
            "dual-track",
            "C-2",
            "SIL",
            "programme state",
        ]
        for section in required_sections:
            assert section.lower() in content.lower(), (
                f"G-17 FAIL: closure report missing required section: '{section}'"
            )

        # Exact token match for each SIL caveat — these strings must appear
        # verbatim to ensure the caveats are substantive, not just referenced.
        required_caveats = [
            ("BASELINE", "IMU calibration caveat (OI-03)"),
            ("RADALT",   "navigation correction caveat (OI-05 / AD-01)"),
            ("DMRL",     "decoy rejection stub caveat (OI-06)"),
            ("AD-15",    "Vehicle A illustrative model caveat"),
            ("EuRoC",    "OpenVINS km-scale validation caveat (OI-07)"),
        ]
        for token, description in required_caveats:
            assert token in content, (
                f"G-17 FAIL: closure report missing mandatory SIL caveat: "
                f"'{token}' ({description})"
            )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke: Group 1 and 2 only (no endurance, no closure report check)
    import subprocess
    import sys as _sys
    _sys.exit(subprocess.call([
        "python", "-m", "pytest",
        __file__,
        "-m", "not endurance",
        "-v",
    ]))
