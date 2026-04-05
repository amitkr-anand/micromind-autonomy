"""
tests/test_prehil_rc8.py
MicroMind Pre-HIL — RC-8 Logger Non-Blocking 200 Hz

Closes OI-18 (RC-8).  Acceptance gate SD-07.

SD-07: RC-8 completeness >= 0.99, no single call > 5 ms.

FusionLogger (core/fusion/fusion_logger.py schema 08.1) is a synchronous
in-memory logger — entries are appended to a Python list per call.
There is no async queue and no separate consumer thread, so "drop_count"
is computed as (submitted - written).  With synchronous append, drops
should be zero; the 1% tolerance accommodates any external interruption.

Mandatory caveat (RC-8, SD-09):
    RC-8 validated on Ryzen 7 9700X.  Jetson Orin renice +10 / 2-core
    taskset variant not characterised (OI-25).  This result does not
    constitute Jetson clearance.
"""

from __future__ import annotations

import time
import unittest

from core.fusion.fusion_logger import FusionLogger


class TestRC8LoggerNonBlocking(unittest.TestCase):
    """RC-8 — Logger non-blocking 200 Hz 60 s drop-rate test.  Gate SD-07.

    Mandatory caveat:
        RC-8 validated on Ryzen 7 9700X.  Jetson Orin renice +10 /
        2-core taskset variant not characterised (OI-25).  This result
        does not constitute Jetson clearance.
    """

    TOTAL_ENTRIES = 12_000      # 200 Hz × 60 s
    MAX_CALL_MS   = 5.0         # SD-07: no single call may block > 5 ms
    MIN_COMPLETENESS = 0.99     # SD-07: written / submitted >= 0.99

    def test_rc8_completeness_and_latency(self) -> None:
        """
        Stimulus:
          Instantiate FusionLogger.
          Submit 12 000 minimal log_propagate() entries (200 Hz × 60 s).
          Measure wall-clock time per submission call.
        Assertions:
          log_completeness = written / submitted >= 0.99.
          No single submission call blocks for > 5 ms.
          drop_count (submitted - written) == 0 or within 1% tolerance.
        """
        logger = FusionLogger(log_path=None, label="RC8_test")

        dt      = 1.0 / 200.0
        t_sim   = 0.0

        worst_call_ms: float       = 0.0
        worst_call_idx: int | None = None
        submitted  = 0
        slow_calls: list[tuple[int, float]] = []

        for i in range(self.TOTAL_ENTRIES):
            t0 = time.perf_counter()
            logger.log_propagate(
                t           = t_sim,
                trace_P     = 3.0,
                vio_mode    = "NOMINAL",
                dt_since_vio= 0.0,
                error_m     = None,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            submitted += 1
            t_sim     += dt

            if elapsed_ms > worst_call_ms:
                worst_call_ms  = elapsed_ms
                worst_call_idx = i

            if elapsed_ms > self.MAX_CALL_MS:
                slow_calls.append((i, elapsed_ms))

        written = len(logger._entries)

        # ── Mandatory caveat ──────────────────────────────────────────────
        print(
            f"\nRC-8: submitted={submitted}, written={written}, "
            f"drop_count={submitted - written}, "
            f"worst_call={worst_call_ms:.3f} ms (step {worst_call_idx})."
        )
        print("RC-8 validated on Ryzen 7 9700X.")
        print("Jetson Orin renice +10 / 2-core taskset variant not characterised (OI-25).")
        print("This result does not constitute Jetson clearance.")

        # ── Assertions ────────────────────────────────────────────────────
        log_completeness = written / submitted
        self.assertGreaterEqual(
            log_completeness, self.MIN_COMPLETENESS,
            f"log_completeness {log_completeness:.4f} must be >= {self.MIN_COMPLETENESS} "
            f"(written={written}, submitted={submitted})",
        )

        drop_count = submitted - written
        max_drops  = int(submitted * (1.0 - self.MIN_COMPLETENESS))
        self.assertLessEqual(
            drop_count, max_drops,
            f"drop_count {drop_count} must be <= {max_drops} "
            f"(1% tolerance of {submitted} submitted entries)",
        )

        if slow_calls:
            details = ", ".join(f"step {idx}: {ms:.2f} ms" for idx, ms in slow_calls[:5])
            self.fail(
                f"RC-8 FAILED: {len(slow_calls)} call(s) exceeded {self.MAX_CALL_MS} ms. "
                f"First 5: [{details}]"
            )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main(verbosity=2)
