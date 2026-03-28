"""
integration/tests/test_prehil_bridge.py
MicroMind Pre-HIL — Phase 1.5 Bridge Tests

Gates — TimeReference:
  G-TREF-01: TimeReference instantiates
  G-TREF-02: is_synced is False before sync_from_px4()
  G-TREF-03: time_boot_ms() returns non-negative int before sync
  G-TREF-04: time_boot_ms() increases monotonically
  G-TREF-05: sync_from_px4() sets is_synced True
  G-TREF-06: px4_offset_ms is correct after sync
  G-TREF-07: time_boot_ms() reflects px4 offset after sync
  G-TREF-08: monotonic_s() increases over time
  G-TREF-09: monotonic_s() starts near zero
"""

import sys, os, time
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from integration.bridge.time_reference import TimeReference


class TestTimeReference:
    def test_G_TREF_01_instantiates(self):
        """G-TREF-01: TimeReference instantiates."""
        assert TimeReference() is not None

    def test_G_TREF_02_not_synced_before_sync(self):
        """G-TREF-02: is_synced is False before sync_from_px4()."""
        ref = TimeReference()
        assert ref.is_synced is False

    def test_G_TREF_03_time_boot_ms_non_negative(self):
        """G-TREF-03: time_boot_ms() returns non-negative int before sync."""
        ref = TimeReference()
        t = ref.time_boot_ms()
        assert isinstance(t, int)
        assert t >= 0

    def test_G_TREF_04_time_boot_ms_monotonic(self):
        """G-TREF-04: time_boot_ms() increases monotonically."""
        ref = TimeReference()
        t1 = ref.time_boot_ms()
        time.sleep(0.02)
        t2 = ref.time_boot_ms()
        assert t2 >= t1

    def test_G_TREF_05_synced_after_sync(self):
        """G-TREF-05: sync_from_px4() sets is_synced True."""
        ref = TimeReference()
        ref.sync_from_px4(px4_time_boot_ms=100_000)
        assert ref.is_synced is True

    def test_G_TREF_06_px4_offset_computed(self):
        """G-TREF-06: px4_offset_ms is non-zero after sync with non-zero px4 time."""
        ref = TimeReference()
        ref.sync_from_px4(px4_time_boot_ms=500_000)
        assert ref.px4_offset_ms != 0.0

    def test_G_TREF_07_time_boot_ms_reflects_offset(self):
        """G-TREF-07: time_boot_ms() reflects PX4 offset after sync."""
        ref = TimeReference()
        # Sync to a large PX4 boot time (e.g. PX4 has been running 100s)
        ref.sync_from_px4(px4_time_boot_ms=100_000)
        t = ref.time_boot_ms()
        # Should be approximately 100000ms (PX4 boot) + small local elapsed
        assert t >= 99_000   # generous lower bound
        assert t <= 200_000  # generous upper bound

    def test_G_TREF_08_monotonic_s_increases(self):
        """G-TREF-08: monotonic_s() increases over time."""
        ref = TimeReference()
        t1 = ref.monotonic_s()
        time.sleep(0.02)
        t2 = ref.monotonic_s()
        assert t2 > t1

    def test_G_TREF_09_monotonic_s_starts_near_zero(self):
        """G-TREF-09: monotonic_s() starts near zero (within 1s of construction)."""
        ref = TimeReference()
        assert ref.monotonic_s() < 1.0
