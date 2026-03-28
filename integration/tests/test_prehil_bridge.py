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


# ---------------------------------------------------------------------------
# BridgeLogger tests
# ---------------------------------------------------------------------------

import json, os, time, tempfile, threading
from integration.bridge.bridge_logger import BridgeLogger, BridgeLogEntry


class TestBridgeLogger:
    def _make_logger(self, tmp_path_str):
        return BridgeLogger(log_path=tmp_path_str, source_type='sim')

    def test_G_BLOG_01_instantiates(self):
        """G-BLOG-01: BridgeLogger instantiates."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            assert BridgeLogger(f.name) is not None
            os.unlink(f.name)

    def test_G_BLOG_02_start_stop(self):
        """G-BLOG-02: start() and stop() complete without error."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            path = f.name
        logger = BridgeLogger(path, source_type='sim')
        logger.start()
        logger.stop()
        os.unlink(path)

    def test_G_BLOG_03_t_log_is_daemon(self):
        """G-BLOG-03: T-LOG thread is a daemon thread."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            path = f.name
        logger = BridgeLogger(path)
        logger.start()
        t_log = next((t for t in threading.enumerate() if t.name == 'T-LOG'), None)
        assert t_log is not None
        assert t_log.daemon is True
        logger.stop()
        os.unlink(path)

    def test_G_BLOG_04_log_writes_entries(self):
        """G-BLOG-04: log() entries appear in JSON-lines file."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False, mode='w') as f:
            path = f.name
        logger = BridgeLogger(path, source_type='sim')
        logger.start()
        logger.log("HEARTBEAT", "RX", seq=1, base_mode=29)
        logger.log("SET_POSITION_TARGET_LOCAL_NED", "TX", seq=2, x_m=0.)
        time.sleep(0.2)
        logger.stop()
        lines = open(path).readlines()
        assert len(lines) >= 2
        first = json.loads(lines[0])
        assert first['msg_type'] == 'HEARTBEAT'
        assert first['direction'] == 'RX'
        assert first['source_type'] == 'sim'
        assert 't_monotonic' in first
        os.unlink(path)

    def test_G_BLOG_05_required_fields_present(self):
        """G-BLOG-05: all five required fields present in every entry."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False, mode='w') as f:
            path = f.name
        logger = BridgeLogger(path, source_type='real')
        logger.start()
        logger.log("HEARTBEAT", "RX", seq=5)
        time.sleep(0.2)
        logger.stop()
        entry = json.loads(open(path).readline())
        for field in ['t_monotonic', 'msg_type', 'direction', 'seq', 'source_type']:
            assert field in entry, f"Missing field: {field}"
        os.unlink(path)

    def test_G_BLOG_06_heartbeat_rx_fields(self):
        """G-BLOG-06: log_heartbeat_rx() writes required fields."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False, mode='w') as f:
            path = f.name
        logger = BridgeLogger(path)
        logger.start()
        logger.log_heartbeat_rx(base_mode=29, custom_mode=393216,
                                 system_status=4, mavlink_version=3,
                                 target_system=1, target_component=0, seq=1)
        time.sleep(0.2)
        logger.stop()
        entry = json.loads(open(path).readline())
        assert entry['msg_type'] == 'HEARTBEAT'
        assert entry['custom_mode'] == 393216
        assert entry['derived_target_system'] == 1
        assert entry['derived_target_component'] == 0
        os.unlink(path)

    def test_G_BLOG_07_setpoint_tx_fields(self):
        """G-BLOG-07: log_setpoint_tx() writes NED fields + coordinate_frame."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False, mode='w') as f:
            path = f.name
        logger = BridgeLogger(path)
        logger.start()
        logger.log_setpoint_tx(x_m=10., y_m=5., z_m=-3., setpoint_hz=20.0)
        time.sleep(0.2)
        logger.stop()
        entry = json.loads(open(path).readline())
        assert entry['x_m'] == 10.
        assert entry['z_m'] == -3.
        assert entry['coordinate_frame'] == 1
        assert entry['setpoint_hz'] == 20.0
        os.unlink(path)

    def test_G_BLOG_08_drop_count_increments_when_full(self):
        """G-BLOG-08: drop_count increments when queue is full."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            path = f.name
        logger = BridgeLogger(path, queue_maxsize=2)
        # Do NOT start — queue fills immediately
        for _ in range(10):
            logger.log("HEARTBEAT", "RX")
        assert logger.drop_count > 0
        os.unlink(path)

    def test_G_BLOG_09_time_ref_used_when_provided(self):
        """G-BLOG-09: BridgeLogger uses TimeReference.monotonic_s() when provided."""
        from integration.bridge.time_reference import TimeReference
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False, mode='w') as f:
            path = f.name
        ref = TimeReference()
        logger = BridgeLogger(path, time_ref=ref)
        logger.start()
        logger.log("HEARTBEAT", "RX")
        time.sleep(0.2)
        logger.stop()
        entry = json.loads(open(path).readline())
        assert entry['t_monotonic'] >= 0.0
        assert entry['t_monotonic'] < 5.0   # within 5s of process start
        os.unlink(path)

    def test_G_BLOG_10_start_idempotent(self):
        """G-BLOG-10: start() is idempotent — safe to call twice."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            path = f.name
        logger = BridgeLogger(path)
        logger.start()
        logger.start()
        t_logs = [t for t in threading.enumerate() if t.name == 'T-LOG']
        assert len(t_logs) == 1
        logger.stop()
        os.unlink(path)
