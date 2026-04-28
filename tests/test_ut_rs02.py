"""
tests/test_ut_rs02.py
MicroMind / NanoCorteX — UT-RS-02: Log Rolling Policy

Validates RollingMissionLog against SRS §12 RS-02 / E-04.
GAP-09 closure gate.

Gates:
    UT-RS-02-01: Roll on size threshold (max_size_bytes exceeded)
    UT-RS-02-02: Roll on time threshold (max_age_min exceeded)
    UT-RS-02-03: CRITICAL (P0) entries never dropped; present in critical_events.log
    UT-RS-02-04: Non-critical drops tracked; LOG_BUFFER_HIGH emitted at 80%
    UT-RS-02-05: LOG_ROLLED entry carries required fields (new_file_path, previous_file_size_mb)

Req IDs: RS-02, E-04
SRS ref: §12
Governance: Code Governance Manual v3.4
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logs.mission_log_schema import (
    LogCategory,
    L10sSERecord,
    MissionLogEntry,
    RollingMissionLog,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nc_entry(state: str = "NOMINAL") -> MissionLogEntry:
    """Non-critical entry (NAVIGATION category)."""
    return MissionLogEntry(
        category=LogCategory.NAVIGATION,
        mission_id="rs02-test",
        state=state,
    )


def _crit_entry() -> MissionLogEntry:
    """CRITICAL P0 entry (L10S_SE category)."""
    return MissionLogEntry(
        category=LogCategory.L10S_SE,
        mission_id="rs02-test",
        state="TERMINAL",
        l10s_se=L10sSERecord(
            decision="CONTINUE",
            eo_lock_confidence=0.95,
            corridor_clear=True,
            l10s_elapsed_s=7.0,
        ),
    )


def _scan_files(log: RollingMissionLog, state: str) -> list[dict]:
    """Return all parsed log entries with the given state field across all files."""
    matches = []
    for fpath in log.all_log_files:
        if not Path(fpath).exists():
            continue
        for line in Path(fpath).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("state") == state:
                    matches.append(obj)
            except json.JSONDecodeError:
                pass
    return matches


# ---------------------------------------------------------------------------
# UT-RS-02-01: Roll on size threshold
# ---------------------------------------------------------------------------

def test_roll_on_size(tmp_path):
    """
    Writing entries past max_size_bytes must trigger a roll.
    LOG_ROLLED entry must appear and a new file must be created.
    """
    log = RollingMissionLog(
        output_dir=tmp_path / "logs",
        critical_log_path=tmp_path / "critical.jsonl",
        max_size_mb=0.001,   # 1024 bytes — crossed after ~2 entries
        max_age_min=60.0,
    )

    for _ in range(10):
        log.add(_nc_entry())
    log.close()

    # At least one roll must have occurred
    assert len(log.all_log_files) >= 2, (
        f"Expected >=2 log files after size-based roll; got {len(log.all_log_files)}"
    )

    # LOG_ROLLED must appear somewhere across all files
    rolled_entries = _scan_files(log, "LOG_ROLLED")
    assert len(rolled_entries) >= 1, "LOG_ROLLED entry not found in any output file"


# ---------------------------------------------------------------------------
# UT-RS-02-02: Roll on time threshold
# ---------------------------------------------------------------------------

def test_roll_on_time(tmp_path):
    """
    Advancing the clock past max_age_min must trigger a roll on the next add().
    """
    t = [0.0]

    def mock_clock() -> float:
        return t[0]

    log = RollingMissionLog(
        output_dir=tmp_path / "logs",
        critical_log_path=tmp_path / "critical.jsonl",
        max_size_mb=200.0,
        max_age_min=30.0,
        _clock_fn=mock_clock,
    )

    # First entry at t=0 — no roll
    log.add(_nc_entry("BEFORE_ROLL"))

    # Advance to 31 minutes
    t[0] = 31 * 60.0

    # This add() should detect the elapsed time and roll
    log.add(_nc_entry("AFTER_ROLL"))
    log.close()

    assert len(log.all_log_files) >= 2, (
        "Expected >=2 log files after time-based roll"
    )

    rolled_entries = _scan_files(log, "LOG_ROLLED")
    assert len(rolled_entries) >= 1, "LOG_ROLLED entry not found after time-based roll"


# ---------------------------------------------------------------------------
# UT-RS-02-03: CRITICAL (P0) entries never dropped
# ---------------------------------------------------------------------------

def test_critical_never_dropped(tmp_path):
    """
    After filling buffer past capacity with non-critical entries,
    CRITICAL entries must still appear in critical_events.log and
    dropped_critical_events_count must remain 0.
    """
    log = RollingMissionLog(
        output_dir=tmp_path / "logs",
        critical_log_path=tmp_path / "critical.jsonl",
        max_size_mb=200.0,
        max_age_min=30.0,
        max_buffer_entries=3,
    )

    # Fill buffer completely with non-critical entries
    for _ in range(3):
        log.add(_nc_entry())

    # One more non-critical: will be dropped (buffer full)
    log.add(_nc_entry("SHOULD_DROP"))

    # CRITICAL entry: must NOT be dropped
    log.add(_crit_entry())
    log.close()

    # Invariant: no CRITICAL events were ever dropped
    assert log.dropped_critical_events_count == 0

    # critical_events.log must exist and contain the L10S_SE entry
    crit_path = tmp_path / "critical.jsonl"
    assert crit_path.exists(), "critical_events.log not created"

    entries = [
        json.loads(ln)
        for ln in crit_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    categories = [e.get("category") for e in entries]
    assert "L10S_SE" in categories, (
        f"L10S_SE entry missing from critical_events.log; found categories: {categories}"
    )


# ---------------------------------------------------------------------------
# UT-RS-02-04: Non-critical drops tracked; LOG_BUFFER_HIGH at 80%
# ---------------------------------------------------------------------------

def test_dropped_count_tracked(tmp_path):
    """
    Non-critical entries written past max_buffer_entries must increment
    dropped_records_total.  LOG_BUFFER_HIGH must be emitted when the
    non-critical entry count reaches 80% of max_buffer_entries.
    """
    log = RollingMissionLog(
        output_dir=tmp_path / "logs",
        critical_log_path=tmp_path / "critical.jsonl",
        max_size_mb=200.0,
        max_age_min=30.0,
        max_buffer_entries=5,
    )

    # Write 6 non-critical entries; 6th exceeds buffer of 5
    for _ in range(6):
        log.add(_nc_entry())
    log.close()

    # At least 1 entry must have been dropped
    assert log.dropped_records_total >= 1, (
        f"Expected dropped_records_total >= 1; got {log.dropped_records_total}"
    )

    # LOG_BUFFER_HIGH must appear in the output files (emitted at 80% = 4 entries)
    hw_entries = _scan_files(log, "LOG_BUFFER_HIGH")
    assert len(hw_entries) >= 1, "LOG_BUFFER_HIGH entry not found in output files"


# ---------------------------------------------------------------------------
# UT-RS-02-05: LOG_ROLLED entry carries required fields
# ---------------------------------------------------------------------------

def test_log_rolled_event_fields(tmp_path):
    """
    After a size-based roll, the LOG_ROLLED entry must carry
    new_file_path and previous_file_size_mb in its notes payload.
    """
    log = RollingMissionLog(
        output_dir=tmp_path / "logs",
        critical_log_path=tmp_path / "critical.jsonl",
        max_size_mb=0.001,   # 1024 bytes — triggers roll quickly
        max_age_min=60.0,
    )

    for _ in range(10):
        log.add(_nc_entry())
    log.close()

    rolled_entries = _scan_files(log, "LOG_ROLLED")
    assert rolled_entries, "No LOG_ROLLED entry found"

    roll_entry = rolled_entries[0]
    assert roll_entry.get("notes"), "LOG_ROLLED entry has empty notes field"

    notes = json.loads(roll_entry["notes"])
    assert "new_file_path" in notes, (
        f"new_file_path missing from LOG_ROLLED notes: {notes}"
    )
    assert "previous_file_size_mb" in notes, (
        f"previous_file_size_mb missing from LOG_ROLLED notes: {notes}"
    )
    # previous_file_size_mb must be a non-negative number
    assert isinstance(notes["previous_file_size_mb"], (int, float))
    assert notes["previous_file_size_mb"] >= 0.0
