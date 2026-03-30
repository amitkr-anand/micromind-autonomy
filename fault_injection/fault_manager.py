"""
BCMP-2 Fault Manager.

Thread-safe singleton registry for all active fault states.

Design
------
The FaultManager is the single source of truth for which faults are
currently active.  Both the scripted injection schedule and the Dash GUI
operator panel write to it.  The sensor and nav source proxies read from
it on every call.

Proxy contract
--------------
  - When no fault is active:  proxies are transparent pass-throughs.
  - When a fault is active:   proxies intercept the relevant data stream.
  - Neither ESKF, BIM, VIOMode, TRNStub nor any frozen core module is
    aware of the fault manager or the proxies.

Thread safety
-------------
All reads and writes are protected by threading.Lock().  The Dash GUI
callback thread and the simulation loop thread both access this object.
Pattern follows the Pre-HIL B-2/B-3 fix (threading.Event /
threading.Lock) documented in ADR-0 v1.1.

Fault IDs
---------
Defined as string constants matching the FI catalogue in the BCMP-2
architecture document v1.1 §6.

JOURNAL
-------
Built: 29 March 2026, micromind-node01.  SB-2 Step 1.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Fault ID constants  (FI catalogue from architecture doc §6)
# ---------------------------------------------------------------------------

FI_GNSS_LOSS          = "FI_GNSS_LOSS"          # FI-01: GNSS denied
FI_VIO_LOSS           = "FI_VIO_LOSS"           # FI-02/03: VIO outage
FI_RADALT_LOSS        = "FI_RADALT_LOSS"        # FI-04: RADALT lost
FI_EO_FREEZE          = "FI_EO_FREEZE"          # FI-05: EO feed frozen
FI_IMU_JITTER         = "FI_IMU_JITTER"         # FI-06: non-monotonic timestamp
FI_MAVLINK_DROP       = "FI_MAVLINK_DROP"       # FI-07: MAVLink disconnect
FI_TERRAIN_CONF_DROP  = "FI_TERRAIN_CONF_DROP"  # FI-08: terrain conf below threshold
FI_CPU_LOAD           = "FI_CPU_LOAD"           # FI-11: CPU overload
FI_MEMORY_PRESSURE    = "FI_MEMORY_PRESSURE"    # FI-12: memory pressure
FI_TIME_SKEW          = "FI_TIME_SKEW"          # FI-13: module time skew

# Composite presets (multi-fault, FI-09/10)
PRESET_VIO_GNSS       = "PRESET_VIO_GNSS"       # FI-09: VIO + GNSS
PRESET_VIO_RADALT_TERM = "PRESET_VIO_RADALT_TERM"  # FI-10: VIO + RADALT in terminal

ALL_FAULT_IDS = [
    FI_GNSS_LOSS, FI_VIO_LOSS, FI_RADALT_LOSS, FI_EO_FREEZE,
    FI_IMU_JITTER, FI_MAVLINK_DROP, FI_TERRAIN_CONF_DROP,
    FI_CPU_LOAD, FI_MEMORY_PRESSURE, FI_TIME_SKEW,
]

PRESETS: Dict[str, List[str]] = {
    PRESET_VIO_GNSS:        [FI_VIO_LOSS, FI_GNSS_LOSS],
    PRESET_VIO_RADALT_TERM: [FI_VIO_LOSS, FI_RADALT_LOSS],
}


# ---------------------------------------------------------------------------
# Fault state record
# ---------------------------------------------------------------------------

@dataclass
class FaultState:
    fault_id:    str
    active:      bool       = False
    start_time:  float      = 0.0   # wall time (time.monotonic())
    duration_s:  float      = 0.0   # 0 = indefinite until cleared
    severity:    float      = 1.0   # 1.0 = full fault; 0–1 for partial
    source:      str        = "scripted"  # "scripted" | "operator" | "preset"
    metadata:    dict       = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Return True if a finite duration fault has elapsed."""
        if not self.active:
            return False
        if self.duration_s <= 0.0:
            return False  # indefinite
        return (time.monotonic() - self.start_time) >= self.duration_s


# ---------------------------------------------------------------------------
# Event log entry
# ---------------------------------------------------------------------------

@dataclass
class FaultEvent:
    timestamp_s:  float    # wall time
    mission_km:   float    # mission distance at event time
    fault_id:     str
    action:       str      # "activated" | "cleared" | "expired"
    source:       str
    duration_s:   float
    severity:     float


# ---------------------------------------------------------------------------
# Fault Manager
# ---------------------------------------------------------------------------

class FaultManager:
    """
    Thread-safe fault state registry for BCMP-2.

    Usage (scripted):
        fm = FaultManager()
        fm.activate(FI_GNSS_LOSS, duration_s=30.0)
        ...
        if fm.is_active(FI_GNSS_LOSS):
            gnss_obs = None   # suppressed by proxy

    Usage (operator via Dash callback):
        fm.activate(FI_VIO_LOSS, duration_s=15.0, source="operator")
        fm.clear(FI_VIO_LOSS)
    """

    def __init__(self):
        self._lock:   threading.Lock = threading.Lock()
        self._faults: Dict[str, FaultState] = {
            fid: FaultState(fault_id=fid) for fid in ALL_FAULT_IDS
        }
        self._events: List[FaultEvent] = []
        self._mission_km: float = 0.0   # updated by runner each tick

    # ── Mission state (updated by runner) ─────────────────────────────────

    def update_mission_km(self, km: float) -> None:
        """Called by runner each tick to track current mission position."""
        with self._lock:
            self._mission_km = km
            # Auto-expire finite faults
            for state in self._faults.values():
                if state.active and state.is_expired():
                    self._do_clear(state, action="expired")

    # ── Activation ────────────────────────────────────────────────────────

    def activate(
        self,
        fault_id:   str,
        duration_s: float = 0.0,
        severity:   float = 1.0,
        source:     str   = "scripted",
        metadata:   dict  = None,
    ) -> None:
        """
        Activate a fault.

        Parameters
        ----------
        fault_id   : one of the FI_* constants
        duration_s : 0 = indefinite (must be cleared manually)
        severity   : 1.0 = full; 0–1 for partial degradation
        source     : "scripted" | "operator" | "preset"
        """
        with self._lock:
            if fault_id not in self._faults:
                raise ValueError(f"Unknown fault_id: {fault_id!r}. "
                                 f"Valid: {ALL_FAULT_IDS}")
            state = self._faults[fault_id]
            state.active     = True
            state.start_time = time.monotonic()
            state.duration_s = duration_s
            state.severity   = severity
            state.source     = source
            state.metadata   = metadata or {}
            self._events.append(FaultEvent(
                timestamp_s = time.monotonic(),
                mission_km  = self._mission_km,
                fault_id    = fault_id,
                action      = "activated",
                source      = source,
                duration_s  = duration_s,
                severity    = severity,
            ))

    def activate_preset(
        self,
        preset_id:  str,
        duration_s: float = 0.0,
        source:     str   = "operator",
    ) -> None:
        """Activate all faults in a named preset simultaneously."""
        if preset_id not in PRESETS:
            raise ValueError(f"Unknown preset: {preset_id!r}. "
                             f"Valid: {list(PRESETS)}")
        for fid in PRESETS[preset_id]:
            self.activate(fid, duration_s=duration_s, source=source)

    # ── Clearing ──────────────────────────────────────────────────────────

    def clear(self, fault_id: str) -> None:
        """Clear (deactivate) a specific fault."""
        with self._lock:
            if fault_id not in self._faults:
                return
            state = self._faults[fault_id]
            if state.active:
                self._do_clear(state, action="cleared")

    def clear_all(self) -> None:
        """Clear all active faults."""
        with self._lock:
            for state in self._faults.values():
                if state.active:
                    self._do_clear(state, action="cleared")

    def _do_clear(self, state: FaultState, action: str) -> None:
        """Internal clear — must be called with lock held."""
        self._events.append(FaultEvent(
            timestamp_s = time.monotonic(),
            mission_km  = self._mission_km,
            fault_id    = state.fault_id,
            action      = action,
            source      = state.source,
            duration_s  = state.duration_s,
            severity    = state.severity,
        ))
        state.active = False

    # ── Query ─────────────────────────────────────────────────────────────

    def is_active(self, fault_id: str) -> bool:
        """Return True if the fault is currently active (and not expired)."""
        with self._lock:
            state = self._faults.get(fault_id)
            if state is None:
                return False
            if state.active and state.is_expired():
                self._do_clear(state, action="expired")
                return False
            return state.active

    def severity(self, fault_id: str) -> float:
        """Return severity of a fault (0.0 if not active)."""
        with self._lock:
            state = self._faults.get(fault_id)
            if state is None or not state.active:
                return 0.0
            return state.severity

    def active_faults(self) -> Dict[str, FaultState]:
        """Return snapshot of all currently active faults."""
        with self._lock:
            return {
                fid: state
                for fid, state in self._faults.items()
                if state.active and not state.is_expired()
            }

    def active_fault_ids(self) -> List[str]:
        """Return list of currently active fault IDs."""
        return list(self.active_faults().keys())

    # ── Event log ─────────────────────────────────────────────────────────

    def event_log(self) -> List[FaultEvent]:
        """Return a copy of the full fault event log."""
        with self._lock:
            return list(self._events)

    def event_log_as_dicts(self) -> List[dict]:
        """Return event log as list of dicts for JSON serialisation."""
        with self._lock:
            return [
                {
                    "timestamp_s": e.timestamp_s,
                    "mission_km":  e.mission_km,
                    "fault_id":    e.fault_id,
                    "action":      e.action,
                    "source":      e.source,
                    "duration_s":  e.duration_s,
                    "severity":    e.severity,
                }
                for e in self._events
            ]

    def reset(self) -> None:
        """Clear all faults and wipe the event log. For test teardown only."""
        with self._lock:
            for state in self._faults.values():
                state.active = False
            self._events.clear()
            self._mission_km = 0.0

    def __repr__(self) -> str:
        active = self.active_fault_ids()
        return f"FaultManager(active={active}, events={len(self._events)})"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_INSTANCE: Optional[FaultManager] = None
_INSTANCE_LOCK = threading.Lock()


def get_fault_manager() -> FaultManager:
    """
    Return the module-level FaultManager singleton.

    The singleton is created on first call and shared by all proxies and
    the Dash GUI.  Use reset() between test runs to clear state.
    """
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = FaultManager()
        return _INSTANCE


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time

    print("FaultManager self-verification")
    print("=" * 45)

    fm = FaultManager()

    # Basic activate / query / clear
    fm.activate(FI_GNSS_LOSS, duration_s=0.0, source="scripted")
    assert fm.is_active(FI_GNSS_LOSS), "should be active"
    assert not fm.is_active(FI_VIO_LOSS), "should not be active"
    print("  activate + query:          PASS")

    fm.clear(FI_GNSS_LOSS)
    assert not fm.is_active(FI_GNSS_LOSS), "should be cleared"
    print("  clear:                     PASS")

    # Duration / auto-expiry
    fm.activate(FI_VIO_LOSS, duration_s=0.05)
    assert fm.is_active(FI_VIO_LOSS), "should be active"
    _time.sleep(0.06)
    assert not fm.is_active(FI_VIO_LOSS), "should have expired"
    print("  auto-expiry (0.05s):       PASS")

    # Preset
    fm.activate_preset(PRESET_VIO_GNSS, duration_s=0.0)
    assert fm.is_active(FI_VIO_LOSS)
    assert fm.is_active(FI_GNSS_LOSS)
    fm.clear_all()
    assert not fm.active_fault_ids()
    print("  preset + clear_all:        PASS")

    # Event log
    fm.activate(FI_EO_FREEZE, duration_s=0.0, source="operator")
    fm.clear(FI_EO_FREEZE)
    log = fm.event_log()
    assert len(log) >= 2
    assert log[-2].action == "activated"
    assert log[-1].action == "cleared"
    print("  event log:                 PASS")

    # Thread safety: activate from two threads simultaneously
    import threading as _th
    errors = []
    def _worker(fid):
        try:
            fm2 = get_fault_manager()
            fm2.activate(fid, duration_s=0.1, source="thread")
            _time.sleep(0.01)
            fm2.clear(fid)
        except Exception as e:
            errors.append(e)

    threads = [_th.Thread(target=_worker, args=(FI_CPU_LOAD,)) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors, f"Thread errors: {errors}"
    print("  thread safety (8 threads): PASS")

    # Singleton
    fm_a = get_fault_manager()
    fm_b = get_fault_manager()
    assert fm_a is fm_b
    print("  singleton:                 PASS")

    print()
    print("All checks passed.")
