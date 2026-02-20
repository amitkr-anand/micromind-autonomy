"""
core/clock/sim_clock.py
MicroMind / NanoCorteX — Simulation Clock
Sprint S1 Deliverable 4 of 4

Monotonic timestep manager for all simulation modules.
All log entries, state transitions, and guard evaluations
reference SimClock.now() to guarantee consistent, ordered
timestamps across every module in the stack.

Design constraints:
  - Monotonic: time never goes backward
  - Deterministic: same dt sequence → same timestamps
  - No wall-clock dependency: runs identically on any machine
  - Thread-safe step() for future parallel module use

References: NFR-002 (≤2 s transition), NFR-013 (log completeness ≥99%)
"""

import threading
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ClockEvent:
    """Record of a clock tick — used for post-run audit."""
    tick:      int
    time_s:    float
    dt_s:      float
    label:     Optional[str] = None


class SimClock:
    """
    Monotonic simulation clock.

    Usage:
        clock = SimClock(dt=0.01)          # 100 Hz
        clock.start()
        t = clock.now()                    # seconds since mission start
        clock.step()                       # advance one tick
        clock.step(label="GNSS_DENIED")    # labelled tick (logged)

    The clock does NOT auto-advance — each simulation loop calls step()
    explicitly, giving deterministic, reproducible runs.
    """

    def __init__(self, dt: float = 0.01, start_time: float = 0.0):
        """
        Args:
            dt:         Simulation timestep in seconds (default 10 ms / 100 Hz).
            start_time: Mission T=0 offset in seconds (default 0.0).
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        self._dt:         float        = dt
        self._start_time: float        = start_time
        self._current:    float        = start_time
        self._tick:       int          = 0
        self._lock:       threading.Lock = threading.Lock()
        self._history:    List[ClockEvent] = []
        self._running:    bool          = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Mark clock as running. Records T=0 event."""
        with self._lock:
            self._running = True
            self._history.append(ClockEvent(
                tick=0, time_s=self._current, dt_s=0.0, label="CLOCK_START"
            ))

    def reset(self) -> None:
        """Reset to initial state. Clears history."""
        with self._lock:
            self._current = self._start_time
            self._tick    = 0
            self._running = False
            self._history.clear()

    # ------------------------------------------------------------------
    # Advance
    # ------------------------------------------------------------------

    def step(self, label: Optional[str] = None) -> float:
        """
        Advance clock by one dt.

        Args:
            label: Optional event label recorded in clock history.
                   Use for significant simulation events (state transitions,
                   jammer activations, etc.) to aid post-run audit.

        Returns:
            New current time in seconds.
        """
        with self._lock:
            if not self._running:
                raise RuntimeError("SimClock.step() called before start()")
            self._current += self._dt
            self._tick    += 1
            if label is not None:
                self._history.append(ClockEvent(
                    tick=self._tick,
                    time_s=self._current,
                    dt_s=self._dt,
                    label=label
                ))
            return self._current

    def step_to(self, target_time: float, label: Optional[str] = None) -> int:
        """
        Advance clock until now() >= target_time.
        Useful for jumping to scenario event timestamps.

        Returns:
            Number of steps taken.
        """
        steps = 0
        while self._current < target_time:
            self.step(label=label if steps == 0 else None)
            steps += 1
        return steps

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def now(self) -> float:
        """Current simulation time in seconds (thread-safe)."""
        with self._lock:
            return self._current

    def tick(self) -> int:
        """Current tick count (0-indexed from start)."""
        with self._lock:
            return self._tick

    def dt(self) -> float:
        """Simulation timestep in seconds."""
        return self._dt

    def elapsed(self) -> float:
        """Time elapsed since mission T=0."""
        with self._lock:
            return self._current - self._start_time

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    # ------------------------------------------------------------------
    # History / audit
    # ------------------------------------------------------------------

    def history(self) -> List[ClockEvent]:
        """Return a snapshot of labelled clock events for audit."""
        with self._lock:
            return list(self._history)

    def labelled_events(self) -> List[ClockEvent]:
        """Return only ticks that carry an event label."""
        with self._lock:
            return [e for e in self._history if e.label is not None]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def time_to_tick(self, time_s: float) -> int:
        """Convert a time in seconds to the nearest tick index."""
        return round((time_s - self._start_time) / self._dt)

    def tick_to_time(self, tick: int) -> float:
        """Convert a tick index to simulation time in seconds."""
        return self._start_time + tick * self._dt

    def __repr__(self) -> str:
        return (
            f"SimClock(t={self._current:.3f}s, "
            f"tick={self._tick}, dt={self._dt}s, "
            f"running={self._running})"
        )
