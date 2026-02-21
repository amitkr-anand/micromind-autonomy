"""
core/l10s_se/l10s_se.py
MicroMind Sprint S5 — L10s-SE (Last-10-Second Safety Envelope)
Implements: FR-105, KPI-T03

Boundary conditions (Part Two V7 §1.12):
  - Abort/continue decision within      : ≤ 2 s of L10s-SE activation
  - Temporal window                     : ≤ 10 s from activation to impact or abort
  - Civilian detection threshold        : confidence ≥ 0.7 in any frame → abort
  - Corridor hard enforcement           : any predicted corridor violation → immediate abort
  - EO lock re-acquisition timeout      : 1.5 s max; fail → abort
  - Secure log entry                    : all events logged with timestamp, decision, sensor state

NO ML in this path. Deterministic and auditable only.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("L10s-SE")

# ─── Boundary Constants ───────────────────────────────────────────────────────
DECISION_TIMEOUT_S         = 2.0    # ≤ 2 s from activation to decision
L10S_WINDOW_S              = 10.0   # hard temporal limit
LOCK_CONFIDENCE_THRESHOLD  = 0.85   # must match DMRL threshold
CIVILIAN_DETECT_THRESHOLD  = 0.70   # any frame ≥ this → abort
REACQUISITION_TIMEOUT_S    = 1.5    # from DMRL boundary conditions


# ─── State & Decision Types ───────────────────────────────────────────────────

class L10sDecision(Enum):
    CONTINUE   = "CONTINUE"    # proceed to simulated impact
    ABORT      = "ABORT"       # abort; transition to ST-06


class AbortReason(Enum):
    NONE                 = "NONE"
    NO_LOCK              = "NO_LOCK"
    LOCK_LOST_TIMEOUT    = "LOCK_LOST_TIMEOUT"
    DECOY_DETECTED       = "DECOY_DETECTED"
    CIVILIAN_DETECTED    = "CIVILIAN_DETECTED"
    CORRIDOR_VIOLATION   = "CORRIDOR_VIOLATION"
    DECISION_TIMEOUT     = "DECISION_TIMEOUT"
    L10S_WINDOW_EXPIRED  = "L10S_WINDOW_EXPIRED"


@dataclass
class L10sInputs:
    """
    All sensor state fed into L10s-SE at activation.
    Drawn from DMRL result and mission envelope state.
    """
    # From DMRL
    lock_acquired: bool
    lock_confidence: float
    is_decoy: bool
    decoy_confidence: float
    lock_lost_timeout: bool       # DMRL re-acq timed out

    # From mission envelope / real-time sense
    civilian_confidence: float    # 0–1; ≥0.70 in any frame → abort
    corridor_violation: bool      # True if predicted trajectory exits envelope
    pre_terminal_zpi_complete: bool   # mandatory per DD-02

    # Timing
    activation_timestamp: float   # monotonic time of L10s-SE activation


@dataclass
class L10sLogEntry:
    """Single auditable log entry."""
    timestamp: float
    event: str
    decision: Optional[str] = None
    sensor_state_snapshot: Optional[dict] = None


@dataclass
class L10sResult:
    """Output of L10s-SE decision engine."""
    decision: L10sDecision
    abort_reason: AbortReason
    decision_latency_s: float       # time from activation to decision (must be ≤ 2 s)
    window_compliance: bool         # decision made within L10S_WINDOW_S
    decision_compliance: bool       # decision latency ≤ DECISION_TIMEOUT_S
    l10s_compliant: bool            # both timing requirements satisfied = KPI-T03
    secure_log: list[L10sLogEntry] = field(default_factory=list)


# ─── L10s-SE Decision Engine ──────────────────────────────────────────────────

class L10sSafetyEnvelope:
    """
    Deterministic, auditable Last-10-Second Safety Envelope.
    No machine learning. No stochastic elements.
    All decisions are traceable to a specific input condition.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._log_entries: list[L10sLogEntry] = []

    def _log(self, event: str, decision: Optional[str] = None,
             snapshot: Optional[dict] = None):
        entry = L10sLogEntry(
            timestamp=time.monotonic(),
            event=event,
            decision=decision,
            sensor_state_snapshot=snapshot,
        )
        self._log_entries.append(entry)
        if self.verbose:
            prefix = f"[L10s-SE][{entry.timestamp:.3f}]"
            if decision:
                logger.info(f"{prefix} {event} → {decision}")
            else:
                logger.info(f"{prefix} {event}")

    def _sensor_snapshot(self, inputs: L10sInputs) -> dict:
        return {
            "lock_acquired":       inputs.lock_acquired,
            "lock_confidence":     round(inputs.lock_confidence, 4),
            "is_decoy":            inputs.is_decoy,
            "decoy_confidence":    round(inputs.decoy_confidence, 4),
            "lock_lost_timeout":   inputs.lock_lost_timeout,
            "civilian_confidence": round(inputs.civilian_confidence, 4),
            "corridor_violation":  inputs.corridor_violation,
            "zpi_complete":        inputs.pre_terminal_zpi_complete,
        }

    def _build_result(
        self,
        decision: L10sDecision,
        abort_reason: AbortReason,
        t_activation: float,
    ) -> L10sResult:
        latency = time.monotonic() - t_activation
        window_ok   = latency <= L10S_WINDOW_S
        decision_ok = latency <= DECISION_TIMEOUT_S
        compliant   = window_ok and decision_ok

        self._log(
            f"Decision={decision.value} reason={abort_reason.value} "
            f"latency={latency*1000:.1f}ms compliance={'PASS' if compliant else 'FAIL'}"
        )
        return L10sResult(
            decision=decision,
            abort_reason=abort_reason,
            decision_latency_s=round(latency, 4),
            window_compliance=window_ok,
            decision_compliance=decision_ok,
            l10s_compliant=compliant,
            secure_log=list(self._log_entries),
        )

    def evaluate(self, inputs: L10sInputs) -> L10sResult:
        """
        Deterministic decision tree.
        Authority chain: Layer 1 (Part Two V7 §1.12).
        Each check is an explicit abort gate; CONTINUE requires ALL gates clear.
        """
        self._log_entries.clear()
        t_activation = inputs.activation_timestamp

        snap = self._sensor_snapshot(inputs)
        self._log("L10s-SE ACTIVATED", snapshot=snap)

        # ── Gate 0: Pre-terminal ZPI burst confirmation (DD-02 mandatory) ─────
        if not inputs.pre_terminal_zpi_complete:
            self._log("GATE 0 FAIL — pre-terminal ZPI burst NOT confirmed (DD-02 violation)")
            return self._build_result(L10sDecision.ABORT, AbortReason.NO_LOCK, t_activation)

        self._log("Gate 0 PASS — pre-terminal ZPI burst confirmed")

        # ── Gate 1: EO lock acquired check ────────────────────────────────────
        if inputs.lock_lost_timeout:
            self._log(
                f"GATE 1 FAIL — EO lock re-acquisition timed out "
                f"(> {REACQUISITION_TIMEOUT_S}s)"
            )
            return self._build_result(
                L10sDecision.ABORT, AbortReason.LOCK_LOST_TIMEOUT, t_activation
            )

        if not inputs.lock_acquired:
            self._log(
                f"GATE 1 FAIL — No EO lock (confidence={inputs.lock_confidence:.4f} "
                f"< {LOCK_CONFIDENCE_THRESHOLD})"
            )
            return self._build_result(
                L10sDecision.ABORT, AbortReason.NO_LOCK, t_activation
            )

        if inputs.lock_confidence < LOCK_CONFIDENCE_THRESHOLD:
            self._log(
                f"GATE 1 FAIL — Lock confidence below threshold "
                f"({inputs.lock_confidence:.4f} < {LOCK_CONFIDENCE_THRESHOLD})"
            )
            return self._build_result(
                L10sDecision.ABORT, AbortReason.NO_LOCK, t_activation
            )

        self._log(
            f"Gate 1 PASS — EO lock acquired | confidence={inputs.lock_confidence:.4f}"
        )

        # ── Gate 2: Decoy detection check ─────────────────────────────────────
        if inputs.is_decoy:
            self._log(
                f"GATE 2 FAIL — Decoy detected (confidence={inputs.decoy_confidence:.4f} "
                f"over {3} consecutive frames)"
            )
            return self._build_result(
                L10sDecision.ABORT, AbortReason.DECOY_DETECTED, t_activation
            )

        self._log("Gate 2 PASS — No decoy detected")

        # ── Gate 3: Civilian presence check ───────────────────────────────────
        if inputs.civilian_confidence >= CIVILIAN_DETECT_THRESHOLD:
            self._log(
                f"GATE 3 FAIL — Civilian signature detected "
                f"(confidence={inputs.civilian_confidence:.4f} ≥ {CIVILIAN_DETECT_THRESHOLD})"
            )
            return self._build_result(
                L10sDecision.ABORT, AbortReason.CIVILIAN_DETECTED, t_activation
            )

        self._log(
            f"Gate 3 PASS — Civilian confidence below threshold "
            f"({inputs.civilian_confidence:.4f} < {CIVILIAN_DETECT_THRESHOLD})"
        )

        # ── Gate 4: Corridor hard enforcement ─────────────────────────────────
        if inputs.corridor_violation:
            self._log(
                "GATE 4 FAIL — Predicted trajectory violates mission envelope corridor"
            )
            return self._build_result(
                L10sDecision.ABORT, AbortReason.CORRIDOR_VIOLATION, t_activation
            )

        self._log("Gate 4 PASS — Corridor clear")

        # ── Gate 5: Decision timing check ─────────────────────────────────────
        elapsed = time.monotonic() - t_activation
        if elapsed > DECISION_TIMEOUT_S:
            self._log(
                f"GATE 5 FAIL — Decision latency exceeded "
                f"({elapsed*1000:.1f}ms > {DECISION_TIMEOUT_S*1000:.0f}ms)"
            )
            return self._build_result(
                L10sDecision.ABORT, AbortReason.DECISION_TIMEOUT, t_activation
            )

        # ── ALL GATES CLEAR — CONTINUE ────────────────────────────────────────
        self._log("All gates CLEAR — issuing CONTINUE")
        return self._build_result(
            L10sDecision.CONTINUE, AbortReason.NONE, t_activation
        )


# ─── Helper: build inputs from DMRL result ────────────────────────────────────

def inputs_from_dmrl(
    dmrl_result,                    # DMRLResult from dmrl_stub
    civilian_confidence: float = 0.0,
    corridor_violation: bool = False,
    pre_terminal_zpi_complete: bool = True,
) -> L10sInputs:
    """
    Convenience factory — constructs L10sInputs from a DMRLResult
    plus environment state.
    """
    return L10sInputs(
        lock_acquired=dmrl_result.lock_acquired,
        lock_confidence=dmrl_result.lock_confidence,
        is_decoy=dmrl_result.is_decoy,
        decoy_confidence=dmrl_result.decoy_confidence,
        lock_lost_timeout=dmrl_result.lock_lost_timeout,
        civilian_confidence=civilian_confidence,
        corridor_violation=corridor_violation,
        pre_terminal_zpi_complete=pre_terminal_zpi_complete,
        activation_timestamp=time.monotonic(),
    )


# ─── Standalone self-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    engine = L10sSafetyEnvelope(verbose=True)

    print("=" * 70)
    print("L10s-SE Self-Test — Deterministic Decision Tree")
    print("=" * 70)

    # Test 1: Clean CONTINUE
    print("\n--- Test 1: Valid lock, no decoy, clear corridor ---")
    inp = L10sInputs(
        lock_acquired=True, lock_confidence=0.912, is_decoy=False,
        decoy_confidence=0.12, lock_lost_timeout=False,
        civilian_confidence=0.05, corridor_violation=False,
        pre_terminal_zpi_complete=True, activation_timestamp=time.monotonic()
    )
    r = engine.evaluate(inp)
    print(f"  Decision: {r.decision.value} | compliant={r.l10s_compliant} "
          f"| latency={r.decision_latency_s*1000:.2f}ms")

    # Test 2: Decoy abort
    print("\n--- Test 2: Decoy detected ---")
    inp2 = L10sInputs(
        lock_acquired=True, lock_confidence=0.79, is_decoy=True,
        decoy_confidence=0.88, lock_lost_timeout=False,
        civilian_confidence=0.02, corridor_violation=False,
        pre_terminal_zpi_complete=True, activation_timestamp=time.monotonic()
    )
    r2 = engine.evaluate(inp2)
    print(f"  Decision: {r2.decision.value} | reason={r2.abort_reason.value} "
          f"| compliant={r2.l10s_compliant}")

    # Test 3: Civilian abort
    print("\n--- Test 3: Civilian presence ---")
    inp3 = L10sInputs(
        lock_acquired=True, lock_confidence=0.91, is_decoy=False,
        decoy_confidence=0.10, lock_lost_timeout=False,
        civilian_confidence=0.82, corridor_violation=False,
        pre_terminal_zpi_complete=True, activation_timestamp=time.monotonic()
    )
    r3 = engine.evaluate(inp3)
    print(f"  Decision: {r3.decision.value} | reason={r3.abort_reason.value}")

    # Test 4: Corridor violation
    print("\n--- Test 4: Corridor violation ---")
    inp4 = L10sInputs(
        lock_acquired=True, lock_confidence=0.93, is_decoy=False,
        decoy_confidence=0.08, lock_lost_timeout=False,
        civilian_confidence=0.02, corridor_violation=True,
        pre_terminal_zpi_complete=True, activation_timestamp=time.monotonic()
    )
    r4 = engine.evaluate(inp4)
    print(f"  Decision: {r4.decision.value} | reason={r4.abort_reason.value}")

    print("\nAll L10s-SE self-tests complete.")
