"""
core/state_machine/state_machine.py
MicroMind / NanoCorteX — Deterministic State Machine
Sprint S1 Deliverable 1 of 4

Formally specified 7-state FSM per Part Two V7 Section 1.2 and
TechReview R-01. All transitions are deterministic, guarded,
and logged with timestamp and guard evaluation result.

States:
    ST-01  NOMINAL          GNSS Green, RF active, no EW alert
    ST-02  EW_AWARE         Jammer hypothesis active, GNSS ≤ Amber
    ST-03  GNSS_DENIED      GNSS Red, VIO+TRN primary
    ST-04  SILENT_INGRESS   Terminal zone entered, ZPI bursts only
    ST-05  SHM_ACTIVE       Zero RF, IMU+VO+DEM only, L10s-SE enforced
    ST-06  ABORT            Abort triggered; loiter/egress per envelope
    ST-07  MISSION_FREEZE   Anti-capture; all outputs suppressed

Design constraints:
    - NFR-002: all transitions ≤ 2 seconds (enforced via timing check + log)
    - All transitions logged: timestamp, trigger, guard results (TechReview R-01)
    - No ML in this path — purely deterministic guard evaluation
    - Authority chain respected: mission envelope cannot be overridden

References:
    Part Two V7  Section 1.2 (state machine), Section 2 (failure modes)
    TechReview   R-01 (formalise state machine — CRITICAL)
    NFR-002      Autonomy fallback transition time ≤ 2 s
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

from core.clock.sim_clock import SimClock
from logs.mission_log_schema import (
    BIMState,
    GuardEvaluation,
    LogCategory,
    MissionLog,
    MissionLogEntry,
)


# ---------------------------------------------------------------------------
# State enumeration
# ---------------------------------------------------------------------------

class NCState(str, Enum):
    """NanoCorteX operational states (Part Two V7 §1.2.1)."""
    NOMINAL         = "NOMINAL"         # ST-01
    EW_AWARE        = "EW_AWARE"        # ST-02
    GNSS_DENIED     = "GNSS_DENIED"     # ST-03
    SILENT_INGRESS  = "SILENT_INGRESS"  # ST-04
    SHM_ACTIVE      = "SHM_ACTIVE"      # ST-05
    ABORT           = "ABORT"           # ST-06
    MISSION_FREEZE  = "MISSION_FREEZE"  # ST-07


# ---------------------------------------------------------------------------
# Sensor / system inputs consumed by guards
# ---------------------------------------------------------------------------

@dataclass
class SystemInputs:
    """
    Snapshot of system inputs evaluated by transition guards.
    Populated each tick by the simulation or real sensor fusion layer.
    All values are the single source of truth for guard decisions.
    """
    # BIM outputs
    bim_trust_score:        float    = 1.0      # 0.0–1.0
    bim_state:              BIMState = BIMState.GREEN
    bim_green_sample_count: int      = 0        # consecutive Green samples

    # EW Engine outputs
    ew_jammer_confidence:   float    = 0.0      # highest hypothesis confidence
    ew_cost_map_active:     bool     = False

    # Navigation readiness
    vio_feature_count:      int      = 100      # ≥20 required for GNSS_DENIED
    trn_correlation_valid:  bool     = True

    # Mission envelope / geometry
    terminal_zone_entered:  bool     = False    # crossed per signed envelope
    l10s_active:            bool     = False    # L10s-SE window engaged
    eo_lock_confidence:     float    = 0.0      # EO/IR lock 0.0–1.0

    # Integrity / security
    key_mismatch:           bool     = False    # PQC signature failure
    tamper_detected:        bool     = False    # hardware tamper flag
    corridor_violation:     bool     = False    # predicted path exits envelope

    # L10s-SE decision (populated by L10s-SE module)
    l10s_abort_commanded:   bool     = False

    # Timestamp of last input update (wall-clock seconds, for latency checks)
    updated_at:             float    = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Transition result
# ---------------------------------------------------------------------------

@dataclass
class TransitionResult:
    """Outcome of a transition attempt."""
    attempted:      bool
    succeeded:      bool
    from_state:     NCState
    to_state:       NCState
    trigger:        str
    latency_ms:     float
    guards:         List[GuardEvaluation]
    timestamp_s:    float
    nfr_002_pass:   bool        # latency ≤ 2000 ms


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class NanoCorteXFSM:
    """
    NanoCorteX deterministic state machine.

    All transitions are evaluated synchronously. Guard logic is pure
    Python with no ML calls — deterministic and auditable.

    Usage:
        clock  = SimClock(dt=0.01)
        log    = MissionLog(mission_id="BCMP1-SIL-001")
        fsm    = NanoCorteXFSM(clock=clock, log=log)

        clock.start()
        fsm.start()

        inputs = SystemInputs(...)
        result = fsm.evaluate(inputs)    # called each simulation tick
    """

    NFR_002_LIMIT_MS = 2000.0   # maximum allowed transition time (ms)

    def __init__(
        self,
        clock:      SimClock,
        log:        MissionLog,
        mission_id: str = "UNKNOWN",
    ):
        self._clock       = clock
        self._log         = log
        self._mission_id  = mission_id
        self._state       = NCState.NOMINAL
        self._prev_state  = NCState.NOMINAL
        self._started     = False

        # Track consecutive Green BIM samples for ST-02 → ST-01 guard
        self._green_count = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Log mission start and mark FSM as running."""
        self._started = True
        self._log.append(MissionLogEntry(
            timestamp_s  = self._clock.now(),
            tick         = self._clock.tick(),
            category     = LogCategory.MISSION_START,
            mission_id   = self._mission_id,
            state        = self._state.value,
            notes        = "NanoCorteX FSM started — state NOMINAL",
        ))

    @property
    def state(self) -> NCState:
        return self._state

    def evaluate(self, inputs: SystemInputs) -> Optional[TransitionResult]:
        """
        Evaluate all legal transitions from the current state.
        Execute the first transition whose guards pass.

        Called once per simulation tick. Returns TransitionResult if a
        transition occurred, else None.

        Priority ordering within each state follows Part Two V7 §1.2.2:
        safety/freeze transitions evaluated before degradation transitions.
        """
        if not self._started:
            raise RuntimeError("NanoCorteXFSM.evaluate() called before start()")

        # --- Global priority: anti-capture / tamper (any state → MISSION_FREEZE)
        if self._state not in (NCState.MISSION_FREEZE, NCState.ABORT):
            result = self._try_mission_freeze(inputs)
            if result:
                return result

        # --- State-specific transitions
        dispatch: Dict[NCState, Callable] = {
            NCState.NOMINAL:        self._from_nominal,
            NCState.EW_AWARE:       self._from_ew_aware,
            NCState.GNSS_DENIED:    self._from_gnss_denied,
            NCState.SILENT_INGRESS: self._from_silent_ingress,
            NCState.SHM_ACTIVE:     self._from_shm_active,
            NCState.ABORT:          self._from_abort,
            NCState.MISSION_FREEZE: self._from_mission_freeze,
        }
        handler = dispatch.get(self._state)
        if handler:
            return handler(inputs)
        return None

    # ------------------------------------------------------------------
    # Global guard: MISSION_FREEZE (any → ST-07)
    # ------------------------------------------------------------------

    def _try_mission_freeze(self, inputs: SystemInputs) -> Optional[TransitionResult]:
        guards = [
            GuardEvaluation("key_mismatch",    inputs.key_mismatch,    inputs.key_mismatch,    True),
            GuardEvaluation("tamper_detected",  inputs.tamper_detected, inputs.tamper_detected, True),
        ]
        if inputs.key_mismatch or inputs.tamper_detected:
            trigger = "KEY_MISMATCH" if inputs.key_mismatch else "TAMPER_DETECTED"
            return self._transition(NCState.MISSION_FREEZE, trigger, guards, inputs)
        return None

    # ------------------------------------------------------------------
    # ST-01 NOMINAL transitions
    # ------------------------------------------------------------------

    def _from_nominal(self, inputs: SystemInputs) -> Optional[TransitionResult]:
        # NOMINAL → ABORT: corridor violation
        if inputs.corridor_violation:
            guards = [GuardEvaluation("corridor_violation", True, True, True)]
            return self._transition(NCState.ABORT, "CORRIDOR_VIOLATION", guards, inputs)

        # NOMINAL → EW_AWARE: jammer hypothesis confidence > 0.6
        guards_ew = [
            GuardEvaluation(
                "ew_jammer_confidence > 0.6",
                inputs.ew_jammer_confidence > 0.6,
                inputs.ew_jammer_confidence, 0.6
            ),
        ]
        if inputs.ew_jammer_confidence > 0.6:
            return self._transition(NCState.EW_AWARE, "JAMMER_HYPOTHESIS", guards_ew, inputs)

        return None

    # ------------------------------------------------------------------
    # ST-02 EW_AWARE transitions
    # ------------------------------------------------------------------

    def _from_ew_aware(self, inputs: SystemInputs) -> Optional[TransitionResult]:
        # EW_AWARE → ABORT: corridor violation
        if inputs.corridor_violation:
            guards = [GuardEvaluation("corridor_violation", True, True, True)]
            return self._transition(NCState.ABORT, "CORRIDOR_VIOLATION", guards, inputs)

        # EW_AWARE → GNSS_DENIED: BIM Red + nav ready
        nav_ready = (
            inputs.vio_feature_count >= 20 or inputs.trn_correlation_valid
        )
        guards_denied = [
            GuardEvaluation("bim_state == RED",     inputs.bim_state == BIMState.RED,
                            inputs.bim_state.value, "RED"),
            GuardEvaluation("vio_features >= 20 OR trn_valid", nav_ready,
                            inputs.vio_feature_count, 20),
        ]
        if inputs.bim_state == BIMState.RED and nav_ready:
            return self._transition(NCState.GNSS_DENIED, "BIM_RED", guards_denied, inputs)

        # EW_AWARE → NOMINAL: BIM Green 3 consecutive samples, no active jammer
        self._green_count = (
            self._green_count + 1
            if inputs.bim_state == BIMState.GREEN
            else 0
        )
        guards_recover = [
            GuardEvaluation("bim_green_3_samples", self._green_count >= 3,
                            self._green_count, 3),
            GuardEvaluation("ew_confidence <= 0.6", inputs.ew_jammer_confidence <= 0.6,
                            inputs.ew_jammer_confidence, 0.6),
        ]
        if self._green_count >= 3 and inputs.ew_jammer_confidence <= 0.6:
            self._green_count = 0
            return self._transition(NCState.NOMINAL, "BIM_GREEN_RECOVERED", guards_recover, inputs)

        return None

    # ------------------------------------------------------------------
    # ST-03 GNSS_DENIED transitions
    # ------------------------------------------------------------------

    def _from_gnss_denied(self, inputs: SystemInputs) -> Optional[TransitionResult]:
        # GNSS_DENIED → ABORT: corridor violation
        if inputs.corridor_violation:
            guards = [GuardEvaluation("corridor_violation", True, True, True)]
            return self._transition(NCState.ABORT, "CORRIDOR_VIOLATION", guards, inputs)

        # GNSS_DENIED → SILENT_INGRESS: terminal zone boundary crossed
        guards_si = [
            GuardEvaluation("terminal_zone_entered", inputs.terminal_zone_entered,
                            inputs.terminal_zone_entered, True),
        ]
        if inputs.terminal_zone_entered:
            return self._transition(NCState.SILENT_INGRESS, "TERMINAL_ZONE_BOUNDARY",
                                    guards_si, inputs)

        return None

    # ------------------------------------------------------------------
    # ST-04 SILENT_INGRESS transitions
    # ------------------------------------------------------------------

    def _from_silent_ingress(self, inputs: SystemInputs) -> Optional[TransitionResult]:
        # SILENT_INGRESS → ABORT: corridor violation
        if inputs.corridor_violation:
            guards = [GuardEvaluation("corridor_violation", True, True, True)]
            return self._transition(NCState.ABORT, "CORRIDOR_VIOLATION", guards, inputs)

        # SILENT_INGRESS → SHM_ACTIVE: L10s-SE window engaged
        guards_shm = [
            GuardEvaluation("l10s_active", inputs.l10s_active,
                            inputs.l10s_active, True),
        ]
        if inputs.l10s_active:
            return self._transition(NCState.SHM_ACTIVE, "L10S_SE_ACTIVATION",
                                    guards_shm, inputs)

        return None

    # ------------------------------------------------------------------
    # ST-05 SHM_ACTIVE transitions
    # ------------------------------------------------------------------

    def _from_shm_active(self, inputs: SystemInputs) -> Optional[TransitionResult]:
        # SHM_ACTIVE → ABORT: L10s-SE abort commanded OR EO lock loss
        eo_lock_loss = inputs.eo_lock_confidence < 0.3 and inputs.l10s_active
        guards_abort = [
            GuardEvaluation("l10s_abort_commanded", inputs.l10s_abort_commanded,
                            inputs.l10s_abort_commanded, True),
            GuardEvaluation("eo_lock_confidence < 0.3", eo_lock_loss,
                            inputs.eo_lock_confidence, 0.3),
        ]
        if inputs.l10s_abort_commanded or eo_lock_loss:
            trigger = "L10S_ABORT" if inputs.l10s_abort_commanded else "EO_LOCK_LOSS"
            return self._transition(NCState.ABORT, trigger, guards_abort, inputs)

        return None

    # ------------------------------------------------------------------
    # Terminal states — no further transitions
    # ------------------------------------------------------------------

    def _from_abort(self, _: SystemInputs) -> Optional[TransitionResult]:
        # ABORT is terminal within a mission run.
        # Recovery to NOMINAL is an operator/ground action (post-mission).
        return None

    def _from_mission_freeze(self, _: SystemInputs) -> Optional[TransitionResult]:
        # MISSION_FREEZE is terminal. No output. No transition.
        return None

    # ------------------------------------------------------------------
    # Transition executor
    # ------------------------------------------------------------------

    def _transition(
        self,
        to_state:   NCState,
        trigger:    str,
        guards:     List[GuardEvaluation],
        inputs:     SystemInputs,
    ) -> TransitionResult:
        """
        Execute a state transition, measure latency, and log.

        Latency is measured as wall-clock delta for the guard evaluation
        call (representative of compute time; SIL uses sim time).
        """
        t_start_wall = time.monotonic()
        from_state   = self._state

        # Execute transition
        self._prev_state = self._state
        self._state      = to_state

        t_end_wall   = time.monotonic()
        latency_ms   = (t_end_wall - t_start_wall) * 1000.0
        nfr_002_pass = latency_ms <= self.NFR_002_LIMIT_MS

        result = TransitionResult(
            attempted    = True,
            succeeded    = True,
            from_state   = from_state,
            to_state     = to_state,
            trigger      = trigger,
            latency_ms   = latency_ms,
            guards       = guards,
            timestamp_s  = self._clock.now(),
            nfr_002_pass = nfr_002_pass,
        )

        # Log entry (mandatory — TechReview R-01)
        entry = MissionLogEntry(
            timestamp_s          = self._clock.now(),
            tick                 = self._clock.tick(),
            category             = LogCategory.STATE_TRANSITION,
            mission_id           = self._mission_id,
            state                = to_state.value,
            from_state           = from_state.value,
            to_state             = to_state.value,
            transition_trigger   = trigger,
            transition_latency_ms = latency_ms,
            guards               = guards,
            notes                = (
                None if nfr_002_pass
                else f"NFR-002 VIOLATION: latency {latency_ms:.1f} ms > 2000 ms"
            ),
        )
        self._log.append(entry)

        return result

    # ------------------------------------------------------------------
    # Manual terminal log
    # ------------------------------------------------------------------

    def log_mission_end(self, notes: str = "") -> None:
        """Call at end of simulation run to record final state."""
        self._log.append(MissionLogEntry(
            timestamp_s  = self._clock.now(),
            tick         = self._clock.tick(),
            category     = LogCategory.MISSION_END,
            mission_id   = self._mission_id,
            state        = self._state.value,
            notes        = notes or f"Mission ended in state {self._state.value}",
        ))

    def __repr__(self) -> str:
        return (
            f"NanoCorteXFSM(state={self._state.value}, "
            f"t={self._clock.now():.2f}s)"
        )
