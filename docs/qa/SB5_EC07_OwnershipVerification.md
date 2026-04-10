# SB-5 EC-07 Recovery Ownership Verification
**Date:** 10 April 2026  
**SRS ref:** §16 Recovery Ownership Matrix  
**Commit (HEAD):** `545de66`  
**Verified by:** Agent 2 (Deputy 1 branch)  
**Governance ref:** Code Governance Manual v3.2 §2.5

---

## §16 Source Extract

Text extracted from `docs/qa/MicroMind_SRS_v1_3.docx` §16 (added in v1.2, correction O-01).

> "This matrix governs implementation. Any module handling an event not designated as its
> owner constitutes an ownership violation and must be corrected."

Full §16 rows relevant to the six events under verification:

| §16 Event Name | Detects | Decides (owns recovery) | Executes | Logs/Consumes |
|---|---|---|---|---|
| GNSS spoof detected | Navigation Manager (BIM) | Navigation Manager | Navigation Manager (GNSS gate, TRN-primary transition) | Mission Manager (trust state consumer), Demo Tool |
| VIO outage | Navigation Manager (VIOMode) | Navigation Manager | Navigation Manager (weight rebalance, mode demotion) | Mission Manager (USV consumer) |
| PX4 reboot | PX4 Bridge (HEARTBEAT seq reset) | PX4 Bridge (D7–D9); Mission Manager (D8a gate) | PX4 Bridge (reconnect + OFFBOARD re-entry) | Mission Manager (SHM during reboot, D8a evaluation), Navigation Manager |
| *(no row)* | — | — | — | — |
| SHM entry | Mission Manager (trigger detection) | Mission Manager | Mission Manager (loiter command to PX4) | All modules (log only) |
| Terminal phase failure (DMRL/L10s) | DMRL / L10s-SE | Mission Manager (gate decision owns abort/proceed) | Mission Manager (ABORT_TERM or SHM) | Navigation Manager, PX4 Bridge (hold setpoint) |

**Note:** §16 has no dedicated row for "Corridor Violation (predicted)". See OI-40.

---

## Verification Table

| Event | §16 Owner (Detects) | Module emitting log | Log event string | File : Line | Compliant | Notes |
|---|---|---|---|---|---|---|
| GNSS Spoofing | Navigation Manager (BIM) | `core/bim/bim.py` | **NOT FOUND** | `bim.py:288` (`spoof_alert=True` in BIMResult only) | **N** | BIM is correct module per §16, but no named log event string (e.g. `GNSS_SPOOF_DETECTED`) is emitted. `spoof_alert` is a struct field, not a logged event. See OI-39. |
| VIO Degradation | Navigation Manager (VIOMode) | `core/fusion/vio_mode.py` | `VIO_OUTAGE_DETECTED` | `vio_mode.py:166` | **Y** | Correct module (VIOMode = Navigation Manager). Logged via stdlib `_log.warning()`; not structured MissionLog but module ownership is correct. |
| PX4 Reboot | PX4 Bridge (HEARTBEAT seq reset) | `integration/bridge/reboot_detector.py` | `PX4_REBOOT_DETECTED` | `reboot_detector.py:151` | **Y** | Correct module (RebootDetector instantiated by MAVLinkBridge = PX4 Bridge). Also forwarded to BridgeLogger at `mavlink_bridge.py:435`. |
| Corridor Violation (predicted) | *(not in §16)* | `core/state_machine/state_machine.py` | `CORRIDOR_VIOLATION` | `state_machine.py:240,263,304,325` | **N** | §16 has **no recovery ownership row** for this event. NanoCorteXFSM emits `CORRIDOR_VIOLATION` → ABORT from multiple states. Ownership unspecified in §16. See OI-40. |
| SHM Trigger | Mission Manager (trigger detection) | `core/state_machine/state_machine.py` | `L10S_SE_ACTIVATION` (STATE_TRANSITION trigger) | `state_machine.py:333` | **Y** | NanoCorteXFSM = Mission Manager component per §16 terminology. STATE_TRANSITION log entry carries `to_state="SHM_ACTIVE"`, `trigger="L10S_SE_ACTIVATION"`. No standalone `SHM_ACTIVATED` event; ownership is correct. |
| Target Lock Loss | DMRL / L10s-SE (detects) → Mission Manager (decides) | `core/state_machine/state_machine.py` (decides) + `core/l10s_se/l10s_se.py` (detects) | `EO_LOCK_LOSS` / `LOCK_LOST_TIMEOUT` | `state_machine.py:352` / `l10s_se.py:188` | **Y** | l10s_se provides detection (`l10s_abort_commanded`, `LOCK_LOST_TIMEOUT`). NanoCorteXFSM (Mission Manager) makes abort decision. Matches §16: DMRL/L10s-SE detects, Mission Manager decides. |

---

## Summary

| Compliant | Count | Events |
|---|---|---|
| **Y** | 4 | VIO Degradation, PX4 Reboot, SHM Trigger, Target Lock Loss |
| **N** | 2 | GNSS Spoofing (no log event), Corridor Violation (not in §16) |

---

## Non-Compliances → OIs Raised

### OI-39: EC-07 — GNSS Spoof event has no dedicated named log event string

**Event:** GNSS Spoofing  
**§16 requirement:** Navigation Manager (BIM) detects and logs GNSS spoof event.  
**Finding:** `core/bim/bim.py` `_detect_spoof()` correctly identifies spoofing and
sets `spoof_alert=True` in the `BIMResult` dataclass (line 288). However, no named
log event string (e.g. `GNSS_SPOOF_DETECTED`) is emitted by any `log()` call.
The `spoof_alert` field is carried in the BIMResult data structure passed upstream
but is not logged as a discrete auditable event.  
**Impact:** §16 requires the detection event to be consumable by Mission Manager
(trust state consumer) and Demo Tool. Without a named event string, neither module
can subscribe to or audit the spoof detection.  
**Priority:** MEDIUM (correct module, missing log call — not a §1.3 forbidden
behaviour violation since BIM is the designated owner).  
**Fix required before:** Phase A exit gate.

### OI-40: EC-07 — Corridor Violation (predicted) absent from §16 Recovery Ownership Matrix

**Event:** Corridor Violation (predicted)  
**§16 requirement:** No row exists for this event.  
**Finding:** `core/state_machine/state_machine.py` emits `"CORRIDOR_VIOLATION"`
transition trigger → ABORT from states NOMINAL (line 240), EW_AWARE (line 263),
GNSS_DENIED (line 304), and SILENT_INGRESS (line 325). `core/l10s_se/l10s_se.py`
also defines `AbortReason.CORRIDOR_VIOLATION` (line 48) used in terminal phase.  
**Impact:** No §16 ownership row means: (a) there is no authoritative definition
of which module should detect a predicted corridor violation, which module decides
recovery action, and which module executes it; (b) the current implementation
(NanoCorteXFSM directly → ABORT) has not been reviewed against §16 ownership rules.  
**Priority:** MEDIUM (§16 documentation gap — the code handles it consistently,
but the ownership is not formally specified).  
**Fix required before:** SRS v1.4 revision. OI raised here to track.

---

## Grep Evidence

### GNSS Spoofing
```
core/bim/bim.py:173:    spoof_alert:        bool  = False
core/bim/bim.py:288:            spoof_alert         = spoof,
core/bim/bim.py:385:    def _detect_spoof(self, m: GNSSMeasurement) -> bool:
core/bim/bim.py:426:        if spoof_alert:   ← state forced to RED, no log event emitted
```
No hit for: `GNSS_SPOOF_DETECTED`, `BIM_SPOOF`, `SPOOF_DETECTED` in any `.py` file.

### VIO Degradation
```
core/fusion/vio_mode.py:166: _log.warning("VIO_OUTAGE_DETECTED: dt_since_vio=%.3fs ...")
core/fusion/vio_mode.py:214: _log.info("VIO_RESUMPTION_STARTED: outage_events=%d", ...)
```

### PX4 Reboot
```
integration/bridge/reboot_detector.py:151: "event": "PX4_REBOOT_DETECTED"
integration/bridge/mavlink_bridge.py:435:  self._logger.log("PX4_REBOOT_DETECTED", evt)
```

### Corridor Violation (predicted)
```
core/state_machine/state_machine.py:240: "CORRIDOR_VIOLATION" → ABORT (from NOMINAL)
core/state_machine/state_machine.py:263: "CORRIDOR_VIOLATION" → ABORT (from EW_AWARE)
core/state_machine/state_machine.py:304: "CORRIDOR_VIOLATION" → ABORT (from GNSS_DENIED)
core/state_machine/state_machine.py:325: "CORRIDOR_VIOLATION" → ABORT (from SILENT_INGRESS)
core/l10s_se/l10s_se.py:48:  AbortReason.CORRIDOR_VIOLATION
```
§16 has no row for this event.

### SHM Trigger
```
core/state_machine/state_machine.py:333: _transition(NCState.SHM_ACTIVE, "L10S_SE_ACTIVATION", ...)
```
No standalone `SHM_ACTIVATED` event string found.

### Target Lock Loss
```
core/state_machine/state_machine.py:352: trigger = "EO_LOCK_LOSS"
core/l10s_se/l10s_se.py:45:  LOCK_LOST_TIMEOUT = "LOCK_LOST_TIMEOUT"
core/l10s_se/l10s_se.py:188: L10sDecision.ABORT, AbortReason.LOCK_LOST_TIMEOUT
```
