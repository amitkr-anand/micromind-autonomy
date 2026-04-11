# TECHNICAL_NOTES — core/bim
**Module:** BIM (Bayesian Integrity Monitor)
**Status:** FROZEN — do not modify without
explicit Deputy 1 unfreeze authorisation.
**Last updated:** 11 April 2026

---

## Frozen File Status

This module is part of the Navigation Core Box
(§2.3). It implements the mathematical integrity
monitor. It must not be modified without:
  1. Explicit Deputy 1 unfreeze notice
  2. Full SIL re-run after any change
  3. Re-freeze confirmation in commit message

---

## OI-39 Authorised Change (10 April 2026)

**Scope:** One log call added at bim.py:252
**Change:** stdlib _log.warning("GNSS_SPOOF_
  DETECTED: bim_score=%.4f", raw) added
  immediately after spoof = self._detect_spoof()
**Authorised by:** Deputy 1 unfreeze notice
  (session 10 April 2026)
**Commit:** fcf73cf
**Rationale:** EC-07 §16 compliance — GNSS
  Spoofing event must be auditable by log grep.
  BIM correctly sets spoof_alert=True; the log
  call makes it externally observable.

---

## OI-41 — Structured Log Debt

**Status:** OPEN — deferred to DD-01 migration
**Finding:** GNSS_SPOOF_DETECTED uses stdlib
  _log.warning() not structured log_event().
  Missing: timestamp_ms, req_id, severity fields
  required by governance CI logging check
  (Appendix B).
**Root cause:** BIM is a pure-computation module
  with no injected self._log or self._clock
  dependencies. Adding structured logging requires
  a constructor signature change.
**Resolution:** Defer to DD-01 phased migration.
  When BIM is unfrozen for HIL integration,
  inject logger + clock and migrate to log_event().

---

*This document is the change justification record
for a frozen module. No OODA rationale is required
beyond the above.*
