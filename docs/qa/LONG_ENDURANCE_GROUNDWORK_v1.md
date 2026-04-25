# Long-Endurance Groundwork v1.0
Authority: Deputy 1 | Date: 26 April 2026

## ST-END-01 Prerequisites

### Memory
- G-14 memory growth slope: excluded from CI baseline.
  Requires dedicated overnight run. Target: < 50 MB over 2 hours.
- Checkpoint retention: CONFIRMED — max_retained=5, purge tested (c38357a)
- Route fragment accumulation: CONFIRMED clean (SB-07)

### Logging (BLOCKING)
- UT-RS-02 (log rolling): NOT STARTED — GAP-09
  Must be implemented before ST-END-01 or disk exhaustion is uncontrolled
- ST-DISK-01 (disk exhaustion): NOT IMPLEMENTED
  Critical events must survive disk full (1000-event RAM ring buffer per Appendix E)

### Stability
- ProcessWatchdog decision logic: IMPLEMENTED (3fc84bc)
- Real SIGKILL stimulus (ST-RESTART-01): Phase D
- CPU gate (≤70% mean): NOT MEASURED — no CI gate exists

## Pre-ST-END-01 Sequence (Mandatory Order)
1. UT-RS-02 — log rolling implementation and test (SIL)
2. ST-DISK-01 — disk exhaustion simulation (SIL)
3. G-14 — memory growth overnight run on node01
4. 30-minute CPU warm-up run — confirm ≤70%
5. ST-END-01 — 2-hour full endurance run (overnight)

## Estimated Timeline
Items 1-2: 1 session each
Item 3: overnight (unattended)
Item 4: 30 minutes
Item 5: 2 hours overnight

## Status
Phase D. Not blocked. Dependent on UT-RS-02 implementation.
