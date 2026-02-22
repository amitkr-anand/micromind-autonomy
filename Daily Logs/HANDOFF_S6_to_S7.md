# MicroMind — Sprint Handoff: S6 → S7
**Generated:** 22 February 2026
**Outgoing sprint:** S6 — CEMS + ZPI Multi-UAV
**Incoming sprint:** S7 — TBD (post-TASL)
**Author:** Amit (amitkr-anand)

---

## 1. Outgoing Sprint: What Was Completed

### Modules delivered
| File | FR | Gate result |
|---|---|---|
| `core/zpi/zpi.py` | FR-104 | 16/16 ZPI tests pass |
| `core/cems/cems.py` | FR-102 | 20/20 CEMS tests pass |
| `sim/bcmp1_cems_sim.py` | FR-102 / FR-104 | 7/7 CEMS criteria |
| `tests/test_s6_zpi_cems.py` | — | 36/36 total |

### Test suite
- **36/36 tests pass** — runtime 0.01s
- Runner: `python tests/test_s6_zpi_cems.py` from repo root
- S5 regression: **111/111 PASS** — nothing broken

### Commit
```
a7633ab  Sprint S6 COMPLETE: ZPI + CEMS multi-UAV — 36/36 tests, 7/7 CEMS criteria
```
Branch: merged to `main` via fast-forward

### Repo state at handoff
All S0–S6 modules on `main`. No open bugs. No deferred items within S6 scope.

---

## 2. Live Interfaces Handed to S7

### ZPI Burst Scheduler
```python
from core.zpi.zpi import make_zpi_scheduler, BurstType, ZPIState

zpi = make_zpi_scheduler(mission_key_hex, centre_freq_hz=433_000_000.0)

# Assess DF risk
risk = zpi.assess_df_risk(jammer_bearing_deg, own_track_deg)  # bool

# Update state
zpi.update_state(df_risk=risk, shm_active=False)
# zpi.state → ZPIState.NOMINAL | DF_ADAPTED | SUPPRESSED

# Transmit a burst
burst = zpi.transmit_burst(BurstType.CEMS_PEER, payload_bytes=144, sim_time_s=t)
# burst is None if SUPPRESSED; burst.confirmed=True if transmitted

# Mandatory pre-terminal burst
burst = zpi.transmit_burst(BurstType.PRE_TERMINAL, 64, sim_time_s=t, pre_terminal=True)
zpi.pre_terminal_confirmed  # bool

# Run a mission segment
result = zpi.run_mission_segment(duration_s, burst_schedule, ...)
# result.duty_cycle, result.zpi_compliant, result.burst_log
```

### CEMS Engine
```python
from core.cems.cems import CEMSEngine, EWObservation, CEMSPacket

engine = CEMSEngine(uav_id="UAV-A", mission_key=bytes.fromhex(key_hex))

# Build and sign a packet for transmission
pkt = engine.build_packet(obs, sim_time_s=t, mission_key=key_bytes)
# pkt.byte_size ≤ 256, pkt.version == 1, pkt.hmac_digest (32 bytes)

# Receive and validate a peer packet
obs = engine.receive_packet(pkt, local_time_s=t)
# obs is None if rejected (replay / low confidence / own / bad version / duplicate)

# Run merge cycle
result = engine.run_merge_cycle(local_obs_list, peer_obs_list, current_time_s=t)
# result.jammer_nodes     → list[MergedJammerNode]
# result.packets_received → int
# result.packets_rejected → int
# result.peers_active     → int
# result.cems_compliant   → bool
```

### MergedJammerNode
```python
# node.node_id          → str (e.g. "JN-000")
# node.position_m       → (x, y) centroid in metres
# node.confidence       → float 0–1 (max of contributing observations)
# node.observation_count → int
# node.source_uav_ids   → list[str] — which UAVs contributed
# node.last_updated_s   → float
```

### BCMP-1 CEMS Simulation
```python
from sim.bcmp1_cems_sim import run_bcmp1_cems

result = run_bcmp1_cems(seed=42)
# result.passed              → bool (all 7 criteria)
# result.criteria            → dict {"CEMS-01": bool, ..., "CEMS-07": bool}
# result.peak_max_sources    → int (max source UAVs in any merged node)
# result.replay_rejections   → int
# result.replans             → list[RouteReplan]
# result.event_log           → list[str]
```

---

## 3. Boundary Constants (S6 — Do Not Change Without Spec Update)

| Constant | Value | Module | FR |
|---|---|---|---|
| ZPI burst duration | ≤ 10 ms | zpi.py | FR-104 |
| ZPI inter-burst interval | 2–30 s randomised | zpi.py | FR-104 |
| ZPI frequency hop range | ± 5 MHz (±10 MHz adapted) | zpi.py | FR-104 |
| ZPI max duty cycle | ≤ 0.5% | zpi.py | FR-104 |
| ZPI power range | −10 to 0 dB | zpi.py | FR-104 |
| ZPI DF risk trigger | jammer bearing within 45° of track | zpi.py | FR-104 |
| ZPI hop plan seed | HKDF-SHA256 from mission key | zpi.py | FR-104 |
| CEMS spatial merge radius | 200 m | cems.py | FR-102 |
| CEMS temporal merge window | 15 s | cems.py | FR-102 |
| CEMS temporal decay rate | 0.1/s after window | cems.py | FR-102 |
| CEMS min peer confidence | ≥ 0.5 | cems.py | FR-102 |
| CEMS replay attack window | 30 s | cems.py | FR-102 |
| CEMS max packet size | 256 bytes | cems.py | FR-102 |
| CEMS packet schema version | 1 (version byte mandatory) | cems.py | FR-102 |
| CEMS merge rate | ≥ 1 Hz when peers active | cems.py | FR-102 |

---

## 4. Decisions Made in S6 (Carry Forward)

| Decision | Detail |
|---|---|
| UAV formation offset | 150 m — within 200 m merge radius so same-jammer observations merge |
| Merge rate compliance threshold | 2 s — flags genuine stalls, not sim cadences |
| Peak node tracking | Mid-mission snapshot used for CEMS-03/05 — nodes expire by mission end (15s temporal window) |
| ZPI hop plan shared key | Two UAVs with same mission key produce identical hop plans — implicit time sync |
| Pre-terminal burst | Sent once only, T-30s before SHM activation, BurstType.PRE_TERMINAL |
| CEMS packet auth | HMAC-SHA256 over packet_id + timestamp + obs_id using mission key |
| run_s5_tests.py | Unchanged — still in repo root, S6 tests run separately |

---

## 5. Incoming Sprint: S7 Scope (TBD)

**Status:** NOT STARTED — scope pending TASL meeting outcome

Candidate modules (from Part Two V7, not yet scheduled):
| File | FR | Description |
|---|---|---|
| `core/cybersec/` | FR-109–112 | Key loading, envelope signature verification, PQC-ready |
| CNN DMRL upgrade | FR-103 | Replace rule-based DMRL stub with trained CNN (requires GPU) |
| ROS2 node wrappers | — | HIL phase — wrap SIL modules as ROS2 nodes |

**Key dependency:** TASL meeting outcome determines whether S7 is cybersecurity hardening, HIL integration, or further SIL expansion.

---

## 6. Session Start Checklist for S7

```bash
git checkout main
git pull origin main
git log --oneline main | head -7

# Verify S6 modules present
ls core/zpi/ core/cems/ sim/bcmp1_cems_sim.py

# Run S6 tests
python tests/test_s6_zpi_cems.py

# Run S5 regression
python run_s5_tests.py

# Expected: 36/36 + 111/111 before starting any S7 work
```

---

## 7. End of Sprint Reminder (for whoever closes S7)

At the end of Sprint S7, generate a new handoff file and:
1. Save as `Daily Logs/HANDOFF_S7_to_S8.md`
2. Commit and push to `main`
3. Upload to Claude Project knowledge
4. Update `SPRINT_STATUS.md` and re-upload

---

# TEMPLATE — Sprint Handoff (copy this for every sprint close)

```markdown
# MicroMind — Sprint Handoff: Sx → Sy
**Generated:** [DATE]
**Outgoing sprint:** Sx — [NAME]
**Incoming sprint:** Sy — [NAME]
**Author:** Amit (amitkr-anand)

## 1. Outgoing Sprint: What Was Completed
### Modules delivered
| File | FR | Gate result |
|---|---|---|

### Test suite
- **x/x tests pass** — runtime xs
- Runner: `python [runner].py` from [location]

### Commit
[HASH]  [message]
Branch: merged to `main`

### Repo state at handoff
[open issues, deferred items, known limitations]

## 2. Live Interfaces Handed to Sy
[document every interface the new sprint depends on]

## 3. Decisions Made in Sx (Carry Forward)
| Decision | Detail |
|---|---|

## 4. Incoming Sprint: Sy Scope
### Modules to build
| File | FR | Description |
|---|---|---|

### Acceptance gate (draft)
- [criterion]

## 5. Session Start Checklist for Sy
git checkout main && git pull origin main
python [previous test runner]   # must be green

## 6. End of Sprint Reminder
Generate HANDOFF_Sy_to_Sz.md → Daily Logs/ → commit → upload to Project knowledge → update SPRINT_STATUS.md
```
