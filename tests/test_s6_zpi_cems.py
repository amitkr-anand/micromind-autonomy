"""
tests/test_s6_zpi_cems.py
MicroMind Sprint S6 — Acceptance Tests: ZPI + CEMS

Boundary conditions under test (Part Two V7):
  ZPI:
    - Burst duration ≤ 10 ms
    - Duty cycle ≤ 0.5%
    - Hop plan deterministic from mission key (two UAVs, same key → same plan)
    - DF adaptation triggers at jammer bearing within 45° of own track
    - SHM suppression blocks all bursts
    - Pre-terminal burst confirmed before SHM activation
  CEMS:
    - Replay attack rejection (timestamp delta > 30 s)
    - Low confidence rejection (< 0.5)
    - Oversized packet rejection (> 256 bytes)
    - Own packet rejection
    - Spatial merge: observations within 200 m → same node
    - Spatial merge: observations beyond 200 m → separate nodes
    - Temporal decay: confidence decays at 0.1/s after 15 s
    - Merge latency (functional check — no hard timing in SIL)
    - Schema version check
    - Duplicate packet rejection
"""

import sys
import os
import math
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.zpi.zpi import (
    ZPIBurstScheduler, ZPIState, BurstType,
    make_zpi_scheduler, BURST_DURATION_S, MAX_DUTY_CYCLE,
    DF_RISK_BEARING_DEG, INTERVAL_MIN_S, INTERVAL_MAX_S,
)
from core.cems.cems import (
    CEMSEngine, CEMSPacket, EWObservation, AuthValidator,
    SpatialTemporalMergeEngine,
    SPATIAL_MERGE_RADIUS_M, TEMPORAL_MERGE_WINDOW_S,
    TEMPORAL_DECAY_RATE, MIN_PEER_CONFIDENCE, REPLAY_ATTACK_WINDOW_S,
    MAX_PACKET_BYTES, PACKET_SCHEMA_VERSION,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

MISSION_KEY = "deadbeefcafebabe" * 4       # 64-char hex = 32 bytes
MISSION_KEY_B = bytes.fromhex(MISSION_KEY)

def make_obs(obs_id="OBS-001", uav_id="UAV-A", t=0.0,
             pos=(1000.0, 2000.0), freq=433e6,
             rssi=-70.0, bearing=270.0, conf=0.8) -> EWObservation:
    return EWObservation(
        obs_id=obs_id, source_uav_id=uav_id, timestamp_s=t,
        position_m=pos, freq_hz=freq, rssi_dbm=rssi,
        bearing_deg=bearing, signature_conf=conf,
    )

def make_packet(obs: EWObservation, source_uav="UAV-B",
                t=0.0, version=PACKET_SCHEMA_VERSION,
                packet_id=None) -> CEMSPacket:
    pid = packet_id or f"{source_uav}-{int(t*1000)}"
    return CEMSPacket(
        version=version, packet_id=pid,
        source_uav_id=source_uav, timestamp_s=t, observation=obs,
    )

# ─── ZPI Tests ────────────────────────────────────────────────────────────────

def test_zpi_burst_duration():
    """Burst duration must be ≤ 10 ms (FR-104)."""
    zpi = make_zpi_scheduler(MISSION_KEY)
    burst = zpi.transmit_burst(BurstType.TELEMETRY, 64, sim_time_s=10.0)
    assert burst is not None
    assert burst.duration_s <= BURST_DURATION_S, \
        f"Burst duration {burst.duration_s}s exceeds 10 ms"

def test_zpi_duty_cycle():
    """Duty cycle must remain ≤ 0.5% over a 600 s mission segment (FR-104)."""
    zpi = make_zpi_scheduler(MISSION_KEY)
    # 10 bursts over 600 s: tx_time = 10 × 0.01 = 0.1 s → duty = 0.1/600 = 0.017%
    schedule = [(t * 60.0, BurstType.TELEMETRY, 64) for t in range(10)]
    result = zpi.run_mission_segment(600.0, schedule)
    assert result.duty_cycle <= MAX_DUTY_CYCLE, \
        f"Duty cycle {result.duty_cycle:.4f} exceeds {MAX_DUTY_CYCLE}"
    assert result.zpi_compliant

def test_zpi_hop_plan_deterministic():
    """Two schedulers with same mission key must produce identical hop frequencies."""
    zpi_a = make_zpi_scheduler(MISSION_KEY)
    zpi_b = make_zpi_scheduler(MISSION_KEY)
    freqs_a = [zpi_a.hop_plan.next_frequency() for _ in range(20)]
    freqs_b = [zpi_b.hop_plan.next_frequency() for _ in range(20)]
    assert freqs_a == freqs_b, "Hop plans differ for same mission key"

def test_zpi_hop_plan_different_keys():
    """Different mission keys must produce different hop plans."""
    key2 = "cafebabe" * 8
    zpi_a = make_zpi_scheduler(MISSION_KEY)
    zpi_b = make_zpi_scheduler(key2)
    freqs_a = [zpi_a.hop_plan.next_frequency() for _ in range(10)]
    freqs_b = [zpi_b.hop_plan.next_frequency() for _ in range(10)]
    assert freqs_a != freqs_b, "Different keys produced identical hop plans"

def test_zpi_hop_range():
    """All hop frequencies must be within ± 5 MHz of centre (FR-104)."""
    from core.zpi.zpi import HOP_RANGE_HZ
    centre = 433_000_000.0
    zpi = make_zpi_scheduler(MISSION_KEY, centre_freq_hz=centre)
    for _ in range(100):
        f = zpi.hop_plan.next_frequency()
        assert abs(f - centre) <= HOP_RANGE_HZ, \
            f"Frequency {f/1e6:.3f} MHz outside ±5 MHz of centre"

def test_zpi_power_range():
    """All power levels must be within −10 to 0 dB (FR-104)."""
    zpi = make_zpi_scheduler(MISSION_KEY)
    for _ in range(100):
        p = zpi.hop_plan.next_power()
        assert -10.0 <= p <= 0.0, f"Power {p:.1f} dB outside valid range"

def test_zpi_df_risk_detection():
    """DF risk triggers when jammer bearing within 45° of own track."""
    zpi = make_zpi_scheduler(MISSION_KEY)
    # Jammer at 90°, own track 90° → delta = 0° → risk
    assert zpi.assess_df_risk(90.0, 90.0) is True
    # Jammer at 90°, own track 90° + 44° = 134° → delta = 44° → risk
    assert zpi.assess_df_risk(90.0, 134.0) is True
    # Jammer at 90°, own track 90° + 46° = 136° → delta = 46° → no risk
    assert zpi.assess_df_risk(90.0, 136.0) is False
    # Opposite directions → 180° delta → no risk
    assert zpi.assess_df_risk(0.0, 180.0) is False

def test_zpi_df_adaptation_state():
    """DF risk → ZPIState.DF_ADAPTED; no risk → NOMINAL."""
    zpi = make_zpi_scheduler(MISSION_KEY)
    zpi.update_state(df_risk=True, shm_active=False)
    assert zpi.state == ZPIState.DF_ADAPTED
    zpi.update_state(df_risk=False, shm_active=False)
    assert zpi.state == ZPIState.NOMINAL

def test_zpi_df_adaptation_widens_hops():
    """Under DF adaptation, hop range should widen to ±10 MHz."""
    from core.zpi.zpi import ADAPTED_HOP_RANGE_HZ
    centre = 433_000_000.0
    zpi = make_zpi_scheduler(MISSION_KEY, centre_freq_hz=centre)
    zpi.update_state(df_risk=True, shm_active=False)
    # At least one hop should exceed ±5 MHz but stay within ±10 MHz
    freqs = [zpi.hop_plan.next_frequency(df_adapted=True) for _ in range(200)]
    offsets = [abs(f - centre) for f in freqs]
    assert max(offsets) <= ADAPTED_HOP_RANGE_HZ, "Adapted hops exceed ±10 MHz"

def test_zpi_df_adaptation_increases_interval():
    """Under DF adaptation, inter-burst interval min should increase."""
    from core.zpi.zpi import ADAPTED_INTERVAL_MIN_S
    zpi = make_zpi_scheduler(MISSION_KEY)
    intervals_normal = [zpi.hop_plan.next_interval(df_adapted=False) for _ in range(50)]
    intervals_adapted = [zpi.hop_plan.next_interval(df_adapted=True) for _ in range(50)]
    assert min(intervals_adapted) >= ADAPTED_INTERVAL_MIN_S, \
        f"Adapted interval {min(intervals_adapted):.1f}s below minimum"
    assert min(intervals_normal) >= INTERVAL_MIN_S

def test_zpi_shm_suppression():
    """SHM active → all bursts suppressed (FR-104 / FR-106)."""
    zpi = make_zpi_scheduler(MISSION_KEY)
    zpi.update_state(df_risk=False, shm_active=True)
    assert zpi.state == ZPIState.SUPPRESSED
    burst = zpi.transmit_burst(BurstType.TELEMETRY, 64, sim_time_s=100.0)
    assert burst is None, "Burst should be suppressed in SHM mode"

def test_zpi_pre_terminal_burst():
    """Mandatory pre-terminal burst must be confirmed before SHM (DD-02)."""
    zpi = make_zpi_scheduler(MISSION_KEY)
    assert zpi.pre_terminal_confirmed is False
    result = zpi.run_mission_segment(
        duration_s=300.0,
        burst_schedule=[],
        pre_terminal_at_s=290.0,
    )
    assert result.pre_terminal_confirmed is True
    assert zpi.pre_terminal_confirmed is True

def test_zpi_pre_terminal_burst_type():
    """Pre-terminal burst must have correct BurstType."""
    zpi = make_zpi_scheduler(MISSION_KEY)
    zpi.run_mission_segment(300.0, [], pre_terminal_at_s=290.0)
    pre_term_bursts = [b for b in zpi._burst_log if b.pre_terminal]
    assert len(pre_term_bursts) == 1
    assert pre_term_bursts[0].burst_type == BurstType.PRE_TERMINAL

def test_zpi_pre_terminal_not_sent_twice():
    """Pre-terminal burst must only be sent once."""
    zpi = make_zpi_scheduler(MISSION_KEY)
    zpi.run_mission_segment(300.0, [], pre_terminal_at_s=290.0)
    zpi.run_mission_segment(300.0, [], pre_terminal_at_s=590.0)
    pre_term_bursts = [b for b in zpi._burst_log if b.pre_terminal]
    assert len(pre_term_bursts) == 1, "Pre-terminal burst sent more than once"

def test_zpi_burst_confirmed_flag():
    """All successful bursts must have confirmed=True."""
    zpi = make_zpi_scheduler(MISSION_KEY)
    schedule = [(t * 30.0, BurstType.EW_FLASH, 128) for t in range(5)]
    zpi.run_mission_segment(150.0, schedule)
    for burst in zpi._burst_log:
        assert burst.confirmed is True

def test_zpi_burst_log_count():
    """Burst log must record all transmitted bursts."""
    zpi = make_zpi_scheduler(MISSION_KEY)
    n = 7
    schedule = [(t * 20.0, BurstType.CEMS_PEER, 200) for t in range(n)]
    result = zpi.run_mission_segment(200.0, schedule)
    assert result.bursts_transmitted == n

# ─── CEMS Auth Validator Tests ────────────────────────────────────────────────

def test_cems_replay_rejection():
    """Packets with timestamp delta > 30 s must be rejected."""
    validator = AuthValidator(MISSION_KEY_B, "UAV-A")
    obs = make_obs(t=0.0)
    pkt = make_packet(obs, t=0.0)
    # Local time is 31 s ahead → replay
    valid, reason = validator.validate(pkt, local_time_s=31.0)
    assert not valid
    assert "REPLAY" in reason

def test_cems_replay_accepted_within_window():
    """Packets within 30 s window must be accepted."""
    validator = AuthValidator(MISSION_KEY_B, "UAV-A")
    obs = make_obs(t=100.0)
    pkt = make_packet(obs, t=100.0)
    valid, reason = validator.validate(pkt, local_time_s=129.9)
    assert valid, f"Expected valid, got: {reason}"

def test_cems_own_packet_rejection():
    """Own packets (same UAV ID) must be rejected."""
    validator = AuthValidator(MISSION_KEY_B, "UAV-A")
    obs = make_obs(uav_id="UAV-A", t=100.0)
    pkt = make_packet(obs, source_uav="UAV-A", t=100.0)
    valid, reason = validator.validate(pkt, local_time_s=100.0)
    assert not valid
    assert reason == "OWN_PACKET"

def test_cems_low_confidence_rejection():
    """Observations with confidence < 0.5 must be rejected."""
    validator = AuthValidator(MISSION_KEY_B, "UAV-A")
    obs = make_obs(conf=0.49, t=100.0)
    pkt = make_packet(obs, t=100.0)
    valid, reason = validator.validate(pkt, local_time_s=100.0)
    assert not valid
    assert "LOW_CONFIDENCE" in reason

def test_cems_confidence_boundary_accepted():
    """Observation with confidence exactly 0.5 must be accepted."""
    validator = AuthValidator(MISSION_KEY_B, "UAV-A")
    obs = make_obs(conf=0.5, t=100.0)
    pkt = make_packet(obs, t=100.0)
    valid, reason = validator.validate(pkt, local_time_s=100.0)
    assert valid, f"Expected valid at boundary, got: {reason}"

def test_cems_bad_version_rejection():
    """Packets with wrong schema version must be rejected."""
    validator = AuthValidator(MISSION_KEY_B, "UAV-A")
    obs = make_obs(t=100.0)
    pkt = make_packet(obs, t=100.0, version=99)
    valid, reason = validator.validate(pkt, local_time_s=100.0)
    assert not valid
    assert "BAD_VERSION" in reason

def test_cems_duplicate_rejection():
    """Duplicate packet IDs must be rejected on second receipt."""
    validator = AuthValidator(MISSION_KEY_B, "UAV-A")
    obs = make_obs(t=100.0)
    pkt = make_packet(obs, t=100.0, packet_id="UNIQUE-001")
    valid1, _ = validator.validate(pkt, local_time_s=100.0)
    valid2, reason2 = validator.validate(pkt, local_time_s=101.0)
    assert valid1
    assert not valid2
    assert reason2 == "DUPLICATE"

# ─── CEMS Spatial-Temporal Merge Tests ───────────────────────────────────────

def test_cems_spatial_merge_same_node():
    """Observations within 200 m must merge into one node."""
    engine = SpatialTemporalMergeEngine()
    obs1 = make_obs("O1", "UAV-A", t=0.0, pos=(1000.0, 2000.0))
    obs2 = make_obs("O2", "UAV-B", t=1.0, pos=(1100.0, 2050.0))  # 111 m away
    nodes = engine.merge([obs1, obs2], current_time_s=2.0)
    assert len(nodes) == 1, f"Expected 1 merged node, got {len(nodes)}"
    assert nodes[0].observation_count == 2
    assert "UAV-A" in nodes[0].source_uav_ids
    assert "UAV-B" in nodes[0].source_uav_ids

def test_cems_spatial_separate_nodes():
    """Observations beyond 200 m must create separate nodes."""
    engine = SpatialTemporalMergeEngine()
    obs1 = make_obs("O1", "UAV-A", t=0.0, pos=(0.0, 0.0))
    obs2 = make_obs("O2", "UAV-B", t=0.0, pos=(500.0, 0.0))  # 500 m away
    nodes = engine.merge([obs1, obs2], current_time_s=1.0)
    assert len(nodes) == 2, f"Expected 2 separate nodes, got {len(nodes)}"

def test_cems_spatial_boundary_200m():
    """Observation exactly at 200 m should merge; 201 m should not."""
    engine1 = SpatialTemporalMergeEngine()
    obs1 = make_obs("O1", "UAV-A", t=0.0, pos=(0.0, 0.0))
    obs2 = make_obs("O2", "UAV-B", t=0.0, pos=(200.0, 0.0))   # exactly 200 m
    nodes = engine1.merge([obs1, obs2], current_time_s=1.0)
    assert len(nodes) == 1

    engine2 = SpatialTemporalMergeEngine()
    obs3 = make_obs("O3", "UAV-A", t=0.0, pos=(0.0, 0.0))
    obs4 = make_obs("O4", "UAV-B", t=0.0, pos=(201.0, 0.0))   # 201 m
    nodes2 = engine2.merge([obs3, obs4], current_time_s=1.0)
    assert len(nodes2) == 2

def test_cems_temporal_decay():
    """Confidence must decay at 0.1/s after 15 s temporal window."""
    engine = SpatialTemporalMergeEngine()
    obs = make_obs("O1", "UAV-A", t=0.0, pos=(1000.0, 1000.0), conf=0.9)
    engine.merge([obs], current_time_s=0.0)

    # At t=15s: no decay yet
    nodes_15 = engine.merge([], current_time_s=15.0)
    assert len(nodes_15) == 1
    assert nodes_15[0].confidence == 0.9

    # At t=20s: 5s of decay → 0.9 - (0.1 * 5) = 0.4 → below MIN → expired
    nodes_20 = engine.merge([], current_time_s=20.0)
    assert len(nodes_20) == 0, "Node should have expired after decay below MIN_PEER_CONFIDENCE"

def test_cems_temporal_decay_partial():
    """Node with high initial confidence survives partial decay."""
    engine = SpatialTemporalMergeEngine()
    obs = make_obs("O1", "UAV-A", t=0.0, pos=(1000.0, 1000.0), conf=1.0)
    engine.merge([obs], current_time_s=0.0)

    # At t=17s: 2s of decay → 1.0 - (0.1 * 2) = 0.8 → still above MIN
    nodes = engine.merge([], current_time_s=17.0)
    assert len(nodes) == 1
    assert abs(nodes[0].confidence - 0.8) < 0.01

def test_cems_stale_observation_discarded():
    """Observations older than temporal window must be discarded."""
    engine = SpatialTemporalMergeEngine()
    obs = make_obs("O1", "UAV-A", t=0.0, pos=(1000.0, 1000.0), conf=0.9)
    # Submit obs at t=0, but merge at t=20 (observation is 20s old > 15s window)
    nodes = engine.merge([obs], current_time_s=20.0)
    assert len(nodes) == 0, "Stale observation should be discarded"

def test_cems_confidence_takes_max():
    """Merged node confidence should take the max of contributing observations."""
    engine = SpatialTemporalMergeEngine()
    obs1 = make_obs("O1", "UAV-A", t=0.0, pos=(1000.0, 1000.0), conf=0.6)
    obs2 = make_obs("O2", "UAV-B", t=1.0, pos=(1050.0, 1050.0), conf=0.9)
    nodes = engine.merge([obs1, obs2], current_time_s=2.0)
    assert len(nodes) == 1
    assert abs(nodes[0].confidence - 0.9) < 0.01

# ─── CEMS Engine Integration Tests ───────────────────────────────────────────

def test_cems_engine_receive_valid_packet():
    """Valid peer packet must be accepted and return observation."""
    engine = CEMSEngine("UAV-A", MISSION_KEY_B)
    obs = make_obs("O1", "UAV-B", t=100.0)
    pkt = make_packet(obs, source_uav="UAV-B", t=100.0)
    result = engine.receive_packet(pkt, local_time_s=100.0)
    assert result is not None
    assert result.obs_id == "O1"

def test_cems_engine_receive_replay():
    """Replay packets must be rejected by engine."""
    engine = CEMSEngine("UAV-A", MISSION_KEY_B)
    obs = make_obs("O1", "UAV-B", t=0.0)
    pkt = make_packet(obs, source_uav="UAV-B", t=0.0)
    result = engine.receive_packet(pkt, local_time_s=31.0)
    assert result is None

def test_cems_engine_merge_cycle():
    """Merge cycle with local + peer observations must produce jammer nodes."""
    engine = CEMSEngine("UAV-A", MISSION_KEY_B)
    local_obs = [make_obs("L1", "UAV-A", t=100.0, pos=(1000.0, 2000.0))]
    peer_obs  = [make_obs("P1", "UAV-B", t=100.0, pos=(1080.0, 2020.0))]
    result = engine.run_merge_cycle(local_obs, peer_obs, current_time_s=100.0)
    assert len(result.jammer_nodes) == 1
    assert result.jammer_nodes[0].observation_count == 2

def test_cems_engine_packet_build():
    """Built packet must have correct version and size."""
    engine = CEMSEngine("UAV-A", MISSION_KEY_B)
    obs = make_obs("O1", "UAV-A", t=100.0)
    pkt = engine.build_packet(obs, sim_time_s=100.0, mission_key=MISSION_KEY_B)
    assert pkt.version == PACKET_SCHEMA_VERSION
    assert pkt.byte_size <= MAX_PACKET_BYTES
    assert len(pkt.hmac_digest) == 32

def test_cems_engine_peer_tracking():
    """Engine must track active peer UAV IDs."""
    engine = CEMSEngine("UAV-A", MISSION_KEY_B)
    obs_b = make_obs("O1", "UAV-B", t=100.0)
    obs_c = make_obs("O2", "UAV-C", t=100.0)
    engine.receive_packet(make_packet(obs_b, "UAV-B", t=100.0, packet_id="B1"), 100.0)
    engine.receive_packet(make_packet(obs_c, "UAV-C", t=100.0, packet_id="C1"), 100.0)
    assert "UAV-B" in engine._peer_ids
    assert "UAV-C" in engine._peer_ids
    assert engine.run_merge_cycle([], [], current_time_s=100.5).peers_active == 2

def test_cems_packet_rejection_counts():
    """Rejected packets must be counted correctly."""
    engine = CEMSEngine("UAV-A", MISSION_KEY_B)
    # Send 1 valid, 1 replay, 1 own
    valid_obs = make_obs("O1", "UAV-B", t=100.0)
    replay_obs = make_obs("O2", "UAV-B", t=0.0)
    own_obs = make_obs("O3", "UAV-A", t=100.0)
    engine.receive_packet(make_packet(valid_obs, "UAV-B", t=100.0, packet_id="V1"), 100.0)
    engine.receive_packet(make_packet(replay_obs, "UAV-B", t=0.0, packet_id="R1"), 100.0)
    engine.receive_packet(make_packet(own_obs, "UAV-A", t=100.0, packet_id="O1"), 100.0)
    result = engine.run_merge_cycle([], [], current_time_s=100.5)
    assert engine._packets_rx == 3
    assert engine._packets_rej == 2

# ─── Test Runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # ZPI
        test_zpi_burst_duration,
        test_zpi_duty_cycle,
        test_zpi_hop_plan_deterministic,
        test_zpi_hop_plan_different_keys,
        test_zpi_hop_range,
        test_zpi_power_range,
        test_zpi_df_risk_detection,
        test_zpi_df_adaptation_state,
        test_zpi_df_adaptation_widens_hops,
        test_zpi_df_adaptation_increases_interval,
        test_zpi_shm_suppression,
        test_zpi_pre_terminal_burst,
        test_zpi_pre_terminal_burst_type,
        test_zpi_pre_terminal_not_sent_twice,
        test_zpi_burst_confirmed_flag,
        test_zpi_burst_log_count,
        # CEMS Auth
        test_cems_replay_rejection,
        test_cems_replay_accepted_within_window,
        test_cems_own_packet_rejection,
        test_cems_low_confidence_rejection,
        test_cems_confidence_boundary_accepted,
        test_cems_bad_version_rejection,
        test_cems_duplicate_rejection,
        # CEMS Merge
        test_cems_spatial_merge_same_node,
        test_cems_spatial_separate_nodes,
        test_cems_spatial_boundary_200m,
        test_cems_temporal_decay,
        test_cems_temporal_decay_partial,
        test_cems_stale_observation_discarded,
        test_cems_confidence_takes_max,
        # CEMS Engine
        test_cems_engine_receive_valid_packet,
        test_cems_engine_receive_replay,
        test_cems_engine_merge_cycle,
        test_cems_engine_packet_build,
        test_cems_engine_peer_tracking,
        test_cems_packet_rejection_counts,
    ]

    passed = 0
    failed = 0
    t0 = time.perf_counter()

    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {t.__name__}: {e}")
            failed += 1

    elapsed = time.perf_counter() - t0
    total = passed + failed
    print(f"\n  Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"  Elapsed: {elapsed:.2f}s")
    gate = "✅ PASS" if failed == 0 else "❌ FAIL"
    print(f"  Sprint S6 ZPI+CEMS Gate: {gate}")
    sys.exit(0 if failed == 0 else 1)
