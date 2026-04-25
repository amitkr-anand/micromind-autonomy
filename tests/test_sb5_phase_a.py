"""
tests/test_sb5_phase_a.py
MicroMind / NanoCorteX — SB-5 Phase A Gates

SA-01  test_sa01_checkpoint_v12_fields_present        Schema completeness
SA-02  test_sa02_checkpoint_restore_after_sigkill     SIGKILL restore, position error < 20 m
SA-03  test_sa03_checkpoint_rolling_purge             Rolling purge ≤5 retained
SA-04  test_sa04_p02_operator_clearance_blocks_resume P-02 operator clearance gate

All gates are independent; no test depends on another test's side effects.
setUp / tearDown isolate filesystem state (checkpoint files) between tests.

Requirements: PX4-05, EC-02
SRS ref:      §10.15, UT-PX4-05, corrections P-01, P-02
Governance:   Code Governance Manual v3.2 §1.3, §2.5, §9.1
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from typing import List

from core.checkpoint.checkpoint import Checkpoint, CheckpointStore
from core.mission_manager.mission_manager import MissionManager, MissionState


# ---------------------------------------------------------------------------
# SA-01 — Schema completeness
# ---------------------------------------------------------------------------

class TestSA01CheckpointV12FieldsPresent(unittest.TestCase):
    """
    SA-01: All six v1.2 fields are present in the serialised output and
    round-trip correctly through dict and JSON serialisation paths.
    """

    def test_sa01_checkpoint_v12_fields_present(self):
        # Instantiate Checkpoint with all six new fields at non-default values
        cp = Checkpoint(
            mission_id="SA-01-TEST",
            timestamp_ms=12345,
            shm_active=True,
            pending_operator_clearance_required=True,
            mission_abort_flag=True,
            eta_to_destination_ms=98765,
            terrain_corridor_phase="PHASE_B",
            route_corridor_half_width_m=150.0,
        )

        # --- Serialise to dict -----------------------------------------------
        d = cp.to_dict()

        v12_fields = [
            "shm_active",
            "pending_operator_clearance_required",
            "mission_abort_flag",
            "eta_to_destination_ms",
            "terrain_corridor_phase",
            "route_corridor_half_width_m",
        ]

        for fname in v12_fields:
            self.assertIn(fname, d,
                f"Field '{fname}' missing from Checkpoint.to_dict() output")

        # --- Assert correct Python types in dict --------------------------------
        self.assertIsInstance(d["shm_active"], bool,
            "shm_active must be bool in serialised dict")
        self.assertIsInstance(d["pending_operator_clearance_required"], bool,
            "pending_operator_clearance_required must be bool in serialised dict")
        self.assertIsInstance(d["mission_abort_flag"], bool,
            "mission_abort_flag must be bool in serialised dict")
        self.assertIsInstance(d["eta_to_destination_ms"], int,
            "eta_to_destination_ms must be int in serialised dict")
        self.assertIsInstance(d["terrain_corridor_phase"], str,
            "terrain_corridor_phase must be str in serialised dict")
        self.assertIsInstance(d["route_corridor_half_width_m"], float,
            "route_corridor_half_width_m must be float in serialised dict")

        # --- Serialise to JSON and confirm keys survive -------------------------
        json_str = cp.to_json()
        data_from_json = json.loads(json_str)
        for fname in v12_fields:
            self.assertIn(fname, data_from_json,
                f"Field '{fname}' missing from JSON-deserialised output")

        # --- Round-trip: dict → Checkpoint → assert values match ---------------
        cp2 = Checkpoint.from_dict(d)

        self.assertEqual(cp2.shm_active, True)
        self.assertEqual(cp2.pending_operator_clearance_required, True)
        self.assertEqual(cp2.mission_abort_flag, True)
        self.assertEqual(cp2.eta_to_destination_ms, 98765)
        self.assertEqual(cp2.terrain_corridor_phase, "PHASE_B")
        self.assertAlmostEqual(cp2.route_corridor_half_width_m, 150.0, places=6,
            msg="route_corridor_half_width_m round-trip value mismatch")

        # --- Round-trip: JSON string → Checkpoint → assert values match ---------
        cp3 = Checkpoint.from_json(json_str)
        self.assertEqual(cp3.shm_active, True)
        self.assertEqual(cp3.eta_to_destination_ms, 98765)
        self.assertEqual(cp3.terrain_corridor_phase, "PHASE_B")
        self.assertAlmostEqual(cp3.route_corridor_half_width_m, 150.0, places=6)


# ---------------------------------------------------------------------------
# SA-02 — SIGKILL restore
# ---------------------------------------------------------------------------

class TestSA02CheckpointRestoreAfterSigkill(unittest.TestCase):
    """
    SA-02: A checkpoint written to disk survives simulated SIGKILL.

    Simulates SIGKILL by writing a checkpoint and then reading it back
    without going through any normal shutdown path.  Asserts:
      - Restored position error < 20 m from written position.
      - All six v1.2 fields are present and correct in the restored checkpoint.

    P-01 verification: shm_active=True is written and restored correctly,
    ensuring a vehicle that was in SHM at checkpoint time will re-enter SHM
    rather than resume autonomous flight on reboot.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="sa02_cp_")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_sa02_checkpoint_restore_after_sigkill(self):
        store = CheckpointStore(checkpoint_dir=self._tmpdir)

        # Write a checkpoint with known ESKF position (NED frame)
        written_pos = [100.0, 200.0, -95.0]
        cp = Checkpoint(
            mission_id="SA-02-SIGKILL-TEST",
            timestamp_ms=1000,
            pos_ned=written_pos,
            fsm_state="SHM_ACTIVE",
            shm_active=True,                           # P-01: SHM state persisted
            pending_operator_clearance_required=False,
            mission_abort_flag=False,
            eta_to_destination_ms=60000,
            terrain_corridor_phase="INGRESS",
            route_corridor_half_width_m=75.5,
        )
        store.write(cp)

        # --- Restore WITHOUT normal shutdown path ----------------------------
        # In a real SIGKILL scenario: process is killed, restarts, reads last
        # written JSON file from disk.  Here we simulate that by calling
        # restore_latest() directly — no cleanup, no finalizer, no atexit.
        restored = store.restore_latest()
        self.assertIsNotNone(restored,
            "restore_latest() returned None — no checkpoint file found on disk")

        # --- Assert position error < 20 m ------------------------------------
        error_m = math.sqrt(sum(
            (r - w) ** 2
            for r, w in zip(restored.pos_ned, written_pos)
        ))
        self.assertLess(error_m, 20.0,
            f"Restored position error {error_m:.6f} m exceeds 20 m threshold")

        # --- Assert all six v1.2 fields present and correct ------------------
        self.assertEqual(restored.shm_active, True,
            "P-01: shm_active must round-trip as True")
        self.assertEqual(restored.pending_operator_clearance_required, False)
        self.assertEqual(restored.mission_abort_flag, False)
        self.assertEqual(restored.eta_to_destination_ms, 60000)
        self.assertEqual(restored.terrain_corridor_phase, "INGRESS")
        self.assertAlmostEqual(restored.route_corridor_half_width_m, 75.5, places=6,
            msg="route_corridor_half_width_m round-trip value mismatch after SIGKILL restore")


# ---------------------------------------------------------------------------
# SA-03 — Rolling purge
# ---------------------------------------------------------------------------

class TestSA03CheckpointRollingPurge(unittest.TestCase):
    """
    SA-03: CheckpointStore purges oldest files so that ≤5 checkpoints are
    retained on disk after six sequential writes.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="sa03_cp_")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_sa03_checkpoint_rolling_purge(self):
        event_log: List = []
        store = CheckpointStore(
            checkpoint_dir=self._tmpdir,
            max_retained=5,
            event_log=event_log,
        )

        # Write six checkpoints in sequence (distinct timestamps)
        for i in range(6):
            cp = Checkpoint(
                mission_id=f"SA-03-TEST-{i}",
                timestamp_ms=i * 1000,   # 0, 1000, 2000, 3000, 4000, 5000 ms
                waypoint_index=i,
                shm_active=False,
            )
            store.write(cp)

        # --- Assert retained count <= 5 --------------------------------------
        retained = store.checkpoints_retained_count
        self.assertLessEqual(retained, 5,
            f"Expected ≤5 retained checkpoints after 6 writes, got {retained}")

        # --- Assert CHECKPOINT_PURGED logged at least once -------------------
        purge_events = [e for e in event_log if e["event"] == "CHECKPOINT_PURGED"]
        self.assertGreater(len(purge_events), 0,
            "Expected at least one CHECKPOINT_PURGED event — none found in event_log")

        # --- Confirm CHECKPOINT_WRITTEN events count == 6 --------------------
        write_events = [e for e in event_log if e["event"] == "CHECKPOINT_WRITTEN"]
        self.assertEqual(len(write_events), 6,
            f"Expected 6 CHECKPOINT_WRITTEN events, got {len(write_events)}")


# ---------------------------------------------------------------------------
# SA-04 — P-02 operator clearance blocks resume
# ---------------------------------------------------------------------------

class TestSA04P02OperatorClearanceBlocksResume(unittest.TestCase):
    """
    SA-04: MissionManager.resume() blocks autonomous mission resume when
    pending_operator_clearance_required=True (P-02 operator clearance gate).

    Verifies:
      - resume() returns False
      - Mission state is NOT active/running
      - SHM is entered (MissionState.SHM)
      - AWAITING_OPERATOR_CLEARANCE is logged with all four required fields:
          req_id, severity, module_name, timestamp_ms
    """

    def test_sa04_p02_operator_clearance_blocks_resume(self):
        # Construct a checkpoint requiring operator clearance (P-02 trigger)
        cp = Checkpoint(
            mission_id="SA-04-P02-TEST",
            timestamp_ms=5000,
            fsm_state="SHM_ACTIVE",
            shm_active=True,
            pending_operator_clearance_required=True,   # P-02 trigger
        )

        event_log: List = []
        current_time_ms = 5000
        mm = MissionManager(
            event_log=event_log,
            clock_fn=lambda: current_time_ms,
        )

        # --- Trigger the Mission Manager resume path -------------------------
        resumed = mm.resume(cp)

        # --- Assert mission did NOT resume -----------------------------------
        self.assertFalse(resumed,
            "resume() must return False when pending_operator_clearance_required=True")

        # --- Assert mission state is NOT active/running ----------------------
        self.assertNotEqual(mm.state, MissionState.ACTIVE,
            "Mission state must NOT be ACTIVE after P-02 blocked resume")
        self.assertNotEqual(mm.state, MissionState.RESUMING,
            "Mission state must NOT be RESUMING after P-02 blocked resume")

        # --- Assert SHM is entered -------------------------------------------
        self.assertEqual(mm.state, MissionState.SHM,
            f"Expected state=SHM after P-02 blocked resume, got {mm.state}")
        self.assertTrue(mm.shm_entered,
            "shm_entered property must be True after P-02 blocks resume")

        # --- Assert AWAITING_OPERATOR_CLEARANCE logged with all four fields --
        clearance_events = [
            e for e in event_log if e.get("event") == "AWAITING_OPERATOR_CLEARANCE"
        ]
        self.assertGreater(len(clearance_events), 0,
            "No AWAITING_OPERATOR_CLEARANCE event found in event_log")

        evt = clearance_events[0]

        self.assertIn("req_id", evt,
            "req_id field missing from AWAITING_OPERATOR_CLEARANCE event")
        self.assertEqual(evt["req_id"], "PX4-05",
            f"req_id: expected 'PX4-05', got '{evt.get('req_id')}'")

        self.assertIn("severity", evt,
            "severity field missing from AWAITING_OPERATOR_CLEARANCE event")
        self.assertEqual(evt["severity"], "WARNING",
            f"severity: expected 'WARNING', got '{evt.get('severity')}'")

        self.assertIn("module_name", evt,
            "module_name field missing from AWAITING_OPERATOR_CLEARANCE event")
        self.assertEqual(evt["module_name"], "MissionManager",
            f"module_name: expected 'MissionManager', got '{evt.get('module_name')}'")

        self.assertIn("timestamp_ms", evt,
            "timestamp_ms field missing from AWAITING_OPERATOR_CLEARANCE event")
        self.assertEqual(evt["timestamp_ms"], 5000,
            f"timestamp_ms: expected 5000, got '{evt.get('timestamp_ms')}'")



# ---------------------------------------------------------------------------
# SA-05 — Reboot detected within 3 s
# ---------------------------------------------------------------------------

class TestSA05RebootDetectedWithin3s(unittest.TestCase):
    """
    SA-05: RebootDetector triggers PX4_REBOOT_DETECTED on a backward sequence-
    number jump of 10, and reports elapsed detection time <= 3000 ms.

    RebootDetector is the sequence-number reset detection component used by
    MAVLinkBridge (integration/bridge/reboot_detector.py).  It is tested here
    directly because pymavlink is absent from the SIL conda environment and
    MAVLinkBridge cannot be imported in SIL context — the detection logic is
    isolated in a pymavlink-free module specifically for this purpose.

    Injection pattern (equivalent to injecting a HEARTBEAT into the bridge
    handler with seq = last_seq - 10):
      1. feed(seq=BASELINE) — establishes last_seq
      2. feed(seq=BASELINE-10) — triggers backward jump detection
    """

    def test_sa05_reboot_detected_within_3s(self):
        from integration.bridge.reboot_detector import RebootDetector

        event_log = []
        detector = RebootDetector(event_log=event_log)

        baseline_seq = 50
        injected_seq = baseline_seq - 10   # = 40 — backward jump of 10

        # Step 1: establish last_seq baseline
        detector.feed(seq=baseline_seq)

        # Step 2: inject HEARTBEAT with seq = last_seq - 10
        detected = detector.feed(seq=injected_seq)

        # --- Assert PX4_REBOOT_DETECTED was logged ---------------------------
        self.assertTrue(detected,
            "RebootDetector.feed() must return True on seq-reset detection")

        reboot_events = [e for e in event_log if e.get("event") == "PX4_REBOOT_DETECTED"]
        self.assertGreater(len(reboot_events), 0,
            "No PX4_REBOOT_DETECTED event found in event_log after backward seq jump of 10")

        evt = reboot_events[0]

        # --- Assert required event fields ------------------------------------
        self.assertEqual(evt["req_id"], "PX4-04")
        self.assertEqual(evt["severity"], "WARNING")
        self.assertEqual(evt["module_name"], "MAVLinkBridge")
        self.assertIn("timestamp_ms", evt)
        self.assertIn("payload", evt)

        # --- Assert elapsed detection time <= 3000 ms -----------------------
        elapsed_ms = evt["payload"]["elapsed_detection_ms"]
        self.assertLessEqual(elapsed_ms, 3000,
            f"elapsed_detection_ms {elapsed_ms} ms exceeds 3000 ms threshold")


# ---------------------------------------------------------------------------
# SA-06 — D8a clearance=False → MISSION_RESUME_AUTHORISED
# ---------------------------------------------------------------------------

class TestSA06D8aClearanceFalseResumes(unittest.TestCase):
    """
    SA-06: D8a gate nominal path — when pending_operator_clearance_required=False,
    MissionManager.on_reboot_detected() logs MISSION_RESUME_AUTHORISED and
    sets mission state to ACTIVE.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="sa06_cp_")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_sa06_d8a_clearance_false_resumes(self):
        # Write a checkpoint with pending_operator_clearance_required=False
        store = CheckpointStore(checkpoint_dir=self._tmpdir)
        cp = Checkpoint(
            mission_id="SA-06-D8A-TEST",
            timestamp_ms=8000,
            fsm_state="GNSS_DENIED",
            waypoint_index=3,
            pending_operator_clearance_required=False,   # nominal path
            shm_active=False,
        )
        store.write(cp)

        event_log = []
        current_time_ms = 8100
        mm = MissionManager(
            event_log=event_log,
            clock_fn=lambda: current_time_ms,
        )

        # Trigger MissionManager reboot-recovery path
        resumed = mm.on_reboot_detected(store)

        # --- Assert mission resumed ------------------------------------------
        self.assertTrue(resumed,
            "on_reboot_detected() must return True when pending_operator_clearance_required=False")
        self.assertEqual(mm.state, MissionState.ACTIVE,
            f"Expected state=ACTIVE after D8a nominal resume, got {mm.state}")

        # --- Assert MISSION_RESUME_AUTHORISED logged with all four required fields ---
        resume_events = [
            e for e in event_log if e.get("event") == "MISSION_RESUME_AUTHORISED"
        ]
        self.assertGreater(len(resume_events), 0,
            "No MISSION_RESUME_AUTHORISED event found in event_log")

        evt = resume_events[0]

        self.assertIn("req_id", evt,
            "req_id field missing from MISSION_RESUME_AUTHORISED event")
        self.assertEqual(evt["req_id"], "PX4-04",
            f"req_id: expected 'PX4-04', got '{evt.get('req_id')}'")

        self.assertIn("severity", evt,
            "severity field missing from MISSION_RESUME_AUTHORISED event")
        self.assertEqual(evt["severity"], "INFO",
            f"severity: expected 'INFO', got '{evt.get('severity')}'")

        self.assertIn("module_name", evt,
            "module_name field missing from MISSION_RESUME_AUTHORISED event")
        self.assertEqual(evt["module_name"], "MissionManager",
            f"module_name: expected 'MissionManager', got '{evt.get('module_name')}'")

        self.assertIn("timestamp_ms", evt,
            "timestamp_ms field missing from MISSION_RESUME_AUTHORISED event")


# ---------------------------------------------------------------------------
# SA-07 — D8a clearance=True → AWAITING_OPERATOR_CLEARANCE (SHM entered)
# ---------------------------------------------------------------------------

class TestSA07D8aClearanceTrueBlocks(unittest.TestCase):
    """
    SA-07: D8a gate P-02 path — when pending_operator_clearance_required=True,
    MissionManager.on_reboot_detected() logs AWAITING_OPERATOR_CLEARANCE and
    enters SHM; mission state is NOT ACTIVE.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="sa07_cp_")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_sa07_d8a_clearance_true_blocks(self):
        # Write a checkpoint with pending_operator_clearance_required=True
        store = CheckpointStore(checkpoint_dir=self._tmpdir)
        cp = Checkpoint(
            mission_id="SA-07-D8A-TEST",
            timestamp_ms=9000,
            fsm_state="SHM_ACTIVE",
            shm_active=True,
            pending_operator_clearance_required=True,    # P-02 trigger
            mission_abort_flag=False,
        )
        store.write(cp)

        event_log = []
        current_time_ms = 9100
        mm = MissionManager(
            event_log=event_log,
            clock_fn=lambda: current_time_ms,
        )

        # Trigger MissionManager reboot-recovery path
        resumed = mm.on_reboot_detected(store)

        # --- Assert mission did NOT resume -----------------------------------
        self.assertFalse(resumed,
            "on_reboot_detected() must return False when pending_operator_clearance_required=True")
        self.assertNotEqual(mm.state, MissionState.ACTIVE,
            "Mission state must NOT be ACTIVE when P-02 gate blocks D8a resume")

        # --- Assert SHM is entered -------------------------------------------
        self.assertEqual(mm.state, MissionState.SHM,
            f"Expected state=SHM after D8a blocked resume, got {mm.state}")
        self.assertTrue(mm.shm_entered,
            "shm_entered must be True after D8a P-02 blocks resume")

        # --- Assert AWAITING_OPERATOR_CLEARANCE logged -----------------------
        clearance_events = [
            e for e in event_log if e.get("event") == "AWAITING_OPERATOR_CLEARANCE"
        ]
        self.assertGreater(len(clearance_events), 0,
            "No AWAITING_OPERATOR_CLEARANCE event found after D8a P-02 trigger")

        evt = clearance_events[0]
        self.assertEqual(evt.get("req_id"), "PX4-05")
        self.assertEqual(evt.get("severity"), "WARNING")
        self.assertEqual(evt.get("module_name"), "MissionManager")
        self.assertIn("timestamp_ms", evt)


# ---------------------------------------------------------------------------
# E-01 / EC-02 — Checkpoint purge confirmation gate
# ---------------------------------------------------------------------------

class TestE01CheckpointPurge(unittest.TestCase):
    """
    E-01 / EC-02: Targeted checkpoint rolling-purge confirmation.

    SA-03 verifies retained <= 5 and that at least one CHECKPOINT_PURGED
    event is emitted.  This gate adds the assertions required by EC-02:
      - Exactly 5 checkpoints retained (not just ≤5).
      - The OLDEST checkpoint is the one purged (verified via checkpoint_id).
      - CHECKPOINT_PURGED event payload inspected: checkpoint_id, req_id.
      - checkpoints_retained_count == 5 via store property.
      - 7th write also triggers a second purge (count stays ≤5).

    Requirements: EC-02, SRS §10.15 PX4-05
    Note: CHECKPOINT_PURGED req_id is "PX4-05" in CheckpointStore — EC-02
    is the SRS requirement; PX4-05 is the implementation log tag. Both
    reference SRS §10.15.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="e01_cp_")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_e01_checkpoint_purge(self):
        event_log: List = []
        store = CheckpointStore(
            checkpoint_dir=self._tmpdir,
            max_retained=5,
            event_log=event_log,
        )

        # Write 6 checkpoints; capture oldest checkpoint_id prefix for purge assertion
        first_id_prefix: str = ""
        for i in range(6):
            cp = Checkpoint(
                mission_id=f"E01-EC02-{i}",
                timestamp_ms=(i + 1) * 1000,   # 1000, 2000, …, 6000 ms (distinct, increasing)
                waypoint_index=i,
            )
            if i == 0:
                first_id_prefix = cp.checkpoint_id[:8]
            store.write(cp)

        # --- Assertion 1: exactly 5 checkpoints retained on disk ----------------
        self.assertEqual(store.checkpoints_retained_count, 5,
            f"Expected exactly 5 retained checkpoints after 6 writes, "
            f"got {store.checkpoints_retained_count}")

        # --- Assertion 2: exactly 1 CHECKPOINT_PURGED event emitted -------------
        purge_events = [e for e in event_log if e["event"] == "CHECKPOINT_PURGED"]
        self.assertEqual(len(purge_events), 1,
            f"Expected exactly 1 CHECKPOINT_PURGED event after 6 writes, "
            f"got {len(purge_events)}")

        purge_evt = purge_events[0]

        # --- Assertion 3: oldest checkpoint was the one purged ------------------
        # CheckpointStore._purge() stores the 8-char id prefix extracted from filename
        self.assertEqual(purge_evt["checkpoint_id"], first_id_prefix,
            f"CHECKPOINT_PURGED checkpoint_id mismatch: "
            f"expected '{first_id_prefix}' (oldest), got '{purge_evt.get('checkpoint_id')}'")

        # --- Assertion 4a: purged_checkpoint_id field present in event ----------
        self.assertIn("checkpoint_id", purge_evt,
            "CHECKPOINT_PURGED event missing 'checkpoint_id' field")

        # --- Assertion 4b: checkpoints_retained_count == 5 (store property) ----
        self.assertEqual(store.checkpoints_retained_count, 5,
            "checkpoints_retained_count property must equal 5 after purge")

        # --- Assertion 4c: req_id = "PX4-05" (EC-02 traced through SRS §10.15 PX4-05)
        self.assertEqual(purge_evt["req_id"], "PX4-05",
            f"CHECKPOINT_PURGED req_id: expected 'PX4-05', "
            f"got '{purge_evt.get('req_id')}'")

        # --- Assertion 5: 7th write also triggers purge; count stays ≤5 --------
        cp7 = Checkpoint(
            mission_id="E01-EC02-6",
            timestamp_ms=7000,
            waypoint_index=6,
        )
        store.write(cp7)

        self.assertLessEqual(store.checkpoints_retained_count, 5,
            f"After 7th write, retained count must remain ≤5, "
            f"got {store.checkpoints_retained_count}")

        purge_events_total = [e for e in event_log if e["event"] == "CHECKPOINT_PURGED"]
        self.assertEqual(len(purge_events_total), 2,
            f"Expected 2 CHECKPOINT_PURGED events after 7 writes, "
            f"got {len(purge_events_total)}")


if __name__ == "__main__":
    unittest.main()
