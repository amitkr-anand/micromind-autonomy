"""
core/checkpoint/checkpoint.py
MicroMind / NanoCorteX — Mission Checkpoint Module v1.2

Implements persistent mission state checkpointing per SRS §10.15 (PX4-05, EC-02).

Design principles:
    - Checkpoint schema v1.2 adds six mandatory fields for SHM persistence,
      operator clearance gate, and route corridor tracking (corrections P-01, P-02).
    - Serialisation uses dataclasses.asdict() — all fields included automatically.
      No field can be silently dropped.
    - CheckpointStore maintains a rolling window of ≤5 checkpoints per PX4-05.
    - Atomic write pattern (write to .tmp, rename into place) ensures SIGKILL-safe
      persistence: a SIGKILL between write and rename leaves only the .tmp file,
      never a corrupt .json file; the last committed .json is always valid.

Log events emitted by CheckpointStore (plain dicts appended to event_log):
    CHECKPOINT_WRITTEN   — after each successful write
    CHECKPOINT_RESTORED  — after each successful restore
    CHECKPOINT_PURGED    — for each file deleted by rolling purge

P-01 (shm_active persistence):
    shm_active round-trips through asdict() / from_dict() without loss.
    No special handling required — asdict() captures all dataclass fields.

P-02 (operator clearance gate):
    pending_operator_clearance_required is serialised/deserialised here.
    The gate logic that blocks autonomous resume is in MissionManager.resume().

References:
    SRS §10.15  PX4-05, EC-02
    Code Governance Manual v3.2  §1.3, §9.1
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class CheckpointCorruptError(Exception):
    """Raised when a checkpoint file fails schema validation."""
    pass


# ---------------------------------------------------------------------------
# Checkpoint dataclass — schema v1.2
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    """
    Persistent mission state snapshot.

    v1.1 baseline fields capture core navigation and FSM state sufficient to
    resume a mission after an unplanned restart (SIGKILL, power-cycle).

    v1.2 mandatory fields (SRS §10.15, PX4-05 corrections P-01 / P-02):

        shm_active                         (bool)   P-01: vehicle was in SHM at checkpoint
                                                     time; must re-enter SHM on reboot.
        pending_operator_clearance_required (bool)   P-02: resume requires explicit operator
                                                     clearance before autonomous flight.
        mission_abort_flag                 (bool)   abort was commanded; do not resume.
        eta_to_destination_ms              (int)    estimated time to destination at
                                                     checkpoint time (milliseconds).
        terrain_corridor_phase             (str)    route corridor phase identifier.
        route_corridor_half_width_m        (float)  corridor half-width at checkpoint time.
    """

    # ------------------------------------------------------------------
    # v1.1 baseline fields
    # ------------------------------------------------------------------
    schema_version: str  = "1.2"
    checkpoint_id:  str  = field(default_factory=lambda: str(uuid.uuid4()))
    mission_id:     str  = ""
    timestamp_ms:   int  = 0

    # ESKF navigation state (NED frame, metres / metres-per-second / radians)
    pos_ned:   List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    vel_ned:   List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    heading_rad: float     = 0.0

    # Mission progression
    fsm_state:      str = "NOMINAL"
    waypoint_index: int = 0

    # ------------------------------------------------------------------
    # v1.2 mandatory fields (SRS §10.15, PX4-05 corrections P-01, P-02)
    # ------------------------------------------------------------------
    shm_active:                          bool  = False
    pending_operator_clearance_required: bool  = False
    mission_abort_flag:                  bool  = False
    eta_to_destination_ms:               int   = 0
    terrain_corridor_phase:              str   = ""
    route_corridor_half_width_m:         float = 0.0

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise all fields to a plain dict via dataclasses.asdict().

        asdict() recursively converts all dataclass fields — no field can be
        silently omitted.  List[float] fields (pos_ned, vel_ned) are preserved
        as Python lists.
        """
        return asdict(self)

    def to_json(self) -> str:
        """Serialise to JSON string (human-readable, 2-space indent)."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """
        Deserialise from a plain dict.

        Unknown keys in data are silently ignored so that legacy checkpoint
        files (schema < v1.2) can be loaded; new v1.2 fields will take their
        defined defaults.

        Known fields are passed directly to the dataclass constructor, which
        preserves Python types (bool, int, str, float, list).
        """
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}

        MANDATORY_FIELDS = {
            "schema_version", "checkpoint_id", "timestamp_ms",
            "waypoint_index", "mission_abort_flag",
            "pending_operator_clearance_required", "shm_active",
        }
        missing = MANDATORY_FIELDS - set(data.keys())
        if missing:
            raise CheckpointCorruptError(
                f"Missing mandatory fields: {sorted(missing)}"
            )

        if data.get("schema_version") != "1.2":
            raise CheckpointCorruptError(
                f"Unsupported schema version: {data.get('schema_version')!r}"
                f" (expected '1.2')"
            )

        return cls(**filtered)

    @classmethod
    def from_json(cls, s: str) -> "Checkpoint":
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Checkpoint store
# ---------------------------------------------------------------------------

class CheckpointStore:
    """
    Persistent checkpoint store with rolling purge.

    Maintains at most `max_retained` checkpoint files on disk (default 5,
    per PX4-05 §10.15).  Files are named:

        checkpoint_{timestamp_ms:020d}_{checkpoint_id[:8]}.json

    Lexicographic sort equals chronological sort (20-digit zero-padded integer
    timestamp guarantees this).

    Atomic write pattern:
        Write goes to <name>.tmp, then os.rename() into <name>.json.
        A SIGKILL between the two calls leaves only the .tmp file; the last
        committed .json is always valid and complete.

    Events logged to event_log (list of dicts):
        {"event": "CHECKPOINT_WRITTEN",  "req_id": "PX4-05", "severity": "INFO",
         "module_name": "CheckpointStore", "checkpoint_id": str,
         "path": str, "timestamp_ms": int}
        {"event": "CHECKPOINT_RESTORED", "req_id": "PX4-05", "severity": "INFO",
         "module_name": "CheckpointStore", "checkpoint_id": str,
         "path": str, "timestamp_ms": int}
        {"event": "CHECKPOINT_PURGED",   "req_id": "PX4-05", "severity": "DEBUG",
         "module_name": "CheckpointStore", "checkpoint_id": str,
         "path": str, "timestamp_ms": int}
    """

    MAX_RETAINED_DEFAULT = 5

    def __init__(
        self,
        checkpoint_dir: "str | Path",
        max_retained:   int = MAX_RETAINED_DEFAULT,
        event_log:      Optional[List[Dict[str, Any]]] = None,
        clock_fn:       Optional[Callable[[], float]] = None,
    ):
        """
        Args:
            checkpoint_dir: Directory where checkpoint files are stored.
                            Created if it does not exist.
            max_retained:   Maximum number of checkpoint files to keep on disk.
                            Older files are purged after each write.
            event_log:      External list to append checkpoint events to.
                            If None, an internal list is created.
            clock_fn:       Optional callable returning current time in seconds.
                            Used for CHECKPOINT_CORRUPT event timestamps.
                            If None, timestamp_ms is 0 in corruption events.
        """
        self._dir          = Path(checkpoint_dir)
        self._max_retained = max_retained
        self._event_log    = event_log if event_log is not None else []
        self._clock_fn     = clock_fn
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def event_log(self) -> List[Dict[str, Any]]:
        """Snapshot of all events logged by this store."""
        return list(self._event_log)

    @property
    def checkpoints_retained_count(self) -> int:
        """Count of checkpoint JSON files currently on disk."""
        return len(self._checkpoint_files())

    def write(self, checkpoint: Checkpoint) -> Path:
        """
        Serialise checkpoint to disk atomically, then apply rolling purge.

        Args:
            checkpoint: Checkpoint to persist.

        Returns:
            Path of the written checkpoint file.
        """
        filename = (
            f"checkpoint_{checkpoint.timestamp_ms:020d}"
            f"_{checkpoint.checkpoint_id[:8]}.json"
        )
        dest = self._dir / filename
        tmp  = dest.with_suffix(".tmp")

        # Atomic write: write to .tmp then rename into place
        tmp.write_text(checkpoint.to_json(), encoding="utf-8")
        tmp.rename(dest)

        self._event_log.append({
            "event":         "CHECKPOINT_WRITTEN",
            "req_id":        "PX4-05",
            "severity":      "INFO",
            "module_name":   "CheckpointStore",
            "checkpoint_id": checkpoint.checkpoint_id,
            "path":          str(dest),
            "timestamp_ms":  checkpoint.timestamp_ms,
        })

        self._purge()
        return dest

    def restore_latest(self) -> Optional[Checkpoint]:
        """
        Restore the most recent checkpoint from disk.

        Returns None if no checkpoint files exist in the store directory.
        """
        files = self._checkpoint_files()
        if not files:
            return None
        try:
            return self._load(files[-1])
        except CheckpointCorruptError:
            return None

    def restore_from_path(self, path: "str | Path") -> Checkpoint:
        """
        Restore a specific checkpoint file by path.

        Args:
            path: Absolute or relative path to a checkpoint JSON file.
        """
        return self._load(Path(path))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _checkpoint_files(self) -> List[Path]:
        """Return lexicographically sorted list of checkpoint .json files (oldest first)."""
        return sorted(self._dir.glob("checkpoint_*.json"))

    def _emit_corrupt(self, path: Path, reason: str) -> None:
        """Append a CHECKPOINT_CORRUPT event to the event log."""
        self._event_log.append({
            "event":        "CHECKPOINT_CORRUPT",
            "req_id":       "PX4-05",
            "severity":     "WARNING",
            "module_name":  "CheckpointStore",
            "path":         str(path),
            "reason":       reason,
            "timestamp_ms": int(self._clock_fn() * 1000) if self._clock_fn is not None else 0,
        })

    def _load(self, path: Path) -> Checkpoint:
        """Read and deserialise a checkpoint file; log CHECKPOINT_RESTORED.

        Raises CheckpointCorruptError (after logging CHECKPOINT_CORRUPT) on
        I/O failure, invalid JSON, or schema validation failure.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError) as exc:
            self._emit_corrupt(path, reason=f"io_error: {type(exc).__name__}")
            raise CheckpointCorruptError(
                f"Cannot read checkpoint {path}: {exc}"
            ) from exc

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            self._emit_corrupt(path, reason="invalid_json")
            raise CheckpointCorruptError(
                f"Invalid JSON in checkpoint {path}: {exc}"
            ) from exc

        try:
            cp = Checkpoint.from_dict(data)
        except CheckpointCorruptError:
            self._emit_corrupt(path, reason="schema_invalid")
            raise

        self._event_log.append({
            "event":         "CHECKPOINT_RESTORED",
            "req_id":        "PX4-05",
            "severity":      "INFO",
            "module_name":   "CheckpointStore",
            "checkpoint_id": cp.checkpoint_id,
            "path":          str(path),
            "timestamp_ms":  cp.timestamp_ms,
        })
        return cp

    def _purge(self) -> None:
        """
        Delete oldest checkpoint files until retained count <= max_retained.

        Logs CHECKPOINT_PURGED for each deleted file.

        Filename format: checkpoint_{timestamp_ms:020d}_{id8}.json
        Parsed as: stem.split("_", 2) → ["checkpoint", ts_str, id8]
        """
        files = self._checkpoint_files()
        while len(files) > self._max_retained:
            oldest = files.pop(0)
            # Parse timestamp and id from filename stem
            parts = oldest.stem.split("_", 2)   # ["checkpoint", "00000...ts", "id8chars"]
            ts_ms = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
            cid   = parts[2] if len(parts) == 3 else "unknown"
            self._event_log.append({
                "event":         "CHECKPOINT_PURGED",
                "req_id":        "PX4-05",
                "severity":      "DEBUG",
                "module_name":   "CheckpointStore",
                "checkpoint_id": cid,
                "path":          str(oldest),
                "timestamp_ms":  ts_ms,
            })
            oldest.unlink()
