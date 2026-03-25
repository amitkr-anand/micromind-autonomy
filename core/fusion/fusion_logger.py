"""
core/fusion/fusion_logger.py
============================
Fusion observability logger — Schema 08.1 (S-NEP-08).

SCHEMA HISTORY:
    06e.3  S-NEP-06  outage/recovery experiments
    08.1   S-NEP-08  VIO navigation mode, drift envelope, spike alerting
                     vel_err_m_s demoted to diagnostic (default off)

BACKWARD COMPATIBILITY:
    Old runners (run_04b, run_05, run_06_*) produce schema 06e.3.
    This module produces schema 08.1 when called with the new fields.
    Old logs are NOT affected.

SEPARATION CONTRACT:
    fusion_logger reads VIONavigationMode.current_mode as a string only.
    No import of ESKF internals. No estimator calls.

LOG FIELDS per VIO_UPDATE entry (schema 08.1):
    t                       float   timestamp (seconds)
    type                    str     "VIO_UPDATE"
    nis                     float   NIS scalar
    innov_mag               float   ‖innovation‖ metres
    trace_P                 float   trace(P[0:3,0:3])
    error_m                 float|None  ‖state.p - p_GT‖ if GT available
    ba_est                  list[float] accelerometer bias estimate
    vio_mode                str     "NOMINAL" | "OUTAGE" | "RESUMPTION"
    dt_since_vio            float   seconds since last accepted update
    drift_envelope_m        float|None  None outside OUTAGE; conservative
                                    over-estimate only (NOT a hard bound)
    innovation_spike_alert  bool    True on first post-outage update if
                                    innov_mag > VIO_INNOVATION_SPIKE_THRESHOLD_M
    vel_err_diagnostic      float|None  DIAGNOSTIC ONLY. Emitted only when
                                    emit_vel_diagnostic=True (default False).
                                    D-05: not a valid operational signal.

LOG FIELDS per PROPAGATE entry (schema 08.1):
    t               float   timestamp
    type            str     "PROPAGATE"
    trace_P         float
    error_m         float|None
    dt_since_vio    float   seconds since last accepted VIO update
    vio_mode        str     current mode at propagation time

SUMMARY FIELDS (schema 08.1 additions):
    n_outage_events         int     NOMINAL→OUTAGE transition count
    n_spike_alerts          int     innovation_spike_alert=True count
    max_dt_since_vio        float   maximum outage duration (seconds)
    max_drift_envelope_m    float   maximum drift_envelope_m reached
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any


# Schema version string — increment on any breaking change
SCHEMA_VERSION = "08.1"


class FusionLogger:
    """
    Collects per-step log entries and writes a structured JSON log on close.

    Schema 08.1: VIO navigation mode fields added, vel_err_m_s demoted.
    """

    def __init__(
        self,
        log_path: str | Path | None = None,
        label: str = "",
        emit_vel_diagnostic: bool = False,
    ) -> None:
        """
        Args:
            log_path:            Output path for the JSON log file.
            label:               Run label (appears in summary).
            emit_vel_diagnostic: If True, vel_err_diagnostic is included
                                 in VIO_UPDATE entries. Default False per D-05.
        """
        import tempfile as _tf
        self._path = Path(log_path) if log_path is not None else Path(_tf.mktemp(suffix=".json"))
        self._label  = label
        self._emit_vel_diag = emit_vel_diagnostic
        self._entries: list[dict[str, Any]] = []

        # Run-level summary accumulators
        self._n_vio_updates  = 0
        self._n_propagations = 0
        self._n_rejections   = 0
        self._health_nan     = False
        self._health_div     = False

        # Mode summary (populated from VIONavigationMode at close)
        self._n_outage_events      = 0
        self._n_spike_alerts       = 0
        self._max_dt_since_vio     = 0.0
        self._max_drift_envelope_m = 0.0

    # ── Entry logging methods ─────────────────────────────────────────────────

    def log_vio_update(
        self,
        *,
        t: float,
        nis: float,
        innov_mag: float,
        trace_P: float,
        vio_mode: str,
        dt_since_vio: float,
        drift_envelope_m: float | None,
        innovation_spike_alert: bool,
        error_m: float | None = None,
        ba_est: list[float] | None = None,
        vel_err_diagnostic: float | None = None,
    ) -> None:
        """Log one accepted VIO update entry (schema 08.1)."""
        entry: dict[str, Any] = {
            "t":                      round(t, 4),
            "type":                   "VIO_UPDATE",
            "nis":                    round(nis, 8),
            "innov_mag":              round(innov_mag, 6),
            "trace_P":                round(trace_P, 8),
            "error_m":                round(error_m, 6) if error_m is not None else None,
            "ba_est":                 [round(v, 6) for v in ba_est] if ba_est else None,
            "vio_mode":               vio_mode,
            "dt_since_vio":           round(dt_since_vio, 4),
            "drift_envelope_m":       round(drift_envelope_m, 4)
                                      if drift_envelope_m is not None else None,
            "innovation_spike_alert": bool(innovation_spike_alert),
        }
        # D-05: vel_err emitted only when explicitly requested
        if self._emit_vel_diag:
            entry["vel_err_diagnostic"] = (
                round(vel_err_diagnostic, 6)
                if vel_err_diagnostic is not None else None
            )
        self._entries.append(entry)
        self._n_vio_updates += 1
        if innovation_spike_alert:
            self._n_spike_alerts += 1

    def log_propagate(
        self,
        *,
        t: float,
        trace_P: float,
        vio_mode: str,
        dt_since_vio: float,
        error_m: float | None = None,
    ) -> None:
        """Log one IMU propagation step (schema 08.1, every N steps)."""
        self._entries.append({
            "t":            round(t, 4),
            "type":         "PROPAGATE",
            "trace_P":      round(trace_P, 8),
            "error_m":      round(error_m, 6) if error_m is not None else None,
            "vio_mode":     vio_mode,
            "dt_since_vio": round(dt_since_vio, 4),
        })
        self._n_propagations += 1

    def log_rejection(self, *, t: float, nis: float, innov_mag: float) -> None:
        """Log a rejected VIO measurement."""
        self._entries.append({
            "t":        round(t, 4),
            "type":     "REJECTION",
            "nis":      round(nis, 8),
            "innov_mag": round(innov_mag, 6),
        })
        self._n_rejections += 1

    def flag_nan(self) -> None:
        """Mark this run as having encountered a NaN in covariance."""
        self._health_nan = True

    def flag_divergence(self) -> None:
        """Mark this run as having diverged (trace_P > threshold)."""
        self._health_div = True

    # ── Close and write ───────────────────────────────────────────────────────

    def close(
        self,
        vio_nav_mode=None,  # VIONavigationMode instance or None
        extra_summary: dict[str, Any] | None = None,
    ) -> None:
        """
        Finalise and write the JSON log.

        Args:
            vio_nav_mode:  VIONavigationMode instance. If provided, its
                           summary statistics are incorporated. Pass None
                           for runs that do not use schema 08.1 mode tracking.
            extra_summary: Additional key-value pairs for the summary block.
        """
        if vio_nav_mode is not None:
            self._n_outage_events      = vio_nav_mode.n_outage_events
            self._n_spike_alerts       = vio_nav_mode.n_spike_alerts
            self._max_dt_since_vio     = vio_nav_mode.max_dt_since_vio
            self._max_drift_envelope_m = vio_nav_mode.max_drift_envelope_m

        summary: dict[str, Any] = {
            "schema":          SCHEMA_VERSION,
            "label":           self._label,
            "health":          {
                "nan": self._health_nan,
                "div": self._health_div,
            },
            "n_vio_updates":   self._n_vio_updates,
            "n_propagations":  self._n_propagations,
            "n_rejections":    self._n_rejections,
            # Schema 08.1 mode fields
            "n_outage_events":      self._n_outage_events,
            "n_spike_alerts":       self._n_spike_alerts,
            "max_dt_since_vio":     round(self._max_dt_since_vio, 3),
            "max_drift_envelope_m": round(self._max_drift_envelope_m, 4),
            # D-05 notice
            "vel_err_note": (
                "vel_err_m_s removed per D-05 (S-NEP-07). "
                "Use emit_vel_diagnostic=True for diagnostic access."
            ),
            # Drift envelope notice
            "drift_envelope_note": (
                "drift_envelope_m is a conservative confidence degradation "
                "signal, NOT a guaranteed position error bound (S-NEP-07 L-05)."
            ),
        }
        if extra_summary:
            summary.update(extra_summary)

        log = {"summary": summary, "time_series": self._entries}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(log, f, indent=2)
