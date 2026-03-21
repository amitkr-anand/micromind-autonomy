# core/fusion/fusion_logger.py
# MicroMind — S-NEP-04 Step 04-A
#
# Records all 8 observability signals per integration run.
# Writes structured JSON on close().
#
# Design rule: NO ESKF instrumentation.
# All logging happens in the fusion layer wrapper — ESKF internals are
# never modified to emit signals.

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


class FusionLogger:
    """
    Observability wrapper for the MicroMind VIO fusion layer.

    Records all 8 observability signals per integration run and writes a
    structured JSON log on close().  Must be instantiated once per run and
    closed explicitly when the run ends.

    Signals
    -------
    O-01  ESKF update sequence — type (PROPAGATE / VIO_UPDATE / REJECTION),
          timestamp, and NIS per event.
    O-02  Fusion-layer NIS time series — (timestamp, NIS) per fused update.
    O-03  ESKF position covariance trace — (timestamp, trace) per propagation
          step.  Caller must supply; not read from ESKF internals.
    O-04  Fused ATE vs ground truth — recorded post-run via log_ate().
    O-05  R matrix log — trace and minimum eigenvalue per measurement.
    O-06  /odomimu message rate at fusion node input — Hz.
    O-07  IFM event log — always present, even if zero events.
    O-08  Camera-IMU time offset convergence — (timestamp, offset_s) series.

    Usage
    -----
    logger = FusionLogger()
    logger.log_update(t, "VIO_UPDATE", nis=1.23)
    logger.log_cov_trace(t, trace_value)
    ...
    logger.close(run_id="mh01_run1", output_path="/tmp/fusion_log.json")
    """

    # Valid update type tokens for O-01
    _VALID_UPDATE_TYPES = frozenset({"PROPAGATE", "VIO_UPDATE", "REJECTION"})

    def __init__(self):
        self._o01: list[dict] = []   # update sequence
        self._o02: list[dict] = []   # NIS time series
        self._o03: list[dict] = []   # covariance trace
        self._o04: Optional[dict] = None  # ATE (post-run)
        self._o05: list[dict] = []   # R matrix log
        self._o06: list[dict] = []   # message rate
        self._o07: list[dict] = []   # IFM events
        self._o08: list[dict] = []   # time offset convergence
        self._created_at = time.time()

    # ------------------------------------------------------------------
    # O-01 / O-02: update sequence + NIS
    # ------------------------------------------------------------------

    def log_update(
        self,
        timestamp: float,
        update_type: str,
        nis: Optional[float] = None,
    ) -> None:
        """
        Record one ESKF update event (O-01).

        If update_type is "VIO_UPDATE" and nis is provided, the NIS value
        is also appended to the O-02 time series.

        Parameters
        ----------
        timestamp   : mission time in seconds
        update_type : one of "PROPAGATE", "VIO_UPDATE", "REJECTION"
        nis         : Normalised Innovation Squared (optional; required for
                      VIO_UPDATE to populate O-02)
        """
        if update_type not in self._VALID_UPDATE_TYPES:
            raise ValueError(
                f"update_type must be one of {sorted(self._VALID_UPDATE_TYPES)}, "
                f"got '{update_type}'"
            )

        entry: dict = {"t": float(timestamp), "type": update_type}
        if nis is not None:
            entry["nis"] = float(nis)
        self._o01.append(entry)

        # O-02: NIS time series — only for successful fusions
        if update_type == "VIO_UPDATE" and nis is not None:
            self._o02.append({"t": float(timestamp), "nis": float(nis)})

    # ------------------------------------------------------------------
    # O-03: position covariance trace
    # ------------------------------------------------------------------

    def log_cov_trace(self, timestamp: float, trace_value: float) -> None:
        """
        Record the ESKF position covariance trace at a given timestep (O-03).

        Caller reads trace(P[0:3, 0:3]) from the ESKF after propagate() or
        inject() and passes it here.  No ESKF instrumentation required.
        """
        self._o03.append({"t": float(timestamp), "trace_P_pos": float(trace_value)})

    # ------------------------------------------------------------------
    # O-04: ATE (post-run)
    # ------------------------------------------------------------------

    def log_ate(
        self,
        ate_rmse_m: float,
        standalone_ate_rmse_m: Optional[float] = None,
        n_samples: Optional[int] = None,
    ) -> None:
        """
        Record fused ATE vs ground truth after the run completes (O-04).

        Parameters
        ----------
        ate_rmse_m            : fused trajectory ATE RMSE (m)
        standalone_ate_rmse_m : standalone VIO ATE for comparison (optional)
        n_samples             : number of aligned pose pairs used
        """
        self._o04 = {
            "ate_rmse_m": float(ate_rmse_m),
            "standalone_ate_rmse_m": (
                float(standalone_ate_rmse_m) if standalone_ate_rmse_m is not None else None
            ),
            "n_samples": n_samples,
        }

    # ------------------------------------------------------------------
    # O-05: R matrix log
    # ------------------------------------------------------------------

    def log_r_matrix(self, timestamp: float, r_matrix: np.ndarray) -> None:
        """
        Record the VIO measurement noise matrix R used in a fusion update (O-05).

        Parameters
        ----------
        timestamp : mission time in seconds
        r_matrix  : (3, 3) position noise covariance matrix (m²)
        """
        r = np.asarray(r_matrix, dtype=np.float64).reshape(3, 3)
        eigenvalues = np.linalg.eigvalsh(r)
        self._o05.append(
            {
                "t": float(timestamp),
                "trace_R": float(np.trace(r)),
                "min_eigenvalue": float(eigenvalues.min()),
            }
        )

    # ------------------------------------------------------------------
    # O-06: message rate
    # ------------------------------------------------------------------

    def log_message_rate(self, timestamp: float, rate_hz: float) -> None:
        """
        Record the /odomimu message rate at the fusion node input (O-06).

        Parameters
        ----------
        timestamp : mission time in seconds
        rate_hz   : measured message rate (Hz)
        """
        self._o06.append({"t": float(timestamp), "rate_hz": float(rate_hz)})

    # ------------------------------------------------------------------
    # O-07: IFM event log
    # ------------------------------------------------------------------

    def log_ifm_event(
        self,
        ifm_id: str,
        timestamp: float,
        details: Optional[dict] = None,
    ) -> None:
        """
        Record an Integration Failure Mode event (O-07).

        IFM IDs: IFM-01 through IFM-06 (see NEP_SPRINT_STATUS.md).
        O-07 is always present in the log, even when no events occurred.

        Parameters
        ----------
        ifm_id    : e.g. "IFM-04"
        timestamp : mission time in seconds
        details   : arbitrary key-value dict for extra context
        """
        self._o07.append(
            {
                "ifm_id": ifm_id,
                "t": float(timestamp),
                "details": details or {},
            }
        )

    # ------------------------------------------------------------------
    # O-08: camera-IMU time offset convergence
    # ------------------------------------------------------------------

    def log_time_offset(self, timestamp: float, offset_s: float) -> None:
        """
        Record the OpenVINS camera-IMU time offset estimate (O-08).

        OpenVINS online-calibrates the temporal offset between camera and
        IMU.  Logging the convergence series confirms the offset has
        stabilised before trusting fusion results.

        Parameters
        ----------
        timestamp : mission time in seconds
        offset_s  : estimated camera-IMU time offset (seconds)
        """
        self._o08.append({"t": float(timestamp), "offset_s": float(offset_s)})

    # ------------------------------------------------------------------
    # close() — write structured JSON
    # ------------------------------------------------------------------

    def close(self, run_id: str, output_path) -> dict:
        """
        Finalise the log and write a structured JSON file.

        Parameters
        ----------
        run_id      : human-readable identifier for this run
        output_path : path to write the JSON log (str or Path)

        Returns
        -------
        summary : dict with high-level statistics for immediate inspection
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Compute summary statistics
        nis_values = [e["nis"] for e in self._o02 if "nis" in e]
        nis_mean = float(np.mean(nis_values)) if nis_values else None
        nis_std = float(np.std(nis_values)) if len(nis_values) > 1 else None

        n_vio_updates = sum(1 for e in self._o01 if e["type"] == "VIO_UPDATE")
        n_rejections = sum(1 for e in self._o01 if e["type"] == "REJECTION")
        n_propagations = sum(1 for e in self._o01 if e["type"] == "PROPAGATE")

        summary = {
            "run_id": run_id,
            "created_at_unix": self._created_at,
            "closed_at_unix": time.time(),
            "n_propagations": n_propagations,
            "n_vio_updates": n_vio_updates,
            "n_rejections": n_rejections,
            "nis_mean": nis_mean,
            "nis_std": nis_std,
            "n_ifm_events": len(self._o07),
            "ate": self._o04,
        }

        log_doc = {
            "schema_version": "04-A.1",
            "run_id": run_id,
            "summary": summary,
            "O-01_update_sequence": self._o01,
            "O-02_nis_time_series": self._o02,
            "O-03_cov_trace": self._o03,
            "O-04_ate": self._o04,
            "O-05_r_matrix": self._o05,
            "O-06_message_rate": self._o06,
            "O-07_ifm_events": self._o07,
            "O-08_time_offset": self._o08,
        }

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(log_doc, fh, indent=2)

        return summary
