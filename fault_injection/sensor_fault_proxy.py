"""
BCMP-2 Sensor Fault Proxy.

Wraps sensor output functions.  When no fault is active the proxy is a
transparent pass-through.  When a fault is active it intercepts the
relevant data stream and injects the degraded or null output.

Frozen core contract
--------------------
No frozen core module is imported or modified here.  The proxy sits
between the runner and the frozen stack.  BIM, ESKF, VIOMode, TRNStub
receive the intercepted (or pass-through) values and are unaware of the
proxy layer.

Supported intercepts
--------------------
  FI_GNSS_LOSS       -> returns a denied GNSSMeasurement (all fields null/worst-case)
  FI_VIO_LOSS        -> returns (accepted=False, innov_mag=0.0) to VIOMode.on_vio_update
  FI_RADALT_LOSS     -> returns None from TRNStub.update (no correction available)
  FI_EO_FREEZE       -> returns a stale/frozen EO frame object

Vehicle A note
--------------
Vehicle A does not use this proxy.  Vehicle A has no correction logic to
suppress.  Only Vehicle B (MicroMind stack) passes through the proxy.

JOURNAL
-------
Built: 29 March 2026, micromind-node01.  SB-2 Step 2.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any
import numpy as np

from fault_injection.fault_manager import (
    FaultManager,
    get_fault_manager,
    FI_GNSS_LOSS,
    FI_VIO_LOSS,
    FI_RADALT_LOSS,
    FI_EO_FREEZE,
)


# ---------------------------------------------------------------------------
# Denied GNSS measurement factory
# ---------------------------------------------------------------------------

def _denied_gnss_measurement():
    """
    Return a GNSSMeasurement that forces BIM to RED within its hysteresis window.
    All quality fields are set to worst-case values.
    Importing GNSSMeasurement here avoids pulling it into the proxy module
    scope at import time — keeps the proxy lightweight.
    """
    from core.bim.bim import GNSSMeasurement
    return GNSSMeasurement(
        pdop=99.9,
        cn0_db=0.0,
        tracked_satellites=0,
        gps_position_enu=None,
        glonass_position_enu=None,
        doppler_deviation_ms=999.0,
        pose_innovation_m=999.0,
        ew_jammer_confidence=1.0,
    )


# ---------------------------------------------------------------------------
# Sensor Fault Proxy
# ---------------------------------------------------------------------------

class SensorFaultProxy:
    """
    Intercepts sensor outputs on behalf of Vehicle B.

    All intercept methods follow the same pattern:
      - Check fault_manager for the relevant fault ID
      - If active:  return the injected (degraded/null) value
      - If not:     return the real value unchanged

    The proxy never modifies the real sensor or the fault_manager state.
    """

    def __init__(self, fault_manager: Optional[FaultManager] = None):
        self._fm = fault_manager or get_fault_manager()
        # Stale EO frame cache for EO_FREEZE intercept
        self._last_eo_frame: Optional[Any] = None

    # ── GNSS intercept ────────────────────────────────────────────────────

    def gnss(self, real_measurement) -> Any:
        """
        Intercept GNSS measurement.

        FI_GNSS_LOSS active -> return denied measurement (BIM will go RED)
        Otherwise           -> return real_measurement unchanged
        """
        if self._fm.is_active(FI_GNSS_LOSS):
            return _denied_gnss_measurement()
        return real_measurement

    def gnss_available(self, real_available: bool) -> bool:
        """
        Intercept GNSS availability flag.

        FI_GNSS_LOSS active -> always False
        Otherwise           -> real_available
        """
        if self._fm.is_active(FI_GNSS_LOSS):
            return False
        return real_available

    # ── VIO intercept ─────────────────────────────────────────────────────

    def vio_update(
        self,
        real_accepted:  bool,
        real_innov_mag: float,
    ) -> Tuple[bool, float]:
        """
        Intercept VIOMode.on_vio_update() inputs.

        FI_VIO_LOSS active -> (accepted=False, innov_mag=0.0)
                              VIOMode will transition to OUTAGE after threshold.
        Otherwise          -> (real_accepted, real_innov_mag)
        """
        if self._fm.is_active(FI_VIO_LOSS):
            return False, 0.0
        return real_accepted, real_innov_mag

    def vio_frame_available(self, real_available: bool) -> bool:
        """
        Intercept VIO frame availability check.

        FI_VIO_LOSS active -> always False
        Otherwise          -> real_available
        """
        if self._fm.is_active(FI_VIO_LOSS):
            return False
        return real_available

    # ── RADALT / TRN intercept ────────────────────────────────────────────

    def trn_correction(self, real_correction) -> Any:
        """
        Intercept TRNStub.update() return value.

        FI_RADALT_LOSS active -> None (no correction available, as if
                                 TRN returned None due to low confidence)
        Otherwise             -> real_correction (TRNCorrection or None)
        """
        if self._fm.is_active(FI_RADALT_LOSS):
            return None
        return real_correction

    def radalt_available(self, real_available: bool) -> bool:
        """
        Intercept RADALT availability flag.

        FI_RADALT_LOSS active -> always False
        Otherwise             -> real_available
        """
        if self._fm.is_active(FI_RADALT_LOSS):
            return False
        return real_available

    # ── EO frame intercept ────────────────────────────────────────────────

    def eo_frame(self, real_frame: Any) -> Any:
        """
        Intercept EO/IR frame.

        FI_EO_FREEZE active -> return last cached (stale) frame.
                               If no frame ever received: return None
                               (DMRL will handle stale-frame detection).
        Otherwise           -> update cache and return real_frame.

        This simulates the EO feed freezing at the last received frame,
        which is exactly what happens in hardware when the camera locks up.
        """
        if self._fm.is_active(FI_EO_FREEZE):
            return self._last_eo_frame   # stale — may be None
        # Cache current frame before returning
        self._last_eo_frame = real_frame
        return real_frame

    def eo_available(self, real_available: bool) -> bool:
        """
        FI_EO_FREEZE active -> True (feed is there, but frozen — not absent)
        Otherwise           -> real_available

        Note: EO_FREEZE does not make EO unavailable — the frame is present
        but stale.  DMRL stale-frame detection handles this case.
        """
        return real_available  # freeze doesn't change availability flag

    # ── Convenience: check any sensor degradation ─────────────────────────

    def any_sensor_fault_active(self) -> bool:
        """Return True if any sensor fault is currently active."""
        return any(
            self._fm.is_active(fid)
            for fid in [FI_GNSS_LOSS, FI_VIO_LOSS, FI_RADALT_LOSS, FI_EO_FREEZE]
        )

    def active_sensor_faults(self) -> list:
        """Return list of active sensor fault IDs."""
        return [
            fid for fid in [FI_GNSS_LOSS, FI_VIO_LOSS, FI_RADALT_LOSS, FI_EO_FREEZE]
            if self._fm.is_active(fid)
        ]


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fault_injection.fault_manager import FaultManager

    print("SensorFaultProxy self-verification")
    print("=" * 45)

    fm = FaultManager()
    proxy = SensorFaultProxy(fault_manager=fm)

    # GNSS pass-through
    class _FakeGNSS:
        pdop = 1.2
    real = _FakeGNSS()
    assert proxy.gnss(real) is real, "pass-through should return real"
    assert proxy.gnss_available(True) is True
    print("  GNSS pass-through:         PASS")

    # GNSS fault
    fm.activate(FI_GNSS_LOSS)
    result = proxy.gnss(real)
    assert result is not real, "should be denied measurement"
    assert proxy.gnss_available(True) is False
    fm.clear(FI_GNSS_LOSS)
    print("  GNSS_LOSS intercept:       PASS")

    # VIO pass-through
    acc, innov = proxy.vio_update(True, 0.5)
    assert acc is True and innov == 0.5
    print("  VIO pass-through:          PASS")

    # VIO fault
    fm.activate(FI_VIO_LOSS)
    acc, innov = proxy.vio_update(True, 0.5)
    assert acc is False and innov == 0.0
    assert proxy.vio_frame_available(True) is False
    fm.clear(FI_VIO_LOSS)
    print("  VIO_LOSS intercept:        PASS")

    # RADALT / TRN pass-through
    class _FakeTRN:
        north_offset_m = 5.0
    trn = _FakeTRN()
    assert proxy.trn_correction(trn) is trn
    assert proxy.radalt_available(True) is True
    print("  RADALT pass-through:       PASS")

    # RADALT fault
    fm.activate(FI_RADALT_LOSS)
    assert proxy.trn_correction(trn) is None
    assert proxy.radalt_available(True) is False
    fm.clear(FI_RADALT_LOSS)
    print("  RADALT_LOSS intercept:     PASS")

    # EO freeze — first frame cached, then stale
    class _FakeFrame:
        frame_id = 1
    frame1 = _FakeFrame()
    frame2 = _FakeFrame(); frame2.frame_id = 2

    out1 = proxy.eo_frame(frame1)
    assert out1 is frame1                     # pass-through, caches frame1
    fm.activate(FI_EO_FREEZE)
    out2 = proxy.eo_frame(frame2)
    assert out2 is frame1                     # returns stale frame1, not frame2
    fm.clear(FI_EO_FREEZE)
    out3 = proxy.eo_frame(frame2)
    assert out3 is frame2                     # pass-through resumes
    print("  EO_FREEZE intercept:       PASS")

    # any_sensor_fault_active
    assert not proxy.any_sensor_fault_active()
    fm.activate(FI_GNSS_LOSS)
    assert proxy.any_sensor_fault_active()
    assert FI_GNSS_LOSS in proxy.active_sensor_faults()
    fm.clear_all()
    print("  any_sensor_fault_active:   PASS")

    print()
    print("All checks passed.")
