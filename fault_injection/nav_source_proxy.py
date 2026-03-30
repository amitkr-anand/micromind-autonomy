"""
BCMP-2 Navigation Source Proxy.

Wraps navigation source selection and correction injection for Vehicle B.
Intercepts TRN correction windows, VIO update injection, and IMU timestamp
quality — without touching any frozen core module.

Frozen core contract
--------------------
No frozen core module is modified.  The proxy wraps the *call* to
TRNStub.update() and the *result* from it, and wraps VIOMode state
checks.  The frozen modules themselves are unaware.

Supported intercepts
--------------------
  FI_TERRAIN_CONF_DROP  -> suppresses TRN correction (returns None from
                           nav_source_proxy.trn_update()) even when
                           TRNStub would normally return a correction.
                           Simulates terrain below NCC threshold.

  FI_VIO_LOSS           -> suppresses VIO source availability flag.
                           (sensor_fault_proxy already intercepts the
                           frame; this proxy handles the nav-source-level
                           selection logic.)

  FI_IMU_JITTER         -> injects a timestamp perturbation into the
                           dt value passed to VIOMode.tick(), simulating
                           non-monotonic or jittered IMU timestamps.
                           IFM-01 guard in the integration layer will
                           fire if used in full SITL mode.

JOURNAL
-------
Built: 29 March 2026, micromind-node01.  SB-2 Step 3.
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional

from fault_injection.fault_manager import (
    FaultManager,
    get_fault_manager,
    FI_TERRAIN_CONF_DROP,
    FI_VIO_LOSS,
    FI_IMU_JITTER,
)


# ---------------------------------------------------------------------------
# Nav Source Proxy
# ---------------------------------------------------------------------------

class NavSourceProxy:
    """
    Wraps navigation source selection for Vehicle B.

    All intercept methods follow the same pattern:
      - Check fault_manager for the relevant fault ID
      - If active:  return the injected/suppressed value
      - If not:     return the real value unchanged

    The proxy never modifies TRNStub, VIOMode, or any frozen module state.
    """

    def __init__(
        self,
        fault_manager: Optional[FaultManager] = None,
        imu_jitter_max_ms: float = 2.0,
        terrain_conf_floor: float = 0.25,
        seed: int = 0,
    ):
        self._fm                = fault_manager or get_fault_manager()
        self._imu_jitter_max_ms = imu_jitter_max_ms   # max timestamp perturbation
        self._terrain_conf_floor = terrain_conf_floor  # conf level when dropped
        self._rng               = random.Random(seed)

    # ── TRN correction intercept ──────────────────────────────────────────

    def trn_update(
        self,
        trn_stub,
        ins_north_m:     float,
        ins_east_m:      float,
        true_north_m:    float,
        true_east_m:     float,
        ground_track_m:  float,
        timestamp_s:     float = 0.0,
    ) -> Any:
        """
        Wrap TRNStub.update().

        FI_TERRAIN_CONF_DROP active -> returns None (TRN suppressed).
                                       Simulates terrain confidence below
                                       NCC threshold (σ_terrain < 40 m).
        Otherwise                   -> calls trn_stub.update() normally
                                       and returns its result.

        Parameters mirror TRNStub.update() exactly so this can be
        substituted transparently in the runner loop.
        """
        if self._fm.is_active(FI_TERRAIN_CONF_DROP):
            return None  # TRN suppressed — no correction this tick

        return trn_stub.update(
            ins_north_m    = ins_north_m,
            ins_east_m     = ins_east_m,
            true_north_m   = true_north_m,
            true_east_m    = true_east_m,
            ground_track_m = ground_track_m,
            timestamp_s    = timestamp_s,
        )

    def terrain_confidence(self, real_confidence: float) -> float:
        """
        Intercept terrain confidence value.

        FI_TERRAIN_CONF_DROP active -> return _terrain_conf_floor
                                       (below NCC threshold, TRN will suppress)
        Otherwise                   -> real_confidence
        """
        if self._fm.is_active(FI_TERRAIN_CONF_DROP):
            return self._terrain_conf_floor
        return real_confidence

    # ── VIO source selection intercept ────────────────────────────────────

    def vio_source_available(self, real_available: bool) -> bool:
        """
        Intercept VIO navigation source availability.

        FI_VIO_LOSS active -> False (nav source unavailable for selection)
        Otherwise          -> real_available

        Note: sensor_fault_proxy.vio_frame_available() intercepts at the
        frame level.  This proxy intercepts at the nav-source selection
        level.  Both are needed: the frame intercept prevents VIOMode from
        receiving updates; this intercept prevents the runner from selecting
        VIO as the active nav source.
        """
        if self._fm.is_active(FI_VIO_LOSS):
            return False
        return real_available

    # ── IMU jitter intercept ──────────────────────────────────────────────

    def dt_ticked(self, real_dt: float) -> float:
        """
        Intercept the dt value passed to VIOMode.tick().

        FI_IMU_JITTER active -> add a random perturbation within
                                ±_imu_jitter_max_ms milliseconds.
                                Severity scales the jitter amplitude.
        Otherwise            -> real_dt unchanged.

        In full SITL mode with the integration layer active, large jitter
        can trigger IFM-01 (timestamp monotonicity guard).  In SIL mode
        this affects drift envelope growth only.
        """
        if self._fm.is_active(FI_IMU_JITTER):
            severity    = self._fm.severity(FI_IMU_JITTER)
            max_jitter  = (self._imu_jitter_max_ms * 1e-3) * severity
            perturbation = self._rng.uniform(-max_jitter, max_jitter)
            # Never return negative dt (that would be non-physical)
            return max(1e-6, real_dt + perturbation)
        return real_dt

    # ── Navigation mode override ──────────────────────────────────────────

    def effective_nav_sources(self, gnss_ok: bool, vio_ok: bool,
                               trn_ok: bool) -> tuple:
        """
        Return effective navigation source availability after proxy intercepts.

        Convenience method for the runner to check all nav sources
        simultaneously without calling each intercept separately.

        Returns (gnss_ok, vio_ok, trn_ok) after applying active faults.
        """
        if self._fm.is_active(FI_VIO_LOSS):
            vio_ok = False
        if self._fm.is_active(FI_TERRAIN_CONF_DROP):
            trn_ok = False
        # GNSS is handled by sensor_fault_proxy.gnss_available()
        # but we include it here for completeness
        return gnss_ok, vio_ok, trn_ok

    # ── Convenience ───────────────────────────────────────────────────────

    def any_nav_fault_active(self) -> bool:
        """Return True if any nav source fault is currently active."""
        return any(
            self._fm.is_active(fid)
            for fid in [FI_TERRAIN_CONF_DROP, FI_VIO_LOSS, FI_IMU_JITTER]
        )


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fault_injection.fault_manager import FaultManager

    print("NavSourceProxy self-verification")
    print("=" * 45)

    fm = FaultManager()
    proxy = NavSourceProxy(fault_manager=fm, seed=42)

    # ── TRN pass-through ──────────────────────────────────────────────────
    class _FakeTRNStub:
        def update(self, **kwargs):
            class _R:
                north_offset_m = 5.0
                confidence = 0.8
            return _R()

    trn = _FakeTRNStub()
    result = proxy.trn_update(trn, 1000, 500, 1005, 505, 2000, 10.0)
    assert result is not None and result.north_offset_m == 5.0
    assert proxy.terrain_confidence(0.9) == 0.9
    print("  TRN pass-through:          PASS")

    # ── TRN suppressed ────────────────────────────────────────────────────
    fm.activate(FI_TERRAIN_CONF_DROP)
    result = proxy.trn_update(trn, 1000, 500, 1005, 505, 2000, 10.0)
    assert result is None, "TRN should be suppressed"
    assert proxy.terrain_confidence(0.9) == proxy._terrain_conf_floor
    fm.clear(FI_TERRAIN_CONF_DROP)
    print("  TERRAIN_CONF_DROP:         PASS")

    # ── VIO source intercept ──────────────────────────────────────────────
    assert proxy.vio_source_available(True) is True
    fm.activate(FI_VIO_LOSS)
    assert proxy.vio_source_available(True) is False
    fm.clear(FI_VIO_LOSS)
    print("  VIO_LOSS nav source:       PASS")

    # ── IMU jitter pass-through ───────────────────────────────────────────
    dt_real = 0.005
    dt_out  = proxy.dt_ticked(dt_real)
    assert dt_out == dt_real
    print("  IMU jitter pass-through:   PASS")

    # ── IMU jitter active ─────────────────────────────────────────────────
    fm.activate(FI_IMU_JITTER, severity=1.0)
    jittered = [proxy.dt_ticked(dt_real) for _ in range(100)]
    assert all(j > 0 for j in jittered), "dt must always be positive"
    assert any(j != dt_real for j in jittered), "some jitter expected"
    max_dev = max(abs(j - dt_real) for j in jittered)
    assert max_dev <= 0.002 + 1e-9, f"jitter exceeded 2ms: {max_dev*1000:.3f}ms"
    fm.clear(FI_IMU_JITTER)
    print("  IMU_JITTER (100 samples):  PASS")

    # ── effective_nav_sources ─────────────────────────────────────────────
    g, v, t = proxy.effective_nav_sources(True, True, True)
    assert (g, v, t) == (True, True, True)
    fm.activate(FI_VIO_LOSS)
    fm.activate(FI_TERRAIN_CONF_DROP)
    g, v, t = proxy.effective_nav_sources(True, True, True)
    assert v is False and t is False and g is True
    fm.clear_all()
    print("  effective_nav_sources:     PASS")

    # ── any_nav_fault_active ──────────────────────────────────────────────
    assert not proxy.any_nav_fault_active()
    fm.activate(FI_IMU_JITTER)
    assert proxy.any_nav_fault_active()
    fm.clear_all()
    print("  any_nav_fault_active:      PASS")

    print()
    print("All checks passed.")
