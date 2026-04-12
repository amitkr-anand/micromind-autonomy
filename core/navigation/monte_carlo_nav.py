"""
core/navigation/monte_carlo_nav.py
MicroMind / NanoCorteX — Monte Carlo Navigation Performance Evaluator

NAV-02: Monte Carlo navigation performance evaluation over real terrain.

Methodology: identical to BCMP-2 AD-16 (N=300 seeds, P5 floor, P99 ceiling).
Applied here to real-terrain navigation (TRN + VIO correction stack) rather
than BCMP-2 dual-track demonstration.

Drift model
-----------
Position error propagates as a 2D random walk per IMU axis (north, east):
    σ_axis(t) = DRIFT_PSD × sqrt(t)    [m, t in seconds]

where DRIFT_PSD = 1.5 m/√s (consistent with Gate 3 50km Shimla corridor).
This encodes the combined effect of STIM300 ARW (0.15°/√hr), gyro bias
instability, and velocity random walk at AVP-02 cruise (100 km/h).

GNSS phase (0 → gnss_denial_start_km): position error held near zero
(GNSS correction noise σ_gnss = 5 m per axis, 1-sigma).

TRN correction (correction_mode='trn_only' or 'vio_plus_trn'):
    At each trn_interval_m fix opportunity, if terrain suitability is
    ACCEPT or CAUTION, position error is reset to N(0, σ_trn) per axis.
    σ_trn = 25.0 m (phase correlation residual accuracy, 1-sigma).

Vectorised: all N seeds run simultaneously using numpy broadcasting.
N=300 over 55km completes in < 1 second on any modern CPU.

References
----------
    SRS v1.3 NAV-02
    Part Two V7.2 §2.2, §2.3
    AD-16: Monte Carlo drift envelope methodology (29 March 2026)
    Gate 3 technical notes: navigation_manager_TECHNICAL_NOTES.md
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.navigation.corridors import MissionCorridor


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_DRIFT_PSD_M_SQRTS:   float = 1.5     # m/√s per axis — matched to Gate 3
_SPEED_MS:            float = 100.0 / 3.6   # AVP-02 cruise: 100 km/h → 27.78 m/s
_SIGMA_GNSS_M:        float = 5.0     # GNSS correction noise 1-sigma per axis (m)
_SIGMA_TRN_M:         float = 25.0    # TRN residual accuracy 1-sigma per axis (m)
_STEP_KM:             float = 0.1     # Simulation step size (100 m)
_DEFAULT_CHECKPOINTS: List[float] = [10.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """
    Output of MonteCarloNavEvaluator.run().

    All drift arrays are indexed to match checkpoints_km.
    Units: metres (2D horizontal drift, Euclidean norm of north/east errors).
    """
    correction_mode:            str              # 'none' | 'trn_only' | 'vio_plus_trn'
    n_seeds:                    int
    checkpoints_km:             List[float]
    p5_drift_m:                 List[float]
    p50_drift_m:                List[float]
    p99_drift_m:                List[float]
    mean_drift_m:               List[float]
    corrections_accepted_mean:  float            # mean corrections accepted per seed
    corrections_suppressed_mean: float           # mean corrections suppressed per seed
    fix_eligibility:            List[bool]       # terrain eligibility at each fix location


@dataclass
class _FixLocation:
    """Precomputed TRN fix location with terrain eligibility."""
    km:         float
    lat:        float
    lon:        float
    eligible:   bool   # ACCEPT or CAUTION → True, SUPPRESS / out-of-bounds → False


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class MonteCarloNavEvaluator:
    """
    NAV-02: Monte Carlo navigation performance evaluator.

    Runs N independent navigation simulations over a MissionCorridor with
    randomised IMU noise seeds. Produces P5/P50/P99 drift envelopes at
    specified km checkpoints.

    Methodology: AD-16 (29 March 2026).
    N=300, P5 floor, P99 ceiling.

    Parameters
    ----------
    corridor        : MissionCorridor definition
    dem_loader      : DEMLoader instance covering the corridor terrain_dir
    n_seeds         : Number of Monte Carlo seeds (default 300; use 10 for CI)
    checkpoint_km   : Mission km values at which to record drift statistics.
                      Clipped to [0, corridor.total_distance_km]. Defaults to
                      [10, 30, 60, 90, 120, 150, 180].
    trn_interval_m  : Minimum metres between TRN correction attempts.
                      Must match NavigationManager trn_interval_m parameter.
                      Default 5000.0 m.
    master_seed     : numpy RNG seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        corridor:       MissionCorridor,
        dem_loader,                         # DEMLoader — avoid circular import
        n_seeds:        int   = 300,
        checkpoint_km:  Optional[List[float]] = None,
        trn_interval_m: float = 5000.0,
        master_seed:    int   = 42,
    ) -> None:
        self._corridor       = corridor
        self._dem            = dem_loader
        self._n_seeds        = n_seeds
        self._trn_interval_m = trn_interval_m
        self._master_seed    = master_seed

        # Clip checkpoints to corridor length
        total = corridor.total_distance_km
        raw_cps = checkpoint_km if checkpoint_km is not None else _DEFAULT_CHECKPOINTS
        self._checkpoints = sorted(
            cp for cp in raw_cps if 0.0 < cp <= total
        )
        if not self._checkpoints:
            self._checkpoints = [total]

        # Precompute TRN fix locations
        self._fix_locations = self._precompute_fix_locations()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, correction_mode: str = "trn_only") -> MonteCarloResult:
        """
        Run N_seeds Monte Carlo simulations.

        Parameters
        ----------
        correction_mode : 'none'         — INS-only, no TRN/VIO corrections
                          'trn_only'     — TRN phase-correlation corrections only
                          'vio_plus_trn' — VIO + TRN (VIO treated as continuous
                                           low-level drift reduction; modelled as
                                           30% reduction in DRIFT_PSD)

        Returns
        -------
        MonteCarloResult
        """
        if correction_mode not in ("none", "trn_only", "vio_plus_trn"):
            raise ValueError(
                f"correction_mode must be 'none', 'trn_only', or 'vio_plus_trn'; "
                f"got {correction_mode!r}"
            )

        rng = np.random.default_rng(self._master_seed)
        n   = self._n_seeds
        corridor = self._corridor

        # Effective drift PSD — VIO mode reduces dead-reckoning drift
        drift_psd = _DRIFT_PSD_M_SQRTS
        if correction_mode == "vio_plus_trn":
            drift_psd *= 0.70  # VIO reduces heading/velocity drift by ~30%

        step_km = _STEP_KM
        step_m  = step_km * 1000.0
        dt_s    = step_m / _SPEED_MS
        drift_std_per_step = drift_psd * math.sqrt(dt_s)

        total_km = corridor.total_distance_km
        trn_interval_km = self._trn_interval_m / 1000.0

        # Sort fix locations by km
        fix_kms     = [f.km for f in self._fix_locations]
        fix_eligible = [f.eligible for f in self._fix_locations]

        # State: per-seed north/east errors
        err_n = np.zeros(n, dtype=np.float64)
        err_e = np.zeros(n, dtype=np.float64)

        # Checkpoint accumulator: km → list of per-seed 2D drift values
        checkpoint_set   = set(self._checkpoints)
        recorded:        dict = {cp: None for cp in self._checkpoints}

        # TRN correction counters
        corrections_accepted_per_seed  = np.zeros(n, dtype=np.int32)
        corrections_suppressed_per_seed = np.zeros(n, dtype=np.int32)

        # Walk corridor
        current_km = 0.0
        last_trn_km = 0.0
        fix_idx = 0  # pointer into fix_locations

        n_steps = int(round(total_km / step_km)) + 1

        for step in range(n_steps):
            current_km = step * step_km
            if current_km > total_km:
                current_km = total_km

            # ── GNSS phase: clamp errors to GNSS accuracy ─────────────────
            if current_km < corridor.gnss_denial_start_km:
                err_n = rng.standard_normal(n) * _SIGMA_GNSS_M
                err_e = rng.standard_normal(n) * _SIGMA_GNSS_M

            else:
                # ── INS propagation ───────────────────────────────────────
                err_n += rng.standard_normal(n) * drift_std_per_step
                err_e += rng.standard_normal(n) * drift_std_per_step

                # ── TRN correction opportunity ─────────────────────────────
                if correction_mode != "none":
                    dist_since_trn_m = (current_km - last_trn_km) * 1000.0
                    if dist_since_trn_m >= self._trn_interval_m:
                        # Find matching fix location (within one step tolerance)
                        while (
                            fix_idx < len(fix_kms)
                            and fix_kms[fix_idx] < current_km - step_km / 2.0
                        ):
                            fix_idx += 1

                        if (
                            fix_idx < len(fix_kms)
                            and abs(fix_kms[fix_idx] - current_km) < step_km
                        ):
                            if fix_eligible[fix_idx]:
                                # Apply correction: reset to TRN residual accuracy
                                err_n = rng.standard_normal(n) * _SIGMA_TRN_M
                                err_e = rng.standard_normal(n) * _SIGMA_TRN_M
                                corrections_accepted_per_seed += 1
                            else:
                                corrections_suppressed_per_seed += 1
                            last_trn_km = current_km
                            fix_idx += 1

            # ── Record at checkpoints ─────────────────────────────────────
            for cp in self._checkpoints:
                if abs(current_km - cp) < step_km / 2.0 and recorded[cp] is None:
                    drift_2d = np.sqrt(err_n ** 2 + err_e ** 2)
                    recorded[cp] = drift_2d.copy()

        # Fill any checkpoints not reached (corridor shorter than requested)
        drift_2d_final = np.sqrt(err_n ** 2 + err_e ** 2)
        for cp in self._checkpoints:
            if recorded[cp] is None:
                recorded[cp] = drift_2d_final

        # Compute statistics
        p5_list    = []
        p50_list   = []
        p99_list   = []
        mean_list  = []
        for cp in self._checkpoints:
            d = recorded[cp]
            p5_list.append(float(np.percentile(d, 5)))
            p50_list.append(float(np.percentile(d, 50)))
            p99_list.append(float(np.percentile(d, 99)))
            mean_list.append(float(np.mean(d)))

        return MonteCarloResult(
            correction_mode             = correction_mode,
            n_seeds                     = n,
            checkpoints_km              = list(self._checkpoints),
            p5_drift_m                  = p5_list,
            p50_drift_m                 = p50_list,
            p99_drift_m                 = p99_list,
            mean_drift_m                = mean_list,
            corrections_accepted_mean   = float(np.mean(corrections_accepted_per_seed)),
            corrections_suppressed_mean = float(np.mean(corrections_suppressed_per_seed)),
            fix_eligibility             = fix_eligible,
        )

    def compare(
        self,
        result_a: MonteCarloResult,
        result_b: MonteCarloResult,
    ) -> dict:
        """
        Compare two MonteCarloResult objects at shared checkpoints.

        Returns a dict keyed by checkpoint_km with sub-dict:
            {
              'p50_reduction_pct': float,   # positive = result_b improved over result_a
              'p99_reduction_pct': float,
              'p50_a': float, 'p50_b': float,
              'p99_a': float, 'p99_b': float,
            }
        """
        if result_a.checkpoints_km != result_b.checkpoints_km:
            raise ValueError("Cannot compare results with different checkpoint lists")

        comparison = {}
        for i, cp in enumerate(result_a.checkpoints_km):
            p50_a = result_a.p50_drift_m[i]
            p50_b = result_b.p50_drift_m[i]
            p99_a = result_a.p99_drift_m[i]
            p99_b = result_b.p99_drift_m[i]
            comparison[cp] = {
                "p50_a": round(p50_a, 2),
                "p50_b": round(p50_b, 2),
                "p99_a": round(p99_a, 2),
                "p99_b": round(p99_b, 2),
                "p50_reduction_pct": round(
                    100.0 * (p50_a - p50_b) / p50_a if p50_a > 0 else 0.0, 1
                ),
                "p99_reduction_pct": round(
                    100.0 * (p99_a - p99_b) / p99_a if p99_a > 0 else 0.0, 1
                ),
            }
        return comparison

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _precompute_fix_locations(self) -> List[_FixLocation]:
        """
        Compute TRN fix locations along the corridor and assess terrain suitability.

        Fix opportunities occur every trn_interval_m starting from
        gnss_denial_start_km. Each fix location is evaluated against the
        DEMLoader terrain to determine eligibility (ACCEPT or CAUTION = eligible,
        SUPPRESS or out-of-bounds = ineligible).
        """
        from core.trn.hillshade_generator import HillshadeGenerator
        from core.trn.terrain_suitability import TerrainSuitabilityScorer

        corridor = self._corridor
        total_km = corridor.total_distance_km
        trn_interval_km = self._trn_interval_m / 1000.0
        denial_start_km = corridor.gnss_denial_start_km

        # First fix at denial_start + trn_interval
        first_fix_km = math.ceil(
            (denial_start_km + trn_interval_km) / trn_interval_km
        ) * trn_interval_km

        fix_locs = []
        scorer = TerrainSuitabilityScorer()
        hillshader = HillshadeGenerator()
        bounds = self._dem.get_bounds()
        dem_resolution_m = bounds["resolution_m"]

        km = first_fix_km
        while km <= total_km + 1e-6:
            lat, lon = corridor.position_at_km(km)

            # Assess terrain suitability
            tile_elev = self._dem.get_tile(
                lat_centre=lat,
                lon_centre=lon,
                tile_size_m=500.0,
                gsd_m=5.0,
            )
            eligible = False
            if not np.all(np.isnan(tile_elev)):
                hs = hillshader.generate(tile_elev, gsd_m=5.0)
                result = scorer.score(
                    elevation_tile=tile_elev,
                    hillshade_tile=hs,
                    gsd_m=5.0,
                    dem_resolution_m=dem_resolution_m,
                )
                eligible = result.recommendation in ("ACCEPT", "CAUTION")

            fix_locs.append(_FixLocation(km=km, lat=lat, lon=lon, eligible=eligible))
            km = round(km + trn_interval_km, 6)

        return fix_locs
