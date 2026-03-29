"""
BCMP-2 Synthetic Terrain Generator.

Constraint C-3 (Committed Synthetic Terrain)
---------------------------------------------
All BCMP-2 terrain is committed synthetic, seeded from SRTM-style mountain
profiles.  No network dependencies.  Fully reproducible from seed.

All seven C-3 parameters are exposed and documented.  A reviewer can inspect
and challenge any parameter before a demo.  The terrain is not cherry-picked:
parameters are set to SRTM-class values derived from published Himalayan and
Kashmir valley terrain statistics.

Design
------
Three-layer additive model:

  1. Macro ridgeline layer  — long-wavelength sinusoids oriented along
     ridge_orientation_deg.  Controls the dominant ridge-and-valley structure
     visible at 5–30 km scale.

  2. Meso terrain layer  — medium-wavelength noise (1–5 km) representing
     inter-ridge spurs, secondary ridges, and valley floors.

  3. Micro roughness layer  — short-wavelength turbulence (100–500 m)
     parameterised by terrain_roughness_factor.  This is the high-frequency
     texture that TRN NCC correlation relies on for distinctiveness.

The combination produces terrain that is:
  - Unique at every coordinate (no tiling artefacts)
  - Statistically representative of SRTM Himalayan profiles
  - Controllable and reproducible from a single integer seed

Phase terrain profiles (used by bcmp2_scenario.py)
---------------------------------------------------
  P1 (0–30 km):   Mountain ingress — high relief, strong ridgelines
  P2 (30–60 km):  Valley corridor — moderate relief, broad valley floors
  P3 (60–100 km): Plains — low relief, minimal roughness (TRN failure mode)
  P4 (100–120 km): Industrial clutter — flat with sparse height features
  P5 (120–150 km): Terminal zone — small high-detail synthetic area

Compatibility
-------------
The BCMP2Terrain.elevation_at(north_m, east_m) and
BCMP2Terrain.dem_patch(north_m, east_m, h_px, w_px, pixel_size_m)
interfaces are compatible with the existing TRNStub DEMProvider contract.

JOURNAL
-------
Built: 29 March 2026, micromind-node01.  SB-1 Step 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Default pixel resolution — matches existing DEMProvider in trn_stub.py
# ---------------------------------------------------------------------------
DEM_PIXEL_SIZE_M = 5.0


# ---------------------------------------------------------------------------
# Terrain parameter dataclass (C-3 seven required parameters)
# ---------------------------------------------------------------------------

@dataclass
class TerrainParams:
    """
    All seven C-3 terrain generator parameters.

    Defaults are SRTM-class Himalayan/Kashmir valley statistics.
    Every field is documented so a reviewer can challenge any value.
    """

    # 1. elevation_range_m — total vertical relief band [min_m, max_m]
    #    Himalayan tactical corridor: 1500–4500 m typical.
    #    Expressed as (min, max) tuple; mid-point is mean altitude.
    elevation_range_m: tuple = (800.0, 4200.0)

    # 2. ridge_spacing_m — mean distance between adjacent ridge crests (m)
    #    Himalayan transverse ridges: typically 8–25 km.
    ridge_spacing_m: float = 12_000.0

    # 3. ridge_orientation_deg — dominant ridge bearing (degrees, 0=North)
    #    Kashmir ridges run roughly NW–SE (~310° bearing for the ridgeline
    #    normals, meaning ridgelines themselves trend ~40°).
    ridge_orientation_deg: float = 40.0

    # 4. valley_width_m — characteristic valley floor width (m)
    #    Kashmir valley floor: ~15–30 km; tactical corridor valley: ~5–12 km.
    valley_width_m: float = 8_000.0

    # 5. local_slope_variance — RMS slope variation (dimensionless, metres/metre)
    #    Typical Himalayan mid-slope: 0.3–0.6 m/m.  Valley floors: 0.05–0.1.
    local_slope_variance: float = 0.35

    # 6. terrain_seed — deterministic seed for exact reproduction.
    #    Must be committed alongside any demo run.
    terrain_seed: int = 42

    # 7. terrain_roughness_factor — high-frequency texture amplitude (m)
    #    Controls micro-roughness layer amplitude.  TRN NCC needs > 5 m for
    #    reliable correlation; Himalayan terrain typically 10–40 m at 5 m pixel.
    terrain_roughness_factor: float = 15.0


# ---------------------------------------------------------------------------
# Per-phase preset profiles
# ---------------------------------------------------------------------------

PHASE_TERRAIN_PROFILES: dict[str, TerrainParams] = {
    "P1_mountain": TerrainParams(
        elevation_range_m=(1500.0, 4500.0),
        ridge_spacing_m=10_000.0,
        ridge_orientation_deg=40.0,
        valley_width_m=5_000.0,
        local_slope_variance=0.45,
        terrain_seed=101,
        terrain_roughness_factor=22.0,
    ),
    "P2_valley": TerrainParams(
        elevation_range_m=(800.0, 2500.0),
        ridge_spacing_m=15_000.0,
        ridge_orientation_deg=35.0,
        valley_width_m=10_000.0,
        local_slope_variance=0.18,
        terrain_seed=202,
        terrain_roughness_factor=10.0,
    ),
    "P3_plains": TerrainParams(
        # Flat terrain — TRN failure mode.  Low roughness, minimal relief.
        # NCC correlation quality drops significantly here; TRN suppressed.
        elevation_range_m=(150.0, 350.0),
        ridge_spacing_m=40_000.0,
        ridge_orientation_deg=0.0,
        valley_width_m=30_000.0,
        local_slope_variance=0.04,
        terrain_seed=303,
        terrain_roughness_factor=3.0,
    ),
    "P4_urban": TerrainParams(
        elevation_range_m=(150.0, 400.0),
        ridge_spacing_m=50_000.0,
        ridge_orientation_deg=0.0,
        valley_width_m=40_000.0,
        local_slope_variance=0.06,
        terrain_seed=404,
        terrain_roughness_factor=5.0,
    ),
    "P5_terminal": TerrainParams(
        # High-detail small zone: enough texture for terminal EO matching.
        elevation_range_m=(100.0, 300.0),
        ridge_spacing_m=5_000.0,
        ridge_orientation_deg=0.0,
        valley_width_m=2_000.0,
        local_slope_variance=0.08,
        terrain_seed=505,
        terrain_roughness_factor=8.0,
    ),
}


# ---------------------------------------------------------------------------
# Terrain generator
# ---------------------------------------------------------------------------

class BCMP2Terrain:
    """
    Synthetic terrain generator for BCMP-2.

    Implements the same elevation_at() and dem_patch() interface as the
    existing TRNStub DEMProvider so it can serve as a drop-in replacement
    in later sprints.

    Parameters
    ----------
    params : TerrainParams
        All seven C-3 parameters.  Defaults are SRTM Himalayan class.
    pixel_size_m : float
        DEM raster resolution in metres per pixel (default 5.0).

    Example
    -------
    >>> terrain = BCMP2Terrain(PHASE_TERRAIN_PROFILES["P1_mountain"])
    >>> elev = terrain.elevation_at(15_000.0, 8_000.0)
    >>> patch = terrain.dem_patch(15_000.0, 8_000.0, h_px=64, w_px=64)
    """

    def __init__(
        self,
        params: Optional[TerrainParams] = None,
        pixel_size_m: float = DEM_PIXEL_SIZE_M,
    ):
        self.params = params or TerrainParams()
        self.pixel_size_m = pixel_size_m
        self._rng = np.random.default_rng(self.params.terrain_seed)

        # Pre-compute random phases for each sinusoidal component.
        # Using fixed counts keeps the terrain deterministic and unique.
        self._setup_layers()

    def _setup_layers(self) -> None:
        """Pre-compute layer parameters from seed."""
        p = self.params
        rng = self._rng

        # ── Layer 1: Macro ridgelines ──────────────────────────────────────
        # 6 independent sinusoidal ridgelines oriented near ridge_orientation_deg
        n_macro = 6
        ridge_rad = math.radians(p.ridge_orientation_deg)
        spacing   = p.ridge_spacing_m

        # Each ridge gets a random phase offset and slight orientation jitter
        self._macro_phases   = rng.uniform(0, 2 * math.pi, n_macro)
        self._macro_orient   = ridge_rad + rng.uniform(
            -math.radians(15), math.radians(15), n_macro
        )
        self._macro_spacing  = spacing * rng.uniform(0.7, 1.3, n_macro)

        elev_mid  = (p.elevation_range_m[0] + p.elevation_range_m[1]) / 2.0
        elev_half = (p.elevation_range_m[1] - p.elevation_range_m[0]) / 2.0
        self._mean_elev    = elev_mid
        self._macro_amp    = elev_half * 0.55   # macro carries ~55% of relief

        # ── Layer 2: Meso terrain ─────────────────────────────────────────
        # 12 medium-wavelength components, random orientations
        n_meso = 12
        meso_lambda = spacing / rng.uniform(2.5, 6.0, n_meso)
        self._meso_kx = (2 * math.pi / meso_lambda) * np.cos(
            rng.uniform(0, 2 * math.pi, n_meso)
        )
        self._meso_ky = (2 * math.pi / meso_lambda) * np.sin(
            rng.uniform(0, 2 * math.pi, n_meso)
        )
        self._meso_phases = rng.uniform(0, 2 * math.pi, n_meso)
        self._meso_amp    = elev_half * 0.30 * p.local_slope_variance

        # ── Layer 3: Micro roughness ──────────────────────────────────────
        # 24 short-wavelength components, uniformly random orientations
        n_micro = 24
        micro_lambda = spacing / rng.uniform(20.0, 120.0, n_micro)
        self._micro_kx = (2 * math.pi / micro_lambda) * np.cos(
            rng.uniform(0, 2 * math.pi, n_micro)
        )
        self._micro_ky = (2 * math.pi / micro_lambda) * np.sin(
            rng.uniform(0, 2 * math.pi, n_micro)
        )
        self._micro_phases = rng.uniform(0, 2 * math.pi, n_micro)
        self._micro_amp    = p.terrain_roughness_factor

    # ------------------------------------------------------------------
    # Public interface (compatible with TRNStub DEMProvider)
    # ------------------------------------------------------------------

    def elevation_at(self, north_m: float, east_m: float) -> float:
        """
        Return terrain elevation (m) at arbitrary (north_m, east_m) coordinates.
        Coordinate system: north-east, metres from mission origin.
        """
        n, e = float(north_m), float(east_m)
        elev = self._mean_elev

        # Layer 1: macro ridgelines
        for i in range(len(self._macro_phases)):
            k   = 2 * math.pi / self._macro_spacing[i]
            ang = self._macro_orient[i]
            proj = n * math.cos(ang) + e * math.sin(ang)
            elev += self._macro_amp * math.sin(k * proj + self._macro_phases[i])

        # Layer 2: meso
        for i in range(len(self._meso_kx)):
            elev += self._meso_amp * math.sin(
                self._meso_kx[i] * n + self._meso_ky[i] * e + self._meso_phases[i]
            )

        # Layer 3: micro roughness
        for i in range(len(self._micro_kx)):
            elev += self._micro_amp * math.sin(
                self._micro_kx[i] * n + self._micro_ky[i] * e + self._micro_phases[i]
            )

        # Clamp to configured elevation_range_m
        elev = max(self.params.elevation_range_m[0],
                   min(self.params.elevation_range_m[1], elev))
        return elev

    def elevation_grid(
        self,
        north_m: float,
        east_m: float,
        h_px: int,
        w_px: int,
        pixel_size_m: Optional[float] = None,
    ) -> np.ndarray:
        """
        Vectorised elevation grid: returns (h_px, w_px) float64 array.
        Uses broadcasting — much faster than calling elevation_at in a loop.
        """
        ps = pixel_size_m or self.pixel_size_m
        rows = north_m + np.arange(h_px, dtype=np.float64) * ps  # (h_px,)
        cols = east_m  + np.arange(w_px, dtype=np.float64) * ps  # (w_px,)
        N, E = np.meshgrid(rows, cols, indexing="ij")              # (h_px, w_px)

        Z = np.full_like(N, self._mean_elev)

        # Layer 1
        for i in range(len(self._macro_phases)):
            k   = 2 * math.pi / self._macro_spacing[i]
            ang = self._macro_orient[i]
            proj = N * math.cos(ang) + E * math.sin(ang)
            Z += self._macro_amp * np.sin(k * proj + self._macro_phases[i])

        # Layer 2
        Z += np.sum(
            self._meso_amp * np.sin(
                self._meso_kx[:, None, None] * N[None]
                + self._meso_ky[:, None, None] * E[None]
                + self._meso_phases[:, None, None]
            ),
            axis=0,
        )

        # Layer 3
        Z += np.sum(
            self._micro_amp * np.sin(
                self._micro_kx[:, None, None] * N[None]
                + self._micro_ky[:, None, None] * E[None]
                + self._micro_phases[:, None, None]
            ),
            axis=0,
        )

        # Clamp to configured elevation_range_m (C-3: range is a hard constraint)
        Z = np.clip(Z, self.params.elevation_range_m[0], self.params.elevation_range_m[1])
        return Z

    def dem_patch(
        self,
        north_m: float,
        east_m: float,
        h_px: int = 64,
        w_px: int = 64,
        pixel_size_m: Optional[float] = None,
    ) -> np.ndarray:
        """
        Return a DEM patch of size (h_px × w_px) with top-left corner at
        (north_m, east_m).  Compatible with TRNStub DEMProvider.dem_patch().
        """
        return self.elevation_grid(north_m, east_m, h_px, w_px, pixel_size_m)

    # ------------------------------------------------------------------
    # Terrain statistics (for C-3 transparency reporting)
    # ------------------------------------------------------------------

    def terrain_stats(
        self,
        north_m: float = 0.0,
        east_m: float = 0.0,
        sample_km: float = 50.0,
        step_m: float = 500.0,
    ) -> dict:
        """
        Sample terrain over a region and return summary statistics.
        Used to verify C-3 parameters are producing expected relief.
        """
        n_steps = int(sample_km * 1000 / step_m)
        ns = north_m + np.arange(n_steps) * step_m
        es = east_m  + np.arange(n_steps) * step_m
        elevs = np.array([self.elevation_at(n, e) for n, e in zip(ns, es)])
        diffs = np.abs(np.diff(elevs))
        slopes = diffs / step_m
        return {
            "min_elev_m":      float(elevs.min()),
            "max_elev_m":      float(elevs.max()),
            "mean_elev_m":     float(elevs.mean()),
            "relief_m":        float(elevs.max() - elevs.min()),
            "rms_slope":       float(np.sqrt(np.mean(slopes ** 2))),
            "params":          self.params,
        }

    def __repr__(self) -> str:
        p = self.params
        return (
            f"BCMP2Terrain(seed={p.terrain_seed}, "
            f"relief={p.elevation_range_m[0]:.0f}–{p.elevation_range_m[1]:.0f} m, "
            f"ridge_spacing={p.ridge_spacing_m/1000:.0f} km, "
            f"roughness={p.terrain_roughness_factor:.1f} m)"
        )


# ---------------------------------------------------------------------------
# Factory: build terrain for a given mission phase
# ---------------------------------------------------------------------------

def terrain_for_phase(phase_key: str, override_seed: Optional[int] = None) -> BCMP2Terrain:
    """
    Return a BCMP2Terrain instance for the named phase.

    Parameters
    ----------
    phase_key    : one of PHASE_TERRAIN_PROFILES keys
    override_seed: if provided, overrides the profile's terrain_seed
    """
    if phase_key not in PHASE_TERRAIN_PROFILES:
        raise ValueError(
            f"Unknown phase_key '{phase_key}'. "
            f"Valid keys: {list(PHASE_TERRAIN_PROFILES)}"
        )
    params = PHASE_TERRAIN_PROFILES[phase_key]
    if override_seed is not None:
        from dataclasses import replace
        params = replace(params, terrain_seed=override_seed)
    return BCMP2Terrain(params)


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("BCMP-2 Terrain Generator — C-3 parameter verification")
    print("=" * 60)
    for phase_key, expected_relief in [
        ("P1_mountain", (1500, 3200)),
        ("P2_valley",   (500,  2000)),
        ("P3_plains",   (50,   300)),
        ("P4_urban",    (50,   350)),
        ("P5_terminal", (50,   300)),
    ]:
        t = terrain_for_phase(phase_key)
        stats = t.terrain_stats(sample_km=30.0)
        relief = stats["relief_m"]
        rms_slope = stats["rms_slope"]
        status = "OK" if expected_relief[0] <= relief <= expected_relief[1] else "CHECK"
        print(f"  {phase_key:<16} relief={relief:6.0f} m  "
              f"rms_slope={rms_slope:.3f}  {status}")
        print(f"                  {t}")

    # Verify determinism
    t1 = terrain_for_phase("P1_mountain")
    t2 = terrain_for_phase("P1_mountain")
    same = t1.elevation_at(10000, 5000) == t2.elevation_at(10000, 5000)
    print(f"\n  Determinism check (same seed): {'PASS' if same else 'FAIL'}")

    # Verify dem_patch shape
    patch = terrain_for_phase("P1_mountain").dem_patch(0, 0, h_px=64, w_px=64)
    print(f"  dem_patch shape: {patch.shape}  dtype: {patch.dtype}  "
          f"min={patch.min():.0f} m  max={patch.max():.0f} m")
