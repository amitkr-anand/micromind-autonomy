"""
core/trn/hillshade_generator.py
MicroMind / NanoCorteX — Hillshade Tile Generator

NAV-02: Generate terrain-shaded reference tiles from DEM elevation data.
These are the reference images used for phase correlation matching.

Illumination model: Lambertian (CAS paper Wan et al. 2021, Eq. 3).
Multi-directional hillshade is the default for TRN reference tiles because
phase correlation robustness to illumination variation is highest when
reference tiles are illumination-invariant (CAS paper §3.2).

Sensor substitution contract:
    Current: Procedurally generated from DEM (numpy only)
    Real replacement: Pre-generated tile library or on-board tile generation
    Interface: identical numpy uint8 array output

References:
    SRS v1.3 NAV-02
    CAS paper: Wan et al. 2021, Eq. 3 (Lambertian illumination)
    docs/interfaces/dem_contract.yaml
"""
from __future__ import annotations

import math

import numpy as np


class HillshadeGenerator:
    """
    NAV-02: Generate terrain-shaded reference tiles for TRN phase correlation.

    Reference tile generation follows the Lambertian illumination model
    (CAS paper Eq. 3). Illumination parameters are configurable to match
    expected lighting conditions at mission time.

    Sensor substitution contract:
        Current: Procedurally generated from DEM
        Real replacement: Pre-generated tile library or on-board tile generation
        Interface: identical numpy uint8 array output — no architecture change
    """

    def __init__(
        self,
        azimuth_deg: float = 315.0,
        elevation_deg: float = 45.0,
    ) -> None:
        """
        azimuth_deg  : solar azimuth (0 = North, clockwise). Default 315 = NW.
        elevation_deg: solar elevation above horizon. Default 45 degrees.
        These match typical mid-morning Indian subcontinent conditions.
        """
        self._azimuth_deg   = azimuth_deg
        self._elevation_deg = elevation_deg

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        elevation_tile: np.ndarray,
        gsd_m: float,
    ) -> np.ndarray:
        """
        Generate single-direction hillshade from elevation tile.

        Lambertian model (CAS paper Eq. 3):
            I(x,y) = L · (p·cos(τ)·cos(σ) + q·sin(τ)·cos(σ) + sin(σ)) · r(x,y)

        where:
            p, q  = elevation gradients in x (east) and y (north) directions
            τ     = solar azimuth (rad, from North, clockwise)
            σ     = solar elevation (rad, above horizon)
            r(x,y)= 1.0 (uniform Lambertian reflectance for reference generation)
            L     = scale constant (1.0)

        Returns: uint8 numpy array [0, 255], same shape as elevation_tile.
        """
        return self._lambertian_hillshade(
            elevation_tile, gsd_m, self._azimuth_deg, self._elevation_deg
        )

    def generate_multidirectional(
        self,
        elevation_tile: np.ndarray,
        gsd_m: float,
        n_directions: int = 8,
    ) -> np.ndarray:
        """
        Generate multi-directional hillshade by averaging N azimuth directions
        equally spaced around the compass. More robust to illumination uncertainty
        than single-direction hillshade. Use this for TRN reference tiles.

        Mathematical basis: CAS paper §3.2 — translation information is in fringe
        density and orientation, not amplitude. Multi-directional hillshade reduces
        amplitude modulation from single sun angle, improving phase correlation
        robustness across mission time windows.

        Returns: uint8 numpy array [0, 255], same shape as elevation_tile.
        """
        if n_directions < 1:
            raise ValueError(f"n_directions must be >= 1 (got {n_directions})")

        step = 360.0 / n_directions
        accum = np.zeros(elevation_tile.shape, dtype=np.float64)
        for i in range(n_directions):
            az = i * step
            hs = self._lambertian_hillshade(
                elevation_tile, gsd_m, az, self._elevation_deg
            )
            accum += hs.astype(np.float64)

        averaged = accum / n_directions
        return np.clip(averaged, 0, 255).astype(np.uint8)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _lambertian_hillshade(
        elevation_tile: np.ndarray,
        gsd_m: float,
        azimuth_deg: float,
        elevation_deg: float,
    ) -> np.ndarray:
        """
        Compute Lambertian hillshade for a given sun position.

        Solar azimuth is converted from geographic convention (0=North, clockwise)
        to mathematical convention (0=East, counter-clockwise) for gradient
        computation. The gradient convention matches numpy.gradient output where
        axis=0 is row (south→north positive) and axis=1 is column (west→east positive).
        """
        elev_f = elevation_tile.astype(np.float64)

        # Gradient: dy = north gradient (row axis, increasing = northward)
        #           dx = east gradient (col axis, increasing = eastward)
        # numpy.gradient returns [d/d_row, d/d_col]
        grad = np.gradient(elev_f, gsd_m)
        # grad[0]: d/d_row — row increases downward → negate for northward positive
        dz_north = -grad[0]   # p in CAS notation: d_elev / d_north
        dz_east  =  grad[1]   # q in CAS notation: d_elev / d_east

        # Sun direction in geographic convention → convert to unit vector
        # τ: azimuth from North, clockwise
        tau_rad   = math.radians(azimuth_deg)
        sigma_rad = math.radians(elevation_deg)

        cos_sigma = math.cos(sigma_rad)
        sin_sigma = math.sin(sigma_rad)
        # Azimuth from North, clockwise: sun_north = cos(τ), sun_east = sin(τ)
        sun_north = math.cos(tau_rad)
        sun_east  = math.sin(tau_rad)

        # Lambertian dot product: I = (p·sun_east + q·sun_north)·cos(σ) + sin(σ)
        # (r(x,y) = 1.0 uniform reflectance)
        intensity = (
            dz_east  * sun_east  * cos_sigma
            + dz_north * sun_north * cos_sigma
            + sin_sigma
        )

        # Clamp to [0, 1] and scale to uint8
        intensity = np.clip(intensity, 0.0, 1.0)
        return (intensity * 255.0).astype(np.uint8)
