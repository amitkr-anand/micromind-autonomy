"""
core/trn/blender_frame_ingestor.py
MicroMind / NanoCorteX — Blender Frame Ingestor

NAV-02: Load and validate Blender-rendered synthetic UAV frames for
cross-modal TRN validation.

These frames simulate what a real nadir EO camera would see flying
over real terrain at 150m AGL. They are the query images for phase
correlation matching against DEM hillshade reference tiles.

Sensor substitution contract:
    Current:  Blender-rendered PNG
    Gate 2+:  Gazebo-rendered frame
    HIL:      Real EO camera frame
    Interface: identical numpy uint8 array in all three cases.
    Breaking change at HIL: NO

Reference: CAS paper (Wan et al. 2021)
    §4 — query image is rendered from DEM with sun model; reference is
    DEM hillshade. Cross-modal matching via phase correlation frequency
    domain is robust to illumination difference between the two.

Req IDs: NAV-02, AD-01, EC-13
Governance: Code Governance Manual v3.4
"""
from __future__ import annotations

import math
import os
import re
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from core.navigation.corridors import MissionCorridor


class BlenderFrameIngestor:
    """
    NAV-02: Load and validate Blender-rendered synthetic UAV frames for
    cross-modal TRN validation.

    These frames simulate what a real nadir EO camera would see flying
    over real terrain at 150m AGL. They are the query images for
    phase correlation matching against DEM hillshade reference tiles.

    Sensor substitution contract:
        Current: Blender-rendered PNG
        Gate 2+: Gazebo-rendered frame
        HIL: Real EO camera frame
        Interface: identical numpy uint8 array in all three cases.
        Breaking change at HIL: NO

    Reference: CAS paper (Wan et al. 2021)
        §4 — query image is rendered from DEM with sun model; reference is
        DEM hillshade. Cross-modal matching via phase correlation frequency
        domain is robust to illumination difference between the two.
    """

    # Expected frame dimensions
    _EXPECTED_WIDTH  = 640
    _EXPECTED_HEIGHT = 640

    # Frame filename pattern: frame_km000.png, frame_km027.png, etc.
    _FRAME_PATTERN = re.compile(r'^frame_km(\d+)\.png$', re.IGNORECASE)

    def __init__(
        self,
        frames_dir: str,
        corridor: 'MissionCorridor',
        altitude_m: float = 150.0,
        camera_fov_deg: float = 60.0,
    ) -> None:
        """
        frames_dir     : directory containing frame_km000.png ... frame_km055.png
        corridor       : MissionCorridor instance for geographic coordinate lookup
        altitude_m     : AGL altitude at which frames were rendered
        camera_fov_deg : camera field of view used in Blender render
        """
        self._frames_dir    = frames_dir
        self._corridor      = corridor
        self._altitude_m    = altitude_m
        self._camera_fov_deg = camera_fov_deg

        # Pre-compute GSD from altitude and FOV at nominal frame width
        self._gsd_m = self._compute_gsd(altitude_m, camera_fov_deg,
                                        self._EXPECTED_WIDTH)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_frame(
        self,
        km: float,
    ) -> tuple[np.ndarray, float, float, float]:
        """
        Load frame for given corridor km.

        Returns:
            (frame_rgb, lat, lon, gsd_m)

            frame_rgb : uint8 (640,640,3) BGR array
            lat, lon  : geographic position at km
            gsd_m     : ground sample distance computed from altitude and FOV:
                          gsd_m = 2 * altitude_m * tan(fov_rad/2) / frame_width_px

        Raises FileNotFoundError if no frame exists for given km.
        """
        filename = f"frame_km{int(round(km)):03d}.png"
        filepath = os.path.join(self._frames_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"No frame found for km={km:.1f}: {filepath}"
            )

        frame_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise FileNotFoundError(
                f"Failed to read frame: {filepath}"
            )

        lat, lon = self._corridor.position_at_km(km)
        return frame_bgr, lat, lon, self._gsd_m

    def load_all_frames(
        self,
    ) -> list[tuple[float, np.ndarray, float, float, float]]:
        """
        Load all available frames.

        Returns list of:
            (km, frame_rgb, lat, lon, gsd_m)
        Sorted by km ascending.
        """
        kms = self.get_available_kms()
        results = []
        for km in kms:
            frame_bgr, lat, lon, gsd_m = self.load_frame(km)
            results.append((km, frame_bgr, lat, lon, gsd_m))
        return results

    def get_available_kms(self) -> list[float]:
        """
        Scan frames_dir for frame_km*.png files and return list of km values.
        """
        if not os.path.isdir(self._frames_dir):
            return []

        kms: list[float] = []
        for fname in os.listdir(self._frames_dir):
            m = self._FRAME_PATTERN.match(fname)
            if m:
                kms.append(float(m.group(1)))

        kms.sort()
        return kms

    def validate_frame(
        self,
        frame: np.ndarray,
    ) -> dict:
        """
        Quality check on a rendered frame.

        Returns:
            {
                shape_valid         : bool,
                laplacian_variance  : float,
                channel_means       : [R, G, B],
                is_colour           : bool   (R-G diff > 5),
                shi_tomasi_corners  : int,
                quality             : 'GOOD' | 'MARGINAL' | 'POOR'
                    GOOD:     lap_var > 200, corners > 100
                    MARGINAL: lap_var 50-200, corners 30-100
                    POOR:     lap_var < 50, corners < 30
            }
        """
        # Shape check
        shape_valid = (
            len(frame.shape) == 3
            and frame.shape[0] == self._EXPECTED_HEIGHT
            and frame.shape[1] == self._EXPECTED_WIDTH
            and frame.shape[2] == 3
        )

        # cv2 loads BGR; report channel means as [R, G, B] for external clarity
        if frame.ndim == 3 and frame.shape[2] == 3:
            b_mean = float(np.mean(frame[:, :, 0]))
            g_mean = float(np.mean(frame[:, :, 1]))
            r_mean = float(np.mean(frame[:, :, 2]))
        else:
            r_mean = g_mean = b_mean = float(np.mean(frame))

        channel_means = [r_mean, g_mean, b_mean]
        is_colour = abs(r_mean - g_mean) > 5.0

        # Laplacian variance on grayscale
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.astype(np.uint8)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(laplacian.var())

        # Shi-Tomasi corner count
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=1000, qualityLevel=0.01, minDistance=10
        )
        n_corners = int(len(corners)) if corners is not None else 0

        # Quality classification
        if lap_var > 200 and n_corners > 100:
            quality = 'GOOD'
        elif lap_var >= 50 and n_corners >= 30:
            quality = 'MARGINAL'
        else:
            quality = 'POOR'

        return {
            'shape_valid':          shape_valid,
            'laplacian_variance':   lap_var,
            'channel_means':        channel_means,
            'is_colour':            is_colour,
            'shi_tomasi_corners':   n_corners,
            'quality':              quality,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_gsd(
        altitude_m: float,
        fov_deg: float,
        frame_width_px: int,
    ) -> float:
        """
        Compute GSD (m/px) from altitude, FOV, and frame width.

        gsd_m = 2 * altitude_m * tan(fov_rad / 2) / frame_width_px

        At 150m AGL, 60° FOV, 640px:
            footprint = 2 * 150 * tan(30°) = 2 * 150 * 0.5774 = 173.2 m
            gsd = 173.2 / 640 = 0.271 m/px
        """
        fov_rad = math.radians(fov_deg)
        footprint_m = 2.0 * altitude_m * math.tan(fov_rad / 2.0)
        return footprint_m / frame_width_px
