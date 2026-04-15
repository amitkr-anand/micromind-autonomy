#!/usr/bin/env python3
"""
Real Sentinel-2 TRN validation — OI-46.
Run:

  python scripts/validate_real_sentinel_trn.py
    --frames  data/synthetic_imagery/shimla_corridor/
    --tci     <path_to_T43RGQ_TCI_10m.jp2>
    --output  docs/qa/real_sentinel_trn_results.md

OI-45 (self-consistency) proved the phase correlation engine is algorithmically
correct (peaks 0.99).  OI-46 requires genuine same-modality validation:

    Query:     Blender-rendered frames (current: rendered from shimla_texture.png,
               a DEM hillshade-colour map — NOT actual Sentinel-2 imagery).
    Reference: Real Sentinel-2 TCI at same lat/lon/GSD (T43RGQ, 10 Oct 2025,
               EPSG:32643, 10 m/px, WGS84 77.09–78.26°E 30.60–31.62°N).

Tile selection:
    Two TCI tiles available.  T43RGQ covers all 12 SHIMLA_LOCAL frames (km 0–55).
    T43RFQ covers only km=0 and km=5 (its eastern edge is at 77.212°E < km=10
    longitude of 77.214°E).  T43RGQ is therefore used as primary reference.

GSD matching:
    Sentinel-2 resolution: 10.0 m/px.
    Camera GSD (150 m AGL, 60° FOV, 640 px): 0.2706 m/px.
    Camera footprint: 173.2 m × 173.2 m.
    TRN GSD: max(camera_gsd, sentinel_res × 0.5) = max(0.27, 5.0) = 5.0 m/px.
    Reference tile pixels: int(173.2 / 5.0) = 34 × 34.
    Source pixels in TCI: int(173.2 / 10.0) = 17 × 17, up-sampled 2× to 34 × 34.
    Query pixels: 640 × 640 Blender frame, down-sampled to 34 × 34 by TRN.match().
    Both reference and query represent the same 173.2 m × 173.2 m area — no
    footprint mismatch.

Known limitation:
    shimla_texture.png (the Blender terrain texture) is a DEM hillshade-colour
    image (viz.hh_hillshade-color.png from OpenTopography), NOT actual Sentinel-2
    imagery.  Genuine same-modality TRN requires the Blender terrain texture to be
    replaced with a crop from the Sentinel-2 TCI.  NCC peaks < 0.3 confirm this
    gap; peaks ≥ 0.3 would validate the current setup as operationally useful.

Req IDs: NAV-02, AD-01, EC-13
Frozen files: none modified.
"""
import argparse
import glob
import math
import os
import sys
import time

sys.path.insert(0, '.')

import cv2
import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform, transform_bounds

from core.trn.phase_correlation_trn import PhaseCorrelationTRN
from core.trn.terrain_suitability import TerrainSuitabilityScorer
from core.trn.blender_frame_ingestor import BlenderFrameIngestor
from core.navigation.corridors import SHIMLA_LOCAL, SHIMLA_MANALI


# ---------------------------------------------------------------------------
# Default TCI path — T43RGQ covers all 12 corridor frames
# ---------------------------------------------------------------------------
_DEFAULT_TCI_GLOB = (
    "data/terrain/shimla_corridor/**/*T43RGQ*TCI_10m.jp2"
)


# ---------------------------------------------------------------------------
# SentinelTCILoader
# DEMLoader-compatible adapter that extracts luminance crops from a
# Sentinel-2 True Colour Image (TCI) JPEG2000 file.
# ---------------------------------------------------------------------------

class SentinelTCILoader:
    """
    DEMLoader-compatible wrapper for a Sentinel-2 TCI JPEG2000 file.

    The file is expected to be in UTM projection (e.g. EPSG:32643 for
    WGS 84 / UTM zone 43N covering the Shimla corridor).

    get_tile() converts the requested WGS84 centre coordinate to UTM,
    extracts a windowed read from the JP2, converts the RGB crop to
    luminance (float32, 0–255), and resamples to the requested pixel
    count.  The return value matches the DEMLoader.get_tile() contract
    so PhaseCorrelationTRN.match() can use it without modification.

    The returned float32 "elevation" tile contains luminance values
    0–255.  PassthroughHillshadeGen normalises this to uint8 as-is,
    producing a same-modality reference tile for phase correlation.
    """

    def __init__(self, tci_path: str) -> None:
        if not os.path.exists(tci_path):
            raise FileNotFoundError(f"TCI file not found: {tci_path}")
        self._path = tci_path

        with rasterio.open(tci_path) as src:
            self._crs       = src.crs
            self._transform = src.transform
            self._width     = src.width
            self._height    = src.height
            self._res_m     = float(src.res[0])   # 10.0 m/px
            self._utm_bounds = src.bounds          # in UTM metres

            # Convert UTM bounds to WGS84 for is_in_bounds() checks
            wgs84 = transform_bounds(
                src.crs, 'EPSG:4326',
                src.bounds.left, src.bounds.bottom,
                src.bounds.right, src.bounds.top,
            )
        self._wgs84_west  = wgs84[0]
        self._wgs84_south = wgs84[1]
        self._wgs84_east  = wgs84[2]
        self._wgs84_north = wgs84[3]

    # ── DEMLoader interface ───────────────────────────────────────────────────

    def get_bounds(self) -> dict:
        return {
            'north':        self._wgs84_north,
            'south':        self._wgs84_south,
            'east':         self._wgs84_east,
            'west':         self._wgs84_west,
            'resolution_m': self._res_m,
        }

    def is_in_bounds(self, lat: float, lon: float) -> bool:
        return (
            self._wgs84_south <= lat <= self._wgs84_north
            and self._wgs84_west  <= lon <= self._wgs84_east
        )

    def get_tile(
        self,
        lat_centre: float,
        lon_centre: float,
        tile_size_m: float,
        gsd_m: float,
    ) -> np.ndarray:
        """
        Extract a square luminance tile from the TCI centred at
        (lat_centre, lon_centre), covering tile_size_m × tile_size_m
        metres, resampled to int(tile_size_m / gsd_m) pixels.

        Returns float32 array (H, W) in [0, 255].
        Returns NaN-filled array if centre is outside tile bounds.
        """
        n_pixels = max(1, int(tile_size_m / gsd_m))

        if not self.is_in_bounds(lat_centre, lon_centre):
            return np.full((n_pixels, n_pixels), float('nan'), dtype=np.float32)

        # ── Convert WGS84 → TCI CRS (UTM) ────────────────────────────────────
        xy = warp_transform(
            'EPSG:4326', self._crs,
            [lon_centre], [lat_centre],
        )
        x_utm = float(xy[0][0])
        y_utm = float(xy[1][0])

        half_m = tile_size_m / 2.0

        # ── UTM bounding box of requested tile ────────────────────────────────
        x_min = x_utm - half_m
        x_max = x_utm + half_m
        y_min = y_utm - half_m
        y_max = y_utm + half_m

        # ── Pixel window in TCI ───────────────────────────────────────────────
        # rowcol expects (transform, xs, ys) in the file's CRS
        row_top, col_left  = rowcol(self._transform, x_min, y_max)
        row_bot, col_right = rowcol(self._transform, x_max, y_min)

        row_top   = int(max(0,                  row_top ))
        col_left  = int(max(0,                  col_left ))
        row_bot   = int(min(self._height - 1,   row_bot ))
        col_right = int(min(self._width  - 1,   col_right))

        if row_bot <= row_top or col_right <= col_left:
            return np.full((n_pixels, n_pixels), float('nan'), dtype=np.float32)

        # ── Windowed read from JP2 ────────────────────────────────────────────
        from rasterio.windows import Window
        win_height = row_bot - row_top + 1
        win_width  = col_right - col_left + 1

        with rasterio.open(self._path) as src:
            window = Window(col_left, row_top, win_width, win_height)
            data = src.read(window=window)   # (3, H, W) uint8

        # ── RGB → luminance ───────────────────────────────────────────────────
        r = data[0].astype(np.float32)
        g = data[1].astype(np.float32)
        b = data[2].astype(np.float32)
        lum = 0.299 * r + 0.587 * g + 0.114 * b   # standard luma

        # ── Resample to (n_pixels, n_pixels) ─────────────────────────────────
        if lum.shape[0] < 2 or lum.shape[1] < 2:
            return np.full((n_pixels, n_pixels), float('nan'), dtype=np.float32)

        from scipy.ndimage import zoom as _zoom
        zr = n_pixels / lum.shape[0]
        zc = n_pixels / lum.shape[1]
        resampled = _zoom(lum, (zr, zc), order=1)
        return resampled.astype(np.float32)


# ---------------------------------------------------------------------------
# PassthroughHillshadeGen (same as OI-45)
# ---------------------------------------------------------------------------

class PassthroughHillshadeGen:
    """
    HillshadeGenerator-compatible wrapper for same-modality TRN.

    The 'elevation_tile' from SentinelTCILoader is already a luminance
    map.  Returning it normalised to uint8 preserves the Sentinel-2
    spatial structure as the phase correlation reference tile.
    """

    def generate(self, elevation_tile: np.ndarray, gsd_m: float) -> np.ndarray:
        return self._normalise(elevation_tile)

    def generate_multidirectional(
        self,
        elevation_tile: np.ndarray,
        gsd_m: float,
        n_directions: int = 8,
    ) -> np.ndarray:
        return self._normalise(elevation_tile)

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return np.zeros(arr.shape, dtype=np.uint8)
        mn, mx = float(valid.min()), float(valid.max())
        if mx - mn < 1e-6:
            return np.full(arr.shape, 128, dtype=np.uint8)
        norm = np.nan_to_num(arr, nan=mn)
        return np.clip((norm - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

from dataclasses import dataclass

@dataclass
class RealSentinelResult:
    km:                     float
    status:                 str
    peak_value:             float
    suitability_score:      float
    suitability_rec:        str
    correction_north_m:     float
    correction_east_m:      float
    correction_mag_m:       float   # magnitude of returned correction
    ref_texture_var:        float   # Laplacian variance of 34×34 TCI reference tile
    ref_relief_m:           float   # max-min of TCI luminance values in tile
    query_quality:          str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Real Sentinel-2 TRN validation (OI-46).'
    )
    parser.add_argument('--frames', required=True,
                        help='Directory containing frame_km*.png files')
    parser.add_argument('--tci', default=None,
                        help='Path to Sentinel-2 TCI JP2 (T43RGQ). '
                             'Auto-detected from default location if omitted.')
    parser.add_argument('--corridor', default='shimla_local',
                        choices=['shimla_local', 'shimla_manali'])
    parser.add_argument('--output', default='docs/qa/real_sentinel_trn_results.md')
    parser.add_argument('--altitude', type=float, default=150.0,
                        help='AGL altitude (m)')
    parser.add_argument('--fov', type=float, default=60.0,
                        help='Camera FOV (degrees)')
    args = parser.parse_args()

    # ── Auto-detect TCI ──────────────────────────────────────────────────────
    tci_path = args.tci
    if tci_path is None:
        candidates = sorted(glob.glob(_DEFAULT_TCI_GLOB, recursive=True))
        if not candidates:
            print('ERROR: No T43RGQ TCI file found. Pass --tci explicitly.')
            sys.exit(1)
        tci_path = candidates[0]

    corridor = SHIMLA_LOCAL if args.corridor == 'shimla_local' else SHIMLA_MANALI

    # ── Camera geometry ───────────────────────────────────────────────────────
    fov_rad      = math.radians(args.fov)
    footprint_m  = 2.0 * args.altitude * math.tan(fov_rad / 2.0)
    camera_gsd   = footprint_m / 640.0

    # TRN GSD: clamped to half sentinel native resolution
    with rasterio.open(tci_path) as _src:
        sentinel_res_m = float(_src.res[0])
    trn_gsd = max(camera_gsd, sentinel_res_m * 0.5)

    n_ref_px = int(footprint_m / trn_gsd)

    print(f'Sentinel-2 TCI:    {tci_path}')
    print(f'Sentinel res:      {sentinel_res_m:.1f} m/px')
    print(f'Camera footprint:  {footprint_m:.1f} m,  camera GSD {camera_gsd:.4f} m/px')
    print(f'TRN GSD:           {trn_gsd:.1f} m/px (max of camera_gsd, sentinel_res/2)')
    print(f'Reference tile:    {n_ref_px}×{n_ref_px} px  '
          f'({footprint_m:.0f} m / {trn_gsd:.1f} m/px)')
    print(f'Source pixels:     ~{footprint_m / sentinel_res_m:.1f}×'
          f'{footprint_m / sentinel_res_m:.1f} TCI pixels, up-sampled '
          f'{trn_gsd / camera_gsd * n_ref_px / (footprint_m / sentinel_res_m):.1f}×')
    print()

    # ── Construct TRN ────────────────────────────────────────────────────────
    tci_loader   = SentinelTCILoader(tci_path)
    passthrough  = PassthroughHillshadeGen()
    scorer       = TerrainSuitabilityScorer()   # default thresholds

    trn = PhaseCorrelationTRN(
        dem_loader=tci_loader,
        hillshade_gen=passthrough,
        suitability_scorer=scorer,
        tile_size_m=footprint_m,    # tile = camera footprint → matched scale
        min_peak_value=0.10,
        clock_fn=lambda: int(time.monotonic() * 1000),
    )

    # ── Load frames ───────────────────────────────────────────────────────────
    ingestor = BlenderFrameIngestor(
        frames_dir=args.frames,
        corridor=corridor,
        altitude_m=args.altitude,
        camera_fov_deg=args.fov,
    )
    available_kms = ingestor.get_available_kms()
    print(f'Frames available: {available_kms}')
    print()

    # ── Check TCI bounds coverage ─────────────────────────────────────────────
    bounds = tci_loader.get_bounds()
    print(f'TCI WGS84 bounds:  W={bounds["west"]:.4f} S={bounds["south"]:.4f} '
          f'E={bounds["east"]:.4f} N={bounds["north"]:.4f}')
    print()

    print('Running real Sentinel-2 validation ...')
    print()

    results: list[RealSentinelResult] = []
    mission_time_ms = 0

    for km in available_kms:
        frame_bgr, lat, lon, _gsd = ingestor.load_frame(km)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Quality
        lap_var  = float(cv2.Laplacian(frame_gray, cv2.CV_64F).var())
        corners  = cv2.goodFeaturesToTrack(
            frame_gray, maxCorners=1000, qualityLevel=0.01, minDistance=10,
        )
        n_corners = len(corners) if corners is not None else 0
        if lap_var > 200 and n_corners > 100:
            quality = 'GOOD'
        elif lap_var >= 50 and n_corners >= 30:
            quality = 'MARGINAL'
        else:
            quality = 'POOR'

        # Pre-compute reference tile statistics for diagnostics
        ref_tile_float = tci_loader.get_tile(lat, lon, footprint_m, trn_gsd)
        if not np.isnan(ref_tile_float).all():
            ref_uint8  = passthrough._normalise(ref_tile_float)
            lap        = cv2.Laplacian(ref_uint8, cv2.CV_64F)
            ref_tex_v  = float(lap.var())
            valid_vals = ref_tile_float[~np.isnan(ref_tile_float)]
            ref_relief = float(valid_vals.max() - valid_vals.min()) if valid_vals.size else 0.0
        else:
            ref_tex_v  = 0.0
            ref_relief = 0.0

        # Run TRN match: no offset — frames rendered at exact corridor positions
        match = trn.match(
            camera_tile=frame_gray,
            lat_estimate=lat,
            lon_estimate=lon,
            alt_m=args.altitude,
            gsd_m=trn_gsd,
            mission_time_ms=mission_time_ms,
        )

        corr_mag = math.sqrt(
            match.correction_north_m ** 2 + match.correction_east_m ** 2
        )

        results.append(RealSentinelResult(
            km=km,
            status=match.status,
            peak_value=match.confidence,
            suitability_score=match.suitability_score,
            suitability_rec=match.suitability_recommendation,
            correction_north_m=match.correction_north_m,
            correction_east_m=match.correction_east_m,
            correction_mag_m=corr_mag,
            ref_texture_var=ref_tex_v,
            ref_relief_m=ref_relief,
            query_quality=quality,
        ))
        mission_time_ms += 5000

    # ── Print results ─────────────────────────────────────────────────────────
    n_accepted   = sum(1 for r in results if r.status == 'ACCEPTED')
    n_suppressed = sum(1 for r in results if r.status == 'SUPPRESSED')
    n_rejected   = sum(1 for r in results if r.status == 'REJECTED')
    peaks        = [r.peak_value for r in results]

    print('Real Sentinel-2 TRN Results (OI-46)')
    print('Query:     Blender frames (DEM hillshade-colour texture)')
    print('Reference: Sentinel-2 TCI T43RGQ, 10 m/px, Oct 2025')
    print(f'Corridor:  {corridor.name},  Altitude: {args.altitude}m AGL')
    print(f'Ref size:  {n_ref_px}×{n_ref_px} px @ {trn_gsd:.1f} m/px '
          f'({footprint_m:.0f}m footprint)')
    print()
    print(f'{"km":>4} | {"Status":>10} | {"Peak":>6} | {"Suit":>5} | {"Rec":>8} | '
          f'{"RefTexV":>7} | {"RefRel":>6} | {"CorrMag_m":>9} | Query')
    print('-' * 88)
    for r in results:
        print(
            f'{r.km:4.0f} | {r.status:>10} | {r.peak_value:6.4f} | '
            f'{r.suitability_score:5.3f} | {r.suitability_rec:>8} | '
            f'{r.ref_texture_var:7.1f} | {r.ref_relief_m:6.1f} | '
            f'{r.correction_mag_m:9.2f} | {r.query_quality}'
        )

    print()
    print('Summary:')
    print(f'  Accepted:    {n_accepted}/{len(results)}')
    print(f'  Rejected:    {n_rejected}/{len(results)}')
    print(f'  Suppressed:  {n_suppressed}/{len(results)}')
    print(f'  Peak range:  {min(peaks):.4f} – {max(peaks):.4f}')
    if peaks:
        print(f'  Mean peak:   {float(np.mean(peaks)):.4f}')
    print(f'  Threshold:   {trn._min_peak:.3f}')
    print()
    print('Baseline comparison:')
    print(f'  OI-44 cross-modal (RGB vs DEM hillshade): 0.0903–0.1136  0/12')
    print(f'  OI-45 same-modal self-offset:             0.9874–0.9932  12/12')
    print(f'  OI-46 real Sentinel-2 TCI:                '
          f'{min(peaks):.4f}–{max(peaks):.4f}  {n_accepted}/12')

    _write_report(results, args.output, corridor, args.altitude, trn,
                  tci_path, sentinel_res_m, trn_gsd, footprint_m, n_ref_px)
    print(f'\nReport written: {args.output}')


def _write_report(results, output_path, corridor, altitude, trn,
                  tci_path, sentinel_res_m, trn_gsd, footprint_m, n_ref_px):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_accepted   = sum(1 for r in results if r.status == 'ACCEPTED')
    n_suppressed = sum(1 for r in results if r.status == 'SUPPRESSED')
    n_rejected   = sum(1 for r in results if r.status == 'REJECTED')
    peaks        = [r.peak_value for r in results]
    non_supp     = [r.peak_value for r in results
                    if r.status not in ('SUPPRESSED', 'OUTSIDE_COVERAGE')]

    with open(output_path, 'w') as f:
        f.write('# Real Sentinel-2 TRN Validation — OI-46\n')
        f.write('## MicroMind Pre-HIL Evidence\n\n')
        f.write(f'**Date:** 16 April 2026\n')
        f.write(f'**Corridor:** {corridor.name}\n')
        f.write(f'**Query:** Blender-rendered RGB frames (shimla_texture.png — '
                'DEM hillshade-colour from OpenTopography)\n')
        f.write(f'**Reference:** Sentinel-2 TCI `{os.path.basename(tci_path)}`  \n')
        f.write(f'  CRS: EPSG:32643 (UTM 43N), {sentinel_res_m:.0f} m/px, uint8 RGB  \n')
        f.write(f'  Scene date: 17 October 2025  \n')
        f.write(f'  WGS84: 77.086–78.264°E, 30.605–31.618°N\n')
        f.write(f'**Altitude:** {altitude}m AGL\n')
        f.write(f'**TRN GSD:** {trn_gsd:.1f} m/px\n')
        f.write(f'**Ref tile:** {n_ref_px}×{n_ref_px} px ({footprint_m:.0f}m footprint)\n\n')

        f.write('## Results\n\n')
        f.write('| km | Status | Peak | Suitability | Rec | '
                'RefTexVar | RefRelief | CorrMag (m) | Query |\n')
        f.write('|----|--------|------|-------------|-----|'
                '----------|-----------|-------------|-------|\n')
        for r in results:
            f.write(
                f'| {r.km:.0f} | {r.status} | {r.peak_value:.4f} | '
                f'{r.suitability_score:.3f} | {r.suitability_rec} | '
                f'{r.ref_texture_var:.1f} | {r.ref_relief_m:.1f} | '
                f'{r.correction_mag_m:.2f} | {r.query_quality} |\n'
            )

        f.write('\n## Summary\n\n')
        f.write(f'- Accepted: {n_accepted}/{len(results)}\n')
        f.write(f'- Rejected: {n_rejected}/{len(results)}\n')
        f.write(f'- Suppressed: {n_suppressed}/{len(results)}\n')
        f.write(f'- Peak range: {min(peaks):.4f} – {max(peaks):.4f}\n')
        if peaks:
            f.write(f'- Mean peak (all): {float(np.mean(peaks)):.4f}\n')
        if non_supp:
            suggested = float(np.clip(np.percentile(non_supp, 10), 0.05, 0.50))
            f.write(f'- Suggested threshold (P10 non-suppressed): {suggested:.3f}\n')
        f.write(f'- Current threshold: {trn._min_peak:.3f}\n\n')

        f.write('## Baseline Comparison\n\n')
        f.write('| Validation | Peak range | Accepted |\n')
        f.write('|-----------|-----------|----------|\n')
        f.write('| OI-44 Cross-modal: RGB vs DEM hillshade | 0.0903–0.1136 | 0/12 |\n')
        f.write('| OI-45 Same-modal self-offset | 0.9874–0.9932 | 12/12 |\n')
        f.write(f'| OI-46 Real Sentinel-2 TCI | '
                f'{min(peaks):.4f}–{max(peaks):.4f} | {n_accepted}/12 |\n\n')

        f.write('## Interpretation and OI-46 Finding\n\n')

        # Interpret results based on acceptance count
        if n_accepted >= 8:
            finding = (
                'OI-46 CLOSED: Real Sentinel-2 TCI reference yields acceptable NCC '
                'peaks.  The Blender frames correlate with genuine Sentinel-2 imagery '
                'at operationally useful levels.  AD-01 same-modality validated with '
                'real satellite data.'
            )
        elif n_accepted >= 3:
            finding = (
                'OI-46 PARTIAL: Real Sentinel-2 TCI reference yields moderate NCC '
                'peaks on some frames.  Frames with low peaks may reflect texture '
                'mismatch between shimla_texture.png (DEM hillshade-colour) and '
                'genuine Sentinel-2 imagery at those positions.  Recommend '
                're-generating Blender terrain texture from the TCI tile crop '
                'to achieve full-corridor same-modality validation.'
            )
        else:
            finding = (
                'OI-46 OPEN: Real Sentinel-2 NCC peaks are below operational '
                'threshold across most or all frames.  Root cause: shimla_texture.png '
                'is a DEM hillshade-colour map (viz.hh_hillshade-color.png), NOT '
                'actual Sentinel-2 imagery.  The Blender frames do not represent '
                'Sentinel-2 visual content.  Action required: replace '
                'shimla_texture.png with a TCI crop matched to the corridor extent, '
                're-render Blender frames, re-run OI-46.'
            )

        f.write(finding + '\n\n')
        f.write(
            '**shimla_texture.png finding:** The texture used to render the Blender '
            'frames is `viz.hh_hillshade-color.png` from OpenTopography — a terrain '
            'elevation visualisation product, not an optical satellite image.  '
            'Genuine Sentinel-2 TRN requires the reference and query images to be '
            'from the same sensor class.  The Sentinel-2 TCI tiles now available '
            'provide the correct reference source; the Blender frames must be '
            're-rendered with a TCI-derived texture to complete the same-modality '
            'validation chain.\n'
        )


if __name__ == '__main__':
    main()
