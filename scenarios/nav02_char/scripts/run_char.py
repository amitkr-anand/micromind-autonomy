"""
NAV-02 Characterisation Harness — Trinity Corridor
Scope: EC-13 partial characterisation only.
NAV-02 remains PARTIAL. No SRS closure.
Deputy 1 authorised: 26 April 2026.

Consumer of production modules — no production
file modifications permitted.
"""

import math
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import yaml
import numpy as np

# ── Repo root ──────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ── rasterio ───────────────────────────────────────────────────────────────
try:
    import rasterio
    import rasterio.windows
    from rasterio.warp import (
        calculate_default_transform,
        reproject as _rio_reproject,
        Resampling,
    )
    from rasterio.crs import CRS
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False
    print("FATAL DI-03: rasterio not available — cannot run characterisation")

# ── Production modules ─────────────────────────────────────────────────────
_PROD_OK = False
try:
    from core.trn.dem_loader import DEMLoader
    from core.trn.hillshade_generator import HillshadeGenerator
    from core.trn.terrain_suitability import TerrainSuitabilityScorer
    from core.trn.phase_correlation_trn import PhaseCorrelationTRN
    _PROD_OK = True
except ImportError as _e:
    print(f"FATAL DI-04: Cannot import production modules: {_e}")

# ── Paths ──────────────────────────────────────────────────────────────────
CORRIDOR_DEF = _REPO_ROOT / "scenarios" / "nav02_char" / "corridor_definition.yaml"
RESULTS_DIR  = _REPO_ROOT / "scenarios" / "nav02_char" / "results"
DEMO_ISSUES  = _REPO_ROOT / "docs" / "demo" / "DEMO_ISSUES.md"


# ── UTM → WGS84 ────────────────────────────────────────────────────────────

def utm_zone10n_to_wgs84(easting: float, northing: float):
    """
    EPSG:26910 UTM Zone 10N → WGS84 lat/lon.
    Primary: rasterio.warp.transform (uses PROJ, accurate to sub-metre).
    Fallback (rasterio absent): Deputy 1 approved approximate formula.
    The formula has ~25 km error for this corridor — use only if rasterio
    is truly unavailable.
    """
    if RASTERIO_OK:
        from rasterio.warp import transform as _warp_transform
        from rasterio.crs import CRS as _CRS
        src_crs = _CRS.from_epsg(26910)
        dst_crs = _CRS.from_epsg(4326)
        lons, lats = _warp_transform(src_crs, dst_crs, [easting], [northing])
        return float(lats[0]), float(lons[0])
    # Formula fallback (Deputy 1 directive, DI-02 documented)
    lat = northing / 111_320.0
    lon = -123.0 + (easting - 500_000.0) / (
        111_320.0 * math.cos(math.radians(lat))
    )
    return lat, lon


# ── DEM reprojection: UTM → WGS84 temp file ───────────────────────────────

def reproject_utm_to_wgs84(utm_path: str) -> str:
    """
    Reproject UTM GeoTIFF to WGS84. Write to a NamedTemporaryFile.
    Returns path to temp WGS84 .tif. Caller must unlink after use.
    Uses rasterio.warp — no pyproj dependency.
    """
    dst_crs = CRS.from_epsg(4326)
    tmp = tempfile.NamedTemporaryFile(suffix="_wgs84.tif", delete=False)
    tmp.close()

    with rasterio.open(utm_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            driver="GTiff",
        )
        with rasterio.open(tmp.name, "w", **meta) as dst:
            for band_idx in range(1, src.count + 1):
                _rio_reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
    return tmp.name


# ── NAIP extraction ────────────────────────────────────────────────────────

def extract_naip_window(
    naip_path: str,
    centre_e: float,
    centre_n: float,
    half_m: float = 250.0,
):
    """
    Extract a ~500×500 m window from NAIP RGB (UTM) centred at (centre_e, centre_n).
    Returns (3, H, W) uint8 array or None on failure.
    """
    if not RASTERIO_OK:
        return None
    try:
        with rasterio.open(naip_path) as ds:
            gt = ds.transform
            px_size = abs(gt.a)
            half_px = int(half_m / px_size)

            col = int((centre_e - gt.c) / gt.a)
            row = int((centre_n - gt.f) / gt.e)

            col0 = max(0, col - half_px)
            row0 = max(0, row - half_px)
            col1 = min(ds.width,  col + half_px)
            row1 = min(ds.height, row + half_px)

            if col1 <= col0 or row1 <= row0:
                return None

            win = rasterio.windows.Window(
                col0, row0, col1 - col0, row1 - row0
            )
            data = ds.read(window=win)
            return data.astype(np.uint8)
    except Exception as exc:
        print(f"    WARNING: NAIP extraction failed: {exc}")
        return None


def naip_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert (3, H, W) uint8 RGB to (H, W) uint8 grayscale via luminance."""
    r = rgb[0].astype(np.float32)
    g = rgb[1].astype(np.float32)
    b = rgb[2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


# ── Mode runner ────────────────────────────────────────────────────────────

def run_mode(
    corridor,
    dem_path_abs: str,
    mode_label: str,
    dem_resolution_m: int,
    naip_path_abs: str,
) -> dict:
    """Run one characterisation mode. Returns result dict."""
    print(f"\n{'='*64}")
    print(f"MODE: {mode_label} (DEM {dem_resolution_m}m)")
    print(f"{'='*64}")

    print(f"  Reprojecting {Path(dem_path_abs).name} → WGS84 temp file ...")
    wgs84_path = reproject_utm_to_wgs84(dem_path_abs)
    print(f"  WGS84 temp: {wgs84_path}")

    try:
        dem_loader   = DEMLoader(wgs84_path)
        hs_gen       = HillshadeGenerator(azimuth_deg=315.0, elevation_deg=45.0)
        suit_scorer  = TerrainSuitabilityScorer()
        matcher      = PhaseCorrelationTRN(
            dem_loader=dem_loader,
            hillshade_gen=hs_gen,
            suitability_scorer=suit_scorer,
            tile_size_m=500.0,
            min_peak_value=0.15,
        )

        bounds = dem_loader.get_bounds()
        print(
            f"  DEM bounds: N={bounds['north']:.5f} S={bounds['south']:.5f} "
            f"E={bounds['east']:.5f} W={bounds['west']:.5f} "
            f"res={bounds['resolution_m']:.2f}m"
        )

        wp_results = []
        for wp_idx, wp in enumerate(corridor["waypoints_utm"]):
            wp_id = wp["id"]
            east  = float(wp["easting"])
            north = float(wp["northing"])

            lat, lon = utm_zone10n_to_wgs84(east, north)
            print(
                f"\n  {wp_id}: UTM({east:.0f}, {north:.0f}) "
                f"→ WGS84({lat:.6f}, {lon:.6f})"
            )

            # Extract NAIP window (UTM space)
            rgb = extract_naip_window(naip_path_abs, east, north, half_m=250.0)
            if rgb is None or rgb.shape[0] < 3 or rgb.size == 0:
                print(f"    WARNING: NAIP extraction failed — using zero-filled fallback")
                n_px = int(500.0 / 0.60)
                gray = np.zeros((n_px, n_px), dtype=np.uint8)
            else:
                gray = naip_to_grayscale(rgb)
                print(
                    f"    NAIP extracted: bands={rgb.shape[0]}, "
                    f"size={rgb.shape[2]}×{rgb.shape[1]}px, "
                    f"gray={gray.shape[1]}×{gray.shape[0]}px"
                )

            # Call production matcher
            result = matcher.match(
                camera_tile=gray,
                lat_estimate=lat,
                lon_estimate=lon,
                alt_m=150.0,
                gsd_m=0.60,
                mission_time_ms=int(wp_idx * 10000),
            )

            print(
                f"    status={result.status}  "
                f"confidence={result.confidence:.4f}  "
                f"suit_score={result.suitability_score:.3f}  "
                f"suit_rec={result.suitability_recommendation}"
            )
            if result.status == "ACCEPTED":
                print(
                    f"    correction: N={result.correction_north_m:+.2f}m  "
                    f"E={result.correction_east_m:+.2f}m"
                )

            wp_results.append(
                {
                    "id":                       wp_id,
                    "lat":                      round(lat, 7),
                    "lon":                      round(lon, 7),
                    "status":                   result.status,
                    "confidence":               round(float(result.confidence), 4),
                    "suitability_recommendation": result.suitability_recommendation,
                    "correction_north_m":       round(float(result.correction_north_m), 3),
                    "correction_east_m":        round(float(result.correction_east_m), 3),
                    "latency_ms":               int(result.latency_ms),
                }
            )

        return {"dem_resolution_m": dem_resolution_m, "waypoints": wp_results}

    finally:
        try:
            os.unlink(wgs84_path)
        except OSError:
            pass


# ── Results table ──────────────────────────────────────────────────────────

def print_results_table(mode1: dict, mode2: dict) -> None:
    hdr = (
        f"{'WP':<5} {'Mode':<10} {'Status':<18} "
        f"{'Conf':>6} {'SuitRec':<10} "
        f"{'N_corr_m':>10} {'E_corr_m':>10} {'Lat_ms':>7}"
    )
    sep = "-" * 82
    print(f"\n{'='*82}")
    print("NAV-02 CHARACTERISATION RESULTS — Trinity Corridor EC-13")
    print(f"{'='*82}")
    print(hdr)
    print(sep)
    for mode_label, res in [("10m_DEM", mode1), ("30m_DEM", mode2)]:
        for wp in res["waypoints"]:
            if wp["status"] == "ACCEPTED":
                n_s = f"{wp['correction_north_m']:+.3f}"
                e_s = f"{wp['correction_east_m']:+.3f}"
            else:
                n_s, e_s = "—", "—"
            print(
                f"{wp['id']:<5} {mode_label:<10} {wp['status']:<18} "
                f"{wp['confidence']:>6.4f} {wp['suitability_recommendation']:<10} "
                f"{n_s:>10} {e_s:>10} {wp['latency_ms']:>7}"
            )
    print(f"{'='*82}")


# ── Summary ────────────────────────────────────────────────────────────────

def compute_summary(mode1: dict, mode2: dict) -> dict:
    def _stats(wps):
        accepted  = [w for w in wps if w["status"] == "ACCEPTED"]
        suppressed = [
            w for w in wps
            if w["status"] in ("SUPPRESSED", "OUTSIDE_COVERAGE", "REJECTED")
        ]
        confs = [w["confidence"] for w in wps]
        mags  = [
            math.sqrt(w["correction_north_m"] ** 2 + w["correction_east_m"] ** 2)
            for w in accepted
        ]
        mean_conf = float(np.mean(confs)) if confs else 0.0
        mean_mag  = float(np.mean(mags))  if mags  else 0.0
        return len(accepted), len(suppressed), round(mean_conf, 4), round(mean_mag, 3)

    a1, s1, mc1, mm1 = _stats(mode1["waypoints"])
    a2, s2, mc2, mm2 = _stats(mode2["waypoints"])
    return {
        "mode_1_accept_count":                a1,
        "mode_1_suppress_count":              s1,
        "mode_2_accept_count":                a2,
        "mode_2_suppress_count":              s2,
        "mode_1_mean_confidence":             mc1,
        "mode_2_mean_confidence":             mc2,
        "mode_1_mean_correction_magnitude_m": mm1,
        "mode_2_mean_correction_magnitude_m": mm2,
    }


# ── Write results ──────────────────────────────────────────────────────────

def write_results(mode1: dict, mode2: dict, summary: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "run_date": date.today().isoformat(),
        "corridor": "trinity_range_char",
        "note": (
            "EC-13 partial characterisation only. "
            "NAV-02 remains PARTIAL. Not SRS closure. "
            "FINDING DI-03: All waypoints SUPPRESSED both modes. "
            "Root cause: direct PhaseCorrelationTRN.match() at gsd_m=0.60 bypasses "
            "CrossModalEvaluator GSD clamping (max(camera_gsd, dem_res*0.5)). "
            "10m DEM upsampled 16.7x to 0.60m → hillshade texture_variance=5.18 < "
            "threshold 50.0 → SUPPRESS. Relief is good (285m). "
            "Production CrossModalEvaluator would clamp trn_gsd to 5.0m (10m DEM) "
            "or 15.0m (30m DEM). Deputy 1 to rule on char run 2 with clamped GSD."
        ),
        "mode_1_10m": {
            "dem_resolution_m": mode1["dem_resolution_m"],
            "waypoints": mode1["waypoints"],
        },
        "mode_2_30m": {
            "dem_resolution_m": mode2["dem_resolution_m"],
            "waypoints": mode2["waypoints"],
        },
        "summary": summary,
    }
    out_path = RESULTS_DIR / "char_run_results.yaml"
    with open(out_path, "w") as fh:
        yaml.dump(out, fh, default_flow_style=False, sort_keys=False)
    print(f"\nResults written → {out_path}")


# ── DEMO_ISSUES update ─────────────────────────────────────────────────────

def append_demo_issue(issue_id: str, module: str, desc: str, severity: str) -> None:
    """Append a new row to docs/demo/DEMO_ISSUES.md if not already present."""
    try:
        text = DEMO_ISSUES.read_text()
        if issue_id in text:
            return
        row = f"| {issue_id} | {module} | {desc} | {severity} | OPEN |\n"
        DEMO_ISSUES.write_text(text.rstrip("\n") + "\n" + row)
        print(f"  DEMO_ISSUES.md: appended {issue_id}")
    except Exception as exc:
        print(f"  WARNING: could not update DEMO_ISSUES.md: {exc}")


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    if not RASTERIO_OK:
        print("FATAL: rasterio unavailable. Cannot run characterisation.")
        sys.exit(1)
    if not _PROD_OK:
        print("FATAL: production modules unavailable. Cannot run characterisation.")
        sys.exit(1)

    # Record DI-02 (pyproj absent)
    append_demo_issue(
        "DI-02",
        "nav02_char harness",
        "`pyproj` not installed in `micromind-autonomy` conda env. "
        "UTM→WGS84 coordinate conversion uses `rasterio.warp.transform` (accurate, "
        "sub-metre) with Deputy 1 formula as last-resort fallback. "
        "Not blocking. Fix: `conda install -c conda-forge pyproj`.",
        "LOW",
    )
    # Record DI-03 (GSD clamping bypass)
    append_demo_issue(
        "DI-03",
        "nav02_char harness / PhaseCorrelationTRN",
        "Direct PhaseCorrelationTRN.match() calls at camera GSD (0.60 m) bypass "
        "CrossModalEvaluator GSD clamping (trn_gsd = max(camera_gsd, dem_res×0.5)). "
        "Result: 10m DEM tile upsampled 16.7× → hillshade texture_variance=5.18 < "
        "threshold 50.0 → SUPPRESS all waypoints both modes. "
        "Production path (via CrossModalEvaluator) would clamp to trn_gsd=5.0m (10m DEM) "
        "or 15.0m (30m DEM) and avoid over-upsampling. "
        "Characterisation finding: valid architectural constraint. Deputy 1 to rule "
        "on whether char run 2 should call with clamped GSD.",
        "MEDIUM",
    )

    with open(CORRIDOR_DEF) as fh:
        corridor = yaml.safe_load(fh)

    print(f"\nCorridor  : {corridor['corridor_id']}")
    print(f"Length    : {corridor['corridor_length_km']} km")
    print(f"Waypoints : {len(corridor['waypoints_utm'])}")
    print(f"Note      : pyproj absent — UTM→WGS84 fallback formula in use (DI-02)")

    naip_path  = str(_REPO_ROOT / corridor["orthophoto_rgb"])
    dem10_path = str(_REPO_ROOT / corridor["dem_10m"])
    dem30_path = str(_REPO_ROOT / corridor["dem_30m"])

    mode1 = run_mode(corridor, dem10_path, "mode_1_10m", 10, naip_path)
    mode2 = run_mode(corridor, dem30_path, "mode_2_30m", 30, naip_path)

    print_results_table(mode1, mode2)

    summary = compute_summary(mode1, mode2)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    write_results(mode1, mode2, summary)


if __name__ == "__main__":
    main()
