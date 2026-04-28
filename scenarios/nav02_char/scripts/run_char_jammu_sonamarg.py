"""
NAV-02 Characterisation Harness — Jammu-Sonamarg Corridor (Run 4)
Scope: EC-13 partial characterisation only.
NAV-02 remains PARTIAL. No SRS closure.
Deputy 1 authorised: 28 April 2026.

Consumer of production modules — no production file modifications permitted.

Design: 6-WP confirmed-good imagery corridor (WP_GAP_START skipped).
Camera: Sentinel TCI per-tile (EPSG:32643, 10m/px, RGB) — one TCI file per WP.
DEM:    TILE1+TILE2 merged via from_directory() (EPSG:4326, ~28m/px nominal COP30).
trn_gsd: max(10.0, dem_res_actual * 0.5)
NOTE: WP00 (lat=32.73N) is south of merged DEM boundary (32.80N) — expect OUTSIDE_COVERAGE.
"""

from __future__ import annotations

import math
import os
import sys
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
    from rasterio.warp import transform as _warp_transform
    from rasterio.crs import CRS
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False
    print("FATAL: rasterio not available")

# ── Production modules ─────────────────────────────────────────────────────
_PROD_OK = False
try:
    from core.trn.dem_loader import DEMLoader
    from core.trn.hillshade_generator import HillshadeGenerator
    from core.trn.terrain_suitability import TerrainSuitabilityScorer
    from core.trn.phase_correlation_trn import PhaseCorrelationTRN
    _PROD_OK = True
except ImportError as _e:
    print(f"FATAL: cannot import production modules: {_e}")

# ── Paths ──────────────────────────────────────────────────────────────────
CORRIDOR_DEF = (
    _REPO_ROOT / "scenarios" / "nav02_char" / "jammu_sonamarg_corridor_definition.yaml"
)
RESULTS_DIR = _REPO_ROOT / "scenarios" / "nav02_char" / "results"
DEM_DIR     = str(_REPO_ROOT / "data" / "terrain" / "Jammu_leh_corridor_COP30")

# TCI tile map: MGRS tile code → absolute path
TCI_BASE = Path("/home/mmuser/micromind_data/raw/imagery/sentinel2/Jammu_Leh_TCI")
TCI_MAP = {
    "T43SDS": str(TCI_BASE / "T43SDS_TCI_10m.tif"),
    "T43SES": str(TCI_BASE / "T43SES_TCI_10m.tif"),
    "T43SDT": str(TCI_BASE / "T43SDT_TCI_10m.tif"),
    "T43SET": str(TCI_BASE / "T43SET_TCI_10m.tif"),
}

# GSD constants
SENTINEL_GSD_M = 10.0   # Sentinel TCI native resolution
TILE_SIZE_M    = 2000.0
ALT_M          = 150.0

# Waypoints to run — skip WP_GAP_START (INS-only suppress zone)
SKIP_IDS = {"WP_GAP_START"}


# ── Coordinate conversion ──────────────────────────────────────────────────

def wgs84_to_utm43n(lon: float, lat: float):
    """WGS84 → EPSG:32643 (UTM Zone 43N) using rasterio.warp.transform."""
    xs, ys = _warp_transform(
        CRS.from_epsg(4326), CRS.from_epsg(32643), [lon], [lat]
    )
    return float(xs[0]), float(ys[0])


# ── Sentinel tile extraction ───────────────────────────────────────────────

def extract_sentinel_window(
    sentinel_path: str,
    centre_e: float,
    centre_n: float,
    half_m: float = 1000.0,
):
    """
    Extract a ~2000×2000m window from Sentinel TCI (UTM43N) centred at
    (centre_e, centre_n). Returns (3, H, W) uint8 array or None on failure.
    """
    try:
        with rasterio.open(sentinel_path) as ds:
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
                print(f"    WARNING: Sentinel window empty at ({centre_e:.0f}, {centre_n:.0f})")
                return None

            win = rasterio.windows.Window(
                col0, row0, col1 - col0, row1 - row0
            )
            data = ds.read(window=win)
            return data.astype(np.uint8)
    except Exception as exc:
        print(f"    WARNING: Sentinel extraction failed: {exc}")
        return None


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert (3, H, W) uint8 RGB to (H, W) uint8 grayscale via luminance."""
    r = rgb[0].astype(np.float32)
    g = rgb[1].astype(np.float32)
    b = rgb[2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


# ── Expected classification ────────────────────────────────────────────────

def quality_to_expected(quality: str) -> str:
    """Map waypoint quality field to expected TRN suitability class."""
    q = quality.upper()
    if "SUPPRESS" in q:
        return "SUPPRESS"
    if "CAUTION" in q:
        return "CAUTION"
    return "ACCEPT"


def _expected_matches(expected: str, actual_status: str) -> bool:
    """Loosely map expected terrain class to actual TRN status."""
    mapping = {
        "ACCEPT":   ("ACCEPTED", "REJECTED", "OUTSIDE_COVERAGE"),
        "CAUTION":  ("REJECTED", "ACCEPTED"),
        "SUPPRESS": ("SUPPRESSED",),
    }
    return actual_status in mapping.get(expected, ())


# ── Results table ──────────────────────────────────────────────────────────

def print_results_table(wp_results: list, trn_gsd: float, n_wps: int) -> None:
    hdr = (
        f"{'WP':<12} {'Label':<24} {'Exp':<8} {'Actual':<18} "
        f"{'Conf':>7} {'SuitRec':<9} {'SuitScore':>9} {'Lat_ms':>7}"
    )
    sep = "─" * 100
    print(f"\n{'═'*100}")
    print("NAV-02 CHARACTERISATION RUN 4 — Jammu-Sonamarg Corridor EC-13")
    print(f"trn_gsd={trn_gsd:.2f}m  tile_size={TILE_SIZE_M:.0f}m  DEM=JL TILE1+TILE2  camera=Sentinel TCI (per-WP)")
    print(f"{'═'*100}")
    print(hdr)
    print(sep)
    match_count = 0
    for wp in wp_results:
        exp = wp["expected"]
        act = wp["actual_status"]
        exp_match = _expected_matches(exp, act)
        if exp_match:
            match_count += 1
        mark = "✓" if exp_match else "✗"
        corr = ""
        if act == "ACCEPTED":
            corr = f"N={wp['correction_north_m']:+.1f}m E={wp['correction_east_m']:+.1f}m"
        conf_str = f"{wp['confidence']:.4f}" if wp['confidence'] is not None else "  n/a "
        suit_str = f"{wp['suitability_score']:.3f}" if wp['suitability_score'] is not None else "   n/a"
        print(
            f"{wp['id']:<12} {wp['label']:<24} {exp:<8} {act:<18} "
            f"{conf_str:>7} {wp['actual_suitability']:<9} "
            f"{suit_str:>9} {wp['latency_ms']:>7}  {mark}"
        )
        if corr:
            print(f"             └─ correction: {corr}")
    print(f"{'═'*100}")
    print(f"Expected vs actual match: {match_count}/{n_wps}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    if not RASTERIO_OK:
        print("FATAL: rasterio unavailable.")
        sys.exit(1)
    if not _PROD_OK:
        print("FATAL: production modules unavailable.")
        sys.exit(1)

    with open(CORRIDOR_DEF) as fh:
        corridor = yaml.safe_load(fh)

    print(f"\nRun 4 — Jammu-Sonamarg Corridor Characterisation")
    print(f"Corridor  : {corridor['corridor_id']}")
    print(f"Tile size : {TILE_SIZE_M:.0f} m")
    print(f"Alt       : {ALT_M} m AGL")

    # Filter to confirmed-good waypoints only
    all_wps = [w for w in corridor["waypoints"] if w["id"] not in SKIP_IDS]
    print(f"Waypoints : {len(all_wps)} (WP_GAP_START excluded)")

    # DEMLoader — merge TILE1+TILE2 via from_directory() (symlinks at top level)
    print(f"\nLoading merged DEM from {DEM_DIR} ...")
    dem_loader  = DEMLoader.from_directory(DEM_DIR)
    hs_gen      = HillshadeGenerator(azimuth_deg=315.0, elevation_deg=45.0)
    suit_scorer = TerrainSuitabilityScorer()
    matcher     = PhaseCorrelationTRN(
        dem_loader=dem_loader,
        hillshade_gen=hs_gen,
        suitability_scorer=suit_scorer,
        tile_size_m=TILE_SIZE_M,
        min_peak_value=0.15,
    )

    bounds = dem_loader.get_bounds()
    dem_res_actual = bounds["resolution_m"]
    print(
        f"DEM bounds: N={bounds['north']:.4f} S={bounds['south']:.4f} "
        f"E={bounds['east']:.4f} W={bounds['west']:.4f} "
        f"res={dem_res_actual:.2f}m"
    )

    trn_gsd = max(SENTINEL_GSD_M, dem_res_actual * 0.5)
    print(f"trn_gsd   : max({SENTINEL_GSD_M}, {dem_res_actual:.2f}×0.5) = {trn_gsd:.2f}m")

    half_m = TILE_SIZE_M / 2.0
    wp_results = []

    for wp_idx, wp in enumerate(all_wps):
        wp_id    = wp["id"]
        lon      = float(wp["lon"])
        lat      = float(wp["lat"])
        label    = wp["name"]
        tile_key = wp["tile"]
        expected = quality_to_expected(wp.get("quality", "GOOD"))

        print(f"\n{'─'*65}")
        print(f"  {wp_id} ({label}): WGS84({lat:.5f}, {lon:.5f})  tile={tile_key}")

        # Convert WGS84 → UTM43N for Sentinel extraction
        east, north = wgs84_to_utm43n(lon, lat)
        print(f"  UTM43N: E={east:.1f} N={north:.1f}")

        # Select and extract from per-waypoint TCI tile
        sentinel_path = TCI_MAP.get(tile_key)
        if sentinel_path is None:
            print(f"  WARNING: no TCI tile mapped for tile_key={tile_key!r} — zero fallback")
            rgb = None
        else:
            rgb = extract_sentinel_window(sentinel_path, east, north, half_m)

        if rgb is None or rgb.shape[0] < 3 or rgb.size == 0:
            print(f"  WARNING: Sentinel extraction failed — zero-filled fallback")
            n_px = int(TILE_SIZE_M / SENTINEL_GSD_M)
            gray = np.zeros((n_px, n_px), dtype=np.uint8)
        else:
            gray = rgb_to_grayscale(rgb)
            print(
                f"  Sentinel window: bands={rgb.shape[0]}, "
                f"{rgb.shape[2]}×{rgb.shape[1]}px → gray {gray.shape[1]}×{gray.shape[0]}px"
            )
            mean_px = float(np.mean(gray))
            std_px  = float(np.std(gray))
            print(f"  Gray stats: mean={mean_px:.1f} std={std_px:.1f}")

        # Call production matcher
        result = matcher.match(
            camera_tile=gray,
            lat_estimate=lat,
            lon_estimate=lon,
            alt_m=ALT_M,
            gsd_m=trn_gsd,
            mission_time_ms=wp_idx * 30000,
        )

        print(
            f"  status={result.status}  "
            f"confidence={result.confidence:.4f}  "
            f"suit_score={result.suitability_score:.3f}  "
            f"suit_rec={result.suitability_recommendation}"
        )
        if result.status == "ACCEPTED":
            print(
                f"  correction: N={result.correction_north_m:+.2f}m  "
                f"E={result.correction_east_m:+.2f}m"
            )
        print(f"  expected={expected}  match={_expected_matches(expected, result.status)}")

        wp_results.append({
            "id":                 wp_id,
            "label":              label,
            "tile":               tile_key,
            "corridor_km":        wp.get("corridor_km"),
            "expected":           expected,
            "actual_status":      result.status,
            "actual_suitability": result.suitability_recommendation,
            "confidence":         round(float(result.confidence), 4),
            "suitability_score":  round(float(result.suitability_score), 4),
            "correction_north_m": round(float(result.correction_north_m), 3)
                                  if result.status == "ACCEPTED" else None,
            "correction_east_m":  round(float(result.correction_east_m), 3)
                                  if result.status == "ACCEPTED" else None,
            "latency_ms":         int(result.latency_ms),
        })

    # Print table
    print_results_table(wp_results, trn_gsd, len(wp_results))

    # Summary
    accept         = sum(1 for w in wp_results if w["actual_status"] == "ACCEPTED")
    suppress       = sum(1 for w in wp_results if w["actual_status"] == "SUPPRESSED")
    rejected       = sum(1 for w in wp_results if w["actual_status"] == "REJECTED")
    outside        = sum(1 for w in wp_results if w["actual_status"] == "OUTSIDE_COVERAGE")
    match_ct       = sum(1 for w in wp_results
                         if _expected_matches(w["expected"], w["actual_status"]))
    accepted_confs = [w["confidence"] for w in wp_results if w["actual_status"] == "ACCEPTED"]
    mean_conf_acc  = round(float(np.mean(accepted_confs)), 4) if accepted_confs else None

    print(f"\nSummary:")
    print(f"  accept_count:                 {accept}")
    print(f"  suppress_count:               {suppress}")
    print(f"  rejected_count:               {rejected}")
    print(f"  outside_coverage_count:       {outside}")
    print(f"  mean_confidence_accepted:     {mean_conf_acc}")
    print(f"  expected_vs_actual_match_count: {match_ct}/{len(wp_results)}")

    # Write YAML
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "run_id":               "jammu_sonamarg_char_run4",
        "run_date":             date.today().isoformat(),
        "corridor":             corridor["corridor_id"],
        "tile_size_m":          TILE_SIZE_M,
        "trn_gsd_m":            round(trn_gsd, 2),
        "dem_resolution_m":     round(dem_res_actual, 1),
        "dem_tiles":            "TILE1+TILE2 merged (from_directory)",
        "tci_tiles":            "T43SDS/T43SES/T43SDT/T43SET per-waypoint",
        "note": (
            "EC-13 partial characterisation. NAV-02 remains PARTIAL. Not SRS closure. "
            "6-WP confirmed-good imagery corridor. WP_GAP_START excluded (INS-only zone). "
            "WP00 (lat=32.73N) is south of merged DEM boundary (32.80N) — OUTSIDE_COVERAGE expected. "
            "Cross-modal TRN: DEM hillshade vs Sentinel TCI RGB. "
            "Peak ceiling documented: 0.09-0.11 cross-modal (Gate 6 finding OI-44 CLOSED ARCHITECTURAL)."
        ),
        "waypoints": [
            {
                "id":                 w["id"],
                "label":              w["label"],
                "tile":               w["tile"],
                "corridor_km":        w["corridor_km"],
                "expected":           w["expected"],
                "actual_status":      w["actual_status"],
                "actual_suitability": w["actual_suitability"],
                "confidence":         w["confidence"],
                "suitability_score":  w["suitability_score"],
                "correction_north_m": w["correction_north_m"],
                "correction_east_m":  w["correction_east_m"],
                "latency_ms":         w["latency_ms"],
            }
            for w in wp_results
        ],
        "summary": {
            "accept_count":                   accept,
            "suppress_count":                 suppress,
            "rejected_count":                 rejected,
            "outside_coverage_count":         outside,
            "mean_confidence_accepted":       mean_conf_acc,
            "expected_vs_actual_match_count": f"{match_ct}/{len(wp_results)}",
        },
    }
    out_path = RESULTS_DIR / "jammu_sonamarg_char_run4_results.yaml"
    with open(out_path, "w") as fh:
        yaml.dump(out, fh, default_flow_style=False, sort_keys=False)
    print(f"\nResults written → {out_path}")


if __name__ == "__main__":
    main()
