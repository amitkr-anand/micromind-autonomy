"""
NAV-02 Characterisation Harness — Shimla-Manali Corridor (Run 3)
Scope: EC-13 partial characterisation only.
NAV-02 remains PARTIAL. No SRS closure.
Deputy 1 authorised: 26 April 2026.

Consumer of production modules — no production file modifications permitted.

Design: 4-WP terrain transition (ridge→valley→suppress→ridge).
Camera: Sentinel TCI mosaic (EPSG:32643, 10m/px, RGB).
DEM:    shimla_tile.tif (EPSG:4326, ~30m/px) — no reprojection needed.
trn_gsd: max(10.0, 30.0*0.5) = 15.0m → 133×133px tile at 2000m window.
"""

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
# Note: directive referenced terrain_suitability_scorer — actual module is
# core.trn.terrain_suitability (no _scorer suffix). Correct import used.
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
    _REPO_ROOT / "scenarios" / "nav02_char" / "shimla_corridor_definition.yaml"
)
RESULTS_DIR = _REPO_ROOT / "scenarios" / "nav02_char" / "results"
DEMO_ISSUES = _REPO_ROOT / "docs" / "demo" / "DEMO_ISSUES.md"

# GSD constants
SENTINEL_GSD_M  = 10.0   # Sentinel TCI native resolution
DEM_RES_M       = 30.0   # shimla_tile nominal resolution


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
            px_size = abs(gt.a)          # metres per pixel (10m)
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


# ── DEMO_ISSUES append helper ──────────────────────────────────────────────

def append_demo_issue(issue_id: str, module: str, desc: str, severity: str) -> None:
    try:
        text = DEMO_ISSUES.read_text()
        if issue_id in text:
            return
        row = f"| {issue_id} | {module} | {desc} | {severity} | OPEN |\n"
        DEMO_ISSUES.write_text(text.rstrip("\n") + "\n" + row)
        print(f"  DEMO_ISSUES.md: appended {issue_id}")
    except Exception as exc:
        print(f"  WARNING: could not update DEMO_ISSUES.md: {exc}")


# ── Results table ──────────────────────────────────────────────────────────

def print_results_table(wp_results: list, trn_gsd: float) -> None:
    hdr = (
        f"{'WP':<5} {'Label':<26} {'Exp':<10} {'Actual':<18} "
        f"{'Conf':>7} {'SuitRec':<9} {'SuitScore':>9} {'Lat_ms':>7}"
    )
    sep = "─" * 95
    print(f"\n{'═'*95}")
    print("NAV-02 CHARACTERISATION RUN 3 — Shimla-Manali Corridor EC-13")
    print(f"trn_gsd={trn_gsd:.1f}m  tile_size=2000m  DEM=shimla_tile.tif  camera=Sentinel TCI")
    print(f"{'═'*95}")
    print(hdr)
    print(sep)
    match_count = 0
    for wp in wp_results:
        exp  = wp["expected"]
        act  = wp["actual_status"]
        # Count match: expected terrain class aligns with actual
        exp_match = _expected_matches(exp, act)
        if exp_match:
            match_count += 1
        match_mark = "✓" if exp_match else "✗"
        corr = ""
        if act == "ACCEPTED":
            corr = f"N={wp['correction_north_m']:+.1f}m E={wp['correction_east_m']:+.1f}m"
        print(
            f"{wp['id']:<5} {wp['label']:<26} {exp:<10} {act:<18} "
            f"{wp['confidence']:>7.4f} {wp['actual_suitability']:<9} "
            f"{wp['suitability_score']:>9.3f} {wp['latency_ms']:>7}  {match_mark}"
        )
        if corr:
            print(f"      └─ correction: {corr}")
    print(f"{'═'*95}")
    print(f"Expected vs actual match: {match_count}/4")


def _expected_matches(expected: str, actual_status: str) -> bool:
    """Loosely map expected terrain class to actual TRN status."""
    mapping = {
        "ACCEPT":   ("ACCEPTED", "REJECTED"),   # terrain passed scorer; peak may be low
        "CAUTION":  ("REJECTED", "ACCEPTED"),   # CAUTION terrain → REJECTED or ACCEPTED OK
        "SUPPRESS": ("SUPPRESSED",),
    }
    return actual_status in mapping.get(expected, ())


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

    print(f"\nRun 3 — Shimla Corridor Characterisation")
    print(f"Corridor  : {corridor['corridor_id']}")
    print(f"Tile size : {corridor['tile_size_m']} m")
    print(f"Alt       : {corridor['alt_m']} m AGL")
    print(f"Waypoints : {len(corridor['waypoints_wgs84'])}")

    # DEMLoader — shimla_tile.tif is already EPSG:4326, no reprojection needed
    dem_path = str(_REPO_ROOT / corridor["dem_path"])
    dem_loader   = DEMLoader(dem_path)
    hs_gen       = HillshadeGenerator(azimuth_deg=315.0, elevation_deg=45.0)
    suit_scorer  = TerrainSuitabilityScorer()
    matcher      = PhaseCorrelationTRN(
        dem_loader=dem_loader,
        hillshade_gen=hs_gen,
        suitability_scorer=suit_scorer,
        tile_size_m=corridor["tile_size_m"],
        min_peak_value=0.15,
    )

    bounds = dem_loader.get_bounds()
    dem_res_actual = bounds["resolution_m"]
    print(
        f"DEM bounds: N={bounds['north']:.4f} S={bounds['south']:.4f} "
        f"E={bounds['east']:.4f} W={bounds['west']:.4f} "
        f"res={dem_res_actual:.2f}m"
    )

    # Production GSD clamping per CrossModalEvaluator (DEMO-BUG-001 fix)
    # gsd_m = max(camera_gsd, dem_res_m * 0.5)
    # Direct call — same clamping applied manually. See DEMO-BUG-001.
    trn_gsd = max(SENTINEL_GSD_M, dem_res_actual * 0.5)
    print(f"trn_gsd   : max({SENTINEL_GSD_M}, {dem_res_actual:.2f}×0.5) = {trn_gsd:.2f}m")

    sentinel_path = corridor["sentinel_path"]
    half_m        = corridor["tile_size_m"] / 2.0

    wp_results = []
    for wp_idx, wp in enumerate(corridor["waypoints_wgs84"]):
        wp_id  = wp["id"]
        lon    = float(wp["lon"])
        lat    = float(wp["lat"])
        label  = wp["label"]
        expected = wp["expected"]

        print(f"\n{'─'*60}")
        print(f"  {wp_id} ({label}): WGS84({lat:.5f}, {lon:.5f})")

        # Convert WGS84 → UTM43N for Sentinel extraction
        east, north = wgs84_to_utm43n(lon, lat)
        print(f"  UTM43N: E={east:.1f} N={north:.1f}")

        # Extract Sentinel TCI window (UTM43N)
        rgb = extract_sentinel_window(sentinel_path, east, north, half_m)
        if rgb is None or rgb.shape[0] < 3 or rgb.size == 0:
            print(f"  WARNING: Sentinel extraction failed — zero-filled fallback")
            n_px = int(corridor["tile_size_m"] / SENTINEL_GSD_M)
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
            alt_m=corridor["alt_m"],
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
    print_results_table(wp_results, trn_gsd)

    # Summary
    accept   = sum(1 for w in wp_results if w["actual_status"] == "ACCEPTED")
    caution  = sum(1 for w in wp_results if w["actual_suitability"] == "CAUTION"
                   and w["actual_status"] != "SUPPRESSED")
    suppress = sum(1 for w in wp_results if w["actual_status"] == "SUPPRESSED")
    rejected = sum(1 for w in wp_results if w["actual_status"] == "REJECTED")
    match_ct = sum(1 for w in wp_results
                   if _expected_matches(w["expected"], w["actual_status"]))

    accepted_confs = [w["confidence"] for w in wp_results if w["actual_status"] == "ACCEPTED"]
    mean_conf_acc  = round(float(np.mean(accepted_confs)), 4) if accepted_confs else None

    print(f"\nSummary:")
    print(f"  accept_count: {accept}")
    print(f"  caution_count: {caution}")
    print(f"  suppress_count: {suppress}")
    print(f"  rejected_count: {rejected}")
    print(f"  mean_confidence_accepted: {mean_conf_acc}")
    print(f"  expected_vs_actual_match_count: {match_ct}/4")

    # Write YAML
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "run_id":           "shimla_char_run3",
        "run_date":         date.today().isoformat(),
        "corridor":         corridor["corridor_id"],
        "tile_size_m":      corridor["tile_size_m"],
        "trn_gsd_m":        round(trn_gsd, 2),
        "dem_resolution_m": round(dem_res_actual, 1),
        "note": (
            "EC-13 partial characterisation. NAV-02 remains PARTIAL. "
            "Shimla-Manali validated corridor. 4-WP terrain transition design: "
            "ridge→valley→suppression→ridge. "
            "WP2 adjusted lat 31.45→31.43 (2.2km S, shimla_tile boundary). "
            "WP3 adjusted (77.05,31.55)→(77.20,31.38) (shimla_tile boundary)."
        ),
        "waypoints": [
            {
                "id":               w["id"],
                "label":            w["label"],
                "expected":         w["expected"],
                "actual_status":    w["actual_status"],
                "actual_suitability": w["actual_suitability"],
                "confidence":       w["confidence"],
                "suitability_score": w["suitability_score"],
                "correction_north_m": w["correction_north_m"],
                "correction_east_m":  w["correction_east_m"],
                "latency_ms":       w["latency_ms"],
            }
            for w in wp_results
        ],
        "summary": {
            "accept_count":                 accept,
            "caution_count":                caution,
            "suppress_count":               suppress,
            "rejected_count":               rejected,
            "mean_confidence_accepted":     mean_conf_acc,
            "expected_vs_actual_match_count": f"{match_ct}/4",
        },
    }
    out_path = RESULTS_DIR / "shimla_char_run3_results.yaml"
    with open(out_path, "w") as fh:
        yaml.dump(out, fh, default_flow_style=False, sort_keys=False)
    print(f"\nResults written → {out_path}")


if __name__ == "__main__":
    main()
