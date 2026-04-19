"""
LightGlue IPC server — runs in hil-h3 (Python 3.10) environment.

Listens on a Unix domain socket.  Handles one request at a time
(L2 correction fires every 2 km; no concurrency required).

Startup: launched automatically by the client when the socket is absent
or unresponsive.  May also be started manually:
    /path/to/hil-h3/python3 -m integration.lightglue_bridge.server

Protocol: newline-delimited JSON.  Each message is a single JSON object
followed by '\\n'.  See docs/interfaces/L2_LIGHTGLUE_IPC.md.
"""
from __future__ import annotations

import json
import logging
import math
import os
import signal
import socket
import sys
import time
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Bootstrap: allow running as both module and direct script
# ---------------------------------------------------------------------------
if __package__ is None or __package__ == "":
    # Direct invocation: add repo root to path so relative imports work
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(_here))
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    from integration.lightglue_bridge.config import (  # type: ignore
        SOCKET_PATH, TILE_DIR, MAX_FEATURES, MATCH_THRESHOLD,
        MIN_INLIERS, TILE_PATCH_RADIUS_M, TILE_GSD_M,
    )
    from integration.lightglue_bridge.tile_resolver import resolve as resolve_tile  # type: ignore
else:
    from .config import (
        SOCKET_PATH, TILE_DIR, MAX_FEATURES, MATCH_THRESHOLD,
        MIN_INLIERS, TILE_PATCH_RADIUS_M, TILE_GSD_M,
    )
    from .tile_resolver import resolve as resolve_tile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [lightglue-server] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LightGlue import — graceful stub fallback when not installed
# ---------------------------------------------------------------------------
_LIGHTGLUE_AVAILABLE = False
try:
    import torch
    import numpy as np
    import cv2
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image, rbd
    _LIGHTGLUE_AVAILABLE = True
    log.info("LightGlue available — GPU: %s", torch.cuda.is_available())
except ImportError as _e:
    log.warning("LightGlue not available (%s) — running in stub mode", _e)

# Attempt rasterio for tile patch extraction
_RASTERIO_AVAILABLE = False
try:
    import rasterio
    from rasterio.windows import from_bounds
    _RASTERIO_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Model (loaded once at startup, cached for all subsequent calls)
# ---------------------------------------------------------------------------
_extractor = None
_matcher = None


def _load_models() -> None:
    global _extractor, _matcher
    if not _LIGHTGLUE_AVAILABLE:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading SuperPoint+LightGlue on %s …", device)
    t0 = time.monotonic()
    _extractor = SuperPoint(max_num_keypoints=MAX_FEATURES).eval().to(device)
    _matcher = LightGlue(features="superpoint",
                         depth_confidence=-1,
                         width_confidence=-1).eval().to(device)
    log.info("Models loaded in %.0f ms", (time.monotonic() - t0) * 1000)


# ---------------------------------------------------------------------------
# Tile patch extraction
# ---------------------------------------------------------------------------
def _extract_tile_patch(tile_path: str, lat: float, lon: float) -> Optional[Any]:
    """Return a float32 numpy array (H×W) from the tile centred on (lat, lon)."""
    if not _RASTERIO_AVAILABLE:
        return None
    try:
        with rasterio.open(tile_path) as src:
            # Convert radius in metres to degrees (approximate)
            deg_per_m_lat = 1.0 / 111_320.0
            deg_per_m_lon = 1.0 / (111_320.0 * math.cos(math.radians(lat)))
            r_lat = TILE_PATCH_RADIUS_M * deg_per_m_lat
            r_lon = TILE_PATCH_RADIUS_M * deg_per_m_lon
            window = from_bounds(
                lon - r_lon, lat - r_lat, lon + r_lon, lat + r_lat,
                src.transform,
            )
            patch = src.read(1, window=window).astype("float32")
            if patch.size == 0:
                return None
            # Normalise to [0, 1]
            p_min, p_max = float(patch.min()), float(patch.max())
            if p_max > p_min:
                patch = (patch - p_min) / (p_max - p_min)
            return patch
    except Exception as exc:
        log.warning("Tile patch extraction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Core match logic
# ---------------------------------------------------------------------------
def _run_lightglue(
    frame_path: str,
    lat: float,
    lon: float,
    alt: float,
    heading_deg: float,
    tile_path: str,
) -> Tuple[float, float, float, float]:
    """
    Run LightGlue match. Returns (delta_lat, delta_lon, confidence, match_ms).
    Raises ValueError if match quality is below threshold.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.monotonic()

    # Load UAV frame
    uav_img = load_image(frame_path).to(device)  # (1, H, W) float32

    # Extract tile patch and convert to LightGlue tensor
    patch = _extract_tile_patch(tile_path, lat, lon)
    if patch is None:
        raise ValueError("tile patch extraction failed")

    # Convert patch to 3-channel tensor expected by SuperPoint
    patch_uint8 = (patch * 255).clip(0, 255).astype("uint8")
    patch_rgb = cv2.cvtColor(patch_uint8, cv2.COLOR_GRAY2RGB)
    tile_tensor = torch.from_numpy(
        patch_rgb.transpose(2, 0, 1).astype("float32") / 255.0
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        feats0 = _extractor.extract(uav_img)
        feats1 = _extractor.extract(tile_tensor)
        matches_data = _matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches_data = rbd(feats0), rbd(feats1), rbd(matches_data)

    kpts0 = feats0["keypoints"]
    kpts1 = feats1["keypoints"]
    matches = matches_data["matches"]

    n_matches = len(matches)
    if n_matches < MIN_INLIERS:
        raise ValueError(f"insufficient matches: {n_matches} < {MIN_INLIERS}")

    # Mean displacement in pixel space → geographic offset
    pts0 = kpts0[matches[:, 0]].cpu().numpy()
    pts1 = kpts1[matches[:, 1]].cpu().numpy()
    mean_dx = float((pts0[:, 0] - pts1[:, 0]).mean())  # pixels, x=lon axis
    mean_dy = float((pts0[:, 1] - pts1[:, 1]).mean())  # pixels, y=lat axis

    # Patch covers 2 × TILE_PATCH_RADIUS_M in each direction across patch.shape
    patch_h, patch_w = patch.shape
    m_per_px_lon = (2.0 * TILE_PATCH_RADIUS_M) / max(patch_w, 1)
    m_per_px_lat = (2.0 * TILE_PATCH_RADIUS_M) / max(patch_h, 1)
    delta_lon = mean_dx * m_per_px_lon / (111_320.0 * math.cos(math.radians(lat)))
    delta_lat = -mean_dy * m_per_px_lat / 111_320.0   # image y increases downward

    # Confidence: ratio of inlier score (capped at 1.0)
    scores = matches_data.get("scores", torch.ones(n_matches)).cpu().numpy()
    confidence = float(scores.mean())

    match_ms = (time.monotonic() - t0) * 1000.0
    return delta_lat, delta_lon, confidence, match_ms


def _run_stub(
    frame_path: str,
    lat: float,
    lon: float,
    alt: float,
    heading_deg: float,
) -> Tuple[float, float, float, float]:
    """Stub result when LightGlue is not installed.  Returns plausible values."""
    import random
    rng = random.Random(hash((round(lat, 4), round(lon, 4))))
    delta_lat = rng.gauss(0.0, 2e-5)   # ~2m noise
    delta_lon = rng.gauss(0.0, 2e-5)
    confidence = rng.uniform(0.72, 0.88)
    match_ms = rng.uniform(55.0, 85.0)  # dev machine RTX timing
    time.sleep(match_ms / 1000.0)
    return delta_lat, delta_lon, confidence, match_ms


# ---------------------------------------------------------------------------
# Request dispatcher
# ---------------------------------------------------------------------------
def _handle(request: Dict[str, Any]) -> Dict[str, Any]:
    cmd = request.get("cmd")

    if cmd == "ping":
        return {
            "status": "pong",
            "version": "1.0",
            "lightglue_available": _LIGHTGLUE_AVAILABLE,
            "python": sys.version,
        }

    if cmd == "match":
        frame_path = request.get("frame_path", "")
        lat = float(request["lat"])
        lon = float(request["lon"])
        alt = float(request["alt"])
        heading_deg = float(request.get("heading_deg", 0.0))

        # Validate coordinates: must be plausible WGS-84
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return {"status": "no_match", "reason": "invalid_coordinates"}

        # Resolve tile
        tile_path = resolve_tile(lat, lon)
        if tile_path is None:
            return {"status": "no_match", "reason": "no_tile_coverage"}

        # Validate frame file
        if not frame_path or not os.path.isfile(frame_path):
            return {"status": "no_match", "reason": "frame_not_found"}

        try:
            if _LIGHTGLUE_AVAILABLE:
                delta_lat, delta_lon, confidence, match_ms = _run_lightglue(
                    frame_path, lat, lon, alt, heading_deg, tile_path
                )
            else:
                delta_lat, delta_lon, confidence, match_ms = _run_stub(
                    frame_path, lat, lon, alt, heading_deg
                )
        except ValueError as exc:
            return {"status": "no_match", "reason": str(exc)}
        except Exception as exc:
            log.exception("Match error")
            return {"status": "error", "message": str(exc)}

        return {
            "status": "ok",
            "delta_lat": delta_lat,
            "delta_lon": delta_lon,
            "confidence": confidence,
            "match_ms": match_ms,
            "tile_path": tile_path,
            "stub_mode": not _LIGHTGLUE_AVAILABLE,
        }

    return {"status": "error", "message": f"unknown command: {cmd!r}"}


# ---------------------------------------------------------------------------
# Socket server
# ---------------------------------------------------------------------------
def _recv_line(conn: socket.socket, bufsize: int = 65536) -> Optional[bytes]:
    """Receive one newline-terminated message from the connection."""
    data = b""
    while True:
        chunk = conn.recv(bufsize)
        if not chunk:
            return None
        data += chunk
        if b"\n" in data:
            return data.split(b"\n", 1)[0]


def serve(socket_path: str = SOCKET_PATH) -> None:
    # Remove stale socket
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass

    _load_models()

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(socket_path)
    os.chmod(socket_path, 0o600)
    srv.listen(1)
    log.info("Listening on %s (stub_mode=%s)", socket_path, not _LIGHTGLUE_AVAILABLE)

    def _shutdown(sig, _frame):
        log.info("Signal %d — shutting down", sig)
        srv.close()
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while True:
        try:
            conn, _ = srv.accept()
        except OSError:
            break
        try:
            raw = _recv_line(conn)
            if raw is None:
                continue
            request = json.loads(raw.decode("utf-8"))
            response = _handle(request)
            conn.sendall(json.dumps(response).encode("utf-8") + b"\n")
        except Exception as exc:
            log.exception("Handler error: %s", exc)
            try:
                err = json.dumps({"status": "error", "message": str(exc)})
                conn.sendall(err.encode("utf-8") + b"\n")
            except Exception:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    serve()
