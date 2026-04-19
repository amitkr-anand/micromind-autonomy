# L2 LightGlue IPC Interface Contract

**Version:** 1.0  
**Date:** 19 April 2026  
**Status:** Baseline — HIL H-4 PASS  
**Owner:** NanoCorteX Technologies / MicroMind Programme  

---

## 1. Purpose

Defines the inter-process communication contract between the
**micromind-autonomy** production stack (Python 3.11) and the
**LightGlue image matcher** (Python 3.10, hil-h3 conda environment).

This bridge implements the L2 Absolute Reset layer of the three-layer
navigation architecture (AD-01, revised 03 April 2026).  L2 fires once
every 2 km over textured terrain to hard-reset accumulated VIO drift.

---

## 2. Architecture Context

```
micromind-autonomy (Python 3.11)          hil-h3 (Python 3.10)
┌──────────────────────────────────┐      ┌────────────────────────────────┐
│  NavigationManager               │      │  LightGlue server              │
│  lightglue_client.match(...)     │◄────►│  SuperPoint + LightGlue GPU    │
│  (client.py)                     │      │  (server.py)                   │
└──────────────────────────────────┘      └────────────────────────────────┘
              Unix domain socket: /tmp/micromind_lightglue.sock
```

Both processes run on the **same host** (Jetson Orin Nano Super in HIL,
or dev machine for SIL testing).  The server is auto-started by the client
on first use.

---

## 3. IPC Mechanism

**Transport:** Unix domain socket (AF_UNIX, SOCK_STREAM)  
**Socket path:** `/tmp/micromind_lightglue.sock` (overridable via `LIGHTGLUE_SOCKET` env var)  
**Framing:** Newline-delimited JSON (one JSON object per message, terminated by `\n`)  
**Concurrency:** Serial — one request in flight at a time (L2 fires every 2 km)  
**Direction:** Client sends request; server sends exactly one response  

### Rationale for Unix socket over alternatives

| Option | Verdict | Reason |
|--------|---------|--------|
| Unix socket | **Chosen** | Zero-dependency (stdlib `socket`), bidirectional, low latency, works identically on Python 3.10 and 3.11 |
| Named pipe | Rejected | Requires two FIFOs for bidirectional comms; more complex framing |
| ZeroMQ | Rejected | External dependency (`pyzmq`) required in both envs; unnecessary for serial single-machine use |
| HTTP/REST | Prohibited | Explicitly excluded by programme constraints |
| Redis | Prohibited | Explicitly excluded by programme constraints |

---

## 4. Client API

**Module:** `integration.lightglue_bridge.client`

### 4.1 Primary call

```python
from integration.lightglue_bridge import client as lightglue_client

result = lightglue_client.match(
    uav_frame_path: str,   # Absolute path to UAV nadir frame (PNG/JPEG)
    lat: float,            # GPS prior latitude (WGS-84 decimal degrees)
    lon: float,            # GPS prior longitude (WGS-84 decimal degrees)
    alt: float,            # Altitude above sea level (metres)
    heading_deg: float,    # UAV heading (degrees, 0=north, 90=east)
)
```

**Return type:**
```python
Optional[Tuple[float, float, float, float]]
# On success: (delta_lat, delta_lon, confidence, latency_ms)
# On failure:  None
```

| Field | Type | Units | Sign convention |
|-------|------|-------|-----------------|
| `delta_lat` | float | decimal degrees | positive = north |
| `delta_lon` | float | decimal degrees | positive = east |
| `confidence` | float | 0.0 – 1.0 | higher = better match quality |
| `latency_ms` | float | milliseconds | total round-trip including IPC overhead |

**Usage — apply correction to NavigationManager:**
```python
result = lightglue_client.match(frame_path, lat, lon, alt, heading)
if result is not None:
    delta_lat, delta_lon, conf, latency_ms = result
    corrected_lat = lat + delta_lat
    corrected_lon = lon + delta_lon
    navigation_manager.update_trn(corrected_lat, corrected_lon, conf)
```

### 4.2 Auxiliary calls

```python
info = lightglue_client.ping()   # dict: version, python, lightglue_available
lightglue_client.shutdown()      # terminate server subprocess if client owns it
```

---

## 5. Wire Protocol

### 5.1 Ping

**Request:**
```json
{"cmd": "ping"}
```

**Response:**
```json
{"status": "pong", "version": "1.0", "lightglue_available": true, "python": "3.10.x ..."}
```

### 5.2 Match

**Request:**
```json
{
  "cmd": "match",
  "frame_path": "/abs/path/to/frame.png",
  "lat": 31.157,
  "lon": 77.256,
  "alt": 300.0,
  "heading_deg": 45.0
}
```

**Response — success:**
```json
{
  "status": "ok",
  "delta_lat": 0.000023,
  "delta_lon": -0.000041,
  "confidence": 0.847,
  "match_ms": 628.3,
  "tile_path": "/abs/path/to/shimla_tile.tif",
  "stub_mode": false
}
```

**Response — no match:**
```json
{"status": "no_match", "reason": "insufficient_matches"}
```

Possible `reason` values: `invalid_coordinates`, `no_tile_coverage`,
`frame_not_found`, `insufficient_matches`, `tile_patch_extraction_failed`.

**Response — server error:**
```json
{"status": "error", "message": "description of error"}
```

---

## 6. Error Handling

| Condition | Client behaviour | Return |
|-----------|-----------------|--------|
| Server not running | Auto-start (up to 15 s) then retry | `None` if start fails |
| Request timeout (> 10 s) | Log warning | `None` |
| `status == "no_match"` | Log at DEBUG | `None` |
| `status == "error"` | Log at WARNING | `None` |
| Invalid coordinates | Server rejects | `None` (reason: `invalid_coordinates`) |
| No tile coverage | Server rejects | `None` (reason: `no_tile_coverage`) |
| Frame file missing | Server rejects | `None` (reason: `frame_not_found`) |

The client **never raises** — it always returns `Optional[MatchResult]`.
The NavigationManager treats `None` as "no correction available this interval"
and continues on VIO-only until the next L2 trigger.

---

## 7. Server Lifecycle

- **Start:** spawned by client as subprocess on first `match()` call
- **Python interpreter:** configured in `config.LIGHTGLUE_PYTHON` (default: hil-h3 env)
- **Fallback:** if hil-h3 interpreter not found, falls back to current Python with stub mode
- **Stub mode:** when LightGlue not importable, server returns plausible synthetic results for SIL testing
- **Stop:** `SIGTERM` or `SIGINT`; client calls `shutdown()` on programme exit
- **Socket cleanup:** server removes socket on exit; client removes stale socket on start

---

## 8. Tile Coverage

The server resolves satellite tiles via `tile_resolver.resolve(lat, lon)`.

| Region | Lat range | Lon range | Tile file |
|--------|-----------|-----------|-----------|
| shimla_local | 30.9–31.6°N | 76.9–77.7°E | `shimla_corridor/SHIMLA-1_COP30.tif` |
| shimla_manali | 31.0–32.5°N | 76.9–77.7°E | `shimla_manali_corridor/shimla_tile.tif` |
| jammu_leh_tile1 | 32.5–33.6°N | 74.5–76.0°E | `Jammu_leh_corridor_COP30/TILE1/…` |
| jammu_leh_tile2 | 33.5–34.6°N | 74.5–76.5°E | `Jammu_leh_corridor_COP30/TILE2/…` |
| jammu_leh_tile3 | 34.0–35.0°N | 75.5–78.0°E | `Jammu_leh_corridor_COP30/TILE3/…` |

---

## 9. Performance Envelope

| Environment | Median | P99 | Budget (2 km @ 27 m/s) | Margin |
|-------------|--------|-----|------------------------|--------|
| Jetson Orin Nano Super (hil-h3, GPU) | 628 ms | 1,630 ms | 74,000 ms | 45× |
| Dev machine RTX 5060 Ti (stub) | ~70 ms | ~85 ms | 74,000 ms | >800× |
| IPC overhead (Unix socket, JSON) | < 2 ms | < 5 ms | — | — |

---

## 10. Configuration Reference

All parameters are in `integration/lightglue_bridge/config.py` and
overridable via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTGLUE_SOCKET` | `/tmp/micromind_lightglue.sock` | Unix socket path |
| `LIGHTGLUE_TILE_DIR` | `<repo>/data/terrain` | Root of terrain tile data |
| `LIGHTGLUE_PYTHON` | `/home/mmuser/miniforge3/envs/hil-h3/bin/python3` | hil-h3 interpreter |
| `LIGHTGLUE_START_TIMEOUT` | `15.0` | Server readiness timeout (s) |
| `LIGHTGLUE_REQUEST_TIMEOUT` | `10.0` | Per-request timeout (s) |
| `LIGHTGLUE_MAX_FEATURES` | `2048` | SuperPoint keypoint limit |
| `LIGHTGLUE_MATCH_THRESHOLD` | `0.2` | LightGlue match threshold |
| `LIGHTGLUE_MIN_INLIERS` | `20` | Minimum matches to accept result |
| `LIGHTGLUE_PATCH_RADIUS` | `500.0` | Tile patch radius (m) |
| `LIGHTGLUE_TILE_GSD` | `30.0` | Tile GSD (m/px, COP30 nominal) |

---

*End of L2_LIGHTGLUE_IPC.md*
