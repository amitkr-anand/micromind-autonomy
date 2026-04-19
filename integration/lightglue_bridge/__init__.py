"""
LightGlue subprocess IPC bridge.

Client API (micromind-autonomy / Python 3.11):
    from integration.lightglue_bridge import client as lightglue_client
    result = lightglue_client.match(frame_path, lat, lon, alt, heading_deg)
    # result: (delta_lat, delta_lon, confidence, latency_ms) or None

Server (hil-h3 / Python 3.10):
    python3 integration/lightglue_bridge/server.py
"""
from .client import match, ping, shutdown

__all__ = ["match", "ping", "shutdown"]
