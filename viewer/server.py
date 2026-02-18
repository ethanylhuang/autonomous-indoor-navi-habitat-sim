"""FastAPI app with WebSocket endpoint for the interactive viewer.

Serves the static dashboard and provides a WebSocket for real-time
agent control. Single-client, request/response pattern.
"""

import base64
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.vehicle import Vehicle
from viewer.renderer import colorize_depth, encode_rgb_jpeg, render_topdown_view

logger = logging.getLogger(__name__)

# Module-level vehicle reference, managed by lifespan
_vehicle: Optional[Vehicle] = None
_navmesh_grid = None
_navmesh_bounds = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create and destroy the Vehicle with the app lifecycle."""
    global _vehicle, _navmesh_grid, _navmesh_bounds
    logger.info("Creating Vehicle instance...")
    _vehicle = Vehicle()
    # Pre-compute navmesh grid (expensive, do once)
    _navmesh_grid = _vehicle.get_topdown_navmesh()
    _navmesh_bounds = _vehicle.get_navmesh_bounds()
    logger.info("Vehicle ready.")
    yield
    logger.info("Shutting down Vehicle...")
    if _vehicle is not None:
        _vehicle.close()
        _vehicle = None
    logger.info("Vehicle closed.")


app = FastAPI(lifespan=lifespan)

# Serve static files
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
async def index():
    """Serve the dashboard HTML."""
    return FileResponse(str(_static_dir / "index.html"))


def _build_frame(obs) -> dict:
    """Encode observations into a JSON-serializable frame dict."""
    fwd_jpg = encode_rgb_jpeg(obs.forward_rgb)
    rear_jpg = encode_rgb_jpeg(obs.rear_rgb)
    depth_jpg = colorize_depth(obs.depth)
    topdown_png = render_topdown_view(
        _navmesh_grid,
        obs.state.position,
        obs.state.rotation,
        _navmesh_bounds,
    )

    return {
        "forward_rgb": base64.b64encode(fwd_jpg).decode("ascii"),
        "rear_rgb": base64.b64encode(rear_jpg).decode("ascii"),
        "depth": base64.b64encode(depth_jpg).decode("ascii"),
        "topdown": base64.b64encode(topdown_png).decode("ascii"),
        "state": {
            "position": obs.state.position.tolist(),
            "rotation": obs.state.rotation.tolist(),
            "step_count": obs.state.step_count,
            "collided": obs.state.collided,
        },
        "imu": {
            "linear_acceleration": obs.imu.linear_acceleration.tolist(),
            "angular_velocity": obs.imu.angular_velocity.tolist(),
            "timestamp_step": obs.imu.timestamp_step,
        },
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for real-time agent control."""
    await ws.accept()
    logger.info("WebSocket client connected.")

    try:
        # Send initial frame
        obs = _vehicle.get_initial_observations()
        frame = _build_frame(obs)
        await ws.send_text(json.dumps(frame))

        # Request/response loop
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "reset":
                obs = _vehicle.reset()
                # Recompute navmesh grid after reset (position changed)
                # Grid itself doesn't change, just agent overlay
            elif "action" in msg:
                action = msg["action"]
                obs = _vehicle.step(action)
            else:
                logger.warning("Unknown message: %s", raw)
                continue

            frame = _build_frame(obs)
            await ws.send_text(json.dumps(frame))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception:
        logger.exception("WebSocket error")
