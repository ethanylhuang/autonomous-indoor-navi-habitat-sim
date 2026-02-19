"""FastAPI app with WebSocket endpoint for the interactive viewer.

Serves the static dashboard and provides a WebSocket for real-time
agent control. Single-client, request/response pattern.
"""

import base64
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from configs.sensor_rig import HFOV
from src.sensors.lidar import (
    PointCloud,
    depth_to_point_cloud,
    merge_point_clouds,
    transform_point_cloud,
)
from src.perception.obstacle_detector import ObstacleDetector
from src.perception.occupancy_grid import OccupancyGrid
from src.perception.visual_odometry import VisualOdometry
from src.vehicle import Vehicle
from viewer.renderer import (
    colorize_depth,
    encode_rgb_jpeg,
    render_occupancy_grid,
    render_point_cloud_bev,
    render_semantic_overlay,
    render_topdown_view,
    render_vo_trajectory,
)

logger = logging.getLogger(__name__)

# Module-level vehicle reference, managed by lifespan
_vehicle: Optional[Vehicle] = None
_navmesh_grid = None
_navmesh_bounds = None

# M2 perception modules
_vo: Optional[VisualOdometry] = None
_detector: Optional[ObstacleDetector] = None
_grid: Optional[OccupancyGrid] = None
_vo_positions: List[NDArray[np.float64]] = []


def _compute_rear_rotation(agent_rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute rear sensor world rotation: agent rotation * 180-degree Y offset."""
    from src.utils.transforms import quat_multiply
    q_180_y = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    return quat_multiply(agent_rotation, q_180_y)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create and destroy the Vehicle with the app lifecycle."""
    global _vehicle, _navmesh_grid, _navmesh_bounds
    global _vo, _detector, _grid, _vo_positions
    logger.info("Creating Vehicle instance...")
    _vehicle = Vehicle()
    # Pre-compute navmesh grid (expensive, do once)
    _navmesh_grid = _vehicle.get_topdown_navmesh()
    _navmesh_bounds = _vehicle.get_navmesh_bounds()
    # M2 perception modules
    _vo = VisualOdometry(hfov_deg=float(HFOV), resolution=(480, 640))
    _detector = ObstacleDetector()
    _grid = OccupancyGrid()
    _vo_positions = []
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

    # -- M2 perception pipeline -----------------------------------------------
    hfov = float(HFOV)

    # 1. LiDAR: depth -> point clouds -> world frame
    fwd_pc = depth_to_point_cloud(obs.depth, hfov)
    fwd_pc_world = transform_point_cloud(fwd_pc, obs.state.position, obs.state.rotation)
    rear_rot = _compute_rear_rotation(obs.state.rotation)
    rear_pc = depth_to_point_cloud(obs.rear_depth, hfov)
    rear_pc_world = transform_point_cloud(rear_pc, obs.state.position, rear_rot)
    merged_pc = merge_point_clouds([fwd_pc_world, rear_pc_world])

    # 2. VO: forward RGB -> relative pose
    vo_estimate = _vo.update(obs.forward_rgb)
    _vo_positions.append(obs.state.position.copy())
    if len(_vo_positions) > 1000:
        _vo_positions[:] = _vo_positions[-1000:]

    # 3. Semantic obstacles
    fwd_det, rear_det = _detector.detect_both_cameras(
        obs.forward_semantic, obs.rear_semantic,
        obs.depth, obs.rear_depth,
    )

    # 4. Occupancy grid: fuse LiDAR + semantic
    grid_data = _grid.update(
        obs.state.position, obs.state.rotation,
        [fwd_pc_world, rear_pc_world],
        obstacle_detections=[
            (fwd_det, obs.depth, obs.state.rotation, hfov),
            (rear_det, obs.rear_depth, rear_rot, hfov),
        ],
    )

    # -- M2 rendering ---------------------------------------------------------
    pc_bev_png = render_point_cloud_bev(merged_pc, obs.state.position)
    occ_grid_png = render_occupancy_grid(
        grid_data, obs.state.position, obs.state.rotation,
    )
    semantic_jpg = render_semantic_overlay(
        obs.forward_rgb, obs.forward_semantic, fwd_det.mask,
    )
    vo_traj_png = render_vo_trajectory(
        _vo_positions, obs.state.position, _navmesh_bounds,
    )

    # -- M2 stats -------------------------------------------------------------
    occupied_count = int(np.sum(grid_data.grid > 0.5))
    free_count = int(np.sum(grid_data.grid < 0.5))
    fwd_obstacle_pixels = int(np.sum(fwd_det.mask))
    rear_obstacle_pixels = int(np.sum(rear_det.mask))

    return {
        "forward_rgb": base64.b64encode(fwd_jpg).decode("ascii"),
        "rear_rgb": base64.b64encode(rear_jpg).decode("ascii"),
        "depth": base64.b64encode(depth_jpg).decode("ascii"),
        "topdown": base64.b64encode(topdown_png).decode("ascii"),
        # M2 images
        "point_cloud_bev": base64.b64encode(pc_bev_png).decode("ascii"),
        "occupancy_grid": base64.b64encode(occ_grid_png).decode("ascii"),
        "semantic_fwd": base64.b64encode(semantic_jpg).decode("ascii"),
        "vo_trajectory": base64.b64encode(vo_traj_png).decode("ascii"),
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
        # M2 stats
        "m2_stats": {
            "vo_inliers": vo_estimate.num_inliers,
            "vo_valid": vo_estimate.is_valid,
            "occupied_cells": occupied_count,
            "free_cells": free_count,
            "fwd_obstacle_pixels": fwd_obstacle_pixels,
            "rear_obstacle_pixels": rear_obstacle_pixels,
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
                # Reset M2 perception state
                _vo.reset()
                _grid.reset()
                _vo_positions.clear()
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
