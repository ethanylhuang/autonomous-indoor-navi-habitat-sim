"""FastAPI app with WebSocket endpoint for the interactive viewer.

Serves the static dashboard and provides a WebSocket for real-time
agent control. Single-client, request/response pattern.

M3 additions: autonomous navigation mode with start_nav/stop_nav/tick
WebSocket message types.
"""

import base64
import json
import logging
import math
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
    render_topdown_with_path,
    render_vo_trajectory,
)

# M3 imports
from src.control.controller import NavigationController, NavigationStatus
from src.planning.global_planner import GlobalPlanner
from src.planning.local_planner import LocalPlanner
from src.state_estimation.estimator import EKFEstimator
from src.utils.transforms import normalize_angle, yaw_from_quaternion

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

# M3 navigation state
_nav_mode: str = "manual"  # "manual" or "autonomous"
_nav_goal: Optional[NDArray[np.float64]] = None
_nav_estimator: Optional[EKFEstimator] = None
_nav_global_planner: Optional[GlobalPlanner] = None
_nav_local_planner: Optional[LocalPlanner] = None
_nav_controller: Optional[NavigationController] = None
_nav_traveled_path: List[NDArray[np.float64]] = []
_nav_last_status: Optional[NavigationStatus] = None
_nav_final_spl: Optional[float] = None
_nav_obs = None  # latest observations during nav mode
_nav_pipeline_cache = None  # cached perception results from _nav_tick()


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
    global _nav_estimator, _nav_global_planner, _nav_local_planner, _nav_controller
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
    # M3 navigation modules (created once, reset per episode)
    _nav_estimator = EKFEstimator()
    _nav_global_planner = GlobalPlanner()
    _nav_local_planner = LocalPlanner()
    _nav_controller = NavigationController(
        global_planner=_nav_global_planner,
        local_planner=_nav_local_planner,
    )
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


def _run_perception_pipeline(obs):
    """Run M2 perception pipeline and return intermediate results.

    Returns:
        (vo_estimate, fwd_det, rear_det, grid_data, fwd_pc_world,
         rear_pc_world, merged_pc, rear_rot)
    """
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

    return (vo_estimate, fwd_det, rear_det, grid_data,
            fwd_pc_world, rear_pc_world, merged_pc, rear_rot)


def _build_frame(obs, nav_status_dict: Optional[dict] = None) -> dict:
    """Encode observations into a JSON-serializable frame dict."""
    fwd_jpg = encode_rgb_jpeg(obs.forward_rgb)
    rear_jpg = encode_rgb_jpeg(obs.rear_rgb)
    depth_jpg = colorize_depth(obs.depth)

    # Use path-overlay topdown when navigating (check nav_status_dict first for final frame)
    is_nav = (
        (nav_status_dict is not None and nav_status_dict.get("mode") == "autonomous")
        or (_nav_mode == "autonomous")
    )
    if is_nav and _nav_goal is not None and _nav_global_planner is not None:
        path = _nav_global_planner.get_path()
        waypoints = path.waypoints if path and path.is_valid else []
        current_idx = path.current_waypoint_idx if path and path.is_valid else 0
        topdown_png = render_topdown_with_path(
            _navmesh_grid,
            obs.state.position,
            obs.state.rotation,
            _navmesh_bounds,
            waypoints,
            current_idx,
            _nav_goal,
            _nav_traveled_path,
        )
    else:
        topdown_png = render_topdown_view(
            _navmesh_grid,
            obs.state.position,
            obs.state.rotation,
            _navmesh_bounds,
        )

    # Run perception pipeline for rendering (use cached results from _nav_tick if available)
    if nav_status_dict is not None and _nav_pipeline_cache is not None:
        (vo_estimate, fwd_det, rear_det, grid_data,
         fwd_pc_world, rear_pc_world, merged_pc, rear_rot) = _nav_pipeline_cache
    else:
        (vo_estimate, fwd_det, rear_det, grid_data,
         fwd_pc_world, rear_pc_world, merged_pc, rear_rot) = _run_perception_pipeline(obs)

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

    frame = {
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
        # NavMesh bounds for client-side pixel-to-world conversion
        "navmesh_bounds": {
            "lower": list(_navmesh_bounds[0]) if _navmesh_bounds else [0, 0, 0],
            "upper": list(_navmesh_bounds[1]) if _navmesh_bounds else [1, 1, 1],
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

    # M3 nav status
    if nav_status_dict is not None:
        frame["nav_status"] = nav_status_dict
    else:
        frame["nav_status"] = {
            "mode": _nav_mode,
            "goal": _nav_goal.tolist() if _nav_goal is not None else None,
            "goal_reached": False,
            "is_stuck": False,
            "distance_to_goal": None,
            "heading_error": None,
            "steps_taken": 0,
            "collisions": 0,
            "path_length": 0.0,
            "spl": _nav_final_spl,
            "waypoints": [],
            "current_waypoint_idx": 0,
            "action": None,
        }

    return frame


def _start_autonomous_nav(goal: Optional[NDArray[np.float64]] = None) -> dict:
    """Initialize autonomous navigation to a goal.

    Returns:
        Dict with status info or error.
    """
    global _nav_mode, _nav_goal, _nav_traveled_path, _nav_last_status
    global _nav_final_spl, _nav_obs

    # Pick random goal if none provided
    if goal is None:
        goal = np.array(
            _vehicle.pathfinder.get_random_navigable_point(), dtype=np.float64,
        )
    _nav_goal = goal

    # Reset M3 modules
    _nav_estimator.reset()
    _nav_global_planner.reset()
    _nav_local_planner.reset()
    _nav_controller.reset()
    _nav_traveled_path.clear()
    _nav_last_status = None
    _nav_final_spl = None

    # Get current observations
    _nav_obs = _vehicle.get_initial_observations()
    start_pos = _nav_obs.state.position.copy()
    start_rot = _nav_obs.state.rotation.copy()
    start_yaw = yaw_from_quaternion(start_rot)

    # Initialize EKF
    _nav_estimator.initialize(start_pos, start_yaw)

    # Plan global path
    global_path = _nav_global_planner.plan(_vehicle.find_path, start_pos, goal)
    if not global_path.is_valid:
        logger.warning("Nav path invalid. Returning to manual mode.")
        _nav_mode = "manual"
        return {"error": "path_invalid"}

    # Initialize controller
    _nav_controller.start_episode(
        start_pos, start_rot, goal, global_path,
        vehicle_find_path=_vehicle.find_path,
    )

    _nav_traveled_path.append(start_pos.copy())

    # Align vehicle heading toward first waypoint before starting nav loop.
    # Compute desired yaw, then issue turn actions to align.
    if global_path.waypoints and len(global_path.waypoints) > 1:
        first_wp = global_path.waypoints[1]  # skip start point
    else:
        first_wp = goal
    dx = first_wp[0] - start_pos[0]
    dz = first_wp[2] - start_pos[2]
    desired_yaw = math.atan2(-dx, -dz)  # habitat-sim convention
    yaw_diff = normalize_angle(desired_yaw - start_yaw)
    num_turns = int(round(abs(yaw_diff) / math.radians(10.0)))
    turn_action = "turn_left" if yaw_diff > 0 else "turn_right"
    for _ in range(num_turns):
        _nav_obs = _vehicle.step(turn_action)

    # Re-read state after alignment turns
    aligned_pos = _nav_obs.state.position.copy()
    aligned_rot = _nav_obs.state.rotation.copy()
    aligned_yaw = yaw_from_quaternion(aligned_rot)
    _nav_estimator.initialize(aligned_pos, aligned_yaw)

    _nav_mode = "autonomous"

    logger.info(
        "Autonomous nav started: goal=[%.2f, %.2f, %.2f], geodesic=%.2f, "
        "aligned %d turns",
        goal[0], goal[1], goal[2], global_path.geodesic_distance, num_turns,
    )
    return {"status": "started", "goal": goal.tolist()}


def _nav_tick() -> dict:
    """Execute one autonomous navigation step.

    Returns:
        Nav status dict for the frame.
    """
    global _nav_mode, _nav_obs, _nav_last_status, _nav_final_spl, _nav_pipeline_cache

    if _nav_mode != "autonomous" or _nav_controller is None:
        return {"mode": "manual"}

    obs = _nav_obs

    # Run perception pipeline once (cached for _build_frame to reuse)
    pipeline = _run_perception_pipeline(obs)
    _nav_pipeline_cache = pipeline
    (vo_est, fwd_det, rear_det, occ_grid,
     _fwd_pc, _rear_pc, _merged_pc, rear_rot) = pipeline

    # -- M3 State Estimation --
    pose = _nav_estimator.predict(obs.imu, dt=1.0)
    if vo_est.is_valid:
        pose = _nav_estimator.update_vo(vo_est)

    # -- M3 Planning + Control --
    rear_warning = (
        rear_det.obstacle_count > 0
        and float(np.min(obs.rear_depth[obs.rear_depth > 0])) < 0.5
        if np.any(obs.rear_depth > 0)
        else False
    )

    nav_status = _nav_controller.step(
        pose, occ_grid, rear_warning, obs.state.collided,
    )
    _nav_last_status = nav_status
    _nav_traveled_path.append(obs.state.position.copy())
    if len(_nav_traveled_path) > 2000:
        _nav_traveled_path[:] = _nav_traveled_path[-2000:]

    # Build waypoint list for viewer
    path = _nav_global_planner.get_path()
    waypoints_list = []
    current_wp_idx = 0
    if path and path.is_valid:
        waypoints_list = [w.tolist() for w in path.waypoints]
        current_wp_idx = path.current_waypoint_idx

    nav_dict = {
        "mode": "autonomous",
        "goal": _nav_goal.tolist() if _nav_goal is not None else None,
        "goal_reached": nav_status.goal_reached,
        "is_stuck": nav_status.is_stuck,
        "distance_to_goal": round(nav_status.distance_to_goal, 2),
        "heading_error": round(nav_status.heading_error, 3),
        "steps_taken": nav_status.steps_taken,
        "collisions": nav_status.total_collisions,
        "path_length": round(nav_status.path_length, 2),
        "spl": None,
        "waypoints": waypoints_list,
        "current_waypoint_idx": current_wp_idx,
        "action": nav_status.action,
    }

    # Check if episode ended
    if _nav_controller.is_episode_done:
        result = _nav_controller.finish_episode()
        nav_dict["spl"] = round(result.spl, 3)
        nav_dict["termination_reason"] = result.termination_reason
        _nav_final_spl = result.spl
        _nav_mode = "manual"
        logger.info(
            "Nav episode ended: %s, SPL=%.3f, steps=%d",
            result.termination_reason, result.spl, result.steps,
        )
    else:
        # Execute action and get new observations
        _nav_obs = _vehicle.step(nav_status.action)

    return nav_dict


def _stop_autonomous_nav() -> dict:
    """Stop autonomous navigation and return to manual mode."""
    global _nav_mode, _nav_final_spl, _nav_pipeline_cache

    if _nav_mode == "autonomous" and _nav_controller is not None:
        result = _nav_controller.finish_episode()
        _nav_final_spl = result.spl
        logger.info("Nav stopped by user. SPL=%.3f", result.spl)

    _nav_mode = "manual"
    _nav_pipeline_cache = None
    return {
        "mode": "manual",
        "spl": _nav_final_spl,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for real-time agent control."""
    global _nav_obs

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
            msg_type = msg.get("type", "")

            if msg_type == "reset":
                # Reset works in both modes
                global _nav_mode, _nav_pipeline_cache
                _nav_mode = "manual"
                _nav_pipeline_cache = None
                obs = _vehicle.reset()
                # Reset M2 perception state
                _vo.reset()
                _grid.reset()
                _vo_positions.clear()
                # Reset M3 state
                _nav_estimator.reset()
                _nav_global_planner.reset()
                _nav_local_planner.reset()
                _nav_controller.reset()
                _nav_traveled_path.clear()
                frame = _build_frame(obs)
                await ws.send_text(json.dumps(frame))

            elif msg_type == "start_nav":
                # Start autonomous navigation to random goal
                _start_autonomous_nav()
                obs = _nav_obs if _nav_obs is not None else _vehicle.get_initial_observations()
                frame = _build_frame(obs)
                await ws.send_text(json.dumps(frame))

            elif msg_type == "start_nav_to":
                # Start autonomous navigation to specific goal
                goal_coords = msg.get("goal")
                goal = np.array(goal_coords, dtype=np.float64) if goal_coords else None
                _start_autonomous_nav(goal)
                obs = _nav_obs if _nav_obs is not None else _vehicle.get_initial_observations()
                frame = _build_frame(obs)
                await ws.send_text(json.dumps(frame))

            elif msg_type == "stop_nav":
                stop_info = _stop_autonomous_nav()
                obs = _vehicle.get_initial_observations()
                frame = _build_frame(obs, nav_status_dict={
                    "mode": "manual",
                    "goal": None,
                    "goal_reached": False,
                    "is_stuck": False,
                    "distance_to_goal": None,
                    "heading_error": None,
                    "steps_taken": 0,
                    "collisions": 0,
                    "path_length": 0.0,
                    "spl": stop_info.get("spl"),
                    "waypoints": [],
                    "current_waypoint_idx": 0,
                    "action": None,
                })
                await ws.send_text(json.dumps(frame))

            elif msg_type == "tick":
                # Autonomous mode: execute one nav step
                if _nav_mode == "autonomous":
                    nav_dict = _nav_tick()
                    obs = _nav_obs if _nav_obs is not None else _vehicle.get_initial_observations()
                    frame = _build_frame(obs, nav_status_dict=nav_dict)
                    await ws.send_text(json.dumps(frame))
                else:
                    # In manual mode, tick just returns current state
                    obs = _vehicle.get_initial_observations()
                    frame = _build_frame(obs)
                    await ws.send_text(json.dumps(frame))

            elif "action" in msg:
                # Manual control: ignore during autonomous mode
                if _nav_mode == "autonomous":
                    logger.debug("Ignoring manual action during autonomous mode.")
                    continue
                action = msg["action"]
                obs = _vehicle.step(action)
                frame = _build_frame(obs)
                await ws.send_text(json.dumps(frame))

            else:
                logger.warning("Unknown message: %s", raw)
                continue

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception:
        logger.exception("WebSocket error")
