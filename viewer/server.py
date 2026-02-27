"""FastAPI app with WebSocket endpoint for the interactive viewer.

Serves the static dashboard and provides a WebSocket for real-time
agent control. Single-client, request/response pattern.

M3 additions: autonomous navigation mode with start_nav/stop_nav/tick
WebSocket message types.

M5 additions: VLM-guided semantic navigation.
"""

import base64
import json
import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import numpy as np
from numpy.typing import NDArray
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from configs.sensor_rig import HFOV
from configs.sim_config import SimParams
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
    render_rgb_with_vlm_target,
    render_semantic_overlay,
    render_topdown_view,
    render_topdown_with_path,
    render_topdown_with_vlm_target,
    render_vo_trajectory,
)

# M3 imports
from src.control.controller import NavigationController, NavigationStatus
from src.planning.global_planner import GlobalPlanner
from src.planning.local_planner import LocalPlanner
from src.state_estimation.estimator import EKFEstimator
from src.utils.transforms import normalize_angle, yaw_from_quaternion

# M5 VLM imports
from src.vlm.client import VLMClient
from src.vlm.navigator import VLMNavigator, VLMNavStatus
from src.vlm.projection import pixel_to_world, snap_to_navmesh

# Semantic scene imports
from src.perception.semantic_scene import (
    SemanticSceneIndex,
    build_semantic_index_from_sim,
    get_navigable_objects,
)

# M5 Semantic constrained navigation
from src.vlm.semantic_navigator import SemanticConstrainedNavigator
from src.vlm.constrained import ObjectCandidateBuilder

# Configure logging to show INFO level for VLM debugging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

# Scene datasets base path
_SCENE_DATASETS_PATH = Path(__file__).parent.parent / "data" / "scene_datasets"
_HM3D_HABITAT_PATH = Path(__file__).parent.parent / "data" / "hm3d-minival-habitat-v0.2"
_HM3D_SEMANTIC_PATH = Path(__file__).parent.parent / "data" / "hm3d-minival-semantic-annots-v0.2"
_PROJECT_ROOT = Path(__file__).parent.parent

# Current scene ID (for reporting to clients)
_current_scene_id: str = SimParams.scene_id


def discover_scenes() -> list[dict]:
    """Scan for available scenes with matching navmesh files.

    Scans both:
    - data/scene_datasets/ for standard .glb + .navmesh pairs
    - data/hm3d-minival-habitat-v0.2/ for HM3D .basis.glb + .basis.navmesh pairs

    Returns:
        List of {"id": "path/to/scene.glb", "name": "scene_name",
                 "has_semantic": bool} dicts.
    """
    scenes = []

    # 1. Standard test scenes (data/scene_datasets/)
    if _SCENE_DATASETS_PATH.exists():
        for glb_path in _SCENE_DATASETS_PATH.rglob("*.glb"):
            navmesh_path = glb_path.with_suffix(".navmesh")
            if navmesh_path.exists():
                try:
                    rel_path = glb_path.relative_to(_PROJECT_ROOT)
                except ValueError:
                    rel_path = glb_path
                scenes.append({
                    "id": str(rel_path),
                    "name": glb_path.stem,
                    "has_semantic": False,
                })
    else:
        logger.warning("Scene datasets path does not exist: %s", _SCENE_DATASETS_PATH)

    # 2. HM3D minival scenes (data/hm3d-minival-habitat-v0.2/)
    if _HM3D_HABITAT_PATH.exists():
        for glb_path in _HM3D_HABITAT_PATH.rglob("*.basis.glb"):
            # Navmesh has same name pattern: *.basis.navmesh
            navmesh_path = glb_path.with_name(
                glb_path.name.replace(".basis.glb", ".basis.navmesh")
            )
            if navmesh_path.exists():
                try:
                    rel_path = glb_path.relative_to(_PROJECT_ROOT)
                except ValueError:
                    rel_path = glb_path

                # Check for corresponding semantic annotations AND scene_dataset_config
                # Both are required for semantic data to actually work
                scene_dir = glb_path.parent.name  # e.g., "00800-TEEsavR23oF"
                scene_id = glb_path.stem.replace(".basis", "")  # e.g., "TEEsavR23oF"
                semantic_glb = _HM3D_SEMANTIC_PATH / scene_dir / f"{scene_id}.semantic.glb"
                scene_config = glb_path.parent / f"{scene_id}.scene_dataset_config.json"
                # Need both semantic GLB and config file for semantic data to work
                has_semantic = semantic_glb.exists() and scene_config.exists()

                # Build display name: "HM3D: 00800-TEEsavR23oF" or with semantic indicator
                display_name = f"HM3D: {scene_dir}"
                if has_semantic:
                    display_name += " [semantic]"

                scenes.append({
                    "id": str(rel_path),
                    "name": display_name,
                    "has_semantic": has_semantic,
                })
    else:
        logger.info("HM3D habitat path not found: %s", _HM3D_HABITAT_PATH)

    # Sort by name for consistent ordering
    scenes.sort(key=lambda s: s["name"])
    return scenes


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

# M5 VLM navigation state
_vlm_client: Optional[VLMClient] = None
_vlm_navigator: Optional[VLMNavigator] = None
_vlm_mode: str = "manual"  # "manual" or "vlm_nav"
_vlm_instruction: Optional[str] = None

# Semantic scene index (built on scene load for HM3D scenes)
_semantic_index: Optional[SemanticSceneIndex] = None

# M5 Semantic navigation state (constrained VLM selection)
_semantic_navigator = None  # SemanticConstrainedNavigator instance


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
    # M5 VLM navigation (optional - only if API key is available)
    global _vlm_client, _vlm_navigator
    try:
        _vlm_client = VLMClient()
        _vlm_navigator = VLMNavigator(
            _vlm_client,
            _nav_global_planner,
            _nav_local_planner,
            _vehicle.pathfinder,
        )
        logger.info("VLM client initialized successfully.")
    except Exception as e:
        logger.warning("VLM client not initialized (API key missing?): %s", e)
        _vlm_client = None
        _vlm_navigator = None
    # Build semantic index for initial scene (if applicable)
    global _semantic_index, _semantic_navigator
    _semantic_index = build_semantic_index_from_sim(_vehicle)
    if _semantic_index is not None:
        logger.info("Semantic index built: %d objects", len(_semantic_index.objects))
    # M5 Semantic constrained navigator (optional - requires VLM + semantic index)
    if _vlm_client is not None and _semantic_index is not None:
        try:
            candidate_builder = ObjectCandidateBuilder(max_candidates=100)
            _semantic_navigator = SemanticConstrainedNavigator(
                vlm_client=_vlm_client,
                semantic_index=_semantic_index,
                global_planner=_nav_global_planner,
                controller=_nav_controller,
                candidate_builder=candidate_builder,
                pathfinder=_vehicle.pathfinder,
            )
            logger.info("Semantic constrained navigator initialized.")
        except Exception as e:
            logger.warning("Semantic navigator not initialized: %s", e)
            _semantic_navigator = None
    logger.info("Vehicle ready.")
    yield
    logger.info("Shutting down Vehicle...")
    if _vehicle is not None:
        _vehicle.close()
        _vehicle = None
    logger.info("Vehicle closed.")


app = FastAPI(lifespan=lifespan)

# Static files directory
_static_dir = Path(__file__).parent / "static"


@app.get("/")
async def index():
    """Serve the dashboard HTML."""
    return FileResponse(
        str(_static_dir / "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/static/app.js")
async def app_js():
    """Serve app.js with no-cache headers."""
    return FileResponse(
        str(_static_dir / "app.js"),
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# Mount static files AFTER explicit routes so app.js route takes precedence
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/api/scenes")
async def list_scenes():
    """Return available scenes and current scene."""
    scenes = discover_scenes()
    return {
        "scenes": scenes,
        "current": _current_scene_id,
    }


@app.get("/api/semantic_objects")
async def list_semantic_objects():
    """Return navigable semantic objects reachable from current agent position."""
    if _semantic_index is None or _vehicle is None:
        return {"objects": [], "has_semantic": False, "count": 0}

    navigable = get_navigable_objects(_semantic_index)

    # Filter to only objects reachable from current agent position
    # (same navmesh island - can find valid path)
    agent_pos = _vehicle._agent.get_state().position
    reachable = []
    for obj in navigable:
        if obj.navmesh_position is None:
            continue
        waypoints, dist = _vehicle.find_path(agent_pos, obj.navmesh_position)
        if len(waypoints) > 0 and dist < float("inf"):
            reachable.append(obj)

    result = {
        "objects": [obj.to_dict() for obj in reachable],
        "has_semantic": True,
        "count": len(reachable),
        "total_navigable": len(navigable),  # For info: total vs reachable
    }
    # Debug: log first few objects
    logger.info("list_semantic_objects: %d reachable objects", len(reachable))
    for obj in reachable[:5]:
        logger.info("  id=%d label=%s pos=%s", obj.object_id, obj.label, obj.navmesh_position)
    return result


def _reinitialize_simulator(scene_id: str) -> bool:
    """Reinitialize the simulator with a new scene.

    Args:
        scene_id: Path to the scene .glb file (relative to project root).

    Returns:
        True if successful, False otherwise.
    """
    global _vehicle, _navmesh_grid, _navmesh_bounds, _current_scene_id
    global _vo, _detector, _grid, _vo_positions
    global _nav_mode, _nav_goal, _nav_estimator, _nav_global_planner
    global _nav_local_planner, _nav_controller, _nav_traveled_path
    global _nav_last_status, _nav_final_spl, _nav_obs, _nav_pipeline_cache
    global _vlm_mode, _vlm_instruction
    global _semantic_index

    # Validate scene exists
    scene_path = Path(__file__).parent.parent / scene_id
    if not scene_path.exists():
        logger.error("Scene file not found: %s", scene_path)
        return False

    # Handle navmesh path for both standard and HM3D scenes
    # HM3D uses *.basis.glb -> *.basis.navmesh
    # Standard uses *.glb -> *.navmesh
    if scene_path.name.endswith(".basis.glb"):
        navmesh_path = scene_path.with_name(
            scene_path.name.replace(".basis.glb", ".basis.navmesh")
        )
    else:
        navmesh_path = scene_path.with_suffix(".navmesh")

    if not navmesh_path.exists():
        logger.error("NavMesh not found for scene: %s", navmesh_path)
        return False

    logger.info("Reinitializing simulator with scene: %s", scene_id)

    # Close existing vehicle
    if _vehicle is not None:
        _vehicle.close()
        _vehicle = None

    # Check if this is an HM3D scene with semantic data available
    # If so, use scene_dataset_config to enable semantic scene access
    scene_dataset_config = None
    navmesh_override = None
    if scene_path.name.endswith(".basis.glb"):
        # Extract scene directory info for HM3D scenes
        # e.g., data/hm3d-minival-habitat-v0.2/00802-wcojb4TFT35/wcojb4TFT35.basis.glb
        scene_dir = scene_path.parent.name  # e.g., "00802-wcojb4TFT35"
        scene_name = scene_path.stem.replace(".basis", "")  # e.g., "wcojb4TFT35"

        # Check if semantic annotations exist
        semantic_glb = _HM3D_SEMANTIC_PATH / scene_dir / f"{scene_name}.semantic.glb"
        if semantic_glb.exists():
            # Use scene_dataset_config for semantic data access
            config_path = scene_path.parent / f"{scene_name}.scene_dataset_config.json"
            if config_path.exists():
                scene_dataset_config = str(config_path.relative_to(_PROJECT_ROOT))
                # DON'T set navmesh_override - the scene_dataset_config.json
                # includes navmesh_asset which auto-loads with correct transforms
                navmesh_override = None
                scene_id = scene_name  # Use scene name, not full path
                logger.info("Using scene_dataset_config for semantic access: %s", scene_dataset_config)

    # Create new vehicle with new scene
    sim_params = SimParams(scene_id=scene_id, scene_dataset_config=scene_dataset_config)
    _vehicle = Vehicle(sim_params=sim_params, navmesh_path=navmesh_override)
    _current_scene_id = scene_id

    # Recompute navmesh grid
    _navmesh_grid = _vehicle.get_topdown_navmesh()
    _navmesh_bounds = _vehicle.get_navmesh_bounds()

    # Reset M2 perception modules
    _vo.reset()
    _grid.reset()
    _vo_positions.clear()

    # Reset M3 navigation state
    _nav_mode = "manual"
    _nav_goal = None
    _nav_estimator.reset()
    _nav_global_planner.reset()
    _nav_local_planner.reset()
    _nav_controller.reset()
    _nav_traveled_path.clear()
    _nav_last_status = None
    _nav_final_spl = None
    _nav_obs = None
    _nav_pipeline_cache = None

    # Reset M5 VLM state
    _vlm_mode = "manual"
    _vlm_instruction = None
    if _vlm_navigator is not None:
        _vlm_navigator.reset()

    # Build semantic index for HM3D scenes with semantic data
    _semantic_index = build_semantic_index_from_sim(_vehicle)
    if _semantic_index is not None:
        navigable_count = len(get_navigable_objects(_semantic_index))
        logger.info(
            "Semantic index built: %d objects, %d navigable",
            len(_semantic_index.objects),
            navigable_count,
        )
    else:
        logger.info("No semantic data available for this scene.")

    # Reinitialize semantic navigator if VLM and semantic index are available
    global _semantic_navigator
    if _vlm_client is not None and _semantic_index is not None:
        try:
            candidate_builder = ObjectCandidateBuilder(max_candidates=100)
            _semantic_navigator = SemanticConstrainedNavigator(
                vlm_client=_vlm_client,
                semantic_index=_semantic_index,
                global_planner=_nav_global_planner,
                controller=_nav_controller,
                candidate_builder=candidate_builder,
                pathfinder=_vehicle.pathfinder,
            )
            logger.info("Semantic constrained navigator reinitialized for new scene.")
        except Exception as e:
            logger.warning("Could not reinitialize semantic navigator: %s", e)
            _semantic_navigator = None
    else:
        _semantic_navigator = None

    logger.info("Simulator reinitialized successfully.")
    return True


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
    rear_jpg = encode_rgb_jpeg(obs.rear_rgb)
    depth_jpg = colorize_depth(obs.depth)

    # Use path-overlay topdown when navigating (check nav_status_dict first for final frame)
    is_nav = (
        (nav_status_dict is not None and nav_status_dict.get("mode") == "autonomous")
        or (_nav_mode == "autonomous")
    )
    is_vlm_nav = _vlm_mode == "vlm_nav"

    # Get VLM pixel target for overlay on forward RGB
    vlm_pixel = None
    vlm_target_valid = False
    if is_vlm_nav and _vlm_navigator is not None:
        if _vlm_navigator._current_pixel is not None and _vlm_navigator._current_pixel.is_valid:
            vlm_pixel = (_vlm_navigator._current_pixel.u, _vlm_navigator._current_pixel.v)
        if _vlm_navigator._current_world_target is not None:
            vlm_target_valid = _vlm_navigator._current_world_target.is_valid

    # Render forward RGB with VLM target overlay if in VLM nav mode
    if is_vlm_nav and vlm_pixel is not None:
        fwd_jpg = render_rgb_with_vlm_target(obs.forward_rgb, vlm_pixel, vlm_target_valid)
    else:
        fwd_jpg = encode_rgb_jpeg(obs.forward_rgb)

    if is_vlm_nav and _vlm_navigator is not None and _nav_global_planner is not None:
        # VLM nav mode: show VLM target point on navmesh
        path = _nav_global_planner.get_path()
        waypoints = path.waypoints if path and path.is_valid else []
        current_idx = path.current_waypoint_idx if path and path.is_valid else 0
        # Get VLM target from navigator's world target
        vlm_target = None
        if (
            _vlm_navigator._current_world_target is not None
            and _vlm_navigator._current_world_target.is_valid
        ):
            vlm_target = _vlm_navigator._current_world_target.navmesh_position
        topdown_png = render_topdown_with_vlm_target(
            _navmesh_grid,
            obs.state.position,
            obs.state.rotation,
            _navmesh_bounds,
            vlm_target,
            waypoints,
            current_idx,
        )
    elif is_nav and _nav_goal is not None and _nav_global_planner is not None:
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
        action = nav_status.action
        # Parse continuous action: "turn_to:{yaw}" or "turn_to:{yaw}:move"
        parts = action.split(":")
        target_yaw = float(parts[1])
        move_forward = len(parts) > 2 and parts[2] == "move"
        _nav_obs = _vehicle.step_with_heading(target_yaw, move_forward)

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


# ---------------------------------------------------------------------------
# M5 VLM Navigation Functions
# ---------------------------------------------------------------------------


def _start_vlm_nav(instruction: str) -> dict:
    """Start VLM-guided navigation to a semantic goal.

    Args:
        instruction: Natural language goal (e.g., "go to the bedroom").

    Returns:
        Dict with status or error.
    """
    global _vlm_mode, _vlm_instruction, _nav_obs

    if _vlm_navigator is None:
        return {"error": "VLM not configured (missing API key?)"}

    # Stop any existing navigation
    if _nav_mode == "autonomous":
        _stop_autonomous_nav()

    # Reset VLM navigator and start episode
    _vlm_navigator.reset()
    _vlm_navigator.start_episode(instruction)

    # Get initial observations
    _nav_obs = _vehicle.get_initial_observations()

    _vlm_mode = "vlm_nav"
    _vlm_instruction = instruction

    logger.info("VLM nav started: '%s'", instruction)
    return {"status": "started", "instruction": instruction}


def _vlm_tick() -> dict:
    """Execute one VLM navigation step.

    Returns:
        VLM status dict for the frame.
    """
    global _vlm_mode, _nav_obs

    if _vlm_mode != "vlm_nav" or _vlm_navigator is None:
        return {"mode": "manual"}

    obs = _nav_obs

    # Run perception pipeline for occupancy grid
    pipeline = _run_perception_pipeline(obs)
    (vo_est, fwd_det, rear_det, occ_grid,
     _fwd_pc, _rear_pc, _merged_pc, rear_rot) = pipeline

    # Check for rear obstacles
    rear_warning = (
        rear_det.obstacle_count > 0
        and float(np.min(obs.rear_depth[obs.rear_depth > 0])) < 0.5
        if np.any(obs.rear_depth > 0)
        else False
    )

    # Execute one VLM step (pass occupancy grid + planners)
    action, vlm_status = _vlm_navigator.step(
        obs,
        occ_grid,
        rear_warning,
        _vehicle.find_path,
    )

    # Execute the action
    if not _vlm_navigator.is_done:
        _nav_obs = _vehicle.step(action)

    # Build status dict
    status_dict = {
        "mode": vlm_status.mode,
        "instruction": vlm_status.instruction,
        "goal_reached": vlm_status.goal_reached,
        "steps_taken": vlm_status.steps_taken,
        "vlm_calls": vlm_status.vlm_calls,
        "last_vlm_reasoning": vlm_status.last_vlm_reasoning,
        "confidence": vlm_status.confidence,
        "target_visible": vlm_status.target_visible,
        "termination_reason": vlm_status.termination_reason,
        "current_pixel": list(vlm_status.current_pixel) if vlm_status.current_pixel else None,
        "current_world_target": list(vlm_status.current_world_target) if vlm_status.current_world_target else None,
        "target_valid": vlm_status.target_valid,
        "target_failure_reason": vlm_status.target_failure_reason,
        "depth_at_target": vlm_status.depth_at_target,
        "action": action,
    }

    # Check if episode ended
    if _vlm_navigator.is_done:
        _vlm_mode = "manual"
        logger.info(
            "VLM nav ended: %s, steps=%d, vlm_calls=%d",
            vlm_status.termination_reason,
            vlm_status.steps_taken,
            vlm_status.vlm_calls,
        )

    return status_dict


def _stop_vlm_nav() -> dict:
    """Stop VLM navigation and return to manual mode."""
    global _vlm_mode, _vlm_instruction

    if _vlm_mode == "vlm_nav" and _vlm_navigator is not None:
        _vlm_navigator.reset()
        logger.info("VLM nav stopped by user.")

    _vlm_mode = "manual"
    _vlm_instruction = None
    return {"mode": "manual"}


# ---------------------------------------------------------------------------
# M5 Semantic Constrained Navigation Functions
# ---------------------------------------------------------------------------


def _start_semantic_nav(instruction: str) -> dict:
    """Start semantic navigation with constrained VLM object selection.

    Args:
        instruction: Natural language instruction (e.g., "find something to sit on").

    Returns:
        Dict with status or error.
    """
    global _nav_mode, _vlm_mode, _nav_obs

    if _semantic_navigator is None:
        return {"error": "Semantic navigator not available (missing VLM or semantic index)"}

    # Stop any existing navigation
    if _nav_mode == "autonomous":
        _stop_autonomous_nav()
    if _vlm_mode == "vlm_nav":
        _stop_vlm_nav()

    # Get initial observations
    obs = _vehicle.get_initial_observations()

    # Initialize EKF state estimator
    start_pos = obs.state.position.copy()
    start_yaw = yaw_from_quaternion(obs.state.rotation)
    _nav_estimator.initialize(start_pos, start_yaw)

    # Start episode (VLM selection + path planning happens here)
    status = _semantic_navigator.start_episode(
        instruction,
        obs.state.position,
        obs.state.rotation,
    )

    # Check if selection succeeded
    if status.phase == "completed":
        # Selection failed
        reason = status.termination_reason or "unknown"
        logger.warning("Semantic nav selection failed: %s", reason)
        return {"error": f"Object selection failed: {reason}"}

    # Selection succeeded, navigation started
    _nav_mode = "autonomous"  # Reuse autonomous mode for execution
    _nav_obs = obs

    logger.info(
        "Semantic nav started: '%s' -> '%s' (ID %d)",
        instruction,
        status.selected_object_label,
        status.selected_object_id,
    )

    return {
        "status": "started",
        "instruction": instruction,
        "selected_object": status.selected_object_label,
        "selected_id": status.selected_object_id,
        "vlm_reasoning": status.vlm_reasoning,
        "confidence": status.vlm_confidence,
    }


def _semantic_nav_tick() -> dict:
    """Execute one semantic navigation step.

    Returns:
        Semantic nav status dict for the frame.
    """
    global _nav_mode, _nav_obs

    if _semantic_navigator is None or _semantic_navigator.is_done:
        return {"mode": "manual", "phase": "idle"}

    obs = _nav_obs

    # Run perception pipeline for occupancy grid + state estimation
    pipeline = _run_perception_pipeline(obs)
    (vo_est, fwd_det, rear_det, occ_grid,
     _fwd_pc, _rear_pc, _merged_pc, rear_rot) = pipeline

    # Update state estimator
    _nav_estimator.predict(obs.imu, dt=1.0)
    if vo_est.is_valid:
        pose_est = _nav_estimator.update_vo(vo_est)
    else:
        pose_est = _nav_estimator.get_estimate()

    # Check for rear obstacles
    rear_warning = (
        rear_det.obstacle_count > 0
        and float(np.min(obs.rear_depth[obs.rear_depth > 0])) < 0.5
        if np.any(obs.rear_depth > 0)
        else False
    )

    # Check collision
    collided = obs.state.collided

    # Execute one navigation step
    action, status = _semantic_navigator.step(
        pose_est,
        occ_grid,
        rear_warning,
        collided,
    )

    # Execute the action
    if not _semantic_navigator.is_done:
        # Parse continuous action: "turn_to:{yaw}" or "turn_to:{yaw}:move"
        parts = action.split(":")
        target_yaw = float(parts[1])
        move_forward = len(parts) > 2 and parts[2] == "move"
        _nav_obs = _vehicle.step_with_heading(target_yaw, move_forward)

    # Build status dict
    status_dict = {
        "mode": status.mode,
        "instruction": status.instruction,
        "phase": status.phase,
        "selected_object_label": status.selected_object_label,
        "selected_object_id": status.selected_object_id,
        "goal_position": status.goal_position,
        "goal_reached": status.goal_reached,
        "steps_taken": status.steps_taken,
        "vlm_calls": status.vlm_calls,
        "vlm_reasoning": status.vlm_reasoning,
        "vlm_confidence": status.vlm_confidence,
        "termination_reason": status.termination_reason,
        "distance_to_goal": status.distance_to_goal,
        "spl": status.spl,
        "action": action,
    }

    # Check if episode ended
    if _semantic_navigator.is_done:
        _nav_mode = "manual"
        metrics = _semantic_navigator.finish_episode()
        logger.info(
            "Semantic nav ended: %s, SPL=%.3f, steps=%d",
            metrics.termination_reason,
            metrics.spl,
            metrics.steps,
        )

    return status_dict


def _stop_semantic_nav() -> dict:
    """Stop semantic navigation and return to manual mode."""
    global _nav_mode

    if _semantic_navigator is not None and not _semantic_navigator.is_done:
        _semantic_navigator.reset()
        logger.info("Semantic nav stopped by user.")

    _nav_mode = "manual"
    return {"mode": "manual"}


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

            if msg_type == "change_scene":
                # Change to a different scene
                scene_id = msg.get("scene_id")
                if not scene_id:
                    await ws.send_text(json.dumps({"error": "No scene_id provided"}))
                    continue

                if scene_id == _current_scene_id:
                    # Already on this scene, just send current frame
                    obs = _vehicle.get_initial_observations()
                    frame = _build_frame(obs)
                    frame["scene_changed"] = False
                    frame["current_scene"] = _current_scene_id
                    await ws.send_text(json.dumps(frame))
                    continue

                success = _reinitialize_simulator(scene_id)
                if not success:
                    await ws.send_text(json.dumps({
                        "error": f"Failed to load scene: {scene_id}",
                        "current_scene": _current_scene_id,
                    }))
                    continue

                obs = _vehicle.get_initial_observations()
                frame = _build_frame(obs)
                frame["scene_changed"] = True
                frame["current_scene"] = _current_scene_id
                await ws.send_text(json.dumps(frame))

            elif msg_type == "reset":
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

            elif msg_type == "start_nav_to_object":
                # Start autonomous navigation to semantic object
                object_id = msg.get("object_id")
                logger.info("start_nav_to_object: received object_id=%s (type=%s)", object_id, type(object_id).__name__)
                if _semantic_index is None:
                    await ws.send_text(json.dumps({"error": "No semantic data for this scene"}))
                    continue
                logger.info("_semantic_index has %d objects, keys sample: %s",
                           len(_semantic_index.objects),
                           list(_semantic_index.objects.keys())[:5])
                if object_id not in _semantic_index.objects:
                    await ws.send_text(json.dumps({"error": f"Object {object_id} not found"}))
                    continue
                obj = _semantic_index.objects[object_id]
                logger.info("Looked up object_id=%s -> obj.label=%s, obj.instance_name=%s",
                           object_id, obj.label, obj.instance_name)
                if obj.navmesh_position is None:
                    await ws.send_text(json.dumps({"error": f"{obj.instance_name} is not navigable"}))
                    continue
                logger.info("Starting navigation to %s at %s (centroid=%s)",
                           obj.instance_name, obj.navmesh_position, obj.centroid)
                _start_autonomous_nav(obj.navmesh_position)
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

            # M5 VLM navigation handlers
            elif msg_type == "start_vlm_nav":
                instruction = msg.get("instruction", "explore the environment")
                result = _start_vlm_nav(instruction)
                if "error" in result:
                    await ws.send_text(json.dumps({"error": result["error"]}))
                    continue
                obs = _nav_obs if _nav_obs is not None else _vehicle.get_initial_observations()
                frame = _build_frame(obs)
                frame["vlm_status"] = {
                    "mode": "vlm_nav",
                    "instruction": instruction,
                    "goal_reached": False,
                    "steps_taken": 0,
                    "vlm_calls": 0,
                    "last_vlm_reasoning": "",
                    "confidence": 0.0,
                    "target_visible": False,
                    "termination_reason": None,
                    "current_pixel": None,
                    "current_world_target": None,
                    "target_valid": False,
                    "target_failure_reason": None,
                    "depth_at_target": 0.0,
                    "action": None,
                }
                await ws.send_text(json.dumps(frame))

            elif msg_type == "vlm_tick":
                if _vlm_mode == "vlm_nav":
                    vlm_dict = _vlm_tick()
                    obs = _nav_obs if _nav_obs is not None else _vehicle.get_initial_observations()
                    frame = _build_frame(obs)
                    frame["vlm_status"] = vlm_dict
                    await ws.send_text(json.dumps(frame))
                else:
                    # Not in VLM mode, just return current state
                    obs = _vehicle.get_initial_observations()
                    frame = _build_frame(obs)
                    await ws.send_text(json.dumps(frame))

            elif msg_type == "stop_vlm_nav":
                _stop_vlm_nav()
                obs = _vehicle.get_initial_observations()
                frame = _build_frame(obs)
                frame["vlm_status"] = {"mode": "manual"}
                await ws.send_text(json.dumps(frame))

            # M5 Semantic constrained navigation handlers
            elif msg_type == "start_semantic_nav":
                instruction = msg.get("instruction", "")
                result = _start_semantic_nav(instruction)
                if "error" in result:
                    await ws.send_text(json.dumps({"error": result["error"]}))
                    continue
                obs = _nav_obs if _nav_obs is not None else _vehicle.get_initial_observations()
                frame = _build_frame(obs)
                frame["vlm_status"] = {
                    "mode": "vlm_nav",
                    "instruction": instruction,
                    "goal_reached": False,
                    "steps_taken": 0,
                    "vlm_calls": 1,
                    "last_vlm_reasoning": result.get("vlm_reasoning", ""),
                    "confidence": result.get("confidence", 0.0),
                    "target_visible": True,
                    "termination_reason": None,
                    "selected_object_label": result.get("selected_object"),
                    "selected_object_id": result.get("selected_id"),
                }
                await ws.send_text(json.dumps(frame))

            elif msg_type == "semantic_nav_tick":
                if _nav_mode == "autonomous" and _semantic_navigator is not None and not _semantic_navigator.is_done:
                    semantic_dict = _semantic_nav_tick()
                    obs = _nav_obs if _nav_obs is not None else _vehicle.get_initial_observations()
                    frame = _build_frame(obs)
                    frame["vlm_status"] = {
                        "mode": "vlm_nav" if semantic_dict.get("mode") == "semantic_nav" else "manual",
                        "instruction": semantic_dict.get("instruction", ""),
                        "goal_reached": semantic_dict.get("goal_reached", False),
                        "steps_taken": semantic_dict.get("steps_taken", 0),
                        "vlm_calls": semantic_dict.get("vlm_calls", 0),
                        "last_vlm_reasoning": semantic_dict.get("vlm_reasoning", ""),
                        "confidence": semantic_dict.get("vlm_confidence", 0.0),
                        "target_visible": True,
                        "termination_reason": semantic_dict.get("termination_reason"),
                        "selected_object_label": semantic_dict.get("selected_object_label"),
                        "selected_object_id": semantic_dict.get("selected_object_id"),
                    }
                    await ws.send_text(json.dumps(frame))
                else:
                    # Not in semantic nav mode, just return current state
                    obs = _vehicle.get_initial_observations()
                    frame = _build_frame(obs)
                    await ws.send_text(json.dumps(frame))

            elif msg_type == "stop_semantic_nav":
                _stop_semantic_nav()
                obs = _vehicle.get_initial_observations()
                frame = _build_frame(obs)
                frame["vlm_status"] = {"mode": "manual"}
                await ws.send_text(json.dumps(frame))

            elif msg_type == "project_pixel":
                # Project a pixel coordinate from forward camera to navmesh
                u = msg.get("u", 0)
                v = msg.get("v", 0)
                logger.info("project_pixel: u=%d, v=%d", u, v)

                obs = _vehicle.get_initial_observations()
                proj_result = pixel_to_world(
                    u, v,
                    obs.depth,
                    obs.state.position,
                    obs.state.rotation,
                )
                logger.info(
                    "project_pixel result: valid=%s, depth=%.2f, reason=%s",
                    proj_result.is_valid, proj_result.depth_value,
                    proj_result.failure_reason,
                )

                projection_data = {
                    "is_valid": False,
                    "pixel": [u, v],
                    "world_point": None,
                    "navmesh_point": None,
                    "depth_value": proj_result.depth_value,
                    "failure_reason": proj_result.failure_reason,
                }

                if proj_result.is_valid and proj_result.world_point is not None:
                    logger.info(
                        "project_pixel world_point: [%.2f, %.2f, %.2f]",
                        proj_result.world_point[0],
                        proj_result.world_point[1],
                        proj_result.world_point[2],
                    )
                    # Snap to navmesh
                    snapped, snap_ok, snap_reason = snap_to_navmesh(
                        proj_result.world_point,
                        _vehicle.pathfinder,
                        max_distance=2.0,
                    )
                    projection_data["world_point"] = proj_result.world_point.tolist()
                    logger.info(
                        "project_pixel snap: ok=%s, reason=%s",
                        snap_ok, snap_reason,
                    )

                    if snap_ok and snapped is not None:
                        projection_data["is_valid"] = True
                        projection_data["navmesh_point"] = snapped.tolist()
                        logger.info(
                            "project_pixel navmesh_point: [%.2f, %.2f, %.2f]",
                            snapped[0], snapped[1], snapped[2],
                        )
                    else:
                        projection_data["failure_reason"] = snap_reason or "snap_failed"

                frame = _build_frame(obs)
                frame["projection_result"] = projection_data
                await ws.send_text(json.dumps(frame))

            elif "action" in msg:
                # Manual control: ignore during autonomous or VLM mode
                if _nav_mode == "autonomous" or _vlm_mode == "vlm_nav":
                    logger.debug("Ignoring manual action during autonomous/VLM mode.")
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
