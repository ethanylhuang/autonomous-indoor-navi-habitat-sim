"""Pixel coordinate to world coordinate projection for VLM navigation.

Transforms a 2D pixel (u, v) from the camera image into a 3D world coordinate
by combining depth data, camera intrinsics, and NavMesh snapping.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from configs.sensor_rig import HFOV, RGB_RESOLUTION, SENSOR_HEIGHT
from src.utils.transforms import focal_length_from_hfov, quat_to_rotation_matrix


@dataclass
class ProjectionResult:
    """Result of pixel-to-camera or pixel-to-world projection."""

    world_point: Optional[NDArray[np.float64]]  # [3] or None if invalid
    depth_value: float
    is_valid: bool
    failure_reason: Optional[str]


def _sample_depth_region(
    u: int,
    v: int,
    depth: NDArray[np.float32],
    patch_size: int = 11,
) -> Tuple[float, bool]:
    """Sample depth from a region around (u, v), returning median of valid values.

    Args:
        u: Center horizontal pixel coordinate.
        v: Center vertical pixel coordinate.
        depth: [H, W] depth image in meters.
        patch_size: Size of square patch to sample (must be odd).

    Returns:
        (depth_value, is_valid) - median depth if valid pixels found, else (0, False).
    """
    H, W = depth.shape
    half = patch_size // 2

    # Compute patch bounds (clipped to image)
    v_min = max(0, v - half)
    v_max = min(H, v + half + 1)
    u_min = max(0, u - half)
    u_max = min(W, u + half + 1)

    patch = depth[v_min:v_max, u_min:u_max]

    # Filter valid depths (finite, positive, within range)
    valid_mask = np.isfinite(patch) & (patch > 0.3) & (patch < 10.0)
    valid_depths = patch[valid_mask]

    if len(valid_depths) == 0:
        return 0.0, False

    return float(np.median(valid_depths)), True


def pixel_to_camera_frame(
    u: int,
    v: int,
    depth: NDArray[np.float32],
    hfov_deg: float = HFOV,
) -> ProjectionResult:
    """Project pixel (u, v) to 3D point in camera frame.

    Camera frame convention (habitat-sim):
      X = right, Y = up, Z = backward (camera looks along -Z)

    If the exact pixel has invalid depth, samples an 11x11 region around it
    and uses the median of valid depths.

    Args:
        u: Horizontal pixel coordinate (0 = left).
        v: Vertical pixel coordinate (0 = top).
        depth: [H, W] depth image in meters.
        hfov_deg: Horizontal field of view.

    Returns:
        ProjectionResult with 3D point in camera frame.
    """
    H, W = depth.shape

    # Bounds check
    if not (0 <= u < W and 0 <= v < H):
        return ProjectionResult(None, 0.0, False, "pixel_out_of_bounds")

    d = float(depth[v, u])

    # Validate depth - if single pixel invalid, try region sampling
    if not np.isfinite(d) or d <= 0.0 or d > 10.0 or d < 0.3:
        d, valid = _sample_depth_region(u, v, depth)
        if not valid:
            return ProjectionResult(None, 0.0, False, "depth_invalid_region")

    # Final validation (should pass if _sample_depth_region succeeded)
    if d > 10.0:
        return ProjectionResult(None, d, False, "depth_too_far")
    if d < 0.3:
        return ProjectionResult(None, d, False, "depth_too_close")

    focal_length = focal_length_from_hfov(hfov_deg, W)
    cx = W / 2.0
    cy = H / 2.0

    # Inverse pinhole projection
    x = (u - cx) * d / focal_length
    y = -(v - cy) * d / focal_length  # negate: pixel Y down, camera Y up
    z = -d  # camera looks along -Z

    point = np.array([x, y, z], dtype=np.float64)
    return ProjectionResult(point, d, True, None)


def camera_to_world_frame(
    camera_point: NDArray[np.float64],
    agent_position: NDArray[np.float64],
    agent_rotation: NDArray[np.float64],
    sensor_height: float = SENSOR_HEIGHT,
) -> NDArray[np.float64]:
    """Transform point from camera frame to world frame.

    Args:
        camera_point: [3] point in camera frame.
        agent_position: [3] agent position in world frame.
        agent_rotation: [4] agent rotation quaternion [w, x, y, z].
        sensor_height: Height of camera sensor above agent origin (Y axis).

    Returns:
        [3] point in world frame.
    """
    sensor_pos = agent_position.copy()
    sensor_pos[1] += sensor_height

    R = quat_to_rotation_matrix(agent_rotation)
    world_point = R @ camera_point + sensor_pos
    return world_point


def pixel_to_world(
    u: int,
    v: int,
    depth: NDArray[np.float32],
    agent_position: NDArray[np.float64],
    agent_rotation: NDArray[np.float64],
) -> ProjectionResult:
    """Full pipeline: pixel (u, v) to world coordinate.

    Args:
        u: Horizontal pixel coordinate (0 = left).
        v: Vertical pixel coordinate (0 = top).
        depth: [H, W] depth image in meters.
        agent_position: [3] agent position in world frame.
        agent_rotation: [4] agent rotation quaternion [w, x, y, z].

    Returns:
        ProjectionResult with world point or error.
    """
    cam_result = pixel_to_camera_frame(u, v, depth)
    if not cam_result.is_valid:
        return cam_result

    world_point = camera_to_world_frame(
        cam_result.world_point,
        agent_position,
        agent_rotation,
    )

    return ProjectionResult(world_point, cam_result.depth_value, True, None)


def snap_to_navmesh(
    world_point: NDArray[np.float64],
    pathfinder,
    max_distance: float = 2.0,
) -> Tuple[Optional[NDArray[np.float64]], bool, Optional[str]]:
    """Snap a world point to the nearest valid NavMesh location.

    Args:
        world_point: [3] point in world frame.
        pathfinder: habitat_sim PathFinder instance.
        max_distance: Maximum snap distance in meters.

    Returns:
        (snapped_point, success, failure_reason)
    """
    # Try direct snap first (if available)
    if hasattr(pathfinder, "snap_point"):
        snapped = pathfinder.snap_point(world_point)
        if snapped is not None and pathfinder.is_navigable(snapped):
            return np.array(snapped, dtype=np.float64), True, None

    # Fallback: get_random_navigable_point_near
    snapped = pathfinder.get_random_navigable_point_near(
        world_point, radius=max_distance, max_tries=30
    )

    if snapped is None:
        return None, False, "no_navigable_point_nearby"

    snapped = np.array(snapped, dtype=np.float64)
    dist = float(np.linalg.norm(snapped - world_point))
    if dist > max_distance:
        return None, False, "snap_distance_exceeded"

    return snapped, True, None
