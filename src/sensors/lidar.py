"""Depth image to 3D point cloud conversion (simulated LiDAR).

Converts pinhole depth images from habitat-sim into 3D point clouds in
camera frame, then optionally transforms to world frame.

Camera frame convention (habitat-sim): X=right, Y=up, Z=backward
(camera looks along -Z). Depth image is z-buffer depth (distance along -Z).
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray

from src.sensors.imu import quat_to_rotation_matrix


@dataclass
class PointCloud:
    points: NDArray[np.float32]  # [N, 3] XYZ in the requested frame
    num_valid: int  # count of non-infinite, non-NaN points


def depth_to_point_cloud(
    depth: NDArray[np.float32],
    hfov_deg: float,
    max_depth: float = 10.0,
) -> PointCloud:
    """Convert a pinhole depth image to a 3D point cloud in camera frame.

    Camera frame convention (habitat-sim):
      X = right, Y = up, Z = backward (camera looks along -Z)

    The depth image contains z-buffer depth (distance along -Z axis).

    Args:
        depth: [H, W] float32 z-buffer depth in meters.
        hfov_deg: Horizontal field of view in degrees.
        max_depth: Discard points beyond this range.

    Returns:
        PointCloud with points in camera-local frame.
    """
    H, W = depth.shape

    # Build valid mask: exclude zero, negative, NaN, inf, and beyond max_depth
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)

    # Focal length from HFOV
    hfov_rad = math.radians(hfov_deg)
    focal_length = W / (2.0 * math.tan(hfov_rad / 2.0))

    # Center of image
    cx = W / 2.0
    cy = H / 2.0

    # Pixel coordinate grids
    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float32),
                                  np.arange(H, dtype=np.float32))

    # Extract valid pixels
    d_valid = depth[valid]
    u_valid = u_grid[valid]
    v_valid = v_grid[valid]

    # Inverse pinhole projection
    x = (u_valid - cx) * d_valid / focal_length
    y = -(v_valid - cy) * d_valid / focal_length  # negate: pixel Y is down, camera Y is up
    z = -d_valid  # camera looks along -Z

    points = np.stack([x, y, z], axis=-1).astype(np.float32)
    return PointCloud(points=points, num_valid=int(points.shape[0]))


def transform_point_cloud(
    pc: PointCloud,
    position: NDArray[np.float64],
    rotation_quat: NDArray[np.float64],
) -> PointCloud:
    """Transform point cloud from camera frame to world frame.

    Args:
        pc: Point cloud in camera frame.
        position: [3] sensor world position.
        rotation_quat: [4] sensor world rotation [w, x, y, z].

    Returns:
        New PointCloud with points in world frame.
    """
    if pc.num_valid == 0:
        return PointCloud(
            points=np.empty((0, 3), dtype=np.float32),
            num_valid=0,
        )

    R = quat_to_rotation_matrix(rotation_quat)
    pos = np.asarray(position, dtype=np.float64)

    # R @ points.T -> [3, N], then transpose + add position
    points_world = (R @ pc.points.astype(np.float64).T).T + pos
    return PointCloud(
        points=points_world.astype(np.float32),
        num_valid=pc.num_valid,
    )


def merge_point_clouds(clouds: List[PointCloud]) -> PointCloud:
    """Concatenate multiple point clouds into one."""
    if not clouds:
        return PointCloud(points=np.empty((0, 3), dtype=np.float32), num_valid=0)

    all_points = [c.points for c in clouds if c.num_valid > 0]
    if not all_points:
        return PointCloud(points=np.empty((0, 3), dtype=np.float32), num_valid=0)

    merged = np.concatenate(all_points, axis=0)
    return PointCloud(points=merged, num_valid=int(merged.shape[0]))
