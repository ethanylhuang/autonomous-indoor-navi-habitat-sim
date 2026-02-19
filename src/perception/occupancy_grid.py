"""Fused 2D bird's-eye occupancy grid from LiDAR + semantic detections.

The grid is centered on the agent and represents a fixed-size local area.
It is rebuilt each step (no temporal accumulation in M2).

Grid coordinate system:
  - grid[0, 0] corresponds to origin in world XZ coordinates.
  - Row index increases with world +Z, column index increases with world +X.
  - grid[row, col] where row = (world_z - origin_z) / resolution,
    col = (world_x - origin_x) / resolution.

Values: 0.0 = free, 1.0 = occupied, 0.5 = unknown.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.perception.obstacle_detector import ObstacleDetection
from src.sensors.lidar import PointCloud


@dataclass
class OccupancyGridData:
    grid: NDArray[np.float32]  # [grid_h, grid_w] values in [0.0, 1.0]
    resolution: float  # meters per cell
    origin: NDArray[np.float64]  # [2] world XZ of grid[0, 0] corner
    shape: Tuple[int, int]  # (grid_h, grid_w)
    timestamp_step: int


class OccupancyGrid:
    """Local ego-centric occupancy grid fused from LiDAR + semantic detections.

    The grid is centered on the agent and represents a fixed-size local area.
    It is rebuilt each step (no temporal accumulation in M2).
    """

    def __init__(
        self,
        grid_size: float = 10.0,
        resolution: float = 0.05,
        height_min: float = 0.1,
        height_max: float = 2.0,
        obstacle_confidence: float = 0.9,
        free_confidence: float = 0.1,
    ) -> None:
        """
        Args:
            grid_size: Total side length of square grid in meters.
            resolution: Meters per cell.
            height_min: Ignore points below this Y value (floor).
            height_max: Ignore points above this Y value (ceiling).
            obstacle_confidence: Value for occupied cells.
            free_confidence: Value for free cells.
        """
        self._grid_size = grid_size
        self._resolution = resolution
        self._grid_dim = int(grid_size / resolution)
        self._height_min = height_min
        self._height_max = height_max
        self._obstacle_confidence = obstacle_confidence
        self._free_confidence = free_confidence
        self._step: int = 0

    def update(
        self,
        agent_position: NDArray[np.float64],
        agent_rotation: NDArray[np.float64],
        point_clouds: List[PointCloud],
        obstacle_detections: Optional[List[Tuple[ObstacleDetection, NDArray, NDArray, float]]] = None,
    ) -> OccupancyGridData:
        """Build the occupancy grid for the current step.

        Args:
            agent_position: [3] world frame position.
            agent_rotation: [4] quaternion [w, x, y, z].
            point_clouds: World-frame point clouds from LiDAR.
            obstacle_detections: Optional list of tuples:
                (detection, depth_image, sensor_rotation_quat, hfov_deg).
                Each provides additional obstacle evidence from semantic sensors.

        Returns:
            OccupancyGridData with the built grid.
        """
        self._step += 1

        dim = self._grid_dim
        grid = np.full((dim, dim), 0.5, dtype=np.float32)

        # Step 1: Compute grid origin (agent-centered)
        origin_x = agent_position[0] - self._grid_size / 2.0
        origin_z = agent_position[2] - self._grid_size / 2.0
        origin = np.array([origin_x, origin_z], dtype=np.float64)

        # Agent cell in grid
        agent_col = int((agent_position[0] - origin_x) / self._resolution)
        agent_row = int((agent_position[2] - origin_z) / self._resolution)
        agent_col = max(0, min(dim - 1, agent_col))
        agent_row = max(0, min(dim - 1, agent_row))

        # Collect all occupied cells for ray casting
        occupied_cells: List[Tuple[int, int]] = []

        # Step 2: Project LiDAR points
        for pc in point_clouds:
            if pc.num_valid == 0:
                continue
            self._project_points_to_grid(
                pc.points, origin_x, origin_z, dim, grid, occupied_cells,
            )

        # Step 3: Project semantic obstacle detections (back-project to 3D)
        if obstacle_detections:
            for det, depth_img, sensor_rot_quat, hfov_deg in obstacle_detections:
                if not np.any(det.mask):
                    continue
                self._project_obstacle_detection(
                    det, depth_img, sensor_rot_quat, hfov_deg,
                    agent_position, origin_x, origin_z, dim, grid, occupied_cells,
                )

        # Step 4: Ray-cast free space
        self._raycast_free_space(
            agent_row, agent_col, occupied_cells, dim, grid,
        )

        return OccupancyGridData(
            grid=grid,
            resolution=self._resolution,
            origin=origin,
            shape=(dim, dim),
            timestamp_step=self._step,
        )

    def reset(self) -> None:
        """Clear any internal state."""
        self._step = 0

    def _project_points_to_grid(
        self,
        points: NDArray[np.float32],
        origin_x: float,
        origin_z: float,
        dim: int,
        grid: NDArray[np.float32],
        occupied_cells: List[Tuple[int, int]],
    ) -> None:
        """Project world-frame 3D points onto the 2D grid."""
        # Height filtering (Y axis)
        y_vals = points[:, 1]
        height_mask = (y_vals >= self._height_min) & (y_vals <= self._height_max)
        filtered = points[height_mask]

        if len(filtered) == 0:
            return

        # Convert world XZ to grid cell indices
        cols = ((filtered[:, 0] - origin_x) / self._resolution).astype(np.int32)
        rows = ((filtered[:, 2] - origin_z) / self._resolution).astype(np.int32)

        # Filter to within grid bounds
        valid = (cols >= 0) & (cols < dim) & (rows >= 0) & (rows < dim)
        valid_cols = cols[valid]
        valid_rows = rows[valid]

        # Mark cells as occupied
        grid[valid_rows, valid_cols] = self._obstacle_confidence

        # Store for ray casting
        for r, c in zip(valid_rows.tolist(), valid_cols.tolist()):
            occupied_cells.append((r, c))

    def _project_obstacle_detection(
        self,
        det: ObstacleDetection,
        depth_img: NDArray[np.float32],
        sensor_rot_quat: NDArray[np.float64],
        hfov_deg: float,
        agent_position: NDArray[np.float64],
        origin_x: float,
        origin_z: float,
        dim: int,
        grid: NDArray[np.float32],
        occupied_cells: List[Tuple[int, int]],
    ) -> None:
        """Back-project obstacle mask pixels to 3D and project to grid."""
        from src.sensors.imu import quat_to_rotation_matrix

        H, W = det.mask.shape
        hfov_rad = math.radians(hfov_deg)
        fx = W / (2.0 * math.tan(hfov_rad / 2.0))
        cx = W / 2.0
        cy = H / 2.0

        # Get obstacle pixel coordinates
        vs, us = np.where(det.mask)
        depths = depth_img[vs, us]

        # Filter valid depths
        valid = np.isfinite(depths) & (depths > 0.0)
        vs = vs[valid]
        us = us[valid]
        depths = depths[valid]

        if len(depths) == 0:
            return

        # Inverse pinhole projection to camera frame
        x_cam = (us.astype(np.float64) - cx) * depths / fx
        y_cam = -(vs.astype(np.float64) - cy) * depths / fx
        z_cam = -depths.astype(np.float64)

        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

        # Transform to world frame
        R = quat_to_rotation_matrix(sensor_rot_quat)
        points_world = (R @ points_cam.T).T + agent_position

        # Project to grid (reuse the same logic)
        self._project_points_to_grid(
            points_world.astype(np.float32),
            origin_x, origin_z, dim, grid, occupied_cells,
        )

    def _raycast_free_space(
        self,
        agent_row: int,
        agent_col: int,
        occupied_cells: List[Tuple[int, int]],
        dim: int,
        grid: NDArray[np.float32],
    ) -> None:
        """Mark cells between agent and occupied cells as free using Bresenham's line."""
        # Deduplicate occupied cells
        occupied_set = set(occupied_cells)

        for occ_r, occ_c in occupied_set:
            # Bresenham's line from agent to occupied cell (exclusive of endpoint)
            for r, c in _bresenham(agent_row, agent_col, occ_r, occ_c):
                if (r, c) == (occ_r, occ_c):
                    break
                if 0 <= r < dim and 0 <= c < dim:
                    # Only mark as free if currently unknown
                    if grid[r, c] == 0.5:
                        grid[r, c] = self._free_confidence


def _bresenham(r0: int, c0: int, r1: int, c1: int):
    """Yield (row, col) cells along a line from (r0, c0) to (r1, c1)."""
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    while True:
        yield r, c
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
