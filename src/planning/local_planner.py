"""Rule-based local planner with continuous heading control.

Uses ray-marching through occupancy grid to measure clearances
and computes target heading with center-seeking behavior.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.perception.occupancy_grid import OccupancyGridData
from src.utils.transforms import normalize_angle


@dataclass
class LocalPlanResult:
    """Result from local planner with continuous heading control."""

    target_yaw: float  # Absolute heading to face (radians)
    move_forward: bool  # Whether to move after turning
    heading_error: float  # radians, angle to waypoint
    is_blocked: bool  # True if no safe action found
    rear_obstacle_warning: bool  # True if rear obstacles detected within threshold
    clearance_left: float  # meters, clearance to left
    clearance_right: float  # meters, clearance to right
    clearance_forward: float  # meters, clearance ahead


class LocalPlanner:
    """Rule-based local planner with continuous heading control.

    Decision logic:
    1. Compute ideal heading toward waypoint
    2. Check forward clearance along ideal heading
    3. If blocked, bias heading toward side with more clearance
    4. If severely blocked, turn toward clearer side
    5. Always maintain minimum_clearance from obstacles

    Uses ray-marching through occupancy grid for clearance measurement.
    """

    def __init__(
        self,
        minimum_clearance: float = 0.4,
        forward_check_distance: float = 1.0,
        side_check_angle: float = 45.0,
        center_seek_gain: float = 0.3,
        heading_tolerance: float = 0.1,
    ) -> None:
        self._minimum_clearance = minimum_clearance
        self._forward_check_distance = forward_check_distance
        self._side_check_angle_rad = math.radians(side_check_angle)
        self._center_seek_gain = center_seek_gain
        self._heading_tolerance = heading_tolerance

        # Ray-march step size for clearance measurement
        self._step_size = 0.05  # 5cm steps
        self._num_steps = int(forward_check_distance / self._step_size)

    def plan(
        self,
        agent_position: NDArray[np.float64],
        agent_yaw: float,
        target_waypoint: NDArray[np.float64],
        occupancy_grid: OccupancyGridData,
        rear_obstacle_detected: bool,
    ) -> LocalPlanResult:
        """Select heading and movement command given current state and grid.

        Args:
            agent_position: [3] current position.
            agent_yaw: Current heading (radians).
            target_waypoint: [3] next waypoint position.
            occupancy_grid: Current occupancy from M2.
            rear_obstacle_detected: From rear camera obstacle detector.

        Returns:
            LocalPlanResult with target_yaw and move_forward fields populated.
        """
        x = float(agent_position[0])
        z = float(agent_position[2])

        # 1. Compute ideal heading toward waypoint
        dx = target_waypoint[0] - x
        dz = target_waypoint[2] - z
        ideal_yaw = math.atan2(-dx, -dz)

        # 2. Measure clearances
        clearance_forward = self._measure_clearance(x, z, ideal_yaw, occupancy_grid)
        clearance_left = self._measure_clearance(
            x, z, normalize_angle(ideal_yaw + self._side_check_angle_rad), occupancy_grid
        )
        clearance_right = self._measure_clearance(
            x, z, normalize_angle(ideal_yaw - self._side_check_angle_rad), occupancy_grid
        )

        # 3. Compute target heading with center-seeking
        target_yaw = self._compute_target_heading(
            ideal_yaw, clearance_forward, clearance_left, clearance_right
        )

        # 4. Decide whether to move forward
        # ALWAYS move forward after turning if path is clear — turn+move is one action
        # Check clearance in the TARGET direction (where we'll be facing after turn)
        clearance_in_target_dir = self._measure_clearance(x, z, target_yaw, occupancy_grid)
        safe_to_move = clearance_in_target_dir > self._minimum_clearance
        move_forward = safe_to_move  # Always move if safe — no alignment check needed

        # 5. Check if completely blocked
        all_clearances = [clearance_forward, clearance_left, clearance_right]
        is_blocked = all(c < self._minimum_clearance for c in all_clearances)

        # 6. Compute heading error to waypoint
        heading_error = abs(normalize_angle(agent_yaw - ideal_yaw))

        return LocalPlanResult(
            target_yaw=target_yaw,
            move_forward=move_forward,
            heading_error=heading_error,
            is_blocked=is_blocked,
            rear_obstacle_warning=rear_obstacle_detected,
            clearance_left=clearance_left,
            clearance_right=clearance_right,
            clearance_forward=clearance_forward,
        )

    def _measure_clearance(
        self,
        x: float,
        z: float,
        yaw: float,
        grid: OccupancyGridData,
    ) -> float:
        """Ray-march through occupancy grid, return distance to first obstacle.

        Args:
            x, z: World position to ray-march from.
            yaw: Direction to ray-march (radians).
            grid: Occupancy grid data.

        Returns:
            Distance to first obstacle in meters, or forward_check_distance if clear.
        """
        dir_x = -math.sin(yaw)
        dir_z = -math.cos(yaw)

        res = grid.resolution
        origin_x = grid.origin[0]
        origin_z = grid.origin[1]
        gh, gw = grid.shape

        for i in range(1, self._num_steps + 1):
            dist = i * self._step_size
            sample_x = x + dir_x * dist
            sample_z = z + dir_z * dist

            # Convert to grid coordinates
            col = int((sample_x - origin_x) / res)
            row = int((sample_z - origin_z) / res)

            # Check bounds
            if col < 0 or col >= gw or row < 0 or row >= gh:
                return dist

            # Check occupancy
            if grid.grid[row, col] > 0.5:
                return max(0.0, dist - self._step_size)

        return self._forward_check_distance

    def _compute_target_heading(
        self,
        ideal_yaw: float,
        clearance_forward: float,
        clearance_left: float,
        clearance_right: float,
    ) -> float:
        """Compute target heading with center-seeking bias.

        Args:
            ideal_yaw: Desired heading toward waypoint.
            clearance_forward: Forward clearance in meters.
            clearance_left: Left side clearance in meters.
            clearance_right: Right side clearance in meters.

        Returns:
            Target heading in radians.
        """
        if clearance_forward > self._minimum_clearance * 2:
            # Path clear - apply center-seeking bias
            imbalance = clearance_left - clearance_right
            bias = imbalance * self._center_seek_gain
            # Clamp to 30 degrees
            bias = max(-math.pi / 6, min(math.pi / 6, bias))
            return normalize_angle(ideal_yaw + bias)

        elif clearance_forward > self._minimum_clearance:
            # Marginal clearance - stronger centering
            imbalance = clearance_left - clearance_right
            bias = imbalance * self._center_seek_gain * 2.0
            # Clamp to 45 degrees
            bias = max(-math.pi / 4, min(math.pi / 4, bias))
            return normalize_angle(ideal_yaw + bias)

        else:
            # Blocked - turn toward clearer side
            if clearance_left > clearance_right:
                turn = math.pi / 4 if clearance_left >= self._minimum_clearance else math.pi / 2
            else:
                turn = -math.pi / 4 if clearance_right >= self._minimum_clearance else -math.pi / 2
            return normalize_angle(ideal_yaw + turn)

    def reset(self) -> None:
        """Reset planner state (no state for this planner)."""
        pass
