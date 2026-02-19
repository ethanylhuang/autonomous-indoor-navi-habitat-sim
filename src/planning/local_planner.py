"""DWA-inspired local planner adapted for discrete action space.

Evaluates candidate action sequences against the occupancy grid.
Scores by: goal heading alignment, obstacle clearance, forward progress.
Only the first action of the best sequence is executed (receding horizon).
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from src.perception.occupancy_grid import OccupancyGridData
from src.utils.transforms import normalize_angle


@dataclass
class LocalPlanResult:
    best_action: str  # "move_forward", "turn_left", or "turn_right"
    score: float  # combined score of the best trajectory
    heading_error: float  # radians, angle to waypoint
    nearest_obstacle_dist: float  # meters, closest obstacle in forward arc
    is_blocked: bool  # True if no safe action found
    rear_obstacle_warning: bool  # True if rear obstacles detected within threshold


class LocalPlanner:
    """DWA-inspired local planner adapted for discrete action space.

    Evaluates candidate action sequences against the occupancy grid.
    Scores by: goal heading alignment, obstacle clearance, forward progress.
    """

    def __init__(
        self,
        lookahead_steps: int = 3,
        heading_weight: float = 1.0,
        clearance_weight: float = 0.8,
        progress_weight: float = 0.5,
        obstacle_check_radius: float = 0.3,
        rear_check_distance: float = 0.5,
        move_amount: float = 0.25,
        turn_amount_deg: float = 10.0,
    ) -> None:
        self._lookahead_steps = lookahead_steps
        self._heading_weight = heading_weight
        self._clearance_weight = clearance_weight
        self._progress_weight = progress_weight
        self._obstacle_check_radius = obstacle_check_radius
        self._rear_check_distance = rear_check_distance
        self._move_amount = move_amount
        self._turn_amount_rad = math.radians(turn_amount_deg)

        # Pre-generate candidate action sequences
        self._candidates = self._generate_candidates()

    def plan(
        self,
        agent_position: NDArray[np.float64],
        agent_yaw: float,
        target_waypoint: NDArray[np.float64],
        occupancy_grid: OccupancyGridData,
        rear_obstacle_detected: bool,
    ) -> LocalPlanResult:
        """Select the best discrete action given current state and grid.

        Args:
            agent_position: [3] current position.
            agent_yaw: Current heading (radians).
            target_waypoint: [3] next waypoint position.
            occupancy_grid: Current occupancy from M2.
            rear_obstacle_detected: From rear camera obstacle detector.

        Returns:
            LocalPlanResult with selected action and diagnostics.
        """
        start_x = float(agent_position[0])
        start_z = float(agent_position[2])
        start_yaw = float(agent_yaw)

        best_score = -float("inf")
        best_action = "turn_left"  # fallback
        best_heading_error = float("inf")
        best_nearest_dist = 0.0
        all_blocked = True

        for candidate in self._candidates:
            traj = self._simulate_trajectory(start_x, start_z, start_yaw, candidate)
            total_score, heading_err, nearest_dist = self._score_trajectory(
                traj, target_waypoint, occupancy_grid,
            )

            # Penalize backward-heading trajectories when rear obstacles present
            if rear_obstacle_detected:
                final_yaw = traj[-1][2]
                yaw_diff = abs(normalize_angle(final_yaw - start_yaw))
                if yaw_diff > math.pi * 0.75:  # turned ~180 degrees
                    total_score -= 2.0

            # Check if this trajectory is safe (not colliding)
            if nearest_dist > self._obstacle_check_radius:
                all_blocked = False

            if total_score > best_score:
                best_score = total_score
                best_action = candidate[0]
                best_heading_error = heading_err
                best_nearest_dist = nearest_dist

        return LocalPlanResult(
            best_action=best_action,
            score=best_score,
            heading_error=best_heading_error,
            nearest_obstacle_dist=best_nearest_dist,
            is_blocked=all_blocked,
            rear_obstacle_warning=rear_obstacle_detected,
        )

    def _generate_candidates(self) -> List[List[str]]:
        """Generate candidate action sequences.

        Produces ~20 candidates mixing turns and forward movement.
        """
        candidates: List[List[str]] = []

        # Single forward
        candidates.append(["move_forward"])

        # Turn k times then forward, for k in 1..5
        for k in range(1, 6):
            candidates.append(["turn_left"] * k + ["move_forward"])
            candidates.append(["turn_right"] * k + ["move_forward"])

        # Pure rotations for k in 1..3
        for k in range(1, 4):
            candidates.append(["turn_left"] * k)
            candidates.append(["turn_right"] * k)

        # Double forward (straight ahead 2 steps)
        candidates.append(["move_forward", "move_forward"])

        # Lane change: forward + turn + forward
        candidates.append(["move_forward", "turn_left", "move_forward"])
        candidates.append(["move_forward", "turn_right", "move_forward"])

        return candidates

    def _simulate_trajectory(
        self,
        start_x: float,
        start_z: float,
        start_yaw: float,
        actions: List[str],
    ) -> List[Tuple[float, float, float]]:
        """Simulate action sequence to get trajectory of (x, z, yaw) states.

        Uses kinematic model: move_forward advances move_amount in heading
        direction, turn_left/right changes yaw by +/- turn_amount.
        """
        x, z, yaw = start_x, start_z, start_yaw
        trajectory = [(x, z, yaw)]

        for action in actions:
            if action == "move_forward":
                # Habitat-sim: forward is -Z direction, yaw about Y
                # dx = -sin(yaw) * amount, dz = -cos(yaw) * amount
                x += -math.sin(yaw) * self._move_amount
                z += -math.cos(yaw) * self._move_amount
            elif action == "turn_left":
                yaw += self._turn_amount_rad
            elif action == "turn_right":
                yaw -= self._turn_amount_rad
            yaw = normalize_angle(yaw)
            trajectory.append((x, z, yaw))

        return trajectory

    def _score_trajectory(
        self,
        trajectory: List[Tuple[float, float, float]],
        target_waypoint: NDArray[np.float64],
        occupancy_grid: OccupancyGridData,
    ) -> Tuple[float, float, float]:
        """Score a candidate trajectory.

        Returns:
            (total_score, heading_error, nearest_obstacle_dist)
        """
        if len(trajectory) < 2:
            return (0.0, float("inf"), 0.0)

        final_x, final_z, final_yaw = trajectory[-1]
        start_x, start_z, _ = trajectory[0]
        target_x = float(target_waypoint[0])
        target_z = float(target_waypoint[2])

        # 1. Heading score: cos(angle between final heading and waypoint direction)
        dx_to_target = target_x - final_x
        dz_to_target = target_z - final_z
        dist_to_target = math.sqrt(dx_to_target * dx_to_target + dz_to_target * dz_to_target)

        if dist_to_target < 1e-6:
            heading_score = 1.0
            heading_error = 0.0
        else:
            # Desired heading: angle from current position to waypoint
            desired_yaw = math.atan2(-dx_to_target, -dz_to_target)
            heading_error = normalize_angle(final_yaw - desired_yaw)
            heading_score = math.cos(heading_error)

        # 2. Clearance score: min distance to occupied cells along trajectory
        min_obstacle_dist = float("inf")
        for tx, tz, _ in trajectory[1:]:  # skip start
            d = self._check_occupancy(tx, tz, occupancy_grid, self._obstacle_check_radius * 3.0)
            if d < min_obstacle_dist:
                min_obstacle_dist = d

        if min_obstacle_dist < self._obstacle_check_radius:
            # Collision penalty: strongly discourage
            clearance_score = -2.0
        elif min_obstacle_dist < self._obstacle_check_radius * 2.0:
            clearance_score = (min_obstacle_dist - self._obstacle_check_radius) / self._obstacle_check_radius
        else:
            clearance_score = 1.0

        # 3. Progress score: distance reduction toward waypoint
        start_dist = math.sqrt(
            (target_x - start_x) ** 2 + (target_z - start_z) ** 2
        )
        end_dist = math.sqrt(
            (target_x - final_x) ** 2 + (target_z - final_z) ** 2
        )
        if start_dist > 1e-6:
            progress_score = (start_dist - end_dist) / start_dist
        else:
            progress_score = 0.0

        total_score = (
            self._heading_weight * heading_score
            + self._clearance_weight * clearance_score
            + self._progress_weight * progress_score
        )

        return (total_score, abs(heading_error), min_obstacle_dist)

    def _check_occupancy(
        self,
        x: float,
        z: float,
        grid: OccupancyGridData,
        radius: float,
    ) -> float:
        """Check occupancy around a point. Returns distance to nearest obstacle.

        Converts world XZ to grid coordinates, checks cells within radius.
        Returns float('inf') if no obstacles in range.
        """
        res = grid.resolution
        origin_x = grid.origin[0]
        origin_z = grid.origin[1]
        gh, gw = grid.shape

        # Convert world coords to grid cell
        col_center = (x - origin_x) / res
        row_center = (z - origin_z) / res

        # Check radius in cells
        radius_cells = int(math.ceil(radius / res))
        col_min = max(0, int(col_center) - radius_cells)
        col_max = min(gw, int(col_center) + radius_cells + 1)
        row_min = max(0, int(row_center) - radius_cells)
        row_max = min(gh, int(row_center) + radius_cells + 1)

        if col_min >= gw or row_min >= gh or col_max <= 0 or row_max <= 0:
            return float("inf")

        min_dist = float("inf")

        # Extract sub-grid for efficiency
        sub_grid = grid.grid[row_min:row_max, col_min:col_max]
        occupied = sub_grid > 0.5

        if not np.any(occupied):
            return float("inf")

        # Get occupied cell coordinates
        occ_rows, occ_cols = np.where(occupied)
        occ_rows = occ_rows + row_min
        occ_cols = occ_cols + col_min

        # Compute distances in world coordinates
        occ_world_x = origin_x + (occ_cols.astype(np.float64) + 0.5) * res
        occ_world_z = origin_z + (occ_rows.astype(np.float64) + 0.5) * res

        dists = np.sqrt((occ_world_x - x) ** 2 + (occ_world_z - z) ** 2)
        min_dist = float(np.min(dists))

        return min_dist

    def reset(self) -> None:
        """Reset planner state (stateless, nothing to clear)."""
        pass
