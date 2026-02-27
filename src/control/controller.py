"""Navigation controller orchestrating the full nav loop per step.

Takes the estimated pose, the global path, and local planner output to
produce the next action. Handles goal detection, stuck detection, and
replanning triggers. Also computes episode metrics (SPL, path length, etc.).
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

from src.perception.occupancy_grid import OccupancyGridData
from src.planning.global_planner import GlobalPath, GlobalPlanner
from src.planning.local_planner import LocalPlanner, LocalPlanResult
from src.state_estimation.estimator import PoseEstimate
from src.utils.transforms import normalize_angle


@dataclass
class NavigationStatus:
    action: str  # action to execute this step
    goal_reached: bool  # True if within goal threshold
    is_stuck: bool  # True if stuck detection triggered
    needs_replan: bool  # True if replanning is needed
    distance_to_goal: float  # Euclidean distance to goal
    heading_error: float  # radians to next waypoint
    steps_taken: int
    total_collisions: int
    path_length: float  # cumulative distance traveled


@dataclass
class EpisodeResult:
    success: bool
    spl: float  # Success weighted by Path Length
    path_length: float  # total distance traveled
    geodesic_distance: float  # optimal path length
    steps: int
    collisions: int
    termination_reason: str  # "goal_reached", "max_steps", "stuck", "path_invalid"


class NavigationController:
    """Orchestrates one navigation episode.

    Combines global planner, local planner, and state estimator into
    a step-by-step navigation loop. Handles goal detection, stuck
    detection, and replanning.
    """

    def __init__(
        self,
        global_planner: GlobalPlanner,
        local_planner: LocalPlanner,
        goal_threshold: float = 0.5,
        stuck_window: int = 20,
        stuck_displacement: float = 0.3,
        max_replan_attempts: int = 3,
        max_steps: int = 500,
    ) -> None:
        self._global_planner = global_planner
        self._local_planner = local_planner
        self._goal_threshold = goal_threshold
        self._stuck_window = stuck_window
        self._stuck_displacement = stuck_displacement
        self._max_replan_attempts = max_replan_attempts
        self._max_steps = max_steps

        # Per-episode state
        self._goal: Optional[NDArray[np.float64]] = None
        self._geodesic_distance: float = 0.0
        self._steps_taken: int = 0
        self._total_collisions: int = 0
        self._path_length: float = 0.0
        self._prev_position: Optional[NDArray[np.float64]] = None
        self._position_history: deque = deque(maxlen=stuck_window)
        self._replan_count: int = 0
        self._goal_reached: bool = False
        self._termination_reason: Optional[str] = None
        self._vehicle_find_path: Optional[Callable] = None

        # Escape state: queued actions when stuck
        self._escape_queue: List[str] = []
        self._escape_turn_deg: float = 30.0  # larger turns for escape
        self._escape_attempts: int = 0
        self._max_escape_attempts: int = 4

    def start_episode(
        self,
        start_position: NDArray[np.float64],
        start_rotation: NDArray[np.float64],
        goal_position: NDArray[np.float64],
        global_path: GlobalPath,
        vehicle_find_path: Optional[Callable] = None,
    ) -> None:
        """Initialize controller state for a new episode.

        Args:
            start_position: [3] initial world position.
            start_rotation: [4] initial rotation quaternion.
            goal_position: [3] target goal position.
            global_path: Pre-computed global path from planner.
            vehicle_find_path: Optional callable for replanning.
        """
        self._goal = np.asarray(goal_position, dtype=np.float64)
        self._geodesic_distance = global_path.geodesic_distance
        self._steps_taken = 0
        self._total_collisions = 0
        self._path_length = 0.0
        self._prev_position = np.asarray(start_position, dtype=np.float64).copy()
        self._position_history.clear()
        self._position_history.append(self._prev_position.copy())
        self._replan_count = 0
        self._goal_reached = False
        self._termination_reason = None
        self._vehicle_find_path = vehicle_find_path
        self._escape_queue.clear()
        self._escape_attempts = 0

    def step(
        self,
        pose_estimate: PoseEstimate,
        occupancy_grid: OccupancyGridData,
        rear_obstacle_detected: bool,
        collided: bool,
    ) -> NavigationStatus:
        """Compute one navigation step.

        1. Check goal arrival
        2. Check stuck condition
        3. Advance global waypoint if reached
        4. Run local planner for best action
        5. If blocked, trigger replan
        6. Update tracking (path length, collisions, history)

        Returns:
            NavigationStatus with action and status flags.
        """
        self._steps_taken += 1
        current_pos = pose_estimate.position.copy()

        # Update collision counter
        if collided:
            self._total_collisions += 1

        # Update path length
        if self._prev_position is not None:
            step_dist = float(np.linalg.norm(current_pos - self._prev_position))
            self._path_length += step_dist
        self._prev_position = current_pos.copy()
        self._position_history.append(current_pos.copy())

        # Compute distance to goal
        distance_to_goal = _xz_distance(current_pos, self._goal)

        # 1. Check goal arrival
        if distance_to_goal < self._goal_threshold:
            self._goal_reached = True
            self._termination_reason = "goal_reached"
            return NavigationStatus(
                action="move_forward",  # final action doesn't matter
                goal_reached=True,
                is_stuck=False,
                needs_replan=False,
                distance_to_goal=distance_to_goal,
                heading_error=0.0,
                steps_taken=self._steps_taken,
                total_collisions=self._total_collisions,
                path_length=self._path_length,
            )

        # 2. Check max steps
        if self._steps_taken >= self._max_steps:
            self._termination_reason = "max_steps"
            return NavigationStatus(
                action="move_forward",
                goal_reached=False,
                is_stuck=False,
                needs_replan=False,
                distance_to_goal=distance_to_goal,
                heading_error=0.0,
                steps_taken=self._steps_taken,
                total_collisions=self._total_collisions,
                path_length=self._path_length,
            )

        # 3. Drain escape queue if active (bypasses normal planning)
        if self._escape_queue:
            action = self._escape_queue.pop(0)
            # When queue empties, replan from new position and reset history
            if not self._escape_queue and self._vehicle_find_path is not None:
                self._global_planner.replan(
                    self._vehicle_find_path, current_pos, self._goal,
                )
                self._position_history.clear()
                self._position_history.append(current_pos.copy())
            return NavigationStatus(
                action=action,
                goal_reached=False,
                is_stuck=False,
                needs_replan=not self._escape_queue,
                distance_to_goal=distance_to_goal,
                heading_error=0.0,
                steps_taken=self._steps_taken,
                total_collisions=self._total_collisions,
                path_length=self._path_length,
            )

        # 4. Check stuck condition (skip when close to goal — fine adjustments
        #    near the goal look like "stuck" but are normal approach behavior)
        near_goal = distance_to_goal < self._goal_threshold * 2.0
        is_stuck = not near_goal and self._check_stuck()
        if is_stuck:
            if self._escape_attempts < self._max_escape_attempts:
                # Generate escape: large turns toward goal + forward steps
                escape_actions = self._generate_escape(
                    pose_estimate.yaw, current_pos,
                )
                self._escape_queue = escape_actions
                self._escape_attempts += 1
                self._position_history.clear()
                self._position_history.append(current_pos.copy())
                # Execute first escape action immediately
                action = self._escape_queue.pop(0)
                return NavigationStatus(
                    action=action,
                    goal_reached=False,
                    is_stuck=True,
                    needs_replan=True,
                    distance_to_goal=distance_to_goal,
                    heading_error=0.0,
                    steps_taken=self._steps_taken,
                    total_collisions=self._total_collisions,
                    path_length=self._path_length,
                )
            else:
                self._termination_reason = "stuck"
                return NavigationStatus(
                    action="move_forward",
                    goal_reached=False,
                    is_stuck=True,
                    needs_replan=False,
                    distance_to_goal=distance_to_goal,
                    heading_error=0.0,
                    steps_taken=self._steps_taken,
                    total_collisions=self._total_collisions,
                    path_length=self._path_length,
                )

        # 5. Making progress — reset escape counter
        if not is_stuck and self._escape_attempts > 0:
            step_dist_check = float(np.linalg.norm(
                current_pos - self._position_history[0]
            )) if len(self._position_history) > 1 else 0.0
            if step_dist_check > self._stuck_displacement:
                self._escape_attempts = 0

        # 6. Advance global waypoint
        self._global_planner.advance_waypoint(current_pos)
        waypoint = self._global_planner.get_current_waypoint()

        if waypoint is None:
            waypoint = self._goal

        # 7. Run local planner
        local_result = self._local_planner.plan(
            current_pos,
            pose_estimate.yaw,
            waypoint,
            occupancy_grid,
            rear_obstacle_detected,
        )

        # 8. If blocked, trigger replan
        needs_replan = False
        if local_result.is_blocked:
            needs_replan = True
            if self._replan_count < self._max_replan_attempts and self._vehicle_find_path is not None:
                self._global_planner.replan(
                    self._vehicle_find_path, current_pos, self._goal,
                )
                self._replan_count += 1

        # 9. Format action: encode target_yaw and move_forward
        action = f"turn_to:{local_result.target_yaw:.6f}"
        if local_result.move_forward:
            action += ":move"

        return NavigationStatus(
            action=action,
            goal_reached=False,
            is_stuck=is_stuck,
            needs_replan=needs_replan,
            distance_to_goal=distance_to_goal,
            heading_error=local_result.heading_error,
            steps_taken=self._steps_taken,
            total_collisions=self._total_collisions,
            path_length=self._path_length,
        )

    def finish_episode(self) -> EpisodeResult:
        """Compute final episode metrics."""
        success = self._goal_reached
        geodesic = self._geodesic_distance
        path_len = self._path_length

        if success and geodesic > 0 and path_len > 0:
            spl = float(geodesic / max(path_len, geodesic))
        else:
            spl = 0.0

        reason = self._termination_reason or "unknown"

        return EpisodeResult(
            success=success,
            spl=spl,
            path_length=path_len,
            geodesic_distance=geodesic,
            steps=self._steps_taken,
            collisions=self._total_collisions,
            termination_reason=reason,
        )

    def _generate_escape(
        self,
        current_yaw: float,
        current_pos: NDArray[np.float64],
    ) -> List[str]:
        """Generate escape action sequence using larger turns.

        Computes desired yaw toward goal, then generates turn actions
        using escape_turn_deg (larger than normal 10 deg) to break free
        from furniture. Alternates direction on successive attempts.
        """
        # Target direction: toward the goal
        dx = self._goal[0] - current_pos[0]
        dz = self._goal[2] - current_pos[2]
        goal_yaw = math.atan2(-dx, -dz)

        # On even attempts, head toward goal. On odd attempts, offset by 90 deg
        # to try a different escape direction.
        if self._escape_attempts % 2 == 1:
            goal_yaw = normalize_angle(goal_yaw + math.pi / 2.0)

        delta = normalize_angle(goal_yaw - current_yaw)
        turn_rad = math.radians(self._escape_turn_deg)
        num_turns = max(1, int(round(abs(delta) / turn_rad)))
        num_turns = min(num_turns, 6)  # cap at 180 deg

        if delta >= 0:
            actions: List[str] = ["turn_left"] * num_turns
        else:
            actions = ["turn_right"] * num_turns

        # Add forward steps to physically move out of the stuck spot
        actions.extend(["move_forward"] * 3)
        return actions

    def _check_stuck(self) -> bool:
        """Check if agent has moved less than stuck_displacement
        over the last stuck_window steps."""
        if len(self._position_history) < self._stuck_window:
            return False

        oldest = self._position_history[0]
        newest = self._position_history[-1]
        displacement = float(np.linalg.norm(newest - oldest))
        return displacement < self._stuck_displacement

    def _get_effective_waypoint(
        self,
        current_pos: NDArray[np.float64],
        waypoint: NDArray[np.float64],
        occupancy_grid: OccupancyGridData,
    ) -> NDArray[np.float64]:
        """Get effective waypoint, blending toward next if current is obstructed.

        Checks if the path to the waypoint passes through occupied cells.
        If obstructed, blends the target toward the lookahead waypoint.
        """
        # Check if waypoint area is occupied
        wp_x, wp_z = waypoint[0], waypoint[2]
        obstacle_dist = self._check_waypoint_clearance(wp_x, wp_z, occupancy_grid)

        # If waypoint has good clearance, use it directly
        if obstacle_dist > 0.5:
            return waypoint

        # Waypoint is near obstacle — try to blend toward next waypoint
        lookahead = self._global_planner.get_lookahead_waypoint()
        if lookahead is None:
            # No next waypoint, use the goal
            lookahead = self._goal

        # Blend: move target partway toward lookahead
        # More obstruction = more blend toward lookahead
        blend_factor = min(1.0, (0.5 - obstacle_dist) / 0.5)  # 0 to 1
        blend_factor = max(0.3, blend_factor)  # at least 30% toward lookahead

        blended = waypoint * (1.0 - blend_factor) + lookahead * blend_factor
        return blended

    def _check_waypoint_clearance(
        self,
        x: float,
        z: float,
        grid: OccupancyGridData,
    ) -> float:
        """Check clearance around a waypoint. Returns distance to nearest obstacle."""
        res = grid.resolution
        origin_x = grid.origin[0]
        origin_z = grid.origin[1]
        gh, gw = grid.shape

        # Convert world coords to grid cell
        col_center = (x - origin_x) / res
        row_center = (z - origin_z) / res

        # Check within 1m radius
        radius_cells = int(math.ceil(1.0 / res))
        col_min = max(0, int(col_center) - radius_cells)
        col_max = min(gw, int(col_center) + radius_cells + 1)
        row_min = max(0, int(row_center) - radius_cells)
        row_max = min(gh, int(row_center) + radius_cells + 1)

        if col_min >= gw or row_min >= gh or col_max <= 0 or row_max <= 0:
            return float("inf")

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
        return float(np.min(dists))

    @property
    def is_episode_done(self) -> bool:
        """True if the episode has terminated for any reason."""
        return self._termination_reason is not None

    def reset(self) -> None:
        """Clear all episode state."""
        self._goal = None
        self._geodesic_distance = 0.0
        self._steps_taken = 0
        self._total_collisions = 0
        self._path_length = 0.0
        self._prev_position = None
        self._position_history.clear()
        self._replan_count = 0
        self._goal_reached = False
        self._termination_reason = None
        self._vehicle_find_path = None
        self._escape_queue.clear()
        self._escape_attempts = 0


def _xz_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Compute Euclidean distance in XZ plane (ignoring Y)."""
    dx = a[0] - b[0]
    dz = a[2] - b[2]
    return float(np.sqrt(dx * dx + dz * dz))
