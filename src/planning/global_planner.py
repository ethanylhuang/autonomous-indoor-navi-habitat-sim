"""NavMesh-based global path planner.

Wraps NavMesh find_path() with waypoint management, replanning, and path
validity checks. Provides a waypoint sequence from start to goal with
advancement and skip logic.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class GlobalPath:
    waypoints: List[NDArray[np.float64]]  # list of [3] world positions
    geodesic_distance: float  # meters
    current_waypoint_idx: int  # index of next target waypoint
    is_valid: bool  # False if pathfinding failed


class GlobalPlanner:
    """NavMesh-based global path planner.

    Manages a waypoint sequence from start to goal. Supports waypoint
    advancement (when agent reaches a waypoint) and full replanning
    (when obstructed or off-path).
    """

    def __init__(
        self,
        waypoint_reach_threshold: float = 0.5,
        max_waypoint_skip: int = 3,
        corner_smooth_radius: float = 0.4,
    ) -> None:
        self._waypoint_reach_threshold = waypoint_reach_threshold
        self._max_waypoint_skip = max_waypoint_skip
        self._corner_smooth_radius = corner_smooth_radius
        self._path: Optional[GlobalPath] = None

    def plan(
        self,
        vehicle_find_path: Callable,
        start: NDArray[np.float64],
        goal: NDArray[np.float64],
    ) -> GlobalPath:
        """Compute global path from start to goal.

        Args:
            vehicle_find_path: Reference to Vehicle.find_path(start, goal).
                Returns (waypoints, geodesic_distance).
            start: [3] current agent world position.
            goal: [3] target world position.

        Returns:
            GlobalPath with waypoints. is_valid=False if pathfinding fails.
        """
        waypoints, geodesic_distance = vehicle_find_path(start, goal)

        if not waypoints or geodesic_distance == float("inf"):
            self._path = GlobalPath(
                waypoints=[],
                geodesic_distance=float("inf"),
                current_waypoint_idx=0,
                is_valid=False,
            )
            return self._path

        wp_list = [np.asarray(w, dtype=np.float64) for w in waypoints]

        # Smooth corner waypoints by pulling them toward the path midline
        wp_list = self._smooth_corners(wp_list)

        self._path = GlobalPath(
            waypoints=wp_list,
            geodesic_distance=geodesic_distance,
            current_waypoint_idx=min(1, len(wp_list) - 1),  # skip start point
            is_valid=True,
        )
        return self._path

    def get_current_waypoint(self) -> Optional[NDArray[np.float64]]:
        """Return the next waypoint to navigate toward, or None if done."""
        if self._path is None or not self._path.is_valid:
            return None
        if self._path.current_waypoint_idx >= len(self._path.waypoints):
            return None
        return self._path.waypoints[self._path.current_waypoint_idx]

    def get_lookahead_waypoint(self) -> Optional[NDArray[np.float64]]:
        """Return the waypoint after the current one, or None if unavailable."""
        if self._path is None or not self._path.is_valid:
            return None
        next_idx = self._path.current_waypoint_idx + 1
        if next_idx >= len(self._path.waypoints):
            return None
        return self._path.waypoints[next_idx]

    def advance_waypoint(
        self,
        agent_position: NDArray[np.float64],
    ) -> bool:
        """Check if agent has reached current waypoint and advance.

        Uses waypoint_reach_threshold. May skip waypoints if agent
        is closer to a later waypoint (handles NavMesh shortcutting).

        Args:
            agent_position: [3] current agent world position.

        Returns:
            True if all waypoints reached (goal arrived via waypoint sequence).
        """
        if self._path is None or not self._path.is_valid:
            return False

        pos = np.asarray(agent_position, dtype=np.float64)
        wp_list = self._path.waypoints
        idx = self._path.current_waypoint_idx

        if idx >= len(wp_list):
            return True

        # Check if we can skip ahead: find the furthest reachable waypoint
        best_idx = idx
        max_check = min(idx + self._max_waypoint_skip + 1, len(wp_list))
        for i in range(idx, max_check):
            dist = _xz_distance(pos, wp_list[i])
            if dist < self._waypoint_reach_threshold:
                best_idx = i + 1  # advance past this waypoint

        if best_idx > idx:
            self._path.current_waypoint_idx = best_idx

        return self._path.current_waypoint_idx >= len(wp_list)

    def replan(
        self,
        vehicle_find_path: Callable,
        current_position: NDArray[np.float64],
        goal: NDArray[np.float64],
    ) -> GlobalPath:
        """Recompute path from current position to goal."""
        return self.plan(vehicle_find_path, current_position, goal)

    def get_remaining_distance(self) -> float:
        """Sum of distances from current waypoint through remaining waypoints."""
        if self._path is None or not self._path.is_valid:
            return float("inf")

        wp_list = self._path.waypoints
        idx = self._path.current_waypoint_idx

        if idx >= len(wp_list):
            return 0.0

        total = 0.0
        for i in range(idx, len(wp_list) - 1):
            total += float(np.linalg.norm(wp_list[i + 1] - wp_list[i]))
        return total

    def is_goal_reached(
        self,
        agent_position: NDArray[np.float64],
        goal: NDArray[np.float64],
        threshold: float = 0.5,
    ) -> bool:
        """Check if agent is within threshold of the goal (XZ distance)."""
        return _xz_distance(agent_position, goal) < threshold

    def get_path(self) -> Optional[GlobalPath]:
        """Return the current global path, or None if not planned."""
        return self._path

    def reset(self) -> None:
        """Clear path state."""
        self._path = None

    def _smooth_corners(
        self,
        waypoints: List[NDArray[np.float64]],
    ) -> List[NDArray[np.float64]]:
        """Smooth sharp corner waypoints by pulling them toward the path center.

        For each interior waypoint, if the turn angle is sharp (>45 deg),
        offset the waypoint inward along the angle bisector. This keeps
        the agent away from tight corners.
        """
        if len(waypoints) < 3:
            return waypoints

        result = [waypoints[0]]

        for i in range(1, len(waypoints) - 1):
            prev = waypoints[i - 1]
            curr = waypoints[i]
            next_wp = waypoints[i + 1]

            # Vectors in XZ plane
            v_in = np.array([curr[0] - prev[0], curr[2] - prev[2]])
            v_out = np.array([next_wp[0] - curr[0], next_wp[2] - curr[2]])

            len_in = np.linalg.norm(v_in)
            len_out = np.linalg.norm(v_out)

            if len_in < 1e-6 or len_out < 1e-6:
                result.append(curr)
                continue

            # Normalize
            v_in_norm = v_in / len_in
            v_out_norm = v_out / len_out

            # Compute turn angle (angle between -v_in and v_out)
            # cos(angle) = dot(-v_in_norm, v_out_norm)
            cos_angle = float(np.dot(-v_in_norm, v_out_norm))
            cos_angle = max(-1.0, min(1.0, cos_angle))

            # If turn is sharp (>45 deg, i.e., cos < ~0.7), smooth it
            if cos_angle < 0.7:
                # Bisector direction: average of incoming and outgoing directions
                # Points "inward" toward the center of the turn
                bisector = v_in_norm + v_out_norm
                bisector_len = np.linalg.norm(bisector)

                if bisector_len > 1e-6:
                    bisector_norm = bisector / bisector_len
                    # Offset amount scales with turn sharpness
                    # Sharper turn (lower cos) = more offset
                    offset_amount = self._corner_smooth_radius * (1.0 - cos_angle)
                    new_x = curr[0] + bisector_norm[0] * offset_amount
                    new_z = curr[2] + bisector_norm[1] * offset_amount
                    smoothed = np.array([new_x, curr[1], new_z], dtype=np.float64)
                    result.append(smoothed)
                else:
                    result.append(curr)
            else:
                result.append(curr)

        result.append(waypoints[-1])
        return result


def _xz_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Compute Euclidean distance in XZ plane (ignoring Y)."""
    dx = a[0] - b[0]
    dz = a[2] - b[2]
    return float(np.sqrt(dx * dx + dz * dz))
