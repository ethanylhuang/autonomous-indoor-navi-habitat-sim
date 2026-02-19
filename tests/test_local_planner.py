"""Tests for the DWA-inspired local planner.

Pure numpy tests, no habitat-sim dependency.
"""

import math

import numpy as np
import pytest

from src.perception.occupancy_grid import OccupancyGridData
from src.planning.local_planner import LocalPlanner


def _make_empty_grid(
    agent_x: float = 0.0,
    agent_z: float = 0.0,
    grid_size: float = 10.0,
    resolution: float = 0.05,
) -> OccupancyGridData:
    """Create an occupancy grid with no obstacles (all unknown/free)."""
    dim = int(grid_size / resolution)
    grid = np.full((dim, dim), 0.1, dtype=np.float32)  # free
    origin = np.array([
        agent_x - grid_size / 2.0,
        agent_z - grid_size / 2.0,
    ], dtype=np.float64)
    return OccupancyGridData(
        grid=grid,
        resolution=resolution,
        origin=origin,
        shape=(dim, dim),
        timestamp_step=1,
    )


def _make_blocked_ahead_grid(
    agent_x: float = 0.0,
    agent_z: float = 0.0,
    grid_size: float = 10.0,
    resolution: float = 0.05,
    block_distance: float = 0.3,
) -> OccupancyGridData:
    """Create an occupancy grid with obstacles directly in front of agent.

    Agent faces -Z direction (yaw=0). Block cells at agent_z - block_distance.
    """
    dim = int(grid_size / resolution)
    grid = np.full((dim, dim), 0.1, dtype=np.float32)  # free
    origin_x = agent_x - grid_size / 2.0
    origin_z = agent_z - grid_size / 2.0
    origin = np.array([origin_x, origin_z], dtype=np.float64)

    # Block a strip ahead (agent faces -Z, so obstacles are at lower Z)
    obstacle_z = agent_z - block_distance
    row = int((obstacle_z - origin_z) / resolution)
    col_center = int((agent_x - origin_x) / resolution)

    # Block a 20-cell wide strip
    for dc in range(-10, 11):
        c = col_center + dc
        if 0 <= row < dim and 0 <= c < dim:
            grid[row, c] = 0.9  # occupied

    return OccupancyGridData(
        grid=grid,
        resolution=resolution,
        origin=origin,
        shape=(dim, dim),
        timestamp_step=1,
    )


def _make_fully_blocked_grid(
    agent_x: float = 0.0,
    agent_z: float = 0.0,
    grid_size: float = 10.0,
    resolution: float = 0.05,
) -> OccupancyGridData:
    """Create an occupancy grid with obstacles in all directions."""
    dim = int(grid_size / resolution)
    grid = np.full((dim, dim), 0.9, dtype=np.float32)  # all occupied
    origin = np.array([
        agent_x - grid_size / 2.0,
        agent_z - grid_size / 2.0,
    ], dtype=np.float64)

    # Small free zone at agent position
    center = dim // 2
    grid[center, center] = 0.1

    return OccupancyGridData(
        grid=grid,
        resolution=resolution,
        origin=origin,
        shape=(dim, dim),
        timestamp_step=1,
    )


class TestLocalPlanner:

    def test_straight_path_selects_forward(self):
        """No obstacles, waypoint straight ahead -> move_forward."""
        planner = LocalPlanner()
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        yaw = 0.0  # facing -Z
        waypoint = np.array([0.0, 0.0, -5.0], dtype=np.float64)  # ahead
        grid = _make_empty_grid()

        result = planner.plan(pos, yaw, waypoint, grid, False)
        assert result.best_action == "move_forward"
        assert not result.is_blocked

    def test_obstacle_ahead_selects_turn(self):
        """Occupied cells ahead -> turn action."""
        planner = LocalPlanner()
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        yaw = 0.0  # facing -Z
        waypoint = np.array([0.0, 0.0, -5.0], dtype=np.float64)
        # block_distance=0.5 places the wall outside obstacle_check_radius
        # from the agent's start position, but inside it after a forward step.
        grid = _make_blocked_ahead_grid(block_distance=0.5)

        result = planner.plan(pos, yaw, waypoint, grid, False)
        assert result.best_action in ("turn_left", "turn_right")

    def test_waypoint_left_selects_left_turn(self):
        """Waypoint to the left -> turn_left."""
        planner = LocalPlanner(
            heading_weight=2.0,
            clearance_weight=0.1,
            progress_weight=0.1,
        )
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        yaw = 0.0  # facing -Z
        # Waypoint to the left (negative X in world = left when facing -Z)
        waypoint = np.array([-5.0, 0.0, 0.0], dtype=np.float64)
        grid = _make_empty_grid()

        result = planner.plan(pos, yaw, waypoint, grid, False)
        assert result.best_action == "turn_left"

    def test_waypoint_right_selects_right_turn(self):
        """Waypoint to the right -> turn_right."""
        planner = LocalPlanner(
            heading_weight=2.0,
            clearance_weight=0.1,
            progress_weight=0.1,
        )
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        yaw = 0.0  # facing -Z
        # Waypoint to the right (positive X)
        waypoint = np.array([5.0, 0.0, 0.0], dtype=np.float64)
        grid = _make_empty_grid()

        result = planner.plan(pos, yaw, waypoint, grid, False)
        assert result.best_action == "turn_right"

    def test_all_blocked_returns_is_blocked(self):
        """Surrounded by obstacles -> is_blocked=True."""
        planner = LocalPlanner()
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        yaw = 0.0
        waypoint = np.array([0.0, 0.0, -5.0], dtype=np.float64)
        grid = _make_fully_blocked_grid()

        result = planner.plan(pos, yaw, waypoint, grid, False)
        assert result.is_blocked

    def test_rear_obstacle_sets_warning(self):
        """Rear obstacle flag propagated to result."""
        planner = LocalPlanner()
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        yaw = 0.0
        waypoint = np.array([0.0, 0.0, -5.0], dtype=np.float64)
        grid = _make_empty_grid()

        result = planner.plan(pos, yaw, waypoint, grid, rear_obstacle_detected=True)
        assert result.rear_obstacle_warning

        result2 = planner.plan(pos, yaw, waypoint, grid, rear_obstacle_detected=False)
        assert not result2.rear_obstacle_warning

    def test_clearance_score_varies_with_distance(self):
        """Closer obstacles = lower score."""
        planner = LocalPlanner()
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        yaw = 0.0
        waypoint = np.array([0.0, 0.0, -5.0], dtype=np.float64)

        # Empty grid: high score
        grid_empty = _make_empty_grid()
        result_empty = planner.plan(pos, yaw, waypoint, grid_empty, False)

        # Blocked ahead: lower score
        grid_blocked = _make_blocked_ahead_grid(block_distance=0.3)
        result_blocked = planner.plan(pos, yaw, waypoint, grid_blocked, False)

        # Forward action should score higher in empty grid
        assert result_empty.score >= result_blocked.score

    def test_deterministic_for_same_inputs(self):
        """Same inputs produce same outputs."""
        planner = LocalPlanner()
        pos = np.array([1.0, 0.0, 2.0], dtype=np.float64)
        yaw = 0.5
        waypoint = np.array([3.0, 0.0, -1.0], dtype=np.float64)
        grid = _make_empty_grid(agent_x=1.0, agent_z=2.0)

        r1 = planner.plan(pos, yaw, waypoint, grid, False)
        r2 = planner.plan(pos, yaw, waypoint, grid, False)

        assert r1.best_action == r2.best_action
        assert abs(r1.score - r2.score) < 1e-10

    def test_result_action_is_valid(self):
        """best_action is always a valid action string."""
        planner = LocalPlanner()
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        yaw = 0.0
        waypoint = np.array([1.0, 0.0, -1.0], dtype=np.float64)
        grid = _make_empty_grid()

        result = planner.plan(pos, yaw, waypoint, grid, False)
        assert result.best_action in ("move_forward", "turn_left", "turn_right")
