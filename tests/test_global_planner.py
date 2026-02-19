"""Tests for the NavMesh-based global planner.

Pure numpy tests with mock find_path, no habitat-sim dependency.
"""

from typing import List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from src.planning.global_planner import GlobalPlanner, GlobalPath


def _mock_find_path_valid(
    start: NDArray, goal: NDArray,
) -> Tuple[List[NDArray], float]:
    """Mock find_path that returns a straight-line path with waypoints."""
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    mid = (start + goal) / 2.0
    dist = float(np.linalg.norm(goal - start))
    return [start.copy(), mid, goal.copy()], dist


def _mock_find_path_empty(
    start: NDArray, goal: NDArray,
) -> Tuple[List[NDArray], float]:
    """Mock find_path that returns no path (different islands)."""
    return [], float("inf")


class TestGlobalPlanner:

    def test_plan_returns_valid_path(self):
        planner = GlobalPlanner()
        start = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        goal = np.array([4.0, 0.0, 0.0], dtype=np.float64)
        path = planner.plan(_mock_find_path_valid, start, goal)

        assert path.is_valid
        assert len(path.waypoints) == 3
        assert path.geodesic_distance == pytest.approx(4.0, abs=0.01)
        # current_waypoint_idx should skip start (idx=1)
        assert path.current_waypoint_idx == 1

    def test_plan_handles_empty_path(self):
        planner = GlobalPlanner()
        start = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        goal = np.array([4.0, 0.0, 0.0], dtype=np.float64)
        path = planner.plan(_mock_find_path_empty, start, goal)

        assert not path.is_valid
        assert len(path.waypoints) == 0

    def test_advance_waypoint_at_threshold(self):
        planner = GlobalPlanner(waypoint_reach_threshold=0.5)
        start = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        goal = np.array([4.0, 0.0, 0.0], dtype=np.float64)
        planner.plan(_mock_find_path_valid, start, goal)

        # Agent near the midpoint (waypoint index 1)
        agent_pos = np.array([2.0, 0.0, 0.1], dtype=np.float64)
        done = planner.advance_waypoint(agent_pos)
        assert not done  # should advance to waypoint 2 (goal) but not be done yet

        # Now at the goal waypoint
        agent_pos = np.array([4.0, 0.0, 0.0], dtype=np.float64)
        done = planner.advance_waypoint(agent_pos)
        assert done  # all waypoints consumed

    def test_advance_waypoint_skips_passed(self):
        planner = GlobalPlanner(waypoint_reach_threshold=0.5, max_waypoint_skip=3)
        # Create a path with multiple waypoints
        waypoints = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([3.0, 0.0, 0.0]),
            np.array([4.0, 0.0, 0.0]),
        ]

        def mock_find(s, g):
            return waypoints, 4.0

        planner.plan(mock_find, waypoints[0], waypoints[-1])

        # Agent near waypoint 2 (index 2) -- should skip past 1 and 2
        agent_pos = np.array([2.1, 0.0, 0.0], dtype=np.float64)
        done = planner.advance_waypoint(agent_pos)
        assert not done
        # Current waypoint should be past index 2
        wp = planner.get_current_waypoint()
        assert wp is not None
        assert wp[0] >= 3.0

    def test_goal_reached_detection(self):
        planner = GlobalPlanner()
        agent_pos = np.array([3.8, 0.0, 0.0], dtype=np.float64)
        goal = np.array([4.0, 0.0, 0.0], dtype=np.float64)
        assert planner.is_goal_reached(agent_pos, goal, threshold=0.5)
        assert not planner.is_goal_reached(agent_pos, goal, threshold=0.1)

    def test_replan_produces_new_path(self):
        planner = GlobalPlanner()
        start = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        goal = np.array([4.0, 0.0, 0.0], dtype=np.float64)
        planner.plan(_mock_find_path_valid, start, goal)

        # Replan from a different position
        new_start = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        new_path = planner.replan(_mock_find_path_valid, new_start, goal)
        assert new_path.is_valid
        # First waypoint should be the new start
        np.testing.assert_allclose(new_path.waypoints[0], new_start, atol=1e-10)

    def test_remaining_distance_calculation(self):
        planner = GlobalPlanner()
        waypoints = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 1.0]),
            np.array([2.0, 0.0, 1.0]),
        ]

        def mock_find(s, g):
            return waypoints, 3.0

        planner.plan(mock_find, waypoints[0], waypoints[-1])
        # current_waypoint_idx = 1 (skipped start)
        remaining = planner.get_remaining_distance()
        # Distance from wp[1] to wp[2] = 1.0, wp[2] to wp[3] = 1.0
        assert abs(remaining - 2.0) < 0.01

    def test_get_current_waypoint_none_when_done(self):
        planner = GlobalPlanner(waypoint_reach_threshold=1.0)
        waypoints = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0]),
        ]

        def mock_find(s, g):
            return waypoints, 0.5

        planner.plan(mock_find, waypoints[0], waypoints[-1])

        # Advance past all waypoints
        planner.advance_waypoint(np.array([0.5, 0.0, 0.0]))
        assert planner.get_current_waypoint() is None

    def test_reset_clears_path(self):
        planner = GlobalPlanner()
        planner.plan(
            _mock_find_path_valid,
            np.array([0.0, 0.0, 0.0]),
            np.array([4.0, 0.0, 0.0]),
        )
        planner.reset()
        assert planner.get_current_waypoint() is None
        assert planner.get_path() is None
