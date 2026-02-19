"""Tests for the navigation controller.

Pure numpy tests with mock planner/estimator, no habitat-sim dependency.
"""

import numpy as np
import pytest

from src.control.controller import (
    EpisodeResult,
    NavigationController,
    NavigationStatus,
)
from src.perception.occupancy_grid import OccupancyGridData
from src.planning.global_planner import GlobalPath, GlobalPlanner
from src.planning.local_planner import LocalPlanner
from src.state_estimation.estimator import PoseEstimate


def _make_pose(x: float = 0.0, z: float = 0.0, yaw: float = 0.0) -> PoseEstimate:
    return PoseEstimate(
        position=np.array([x, 0.0, z], dtype=np.float64),
        yaw=yaw,
        covariance=np.eye(3, dtype=np.float64) * 0.01,
        timestamp_step=0,
    )


def _make_grid() -> OccupancyGridData:
    dim = 200
    return OccupancyGridData(
        grid=np.full((dim, dim), 0.1, dtype=np.float32),
        resolution=0.05,
        origin=np.array([-5.0, -5.0], dtype=np.float64),
        shape=(dim, dim),
        timestamp_step=1,
    )


def _make_global_path(
    start=None,
    goal=None,
    geodesic: float = 5.0,
) -> GlobalPath:
    if start is None:
        start = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    if goal is None:
        goal = np.array([5.0, 0.0, 0.0], dtype=np.float64)
    mid = (start + goal) / 2.0
    return GlobalPath(
        waypoints=[start.copy(), mid, goal.copy()],
        geodesic_distance=geodesic,
        current_waypoint_idx=1,
        is_valid=True,
    )


class TestNavigationController:

    def _make_controller(self, max_steps=500, stuck_window=20) -> NavigationController:
        gp = GlobalPlanner()
        lp = LocalPlanner()
        return NavigationController(
            global_planner=gp,
            local_planner=lp,
            max_steps=max_steps,
            stuck_window=stuck_window,
            stuck_displacement=0.3,
        )

    def test_goal_reached_terminates(self):
        ctrl = self._make_controller()
        goal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        path = _make_global_path(goal=goal, geodesic=1.0)
        ctrl.start_episode(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
            goal, path,
        )

        # Agent at goal position
        pose = _make_pose(x=1.0, z=0.0)
        grid = _make_grid()
        status = ctrl.step(pose, grid, False, False)

        assert status.goal_reached
        assert ctrl.is_episode_done

    def test_stuck_detection_triggers(self):
        ctrl = self._make_controller(stuck_window=5, max_steps=100)
        goal = np.array([10.0, 0.0, 0.0], dtype=np.float64)
        path = _make_global_path(goal=goal, geodesic=10.0)
        ctrl.start_episode(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
            goal, path,
        )

        grid = _make_grid()

        # Take many steps at the same position (stuck)
        for _ in range(10):
            pose = _make_pose(x=0.0, z=0.0)
            status = ctrl.step(pose, grid, False, False)

        # Should be stuck after enough stationary steps
        assert status.is_stuck or ctrl.is_episode_done

    def test_collision_counter_increments(self):
        ctrl = self._make_controller()
        goal = np.array([10.0, 0.0, 0.0], dtype=np.float64)
        path = _make_global_path(goal=goal, geodesic=10.0)
        ctrl.start_episode(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
            goal, path,
        )

        grid = _make_grid()
        pose = _make_pose(x=1.0, z=0.0)
        status = ctrl.step(pose, grid, False, collided=True)
        assert status.total_collisions == 1

        pose = _make_pose(x=2.0, z=0.0)
        status = ctrl.step(pose, grid, False, collided=True)
        assert status.total_collisions == 2

    def test_path_length_accumulates(self):
        ctrl = self._make_controller()
        goal = np.array([10.0, 0.0, 0.0], dtype=np.float64)
        path = _make_global_path(goal=goal, geodesic=10.0)
        ctrl.start_episode(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
            goal, path,
        )

        grid = _make_grid()

        pose1 = _make_pose(x=1.0, z=0.0)
        status1 = ctrl.step(pose1, grid, False, False)

        pose2 = _make_pose(x=2.0, z=0.0)
        status2 = ctrl.step(pose2, grid, False, False)

        assert status2.path_length > status1.path_length
        assert status2.path_length > 0.0

    def test_episode_result_spl_calculation(self):
        ctrl = self._make_controller()
        goal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        path = _make_global_path(
            start=np.array([0.0, 0.0, 0.0]),
            goal=goal,
            geodesic=1.0,
        )
        ctrl.start_episode(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
            goal, path,
        )

        grid = _make_grid()
        # Step to the goal
        pose = _make_pose(x=1.0, z=0.0)
        ctrl.step(pose, grid, False, False)

        result = ctrl.finish_episode()
        assert result.success
        assert result.spl > 0.0
        assert result.spl <= 1.0

    def test_episode_result_failure_spl_zero(self):
        ctrl = self._make_controller(max_steps=2)
        goal = np.array([100.0, 0.0, 0.0], dtype=np.float64)
        path = _make_global_path(goal=goal, geodesic=100.0)
        ctrl.start_episode(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
            goal, path,
        )

        grid = _make_grid()

        # Take 2 steps (not reaching goal)
        ctrl.step(_make_pose(x=1.0), grid, False, False)
        ctrl.step(_make_pose(x=2.0), grid, False, False)

        # Episode should be terminated by max_steps
        result = ctrl.finish_episode()
        assert not result.success
        assert result.spl == 0.0

    def test_replan_on_blocked(self):
        """When local planner returns is_blocked, needs_replan should be True."""
        ctrl = self._make_controller()
        goal = np.array([10.0, 0.0, 0.0], dtype=np.float64)
        path = _make_global_path(goal=goal, geodesic=10.0)
        ctrl.start_episode(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
            goal, path,
        )

        # Use a fully blocked grid
        dim = 200
        blocked_grid = OccupancyGridData(
            grid=np.full((dim, dim), 0.9, dtype=np.float32),
            resolution=0.05,
            origin=np.array([-5.0, -5.0], dtype=np.float64),
            shape=(dim, dim),
            timestamp_step=1,
        )

        pose = _make_pose(x=1.0, z=0.0)
        status = ctrl.step(pose, blocked_grid, False, False)

        # The local planner should report blocked, triggering replan need
        assert status.needs_replan

    def test_max_steps_terminates(self):
        ctrl = self._make_controller(max_steps=3)
        goal = np.array([100.0, 0.0, 0.0], dtype=np.float64)
        path = _make_global_path(goal=goal, geodesic=100.0)
        ctrl.start_episode(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
            goal, path,
        )

        grid = _make_grid()

        for i in range(3):
            pose = _make_pose(x=float(i + 1))
            ctrl.step(pose, grid, False, False)

        assert ctrl.is_episode_done
        result = ctrl.finish_episode()
        assert result.termination_reason == "max_steps"

    def test_reset_clears_state(self):
        ctrl = self._make_controller()
        goal = np.array([5.0, 0.0, 0.0], dtype=np.float64)
        path = _make_global_path(goal=goal)
        ctrl.start_episode(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
            goal, path,
        )
        ctrl.step(_make_pose(x=1.0), _make_grid(), False, False)
        ctrl.reset()
        assert not ctrl.is_episode_done
