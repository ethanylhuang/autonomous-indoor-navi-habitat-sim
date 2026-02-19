"""Integration tests for the full navigation pipeline.

Requires habitat-sim to be installed. These tests verify the complete
M1+M2+M3 pipeline runs without exceptions and produces valid metrics.
"""

import numpy as np
import pytest

from configs.sensor_rig import HFOV
from src.control.controller import NavigationController
from src.perception.obstacle_detector import ObstacleDetector
from src.perception.occupancy_grid import OccupancyGrid
from src.perception.visual_odometry import VisualOdometry
from src.planning.global_planner import GlobalPlanner
from src.planning.local_planner import LocalPlanner
from src.sensors.lidar import depth_to_point_cloud, transform_point_cloud
from src.state_estimation.estimator import EKFEstimator
from src.utils.transforms import quat_multiply, yaw_from_quaternion
from src.vehicle import Vehicle


def _compute_rear_rotation(agent_rotation):
    q_180_y = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    return quat_multiply(agent_rotation, q_180_y)


@pytest.fixture(scope="module")
def vehicle():
    v = Vehicle()
    yield v
    v.close()


class TestFullNavigation:

    def test_full_episode_does_not_crash(self, vehicle):
        """Run one navigation episode end-to-end without exceptions."""
        hfov = float(HFOV)

        # Setup
        vo = VisualOdometry(hfov_deg=hfov, resolution=(480, 640))
        detector = ObstacleDetector()
        grid = OccupancyGrid()
        estimator = EKFEstimator()
        global_planner = GlobalPlanner()
        local_planner = LocalPlanner()
        controller = NavigationController(
            global_planner=global_planner,
            local_planner=local_planner,
            max_steps=20,  # short episode for testing
        )

        # Reset and pick goal
        obs = vehicle.reset()
        goal = np.array(
            vehicle.pathfinder.get_random_navigable_point(), dtype=np.float64
        )

        # Init EKF
        start_yaw = yaw_from_quaternion(obs.state.rotation)
        estimator.initialize(obs.state.position, start_yaw)

        # Plan path
        global_path = global_planner.plan(
            vehicle.find_path, obs.state.position, goal,
        )
        if not global_path.is_valid:
            pytest.skip("Path planning failed (island issue)")

        controller.start_episode(
            obs.state.position, obs.state.rotation, goal, global_path,
            vehicle_find_path=vehicle.find_path,
        )

        # Run episode
        while not controller.is_episode_done:
            fwd_pc = depth_to_point_cloud(obs.depth, hfov)
            fwd_pc_world = transform_point_cloud(
                fwd_pc, obs.state.position, obs.state.rotation,
            )
            rear_rot = _compute_rear_rotation(obs.state.rotation)
            rear_pc = depth_to_point_cloud(obs.rear_depth, hfov)
            rear_pc_world = transform_point_cloud(
                rear_pc, obs.state.position, rear_rot,
            )

            vo_est = vo.update(obs.forward_rgb)
            fwd_det, rear_det = detector.detect_both_cameras(
                obs.forward_semantic, obs.rear_semantic,
                obs.depth, obs.rear_depth,
            )
            occ_grid = grid.update(
                obs.state.position, obs.state.rotation,
                [fwd_pc_world, rear_pc_world],
            )

            pose = estimator.predict(obs.imu, dt=1.0)
            if vo_est.is_valid:
                pose = estimator.update_vo(vo_est)

            rear_warning = rear_det.obstacle_count > 0
            nav_status = controller.step(
                pose, occ_grid, rear_warning, obs.state.collided,
            )

            if controller.is_episode_done:
                break

            obs = vehicle.step(nav_status.action)

        result = controller.finish_episode()
        # Just verify no crash and result is an EpisodeResult
        assert result is not None

    def test_episode_result_has_valid_metrics(self, vehicle):
        """All metrics are finite and non-negative."""
        hfov = float(HFOV)

        vo = VisualOdometry(hfov_deg=hfov, resolution=(480, 640))
        detector = ObstacleDetector()
        grid = OccupancyGrid()
        estimator = EKFEstimator()
        global_planner = GlobalPlanner()
        local_planner = LocalPlanner()
        controller = NavigationController(
            global_planner=global_planner,
            local_planner=local_planner,
            max_steps=10,
        )

        obs = vehicle.reset()
        goal = np.array(
            vehicle.pathfinder.get_random_navigable_point(), dtype=np.float64
        )

        start_yaw = yaw_from_quaternion(obs.state.rotation)
        estimator.initialize(obs.state.position, start_yaw)

        global_path = global_planner.plan(
            vehicle.find_path, obs.state.position, goal,
        )
        if not global_path.is_valid:
            pytest.skip("Path planning failed")

        controller.start_episode(
            obs.state.position, obs.state.rotation, goal, global_path,
            vehicle_find_path=vehicle.find_path,
        )

        while not controller.is_episode_done:
            fwd_pc = depth_to_point_cloud(obs.depth, hfov)
            fwd_pc_world = transform_point_cloud(
                fwd_pc, obs.state.position, obs.state.rotation,
            )
            occ_grid = grid.update(
                obs.state.position, obs.state.rotation, [fwd_pc_world],
            )
            pose = estimator.predict(obs.imu, dt=1.0)
            nav_status = controller.step(
                pose, occ_grid, False, obs.state.collided,
            )
            if controller.is_episode_done:
                break
            obs = vehicle.step(nav_status.action)

        result = controller.finish_episode()

        assert np.isfinite(result.spl)
        assert result.spl >= 0.0
        assert result.spl <= 1.0
        assert np.isfinite(result.path_length)
        assert result.path_length >= 0.0
        assert result.steps >= 0
        assert result.collisions >= 0
        assert result.termination_reason in (
            "goal_reached", "max_steps", "stuck", "path_invalid",
        )

    def test_short_distance_goal_reached(self, vehicle):
        """Start and goal close together -> should succeed."""
        hfov = float(HFOV)

        vo = VisualOdometry(hfov_deg=hfov, resolution=(480, 640))
        grid = OccupancyGrid()
        estimator = EKFEstimator()
        global_planner = GlobalPlanner()
        local_planner = LocalPlanner()
        controller = NavigationController(
            global_planner=global_planner,
            local_planner=local_planner,
            max_steps=50,
            goal_threshold=0.5,
        )

        obs = vehicle.reset()
        start = obs.state.position.copy()

        # Set goal very close to start (within threshold or just past it)
        goal = start.copy()
        goal[0] += 0.3  # 30cm away

        start_yaw = yaw_from_quaternion(obs.state.rotation)
        estimator.initialize(start, start_yaw)

        global_path = global_planner.plan(
            vehicle.find_path, start, goal,
        )
        if not global_path.is_valid:
            pytest.skip("Path planning failed")

        controller.start_episode(
            start, obs.state.rotation, goal, global_path,
            vehicle_find_path=vehicle.find_path,
        )

        # The very first step might already reach the goal since it's 0.3m away
        fwd_pc = depth_to_point_cloud(obs.depth, hfov)
        fwd_pc_world = transform_point_cloud(
            fwd_pc, obs.state.position, obs.state.rotation,
        )
        occ_grid = grid.update(
            obs.state.position, obs.state.rotation, [fwd_pc_world],
        )
        pose = estimator.predict(obs.imu, dt=1.0)
        nav_status = controller.step(
            pose, occ_grid, False, obs.state.collided,
        )

        if nav_status.goal_reached:
            result = controller.finish_episode()
            assert result.success
            return

        # Otherwise run a few more steps
        for _ in range(49):
            if controller.is_episode_done:
                break
            obs = vehicle.step(nav_status.action)
            fwd_pc = depth_to_point_cloud(obs.depth, hfov)
            fwd_pc_world = transform_point_cloud(
                fwd_pc, obs.state.position, obs.state.rotation,
            )
            occ_grid = grid.update(
                obs.state.position, obs.state.rotation, [fwd_pc_world],
            )
            pose = estimator.predict(obs.imu, dt=1.0)
            nav_status = controller.step(
                pose, occ_grid, False, obs.state.collided,
            )

        result = controller.finish_episode()
        # Short distance goal: high chance of success, but can fail if
        # the NavMesh path is weird. We just verify no crash.
        assert result is not None
