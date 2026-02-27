"""Run classical navigation pipeline over multiple episodes.

Usage:
    python -m scripts.run_classical --episodes 10 --scene skokloster-castle
    python -m scripts.run_classical --episodes 50 --max-steps 500
"""

import argparse
import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from configs.sensor_rig import HFOV
from configs.sim_config import SimParams
from src.control.controller import EpisodeResult, NavigationController
from src.perception.obstacle_detector import ObstacleDetector
from src.perception.occupancy_grid import OccupancyGrid
from src.perception.visual_odometry import VisualOdometry
from src.planning.global_planner import GlobalPlanner
from src.planning.local_planner import LocalPlanner
from src.sensors.lidar import depth_to_point_cloud, transform_point_cloud
from src.state_estimation.estimator import EKFEstimator
from src.utils.transforms import quat_multiply, yaw_from_quaternion
from src.vehicle import Vehicle

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _compute_rear_rotation(agent_rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute rear sensor world rotation: agent rotation * 180-degree Y offset."""
    q_180_y = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    return quat_multiply(agent_rotation, q_180_y)


def run_episode(
    vehicle: Vehicle,
    vo: VisualOdometry,
    detector: ObstacleDetector,
    grid: OccupancyGrid,
    estimator: EKFEstimator,
    global_planner: GlobalPlanner,
    local_planner: LocalPlanner,
    controller: NavigationController,
    goal: NDArray[np.float64],
) -> Optional[EpisodeResult]:
    """Execute one navigation episode from current vehicle position to goal.

    Returns:
        EpisodeResult, or None if path planning failed (episode skipped).
    """
    hfov = float(HFOV)

    # 1. Reset perception + estimation modules
    vo.reset()
    grid.reset()
    estimator.reset()
    global_planner.reset()
    local_planner.reset()
    controller.reset()

    # 2. Get initial observations, initialize EKF
    obs = vehicle.get_initial_observations()
    start_pos = obs.state.position.copy()
    start_rot = obs.state.rotation.copy()
    start_yaw = yaw_from_quaternion(start_rot)
    estimator.initialize(start_pos, start_yaw)

    # 3. Plan global path
    global_path = global_planner.plan(vehicle.find_path, start_pos, goal)
    if not global_path.is_valid:
        logger.warning("Path planning failed (islands?). Skipping episode.")
        return None

    # 4. Initialize controller
    controller.start_episode(
        start_pos, start_rot, goal, global_path,
        vehicle_find_path=vehicle.find_path,
    )

    # 5. Main navigation loop
    while not controller.is_episode_done:
        # -- M2 Perception Pipeline --
        # LiDAR: depth -> point clouds -> world frame
        fwd_pc = depth_to_point_cloud(obs.depth, hfov)
        fwd_pc_world = transform_point_cloud(fwd_pc, obs.state.position, obs.state.rotation)
        rear_rot = _compute_rear_rotation(obs.state.rotation)
        rear_pc = depth_to_point_cloud(obs.rear_depth, hfov)
        rear_pc_world = transform_point_cloud(rear_pc, obs.state.position, rear_rot)

        # VO
        vo_est = vo.update(obs.forward_rgb)

        # Obstacle detection
        fwd_det, rear_det = detector.detect_both_cameras(
            obs.forward_semantic, obs.rear_semantic,
            obs.depth, obs.rear_depth,
        )

        # Occupancy grid
        occ_grid = grid.update(
            obs.state.position, obs.state.rotation,
            [fwd_pc_world, rear_pc_world],
            obstacle_detections=[
                (fwd_det, obs.depth, obs.state.rotation, hfov),
                (rear_det, obs.rear_depth, rear_rot, hfov),
            ],
        )

        # -- M3 State Estimation --
        pose = estimator.predict(obs.imu, dt=1.0)
        if vo_est.is_valid:
            pose = estimator.update_vo(vo_est)

        # -- M3 Planning + Control --
        rear_warning = (
            rear_det.obstacle_count > 0
            and float(np.min(obs.rear_depth[obs.rear_depth > 0])) < 0.5
            if np.any(obs.rear_depth > 0)
            else False
        )

        nav_status = controller.step(
            pose, occ_grid, rear_warning, obs.state.collided,
        )

        # Check if episode ended
        if controller.is_episode_done:
            break

        # -- Action Execution --
        # Parse continuous action: "turn_to:{yaw}" or "turn_to:{yaw}:move"
        action = nav_status.action
        parts = action.split(":")
        target_yaw = float(parts[1])
        move_forward = len(parts) > 2 and parts[2] == "move"
        obs = vehicle.step_with_heading(target_yaw, move_forward)

    return controller.finish_episode()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classical navigation pipeline.")
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to run."
    )
    parser.add_argument(
        "--max-steps", type=int, default=500, help="Max steps per episode."
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="skokloster-castle",
        help="Scene name (e.g., skokloster-castle).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    args = parser.parse_args()

    # Build scene path
    scene_path = f"data/scene_datasets/habitat-test-scenes/{args.scene}.glb"

    logger.info("Initializing vehicle with scene: %s", scene_path)
    sim_params = SimParams(scene_id=scene_path, random_seed=args.seed)
    vehicle = Vehicle(sim_params=sim_params)

    # Initialize M2 perception modules
    vo = VisualOdometry(hfov_deg=float(HFOV), resolution=(480, 640))
    detector = ObstacleDetector()
    grid = OccupancyGrid()

    # Initialize M3 modules
    estimator = EKFEstimator()
    global_planner = GlobalPlanner()
    local_planner = LocalPlanner()
    controller = NavigationController(
        global_planner=global_planner,
        local_planner=local_planner,
        max_steps=args.max_steps,
    )

    rng = np.random.RandomState(args.seed)
    results = []
    skipped = 0

    logger.info("Running %d episodes...", args.episodes)

    for ep in range(args.episodes):
        # Reset vehicle to random position
        vehicle.reset()

        # Pick a random navigable goal
        goal = np.array(
            vehicle.pathfinder.get_random_navigable_point(), dtype=np.float64
        )

        logger.info(
            "Episode %d/%d: goal=[%.2f, %.2f, %.2f]",
            ep + 1, args.episodes, goal[0], goal[1], goal[2],
        )

        result = run_episode(
            vehicle, vo, detector, grid, estimator,
            global_planner, local_planner, controller, goal,
        )

        if result is None:
            skipped += 1
            logger.info("  Skipped (invalid path)")
            continue

        results.append(result)
        logger.info(
            "  %s | SPL=%.3f | steps=%d | path=%.2fm | geo=%.2fm | collisions=%d",
            result.termination_reason,
            result.spl,
            result.steps,
            result.path_length,
            result.geodesic_distance,
            result.collisions,
        )

    # -- Aggregate Summary --
    print("\n" + "=" * 60)
    print("CLASSICAL NAVIGATION SUMMARY")
    print("=" * 60)

    if not results:
        print("No completed episodes.")
        vehicle.close()
        return

    n = len(results)
    successes = sum(1 for r in results if r.success)
    success_rate = successes / n
    mean_spl = sum(r.spl for r in results) / n
    mean_steps = sum(r.steps for r in results) / n
    mean_path = sum(r.path_length for r in results) / n
    total_collisions = sum(r.collisions for r in results)

    # Path efficiency: geodesic / path_length (only for successful episodes)
    successful = [r for r in results if r.success and r.path_length > 0]
    if successful:
        mean_efficiency = sum(
            r.geodesic_distance / r.path_length for r in successful
        ) / len(successful)
    else:
        mean_efficiency = 0.0

    print(f"Episodes:          {n} ({skipped} skipped)")
    print(f"Success rate:      {success_rate:.1%} ({successes}/{n})")
    print(f"Mean SPL:          {mean_spl:.3f}")
    print(f"Mean steps:        {mean_steps:.1f}")
    print(f"Mean path length:  {mean_path:.2f}m")
    print(f"Total collisions:  {total_collisions}")
    print(f"Path efficiency:   {mean_efficiency:.3f}")

    # Per-termination-reason breakdown
    reasons = {}
    for r in results:
        reasons[r.termination_reason] = reasons.get(r.termination_reason, 0) + 1
    print("\nTermination reasons:")
    for reason, count in sorted(reasons.items()):
        print(f"  {reason}: {count}")

    print("=" * 60)

    vehicle.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
