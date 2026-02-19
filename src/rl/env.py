"""Gymnasium environment wrapper for RL navigation training.

Wraps the Vehicle class to provide a standard Gymnasium interface with
Dict observation space (forward_rgb, rear_rgb, depth, goal_vector, imu)
and Discrete(3) action space.
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

import cv2
import gymnasium
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from configs.rl_config import RLConfig
from configs.sim_config import AgentParams, SimParams
from src.utils.transforms import yaw_from_quaternion
from src.vehicle import Observations, Vehicle

logger = logging.getLogger(__name__)

# Action mapping: int -> habitat-sim action string
ACTION_MAP: Dict[int, str] = {
    0: "move_forward",
    1: "turn_left",
    2: "turn_right",
}

# Maximum distance for goal vector normalization (meters)
_MAX_GOAL_DISTANCE: float = 30.0

# Depth clipping range (meters)
_DEPTH_MIN: float = 0.0
_DEPTH_MAX: float = 10.0

# Maximum retries for sampling a valid goal
_MAX_GOAL_RETRIES: int = 100

# Minimum/maximum geodesic distance for a valid episode
_MIN_GEODESIC: float = 1.0
_MAX_GEODESIC: float = 30.0


class NavigationEnv(gymnasium.Env):
    """Gymnasium environment for indoor PointNav using habitat-sim.

    Each instance owns its own Vehicle/Simulator. For vectorized training,
    use SubprocVecEnv so each process gets its own OpenGL context.

    Observation space (Dict):
        forward_rgb: (image_size, image_size, 3) uint8
        rear_rgb:    (image_size, image_size, 3) uint8  (if use_rear_rgb)
        depth:       (image_size, image_size, 1) float32, values in [0, 1]
        goal_vector: (2,) float32 [normalized_distance, normalized_angle]
        imu:         (6,) float32 [linear_vel_x, y, z, angular_vel_x, y, z]

    Action space: Discrete(3) -> {move_forward, turn_left, turn_right}
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: Optional[RLConfig] = None) -> None:
        super().__init__()
        self._config = config if config is not None else RLConfig()
        self._img_size = self._config.image_size

        # Build observation space
        obs_spaces: Dict[str, spaces.Space] = {
            "forward_rgb": spaces.Box(
                low=0, high=255,
                shape=(self._img_size, self._img_size, 3),
                dtype=np.uint8,
            ),
        }
        if self._config.use_rear_rgb:
            obs_spaces["rear_rgb"] = spaces.Box(
                low=0, high=255,
                shape=(self._img_size, self._img_size, 3),
                dtype=np.uint8,
            )
        if self._config.use_depth:
            obs_spaces["depth"] = spaces.Box(
                low=0.0, high=1.0,
                shape=(self._img_size, self._img_size, 1),
                dtype=np.float32,
            )
        obs_spaces["goal_vector"] = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        if self._config.use_imu:
            obs_spaces["imu"] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(6,),
                dtype=np.float32,
            )

        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Discrete(3)

        # Create Vehicle (owns the simulator)
        sim_params = SimParams(
            scene_id=self._config.scene_id,
            random_seed=self._config.seed,
        )
        self._vehicle = Vehicle(sim_params=sim_params)

        # Episode state
        self._goal: Optional[NDArray[np.float32]] = None
        self._geodesic_distance: float = 0.0
        self._prev_distance_to_goal: float = 0.0
        self._step_count: int = 0
        self._total_collisions: int = 0
        self._rng = np.random.RandomState(self._config.seed)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, NDArray], Dict[str, Any]]:
        """Reset the environment for a new episode.

        Randomizes start position and samples a navigable goal with
        geodesic distance in [1.0, 30.0] meters.
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        # Reset vehicle to random navigable position
        vehicle_obs = self._vehicle.reset()

        # Sample a valid goal
        start_pos = vehicle_obs.state.position
        self._goal = self._sample_goal(start_pos)

        # Compute geodesic distance for SPL
        _, self._geodesic_distance = self._vehicle.find_path(
            start_pos, self._goal,
        )

        # Initialize episode state
        self._prev_distance_to_goal = self._xz_distance(
            start_pos, self._goal,
        )
        self._step_count = 0
        self._total_collisions = 0

        obs = self._build_obs(vehicle_obs)
        info = self._build_info(
            success=False,
            distance_to_goal=self._prev_distance_to_goal,
        )

        return obs, info

    def step(
        self, action: int,
    ) -> Tuple[Dict[str, NDArray], float, bool, bool, Dict[str, Any]]:
        """Execute one action and return (obs, reward, terminated, truncated, info)."""
        action_str = ACTION_MAP[int(action)]
        vehicle_obs = self._vehicle.step(action_str)
        self._step_count += 1

        # Current state
        position = vehicle_obs.state.position
        collided = vehicle_obs.state.collided
        if collided:
            self._total_collisions += 1

        curr_distance = self._xz_distance(position, self._goal)
        goal_reached = curr_distance < self._config.goal_threshold

        # Reward
        reward = self._compute_reward(
            self._prev_distance_to_goal,
            curr_distance,
            collided,
            goal_reached,
        )
        self._prev_distance_to_goal = curr_distance

        # Termination
        terminated = goal_reached
        truncated = self._step_count >= self._config.max_episode_steps

        # Build outputs
        obs = self._build_obs(vehicle_obs)
        info = self._build_info(
            success=goal_reached,
            distance_to_goal=curr_distance,
        )

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Release simulator resources."""
        if self._vehicle is not None:
            self._vehicle.close()
            self._vehicle = None

    def render(self) -> NDArray[np.uint8]:
        """Return the forward RGB observation as an image for rendering."""
        vehicle_obs = self._vehicle.get_initial_observations()
        rgb = vehicle_obs.forward_rgb[:, :, :3]  # drop alpha
        return np.ascontiguousarray(rgb)

    # -- Internal helpers --------------------------------------------------

    def _build_obs(self, vehicle_obs: Observations) -> Dict[str, NDArray]:
        """Convert Vehicle observations to Dict observation for the policy."""
        obs: Dict[str, NDArray] = {}

        # Forward RGB: RGBA -> RGB, resize
        fwd_rgb = vehicle_obs.forward_rgb[:, :, :3]
        obs["forward_rgb"] = self._resize_image(fwd_rgb, self._img_size)

        # Rear RGB
        if self._config.use_rear_rgb:
            rear_rgb = vehicle_obs.rear_rgb[:, :, :3]
            obs["rear_rgb"] = self._resize_image(rear_rgb, self._img_size)

        # Depth: resize, clip, normalize to [0, 1], add channel dim
        if self._config.use_depth:
            depth = vehicle_obs.depth.copy()
            depth = np.nan_to_num(depth, nan=0.0, posinf=_DEPTH_MAX)
            depth = cv2.resize(
                depth,
                (self._img_size, self._img_size),
                interpolation=cv2.INTER_NEAREST,
            )
            depth = np.clip(depth, _DEPTH_MIN, _DEPTH_MAX)
            depth = depth / _DEPTH_MAX  # normalize to [0, 1]
            obs["depth"] = depth[:, :, np.newaxis].astype(np.float32)

        # Goal vector
        obs["goal_vector"] = self._compute_goal_vector(
            vehicle_obs.state.position,
            vehicle_obs.state.rotation,
        )

        # IMU: [linear_vel_x, y, z, angular_vel_x, y, z]
        if self._config.use_imu:
            imu = vehicle_obs.imu
            imu_vec = np.concatenate([
                imu.linear_velocity,
                imu.angular_velocity,
            ]).astype(np.float32)
            obs["imu"] = imu_vec

        return obs

    def _compute_goal_vector(
        self,
        position: NDArray,
        rotation: NDArray,
    ) -> NDArray[np.float32]:
        """Compute relative goal in agent body frame as (distance, angle).

        Distance is normalized by _MAX_GOAL_DISTANCE, clipped to [0, 1].
        Angle is normalized from [-pi, pi] to [-1, 1].
        """
        # Vector from agent to goal in world XZ plane
        dx = float(self._goal[0] - position[0])
        dz = float(self._goal[2] - position[2])
        distance = math.sqrt(dx * dx + dz * dz)

        # World angle to goal (habitat: forward is -Z, yaw about Y)
        world_angle = math.atan2(-dx, -dz)

        # Agent yaw from quaternion
        agent_yaw = yaw_from_quaternion(rotation)

        # Relative angle in body frame
        relative_angle = world_angle - agent_yaw
        # Normalize to [-pi, pi]
        relative_angle = math.atan2(
            math.sin(relative_angle), math.cos(relative_angle),
        )

        # Normalize
        norm_distance = min(distance / _MAX_GOAL_DISTANCE, 1.0)
        norm_angle = relative_angle / math.pi  # [-1, 1]

        return np.array([norm_distance, norm_angle], dtype=np.float32)

    def _compute_reward(
        self,
        prev_dist: float,
        curr_dist: float,
        collided: bool,
        goal_reached: bool,
    ) -> float:
        """Compute step reward: dense progress + sparse bonuses/penalties."""
        reward = 0.0

        # Dense progress reward
        reward += (prev_dist - curr_dist) * self._config.progress_reward_scale

        # Slack penalty (per step)
        reward += self._config.slack_penalty

        # Collision penalty
        if collided:
            reward += self._config.collision_penalty

        # Success bonus
        if goal_reached:
            reward += self._config.success_reward

        return reward

    def _resize_image(self, img: NDArray, size: int) -> NDArray[np.uint8]:
        """Resize an image to (size, size) using bilinear interpolation."""
        resized = cv2.resize(
            img, (size, size), interpolation=cv2.INTER_LINEAR,
        )
        return np.ascontiguousarray(resized, dtype=np.uint8)

    def _sample_goal(self, start_pos: NDArray) -> NDArray[np.float32]:
        """Sample a navigable goal with geodesic distance in [1.0, 30.0]m.

        Retries up to _MAX_GOAL_RETRIES times. Falls back to best candidate
        if no valid goal found.
        """
        best_goal: Optional[NDArray] = None
        best_geodesic: float = float("inf")

        for _ in range(_MAX_GOAL_RETRIES):
            goal = np.array(
                self._vehicle.pathfinder.get_random_navigable_point(),
                dtype=np.float32,
            )
            _, geodesic = self._vehicle.find_path(start_pos, goal)

            if _MIN_GEODESIC <= geodesic <= _MAX_GEODESIC:
                return goal

            # Track best candidate (closest to valid range)
            if geodesic < float("inf"):
                dist_to_valid = min(
                    abs(geodesic - _MIN_GEODESIC),
                    abs(geodesic - _MAX_GEODESIC),
                )
                if dist_to_valid < abs(best_geodesic - (_MIN_GEODESIC + _MAX_GEODESIC) / 2.0):
                    best_goal = goal
                    best_geodesic = geodesic

        if best_goal is not None:
            logger.warning(
                "Could not find goal in [%.1f, %.1f]m range after %d tries. "
                "Using best candidate with geodesic=%.2fm.",
                _MIN_GEODESIC, _MAX_GEODESIC, _MAX_GOAL_RETRIES, best_geodesic,
            )
            return best_goal

        # Absolute fallback: use any navigable point
        logger.warning("Falling back to arbitrary navigable goal.")
        return np.array(
            self._vehicle.pathfinder.get_random_navigable_point(),
            dtype=np.float32,
        )

    def _build_info(
        self,
        success: bool,
        distance_to_goal: float,
    ) -> Dict[str, Any]:
        """Build the info dict returned by step()/reset()."""
        # SPL: only meaningful when episode ends with success
        if success and self._geodesic_distance > 0:
            path_length = self._step_count * 0.25  # approximate from step count
            spl = float(
                self._geodesic_distance
                / max(path_length, self._geodesic_distance)
            )
        else:
            spl = 0.0

        return {
            "success": success,
            "spl": spl,
            "distance_to_goal": distance_to_goal,
            "collisions": self._total_collisions,
            "geodesic_distance": self._geodesic_distance,
        }

    @staticmethod
    def _xz_distance(a: NDArray, b: NDArray) -> float:
        """Euclidean distance in XZ plane (ignoring Y)."""
        dx = float(a[0] - b[0])
        dz = float(a[2] - b[2])
        return math.sqrt(dx * dx + dz * dz)
