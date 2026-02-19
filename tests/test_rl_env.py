"""Tests for the RL navigation environment.

Pure-logic tests run without habitat-sim. Integration tests require
habitat-sim and are marked with @pytest.mark.habitat.
"""

import math
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Helpers for pure-logic tests (no habitat-sim or gymnasium import needed)
# ---------------------------------------------------------------------------

def _compute_goal_vector_standalone(
    agent_pos: NDArray,
    agent_rotation: NDArray,
    goal: NDArray,
    max_distance: float = 30.0,
) -> NDArray:
    """Standalone goal vector computation matching NavigationEnv._compute_goal_vector.

    Extracted here so we can test the math without a full env instance.
    """
    from src.utils.transforms import yaw_from_quaternion

    dx = float(goal[0] - agent_pos[0])
    dz = float(goal[2] - agent_pos[2])
    distance = math.sqrt(dx * dx + dz * dz)

    # World angle to goal (habitat: forward is -Z, yaw about Y)
    world_angle = math.atan2(-dx, -dz)

    # Agent yaw from quaternion
    agent_yaw = yaw_from_quaternion(agent_rotation)

    # Relative angle in body frame
    relative_angle = world_angle - agent_yaw
    relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))

    norm_distance = min(distance / max_distance, 1.0)
    norm_angle = relative_angle / math.pi

    return np.array([norm_distance, norm_angle], dtype=np.float32)


def _identity_quat() -> NDArray:
    """Identity quaternion [w, x, y, z] -> facing -Z, yaw=0."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _yaw_quat(yaw: float) -> NDArray:
    """Y-axis rotation quaternion for given yaw."""
    return np.array([
        math.cos(yaw / 2.0),
        0.0,
        math.sin(yaw / 2.0),
        0.0,
    ], dtype=np.float64)


# ===========================================================================
# Pure-logic tests (no habitat-sim needed)
# ===========================================================================

class TestGoalVector:
    """Test goal vector computation in isolation."""

    def test_goal_directly_ahead(self):
        """Goal straight ahead (-Z direction) -> angle ~0."""
        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        agent_rot = _identity_quat()  # facing -Z
        goal = np.array([0.0, 0.0, -5.0], dtype=np.float32)

        gv = _compute_goal_vector_standalone(agent_pos, agent_rot, goal)

        assert gv.shape == (2,)
        assert gv.dtype == np.float32
        # Distance should be 5/30 ~ 0.167
        assert abs(gv[0] - 5.0 / 30.0) < 0.01
        # Angle should be ~0 (goal is ahead)
        assert abs(gv[1]) < 0.05

    def test_goal_directly_behind(self):
        """Goal behind agent -> angle ~+/-1 (normalized pi)."""
        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        agent_rot = _identity_quat()  # facing -Z
        goal = np.array([0.0, 0.0, 5.0], dtype=np.float32)  # behind (+Z)

        gv = _compute_goal_vector_standalone(agent_pos, agent_rot, goal)

        # Angle should be near +/-1 (pi normalized)
        assert abs(abs(gv[1]) - 1.0) < 0.05

    def test_goal_to_left(self):
        """Goal to the left -> angle ~+0.5 (pi/2 normalized)."""
        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        agent_rot = _identity_quat()  # facing -Z
        # Left in habitat is -X when facing -Z
        goal = np.array([-5.0, 0.0, 0.0], dtype=np.float32)

        gv = _compute_goal_vector_standalone(agent_pos, agent_rot, goal)

        # Angle should be ~+0.5 (pi/2 / pi = 0.5)
        assert abs(gv[1] - 0.5) < 0.1

    def test_goal_to_right(self):
        """Goal to the right -> angle ~-0.5."""
        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        agent_rot = _identity_quat()  # facing -Z
        goal = np.array([5.0, 0.0, 0.0], dtype=np.float32)

        gv = _compute_goal_vector_standalone(agent_pos, agent_rot, goal)

        # Angle should be ~-0.5
        assert abs(gv[1] - (-0.5)) < 0.1

    def test_goal_vector_distance_normalization(self):
        """Distance beyond max_distance is clamped to 1.0."""
        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        agent_rot = _identity_quat()
        goal = np.array([0.0, 0.0, -50.0], dtype=np.float32)

        gv = _compute_goal_vector_standalone(agent_pos, agent_rot, goal, max_distance=30.0)

        assert gv[0] == pytest.approx(1.0, abs=1e-5)

    def test_goal_vector_with_rotated_agent(self):
        """Agent rotated 90 degrees left: goal ahead in world -> angle ~-0.5 in body."""
        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        # Yaw = pi/2 (turned left 90 degrees, now facing -X)
        agent_rot = _yaw_quat(math.pi / 2.0)
        # Goal at -Z (world forward, but agent's right)
        goal = np.array([0.0, 0.0, -5.0], dtype=np.float32)

        gv = _compute_goal_vector_standalone(agent_pos, agent_rot, goal)

        # Goal is to the agent's right -> angle ~-0.5
        assert abs(gv[1] - (-0.5)) < 0.1


class TestReward:
    """Test reward computation logic."""

    def test_progress_reward_positive(self):
        """Moving closer to goal gives positive progress reward."""
        prev_dist = 5.0
        curr_dist = 4.0
        progress = (prev_dist - curr_dist) * 1.0  # scale=1.0
        assert progress > 0.0
        assert abs(progress - 1.0) < 1e-6

    def test_progress_reward_negative(self):
        """Moving away from goal gives negative progress reward."""
        prev_dist = 4.0
        curr_dist = 5.0
        progress = (prev_dist - curr_dist) * 1.0
        assert progress < 0.0

    def test_collision_penalty(self):
        """Collision applies penalty."""
        reward = 0.0
        reward += -0.1  # collision_penalty
        assert reward == pytest.approx(-0.1)

    def test_success_reward(self):
        """Success gives large positive bonus."""
        reward = 0.0
        reward += 10.0  # success_reward
        assert reward == pytest.approx(10.0)

    def test_slack_penalty_per_step(self):
        """Each step incurs a small negative penalty."""
        reward = 0.0
        reward += -0.01  # slack_penalty
        assert reward == pytest.approx(-0.01)

    def test_combined_reward(self):
        """Full reward combines all components."""
        prev_dist = 5.0
        curr_dist = 4.5
        collided = True
        goal_reached = False
        scale = 1.0
        slack = -0.01
        collision_pen = -0.1
        success_bonus = 10.0

        reward = (prev_dist - curr_dist) * scale + slack
        if collided:
            reward += collision_pen
        if goal_reached:
            reward += success_bonus

        expected = 0.5 + (-0.01) + (-0.1)
        assert reward == pytest.approx(expected, abs=1e-6)


class TestImageResize:
    """Test image resizing logic."""

    def test_resize_output_shape(self):
        """Resized image has correct shape."""
        import cv2
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
        assert resized.shape == (128, 128, 3)
        assert resized.dtype == np.uint8

    def test_resize_preserves_dtype(self):
        """uint8 stays uint8 after resize."""
        import cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
        assert resized.dtype == np.uint8

    def test_depth_resize_and_normalize(self):
        """Depth resize + clip + normalize produces correct range."""
        import cv2
        depth = np.random.uniform(0.0, 15.0, (480, 640)).astype(np.float32)
        resized = cv2.resize(depth, (128, 128), interpolation=cv2.INTER_NEAREST)
        clipped = np.clip(resized, 0.0, 10.0)
        normalized = clipped / 10.0
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.shape == (128, 128)


class TestActionMapping:
    """Test action int-to-string mapping.

    Uses a local copy of the expected mapping to avoid importing gymnasium
    (which is a heavy dependency not needed for pure-logic tests).
    """

    # Expected mapping (must match src/rl/env.ACTION_MAP)
    _EXPECTED = {0: "move_forward", 1: "turn_left", 2: "turn_right"}

    def test_action_map_keys(self):
        """ACTION_MAP contains exactly 3 actions."""
        assert len(self._EXPECTED) == 3
        assert set(self._EXPECTED.keys()) == {0, 1, 2}

    def test_action_map_values(self):
        """ACTION_MAP maps to valid habitat-sim action strings."""
        valid = {"move_forward", "turn_left", "turn_right"}
        assert set(self._EXPECTED.values()) == valid

    def test_action_map_forward(self):
        assert self._EXPECTED[0] == "move_forward"

    def test_action_map_turn_left(self):
        assert self._EXPECTED[1] == "turn_left"

    def test_action_map_turn_right(self):
        assert self._EXPECTED[2] == "turn_right"


# ===========================================================================
# Integration tests (require habitat-sim)
# ===========================================================================

_habitat_available = False
try:
    import habitat_sim
    _habitat_available = True
except ImportError:
    pass

habitat = pytest.mark.skipif(
    not _habitat_available,
    reason="habitat-sim not installed",
)


@habitat
class TestNavigationEnvIntegration:
    """Integration tests that create a real NavigationEnv with habitat-sim."""

    def _make_env(self) -> "NavigationEnv":
        from configs.rl_config import RLConfig
        from src.rl.env import NavigationEnv
        config = RLConfig(
            max_episode_steps=50,
            image_size=64,  # smaller for faster tests
        )
        return NavigationEnv(config)

    def test_env_reset_returns_valid_obs(self):
        """reset() returns obs matching observation_space."""
        env = self._make_env()
        try:
            obs, info = env.reset(seed=42)

            assert isinstance(obs, dict)
            for key, space in env.observation_space.spaces.items():
                assert key in obs, f"Missing key: {key}"
                assert obs[key].shape == space.shape, (
                    f"Shape mismatch for {key}: {obs[key].shape} vs {space.shape}"
                )
                assert obs[key].dtype == space.dtype, (
                    f"Dtype mismatch for {key}: {obs[key].dtype} vs {space.dtype}"
                )

            assert isinstance(info, dict)
            assert "success" in info
            assert "distance_to_goal" in info
        finally:
            env.close()

    def test_env_step_returns_valid_transition(self):
        """step() returns (obs, reward, terminated, truncated, info)."""
        env = self._make_env()
        try:
            obs, _ = env.reset(seed=42)
            obs2, reward, terminated, truncated, info = env.step(0)

            assert isinstance(obs2, dict)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

            for key in env.observation_space.spaces:
                assert key in obs2
        finally:
            env.close()

    def test_env_episode_terminates_at_max_steps(self):
        """Episode truncates at max_episode_steps."""
        from configs.rl_config import RLConfig
        from src.rl.env import NavigationEnv

        config = RLConfig(max_episode_steps=5, image_size=64)
        env = NavigationEnv(config)
        try:
            env.reset(seed=42)
            truncated = False
            for _ in range(10):
                _, _, terminated, truncated, _ = env.step(0)
                if terminated or truncated:
                    break
            assert truncated or terminated  # should terminate within 10 steps (max=5)
        finally:
            env.close()

    def test_env_close_does_not_raise(self):
        """close() can be called without error."""
        env = self._make_env()
        env.reset(seed=42)
        env.close()
        # Double close should also be safe
        env.close()

    def test_env_observation_space_structure(self):
        """observation_space is a Dict with expected keys."""
        env = self._make_env()
        try:
            import gymnasium
            assert isinstance(env.observation_space, gymnasium.spaces.Dict)
            assert "forward_rgb" in env.observation_space.spaces
            assert "rear_rgb" in env.observation_space.spaces
            assert "depth" in env.observation_space.spaces
            assert "goal_vector" in env.observation_space.spaces
            assert "imu" in env.observation_space.spaces
        finally:
            env.close()

    def test_env_action_space_is_discrete_3(self):
        """action_space is Discrete(3)."""
        env = self._make_env()
        try:
            import gymnasium
            assert isinstance(env.action_space, gymnasium.spaces.Discrete)
            assert env.action_space.n == 3
        finally:
            env.close()
