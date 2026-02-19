"""RL training and environment configuration.

All RL hyperparameters and environment settings in a single dataclass.
Follows the pattern of configs/sim_config.py.
"""

from dataclasses import dataclass


@dataclass
class RLConfig:
    """Configuration for RL navigation training and evaluation."""

    # Environment
    scene_id: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    max_episode_steps: int = 500
    goal_threshold: float = 0.5  # meters, same as M3
    image_size: int = 128  # resize RGB/depth to (image_size, image_size)
    use_rear_rgb: bool = True  # include rear camera in obs
    use_depth: bool = True  # include depth in obs
    use_imu: bool = True  # include IMU vector in obs
    slack_penalty: float = -0.01  # per-step penalty
    success_reward: float = 10.0  # sparse bonus on goal reach
    collision_penalty: float = -0.1
    progress_reward_scale: float = 1.0  # dense reward per meter of progress toward goal
    seed: int = 42

    # Training
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 256  # PPO rollout length
    batch_size: int = 64
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_envs: int = 4  # parallel environments
    log_dir: str = "logs/rl"
    save_dir: str = "checkpoints/rl"
    save_interval: int = 50_000  # save every N timesteps

    # Evaluation
    eval_episodes: int = 50
    eval_seed: int = 99  # different from training seed
