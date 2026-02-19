"""PPO training entrypoint using stable-baselines3.

Usage:
    python -m src.rl.train
    python -m src.rl.train --total-timesteps 500000 --n-envs 2
"""

import argparse
import logging
import os
from typing import Callable, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from configs.rl_config import RLConfig
from src.rl.env import NavigationEnv
from src.rl.policy import NavigationFeaturesExtractor

logger = logging.getLogger(__name__)


def make_env(
    config: RLConfig,
    seed: int,
    rank: int,
) -> Callable[[], NavigationEnv]:
    """Factory function for creating environments in SubprocVecEnv.

    Each subprocess calls this to create its own NavigationEnv with
    its own Vehicle/Simulator instance and unique seed.
    """
    def _init() -> NavigationEnv:
        env_config = RLConfig(
            scene_id=config.scene_id,
            max_episode_steps=config.max_episode_steps,
            goal_threshold=config.goal_threshold,
            image_size=config.image_size,
            use_rear_rgb=config.use_rear_rgb,
            use_depth=config.use_depth,
            use_imu=config.use_imu,
            slack_penalty=config.slack_penalty,
            success_reward=config.success_reward,
            collision_penalty=config.collision_penalty,
            progress_reward_scale=config.progress_reward_scale,
            seed=seed + rank,
        )
        env = NavigationEnv(env_config)
        env = Monitor(env)
        return env
    return _init


def train(config: Optional[RLConfig] = None) -> None:
    """Train a PPO agent on NavigationEnv.

    Steps:
    1. Create vectorized environment (SubprocVecEnv with n_envs)
    2. Create PPO model with custom feature extractor
    3. Set up callbacks: CheckpointCallback, EvalCallback
    4. Train
    5. Save final model
    """
    if config is None:
        config = RLConfig()

    # Create output directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)

    logger.info(
        "Creating %d training environments (scene: %s)...",
        config.n_envs, config.scene_id,
    )

    # Create vectorized training environment
    # Use SubprocVecEnv so each env gets its own process and OpenGL context.
    # Fall back to DummyVecEnv if SubprocVecEnv fails.
    env_fns = [
        make_env(config, config.seed, rank=i)
        for i in range(config.n_envs)
    ]
    try:
        train_env = SubprocVecEnv(env_fns)
        logger.info("Using SubprocVecEnv with %d workers.", config.n_envs)
    except Exception as e:
        logger.warning(
            "SubprocVecEnv failed (%s), falling back to DummyVecEnv.", e,
        )
        train_env = DummyVecEnv(env_fns)

    # Create eval environment (single process)
    eval_env = DummyVecEnv([make_env(config, config.eval_seed, rank=0)])

    # Policy kwargs with custom feature extractor
    policy_kwargs = {
        "features_extractor_class": NavigationFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "share_rgb_weights": True,
        },
    }

    # Create PPO model
    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=config.log_dir,
        verbose=1,
        seed=config.seed,
    )

    logger.info("PPO model created. Policy architecture:")
    logger.info(model.policy)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(config.save_interval // config.n_envs, 1),
        save_path=config.save_dir,
        name_prefix="ppo_nav",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(config.save_dir, "best"),
        log_path=os.path.join(config.log_dir, "eval"),
        eval_freq=max(config.save_interval // config.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    # Train
    logger.info(
        "Starting training: %d timesteps, lr=%.1e, n_envs=%d",
        config.total_timesteps, config.learning_rate, config.n_envs,
    )
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(config.save_dir, "ppo_nav_final")
    model.save(final_path)
    logger.info("Training complete. Final model saved to %s.", final_path)

    # Cleanup
    train_env.close()
    eval_env.close()


def main() -> None:
    """CLI entrypoint for training."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train PPO navigation agent.")
    parser.add_argument(
        "--total-timesteps", type=int, default=1_000_000,
        help="Total training timesteps.",
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--scene", type=str, default="skokloster-castle",
        help="Scene name.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from.",
    )
    args = parser.parse_args()

    scene_path = f"data/scene_datasets/habitat-test-scenes/{args.scene}.glb"

    config = RLConfig(
        scene_id=scene_path,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        learning_rate=args.lr,
    )

    if args.resume is not None:
        logger.info("Resuming training from %s", args.resume)
        # Load existing model and continue training
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.save_dir, exist_ok=True)

        env_fns = [
            make_env(config, config.seed, rank=i)
            for i in range(config.n_envs)
        ]
        try:
            train_env = SubprocVecEnv(env_fns)
        except Exception:
            train_env = DummyVecEnv(env_fns)

        model = PPO.load(args.resume, env=train_env)
        eval_env = DummyVecEnv([make_env(config, config.eval_seed, rank=0)])

        checkpoint_cb = CheckpointCallback(
            save_freq=max(config.save_interval // config.n_envs, 1),
            save_path=config.save_dir,
            name_prefix="ppo_nav",
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(config.save_dir, "best"),
            log_path=os.path.join(config.log_dir, "eval"),
            eval_freq=max(config.save_interval // config.n_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
        )

        model.learn(
            total_timesteps=config.total_timesteps,
            callback=[checkpoint_cb, eval_cb],
            progress_bar=True,
            reset_num_timesteps=False,
        )

        final_path = os.path.join(config.save_dir, "ppo_nav_final")
        model.save(final_path)
        logger.info("Resumed training complete. Model saved to %s.", final_path)

        train_env.close()
        eval_env.close()
    else:
        train(config)


if __name__ == "__main__":
    main()
