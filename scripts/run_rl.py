"""Evaluate a trained RL navigation agent and optionally compare with classical baseline.

Usage:
    python -m scripts.run_rl --model-path checkpoints/rl/ppo_nav_final
    python -m scripts.run_rl --model-path checkpoints/rl/ppo_nav_final --episodes 50 --seed 99
    python -m scripts.run_rl --model-path checkpoints/rl/ppo_nav_final --classical-results results/classical.npz
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from configs.rl_config import RLConfig
from src.rl.env import NavigationEnv

logger = logging.getLogger(__name__)


@dataclass
class RLEpisodeResult:
    success: bool
    spl: float
    path_length_steps: int
    distance_to_goal: float
    collisions: int
    geodesic_distance: float


def evaluate(
    model_path: str,
    config: RLConfig,
    episodes: int,
    seed: int,
    deterministic: bool = True,
) -> List[RLEpisodeResult]:
    """Run evaluation episodes with a trained PPO model.

    Args:
        model_path: Path to the saved SB3 PPO model (without .zip extension).
        config: RL configuration.
        episodes: Number of episodes to evaluate.
        seed: Random seed for episode generation.
        deterministic: Whether to use deterministic actions.

    Returns:
        List of per-episode results.
    """
    from stable_baselines3 import PPO

    # Create single evaluation environment (not vectorized)
    eval_config = RLConfig(
        scene_id=config.scene_id,
        max_episode_steps=config.max_episode_steps,
        goal_threshold=config.goal_threshold,
        image_size=config.image_size,
        use_rear_rgb=config.use_rear_rgb,
        use_depth=config.use_depth,
        use_imu=config.use_imu,
        seed=seed,
    )
    env = NavigationEnv(eval_config)

    # Load trained model
    model = PPO.load(model_path, env=None)
    logger.info("Loaded model from %s", model_path)

    results: List[RLEpisodeResult] = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        episode_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_steps += 1
            done = terminated or truncated

        result = RLEpisodeResult(
            success=info["success"],
            spl=info["spl"],
            path_length_steps=episode_steps,
            distance_to_goal=info["distance_to_goal"],
            collisions=info["collisions"],
            geodesic_distance=info["geodesic_distance"],
        )
        results.append(result)

        logger.info(
            "Episode %d/%d: %s | SPL=%.3f | steps=%d | dist=%.2fm | collisions=%d",
            ep + 1, episodes,
            "SUCCESS" if result.success else "FAIL",
            result.spl,
            result.path_length_steps,
            result.distance_to_goal,
            result.collisions,
        )

    env.close()
    return results


def print_summary(results: List[RLEpisodeResult], label: str = "RL") -> Dict[str, float]:
    """Print evaluation summary and return metrics dict."""
    n = len(results)
    if n == 0:
        print("No completed episodes.")
        return {}

    successes = sum(1 for r in results if r.success)
    success_rate = successes / n
    mean_spl = sum(r.spl for r in results) / n
    mean_steps = sum(r.path_length_steps for r in results) / n
    total_collisions = sum(r.collisions for r in results)

    print("\n" + "=" * 60)
    print(f"{label} NAVIGATION SUMMARY")
    print("=" * 60)
    print(f"Episodes:          {n}")
    print(f"Success rate:      {success_rate:.1%} ({successes}/{n})")
    print(f"Mean SPL:          {mean_spl:.3f}")
    print(f"Mean steps:        {mean_steps:.1f}")
    print(f"Total collisions:  {total_collisions}")
    print("=" * 60)

    return {
        "success_rate": success_rate,
        "mean_spl": mean_spl,
        "mean_steps": mean_steps,
        "total_collisions": total_collisions,
    }


def print_comparison(
    rl_metrics: Dict[str, float],
    classical_path: str,
) -> None:
    """Print side-by-side comparison of RL vs classical results.

    Expects classical results as a JSON file with keys:
    success_rate, mean_spl, mean_steps, total_collisions.
    """
    try:
        with open(classical_path, "r") as f:
            classical = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("Could not load classical results from %s: %s", classical_path, e)
        return

    print("\n" + "=" * 60)
    print("RL vs CLASSICAL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20s} {'RL':<12s} {'Classical':<12s}")

    fmt_map = {
        "success_rate": ("{:.1%}", "{:.1%}"),
        "mean_spl": ("{:.3f}", "{:.3f}"),
        "mean_steps": ("{:.1f}", "{:.1f}"),
        "total_collisions": ("{:.0f}", "{:.0f}"),
    }

    labels = {
        "success_rate": "Success rate:",
        "mean_spl": "Mean SPL:",
        "mean_steps": "Mean steps:",
        "total_collisions": "Total collisions:",
    }

    for key in ["success_rate", "mean_spl", "mean_steps", "total_collisions"]:
        rl_val = rl_metrics.get(key, 0.0)
        cl_val = classical.get(key, 0.0)
        rl_fmt, cl_fmt = fmt_map[key]
        print(f"{labels[key]:<20s} {rl_fmt.format(rl_val):<12s} {cl_fmt.format(cl_val):<12s}")

    print("=" * 60)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate trained RL navigation agent.",
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to saved PPO model (without .zip).",
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--scene", type=str, default="skokloster-castle",
        help="Scene name.",
    )
    parser.add_argument(
        "--seed", type=int, default=99,
        help="Evaluation seed (should match classical eval seed).",
    )
    parser.add_argument(
        "--classical-results", type=str, default=None,
        help="Path to classical results JSON for comparison.",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic (non-deterministic) actions.",
    )
    args = parser.parse_args()

    scene_path = f"data/scene_datasets/habitat-test-scenes/{args.scene}.glb"
    config = RLConfig(scene_id=scene_path, seed=args.seed)

    results = evaluate(
        model_path=args.model_path,
        config=config,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
    )

    rl_metrics = print_summary(results, label="RL")

    if args.classical_results is not None:
        print_comparison(rl_metrics, args.classical_results)

    logger.info("Done.")


if __name__ == "__main__":
    main()
