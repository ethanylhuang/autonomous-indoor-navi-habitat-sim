"""Custom multi-modal feature extractor for navigation observations.

Processes forward_rgb, rear_rgb, depth, goal_vector, and imu observations
through separate CNN/pass-through branches and concatenates into a single
feature vector for the SB3 PPO policy head.
"""

from typing import Dict

import gymnasium
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NavigationFeaturesExtractor(BaseFeaturesExtractor):
    """Multi-modal feature extractor for navigation observations.

    Architecture:
    - forward_rgb (128x128x3) -> NatureCNN -> 256-dim
    - rear_rgb (128x128x3)    -> NatureCNN (shared weights) -> 256-dim
    - depth (128x128x1)       -> small CNN -> 128-dim
    - goal_vector (2,)        -> pass-through
    - imu (6,)                -> pass-through

    Concat all -> Linear(total, features_dim) -> ReLU -> output
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Dict,
        features_dim: int = 256,
        share_rgb_weights: bool = True,
    ) -> None:
        # Must call super with the final features_dim
        super().__init__(observation_space, features_dim)

        self._share_rgb_weights = share_rgb_weights
        self._obs_keys = list(observation_space.spaces.keys())

        # Track total concatenated feature size
        concat_size = 0

        # ---- RGB CNN (NatureCNN-style) ----
        # For 128x128 input:
        # Conv(3, 32, 8, stride=4) -> 31x31
        # Conv(32, 64, 4, stride=2) -> 14x14
        # Conv(64, 64, 3, stride=1) -> 12x12
        # Flatten -> 64*12*12 = 9216 -> Linear(9216, 256)
        rgb_in_channels = 3
        self._rgb_cnn = self._build_nature_cnn(rgb_in_channels, 256)
        concat_size += 256  # forward_rgb

        if "rear_rgb" in self._obs_keys:
            if share_rgb_weights:
                self._rear_rgb_cnn = self._rgb_cnn  # weight sharing
            else:
                self._rear_rgb_cnn = self._build_nature_cnn(rgb_in_channels, 256)
            concat_size += 256

        # ---- Depth CNN (smaller) ----
        if "depth" in self._obs_keys:
            self._depth_cnn = self._build_depth_cnn(1, 128)
            concat_size += 128

        # ---- Goal vector (pass-through) ----
        if "goal_vector" in self._obs_keys:
            goal_shape = observation_space.spaces["goal_vector"].shape
            concat_size += goal_shape[0]  # 2

        # ---- IMU (pass-through) ----
        if "imu" in self._obs_keys:
            imu_shape = observation_space.spaces["imu"].shape
            concat_size += imu_shape[0]  # 6

        # ---- Final projection ----
        self._fc = nn.Sequential(
            nn.Linear(concat_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through all branches, concat, and project."""
        features = []

        # Forward RGB: (B, H, W, C) -> (B, C, H, W) and normalize to [0, 1]
        fwd_rgb = observations["forward_rgb"].float() / 255.0
        fwd_rgb = fwd_rgb.permute(0, 3, 1, 2)
        features.append(self._rgb_cnn(fwd_rgb))

        # Rear RGB
        if "rear_rgb" in self._obs_keys:
            rear_rgb = observations["rear_rgb"].float() / 255.0
            rear_rgb = rear_rgb.permute(0, 3, 1, 2)
            if self._share_rgb_weights:
                features.append(self._rgb_cnn(rear_rgb))
            else:
                features.append(self._rear_rgb_cnn(rear_rgb))

        # Depth: (B, H, W, 1) -> (B, 1, H, W)
        if "depth" in self._obs_keys:
            depth = observations["depth"].float()
            depth = depth.permute(0, 3, 1, 2)
            features.append(self._depth_cnn(depth))

        # Goal vector: (B, 2) pass-through
        if "goal_vector" in self._obs_keys:
            features.append(observations["goal_vector"].float())

        # IMU: (B, 6) pass-through
        if "imu" in self._obs_keys:
            features.append(observations["imu"].float())

        # Concat and project
        concatenated = torch.cat(features, dim=1)
        return self._fc(concatenated)

    @staticmethod
    def _build_nature_cnn(in_channels: int, output_dim: int) -> nn.Sequential:
        """Build a NatureCNN backbone (Mnih et al. 2015).

        For 128x128 input:
            Conv(in_ch, 32, 8, stride=4) -> 31x31
            Conv(32, 64, 4, stride=2) -> 14x14
            Conv(64, 64, 3, stride=1) -> 12x12
            Flatten -> 64*12*12 = 9216 -> Linear(9216, output_dim)
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, output_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _build_depth_cnn(in_channels: int, output_dim: int) -> nn.Sequential:
        """Build a smaller CNN for depth observations.

        For 128x128 input:
            Conv(in_ch, 16, 8, stride=4) -> 31x31
            Conv(16, 32, 4, stride=2) -> 14x14
            Flatten -> 32*14*14 = 6272 -> Linear(6272, output_dim)
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, output_dim),
            nn.ReLU(),
        )
