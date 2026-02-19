"""Semantic-sensor-based obstacle detection.

Uses habitat-sim's semantic sensor output to identify obstacle pixels.
In scenes without semantic annotations, falls back to treating all
non-zero semantic IDs as potential obstacles.
"""

from dataclasses import dataclass
from typing import Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class ObstacleDetection:
    mask: NDArray[np.bool_]  # [H, W] True where obstacle detected
    object_ids: NDArray[np.uint32]  # [H, W] semantic object IDs (filtered)
    obstacle_count: int  # number of distinct obstacle regions
    timestamp_step: int


# Configurable set of semantic IDs considered "obstacle".
# Scene-dependent; empty means "everything non-zero is obstacle".
DEFAULT_OBSTACLE_IDS: Set[int] = set()


class ObstacleDetector:
    """Semantic-sensor-based obstacle detector.

    Uses habitat-sim's semantic sensor to identify obstacle pixels.
    In scenes without semantic annotations, falls back to treating all
    non-zero semantic IDs as potential obstacles.
    """

    def __init__(
        self,
        obstacle_ids: Optional[Set[int]] = None,
        min_obstacle_pixels: int = 100,
    ) -> None:
        """
        Args:
            obstacle_ids: Set of semantic IDs considered obstacles.
                If None or empty, all non-zero IDs are treated as obstacles.
            min_obstacle_pixels: Ignore detections with fewer pixels than this.
        """
        self._obstacle_ids: Set[int] = obstacle_ids if obstacle_ids else set()
        self._min_obstacle_pixels = min_obstacle_pixels
        self._step: int = 0

    def detect(
        self,
        semantic_image: NDArray[np.uint32],
        depth_image: NDArray[np.float32],
        max_range: float = 5.0,
    ) -> ObstacleDetection:
        """Detect obstacles in a single camera view.

        Steps:
        1. Filter semantic image: keep only pixels with IDs in obstacle_ids
           (or all non-zero if obstacle_ids is empty)
        2. Filter by depth range: discard pixels beyond max_range
        3. Apply minimum pixel count threshold
        4. Return binary mask + metadata

        Args:
            semantic_image: [H, W] uint32 semantic object IDs.
            depth_image: [H, W] float32 depth for range filtering.
            max_range: Only detect obstacles within this range (meters).

        Returns:
            ObstacleDetection with mask and metadata.
        """
        self._step += 1

        # Step 1: Semantic filtering
        if self._obstacle_ids:
            # Only keep pixels whose semantic ID is in the obstacle set
            sem_mask = np.isin(semantic_image, list(self._obstacle_ids))
        else:
            # All non-zero IDs are potential obstacles
            sem_mask = semantic_image > 0

        # Step 2: Depth range filtering
        depth_mask = np.isfinite(depth_image) & (depth_image > 0.0) & (depth_image <= max_range)
        combined_mask = sem_mask & depth_mask

        # Step 3: Minimum pixel count threshold
        total_obstacle_pixels = int(np.sum(combined_mask))
        if total_obstacle_pixels < self._min_obstacle_pixels:
            combined_mask = np.zeros_like(combined_mask)
            total_obstacle_pixels = 0

        # Count distinct obstacle IDs in the filtered mask
        if total_obstacle_pixels > 0:
            unique_ids = np.unique(semantic_image[combined_mask])
            obstacle_count = int(len(unique_ids))
        else:
            obstacle_count = 0

        # Build filtered object_ids array
        filtered_ids = np.where(combined_mask, semantic_image, np.uint32(0))

        return ObstacleDetection(
            mask=combined_mask,
            object_ids=filtered_ids,
            obstacle_count=obstacle_count,
            timestamp_step=self._step,
        )

    def detect_both_cameras(
        self,
        forward_semantic: NDArray[np.uint32],
        rear_semantic: NDArray[np.uint32],
        forward_depth: NDArray[np.float32],
        rear_depth: NDArray[np.float32],
        max_range: float = 5.0,
    ) -> Tuple[ObstacleDetection, ObstacleDetection]:
        """Detect obstacles from both cameras. Convenience method.

        Args:
            forward_semantic: [H, W] uint32 forward camera semantic IDs.
            rear_semantic: [H, W] uint32 rear camera semantic IDs.
            forward_depth: [H, W] float32 forward depth.
            rear_depth: [H, W] float32 rear depth.
            max_range: Only detect within this range.

        Returns:
            Tuple of (forward_detection, rear_detection).
        """
        fwd = self.detect(forward_semantic, forward_depth, max_range)
        rear = self.detect(rear_semantic, rear_depth, max_range)
        return fwd, rear
