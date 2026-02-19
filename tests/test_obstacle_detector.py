"""Tests for obstacle detector.

Covers AC4: all-zero semantic, non-zero IDs, depth filtering,
min_obstacle_pixels, custom obstacle_ids, detect_both_cameras.
All tests are pure numpy -- no habitat-sim dependency.
"""

import numpy as np
import pytest

from src.perception.obstacle_detector import ObstacleDetector, ObstacleDetection


class TestObstacleDetectorZeroSemantic:
    def test_all_zero_returns_empty_mask(self):
        """All-zero semantic image returns no obstacles."""
        det = ObstacleDetector(min_obstacle_pixels=10)
        semantic = np.zeros((480, 640), dtype=np.uint32)
        depth = np.full((480, 640), 3.0, dtype=np.float32)
        result = det.detect(semantic, depth)
        assert result.obstacle_count == 0
        assert not np.any(result.mask)


class TestObstacleDetectorNonZeroIDs:
    def test_nonzero_ids_detected(self):
        """Non-zero semantic IDs are detected as obstacles."""
        det = ObstacleDetector(min_obstacle_pixels=10)
        semantic = np.zeros((480, 640), dtype=np.uint32)
        # Fill a 50x50 block with object ID 5
        semantic[100:150, 100:150] = 5
        depth = np.full((480, 640), 3.0, dtype=np.float32)
        result = det.detect(semantic, depth)
        assert result.obstacle_count >= 1
        assert np.sum(result.mask) == 50 * 50


class TestObstacleDetectorDepthFilter:
    def test_far_objects_excluded(self):
        """Obstacles at depth > max_range are excluded."""
        det = ObstacleDetector(min_obstacle_pixels=10)
        semantic = np.zeros((480, 640), dtype=np.uint32)
        semantic[100:200, 100:200] = 3  # 100x100 = 10000 pixels
        depth = np.full((480, 640), 8.0, dtype=np.float32)  # beyond max_range
        result = det.detect(semantic, depth, max_range=5.0)
        assert result.obstacle_count == 0
        assert not np.any(result.mask)

    def test_near_objects_detected(self):
        """Obstacles within max_range are detected."""
        det = ObstacleDetector(min_obstacle_pixels=10)
        semantic = np.zeros((480, 640), dtype=np.uint32)
        semantic[100:200, 100:200] = 3
        depth = np.full((480, 640), 3.0, dtype=np.float32)
        result = det.detect(semantic, depth, max_range=5.0)
        assert result.obstacle_count >= 1


class TestObstacleDetectorMinPixels:
    def test_small_detections_filtered(self):
        """Detections with fewer pixels than min_obstacle_pixels are suppressed."""
        det = ObstacleDetector(min_obstacle_pixels=100)
        semantic = np.zeros((480, 640), dtype=np.uint32)
        # Only 25 pixels (5x5 block) -- below threshold
        semantic[0:5, 0:5] = 1
        depth = np.full((480, 640), 3.0, dtype=np.float32)
        result = det.detect(semantic, depth)
        assert result.obstacle_count == 0
        assert not np.any(result.mask)

    def test_large_detections_pass(self):
        """Detections with enough pixels pass the threshold."""
        det = ObstacleDetector(min_obstacle_pixels=100)
        semantic = np.zeros((480, 640), dtype=np.uint32)
        # 400 pixels (20x20 block) -- above threshold
        semantic[0:20, 0:20] = 1
        depth = np.full((480, 640), 3.0, dtype=np.float32)
        result = det.detect(semantic, depth)
        assert result.obstacle_count >= 1
        assert np.sum(result.mask) == 400


class TestObstacleDetectorCustomIDs:
    def test_custom_obstacle_ids_restricts_detection(self):
        """Custom obstacle_ids set restricts which IDs are detected."""
        det = ObstacleDetector(obstacle_ids={5, 10}, min_obstacle_pixels=10)
        semantic = np.zeros((480, 640), dtype=np.uint32)
        # Block with ID 5 (should be detected)
        semantic[0:20, 0:20] = 5
        # Block with ID 3 (should NOT be detected)
        semantic[100:120, 100:120] = 3
        depth = np.full((480, 640), 3.0, dtype=np.float32)
        result = det.detect(semantic, depth)
        # Only ID=5 block should be detected
        assert np.sum(result.mask[0:20, 0:20]) == 400
        assert not np.any(result.mask[100:120, 100:120])


class TestObstacleDetectorBothCameras:
    def test_detect_both_returns_two_independent(self):
        """detect_both_cameras returns two independent ObstacleDetection objects."""
        det = ObstacleDetector(min_obstacle_pixels=10)

        fwd_semantic = np.zeros((480, 640), dtype=np.uint32)
        fwd_semantic[0:20, 0:20] = 1  # 400 pixels forward
        rear_semantic = np.zeros((480, 640), dtype=np.uint32)
        rear_semantic[200:250, 200:250] = 2  # 2500 pixels rear

        depth = np.full((480, 640), 3.0, dtype=np.float32)
        fwd_det, rear_det = det.detect_both_cameras(
            fwd_semantic, rear_semantic, depth, depth,
        )

        assert isinstance(fwd_det, ObstacleDetection)
        assert isinstance(rear_det, ObstacleDetection)
        assert np.sum(fwd_det.mask) == 400
        assert np.sum(rear_det.mask) == 2500
