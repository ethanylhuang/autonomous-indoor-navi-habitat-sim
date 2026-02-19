"""Tests for LiDAR point cloud conversion.

Covers AC1: depth_to_point_cloud, transform_point_cloud, merge_point_clouds.
All tests are pure numpy -- no habitat-sim dependency.
"""

import math

import numpy as np
import pytest

from src.sensors.lidar import (
    PointCloud,
    depth_to_point_cloud,
    merge_point_clouds,
    transform_point_cloud,
)


class TestDepthToPointCloud:
    def test_uniform_depth_returns_correct_count(self):
        """All-valid uniform depth image produces H*W points."""
        depth = np.full((480, 640), 5.0, dtype=np.float32)
        pc = depth_to_point_cloud(depth, hfov_deg=90.0, max_depth=10.0)
        assert pc.points.shape == (480 * 640, 3)
        assert pc.num_valid == 480 * 640
        assert pc.points.dtype == np.float32

    def test_center_pixel_has_z_equals_neg_depth(self):
        """Point at image center (240, 320) has z = -depth_value."""
        depth = np.full((480, 640), 3.0, dtype=np.float32)
        pc = depth_to_point_cloud(depth, hfov_deg=90.0)
        # Center pixel: u=320, v=240. cx=320, cy=240
        # x = (320 - 320) * 3.0 / f = 0
        # y = -(240 - 240) * 3.0 / f = 0
        # z = -3.0
        center_idx = 240 * 640 + 320
        np.testing.assert_allclose(pc.points[center_idx, 2], -3.0, rtol=0.01)
        np.testing.assert_allclose(pc.points[center_idx, 0], 0.0, atol=0.01)
        np.testing.assert_allclose(pc.points[center_idx, 1], 0.0, atol=0.01)

    def test_edge_pixels_have_correct_angular_spread(self):
        """Points at image edges have |x/z| = tan(hfov/2) for leftmost/rightmost columns."""
        depth = np.full((480, 640), 5.0, dtype=np.float32)
        pc = depth_to_point_cloud(depth, hfov_deg=90.0)

        # At u=0 (leftmost): x = (0 - 320) * 5.0 / f, z = -5.0
        # f = 640 / (2 * tan(45)) = 640 / 2 = 320
        # x = -320 * 5.0 / 320 = -5.0
        # |x/z| = 5.0/5.0 = 1.0 = tan(45deg)
        # Row 240, col 0: index = 240 * 640 + 0
        left_idx = 240 * 640 + 0
        x_left = pc.points[left_idx, 0]
        z_left = pc.points[left_idx, 2]
        ratio = abs(x_left / z_left)
        expected_ratio = math.tan(math.radians(45.0))
        np.testing.assert_allclose(ratio, expected_ratio, rtol=0.02)

    def test_zero_depth_excluded(self):
        """Zero depth pixels are excluded from the point cloud."""
        depth = np.full((10, 10), 0.0, dtype=np.float32)
        pc = depth_to_point_cloud(depth, hfov_deg=90.0)
        assert pc.num_valid == 0
        assert pc.points.shape == (0, 3)

    def test_nan_inf_excluded(self):
        """NaN and inf depth pixels are excluded."""
        depth = np.full((10, 10), 5.0, dtype=np.float32)
        depth[0, 0] = np.nan
        depth[0, 1] = np.inf
        depth[0, 2] = -np.inf
        pc = depth_to_point_cloud(depth, hfov_deg=90.0)
        assert pc.num_valid == 10 * 10 - 3

    def test_max_depth_filtering(self):
        """Points beyond max_depth are excluded."""
        depth = np.full((10, 10), 15.0, dtype=np.float32)
        depth[5, 5] = 3.0
        pc = depth_to_point_cloud(depth, hfov_deg=90.0, max_depth=10.0)
        assert pc.num_valid == 1


class TestTransformPointCloud:
    def test_identity_rotation_zero_translation_preserves(self):
        """Identity rotation + zero translation preserves all coordinates."""
        points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 5.0]], dtype=np.float32)
        pc = PointCloud(points=points, num_valid=2)
        identity_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        zero_pos = np.zeros(3, dtype=np.float64)
        transformed = transform_point_cloud(pc, zero_pos, identity_rot)
        np.testing.assert_allclose(transformed.points, points, atol=1e-5)

    def test_90deg_y_rotation_swaps_xz(self):
        """90-degree Y rotation rotates X->-Z, Z->X."""
        points = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        pc = PointCloud(points=points, num_valid=1)
        # 90 deg about Y: quat = [cos(pi/4), 0, sin(pi/4), 0]
        angle = math.pi / 2.0
        rot_quat = np.array([
            math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0
        ], dtype=np.float64)
        zero_pos = np.zeros(3, dtype=np.float64)
        transformed = transform_point_cloud(pc, zero_pos, rot_quat)
        # [1, 0, 0] rotated 90 deg Y -> [0, 0, -1]
        np.testing.assert_allclose(transformed.points[0], [0.0, 0.0, -1.0], atol=1e-5)

    def test_empty_point_cloud(self):
        """Empty point cloud transforms without error."""
        pc = PointCloud(points=np.empty((0, 3), dtype=np.float32), num_valid=0)
        identity_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        zero_pos = np.zeros(3, dtype=np.float64)
        transformed = transform_point_cloud(pc, zero_pos, identity_rot)
        assert transformed.num_valid == 0
        assert transformed.points.shape == (0, 3)


class TestMergePointClouds:
    def test_merge_concatenates_and_updates_count(self):
        """Merging two point clouds concatenates points."""
        pc1 = PointCloud(
            points=np.array([[1.0, 2.0, 3.0]], dtype=np.float32), num_valid=1,
        )
        pc2 = PointCloud(
            points=np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
            num_valid=2,
        )
        merged = merge_point_clouds([pc1, pc2])
        assert merged.num_valid == 3
        assert merged.points.shape == (3, 3)
        np.testing.assert_array_equal(merged.points[0], [1.0, 2.0, 3.0])

    def test_merge_empty_list(self):
        """Merging empty list returns empty point cloud."""
        merged = merge_point_clouds([])
        assert merged.num_valid == 0
        assert merged.points.shape == (0, 3)
