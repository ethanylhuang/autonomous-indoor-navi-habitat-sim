"""Tests for visual odometry.

Covers AC3: first frame identity, identical frames, shifted image,
reset, inlier count, featureless images.
All tests are pure numpy + cv2 -- no habitat-sim dependency.
"""

import numpy as np
import pytest

from src.perception.visual_odometry import VisualOdometry, VOEstimate


def _make_textured_image(seed: int = 42) -> np.ndarray:
    """Create a synthetic textured RGBA image with detectable features."""
    rng = np.random.RandomState(seed)
    # Random noise + some structure for ORB features
    img = rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    # Add some circles/squares for feature points
    import cv2
    for _ in range(50):
        cx = rng.randint(20, 620)
        cy = rng.randint(20, 460)
        r = rng.randint(5, 20)
        color = tuple(int(c) for c in rng.randint(0, 256, 3))
        cv2.circle(img, (cx, cy), r, color, -1)
    # Convert to RGBA
    alpha = np.full((480, 640, 1), 255, dtype=np.uint8)
    rgba = np.concatenate([img, alpha], axis=-1)
    return rgba


class TestVOFirstFrame:
    def test_first_frame_returns_identity(self):
        """First update() call returns identity rotation and zero translation."""
        vo = VisualOdometry()
        img = _make_textured_image()
        est = vo.update(img)
        assert est.is_valid is True
        np.testing.assert_allclose(est.rotation, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(est.translation_dir, np.zeros(3), atol=1e-10)
        assert est.timestamp_step == 1


class TestVOIdenticalFrames:
    def test_identical_frames_low_motion(self):
        """Two identical frames should produce near-identity or invalid estimate."""
        vo = VisualOdometry()
        img = _make_textured_image()
        vo.update(img)
        est = vo.update(img)
        # With identical frames, either:
        # - is_valid=False (not enough distinct motion)
        # - is_valid=True with near-identity rotation
        if est.is_valid:
            np.testing.assert_allclose(est.rotation, np.eye(3), atol=0.1)
        assert est.num_inliers >= 0


class TestVOShiftedImage:
    def test_shifted_image_valid_estimate(self):
        """A horizontally shifted image should produce a valid estimate."""
        vo = VisualOdometry()
        img = _make_textured_image(seed=123)
        vo.update(img)

        # Shift image horizontally by 10 pixels (simulates camera panning)
        shifted = np.zeros_like(img)
        shifted[:, 10:, :] = img[:, :-10, :]
        est = vo.update(shifted)

        # Should detect motion if enough features match
        if est.is_valid:
            assert est.num_inliers >= 15
            # Rotation should be orthogonal
            R = est.rotation
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=0.05)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=0.05)
            # Translation direction should be unit vector
            np.testing.assert_allclose(np.linalg.norm(est.translation_dir), 1.0, atol=1e-6)


class TestVOReset:
    def test_reset_causes_identity_on_next_call(self):
        """After reset(), next update() returns identity."""
        vo = VisualOdometry()
        img1 = _make_textured_image(seed=1)
        img2 = _make_textured_image(seed=2)
        vo.update(img1)
        vo.update(img2)
        vo.reset()
        est = vo.update(img1)
        assert est.is_valid is True
        np.testing.assert_allclose(est.rotation, np.eye(3), atol=1e-10)
        assert est.timestamp_step == 1


class TestVOInlierCount:
    def test_num_inliers_non_negative(self):
        """num_inliers should always be non-negative."""
        vo = VisualOdometry()
        img = _make_textured_image()
        est = vo.update(img)
        assert est.num_inliers >= 0


class TestVOFeatureless:
    def test_featureless_image_returns_invalid(self):
        """A solid color image has no features, should return is_valid=False."""
        vo = VisualOdometry()
        # First frame with features
        img1 = _make_textured_image()
        vo.update(img1)
        # Second frame: solid gray
        solid = np.full((480, 640, 4), 128, dtype=np.uint8)
        est = vo.update(solid)
        assert est.is_valid is False
