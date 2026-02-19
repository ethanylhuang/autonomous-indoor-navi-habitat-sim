"""Feature-based visual odometry using ORB + Essential Matrix.

Estimates frame-to-frame ego-motion from consecutive RGB images.
Uses the forward camera only.

Scale ambiguity: Essential matrix decomposition gives rotation and
translation direction but not magnitude. M3's EKF fuses VO rotation
with IMU-derived displacement to resolve scale.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from src.utils.transforms import focal_length_from_hfov


@dataclass
class VOEstimate:
    rotation: NDArray[np.float64]  # [3, 3] rotation matrix (frame t-1 -> frame t)
    translation_dir: NDArray[np.float64]  # [3] unit vector, translation direction (scale unknown)
    num_inliers: int  # RANSAC inliers
    is_valid: bool  # True if enough inliers for reliable estimate
    timestamp_step: int


class VisualOdometry:
    """Feature-based visual odometry using ORB + Essential Matrix.

    Estimates frame-to-frame ego-motion from consecutive RGB images.
    Uses the forward camera only (rear camera for cross-validation later).
    """

    def __init__(
        self,
        hfov_deg: float = 90.0,
        resolution: Tuple[int, int] = (480, 640),  # (H, W)
        min_inliers: int = 15,
        max_features: int = 1000,
    ) -> None:
        """Initialize VO with camera intrinsics computed from HFOV + resolution.

        Args:
            hfov_deg: Horizontal field of view in degrees.
            resolution: (height, width) of input images.
            min_inliers: Minimum RANSAC inliers for a valid estimate.
            max_features: Maximum ORB features to detect.
        """
        H, W = resolution
        fx = focal_length_from_hfov(hfov_deg, W)
        cx = W / 2.0
        cy = H / 2.0

        self._K = np.array([
            [fx, 0.0, cx],
            [0.0, fx, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        self._min_inliers = min_inliers
        self._orb = cv2.ORB_create(nfeatures=max_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self._prev_gray: Optional[NDArray[np.uint8]] = None
        self._prev_kp = None
        self._prev_des: Optional[NDArray] = None
        self._step: int = 0

    def update(
        self,
        rgb_frame: NDArray[np.uint8],
    ) -> VOEstimate:
        """Process new frame, estimate motion relative to previous frame.

        First call: stores frame, returns identity rotation + zero translation.
        Subsequent calls: ORB detect+match -> Essential matrix -> decompose.

        Args:
            rgb_frame: [H, W, 4] RGBA or [H, W, 3] RGB image.

        Returns:
            VOEstimate with rotation, translation direction, and validity.
        """
        self._step += 1

        # Guard against degenerate image sizes
        if rgb_frame.shape[0] < 10 or rgb_frame.shape[1] < 10:
            return VOEstimate(
                rotation=np.eye(3, dtype=np.float64),
                translation_dir=np.zeros(3, dtype=np.float64),
                num_inliers=0,
                is_valid=False,
                timestamp_step=self._step,
            )

        # Convert to grayscale
        if rgb_frame.ndim == 3 and rgb_frame.shape[2] == 4:
            gray = cv2.cvtColor(rgb_frame[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif rgb_frame.ndim == 3 and rgb_frame.shape[2] == 3:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb_frame

        # Detect ORB keypoints and descriptors
        kp, des = self._orb.detectAndCompute(gray, None)

        # First frame: store and return identity
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_des = des
            return VOEstimate(
                rotation=np.eye(3, dtype=np.float64),
                translation_dir=np.zeros(3, dtype=np.float64),
                num_inliers=0,
                is_valid=True,
                timestamp_step=self._step,
            )

        # If current or previous frame has no descriptors, return invalid
        if des is None or self._prev_des is None or len(kp) < 2 or len(self._prev_kp) < 2:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_des = des
            return VOEstimate(
                rotation=np.eye(3, dtype=np.float64),
                translation_dir=np.zeros(3, dtype=np.float64),
                num_inliers=0,
                is_valid=False,
                timestamp_step=self._step,
            )

        # Match against previous frame using BFMatcher with knn
        matches = self._bf.knnMatch(self._prev_des, des, k=2)

        # Lowe's ratio test (0.75)
        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        # Not enough matches
        if len(good_matches) < self._min_inliers:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_des = des
            return VOEstimate(
                rotation=np.eye(3, dtype=np.float64),
                translation_dir=np.zeros(3, dtype=np.float64),
                num_inliers=len(good_matches),
                is_valid=False,
                timestamp_step=self._step,
            )

        # Extract matched point coordinates
        pts_prev = np.array(
            [self._prev_kp[m.queryIdx].pt for m in good_matches], dtype=np.float32
        )
        pts_curr = np.array(
            [kp[m.trainIdx].pt for m in good_matches], dtype=np.float32
        )

        # Find essential matrix
        E, mask_E = cv2.findEssentialMat(
            pts_prev, pts_curr, self._K,
            method=cv2.RANSAC, threshold=1.0,
        )

        if E is None or mask_E is None:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_des = des
            return VOEstimate(
                rotation=np.eye(3, dtype=np.float64),
                translation_dir=np.zeros(3, dtype=np.float64),
                num_inliers=0,
                is_valid=False,
                timestamp_step=self._step,
            )

        # Recover pose from essential matrix
        num_inliers, R, t, mask_pose = cv2.recoverPose(
            E, pts_prev, pts_curr, self._K, mask=mask_E.copy(),
        )

        # Store current frame data for next iteration
        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_des = des

        # Check inlier count
        if num_inliers < self._min_inliers:
            return VOEstimate(
                rotation=R.astype(np.float64),
                translation_dir=t.flatten().astype(np.float64),
                num_inliers=int(num_inliers),
                is_valid=False,
                timestamp_step=self._step,
            )

        # Normalize translation to unit vector
        t_flat = t.flatten().astype(np.float64)
        t_norm = np.linalg.norm(t_flat)
        if t_norm > 1e-10:
            t_flat = t_flat / t_norm

        return VOEstimate(
            rotation=R.astype(np.float64),
            translation_dir=t_flat,
            num_inliers=int(num_inliers),
            is_valid=True,
            timestamp_step=self._step,
        )

    def reset(self) -> None:
        """Clear previous frame. Next update() returns identity."""
        self._prev_gray = None
        self._prev_kp = None
        self._prev_des = None
        self._step = 0
