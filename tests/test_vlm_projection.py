"""Unit tests for VLM pixel-to-world projection.

Tests coordinate transformations without requiring habitat_sim.
"""

import numpy as np


def test_pixel_to_camera_center():
    """Test pixel at image center with known depth."""
    # Direct import to avoid __init__ chain
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Import only projection module, not the package
    import importlib.util
    proj_path = Path(__file__).parent.parent / "src" / "vlm" / "projection.py"
    spec = importlib.util.spec_from_file_location("projection", proj_path)
    projection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(projection)

    pixel_to_camera_frame = projection.pixel_to_camera_frame

    # 480x640 image, center pixel at (320, 240)
    H, W = 480, 640
    depth = np.ones((H, W), dtype=np.float32) * 2.0  # 2m depth

    # Center pixel
    u = W // 2  # 320
    v = H // 2  # 240

    result = pixel_to_camera_frame(u, v, depth, hfov_deg=90.0)

    assert result.is_valid, f"Center pixel should be valid: {result.failure_reason}"
    assert result.depth_value == 2.0

    # Center pixel should map to (0, 0, -d) in camera frame
    point = result.world_point
    assert abs(point[0]) < 0.01, f"Center X should be ~0, got {point[0]}"
    assert abs(point[1]) < 0.01, f"Center Y should be ~0, got {point[1]}"
    assert abs(point[2] + 2.0) < 0.01, f"Center Z should be -2, got {point[2]}"


def test_pixel_to_camera_depth_validation():
    """Test depth validation rejects invalid values."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.vlm.projection import pixel_to_camera_frame

    H, W = 480, 640

    # Test cases: (depth_value, expected_valid, expected_reason)
    cases = [
        (0.0, False, "depth_invalid"),
        (-1.0, False, "depth_invalid"),
        (np.nan, False, "depth_invalid"),
        (np.inf, False, "depth_invalid"),
        (0.2, False, "depth_too_close"),
        (15.0, False, "depth_too_far"),
        (1.5, True, None),
        (5.0, True, None),
    ]

    for depth_val, expected_valid, expected_reason in cases:
        depth = np.full((H, W), depth_val, dtype=np.float32)
        result = pixel_to_camera_frame(W // 2, H // 2, depth)

        assert result.is_valid == expected_valid, \
            f"Depth {depth_val}: expected valid={expected_valid}, got {result.is_valid}"

        if not expected_valid:
            assert result.failure_reason == expected_reason, \
                f"Depth {depth_val}: expected reason={expected_reason}, got {result.failure_reason}"


def test_pixel_to_camera_bounds_check():
    """Test out-of-bounds pixel coordinates."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.vlm.projection import pixel_to_camera_frame

    H, W = 480, 640
    depth = np.ones((H, W), dtype=np.float32) * 2.0

    # Out of bounds cases
    oob_cases = [
        (-1, 240),
        (640, 240),
        (320, -1),
        (320, 480),
        (1000, 1000),
    ]

    for u, v in oob_cases:
        result = pixel_to_camera_frame(u, v, depth)
        assert not result.is_valid, f"({u}, {v}) should be out of bounds"
        assert result.failure_reason == "pixel_out_of_bounds"


def test_camera_to_world_frame_identity():
    """Test camera to world transform with identity rotation."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.vlm.projection import camera_to_world_frame

    # Camera point: [1, 0, -2] (1m right, 2m forward in camera frame)
    camera_point = np.array([1.0, 0.0, -2.0], dtype=np.float64)

    # Agent at origin with identity rotation (w=1, x=0, y=0, z=0)
    agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    agent_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # identity quaternion
    sensor_height = 1.5

    world_point = camera_to_world_frame(camera_point, agent_pos, agent_rot, sensor_height)

    # With identity rotation, world point should be camera_point + [0, sensor_height, 0]
    expected = np.array([1.0, 1.5, -2.0], dtype=np.float64)
    np.testing.assert_allclose(world_point, expected, atol=1e-6)


def test_pixel_to_world_integration():
    """Test full pipeline: pixel -> camera -> world."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.vlm.projection import pixel_to_world

    H, W = 480, 640
    depth = np.ones((H, W), dtype=np.float32) * 3.0

    # Center pixel
    u, v = W // 2, H // 2

    # Agent at [5, 0, 5] with identity rotation
    agent_pos = np.array([5.0, 0.0, 5.0], dtype=np.float64)
    agent_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    result = pixel_to_world(u, v, depth, agent_pos, agent_rot)

    assert result.is_valid, f"Integration test failed: {result.failure_reason}"
    assert result.depth_value == 3.0

    # Center pixel at depth 3m, with identity rotation:
    # camera frame: [0, 0, -3] -> world: [5, 1.5, 2]
    # (agent at [5, 0, 5] + sensor height 1.5 + camera Z=-3)
    expected = np.array([5.0, 1.5, 2.0], dtype=np.float64)
    np.testing.assert_allclose(result.world_point, expected, atol=0.1)


if __name__ == "__main__":
    test_pixel_to_camera_center()
    test_pixel_to_camera_depth_validation()
    test_pixel_to_camera_bounds_check()
    test_camera_to_world_frame_identity()
    test_pixel_to_world_integration()
    print("All projection unit tests passed!")
