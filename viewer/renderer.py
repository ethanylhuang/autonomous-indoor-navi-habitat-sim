"""Observation encoding for browser delivery.

Stateless functions that convert raw numpy observations to browser-friendly
byte buffers (JPEG for RGB, PNG for depth colormap and top-down view).
Uses OpenCV (cv2) for image encoding.
"""

import math

import cv2
import numpy as np
from numpy.typing import NDArray


def encode_rgb_jpeg(rgba: NDArray[np.uint8], quality: int = 80) -> bytes:
    """Encode an RGBA image as JPEG bytes (drops alpha channel).

    Args:
        rgba: [H, W, 4] uint8 RGBA image.
        quality: JPEG quality 0-100.

    Returns:
        JPEG-encoded bytes.
    """
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def colorize_depth(
    depth: NDArray[np.float32],
    min_depth: float = 0.0,
    max_depth: float = 10.0,
) -> bytes:
    """Colorize a depth map using TURBO colormap and encode as JPEG.

    Args:
        depth: [H, W] float32 depth in meters.
        min_depth: Minimum depth for normalization.
        max_depth: Maximum depth for normalization.

    Returns:
        JPEG-encoded bytes of the colorized depth.
    """
    # Clamp and normalize to 0-255
    clamped = np.clip(depth, min_depth, max_depth)
    if max_depth > min_depth:
        normalized = ((clamped - min_depth) / (max_depth - min_depth) * 255).astype(
            np.uint8
        )
    else:
        normalized = np.zeros_like(depth, dtype=np.uint8)

    # Apply TURBO colormap
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    _, buf = cv2.imencode(".jpg", colored, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def render_topdown_view(
    navmesh_grid: NDArray[np.bool_],
    agent_position: NDArray[np.float64],
    agent_rotation: NDArray[np.float64],
    navmesh_bounds: tuple,
    meters_per_pixel: float = 0.05,
    canvas_size: int = 400,
) -> bytes:
    """Render a top-down NavMesh view with agent position and heading.

    Args:
        navmesh_grid: 2D bool array (True = navigable).
        agent_position: [3] world-frame position.
        agent_rotation: [4] quaternion [w, x, y, z].
        navmesh_bounds: (lower_bound, upper_bound), each [3] arrays.
        meters_per_pixel: Resolution of the navmesh grid.
        canvas_size: Output image side length in pixels.

    Returns:
        PNG-encoded bytes.
    """
    lower, upper = navmesh_bounds

    # Create grayscale image from navmesh grid
    grid_img = np.where(navmesh_grid, 220, 40).astype(np.uint8)

    # Resize to canvas
    canvas = cv2.resize(
        grid_img, (canvas_size, canvas_size), interpolation=cv2.INTER_NEAREST
    )
    # Convert to BGR for colored overlays
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # Guard against zero-range bounds
    x_range = max(upper[0] - lower[0], 1e-6)
    z_range = max(upper[2] - lower[2], 1e-6)

    # Project agent world XZ to pixel coords
    agent_px_x = int((agent_position[0] - lower[0]) / x_range * canvas_size)
    agent_px_y = int((agent_position[2] - lower[2]) / z_range * canvas_size)

    # Clamp to canvas bounds
    agent_px_x = max(0, min(canvas_size - 1, agent_px_x))
    agent_px_y = max(0, min(canvas_size - 1, agent_px_y))

    # Draw agent as red circle
    cv2.circle(canvas, (agent_px_x, agent_px_y), 6, (0, 0, 255), -1)

    # Compute heading from quaternion (yaw around Y axis)
    # For quaternion [w, x, y, z], yaw = atan2(2(wy + xz), 1 - 2(y^2 + z^2))
    # But in habitat-sim Y-up coords, heading in XZ plane:
    w, x, y, z = agent_rotation
    # Agent forward is -Z in local frame. Yaw from quaternion:
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # Heading arrow: agent forward is -Z, so in pixel coords:
    # pixel_x maps to world X, pixel_y maps to world Z
    # heading direction in world XZ: (-sin(yaw), -cos(yaw))
    arrow_len = 20
    dx = -math.sin(yaw) * arrow_len * (canvas_size / x_range) * meters_per_pixel
    dy = -math.cos(yaw) * arrow_len * (canvas_size / z_range) * meters_per_pixel
    end_x = int(agent_px_x + dx)
    end_y = int(agent_px_y + dy)

    # Draw heading arrow in green
    cv2.arrowedLine(
        canvas,
        (agent_px_x, agent_px_y),
        (end_x, end_y),
        (0, 255, 0),
        2,
        tipLength=0.3,
    )

    _, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()
