# M2 Architect Artifact -- Perception Pipeline

## Problem Restatement

Design the perception layer that converts raw sensor observations from M1's `Vehicle.step()` into actionable representations for M3's navigation stack. M2 consumes the `Observations` dataclass (forward RGB, rear RGB, depth, IMU) and produces: a 3D point cloud from simulated LiDAR, visual odometry pose estimates from consecutive RGB frames, semantic obstacle detections from cameras, and a fused 2D occupancy grid that serves as the shared world representation for M3's global and local planners. The existing IMU simulation must be evaluated for M3 readiness and extended if needed. The browser viewer must be updated to visualize all new M2 outputs.

---

## ASSUMPTIONS

1. **M1 is code-complete and verified.** `Vehicle.step()` returns an `Observations` dataclass with the shapes/types defined in `src/vehicle.py`. The viewer streams frames over WebSocket.

2. **Depth sensor is co-located with forward RGB camera.** Both face forward at position `[0.0, 1.5, 0.0]` with HFOV=90 degrees. Depth values are in meters (float32). There is no rear-facing depth sensor.

3. **No equirectangular depth sensor.** The current depth sensor is a standard pinhole camera (480x640, 90-degree HFOV). LiDAR simulation must work from this single forward-facing depth image, not a 360-degree panoramic depth. For full-surround coverage, we can either (a) add additional depth sensors at other orientations, or (b) accumulate point clouds across multiple steps as the agent rotates. **Decision: add a rear depth sensor to match rear RGB, giving 180-degree coverage (front + back). Full 360 is out of scope for M2.**

4. **habitat-sim has a built-in semantic sensor.** Adding a `CameraSensorSpec` with `sensor_type = SensorType.SEMANTIC` returns a `[H, W]` uint32 array where each pixel contains a semantic object ID. This requires the scene to have semantic annotations. The bundled test scenes (`skokloster-castle.glb`) have limited or no semantic annotations. **Assumption: we implement the sensor pipeline and detector code, but accept that test scenes may return all-zero semantic maps. Validation uses synthetic test data or scenes with known annotations.**

5. **Coordinate system is Y-up, right-handed (habitat-sim default).** Camera forward is -Z in agent-local frame. The depth image measures distance along the camera's -Z axis (z-buffer depth), not radial Euclidean distance. This matters for point cloud projection.

6. **Discrete actions with fixed step size.** `move_forward` = 0.25m, `turn_left`/`turn_right` = 10 degrees. Each `sim.step()` is one discrete tick. The IMU `dt=1.0` represents one step, not one second. All temporal computations use step-based timing.

7. **Python 3.9+ with numpy, scipy, cv2 available.** ORB feature detection and matching are available in `cv2`. No external deep learning framework needed for M2.

8. **Occupancy grid resolution = 5cm per cell.** This matches the existing navmesh `meters_per_pixel=0.05` in `Vehicle.get_topdown_navmesh()`. Grid is XZ-plane (bird's-eye), Y is used only for height filtering.

9. **Vehicle owns all sensor access.** M2 perception modules consume numpy arrays from `Observations`, never touch `habitat_sim.Simulator` directly. The `Vehicle` class remains the sole sim facade.

10. **No real-time constraint.** Perception can take 10-50ms per step. The viewer operates at manual driving speed (1-3 actions/sec). Optimization is deferred.

11. **IMU from M1 is sufficient for M3's EKF.** The current `SimulatedIMU` produces world-frame linear acceleration and angular velocity. M3's EKF needs body-frame IMU readings. **M2 will add a body-frame conversion method** to the existing `SimulatedIMU` class rather than rewriting it.

12. **cv2 ORB is sufficient for visual odometry.** Feature-based VO (ORB + BFMatcher + Essential Matrix decomposition) is preferred over dense optical flow for its lower computational cost and better rotation handling.

---

## IN_SCOPE

### New Files
- `src/sensors/lidar.py` -- Depth image to 3D point cloud conversion
- `src/perception/__init__.py` -- Package init
- `src/perception/occupancy_grid.py` -- Fused 2D bird's-eye occupancy grid
- `src/perception/visual_odometry.py` -- Feature-based VO from consecutive RGB frames
- `src/perception/obstacle_detector.py` -- Semantic-sensor-based obstacle detection
- `tests/test_lidar.py` -- LiDAR point cloud validation
- `tests/test_occupancy_grid.py` -- Occupancy grid tests
- `tests/test_visual_odometry.py` -- VO tests
- `tests/test_obstacle_detector.py` -- Obstacle detector tests

### Modified Files
- `configs/sensor_rig.py` -- Add `rear_depth_spec()` and `forward_semantic_spec()` factories, update `all_sensor_specs()`
- `src/sensors/imu.py` -- Add `to_body_frame()` method on `IMUReading` + expose quaternion rotation utility
- `src/vehicle.py` -- Extend `Observations` dataclass with `rear_depth`, `forward_semantic`, `rear_semantic` fields
- `viewer/server.py` -- Add M2 visualization data to WebSocket frame
- `viewer/renderer.py` -- Add point cloud bird's-eye renderer, occupancy grid renderer, semantic overlay renderer, VO trajectory renderer
- `viewer/static/index.html` -- Add M2 visualization panels
- `viewer/static/app.js` -- Handle new visualization data fields

### Unchanged
- `configs/sim_config.py` -- No changes needed (picks up new sensors from `all_sensor_specs()` automatically)

---

## OUT_OF_SCOPE

- **360-degree LiDAR** -- Two depth sensors (front + rear) give 180-degree coverage. Full surround requires 4+ sensors or agent rotation accumulation; deferred.
- **Learned semantic models** -- M2 uses habitat-sim's built-in semantic sensor only. CNN/transformer-based detection is M4+ territory.
- **Temporal filtering / map persistence** -- Occupancy grid is rebuilt each step from current observations. Persistent mapping (SLAM) is M3+ if needed.
- **Loop closure / bundle adjustment** -- VO provides frame-to-frame relative pose only. Global consistency is the EKF's job in M3.
- **Point cloud downsampling / voxel filtering** -- Not needed at current resolution (480x640 = 307K points max). Add in M3 if performance requires.
- **Continuous actions** -- Remain discrete.
- **Multi-scene validation** -- Test against bundled scenes only.
- **State estimation (EKF)** -- M3. M2 outputs are designed to feed into it.

---

## DESIGN

### 1. File Map and Dependency Graph

```
configs/
  sensor_rig.py         -- MODIFIED: +rear_depth_spec(), +forward_semantic_spec(), +rear_semantic_spec()
                           (no new deps, still pure data + habitat_sim)

src/
  sensors/
    imu.py              -- MODIFIED: +IMUReading.to_body_frame(), +rotation helpers
    lidar.py            -- NEW: depth_to_point_cloud(), depends on numpy only

  perception/
    __init__.py          -- NEW: empty
    occupancy_grid.py    -- NEW: OccupancyGrid class, depends on numpy
    visual_odometry.py   -- NEW: VisualOdometry class, depends on numpy + cv2
    obstacle_detector.py -- NEW: ObstacleDetector class, depends on numpy

  vehicle.py            -- MODIFIED: Observations gains rear_depth, forward_semantic, rear_semantic

viewer/
  server.py             -- MODIFIED: encode + send M2 data
  renderer.py           -- MODIFIED: +render_point_cloud_bev(), +render_occupancy_grid(),
                           +render_semantic_overlay(), +render_vo_trajectory()
  static/
    index.html           -- MODIFIED: new visualization panels
    app.js               -- MODIFIED: handle new fields

tests/
  test_lidar.py          -- NEW
  test_occupancy_grid.py -- NEW
  test_visual_odometry.py -- NEW
  test_obstacle_detector.py -- NEW
```

Dependency direction remains acyclic:
```
viewer/ --> src/vehicle.py --> configs/ + src/sensors/
                          `--> src/perception/ --> src/sensors/lidar.py (for point cloud)
                                               `--> numpy, cv2
```

Perception modules are **stateless processors** (except VO which keeps the previous frame). They consume numpy arrays, never touch `habitat_sim.Simulator`.

### 2. configs/sensor_rig.py -- Additions

```python
# New constants
SEMANTIC_RESOLUTION: Tuple[int, int] = (480, 640)  # match RGB resolution for pixel alignment

# New factory functions
def rear_depth_spec() -> CameraSensorSpec:
    """Rear-facing depth sensor, co-located with rear_rgb."""
    # uuid="rear_depth", orientation=[0.0, pi, 0.0], SensorType.DEPTH

def forward_semantic_spec() -> CameraSensorSpec:
    """Forward-facing semantic sensor, co-located with forward_rgb."""
    # uuid="forward_semantic", SensorType.SEMANTIC, same position/orientation as forward_rgb

def rear_semantic_spec() -> CameraSensorSpec:
    """Rear-facing semantic sensor, co-located with rear_rgb."""
    # uuid="rear_semantic", SensorType.SEMANTIC, orientation=[0.0, pi, 0.0]

def all_sensor_specs() -> List[CameraSensorSpec]:
    """All sensor specs for M1+M2."""
    return [
        forward_rgb_spec(),
        rear_rgb_spec(),
        depth_spec(),           # forward depth (M1)
        rear_depth_spec(),      # NEW
        forward_semantic_spec(),# NEW
        rear_semantic_spec(),   # NEW
    ]
```

Design notes:
- Semantic sensors share position/orientation with their corresponding RGB cameras for pixel-aligned fusion.
- Resolution matches RGB so that semantic labels can be directly overlaid on RGB images.
- `sim_config.py` needs no changes because `make_agent_config()` calls `all_sensor_specs()`.

### 3. src/sensors/lidar.py -- Depth to Point Cloud

Purpose: Convert a depth image from a pinhole camera into a 3D point cloud in the camera frame, then optionally transform to world frame.

```python
@dataclass
class PointCloud:
    points: NDArray[np.float32]     # [N, 3] XYZ in the requested frame
    num_valid: int                  # count of non-infinite, non-NaN points

def depth_to_point_cloud(
    depth: NDArray[np.float32],     # [H, W] z-buffer depth in meters
    hfov_deg: float,                # horizontal field of view in degrees
    max_depth: float = 10.0,        # discard points beyond this range
) -> PointCloud:
    """Convert a pinhole depth image to a 3D point cloud in camera frame.

    Camera frame convention (habitat-sim):
      X = right, Y = up, Z = backward (camera looks along -Z)

    The depth image contains z-buffer depth (distance along -Z axis).

    Returns:
        PointCloud with points in camera-local frame.
    """

def transform_point_cloud(
    pc: PointCloud,
    position: NDArray[np.float64],      # [3] sensor world position
    rotation_quat: NDArray[np.float64], # [4] sensor world rotation [w,x,y,z]
) -> PointCloud:
    """Transform point cloud from camera frame to world frame."""

def merge_point_clouds(clouds: List[PointCloud]) -> PointCloud:
    """Concatenate multiple point clouds into one."""
```

**Depth-to-3D math (pinhole projection inverse):**

Given depth image `D[v, u]` (z-buffer depth = distance along camera -Z):
```
focal_length = W / (2 * tan(hfov / 2))
x = (u - cx) * D[v, u] / focal_length
y = -(v - cy) * D[v, u] / focal_length   # negate because pixel Y is down, camera Y is up
z = -D[v, u]                              # camera looks along -Z
```
where `cx = W/2`, `cy = H/2`.

**Valid point filtering:** Discard pixels where `depth <= 0`, `depth > max_depth`, or `depth` is NaN/inf.

**World frame transform:**
```
R = quaternion_to_rotation_matrix(rotation_quat)
points_world = (R @ points_camera.T).T + position
```

Data shapes at boundary:
- Input: `depth [480, 640] float32`, `hfov_deg float`, `max_depth float`
- Output: `PointCloud.points [N, 3] float32` where N <= 480*640 = 307,200

### 4. src/sensors/imu.py -- Modifications

Add to `IMUReading`:
```python
@dataclass
class IMUReading:
    linear_acceleration: NDArray[np.float64]  # [3] m/s^2, world frame
    angular_velocity: NDArray[np.float64]     # [3] rad/s, world frame
    timestamp_step: int

    def to_body_frame(self, rotation_quat: NDArray[np.float64]) -> "IMUReading":
        """Convert world-frame readings to body-frame using agent rotation.

        Body frame convention: X=right, Y=up, Z=backward (agent-local).

        Args:
            rotation_quat: [4] agent rotation quaternion [w, x, y, z] in world frame.

        Returns:
            New IMUReading with acceleration and angular velocity in body frame.
        """
```

Add module-level utility:
```python
def quat_to_rotation_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
```

This rotation matrix function is also needed by `lidar.py` for point cloud transforms. To avoid circular imports, place it in `src/sensors/imu.py` (already the quaternion utility home) and have `lidar.py` import from there.

**Why body frame matters for M3:** The EKF prediction step uses body-frame accelerometer readings to propagate position. World-frame acceleration requires knowing the current orientation to integrate correctly, creating a chicken-and-egg problem. Body-frame readings let the EKF integrate directly in the body frame and rotate to world frame using its own orientation estimate.

### 5. src/perception/visual_odometry.py -- Feature-Based VO

Purpose: Estimate relative pose (rotation + translation direction) between consecutive RGB frames using ORB feature matching and essential matrix decomposition.

```python
@dataclass
class VOEstimate:
    rotation: NDArray[np.float64]       # [3, 3] rotation matrix (frame t-1 -> frame t)
    translation_dir: NDArray[np.float64] # [3] unit vector, translation direction (scale unknown)
    num_inliers: int                    # RANSAC inliers
    is_valid: bool                      # True if enough inliers for reliable estimate
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
        """
        Computes camera intrinsic matrix from HFOV + resolution:
          fx = fy = W / (2 * tan(hfov/2))
          cx, cy = W/2, H/2
        """

    def update(
        self,
        rgb_frame: NDArray[np.uint8],  # [H, W, 4] RGBA or [H, W, 3] RGB
    ) -> VOEstimate:
        """Process new frame, estimate motion relative to previous frame.

        First call: stores frame, returns identity rotation + zero translation.
        Subsequent calls: ORB detect+match -> Essential matrix -> decompose -> choose correct pose.
        """

    def reset(self) -> None:
        """Clear previous frame. Next update() returns identity."""
```

**VO Pipeline (per update call):**
1. Convert RGBA to grayscale (`cv2.cvtColor`)
2. Detect ORB keypoints + descriptors on current frame
3. Match against previous frame's descriptors using `cv2.BFMatcher(cv2.NORM_HAMMING)` with ratio test (Lowe's ratio = 0.75)
4. If fewer than `min_inliers` matches, return `is_valid=False`
5. Extract matched point pairs as `[N, 2]` float32 arrays
6. `cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)`
7. `cv2.recoverPose(E, pts1, pts2, K)` returns R, t, inlier mask
8. If inlier count < `min_inliers`, return `is_valid=False`
9. Store current frame's keypoints + descriptors for next call
10. Return `VOEstimate(R, t_normalized, num_inliers, True, step)`

**Scale ambiguity:** Essential matrix decomposition gives translation direction but not magnitude. M3's EKF fuses VO rotation with IMU-derived displacement to resolve scale. This is a known limitation of monocular VO.

**Why not use rear camera too:** The forward camera sees where the agent is going (more informative for forward motion). Using both cameras for stereo VO would require known baseline + sync, which adds complexity for marginal gain in a simulation where IMU is perfect. Deferred.

Data shapes:
- Input: `[480, 640, 4] uint8` RGBA
- Output: `VOEstimate.rotation [3, 3] float64`, `translation_dir [3] float64`

### 6. src/perception/obstacle_detector.py -- Semantic Obstacle Detection

Purpose: Identify obstacle regions in camera images using habitat-sim's semantic sensor output.

```python
@dataclass
class ObstacleDetection:
    mask: NDArray[np.bool_]          # [H, W] True where obstacle detected
    object_ids: NDArray[np.uint32]   # [H, W] semantic object IDs
    obstacle_count: int              # number of distinct obstacle regions
    timestamp_step: int

# Configurable set of semantic IDs considered "obstacle"
# This is scene-dependent; default includes common indoor obstacles.
DEFAULT_OBSTACLE_IDS: Set[int] = set()  # populated per-scene or left empty for "everything is obstacle"

class ObstacleDetector:
    """Semantic-sensor-based obstacle detector.

    Uses habitat-sim's semantic sensor to identify obstacle pixels.
    In scenes without semantic annotations, falls back to treating all
    non-zero semantic IDs as potential obstacles.
    """

    def __init__(
        self,
        obstacle_ids: Optional[Set[int]] = None,
        min_obstacle_pixels: int = 100,  # ignore tiny detections
    ) -> None:

    def detect(
        self,
        semantic_image: NDArray[np.uint32],  # [H, W] semantic object IDs
        depth_image: NDArray[np.float32],     # [H, W] depth for range filtering
        max_range: float = 5.0,              # only detect within this range
    ) -> ObstacleDetection:
        """Detect obstacles in a single camera view.

        Steps:
        1. Filter semantic image: keep only pixels with IDs in obstacle_ids
           (or all non-zero if obstacle_ids is empty)
        2. Filter by depth range: discard pixels beyond max_range
        3. Apply minimum pixel count threshold
        4. Return binary mask + metadata
        """

    def detect_both_cameras(
        self,
        forward_semantic: NDArray[np.uint32],
        rear_semantic: NDArray[np.uint32],
        forward_depth: NDArray[np.float32],
        rear_depth: NDArray[np.float32],
        max_range: float = 5.0,
    ) -> Tuple[ObstacleDetection, ObstacleDetection]:
        """Detect obstacles from both cameras. Convenience method."""
```

**Why semantic sensor as ground truth:** The project plan specifies starting with habitat-sim's semantic sensor. This provides per-pixel object IDs without a trained model. The detector's job is to classify which IDs are obstacles, apply depth-range filtering, and produce a binary mask. When a learned model replaces the semantic sensor later, only the input changes (semantic segmentation output instead of habitat-sim semantic sensor); the rest of the pipeline is identical.

**Obstacle-to-grid projection:** The `ObstacleDetector` produces masks in image space. The `OccupancyGrid` (below) is responsible for projecting these detections into the 2D bird's-eye grid using the corresponding depth images + camera intrinsics.

Data shapes:
- Input: `semantic [480, 640] uint32`, `depth [480, 640] float32`
- Output: `ObstacleDetection.mask [480, 640] bool`, `object_ids [480, 640] uint32`

### 7. src/perception/occupancy_grid.py -- Fused 2D Occupancy Grid

Purpose: Build a local 2D bird's-eye occupancy grid by projecting LiDAR point clouds and semantic obstacle detections into a common grid.

```python
@dataclass
class OccupancyGridData:
    grid: NDArray[np.float32]        # [grid_h, grid_w] values in [0.0, 1.0]
                                     # 0.0 = free, 1.0 = occupied, 0.5 = unknown
    resolution: float                # meters per cell
    origin: NDArray[np.float64]      # [2] world XZ of grid[0, 0] corner
    shape: Tuple[int, int]           # (grid_h, grid_w)
    timestamp_step: int

class OccupancyGrid:
    """Local ego-centric occupancy grid fused from LiDAR + semantic detections.

    The grid is centered on the agent and represents a fixed-size local area.
    It is rebuilt each step (no temporal accumulation in M2).
    """

    def __init__(
        self,
        grid_size: float = 10.0,      # meters, total side length of square grid
        resolution: float = 0.05,      # meters per cell
        height_min: float = 0.1,       # ignore points below this (floor)
        height_max: float = 2.0,       # ignore points above this (ceiling)
        obstacle_confidence: float = 0.9,
        free_confidence: float = 0.1,
    ) -> None:
        """
        Grid dimensions: int(grid_size / resolution) x int(grid_size / resolution)
        Default: 200 x 200 cells covering 10m x 10m area.
        """

    def update(
        self,
        agent_position: NDArray[np.float64],   # [3] world frame
        agent_rotation: NDArray[np.float64],    # [4] quaternion [w,x,y,z]
        point_clouds: List[PointCloud],         # world-frame point clouds from LiDAR
        obstacle_detections: Optional[List[Tuple[ObstacleDetection, NDArray, NDArray, float]]] = None,
        # Each tuple: (detection, depth_image, sensor_rotation_quat, hfov_deg)
    ) -> OccupancyGridData:
        """Build the occupancy grid for the current step.

        Steps:
        1. Compute grid origin (agent-centered)
        2. Project LiDAR points:
           a. Filter by height (height_min < y < height_max)
           b. Convert world XZ to grid cell indices
           c. Mark cells as occupied
        3. Project semantic obstacle detections:
           a. For each obstacle pixel, use depth to compute 3D world position
           b. Filter by height, project to grid cells
           c. Mark as occupied (these catch glass/thin objects depth might miss)
        4. Ray-cast free space: cells between agent and occupied cells are free
        5. Return OccupancyGridData
        """

    def reset(self) -> None:
        """Clear any internal state (minimal for M2 since grid is rebuilt each step)."""
```

**Fusion strategy:**
- **LiDAR points** are the primary source: each point with height in `[height_min, height_max]` marks a cell as occupied.
- **Semantic detections** supplement LiDAR: obstacle mask pixels are back-projected to 3D using the corresponding depth image, then projected to the grid. This catches objects that depth sensors represent poorly (glass, mirrors) IF the semantic sensor detects them.
- **Free space marking:** Simple ray casting from agent center through each occupied cell. Cells along the ray that are closer than the occupied cell are marked free. This uses Bresenham's line algorithm on the grid.
- **Unknown:** Cells that are neither observed as free nor occupied remain at 0.5.

**Grid coordinate system:**
- Grid `[0, 0]` corresponds to `origin` in world XZ coordinates.
- Grid row index increases with world Z, column index increases with world X.
- `grid[row, col]` where `row = int((world_z - origin_z) / resolution)`, `col = int((world_x - origin_x) / resolution)`.

Data shapes:
- Output: `OccupancyGridData.grid [200, 200] float32` (default 10m at 5cm resolution)

### 8. src/vehicle.py -- Modifications

Extend the `Observations` dataclass:

```python
@dataclass
class Observations:
    # M1 fields (unchanged)
    forward_rgb: NDArray[np.uint8]       # [480, 640, 4] RGBA
    rear_rgb: NDArray[np.uint8]          # [480, 640, 4] RGBA
    depth: NDArray[np.float32]           # [480, 640] meters (forward)
    imu: IMUReading
    state: VehicleState

    # M2 additions
    rear_depth: NDArray[np.float32]      # [480, 640] meters (rear)
    forward_semantic: NDArray[np.uint32] # [480, 640] semantic IDs (forward)
    rear_semantic: NDArray[np.uint32]    # [480, 640] semantic IDs (rear)
```

Modify `_build_observations()` to extract the three new sensor outputs from `raw_obs`:
```python
rear_depth=raw_obs["rear_depth"],
forward_semantic=raw_obs["forward_semantic"],
rear_semantic=raw_obs["rear_semantic"],
```

**No perception pipeline execution in Vehicle.** The `Vehicle` class remains a thin sim facade. Perception modules are called by the viewer/scripts layer, not by `Vehicle.step()`. This keeps `Vehicle` fast and gives callers control over which perception modules to run.

### 9. Perception Orchestration Pattern

The perception pipeline is NOT embedded in `Vehicle`. Instead, the caller (viewer server or script) orchestrates it:

```python
# In viewer/server.py or scripts/run_classical.py:

from src.sensors.lidar import depth_to_point_cloud, transform_point_cloud, merge_point_clouds
from src.perception.visual_odometry import VisualOdometry
from src.perception.obstacle_detector import ObstacleDetector
from src.perception.occupancy_grid import OccupancyGrid
from configs.sensor_rig import HFOV, SENSOR_HEIGHT

# Initialize once
vo = VisualOdometry(hfov_deg=HFOV, resolution=(480, 640))
detector = ObstacleDetector()
grid = OccupancyGrid()

# Per step:
obs = vehicle.step(action)

# 1. LiDAR: depth -> point clouds -> world frame
fwd_pc = depth_to_point_cloud(obs.depth, HFOV)
fwd_pc_world = transform_point_cloud(fwd_pc, obs.state.position, obs.state.rotation)
rear_pc = depth_to_point_cloud(obs.rear_depth, HFOV)
# rear sensor rotation = agent rotation * 180deg Y offset
rear_pc_world = transform_point_cloud(rear_pc, obs.state.position, rear_sensor_world_rot)
merged_pc = merge_point_clouds([fwd_pc_world, rear_pc_world])

# 2. VO: forward RGB -> relative pose
vo_estimate = vo.update(obs.forward_rgb)

# 3. Semantic obstacles
fwd_det, rear_det = detector.detect_both_cameras(
    obs.forward_semantic, obs.rear_semantic,
    obs.depth, obs.rear_depth,
)

# 4. Occupancy grid: fuse LiDAR + semantic
grid_data = grid.update(
    obs.state.position, obs.state.rotation,
    [fwd_pc_world, rear_pc_world],
    obstacle_detections=[(fwd_det, obs.depth, obs.state.rotation, HFOV), ...]
)
```

**Why not in Vehicle:** Perception is computationally expensive and not always needed. The viewer may want to skip VO during manual driving. M3's autonomous loop needs all modules. M4's RL policy may bypass classical perception entirely. Keeping perception external to Vehicle maintains flexibility.

### 10. Viewer Updates

#### 10a. viewer/renderer.py -- New Functions

```python
def render_point_cloud_bev(
    point_cloud: PointCloud,
    agent_position: NDArray[np.float64],
    view_range: float = 10.0,        # meters, total side length
    canvas_size: int = 400,
) -> bytes:
    """Render point cloud as bird's-eye-view image (PNG).

    Projects 3D points onto XZ plane. Agent at center.
    Color encodes height (Y value) using a colormap.
    """

def render_occupancy_grid(
    grid_data: OccupancyGridData,
    agent_position: NDArray[np.float64],
    agent_rotation: NDArray[np.float64],
    canvas_size: int = 400,
) -> bytes:
    """Render occupancy grid as colored image (PNG).

    Color scheme: green=free, red=occupied, gray=unknown.
    Agent position + heading arrow overlaid.
    """

def render_semantic_overlay(
    rgb: NDArray[np.uint8],          # [H, W, 4] RGBA
    semantic: NDArray[np.uint32],    # [H, W] semantic IDs
    obstacle_mask: NDArray[np.bool_], # [H, W] True where obstacle
    alpha: float = 0.4,
) -> bytes:
    """Overlay obstacle detections on RGB image as semi-transparent red (JPEG)."""

def render_vo_trajectory(
    positions: List[NDArray[np.float64]],  # list of [3] world positions
    current_position: NDArray[np.float64],
    navmesh_bounds: tuple,
    canvas_size: int = 400,
) -> bytes:
    """Render VO-estimated trajectory on top-down canvas (PNG).

    Shows breadcrumb trail of estimated positions.
    """
```

#### 10b. viewer/server.py -- Changes

Add perception module instances alongside the Vehicle in the lifespan:

```python
# In lifespan:
_vo = VisualOdometry(...)
_detector = ObstacleDetector(...)
_grid = OccupancyGrid(...)
_vo_positions = []  # accumulate for trajectory rendering

# In _build_frame():
# After getting obs, run perception pipeline, add to frame:
frame["point_cloud_bev"] = base64.b64encode(render_point_cloud_bev(...)).decode("ascii")
frame["occupancy_grid"] = base64.b64encode(render_occupancy_grid(...)).decode("ascii")
frame["semantic_fwd"] = base64.b64encode(render_semantic_overlay(...)).decode("ascii")
frame["vo_trajectory"] = base64.b64encode(render_vo_trajectory(...)).decode("ascii")
```

Perception execution happens inside `_build_frame()`. This adds ~20-50ms per frame, acceptable at manual driving speed.

#### 10c. viewer/static/index.html -- Layout Update

Add a second row of panels below the existing bottom row:

```
+----------------------------------------------------------+
|  Forward RGB       |  Rear RGB        |  Depth (color)    |    <- M1 (unchanged)
+----------------------------------------------------------+
|  Top-Down NavMesh  |  Agent State                         |    <- M1 (unchanged)
+----------------------------------------------------------+
|  Point Cloud BEV   |  Occupancy Grid  |  Semantic Overlay |    <- M2 NEW
+----------------------------------------------------------+
|  VO Trajectory     |  M2 Stats (VO inliers, grid cells)  |    <- M2 NEW
+----------------------------------------------------------+
```

New `<img>` elements:
- `id="pointCloudBev"` -- point cloud bird's-eye view
- `id="occGrid"` -- occupancy grid
- `id="semanticFwd"` -- forward RGB with semantic overlay
- `id="voTrajectory"` -- VO trajectory trail

New state display fields:
- VO inliers count, VO validity
- Occupied cell count, free cell count
- Obstacle pixel count (forward + rear)

#### 10d. viewer/static/app.js -- Changes

In `updateFrame()`, add handling for new base64 image fields and state fields. Same pattern as existing: `data.point_cloud_bev`, `data.occupancy_grid`, etc.

### 11. Coordinate Frame Reference

All modules must be explicit about which frame they operate in. This table is the single source of truth:

| Frame | Convention | Used By |
|-------|-----------|---------|
| **World** | Y-up, right-handed. Origin at scene center. | `VehicleState.position/rotation`, occupancy grid, NavMesh |
| **Agent body** | X=right, Y=up, Z=backward (facing -Z). Origin at agent center. | IMU body-frame readings |
| **Camera** | X=right, Y=up, Z=backward (looking along -Z). Origin at sensor mount point. | `depth_to_point_cloud()` output (before transform) |
| **Image** | Row=down (v), Col=right (u). Origin at top-left. | All sensor images `[H, W, ...]` |
| **Grid** | Row increases with world +Z, Col increases with world +X. | `OccupancyGridData.grid[row, col]` |

**Key transforms:**
- Camera -> World: `R_world_cam @ p_cam + t_sensor_world` (rotation from agent rotation + sensor mounting offset)
- World XZ -> Grid cell: `col = (x - origin_x) / resolution`, `row = (z - origin_z) / resolution`
- Forward camera rotation = agent rotation (both face -Z in agent frame)
- Rear camera rotation = agent rotation * `quat_from_axis_angle(Y, pi)` (180-degree Y offset)

### 12. Data Flow Summary (Full Pipeline)

```
habitat_sim.step(action)
    |
    v
raw_obs dict {forward_rgb, rear_rgb, depth, rear_depth, forward_semantic, rear_semantic}
    |
    v
Vehicle._build_observations()  -->  Observations dataclass
    |
    +---> depth [480,640] float32 -----> depth_to_point_cloud() --> PointCloud [N,3] float32
    |                                         |
    +---> rear_depth [480,640] float32 ----> depth_to_point_cloud() --> PointCloud [M,3] float32
    |                                         |
    |     transform_point_cloud() x2          |
    |     merge_point_clouds()                |
    |          |                              |
    |          v                              |
    |     merged PointCloud [N+M, 3] float32  |
    |          |                              |
    +---> forward_rgb [480,640,4] uint8       |
    |         |                               |
    |         v                               |
    |     VisualOdometry.update()             |
    |         |                               |
    |         v                               |
    |     VOEstimate (R[3,3], t[3], inliers)  |
    |                                         |
    +---> forward_semantic [480,640] uint32   |
    |     rear_semantic [480,640] uint32      |
    |         |                               |
    |         v                               |
    |     ObstacleDetector.detect_both()      |
    |         |                               |
    |         v                               |
    |     (ObstacleDetection, ObstacleDetection)  -- masks [480,640] bool
    |                                         |
    +-------> all outputs feed into:          |
              OccupancyGrid.update()          |
                  |                           |
                  v                           |
              OccupancyGridData               |
                grid [200,200] float32        |
                resolution: 0.05              |
                origin: [2] float64           |

TO M3:
  - OccupancyGridData --> local_planner (DWA), global_planner (cost overlay)
  - VOEstimate --> state_estimation/estimator.py (EKF measurement)
  - IMUReading.to_body_frame() --> state_estimation/estimator.py (EKF prediction)
  - PointCloud (optional) --> local_planner for fine-grained obstacle avoidance
```

### 13. Test Design

**tests/test_lidar.py** (8 cases):
- `depth_to_point_cloud` with uniform depth returns correct point count
- Points at image center have Z = -depth (camera looks along -Z)
- Points at image edges have correct X/Y spread from HFOV
- Zero/NaN/inf depth pixels are excluded
- `max_depth` filtering works
- `transform_point_cloud` with identity rotation preserves points (up to translation)
- `transform_point_cloud` with 90-degree Y rotation rotates X<->Z correctly
- `merge_point_clouds` concatenates and updates `num_valid`

**tests/test_occupancy_grid.py** (7 cases):
- Grid has correct shape for given `grid_size` and `resolution` (200x200 default)
- Single point at known XZ position marks correct grid cell as occupied
- Points outside grid bounds are ignored (no IndexError)
- Points outside height range are filtered
- Free space ray casting marks cells between agent and obstacle as free
- Unknown cells remain at 0.5
- Grid origin is correctly centered on agent

**tests/test_visual_odometry.py** (6 cases):
- First frame returns identity rotation and zero translation
- Identical consecutive frames return identity (no motion detected) -- may have low inliers
- Synthetic shifted image produces `is_valid=True` with correct translation direction
- `reset()` causes next call to return identity
- `num_inliers` is non-negative integer
- Graceful handling of featureless images (returns `is_valid=False`)

**tests/test_obstacle_detector.py** (6 cases):
- All-zero semantic image returns empty mask (no obstacles)
- Non-zero semantic IDs are detected as obstacles
- Depth range filtering excludes far objects
- `min_obstacle_pixels` threshold filters small detections
- Custom `obstacle_ids` set restricts which IDs are obstacles
- `detect_both_cameras` returns two independent detections

### 14. Requirements Updates

Add to `requirements.txt`:
```
scipy>=1.10.0          # for spatial transforms if needed; also M3 EKF prep
```

No other new dependencies. `cv2` (already present for M1 renderer) provides ORB, BFMatcher, findEssentialMat, recoverPose. `numpy` provides all linear algebra.

---

## RISKS

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| R1 | **Semantic sensor returns all-zero on test scenes.** Bundled test scenes likely lack semantic annotations. Obstacle detector produces no detections, reducing grid quality. | HIGH | Test with synthetic data (manually constructed semantic arrays). Document that full obstacle detection requires annotated scenes (Replica/HM3D). Occupancy grid still functions from LiDAR alone. |
| R2 | **VO scale ambiguity.** Essential matrix gives rotation + translation direction, not magnitude. VO cannot provide metric displacement alone. | MEDIUM | Documented as expected. M3's EKF fuses VO rotation with IMU displacement. For M2 validation, compare VO rotation against ground truth from `VehicleState.rotation` delta. |
| R3 | **VO fails on textureless surfaces.** Indoor scenes with plain walls produce few ORB features. | MEDIUM | Set `min_inliers` threshold. Return `is_valid=False` gracefully. M3's EKF falls back to IMU-only when VO is invalid. Increase `max_features` to 2000 if default fails. |
| R4 | **Depth z-buffer vs. Euclidean distance.** If habitat-sim returns Euclidean distance instead of z-buffer depth, the pinhole projection math is wrong. | HIGH | Verify experimentally: depth at image center should equal Euclidean distance (z-buffer and Euclidean agree at center). Depth at edges will differ if z-buffer. Test with known scene geometry. Add a flag `is_euclidean_depth` to `depth_to_point_cloud()` to handle both cases. |
| R5 | **Occupancy grid size vs. scene size mismatch.** 10m grid may be too small for large scenes or too large for small rooms. | LOW | Grid size is configurable. Default 10m covers typical indoor rooms. M3 can adjust based on scene bounds. |
| R6 | **Rear depth sensor adds ~5ms per frame.** Extra sensor increases sim.step() time by ~15-20%. | LOW | Acceptable at manual driving speed. Profile and optimize only if needed. |
| R7 | **ORB is non-deterministic.** OpenCV's ORB may produce slightly different keypoints across runs. | LOW | Tests use relative thresholds (inlier count > 0) rather than exact values. Seed OpenCV RNG if reproducibility needed. |
| R8 | **Adding fields to `Observations` dataclass breaks M1 tests.** Existing tests construct or inspect `Observations` by position. | MEDIUM | M1 tests access fields by name, not position. Adding new fields at the end of the dataclass is safe. Verify all M1 tests still pass after modification. |
| R9 | **Viewer payload size increases.** Four new base64 images add ~200-400KB per frame. | LOW | Total frame ~400-600KB. Acceptable for localhost. Compress point cloud BEV aggressively (JPEG quality=60). Add a toggle in the viewer to enable/disable M2 panels. |
| R10 | **`quat_to_rotation_matrix` in imu.py creates a coupling.** lidar.py imports from imu.py, which is semantically odd. | LOW | Acceptable for M2. If coupling becomes problematic, extract a `src/utils/transforms.py` module in M3. |

---

## ACCEPTANCE_CRITERIA

### AC1: LiDAR Point Cloud
- `depth_to_point_cloud()` with a `[480, 640] float32` depth image returns a `PointCloud` with `points` shape `[N, 3]` and dtype `float32`, where `N <= 307200`.
- Points at image center `(240, 320)` have `z = -depth_value` (within 1% tolerance).
- Points at image edges have correct angular spread: `|x| / |z| = tan(hfov/2)` at column 0 and column 639.
- NaN, inf, zero, and beyond-max-depth pixels are excluded from the point cloud.
- `transform_point_cloud()` with identity rotation and zero translation preserves all point coordinates.
- `transform_point_cloud()` with known rotation produces expected coordinate changes.

### AC2: IMU Body Frame
- `IMUReading.to_body_frame()` with identity rotation returns identical readings.
- `to_body_frame()` with a 90-degree Y rotation swaps X and Z components correctly.
- All dtypes remain `float64`, shapes remain `(3,)`.

### AC3: Visual Odometry
- First `update()` call returns `is_valid=True` with identity rotation (3x3) and zero translation (3,).
- Consecutive calls with real RGB frames from habitat-sim produce `VOEstimate` with `num_inliers >= 0`.
- When VO is valid, rotation matrix is orthogonal (`R @ R.T = I`) and `det(R) = 1.0`.
- `translation_dir` is a unit vector (`|t| = 1.0`) when `is_valid=True`.
- `reset()` causes next call to return identity.
- Featureless input (solid color image) returns `is_valid=False`.

### AC4: Obstacle Detector
- All-zero semantic image returns `obstacle_count=0` and all-False mask.
- Semantic image with known non-zero IDs returns correct mask coverage.
- Depth range filtering: obstacles at depth > `max_range` are excluded.
- `min_obstacle_pixels` correctly suppresses small detections.
- `detect_both_cameras()` returns two independent `ObstacleDetection` objects.

### AC5: Occupancy Grid
- Grid shape is `[200, 200]` for default parameters (10m at 5cm resolution).
- Grid values are in `[0.0, 1.0]`.
- A point cloud with one point at a known XZ position marks the correct grid cell as occupied (value > 0.5).
- Points outside the grid area do not cause errors.
- Points outside the height range `[height_min, height_max]` are filtered.
- Agent-centered: grid origin shifts with agent position.
- Free space cells (between agent and nearest obstacle along a ray) have value < 0.5.
- Unobserved cells have value = 0.5.

### AC6: Sensor Extension
- `Observations.rear_depth` has shape `[480, 640]` and dtype `float32`.
- `Observations.forward_semantic` has shape `[480, 640]` and dtype `uint32`.
- `Observations.rear_semantic` has shape `[480, 640]` and dtype `uint32`.
- All M1 tests continue to pass without modification.

### AC7: Viewer Integration
- Browser dashboard shows four new panels: Point Cloud BEV, Occupancy Grid, Semantic Overlay, VO Trajectory.
- All new panels update on each action (W/A/D keys).
- Occupancy grid shows green (free), red (occupied), gray (unknown) regions.
- Point cloud BEV shows height-colored dots centered on agent.
- Semantic overlay highlights obstacle regions in semi-transparent red on forward RGB.
- VO trajectory shows breadcrumb trail of estimated positions.
- New stats displayed: VO inlier count, VO validity, occupied cell count, obstacle pixel count.
- Reset (R key) clears VO trajectory and resets all perception state.

### AC8: M3 Interface Readiness
- `OccupancyGridData` is importable from `src.perception.occupancy_grid` and contains `grid`, `resolution`, `origin`, and `shape` fields as specified.
- `VOEstimate` is importable from `src.perception.visual_odometry` and contains `rotation`, `translation_dir`, `num_inliers`, and `is_valid` fields.
- `IMUReading.to_body_frame()` is callable and returns a new `IMUReading`.
- `PointCloud` is importable from `src.sensors.lidar`.
- All perception modules can be instantiated and called independently of the viewer.
