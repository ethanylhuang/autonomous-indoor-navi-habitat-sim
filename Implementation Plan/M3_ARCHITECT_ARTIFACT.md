# M3 Architect Artifact — Classical Navigation Stack

## ASSUMPTIONS

1. **M1 and M2 are code-complete and tested.** All sensor, perception, viewer, and vehicle modules are stable and their interfaces are frozen. M3 builds on top without modifying M1/M2 files beyond additive changes (new imports, new WebSocket message types in `server.py`, new UI panels in the viewer).

2. **Discrete action space only.** The controller maps planned velocities into sequences of `move_forward`, `turn_left`, `turn_right`. Step sizes from `AgentParams`: forward = 0.25m, turn = 10 degrees. No continuous actions.

3. **Ground-truth position is available** from `AgentState` for EKF initialization and evaluation metrics, but the EKF's *output* (fused IMU+VO estimate) is what the planner uses during navigation. This creates a "realistic" localization pipeline while allowing metric computation against ground truth.

4. **Single episode = one (start, goal) pair.** The agent navigates from a random navigable start to a random navigable goal. An episode ends on: goal reached, max steps exceeded, or stuck detection triggers.

5. **NavMesh `find_path()` may fail** (return empty waypoints or `inf` distance) if start/goal are on different NavMesh islands. The system must detect this and skip the episode.

6. **The occupancy grid from M2 is rebuilt each step** (no temporal accumulation). The DWA local planner uses the current-step grid for obstacle scoring. This is sufficient for static environments (test scenes have no moving objects).

7. **VO provides rotation only** (translation direction is unit vector, scale unknown). The EKF fuses VO rotation with IMU-derived displacement to form the state estimate. Scale comes from the IMU's position differencing (which itself comes from ground truth `AgentState` -- an acknowledged sim-specific simplification).

8. **"Rear camera collision avoidance"** means: when the occupancy grid shows obstacles behind the agent (within a close-range threshold), the local planner penalizes or blocks `move_forward` if the agent is heading toward those rear obstacles after a recent reversal. In practice with discrete forward-only motion, rear camera awareness triggers replanning rather than enabling explicit reversing.

9. **scipy is already in requirements.txt** (confirmed: `scipy>=1.10.0`). The EKF will use numpy for matrix operations; scipy is available for `scipy.spatial.transform.Rotation` if needed but we prefer our existing `transforms.py` helpers.

10. **Viewer integration is additive.** New autonomous navigation mode adds a new WebSocket message type (`start_nav`, `stop_nav`) alongside existing manual WASD control. The user can switch between manual and autonomous modes.

11. **SPL (Success weighted by Path Length)** is defined as: `success * (geodesic_distance / max(path_length, geodesic_distance))` per Habitat challenge convention.

12. **Max steps per episode** defaults to 500. With 0.25m per step, this allows ~125m of travel, far exceeding typical indoor geodesic distances.

---

## IN_SCOPE

- `src/state_estimation/estimator.py` -- EKF fusing IMU + VO into pose estimate
- `src/state_estimation/__init__.py` -- package init
- `src/planning/global_planner.py` -- NavMesh pathfinding wrapper with waypoint management
- `src/planning/local_planner.py` -- DWA local planner using occupancy grid
- `src/planning/__init__.py` -- package init
- `src/control/controller.py` -- converts planned velocity to discrete actions, detects goal arrival and stuck conditions
- `src/control/__init__.py` -- package init
- `scripts/run_classical.py` -- executes navigation episodes, computes metrics, prints summary
- Viewer updates: `viewer/server.py` (new autonomous nav WebSocket handler), `viewer/renderer.py` (waypoint rendering on top-down view), `viewer/static/index.html` (nav controls panel), `viewer/static/app.js` (nav mode toggle, goal display)
- `tests/test_estimator.py` -- EKF unit tests
- `tests/test_global_planner.py` -- global planner tests
- `tests/test_local_planner.py` -- DWA tests
- `tests/test_controller.py` -- controller tests
- `tests/test_navigation.py` -- integration test for full nav loop

---

## OUT_OF_SCOPE

- Continuous action space (M4)
- RL policy training or evaluation (M4)
- Multi-scene benchmarking suite (M5)
- Temporal occupancy grid accumulation (potential M3.5 enhancement)
- Dynamic obstacle avoidance (no moving objects in test scenes)
- SLAM / loop closure (EKF is filtering only, no graph optimization)
- Explicit reversing / multi-point-turn maneuvers (forward-only with turning)
- Click-on-NavMesh goal setting in viewer (can be added as optional follow-up)

---

## DESIGN

### 1. File Map and Dependency Graph

```
src/
  state_estimation/
    __init__.py            (empty)
    estimator.py           depends on: numpy, src/sensors/imu.py, src/perception/visual_odometry.py, src/utils/transforms.py
  planning/
    __init__.py            (empty)
    global_planner.py      depends on: numpy, src/vehicle.py (types only: Vehicle.find_path signature)
    local_planner.py       depends on: numpy, src/perception/occupancy_grid.py (OccupancyGridData type)
  control/
    __init__.py            (empty)
    controller.py          depends on: numpy, src/utils/transforms.py

scripts/
  run_classical.py         depends on: src/vehicle.py, all M2 perception modules, all M3 modules, configs/*

viewer/ (modifications)
  server.py                adds: navigation runner imports, new WebSocket message handlers
  renderer.py              adds: render_waypoints_on_topdown(), render_nav_status()
  static/index.html        adds: nav control panel (Start Nav, Stop Nav, goal display, metrics)
  static/app.js            adds: nav mode state, start/stop handlers, nav status display

tests/
  test_estimator.py
  test_global_planner.py
  test_local_planner.py
  test_controller.py
  test_navigation.py
```

Dependency direction:
```
scripts/run_classical.py
    --> src/control/controller.py
        --> src/planning/local_planner.py
            --> src/perception/occupancy_grid.py (OccupancyGridData)
        --> src/planning/global_planner.py
            --> src/vehicle.py (find_path)
        --> src/state_estimation/estimator.py
            --> src/sensors/imu.py (IMUReading)
            --> src/perception/visual_odometry.py (VOEstimate)
    --> src/vehicle.py
    --> all M2 perception modules
```

No circular dependencies. M3 modules are leaf consumers of M1/M2 data types.

### 2. src/state_estimation/estimator.py

**Purpose:** Extended Kalman Filter fusing IMU-derived displacement with VO-derived rotation to produce a robust 2D pose estimate (x, z, yaw). The vehicle operates on a floor plane, so the EKF state is 2D+heading.

**State vector:** `[x, z, yaw]` (3-dimensional). Position in world XZ plane, yaw about Y axis.

**Design rationale for 2D EKF:** Indoor navigation on a NavMesh is inherently planar. The Y coordinate is fixed by the NavMesh height. A 3D EKF would add complexity without benefit. The 2D formulation also avoids quaternion normalization issues in the state vector.

```python
@dataclass
class PoseEstimate:
    position: NDArray[np.float64]  # [3] world frame [x, y, z] (y from navmesh)
    yaw: float                     # radians, rotation about Y axis
    covariance: NDArray[np.float64]  # [3, 3] uncertainty in [x, z, yaw]
    timestamp_step: int

class EKFEstimator:
    """Extended Kalman Filter for 2D+heading pose estimation.

    State: [x, z, yaw]
    Predict: IMU-derived displacement (dx, dz) and rotation (dyaw)
    Update: VO-derived rotation (dyaw) when valid
    """

    def __init__(
        self,
        process_noise_pos: float = 0.01,    # meters^2 per step
        process_noise_yaw: float = 0.005,   # rad^2 per step
        measurement_noise_yaw: float = 0.02, # rad^2 per measurement
        floor_height: float = 0.0,          # Y coordinate of the floor
    ) -> None: ...

    def initialize(
        self,
        position: NDArray[np.float64],  # [3] initial world position
        yaw: float,                      # initial yaw
    ) -> PoseEstimate: ...

    def predict(
        self,
        imu_reading: IMUReading,
        dt: float,
    ) -> PoseEstimate:
        """Predict step using IMU-derived displacement.

        Extracts displacement from IMU acceleration (double integration over dt)
        or, more practically, from the IMU's velocity estimate.
        Updates state mean and covariance.
        """
        ...

    def update_vo(
        self,
        vo_estimate: VOEstimate,
    ) -> PoseEstimate:
        """Update step using VO rotation measurement.

        Extracts yaw change from VO rotation matrix.
        Only applied when vo_estimate.is_valid is True.
        """
        ...

    def get_estimate(self) -> PoseEstimate: ...

    def reset(self) -> None: ...
```

**EKF mechanics:**

Prediction (from IMU):
- State transition: `x' = x + dx`, `z' = z + dz`, `yaw' = yaw + dyaw`
- `dx`, `dz` derived from IMU linear velocity integrated over dt (one step)
- `dyaw` derived from IMU angular velocity Y component * dt
- F (Jacobian) = 3x3 identity (linear state transition for discrete steps)
- Q (process noise) = `diag(process_noise_pos, process_noise_pos, process_noise_yaw)`

Update (from VO):
- Measurement: `z_meas = dyaw_vo` (yaw change from VO rotation matrix)
- H = `[0, 0, 1]` (measures only yaw)
- R = `measurement_noise_yaw`
- Standard EKF update: `K = P H^T (H P H^T + R)^{-1}`, `x += K(z - Hx)`, `P = (I - KH) P`

**Key design decisions:**
- VO provides rotation correction only (scale-free). Translation direction from VO is informational but not fused -- IMU displacement is more reliable in simulation (derived from ground-truth state differencing).
- When VO is invalid (`is_valid=False`), the update step is skipped, and the state drifts on IMU prediction alone. This is safe for short gaps.
- Floor height Y is constant (set from initial agent position).

### 3. src/planning/global_planner.py

**Purpose:** Wraps NavMesh `find_path()` with waypoint management, replanning, and path validity checks.

```python
@dataclass
class GlobalPath:
    waypoints: List[NDArray[np.float64]]  # list of [3] world positions
    geodesic_distance: float               # meters
    current_waypoint_idx: int              # index of next target waypoint
    is_valid: bool                         # False if pathfinding failed

class GlobalPlanner:
    """NavMesh-based global path planner.

    Manages a waypoint sequence from start to goal. Supports waypoint
    advancement (when agent reaches a waypoint) and full replanning
    (when obstructed or off-path).
    """

    def __init__(
        self,
        waypoint_reach_threshold: float = 0.5,  # meters
        max_waypoint_skip: int = 3,              # allow skipping ahead
    ) -> None: ...

    def plan(
        self,
        vehicle_find_path: Callable,  # Vehicle.find_path method reference
        start: NDArray[np.float64],
        goal: NDArray[np.float64],
    ) -> GlobalPath:
        """Compute global path from start to goal.

        Args:
            vehicle_find_path: Reference to Vehicle.find_path(start, goal).
            start: [3] current agent world position.
            goal: [3] target world position.

        Returns:
            GlobalPath with waypoints. is_valid=False if pathfinding fails.
        """
        ...

    def get_current_waypoint(self) -> Optional[NDArray[np.float64]]:
        """Return the next waypoint to navigate toward, or None if done."""
        ...

    def advance_waypoint(
        self,
        agent_position: NDArray[np.float64],
    ) -> bool:
        """Check if agent has reached current waypoint and advance.

        Uses waypoint_reach_threshold. May skip waypoints if agent
        is closer to a later waypoint (handles NavMesh shortcutting).

        Returns:
            True if all waypoints reached (goal arrived).
        """
        ...

    def replan(
        self,
        vehicle_find_path: Callable,
        current_position: NDArray[np.float64],
        goal: NDArray[np.float64],
    ) -> GlobalPath:
        """Recompute path from current position to goal."""
        ...

    def get_remaining_distance(self) -> float:
        """Sum of distances from current position through remaining waypoints."""
        ...

    def is_goal_reached(
        self,
        agent_position: NDArray[np.float64],
        goal: NDArray[np.float64],
        threshold: float = 0.5,
    ) -> bool:
        """Check if agent is within threshold of the goal."""
        ...

    def reset(self) -> None: ...
```

**Design notes:**
- `vehicle_find_path` is passed as a callable rather than importing `Vehicle` to avoid circular dependency. The caller (controller/script) binds it.
- Waypoint skipping handles the case where NavMesh paths include intermediate points the agent can skip past (e.g., when the NavMesh path goes around a corner but the agent is already past it).
- `waypoint_reach_threshold = 0.5m` is generous to avoid the agent oscillating around a waypoint. With 0.25m steps, the agent can overshoot by at most 0.25m.

### 4. src/planning/local_planner.py

**Purpose:** Dynamic Window Approach (DWA) that evaluates candidate velocity pairs (linear_v, angular_v) against the occupancy grid and scores them by: (a) heading toward the next waypoint, (b) clearance from obstacles, (c) forward progress.

**Adaptation for discrete actions:** Standard DWA operates in continuous (v, omega) space. Since our action space is discrete, the DWA evaluates a fixed set of candidate "trajectories" corresponding to action sequences of length N (lookahead depth). Each candidate is a short sequence like `[move_forward, move_forward]`, `[turn_left, move_forward]`, `[turn_right, turn_right, move_forward]`, etc.

```python
@dataclass
class LocalPlanResult:
    best_action: str                       # "move_forward", "turn_left", or "turn_right"
    score: float                           # combined score of the best trajectory
    heading_error: float                   # radians, angle to waypoint
    nearest_obstacle_dist: float           # meters, closest obstacle in forward arc
    is_blocked: bool                       # True if no safe action found
    rear_obstacle_warning: bool            # True if rear obstacles detected within threshold

class LocalPlanner:
    """DWA-inspired local planner adapted for discrete action space.

    Evaluates candidate action sequences against the occupancy grid.
    Scores by: goal heading alignment, obstacle clearance, forward progress.
    """

    def __init__(
        self,
        lookahead_steps: int = 3,          # steps to simulate ahead
        heading_weight: float = 1.0,       # weight for goal heading alignment
        clearance_weight: float = 0.8,     # weight for obstacle clearance
        progress_weight: float = 0.5,      # weight for forward progress toward goal
        obstacle_check_radius: float = 0.3, # meters, agent + safety margin
        rear_check_distance: float = 0.5,  # meters, rear obstacle warning zone
        move_amount: float = 0.25,         # meters per move_forward
        turn_amount_deg: float = 10.0,     # degrees per turn
    ) -> None: ...

    def plan(
        self,
        agent_position: NDArray[np.float64],  # [3] current position
        agent_yaw: float,                      # current heading (radians)
        target_waypoint: NDArray[np.float64],  # [3] next waypoint position
        occupancy_grid: OccupancyGridData,     # current occupancy from M2
        rear_obstacle_detected: bool,          # from rear camera obstacle detector
    ) -> LocalPlanResult:
        """Select the best discrete action given current state and grid.

        Generates candidate action sequences, simulates each forward
        (position + heading updates), scores against occupancy grid,
        and returns the best first action.
        """
        ...

    def _generate_candidates(self) -> List[List[str]]:
        """Generate candidate action sequences of length up to lookahead_steps.

        Examples: ["move_forward"], ["turn_left", "move_forward"],
                  ["turn_right", "turn_right", "move_forward"], etc.
        """
        ...

    def _simulate_trajectory(
        self,
        start_x: float,
        start_z: float,
        start_yaw: float,
        actions: List[str],
    ) -> List[Tuple[float, float, float]]:
        """Simulate action sequence to get trajectory of (x, z, yaw) states.

        Uses kinematic model: move_forward advances 0.25m in heading direction,
        turn_left/right changes yaw by +/-10 degrees.
        """
        ...

    def _score_trajectory(
        self,
        trajectory: List[Tuple[float, float, float]],
        target_waypoint: NDArray[np.float64],
        occupancy_grid: OccupancyGridData,
    ) -> Tuple[float, float, float]:
        """Score a candidate trajectory.

        Returns:
            (total_score, heading_error, nearest_obstacle_dist)

        Scoring components:
        1. heading_score: cos(angle between final heading and waypoint direction)
        2. clearance_score: min distance to occupied cells along trajectory
        3. progress_score: distance reduction toward waypoint
        """
        ...

    def _check_occupancy(
        self,
        x: float,
        z: float,
        grid: OccupancyGridData,
        radius: float,
    ) -> float:
        """Check occupancy around a point. Returns distance to nearest obstacle.

        Converts world XZ to grid coordinates, checks cells within radius.
        Returns float('inf') if no obstacles in range.
        """
        ...

    def reset(self) -> None: ...
```

**Candidate generation strategy:**
- At each step, generate ~15-20 candidate action sequences:
  - `[move_forward]` (go straight)
  - `[turn_left * k, move_forward]` for k in 1..5 (turn then go)
  - `[turn_right * k, move_forward]` for k in 1..5
  - `[turn_left * k]` for k in 1..3 (pure rotation)
  - `[turn_right * k]` for k in 1..3
  - `[move_forward, move_forward]` (straight ahead 2 steps)
  - `[move_forward, turn_left, move_forward]` (lane change left)
  - `[move_forward, turn_right, move_forward]` (lane change right)
- Only the first action of the best sequence is executed (receding horizon).

**Rear camera integration:**
- The `rear_obstacle_detected` flag is set when the rear `ObstacleDetection.obstacle_count > 0` and the nearest rear obstacle depth pixel is below `rear_check_distance`.
- When active, the planner adds a penalty to any trajectory that would move the agent backward toward those obstacles (in practice, this means penalizing trajectories that turn 180 degrees then move forward).
- This flag is also surfaced in `LocalPlanResult.rear_obstacle_warning` for UI display.

### 5. src/control/controller.py

**Purpose:** Orchestrates the full navigation loop per step. Takes the estimated pose, the global path, the local planner output, and produces the action. Also handles goal detection, stuck detection, and replanning triggers.

```python
@dataclass
class NavigationStatus:
    action: str                     # action to execute this step
    goal_reached: bool              # True if within goal threshold
    is_stuck: bool                  # True if stuck detection triggered
    needs_replan: bool              # True if replanning is needed
    distance_to_goal: float         # Euclidean distance to goal
    heading_error: float            # radians to next waypoint
    steps_taken: int
    total_collisions: int
    path_length: float              # cumulative distance traveled

@dataclass
class EpisodeResult:
    success: bool
    spl: float                     # Success weighted by Path Length
    path_length: float             # total distance traveled
    geodesic_distance: float       # optimal path length
    steps: int
    collisions: int
    termination_reason: str        # "goal_reached", "max_steps", "stuck", "path_invalid"

class NavigationController:
    """Orchestrates one navigation episode.

    Combines global planner, local planner, and state estimator into
    a step-by-step navigation loop. Handles goal detection, stuck
    detection, and replanning.
    """

    def __init__(
        self,
        goal_threshold: float = 0.5,       # meters to consider goal reached
        stuck_window: int = 20,            # steps to check for stuck
        stuck_displacement: float = 0.3,   # minimum displacement over stuck_window
        max_replan_attempts: int = 3,      # max replans per episode
        max_steps: int = 500,              # max steps per episode
    ) -> None: ...

    def start_episode(
        self,
        start_position: NDArray[np.float64],
        start_rotation: NDArray[np.float64],
        goal_position: NDArray[np.float64],
        global_path: GlobalPath,
    ) -> None:
        """Initialize controller state for a new episode."""
        ...

    def step(
        self,
        pose_estimate: PoseEstimate,
        occupancy_grid: OccupancyGridData,
        rear_obstacle_detected: bool,
        collided: bool,
    ) -> NavigationStatus:
        """Compute one navigation step.

        1. Check goal arrival
        2. Check stuck condition
        3. Advance global waypoint if reached
        4. Run local planner for best action
        5. If blocked, trigger replan
        6. Update tracking (path length, collisions, history)

        Returns:
            NavigationStatus with action and status flags.
        """
        ...

    def finish_episode(self) -> EpisodeResult:
        """Compute final episode metrics."""
        ...

    def _check_stuck(self) -> bool:
        """Check if agent has moved less than stuck_displacement
        over the last stuck_window steps."""
        ...

    def reset(self) -> None: ...
```

**Stuck detection:** Maintain a ring buffer of the last `stuck_window` positions. If `norm(position[now] - position[now - stuck_window]) < stuck_displacement`, the agent is stuck. On stuck detection:
1. Attempt replan (up to `max_replan_attempts`).
2. If replanning also fails, terminate episode with `termination_reason="stuck"`.

**SPL computation:**
```python
spl = success * (geodesic_distance / max(path_length, geodesic_distance))
```
Where `success = 1.0` if `goal_reached`, else `0.0`.

### 6. scripts/run_classical.py

**Purpose:** Main entry point to run classical navigation episodes. Creates the vehicle, perception pipeline, and M3 modules, runs N episodes, and reports aggregate metrics.

```python
"""Run classical navigation pipeline over multiple episodes.

Usage:
    python -m scripts.run_classical --episodes 10 --scene skokloster-castle
    python -m scripts.run_classical --episodes 50 --max-steps 500
"""

def run_episode(
    vehicle: Vehicle,
    vo: VisualOdometry,
    detector: ObstacleDetector,
    grid: OccupancyGrid,
    estimator: EKFEstimator,
    global_planner: GlobalPlanner,
    local_planner: LocalPlanner,
    controller: NavigationController,
    goal: NDArray[np.float64],
) -> EpisodeResult:
    """Execute one navigation episode from current vehicle position to goal.

    Steps:
    1. Reset perception + estimation modules
    2. Get initial observations, initialize EKF
    3. Plan global path
    4. Loop:
       a. Run perception pipeline (LiDAR, VO, obstacle detect, occupancy grid)
       b. EKF predict (IMU) + update (VO)
       c. Controller.step() -> action
       d. Vehicle.step(action)
       e. Check termination
    5. Return EpisodeResult
    """
    ...

def main():
    """Parse args, run episodes, print summary."""
    ...
    # Aggregate metrics:
    # - Success rate: sum(success) / num_episodes
    # - Mean SPL
    # - Mean path length
    # - Mean steps
    # - Total collisions
    # - Mean path efficiency: geodesic_distance / path_length
    ...

if __name__ == "__main__":
    main()
```

**Episode execution flow (one step):**
```
obs = vehicle.get_initial_observations()  # or vehicle.step(action)
    |
    v
[M2 Perception Pipeline]
    lidar_pc = depth_to_point_cloud(obs.depth) -> transform to world
    rear_pc = depth_to_point_cloud(obs.rear_depth) -> transform to world
    vo_est = vo.update(obs.forward_rgb)
    fwd_det, rear_det = detector.detect_both_cameras(...)
    occ_grid = grid.update(agent_pos, agent_rot, [lidar_pc, rear_pc], obstacle_detections)
    |
    v
[M3 State Estimation]
    pose = estimator.predict(obs.imu, dt=1.0)
    if vo_est.is_valid:
        pose = estimator.update_vo(vo_est)
    |
    v
[M3 Planning + Control]
    global_planner.advance_waypoint(pose.position)
    waypoint = global_planner.get_current_waypoint()
    rear_warning = rear_det.obstacle_count > 0 and min_rear_depth < 0.5
    nav_status = controller.step(pose, occ_grid, rear_warning, obs.state.collided)
    |
    v
[Action Execution]
    obs = vehicle.step(nav_status.action)
```

### 7. Viewer Integration

**New WebSocket message types:**

Client -> Server:
```json
{"type": "start_nav"}           // Start autonomous navigation to random goal
{"type": "start_nav_to", "goal": [x, y, z]}  // Navigate to specific goal
{"type": "stop_nav"}            // Stop autonomous navigation, return to manual
{"type": "reset"}               // Existing: reset agent position
{"action": "move_forward"}      // Existing: manual control (disabled during nav)
```

Server -> Client (extends existing frame):
```json
{
    // ... existing M1/M2 fields ...
    "nav_status": {
        "mode": "autonomous",        // "manual" or "autonomous"
        "goal": [x, y, z],           // null if manual mode
        "goal_reached": false,
        "is_stuck": false,
        "distance_to_goal": 3.45,
        "heading_error": 0.23,
        "steps_taken": 42,
        "collisions": 1,
        "path_length": 5.67,
        "spl": null,                 // populated on episode end
        "waypoints": [[x1,y1,z1], ...],  // current global path
        "current_waypoint_idx": 3,
        "action": "move_forward"
    }
}
```

**Server changes (`viewer/server.py`):**
- Add module-level navigation state: `_nav_mode`, `_nav_goal`, `_global_planner`, `_local_planner`, `_estimator`, `_controller`.
- On `start_nav`: pick random goal (or use provided), plan global path, initialize EKF and controller. Set `_nav_mode = "autonomous"`.
- On each WebSocket receive in autonomous mode: instead of waiting for client action, server auto-steps. Use a loop: execute one nav step per WebSocket tick. Client sends `{"type": "tick"}` to request next frame (prevents browser flooding).
- On `stop_nav` or goal reached: set `_nav_mode = "manual"`, send final metrics.
- Manual WASD keys are ignored while `_nav_mode == "autonomous"`.

**Renderer additions (`viewer/renderer.py`):**
- `render_topdown_with_path()`: Extends `render_topdown_view()` to overlay waypoints (blue dots), current waypoint (yellow), goal (green star), and the agent's traveled path (cyan trail). This replaces the basic topdown during navigation.
- `render_nav_status_overlay()`: Small overlay image showing distance-to-goal, heading arrow, stuck/blocked status.

**Frontend additions:**
- New control panel section below existing controls:
  ```
  Navigation: [Start Nav] [Stop Nav]  Mode: Manual
  Goal: (--) | Distance: -- | SPL: --
  ```
- `Start Nav` button sends `start_nav`, `Stop Nav` sends `stop_nav`.
- During autonomous mode, WASD keys are disabled (visual indicator).
- Navigation metrics update in real-time from `nav_status` in frame data.
- On episode complete, flash success/failure indicator and display final SPL.

### 8. Data Flow Summary (Full M3 Step)

```
[Autonomous Mode Active]
    Browser sends {"type": "tick"}
        --> server.py websocket handler
            --> [M2 Perception Pipeline]
                lidar_pc_fwd, lidar_pc_rear -> merged point cloud
                vo.update(forward_rgb) -> VOEstimate
                detector.detect_both_cameras() -> fwd_det, rear_det
                grid.update() -> OccupancyGridData
            --> [M3 State Estimation]
                estimator.predict(imu_reading)
                estimator.update_vo(vo_estimate)  // if valid
                --> PoseEstimate
            --> [M3 Planning]
                global_planner.advance_waypoint(pose.position)
                local_planner.plan(pose, occ_grid, rear_flag)
                --> LocalPlanResult
            --> [M3 Control]
                controller.step(pose, occ_grid, rear_flag, collided)
                --> NavigationStatus
            --> vehicle.step(nav_status.action)
                --> Observations
            --> _build_frame(obs) + nav_status overlay
            --> JSON -> WebSocket -> Browser
    Browser updates all panels + nav status display
```

**Observation shapes at module boundaries (M3 additions):**

| Boundary | Type | Shape/Content |
|----------|------|---------------|
| IMUReading -> EKF | linear_acceleration, angular_velocity | [3] float64 each |
| VOEstimate -> EKF | rotation [3,3] float64, is_valid bool | yaw extracted |
| EKF output | PoseEstimate | position [3] float64, yaw float, cov [3,3] float64 |
| OccupancyGridData -> LocalPlanner | grid [200,200] float32, origin [2] float64 | resolution=0.05 |
| GlobalPath -> Controller | waypoints list of [3] float64 | geodesic_distance float |
| LocalPlanResult -> Controller | best_action str, score float | heading_error float |
| NavigationStatus -> Vehicle | action str | one of {"move_forward","turn_left","turn_right"} |
| EpisodeResult | metrics | success bool, spl float, path_length float, etc. |

### 9. transforms.py Additions

One new utility function needed:

```python
def quaternion_from_yaw(yaw: float) -> NDArray[np.float64]:
    """Create a Y-axis rotation quaternion [w, x, y, z] from yaw angle.

    Args:
        yaw: Rotation about Y axis in radians.

    Returns:
        [4] quaternion [w, x, y, z].
    """
    return np.array([
        math.cos(yaw / 2.0),
        0.0,
        math.sin(yaw / 2.0),
        0.0,
    ], dtype=np.float64)
```

This is used by the EKF to convert its yaw state back to a quaternion for compatibility with the rest of the system.

### 10. Test Design

**tests/test_estimator.py** (pure numpy, no habitat-sim):
- `test_initial_pose_matches_input` -- initialize returns same position/yaw
- `test_predict_stationary_no_drift` -- zero IMU prediction doesn't change pose
- `test_predict_forward_motion` -- forward IMU updates position correctly
- `test_predict_turn_updates_yaw` -- angular velocity IMU updates yaw
- `test_update_vo_corrects_yaw` -- VO measurement corrects yaw estimate
- `test_update_vo_invalid_skipped` -- invalid VO does not change state
- `test_covariance_grows_on_predict` -- uncertainty increases without measurement
- `test_covariance_shrinks_on_update` -- uncertainty decreases with VO update
- `test_reset_clears_state` -- after reset, state is uninitialized

**tests/test_global_planner.py** (pure numpy, mock find_path):
- `test_plan_returns_valid_path` -- mock find_path returns waypoints
- `test_plan_handles_empty_path` -- find_path returns no waypoints -> is_valid=False
- `test_advance_waypoint_at_threshold` -- agent near waypoint advances index
- `test_advance_waypoint_skips_passed` -- agent past intermediate waypoints skips them
- `test_goal_reached_detection` -- agent within threshold of goal
- `test_replan_produces_new_path` -- replan updates waypoints
- `test_remaining_distance_calculation` -- correct distance sum

**tests/test_local_planner.py** (pure numpy):
- `test_straight_path_selects_forward` -- no obstacles, waypoint ahead -> move_forward
- `test_obstacle_ahead_selects_turn` -- occupied cells ahead -> turn action
- `test_waypoint_left_selects_left_turn` -- waypoint to left -> turn_left
- `test_waypoint_right_selects_right_turn` -- waypoint to right -> turn_right
- `test_all_blocked_returns_is_blocked` -- surrounded by obstacles -> is_blocked=True
- `test_rear_obstacle_sets_warning` -- rear flag propagated to result
- `test_clearance_score_varies_with_distance` -- closer obstacles = lower score

**tests/test_controller.py** (pure numpy, mock planner/estimator):
- `test_goal_reached_terminates` -- within threshold -> goal_reached=True
- `test_stuck_detection_triggers` -- no displacement over window -> is_stuck=True
- `test_collision_counter_increments` -- collided=True increments count
- `test_path_length_accumulates` -- path_length grows with movement
- `test_episode_result_spl_calculation` -- SPL computed correctly on success
- `test_episode_result_failure_spl_zero` -- SPL=0 on failure
- `test_replan_on_blocked` -- blocked local plan triggers needs_replan
- `test_max_steps_terminates` -- exceeding max_steps ends episode

**tests/test_navigation.py** (integration, requires habitat-sim):
- `test_full_episode_does_not_crash` -- run one episode, no exceptions
- `test_episode_result_has_valid_metrics` -- all metrics are finite, non-negative
- `test_short_distance_goal_reached` -- start and goal close -> success

---

## RISKS

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| R1 | **EKF divergence from ground truth.** The simulated IMU derives from ground-truth state differencing, so the EKF is effectively smoothing ground truth. This is not representative of real IMU drift. | MEDIUM | Acceptable for M3 (classical baseline). Add Gaussian noise injection to IMU in M4 to test robustness. Document this limitation. |
| R2 | **DWA candidate explosion.** With lookahead_steps=3 and 3 actions, there are 3^3=27 candidates per step. Simulating each against the occupancy grid adds latency. | LOW | 27 candidates with simple grid lookups is sub-millisecond. Pre-generate candidate lists at init time. If latency is an issue, reduce lookahead to 2 (9 candidates). |
| R3 | **Discrete action granularity limits DWA effectiveness.** 10-degree turns and 0.25m steps are coarse. The agent may oscillate when fine alignment is needed (e.g., narrow doorways). | MEDIUM | Waypoint reach threshold (0.5m) is deliberately generous. If oscillation is observed: (a) reduce turn_amount to 5 degrees, or (b) add a "coast" heuristic that repeats the last action for 2 steps before re-planning. |
| R4 | **Occupancy grid is single-step, no memory.** Obstacles seen in previous steps but now out of sensor range are forgotten. The agent could backtrack into previously-seen obstacles. | MEDIUM | Acceptable for M3 with static scenes. For M4+, add temporal grid accumulation with exponential decay. Current grid is already cleared and rebuilt per step by design. |
| R5 | **NavMesh find_path returns paths through furniture.** The NavMesh is baked and may not reflect all obstacles the depth sensor sees (thin objects, objects added after NavMesh bake). | MEDIUM | The local planner provides obstacle avoidance using the occupancy grid, overriding the global path when needed. Replanning handles cases where the local planner is blocked for too long. |
| R6 | **WebSocket autonomous mode may flood the browser.** If the server steps faster than the browser can render, frames queue up and lag increases. | LOW | Use tick-based protocol: server waits for client `{"type": "tick"}` before computing next step. Client sends tick after rendering previous frame. This naturally throttles to browser render speed. |
| R7 | **Goal on different NavMesh island.** `find_path()` may return `inf` distance or empty waypoints. | LOW | `GlobalPlanner.plan()` checks `is_valid` and `geodesic_distance < inf`. `run_classical.py` skips invalid episodes and picks a new goal. |
| R8 | **VO fails consistently in low-texture environments.** Some test scenes have plain walls where ORB finds few features. | MEDIUM | EKF falls back to IMU-only prediction when VO is invalid. The simulated IMU is derived from ground truth, so IMU-only is actually more accurate than real-world. Log VO failure rate per episode for diagnostics. |
| R9 | **Stuck detection false positives.** Agent turning in place (aligning to waypoint) may trigger stuck detection since position doesn't change. | MEDIUM | Use displacement threshold (0.3m over 20 steps) that accounts for pure rotation phases. 20 steps of turning = 200 degrees, more than a full rotation. If false positives occur, increase window or add a "turning" state exemption. |
| R10 | **Viewer autonomous mode conflicts with manual controls.** Race condition if user sends WASD while autonomous mode is active. | LOW | Server ignores manual action messages when `_nav_mode == "autonomous"`. Frontend disables keyboard input visually. |

---

## ACCEPTANCE_CRITERIA

### AC1: EKF State Estimation
- EKF initializes from ground-truth position and yaw without error
- Predict step with zero IMU readings does not change pose
- Predict step with forward IMU reading moves position in heading direction
- VO update corrects yaw when valid, is skipped when invalid
- Covariance grows on predict, shrinks on update
- `PoseEstimate.position` is [3] float64, `PoseEstimate.yaw` is float, `PoseEstimate.covariance` is [3,3] float64
- EKF resets cleanly between episodes

### AC2: Global Planner
- `plan()` returns `GlobalPath` with non-empty waypoints for connected start/goal
- `plan()` returns `is_valid=False` for disconnected or invalid points
- `advance_waypoint()` increments `current_waypoint_idx` when agent is within threshold
- `is_goal_reached()` returns True when agent is within 0.5m of goal
- `replan()` produces a new path from current position
- Waypoints are [3] float64 arrays in world frame

### AC3: Local Planner (DWA)
- With no obstacles and waypoint straight ahead, selects `move_forward`
- With obstacles directly ahead, selects a turn action
- With waypoint to the left, prefers `turn_left`
- When completely blocked, returns `is_blocked=True`
- `rear_obstacle_warning` is True when rear flag is set
- `LocalPlanResult.best_action` is always a valid action string
- Scores are deterministic for the same inputs

### AC4: Navigation Controller
- Detects goal arrival when within `goal_threshold` of goal
- Detects stuck condition when displacement < `stuck_displacement` over `stuck_window` steps
- Triggers replanning when local planner returns `is_blocked`
- Terminates episode at `max_steps`
- `path_length` monotonically increases
- `total_collisions` monotonically increases
- `EpisodeResult.spl` = 0 on failure, > 0 on success
- `EpisodeResult.spl` <= 1.0 always

### AC5: run_classical.py
- `python -m scripts.run_classical --episodes 10` runs without exception
- Prints per-episode results (success, SPL, steps, collisions)
- Prints aggregate summary (success rate, mean SPL, mean steps, total collisions)
- Handles pathfinding failures gracefully (skip episode, report)
- Episodes that succeed have SPL > 0
- All episodes terminate (no infinite loops)

### AC6: Viewer Autonomous Navigation
- "Start Nav" button initiates autonomous navigation
- "Stop Nav" button returns to manual mode
- During autonomous mode, WASD keys are ignored
- Top-down view shows waypoints (blue), current waypoint (yellow), goal (green)
- Navigation status panel shows distance, heading error, steps, collisions
- On goal reached, displays success indicator and final SPL
- On failure (stuck/max_steps), displays failure reason
- Manual WASD control works normally when not in autonomous mode
- Reset button works in both modes

### AC7: Integration
- Full pipeline (M1 sensors -> M2 perception -> M3 planning -> action) executes in < 200ms per step on typical hardware (excluding GPU rendering time)
- No circular import dependencies between M3 modules
- M3 modules can be imported independently of habitat-sim for unit testing (except integration tests)
- All M3 dataclasses are serializable to JSON (for viewer transport)

---

### Critical Files for Implementation
- `/home/etem/cursor/habitat-sim-2/src/state_estimation/estimator.py` - Core EKF implementation fusing IMU + VO, central to the navigation pipeline
- `/home/etem/cursor/habitat-sim-2/src/planning/local_planner.py` - DWA adapted for discrete actions, most complex algorithm in M3, consults occupancy grid
- `/home/etem/cursor/habitat-sim-2/src/control/controller.py` - Orchestrates all M3 components per step, handles goal/stuck/replan logic, computes metrics
- `/home/etem/cursor/habitat-sim-2/scripts/run_classical.py` - Main entry point tying M1+M2+M3 together, episode loop, metric aggregation
- `/home/etem/cursor/habitat-sim-2/viewer/server.py` - Must be extended with autonomous nav WebSocket handlers; pattern to follow for integration
