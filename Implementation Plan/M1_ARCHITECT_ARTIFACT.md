# M1 Architect Artifact -- Environment & Sensor Rig

## Problem Restatement

Design the foundation layer for an autonomous indoor vehicle simulator. M1 delivers a working habitat-sim instance with Bullet physics, a cylinder agent with four sensors (forward RGB, rear RGB, depth, simulated IMU), NavMesh pathfinding, and a browser-based interactive viewer for manual validation. Every component must be structured for reuse in M2-M5 without refactoring.

---

## ASSUMPTIONS

1. **Python 3.9+** in a conda environment with `habitat-sim` installed via `conda install habitat-sim withbullet -c conda-forge -c aihabitat`.
2. **Bundled test scenes** are available at `data/scene_datasets/habitat-test-scenes/` (e.g., `skokloster-castle.glb`, `van-gogh-room.glb`). These ship with habitat-sim and require no separate download beyond `python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/`.
3. **habitat-lab is NOT used** in M1. We program directly against `habitat_sim` Python bindings.
4. **Single-threaded sim** -- the viewer backend owns the sole `habitat_sim.Simulator` instance. No concurrent sim access.
5. **Discrete action space only** -- `move_forward` (0.25m), `turn_left` (10 deg), `turn_right` (10 deg). Step sizes are configurable.
6. **`CameraSensorSpec.orientation`** accepts a numpy array of euler angles `[x, y, z]` in radians, relative to the agent node. A rear camera uses `[0.0, pi, 0.0]`.
7. **Coordinate system** is Y-up, right-handed. Camera forward is -Z. Sensor positions are agent-relative in meters.
8. **IMU is not a habitat-sim sensor.** It is a pure Python class that computes linear acceleration and angular velocity by differencing consecutive `AgentState` snapshots.
9. **NavMesh is pre-baked** into the `.glb`/`.navmesh` files. We load it, we do not recompute it.
10. **The viewer serves one client at a time.** Multi-client is out of scope.
11. **FastAPI** and **uvicorn** are the web server stack. Frontend is vanilla HTML/JS (no React/Vue).
12. **WebSocket frame rate** target is approximately 10 FPS (sufficient for manual driving). JPEG compression for RGB, PNG for depth colormap.

---

## IN_SCOPE

- `configs/sensor_rig.py` -- sensor spec factory functions, constants for resolution/FOV/position
- `configs/sim_config.py` -- simulator + agent configuration factory
- `src/vehicle.py` -- creates the simulator, mounts sensors, exposes `step()` + `get_observations()` + `get_state()`
- `src/sensors/__init__.py` -- package init
- `src/sensors/imu.py` -- simulated IMU via AgentState differencing
- `viewer/server.py` -- FastAPI app, WebSocket endpoint, static file serving
- `viewer/renderer.py` -- observation encoding (JPEG/PNG), depth colorization, top-down NavMesh rendering
- `viewer/static/index.html` -- dashboard layout
- `viewer/static/app.js` -- WebSocket client, keyboard handler, canvas rendering
- `tests/test_sensors.py` -- sensor observation shape/dtype validation
- `tests/test_navmesh.py` -- pathfinding smoke test
- `tests/test_imu.py` -- IMU output validation
- `requirements.txt` -- pinned dependencies

---

## OUT_OF_SCOPE

- LiDAR simulation (depth-to-point-cloud) -- M2
- Visual odometry -- M2
- Occupancy grid -- M2
- Obstacle detection / semantic sensor -- M2
- Global/local planners -- M3
- State estimation (EKF) -- M3
- RL environment wrapper -- M4
- Multi-scene benchmarking -- M5
- Click-on-NavMesh goal setting (listed as optional in plan; defer to M3)
- Continuous action space
- Multi-agent support
- Docker/deployment packaging

---

## DESIGN

### 1. File Map and Dependency Graph

```
configs/
  sensor_rig.py      -- pure data, no sim dependency
  sim_config.py      -- depends on sensor_rig, habitat_sim

src/
  sensors/
    __init__.py
    imu.py           -- depends on numpy only
  vehicle.py         -- depends on configs/*, src/sensors/imu.py, habitat_sim

viewer/
  server.py          -- depends on src/vehicle.py, viewer/renderer.py, fastapi, uvicorn
  renderer.py        -- depends on numpy, cv2
  static/
    index.html
    app.js

tests/
  test_sensors.py    -- depends on src/vehicle.py
  test_navmesh.py    -- depends on src/vehicle.py
  test_imu.py        -- depends on src/sensors/imu.py
```

Dependency direction: `viewer/ --> src/vehicle.py --> configs/ + src/sensors/`

No circular dependencies. The viewer imports vehicle, vehicle imports configs and sensors. Sensors are leaf nodes.

### 2. configs/sensor_rig.py

Purpose: Define sensor specifications as pure data. Return `habitat_sim.CameraSensorSpec` instances. All magic numbers live here.

Interface:
```python
# Constants
SENSOR_HEIGHT: float = 1.5        # meters above agent origin
RGB_RESOLUTION: tuple[int, int] = (480, 640)  # (H, W)
DEPTH_RESOLUTION: tuple[int, int] = (480, 640)
HFOV: int = 90                    # degrees

# Factory functions
def forward_rgb_spec() -> CameraSensorSpec   # uuid="forward_rgb", forward-facing
def rear_rgb_spec() -> CameraSensorSpec      # uuid="rear_rgb", orientation=[0, pi, 0]
def depth_spec() -> CameraSensorSpec         # uuid="depth", co-located with forward_rgb
def all_sensor_specs() -> List[CameraSensorSpec]  # returns all three
```

Design notes:
- `orientation` is `[x, y, z]` euler angles in radians. Rear camera sets `y = pi`.
- Resolution is `[H, W]` per habitat-sim convention.
- Position `[0.0, 1.5, 0.0]` places sensors 1.5m above agent origin on Y axis.
- IMU is NOT listed here -- it is not a habitat-sim sensor.
- Each sensor has its own factory so M2 can add sensors by adding one function + appending to `all_sensor_specs()`.

### 3. configs/sim_config.py

Purpose: Build the complete `habitat_sim.Configuration` from sensor rig specs and scene parameters.

Interface:
```python
@dataclass
class AgentParams:
    radius: float = 0.1          # meters
    height: float = 1.8          # meters
    move_forward_amount: float = 0.25   # meters per step
    turn_amount: float = 10.0           # degrees per step

@dataclass
class SimParams:
    scene_id: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    enable_physics: bool = True
    allow_sliding: bool = True
    random_seed: int = 42

def make_agent_config(params: Optional[AgentParams] = None) -> AgentConfiguration
def make_sim_config(sim_params=None, agent_params=None) -> habitat_sim.Configuration
```

Design notes:
- Dataclasses for params -- type-safe, IDE-friendly, easy to override.
- `turn_amount = 10.0` degrees for finer manual control.
- `allow_sliding = True` so agent doesn't get stuck during manual testing.
- Factory returns a complete `habitat_sim.Configuration`.

### 4. src/sensors/imu.py

Purpose: Simulate IMU readings by differencing consecutive `AgentState` snapshots.

Interface:
```python
@dataclass
class IMUReading:
    linear_acceleration: NDArray[np.float64]   # [3] m/s^2, world frame
    angular_velocity: NDArray[np.float64]       # [3] rad/s, world frame
    timestamp_step: int                         # sim step count

class SimulatedIMU:
    def __init__(self, dt: float = 1.0)
    def update(self, position: NDArray, rotation_quat: NDArray) -> IMUReading
    def reset(self) -> None
```

Design notes:
- Stateful class -- needs history (previous position, velocity, rotation).
- Quaternion convention: `[w, x, y, z]` matching habitat-sim's Magnum quaternion.
- First call returns zeros (no spike). Second call has velocity but zero acceleration.
- Angular velocity uses small-angle approximation `2 * vec(q_delta) / dt` (< 0.5% error at 10-degree turns).
- `reset()` for episode boundaries.
- Private quaternion helpers: `_quat_inverse()`, `_quat_multiply()`.

### 5. src/vehicle.py

Purpose: Central facade. Owns the simulator instance. Only class that touches `habitat_sim.Simulator`.

Interface:
```python
@dataclass
class VehicleState:
    position: NDArray[np.float64]       # [3] world frame
    rotation: NDArray[np.float64]       # [4] quaternion wxyz
    step_count: int
    collided: bool
    imu: IMUReading

@dataclass
class Observations:
    forward_rgb: NDArray[np.uint8]      # [480, 640, 4] RGBA
    rear_rgb: NDArray[np.uint8]         # [480, 640, 4] RGBA
    depth: NDArray[np.float32]          # [480, 640] meters
    imu: IMUReading
    state: VehicleState

class Vehicle:
    VALID_ACTIONS = {"move_forward", "turn_left", "turn_right"}

    def __init__(self, sim_params=None, agent_params=None, imu_dt=1.0)
    def step(self, action: str) -> Observations
    def get_initial_observations(self) -> Observations
    def find_path(self, start, goal) -> Tuple[List[NDArray], float]
    def get_navmesh_bounds(self) -> Tuple[NDArray, NDArray]
    def get_topdown_navmesh(self, meters_per_pixel=0.05) -> NDArray[np.bool_]
    def reset(self) -> Observations
    def close(self) -> None
    @property
    def pathfinder(self) -> PathFinder
```

Design notes:
- Vehicle is the ONLY class that touches `habitat_sim.Simulator`.
- `Observations` is a typed dataclass, not a raw dict.
- NavMesh load checked at init -- fails fast with clear error.
- `get_initial_observations()` for viewer's first frame before any action.
- `_magnum_quat_to_array()` helper isolates Magnum quaternion conversion.

### 6. viewer/renderer.py

Purpose: Convert raw numpy observations to browser-friendly bytes. Stateless functions.

Interface:
```python
def encode_rgb_jpeg(rgba: NDArray[np.uint8], quality=80) -> bytes
def colorize_depth(depth: NDArray[np.float32], min_depth=0.0, max_depth=10.0) -> bytes
def render_topdown_view(navmesh_grid, agent_position, agent_rotation, navmesh_bounds, meters_per_pixel=0.05, canvas_size=400) -> bytes
```

Design notes:
- Uses `cv2` (OpenCV) for encoding. JPEG for RGB, PNG for top-down.
- Depth colorized with TURBO colormap (better perceptual uniformity than JET).
- Top-down: navmesh resized to 400x400 canvas, agent projected from world XZ to pixel coords, heading arrow from quaternion yaw.
- Guard against zero-range navmesh bounds.

### 7. viewer/server.py

Purpose: FastAPI app, WebSocket endpoint, static file serving.

Interface:
```
GET /          -> serves index.html
WS  /ws        -> WebSocket endpoint

WebSocket protocol:
  Client sends: {"action": "move_forward"} or {"type": "reset"}
  Server sends: {"forward_rgb": "<b64>", "rear_rgb": "<b64>", "depth": "<b64>",
                  "topdown": "<b64>", "state": {...}, "imu": {...}}
```

Design notes:
- `lifespan` context manager creates/destroys Vehicle.
- Single WebSocket, request/response pattern (client sends action, server responds with frame).
- Base64-encoded images in JSON (~150-250KB per frame, acceptable for local dev).
- Synchronous sim access -- `await ws.receive_text()` serializes requests naturally.

### 8. viewer/static/ (index.html + app.js)

Dashboard layout:
```
+----------------------------------------------------------+
|  Autonomous Vehicle Viewer         Step: 0     [Reset]    |
+----------------------------------------------------------+
|  Forward RGB       |  Rear RGB        |  Depth (color)    |
|  <img id="fwd">   |  <img id="rear"> |  <img id="depth"> |
+----------------------------------------------------------+
|  Top-Down NavMesh  |  Agent State                         |
|  <img id="topdown">|  Position / Rotation / IMU / etc     |
+----------------------------------------------------------+
|  Controls: W=Forward | A=Left | D=Right | R=Reset         |
+----------------------------------------------------------+
```

- Vanilla HTML + CSS grid. Four `<img>` elements receive base64 data URLs.
- Keyboard throttling: boolean `waitingForResponse` flag prevents queuing.
- No framework dependencies.

### 9. Data Flow Summary

```
Browser (app.js) --> keydown {"action":"move_forward"}
    --> WebSocket (server.py)
        --> Vehicle.step(action)
            --> habitat_sim.Simulator.step()
            --> SimulatedIMU.update()
            --> returns Observations dataclass
        --> renderer encodes to JPEG/PNG bytes
        --> base64 encode into JSON
    --> WebSocket sends JSON to browser
Browser updates <img> elements + state text
```

**Observation shapes at each boundary:**

| Boundary | forward_rgb | rear_rgb | depth | IMU |
|----------|-------------|----------|-------|-----|
| habitat-sim output | [480,640,4] uint8 | [480,640,4] uint8 | [480,640] float32 | N/A |
| Observations dataclass | same | same | same | IMUReading([3] f64, [3] f64) |
| renderer output | JPEG bytes | JPEG bytes | JPEG bytes (colorized) | N/A |
| WebSocket payload | base64 string | base64 string | base64 string | JSON arrays |

### 10. Test Design

**tests/test_sensors.py** (11 cases):
- Vehicle creates successfully, initial observations correct shapes/dtypes
- Forward and rear RGB are not identical
- Depth center patch is finite and >= 0
- Each action returns valid Observations, invalid action raises ValueError

**tests/test_navmesh.py** (8 cases):
- Pathfinder loaded, random navigable points have shape (3,)
- find_path returns non-empty waypoints with finite distance
- Top-down navmesh is 2D bool with both True and False values
- Bounds are valid (lower < upper on X and Z)

**tests/test_imu.py** (8 cases):
- First update returns zeros, third forward step has non-zero acceleration
- Turn produces non-zero angular velocity Y component
- Reset clears state, dtypes are float64, shape is (3,)

### 11. requirements.txt

```
habitat-sim          # installed via conda, listed for documentation
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
websockets>=11.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
```

---

## RISKS

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| R1 | **`CameraSensorSpec.orientation` format uncertain.** If the convention is not `[pitch, yaw, roll]` mapping to `[X, Y, Z]`, the rear camera will render the wrong direction. | HIGH | Smoke test immediately: render forward + rear, visually confirm rear shows opposite wall. Fallback: set rotation via `sensor_states["rear_rgb"].rotation` after sim creation. |
| R2 | **Magnum quaternion convention.** `AgentState.rotation` may use `.scalar`/`.vector` accessors, not array indexing. | MEDIUM | `_magnum_quat_to_array()` isolates this. Test with `type()` and `dir()` on the object. |
| R3 | **`previous_step_collided` may not exist** across habitat-sim versions. | MEDIUM | `getattr(self._sim, 'previous_step_collided', False)` as safe fallback. Collision display is informational. |
| R4 | **NavMesh not loaded for some test scenes.** | MEDIUM | Default to `skokloster-castle.glb`. Constructor asserts `pathfinder.is_loaded` and fails fast. |
| R5 | **Base64 payload size** (~150-250KB per frame). | LOW | JPEG quality=80 is reasonable. Monitor during testing. |
| R6 | **IMU angular velocity approximation** breaks for large turn angles. | LOW | < 0.5% error at 10-degree steps. Replace with full axis-angle extraction in M3 if needed. |
| R7 | **`allow_sliding = True` masks collision bugs.** | LOW | Acceptable for M1 manual testing. M3 can set False. |
| R8 | **Single-threaded sim blocks async event loop** (~5-20ms per step). | LOW | At manual speeds (1-3 actions/sec), no perceptible delay. Use `run_in_executor()` in future milestones. |
| R9 | **`physics_config_file` path may not exist.** | MEDIUM | Test with and without. If not found, omit and let habitat-sim use Bullet defaults. Log warning. |

---

## ACCEPTANCE_CRITERIA

### AC1: Simulator initializes successfully
- `Vehicle()` creates without exception
- `vehicle.pathfinder.is_loaded` returns True
- `vehicle.close()` releases resources without exception
- Second `Vehicle()` after close works (no leaked state)

### AC2: Sensor observations have correct shapes and types
- `forward_rgb`: (480, 640, 4), uint8
- `rear_rgb`: (480, 640, 4), uint8
- `depth`: (480, 640), float32
- Forward and rear are visually different (`np.array_equal` returns False)
- Depth center patch (100x100) is finite and >= 0

### AC3: Discrete actions produce expected state changes
- `move_forward` changes position (delta > 0.01)
- `turn_left` / `turn_right` change rotation quaternion
- Invalid action raises ValueError
- `step_count` increments by 1

### AC4: IMU produces valid readings
- First update: zeros for both acceleration and angular velocity
- After 3 forward steps: non-zero acceleration
- After turn_left: non-zero angular velocity Y component
- After reset: next update returns zeros
- All arrays: dtype float64, shape (3,)

### AC5: NavMesh pathfinding works
- find_path between random navigable points returns non-empty waypoints
- geodesic_distance is finite and > 0
- Top-down navmesh is 2D bool array with both True and False
- Bounds: lower < upper on X and Z axes

### AC6: Browser viewer is fully functional
- `uvicorn viewer.server:app --host 0.0.0.0 --port 8000` starts without error
- `http://localhost:8000` shows dashboard with all four panels
- W/A/D keys control agent; all images update per action
- Forward and rear show different perspectives at all times
- Depth shows colorized map (not black, not uniform)
- Top-down shows NavMesh + red dot + green heading arrow
- State and IMU displays update per action
- R key resets agent to random position

### AC7: Configuration is reusable for future milestones
- Changing `SimParams(scene_id=...)` switches scenes without code changes
- Changing `AgentParams(radius=..., turn_amount=...)` changes embodiment without code changes
- `all_sensor_specs()` callable independently of simulator
- Adding M2 sensor requires only: new factory function + append to list + read new obs key
