# Autonomous Indoor Vehicle — Habitat-Sim

## Overview

Autonomous vehicle with simulated LiDAR (dual depth sensors), dual RGB cameras (forward + rear), dual semantic cameras, and simulated IMU, navigating indoor environments from a given start to end point. Built on Meta's Habitat-Sim.

## Milestone Status

| Milestone | Status | Description |
|-----------|--------|-------------|
| **M1** | COMPLETE | Environment & sensor rig |
| **M2** | COMPLETE | Perception pipeline |
| **M3** | COMPLETE | Classical navigation stack |
| **M4** | INFRASTRUCTURE | RL navigation policy (env, policy, train script ready) |
| **M5** | IN PROGRESS | VLM/semantic navigation |

## Habitat-Sim Constraints

- **No native LiDAR** — simulated via depth sensor + ray casting or equirectangular depth + column sampling
- **No native IMU** — derived from successive `AgentState` queries (position/rotation differencing)
- **Multi-camera is trivial** — multiple `CameraSensorSpec` with unique UUIDs and independent positions
- **NavMesh pathfinding built-in** — Recast/Detour: shortest path, navigability checks, `GreedyGeodesicFollower`
- **Custom embodiment supported** — URDF-based articulated objects or simple cylinder agents
- **Two-layer architecture** — habitat-sim (low-level engine) vs habitat-lab (RL/task/config layer)

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Navigation approach | Classical first, then RL | Deterministic baseline before learned policy |
| Scene dataset | Test scenes (bundled) | Zero setup cost; upgrade to Replica/HM3D later |
| Vehicle model | Cylinder agent | Fastest to set up; sufficient for navigation research |
| Camera usage | Full utilization | Visual odometry, obstacle detection, and RL observations |
| habitat-lab | Deferred to M4 | Standalone habitat-sim gives direct control for M1-M3 |
| Action space | Discrete first | Simpler to debug; matches NavMesh GreedyGeodesicFollower |
| Semantic detection | Built-in semantic sensor first | Ground-truth stand-in; swap for learned model in RL phase |
| M1 validation | Browser-based interactive viewer | Manual WASD control with live sensor feeds; most intuitive proof of completion |
| Visualization backend | FastAPI + WebSocket | Lightweight, async-native, minimal deps |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Main Loop                         │
│  start/goal → perceive → plan → act → repeat → done │
└──────┬──────────┬──────────┬──────────┬─────────────┘
       │          │          │          │
  ┌────▼────┐ ┌──▼───┐ ┌───▼────┐ ┌───▼────┐
  │ Sensors │ │Fusion│ │Planner │ │Control │
  │  Rig    │ │      │ │        │ │        │
  ├─────────┤ ├──────┤ ├────────┤ ├────────┤
  │Fwd RGB  │ │Occ.  │ │NavMesh │ │Action  │
  │Rear RGB │ │Grid  │ │(global)│ │Executor│
  │Depth→PC │ │(fused│ │DWA     │ │        │
  │IMU(sim) │ │lidar │ │(local) │ │        │
  │         │ │+vis) │ │        │ │        │
  └─────────┘ └──────┘ └────────┘ └────────┘

    M1            M2         M3          M3

                         ┌────────────────────┐
                         │  M4: RL Policy      │
                         │  (replaces planner) │
                         └────────────────────┘
```

### State Estimation

```
IMU (accel/gyro) ──┐
                    ├──→ EKF/Complementary Filter ──→ Fused Pose Estimate
Visual Odometry ───┘
```

Redundant localization: VO corrects long-term IMU drift; IMU corrects short-term VO gaps.

## Milestones

### M1 — Environment & Sensor Rig

**Goal:** Simulator running with all sensors mounted and producing valid observations.

- Conda environment setup, habitat-sim with Bullet physics
- Download bundled test scenes
- Cylinder agent with configurable radius/height
- Sensor suite:
  - `forward_rgb` — pinhole camera, facing forward
  - `rear_rgb` — pinhole camera, rotated 180 deg
  - `depth` — depth sensor (basis for LiDAR simulation)
  - IMU — computed from `AgentState` deltas (position/rotation differencing)
- NavMesh loads and pathfinding works
- **Browser-based interactive viewer** for M1 validation:
  - FastAPI + WebSocket backend streaming sensor frames
  - Frontend dashboard: forward RGB | rear RGB | depth (colorized) | top-down NavMesh
  - WASD keyboard control (W=forward, A=turn left, D=turn right)
  - Live agent state display (position, rotation, step count, collisions)
  - Click-on-NavMesh to set goal (optional, for pathfinding validation)
- **Validation:** manually drive the vehicle in browser, visually confirm all sensor feeds are correct, NavMesh renders, agent responds to controls

### M2 — Perception Pipeline

**Goal:** Raw sensor data converted into actionable representations.

- **LiDAR simulation:** equirectangular depth or multi-ray depth sampling → 3D point cloud
- **IMU simulation:** linear acceleration + angular velocity from consecutive agent states
- **Visual odometry:** estimate ego-motion from consecutive front/rear RGB frames (feature matching or optical flow). Cross-checks IMU, provides drift correction.
- **Semantic obstacle detection:** front + rear RGB through segmentation to identify obstacles depth may miss (glass, mirrors, thin objects, open doors). Start with habitat-sim's semantic sensor as ground truth.
- **Occupancy grid:** fused from both LiDAR point cloud + camera-based visual detections (2D bird's-eye projection)
- **Validation:** visualize point cloud, occupancy grid, VO trajectory, semantic overlays

### M3 — Classical Navigation Stack

**Goal:** Vehicle autonomously navigates from start to goal using classical methods.

- **Global planner:** NavMesh `PathFinder.find_path()` → waypoint sequence
- **Local planner:** DWA (Dynamic Window Approach) using fused occupancy grid (LiDAR + visual detections)
- **Rear camera collision avoidance:** rear feed actively checked during reversing or tight-space maneuvering
- **State estimation:** IMU + VO fusion (EKF or complementary filter) for robust localization
- **Controller:** convert waypoints + local plan → discrete actions (`move_forward`, `turn_left`, `turn_right`)
- **Stopping logic:** goal reached, stuck detection, replanning on obstruction
- **Metrics:** SPL (Success weighted by Path Length), success rate, collision count, path efficiency
- **Validation:** navigate multiple start/goal pairs across test scenes

### M4 — RL Navigation Policy

**Goal:** Learned policy matches or exceeds classical baseline.

- Gym-style environment wrapper (habitat-lab or custom)
- Observation space: forward RGB + rear RGB + depth + IMU + goal vector
- Action space: discrete (matching M3) or continuous
- Train with PPO/DDPPO via habitat-baselines
- Compare against classical baseline on same episode set
- **Validation:** trained policy achieves comparable or better SPL

### M5 — VLM-Guided Semantic Navigation

**Goal:** Natural language instruction following via Vision-Language Models.

**Completed:**
- Semantic scene parsing from HM3D `.semantic.txt` annotation files
- `SemanticSceneParser` — extracts object labels, IDs, regions
- `SemanticObject` dataclass — object_id, label, centroid, navmesh_position, instance_number
- Multi-view centroid computation (360-degree sweep for accurate world-space positions)
- NavMesh snapping of object positions
- Structural element filtering (walls, doors, ceilings excluded)
- Viewer integration — semantic object dropdown for goal selection

**In Progress:**
- `VLMClient` — Anthropic Claude API wrapper for pixel-based navigation queries
- `VLMNavigator` — Hierarchical controller: VLM pixel selection + classical planning
- Pixel-to-world projection (`vlm/projection.py`)
- VLM prompt engineering (`vlm/prompts.py`)

**Architecture:**
```
User Instruction ("go to the red couch")
         │
         ▼
   ┌───────────┐
   │ VLMClient │ ◄── Claude API (pixel target selection)
   └─────┬─────┘
         │ (u, v) pixel
         ▼
   ┌─────────────┐
   │ Projection  │ ◄── Depth + camera intrinsics
   └─────┬───────┘
         │ 3D world point
         ▼
   ┌─────────────┐
   │ NavMesh Snap│
   └─────┬───────┘
         │ navigable goal
         ▼
   ┌─────────────┐
   │ Classical   │ ◄── GlobalPlanner + LocalPlanner + Controller
   │ Nav Stack   │
   └─────────────┘
```

**Validation:**
- Semantic object lookup by label (e.g., "couch", "table")
- VLM-selected pixels project to valid navigable positions
- End-to-end instruction following across HM3D scenes

### M6 — Evaluation & Benchmarking (Future)

**Goal:** Comprehensive benchmarking and debugging tools.

- Top-down trajectory rendering with NavMesh overlay
- Side-by-side classical vs RL vs VLM comparison
- Sensor observation recording/replay
- Benchmark suite across multiple HM3D scenes
- Instruction-following success metrics

## File Structure

```
autonomous-indoor-navi-habitat-sim/
├── CLAUDE.md                    # Project operating contract
├── .claude/agents/
│   ├── orchestrator.md          # Main agent coordination protocol
│   ├── architect.md             # Design phase agent
│   ├── builder.md               # Implementation agent
│   └── shield.md                # QA/verification agent
├── Implementation Plan/
│   ├── PROJECT_PLAN.md          # This file
│   ├── M1_ARCHITECT_ARTIFACT.md # M1 design spec
│   ├── M2_ARCHITECT_ARTIFACT.md # M2 design spec
│   ├── M3_ARCHITECT_ARTIFACT.md # M3 design spec
│   └── M5_VLM_NAVIGATION.md     # M5 VLM design spec
├── configs/
│   ├── sensor_rig.py            # Sensor specs (6 sensors: RGB, depth, semantic x2)
│   ├── sim_config.py            # Simulator + agent configuration
│   └── rl_config.py             # RL hyperparameters
├── src/
│   ├── sensors/
│   │   ├── lidar.py             # Depth → point cloud conversion
│   │   └── imu.py               # State-differencing IMU simulation
│   ├── perception/
│   │   ├── occupancy_grid.py    # Fused grid (LiDAR + visual)
│   │   ├── visual_odometry.py   # ORB + RANSAC VO from RGB frames
│   │   ├── obstacle_detector.py # Semantic obstacle detection
│   │   └── semantic_scene.py    # HM3D semantic scene parsing (M5)
│   ├── planning/
│   │   ├── global_planner.py    # NavMesh pathfinding + corner smoothing
│   │   └── local_planner.py     # Rule-based obstacle avoidance
│   ├── control/
│   │   └── controller.py        # Full nav loop orchestration
│   ├── state_estimation/
│   │   └── estimator.py         # EKF (IMU + VO fusion)
│   ├── rl/                      # M4
│   │   ├── env.py               # Gymnasium environment wrapper
│   │   ├── policy.py            # Multi-modal feature extractor (SB3)
│   │   └── train.py             # PPO training loop
│   ├── vlm/                     # M5
│   │   ├── client.py            # Anthropic Claude API wrapper
│   │   ├── navigator.py         # Hierarchical VLM + classical planner
│   │   ├── projection.py        # Pixel → world coordinate projection
│   │   └── prompts.py           # VLM prompt templates
│   ├── utils/
│   │   └── transforms.py        # Shared quaternion/camera/angle utilities
│   └── vehicle.py               # Simulator facade, sensor mounting, stepping
├── viewer/
│   ├── server.py                # FastAPI + WebSocket backend
│   ├── renderer.py              # Visualization (RGB, depth, semantic, occupancy, top-down)
│   └── static/
│       ├── index.html           # Dashboard UI
│       └── app.js               # Keyboard controls, scene/goal selection, WebSocket
├── scripts/
│   ├── run_classical.py         # Classical nav evaluation
│   └── run_rl.py                # RL policy evaluation
├── tests/                       # 20+ test files covering all modules
├── data/
│   └── scene_datasets/          # HM3D/test scenes (gitignored)
└── requirements.txt
```
