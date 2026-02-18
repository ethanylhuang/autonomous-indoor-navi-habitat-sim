# Autonomous Indoor Vehicle — Habitat-Sim

## Overview

Autonomous vehicle with 1 LiDAR (simulated), 2 RGB cameras (forward + rear), and IMU (simulated), navigating indoor environments from a given start to end point. Built on Meta's Habitat-Sim.

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

### M5 — Evaluation & Visualization

**Goal:** Comprehensive benchmarking and debugging tools.

- Top-down trajectory rendering with NavMesh overlay
- Side-by-side classical vs RL comparison
- Sensor observation recording/replay
- Benchmark suite across multiple scenes (upgrade to Replica/HM3D)

## File Structure

```
habitat-sim-2/
├── CLAUDE.md
├── .claude/agents/
│   ├── architect.md
│   ├── builder.md
│   └── shield.md
├── Implementation Plan/
│   └── PROJECT_PLAN.md          # This file
├── configs/
│   ├── sensor_rig.py            # Sensor specs (cameras, depth, IMU params)
│   └── sim_config.py            # Simulator + agent configuration
├── src/
│   ├── sensors/
│   │   ├── lidar.py             # Depth → point cloud conversion
│   │   ├── imu.py               # State-differencing IMU simulation
│   │   └── cameras.py           # Camera config, image preprocessing
│   ├── perception/
│   │   ├── occupancy_grid.py    # Fused grid (LiDAR + visual)
│   │   ├── visual_odometry.py   # Feature-based VO from RGB frames
│   │   └── obstacle_detector.py # Semantic segmentation pipeline
│   ├── planning/
│   │   ├── global_planner.py    # NavMesh pathfinding wrapper
│   │   └── local_planner.py     # DWA / obstacle avoidance
│   ├── control/
│   │   └── controller.py        # Waypoint → action conversion
│   ├── state_estimation/
│   │   └── estimator.py         # IMU + VO fusion (EKF or similar)
│   ├── rl/                      # M4
│   │   ├── env.py               # Gym-style environment wrapper
│   │   ├── policy.py            # Policy network
│   │   └── train.py             # Training loop
│   └── vehicle.py               # Agent setup, sensor mounting, main loop
├── viewer/
│   ├── server.py                # FastAPI + WebSocket backend
│   ├── static/
│   │   ├── index.html           # Dashboard UI
│   │   └── app.js               # Keyboard controls, WebSocket client, canvas rendering
│   └── renderer.py              # Sensor frame encoding, top-down view rendering
├── scripts/
│   ├── run_classical.py         # Run classical nav pipeline
│   ├── run_rl.py                # Run RL policy
│   └── visualize.py             # Trajectory + sensor visualization
├── tests/
│   ├── test_sensors.py
│   ├── test_planner.py
│   └── test_navigation.py
├── data/
│   └── scene_datasets/          # Downloaded scenes (gitignored)
└── requirements.txt
```
