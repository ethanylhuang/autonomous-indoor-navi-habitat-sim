"""Simulator and agent configuration factory.

Builds the complete habitat_sim.Configuration from sensor rig specs and scene
parameters. Dataclasses provide type-safe, IDE-friendly defaults that are easy
to override per-scenario.
"""

from dataclasses import dataclass
from typing import Optional

import habitat_sim
from habitat_sim.agent import ActionSpec, ActuationSpec, AgentConfiguration

from configs.sensor_rig import all_sensor_specs


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AgentParams:
    radius: float = 0.1  # meters
    height: float = 1.8  # meters
    move_forward_amount: float = 0.25  # meters per step
    turn_amount: float = 10.0  # degrees per step


@dataclass
class SimParams:
    scene_id: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    enable_physics: bool = True
    allow_sliding: bool = True
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_agent_config(params: Optional[AgentParams] = None) -> AgentConfiguration:
    """Build an AgentConfiguration with sensors and discrete action space."""
    if params is None:
        params = AgentParams()

    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = all_sensor_specs()
    agent_cfg.radius = params.radius
    agent_cfg.height = params.height

    agent_cfg.action_space = {
        "move_forward": ActionSpec(
            "move_forward",
            ActuationSpec(amount=params.move_forward_amount),
        ),
        "turn_left": ActionSpec(
            "turn_left",
            ActuationSpec(amount=params.turn_amount),
        ),
        "turn_right": ActionSpec(
            "turn_right",
            ActuationSpec(amount=params.turn_amount),
        ),
    }

    return agent_cfg


def make_sim_config(
    sim_params: Optional[SimParams] = None,
    agent_params: Optional[AgentParams] = None,
) -> habitat_sim.Configuration:
    """Build the complete habitat_sim.Configuration."""
    if sim_params is None:
        sim_params = SimParams()

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = sim_params.scene_id
    sim_cfg.enable_physics = sim_params.enable_physics
    sim_cfg.allow_sliding = sim_params.allow_sliding
    sim_cfg.random_seed = sim_params.random_seed
    sim_cfg.gpu_device_id = 0

    agent_cfg = make_agent_config(agent_params)

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])
