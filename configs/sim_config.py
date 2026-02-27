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
    scene_dataset_config: Optional[str] = None  # Path to .scene_dataset_config.json
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
    """Build the complete habitat_sim.Configuration.

    When scene_dataset_config is provided, uses habitat-sim's make_cfg for proper
    scene/navmesh coordinate alignment.
    """
    if sim_params is None:
        sim_params = SimParams()
    if agent_params is None:
        agent_params = AgentParams()

    # When using scene_dataset_config, use habitat-sim's make_cfg for proper alignment
    if sim_params.scene_dataset_config:
        from habitat_sim.utils.settings import default_sim_settings, make_cfg

        settings = default_sim_settings.copy()
        settings["scene_dataset_config_file"] = sim_params.scene_dataset_config
        settings["scene"] = sim_params.scene_id
        settings["enable_physics"] = sim_params.enable_physics
        settings["seed"] = sim_params.random_seed

        # Build base config using habitat-sim's make_cfg
        cfg = make_cfg(settings)

        # Replace agent config with our custom one (preserves our sensor specs)
        agent_cfg = make_agent_config(agent_params)
        cfg.agents = [agent_cfg]

        return cfg

    # Standard path for direct scene loading (no scene_dataset_config)
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = sim_params.scene_id
    sim_cfg.enable_physics = sim_params.enable_physics
    sim_cfg.allow_sliding = sim_params.allow_sliding
    sim_cfg.random_seed = sim_params.random_seed
    sim_cfg.gpu_device_id = 0

    agent_cfg = make_agent_config(agent_params)

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])
