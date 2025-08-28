from typing import Union, Optional, Dict, List

import numpy as np
import torch as th

from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from VisFly.utils.randomization import UniformStateRandomizer, NormalStateRandomizer

g = th.tensor([[0, 0, -9.8]])


class ball:
    def __init__(self,
                 num_agents,
                 random_kwargs=None,
                 dt=0.2,
                 ):
        self.num_agents = num_agents
        random_kwargs = dict(
            position={"mean": [1., 0., 1.5], "half": [0.0, 2., 1.]},
            orientation={"mean": [0., 0., 0.], "half": [0., 0., 0.]},
            velocity={"mean": [0., 0., 0.], "half": [1., 1., 0.]},
            angular_velocity={"mean": [0., 0., 0.], "half": [0., 0., 0.]},
        ) if random_kwargs is None else random_kwargs
        self.randomizer = UniformStateRandomizer(
            **random_kwargs
        )
        self.position = th.empty((self.num_agents, 3))
        self.orientation = th.empty((self.num_agents, 3))
        self.velocity = th.empty((self.num_agents, 3))
        self.angular_velocity = th.empty((self.num_agents, 3))
        self.is_collision = th.zeros(self.num_agents, dtype=th.bool)

        self.dt = dt

    def reset(self):
        self.reset_by_id()

    def reset_by_id(self, indices=None):
        indices = th.arange(self.num_agents) if indices is None else indices
        pos, _, vel, _ = self.randomizer.generate(len(indices))
        self.position[indices] = pos
        self.velocity[indices] = vel
        self.is_collision[indices] = False

    def step(self):
        self.position += self.velocity * self.dt
        self.velocity += g * self.dt
        self.is_collision = self.position[:, 2] < 0.1


class CatchEnv(DroneGymEnvsBase):
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = False,
            max_episode_steps: int = 1000,
            device: Optional[th.device] = th.device("cpu"),
            dynamics_kwargs=None,
            random_kwargs=None,
            requires_grad: bool = False,
            scene_kwargs: Optional[Dict] = None,
            sensor_kwargs: Optional[List] = None,
            tensor_output: bool = False,

    ):
        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            max_episode_steps=max_episode_steps,
            device=device,
            dynamics_kwargs=dynamics_kwargs,
            random_kwargs=random_kwargs,
            requires_grad=requires_grad,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            tensor_output=tensor_output,
        )
        scene_kwargs["object_kwargs"] = {
            "object_setting_path": "VisFly/datasets/visfly-beta/configs/free_falling_objects.json",
            "isolated": True,
        }

    def get_observation(self, indices=None) -> dict:
        """Get basic state observation for catch task"""
        from VisFly.utils.type import TensorDict
        
        # Basic state observation - position, orientation, velocity, angular velocity
        state = th.hstack([
            self.position.flatten(),  # 3 values per agent
            self.orientation.flatten(),  # 4 values per agent  
            self.velocity.flatten(),  # 3 values per agent
            self.angular_velocity.flatten(),  # 3 values per agent
        ]).to(self.device)
        
        return TensorDict({
            "state": state.view(self.num_agent, -1),  # Reshape to [num_agents, features]
        })

    def get_success(self) -> th.Tensor:
        """Basic success check - placeholder"""
        return th.full((self.num_agent,), False)

    def get_reward(self) -> th.Tensor:
        """Basic reward function for catch task"""
        # Simple reward based on maintaining stable flight
        base_reward = 0.1 * th.ones((self.num_agent,), dtype=th.float32)
        
        # Penalize large velocities and angular velocities (encourage stable flight)
        velocity_penalty = -0.01 * (self.velocity - 0).norm(dim=1)
        angular_penalty = -0.01 * (self.angular_velocity - 0).norm(dim=1)
        
        reward = base_reward + velocity_penalty + angular_penalty
        return reward
