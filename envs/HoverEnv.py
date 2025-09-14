import numpy as np
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch as th
from gymnasium import spaces
from VisFly.utils.type import TensorDict


class HoverEnv(DroneGymEnvsBase):
    """
    A standalone hover environment for testing BPTT baseline training.
    This environment only provides state observations (no depth/visual).
    """

    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        visual: bool = False,
        requires_grad: bool = False,
        random_kwargs: dict = {},
        dynamics_kwargs: dict = {},
        scene_kwargs: dict = {},
        sensor_kwargs: list = [],
        device: str = "cpu",
        target: Optional[th.Tensor] = None,
        max_episode_steps: int = 256,
        tensor_output: bool = False,
    ):
        
        # Set default random spawn configuration if not provided
        if not random_kwargs:
            random_kwargs = {
                "state_generator": {
                    "class": "Uniform", 
                    "kwargs": [
                        {"position": {"mean": [15.0, 0.0, 1.5], "half": [2.0, 2.0, 0.5]}},
                    ]
                }
            }
        
        # No sensors needed for state-only hover environment
        sensor_kwargs = []
        
        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,
        )
        
        # Set target position
        target_pos = [15.0, 0.0, 1.5] if target is None else target
        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor(target_pos).reshape(1, -1)
        
        # Set success criteria
        self.success_radius = 0.1
        self.success_speed = 0.1
        # Override observation space to only include state
        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
            )
        })

    def get_reward(self) -> th.Tensor:
        base_r = 0.1
        pos_factor = -0.1 * 1/9 / 2

        reward = (
                base_r +
                 (self.position - self.target).norm(dim=1) * pos_factor +
                 (self.velocity - 0).norm(dim=1) * -0.002 +
                 (self.angular_velocity - 0).norm(dim=1) * -0.01
        )

        return reward

    def get_observation(self, indices=None) -> Dict:
        """Get state-only observation for hover task"""
        # Get relative position to target in drone's local frame
        orientation = self.envs.dynamics._orientation.clone()
        rela = self.target - self.position
        head_target = orientation.world_to_head(rela.T).T
        head_velocity = orientation.world_to_head((self.velocity - 0).T).T
        
        # Construct state vector: [relative_target(3), orientation(4), velocity(3), angular_velocity(3)]
        state = th.hstack([
            head_target / 10,
            self.orientation,
            head_velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        return TensorDict({
            "state": state,
        })
    
    def get_success(self) -> th.Tensor:
        """Check if agents are hovering successfully near target"""
        position_success = (self.position - self.target).norm(dim=1) <= self.success_radius
        velocity_success = self.velocity.norm(dim=1) <= self.success_speed
        return position_success & velocity_success
    
    def get_failure(self) -> th.Tensor:
        """Check for collision or other failure conditions"""
        # Use collision detection from base class if available
        return self.is_collision
        # return th.full((self.num_agent,), False)
