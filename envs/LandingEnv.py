import numpy as np
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt

from VisFly.utils.type import TensorDict


class LandingEnv(DroneGymEnvsBase):
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: dict = {},
            dynamics_kwargs: dict = {},
            scene_kwargs: dict = {},
            sensor_kwargs: list = [],
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            landing_target: Optional[th.Tensor] = None,
            max_episode_steps: int = 128,
            tensor_output: bool = False,
    ):
        # Use config random_kwargs if provided, otherwise use default
        if not random_kwargs:
            random_kwargs = {
                "state_generator":
                    {
                        "class": "Uniform",
                        "kwargs": [
                            {"position": {"mean": [2., 0., 2.5], "half": [1.0, 1.0, 1.0]}},
                        ]
                    }
            }

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

        # Use landing_target if provided (eval config), otherwise target, otherwise default
        target_pos = landing_target if landing_target is not None else (target if target is not None else [2., 0., 0.2])
        self.target = th.ones((self.num_envs, 1), device=self.device) @ th.as_tensor(target_pos, device=self.device, dtype=th.float32).reshape(1, -1)
        
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32),
        })

    def get_failure(self) -> th.Tensor:
        return self.is_collision

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        state = th.hstack([
            (self.target - self.position) / self.max_sense_radius,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        return TensorDict({
            "state": state,
        })


    def get_success(self) -> th.Tensor:
        landing_half = 0.3
        # return th.full((self.num_envs,), False)
        return (self.position[:, 2] <= 0.2) \
            & (self.position[:, :2] < (self.target[:, :2] + landing_half)).all(dim=1)\
               & (self.position[:, :2] > (self.target[:, :2] - landing_half)).all(dim=1) \
               & ((self.velocity - 0).norm(dim=1) <= 0.3)  #
        # & \
        # ((self.position[:, :2] < self.target[:2] + landing_half).all(dim=1) & (self.position[:, :2] > self.target[:2] - landing_half).all(dim=1))


    def get_reward(self, predicted_obs=None) -> th.Tensor:
        eta = th.as_tensor(1.2)
        v_l = 1 * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
        r_p = -0.1
        descent_v = -self.velocity[:, 2] - 0
        # r_z_punish = ((descent_v > v_l) | (descent_v < 0))
        # r_z = r_z_punish * r_p + \
        #       ~r_z_punish * (eta.pow(descent_v / v_l) - 1) / (eta-1) * 0.1

        r_z_first = descent_v <= v_l
        r_z = ~r_z_first * (eta.pow(-4 * descent_v / v_l + 5) - 1) / (eta - 1) * 0.1 + \
              r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1

        rho = th.as_tensor(1.2)
        d_s = 2. * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
        d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
        r_xy_punish = d_xy > d_s
        r_xy = (rho.pow(1 - d_xy / d_s) - 1) / (rho - 1) * 0.1

        # toward_v = ((self.velocity[:, :2] -0)* ((self.target - self.position)[:, :2])).sum(dim=1) / d_xy
        # r_xy_is_first_sec = toward_v <= v_l
        # r_xy = r_xy_is_first_sec * 0.1 * (rho.pow(toward_v/v_l)-1)/(rho-1)+ \
        #     ~r_xy_is_first_sec * 0.1*(rho.pow(-4 * descent_v / v_l + 5) - 1) / (rho-1) * 0.1

        r_s = 20.
        r_l = self.success * r_s + self.failure * -0.1
        reward = 1. * r_l + 1. * r_xy + 1. * r_z

        return reward
