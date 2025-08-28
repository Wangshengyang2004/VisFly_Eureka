import numpy as np
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt

from VisFly.utils.type import TensorDict


class VisLandingEnv(DroneGymEnvsBase):
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
            max_episode_steps: int = 128,
            tensor_output: bool = False,
            is_eval: bool = False,
    ):
        sensor_kwargs = [{
            "sensor_type": SensorType.COLOR,
            "uuid": "color",
            "resolution": [64, 64],
            "orientation": [-np.pi / 2, 0, 0]
        }]
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
            tensor_output=tensor_output
        )

        self.target = th.tensor([2, 0, 0], device=device)
        self.success_radius = 0.5
        self.observation_space["target"] = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.centers = None

    def get_failure(self):
        if self.centers is None:
            return th.zeros(self.num_envs, dtype=th.bool, device=self.device)
        out_of_vision = th.isnan(self.centers).any(dim=1)
        return out_of_vision

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        two_value = (self.sensor_obs["color"].mean(axis=1) < 70)
        self._pre_centers = self.centers if self.centers is not None else None
        # Compute centers with proper NaN handling and dtype conversion
        center_coords = []
        for each_img in two_value:
            center_coord = center_of_mass(each_img)
            # Replace NaN with previous center or default position
            if np.any(np.isnan(center_coord)):
                center_coord = (0.0, 0.0)  # Default to center of image
            center_coords.append(center_coord)
        
        self.centers = th.as_tensor(center_coords, dtype=th.float32, device=self.device) \
                       / self.observation_space["color"].shape[1] - 0.5
        # Check for out of vision after center computation
        out_of_vision = th.isnan(self.centers).any(dim=1)
        if self._pre_centers is not None:
            for i in th.arange(self.num_envs, device=self.device)[out_of_vision]:
                self.centers[i] = self._pre_centers[i]
        # debug
        # plt.imshow(self.sensor_obs["color"][0])
        # plt.show()
        # import cv2 as cv
        # cv.imshow("2 value", np.full_like(two_value[0], 255, dtype=np.uint8) * two_value[0])
        # cv.imshow("color", self.sensor_obs["color"][0])
        # cv.waitKey(0)
        return TensorDict({
            "state": self.state,
            "color": th.as_tensor(self.sensor_obs["color"], dtype=th.float32) / 255.0,
            "target": self.centers,
        })

    def get_success(self) -> th.Tensor:
        landing_half = 0.3
        # return th.full((self.num_envs,), False)
        return ((self.position - 0)[:, 2] <= 0.2) \
            & ((self.position - 0)[:, :2] < (self.target[:2] + landing_half)).all(dim=1)\
               & ((self.position - 0)[:, :2] > (self.target[:2] - landing_half)).all(dim=1) \
               & ((self.velocity - 0).norm(dim=1) <= 0.3)  #
        # & \
        # ((self.position[:, :2] < self.target[:2] + landing_half).all(dim=1) & (self.position[:, :2] > self.target[:2] - landing_half).all(dim=1))


    def get_reward(self) -> th.Tensor:
        if self.centers is None:
            return th.zeros((self.num_envs,), dtype=th.float32, device=self.device)
        
        # Ensure all tensors are float32
        centers = self.centers.to(dtype=th.float32)
        position = self.position.to(dtype=th.float32)
        velocity = self.velocity.to(dtype=th.float32)
        orientation = self.orientation.to(dtype=th.float32)
        angular_velocity = self.angular_velocity.to(dtype=th.float32)
        
        # Handle NaN values in centers by replacing with large penalty distance
        centers_norm = th.where(th.isnan(centers).any(dim=1, keepdim=True), 
                               th.full_like(centers.norm(dim=1, keepdim=True), 2.0), 
                               centers.norm(dim=1, keepdim=True)).squeeze()
        
        # precise and stable target flight
        base_r = 0.1
        """reward function"""
        reward = base_r + 0.2 * (1.25 - centers_norm / 1).clamp_max(1.) + \
                (orientation[:, [0, 1]]).norm(dim=1) * -0.2 + \
                0.1 * (3 - position[:, 2]).clamp(0, 3) / 3 * 2 + \
                -0.02 * (velocity - 0).norm(dim=1) + \
                -0.01 * (angular_velocity - 0).norm(dim=1)

        # Ensure reward is float32 and handle any remaining NaN values
        reward = reward.to(dtype=th.float32)
        reward = th.where(th.isnan(reward), th.tensor(-1.0, dtype=th.float32, device=self.device), reward)
        
        return reward


class VisLandingEnv2(VisLandingEnv):
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
            max_episode_steps: int = 128,
            tensor_output: bool = False,
            is_eval: bool = False,
    ):
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
            target=target,
            max_episode_steps=max_episode_steps,
            is_eval=is_eval
        )
        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([2., 0., 2.5] if target is None else target).reshape(1, -1)
        if is_eval:
            self.target = th.as_tensor([[2., 1., 2.5],[2., 0., 2.5],[2., -1., 2.5]])
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32),
        })

    def get_failure(self) -> th.Tensor:
        return self.is_collision

    def get_reward(self) -> th.Tensor:
        eta = th.as_tensor(1.2, dtype=th.float32, device=self.device)
        v_l = 1 * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach().to(dtype=th.float32)
        r_p = -0.1
        descent_v = (-self.velocity[:, 2] - 0).to(dtype=th.float32)
        # r_z_punish = ((descent_v > v_l) | (descent_v < 0))
        # r_z = r_z_punish * r_p + \
        #       ~r_z_punish * (eta.pow(descent_v / v_l) - 1) / (eta-1) * 0.1

        r_z_first = descent_v <= v_l
        r_z = ~r_z_first * (eta.pow(-4 * descent_v / v_l + 5) - 1) / (eta - 1) * 0.1 + \
              r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1

        rho = th.as_tensor(1.2, dtype=th.float32, device=self.device)
        d_s = 2. * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach().to(dtype=th.float32)
        d_xy = ((self.target - self.position)[:, :2].norm(dim=1) - 0).to(dtype=th.float32)
        r_xy_punish = d_xy > d_s
        r_xy = (rho.pow(1 - d_xy / d_s) - 1) / (rho - 1) * 0.1

        # toward_v = ((self.velocity[:, :2] -0)* ((self.target - self.position)[:, :2])).sum(dim=1) / d_xy
        # r_xy_is_first_sec = toward_v <= v_l
        # r_xy = r_xy_is_first_sec * 0.1 * (rho.pow(toward_v/v_l)-1)/(rho-1)+ \
        #     ~r_xy_is_first_sec * 0.1*(rho.pow(-4 * descent_v / v_l + 5) - 1) / (rho-1) * 0.1

        r_s = 20.
        r_l = (self.success * r_s + self.failure * -0.1).to(dtype=th.float32)
        reward = (1. * r_l + 1. * r_xy + 1. * r_z).to(dtype=th.float32)
        
        # Handle any NaN values that might have been introduced
        reward = th.where(th.isnan(reward), th.tensor(-1.0, dtype=th.float32, device=self.device), reward)
        
        return reward

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
