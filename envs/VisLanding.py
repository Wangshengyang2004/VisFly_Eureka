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
        # debug - commented out for remote execution
        # # Transpose from (C, H, W) to (H, W, C) for matplotlib
        # plt.imshow(np.transpose(self.sensor_obs["color"][0], (1, 2, 0)))
        # plt.show()
        # import cv2 as cv
        # cv.imshow("2 value", np.full_like(two_value[0], 255, dtype=np.uint8) * two_value[0])
        # # Transpose from (C, H, W) to (H, W, C) for OpenCV
        # cv.imshow("color", np.transpose(self.sensor_obs["color"][0], (1, 2, 0)))
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
            return th.zeros((self.num_envs,), device=self.device)
        # precise and stable target flight
        base_r = 0.1
        """reward function"""
        reward = 0.2 * (1.25 - self.centers.norm(dim=1) / 1).clamp_max(1.) + \
                (self.orientation[:, [0, 1]]).norm(dim=1) * -0.2 + \
                0.1 * (3 - self.position[:, 2]).clamp(0, 3) / 3 * 2 + \
                -0.02 * (self.velocity - 0).norm(dim=1) + \
                -0.01 * (self.angular_velocity - 0).norm(dim=1) + \
                0.1 * 20 * self._success * (10 + (th.tensor(self.max_episode_steps, device=self.device) - th.tensor(self._step_count, device=self.device))) / (1 + 2 * (self.velocity - 0).norm(dim=1))  # / (self.velocity.norm(dim=1) + 1)

        return reward
