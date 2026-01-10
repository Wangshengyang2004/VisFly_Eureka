import numpy as np
from typing import Dict, Optional

import torch as th
from gymnasium import spaces
from habitat_sim import SensorType
from scipy.ndimage import center_of_mass

from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from VisFly.utils.type import TensorDict


class VisLandingEnv(DroneGymEnvsBase):
    """Visual landing environment using a color camera for pad localisation."""

    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        visual: bool = True,
        requires_grad: bool = False,
        random_kwargs: Optional[dict] = None,
        dynamics_kwargs: Optional[dict] = None,
        scene_kwargs: Optional[dict] = None,
        sensor_kwargs: Optional[list] = None,
        device: str = "cpu",
        target: Optional[th.Tensor] = None,
        max_episode_steps: int = 128,
        tensor_output: bool = False,
        is_eval: bool = False,
    ):
        if not sensor_kwargs:
            sensor_kwargs = [
                {
                    "sensor_type": SensorType.COLOR,
                    "uuid": "color",
                    "resolution": [64, 64],
                    "orientation": [-np.pi / 2, 0.0, 0.0],
                }
            ]

        if not random_kwargs:
            random_kwargs = {
                "state_generator": {
                    "class": "Uniform",
                    "kwargs": [
                        {
                            "position": {
                                "mean": [2.0, 0.0, 2.5],
                                "half": [1.0, 1.0, 0.5],
                            }
                        }
                    ],
                }
            }

        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs or {},
            scene_kwargs=scene_kwargs or {},
            sensor_kwargs=sensor_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,
        )

        base_target = th.as_tensor(
            [2.0, 0.0, 0.0] if target is None else target, device=self.device, dtype=th.float32
        )
        self.target_world = th.ones((self.num_envs, 1), device=self.device) @ base_target.view(1, -1)
        flattened_target = base_target.view(-1)
        self.target_height = float(flattened_target[-1].item()) if flattened_target.numel() >= 3 else 0.0
        self.success_radius = 0.4
        self._image_height = sensor_kwargs[0]["resolution"][0]
        self._image_width = sensor_kwargs[0]["resolution"][1]
        self._pad_offset = th.zeros((self.num_envs, 2), device=self.device, dtype=th.float32)
        self._target_visible = th.ones(self.num_envs, dtype=th.bool, device=self.device)
        self.observation_space["target"] = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def get_failure(self) -> th.Tensor:
        return self.is_collision | (~self._target_visible)

    def get_observation(self, indices=None) -> Dict:
        color_obs = self.sensor_obs["color"]
        grayscale = color_obs.mean(axis=1)
        mask_stack = grayscale < 70

        pad_offsets = self._pad_offset.clone()
        visibility = th.zeros(self.num_envs, dtype=th.bool, device=self.device)

        for idx, mask in enumerate(mask_stack):
            if not mask.any():
                continue
            center = center_of_mass(mask)
            if center[0] is None or np.isnan(center[0]) or np.isnan(center[1]):
                continue

            row, col = center
            width = max(self._image_width - 1, 1)
            height = max(self._image_height - 1, 1)
            norm_x = (col / width) * 2.0 - 1.0
            norm_y = (row / height) * 2.0 - 1.0
            pad_offsets[idx, 0] = float(np.clip(norm_x, -1.0, 1.0))
            pad_offsets[idx, 1] = float(np.clip(norm_y, -1.0, 1.0))
            visibility[idx] = True

        self._pad_offset = pad_offsets
        self._target_visible = visibility

        return TensorDict(
            {
                "state": self.state,
                "color": th.as_tensor(color_obs, dtype=th.float32) / 255.0,
                "target": self._pad_offset,
            }
        )

    def get_success(self) -> th.Tensor:
        horizontal_error = (self.position[:, :2] - self.target_world[:, :2]).norm(dim=1)
        height_condition = self.position[:, 2] <= self.target_height + 0.1
        speed_condition = self.velocity.norm(dim=1) <= 0.3
        return (horizontal_error <= self.success_radius) & height_condition & speed_condition

    def get_reward(self) -> th.Tensor:
        frame_error = self._pad_offset.norm(dim=1)
        frame_reward = (1.0 - frame_error.clamp(max=1.0)) * 0.6

        vertical_error = th.abs(self.position[:, 2] - self.target_height)
        altitude_reward = (1.0 - (vertical_error / 2.0).clamp(max=1.0)) * 0.3

        downward_speed = (-self.velocity[:, 2]).clamp(min=0.0)
        descent_reward = 0.1 * downward_speed

        stability_penalty = 0.05 * self.velocity.norm(dim=1)
        angular_penalty = 0.02 * self.angular_velocity.norm(dim=1)

        success_bonus = self.success.float() * 5.0
        collision_penalty = self.is_collision.float() * 5.0
        visibility_penalty = (~self._target_visible).float() * 0.5

        reward = (
            frame_reward
            + altitude_reward
            + descent_reward
            - stability_penalty
            - angular_penalty
            + success_bonus
            - collision_penalty
            - visibility_penalty
        )

        return reward
