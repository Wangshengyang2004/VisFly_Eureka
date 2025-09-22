from typing import Dict, Optional

import torch as th
from habitat_sim.sensor import SensorType

from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from VisFly.utils.type import TensorDict


class CirclingEnv(DroneGymEnvsBase):
    """Environment that encourages the drone to orbit a fixed target."""

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
        desired_radius: float = 3.0,
        desired_height: float = 1.5,
        max_episode_steps: int = 256,
        tensor_output: bool = False,
    ):
        if not sensor_kwargs:
            sensor_kwargs = [
                {
                    "sensor_type": SensorType.DEPTH,
                    "uuid": "depth",
                    "resolution": [64, 64],
                }
            ]

        if not random_kwargs:
            random_kwargs = {
                "state_generator": {
                    "class": "Uniform",
                    "kwargs": [
                        {
                            "position": {
                                "mean": [desired_radius + 0.5, 0.0, desired_height + 0.2],
                                "half": [1.0, 1.0, 0.3],
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

        base_center = th.as_tensor(
            [0.0, 0.0, desired_height] if target is None else target,
            device=self.device,
            dtype=th.float32,
        )
        self.center = th.ones((self.num_envs, 1), device=self.device) @ base_center.view(1, -1)
        self.desired_radius = desired_radius
        self.desired_height = desired_height
        self.radius_tolerance = max(desired_radius, 1.0)
        self.height_tolerance = 0.5

    def get_observation(self, indices=None) -> Dict:
        orientation = self.envs.dynamics._orientation.clone()
        relative_to_center = self.center - self.position
        head_target = orientation.world_to_head(relative_to_center.T).T
        head_velocity = orientation.world_to_head(self.velocity.T).T

        radius = (self.position[:, :2] - self.center[:, :2]).norm(dim=1, keepdim=True)
        radius_error = radius - self.desired_radius
        height_error = (self.position[:, 2] - self.desired_height).unsqueeze(1)

        state = th.hstack(
            [
                head_target / 10.0,
                self.orientation,
                head_velocity / 10.0,
                self.angular_velocity / 10.0,
            ]
        )

        if self.visual and "depth" in self.sensor_obs:
            return TensorDict(
                {
                    "state": state,
                    "depth": th.as_tensor(self.sensor_obs["depth"], dtype=th.float32).clamp(max=10.0),
                }
            )
        else:
            return TensorDict({"state": state})

    def get_success(self) -> th.Tensor:
        return th.full((self.num_envs,), False, dtype=th.bool, device=self.device)

    def get_failure(self) -> th.Tensor:
        return self.is_collision

    def get_reward(self) -> th.Tensor:
        relative = self.position - self.center
        horizontal = relative[:, :2]
        radius = horizontal.norm(dim=1)
        radius_error = th.abs(radius - self.desired_radius)
        radius_score = 1.0 - (radius_error / (self.radius_tolerance + 1e-6)).clamp(max=1.0)
        radius_reward = radius_score * 0.5

        height_error = th.abs(self.position[:, 2] - self.desired_height)
        height_score = 1.0 - (height_error / self.height_tolerance).clamp(max=1.0)
        height_reward = height_score * 0.2

        tangent = self._tangent_direction(relative)
        heading_reward = 0.2 * (self.direction * tangent).sum(dim=1)

        tangential_speed = (self.velocity * tangent).sum(dim=1)
        speed_reward = 0.2 * th.tanh(tangential_speed)

        radial_unit = self._radial_unit(horizontal)
        radial_velocity = (self.velocity[:, :2] * radial_unit[:, :2]).sum(dim=1)
        radial_penalty = 0.1 * radial_velocity.abs()

        stability_penalty = 0.02 * self.angular_velocity.norm(dim=1)
        speed_penalty = 0.02 * self.velocity.norm(dim=1)
        collision_penalty = self.is_collision.float() * 5.0

        reward = (
            radius_reward
            + height_reward
            + heading_reward
            + speed_reward
            - radial_penalty
            - stability_penalty
            - speed_penalty
            - collision_penalty
        )

        return reward

    def _tangent_direction(self, relative: th.Tensor) -> th.Tensor:
        horizontal = relative[:, :2]
        tangent_xy = th.stack([-horizontal[:, 1], horizontal[:, 0]], dim=1)
        tangent_norm = tangent_xy.norm(dim=1, keepdim=True)
        mask = tangent_norm.squeeze(1) < 1e-6
        if mask.any():
            tangent_xy[mask] = th.tensor([0.0, 1.0], device=self.device)
            tangent_norm[mask] = 1.0
        tangent_xy = tangent_xy / tangent_norm
        tangent = th.zeros((self.num_envs, 3), device=self.device)
        tangent[:, :2] = tangent_xy
        return tangent

    def _radial_unit(self, horizontal: th.Tensor) -> th.Tensor:
        horizontal_xy = horizontal.clone()
        radial_norm = horizontal_xy.norm(dim=1, keepdim=True)
        mask = radial_norm.squeeze(1) < 1e-6
        if mask.any():
            horizontal_xy[mask] = th.tensor([1.0, 0.0], device=self.device)
            radial_norm[mask] = 1.0
        radial_xy = horizontal_xy / radial_norm
        radial = th.zeros((self.num_envs, 3), device=self.device)
        radial[:, :2] = radial_xy
        return radial
