import os
import sys

import numpy as np
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
from VisFly.utils.tools.train_encoder import model as encoder
from VisFly.utils.type import TensorDict
class intrinsic:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def to(self, device):
        self.fx = th.tensor(self.fx).to(device)
        self.fy = th.tensor(self.fy).to(device)
        self.cx = th.tensor(self.cx).to(device)
        self.cy = th.tensor(self.cy).to(device)
        return self


def get_batch_mask_centers_torch(mask_batch):
    """PyTorch版本"""
    B, H, W = mask_batch.shape
    centers = []

    for b in range(B):
        mask = mask_batch[b]
        indices = th.nonzero(mask, as_tuple=True)

        if len(indices[0]) == 0:
            centers.append(None)
        else:
            center_y = th.mean(indices[0].float())
            center_x = th.mean(indices[1].float())
            centers.append((center_x.item(), center_y.item()))

    return centers


class ObjectTrackingEnv(DroneGymEnvsBase):
    semantic_alias = {
        "stone": 5,
        "cup":8,
        "human":7,
        "ball": 2
    }
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: dict = None,
            dynamics_kwargs: dict = {},
            scene_kwargs: dict = {},
            sensor_kwargs: list = [],
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
            tensor_output: bool = False,
            keep_dis=1.5,
            box_noise=1.0,
            semantic_id =2
    ):
        assert "obj_settings" in scene_kwargs, "scene_kwargs must contain 'obj_settings' for ObjectTrackingEnv"

        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            sensor_kwargs=sensor_kwargs,
            scene_kwargs=scene_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,

        )
        self.target = th.zeros((self.num_envs, 3), dtype=th.float32, device=self.device)
        self.center = th.as_tensor([5, 0, 1.])
        self.radius_spd = 0.2 * th.pi / 1
        self.height = 0.3
        self.radius = 2
        self.box_center = th.ones((self.num_envs, 3), dtype=th.float32, device=self.device) * 0.5
        # self.update_target()
        self.observation_space["state"] = spaces.Box(
            shape=(16,), low=-th.inf, high=th.inf, dtype=np.float32)
        # self.observation_space["state"] = spaces.Box(
        #     shape=(19,), low=-th.inf, high=th.inf, dtype=np.float32)
        test = 1
        # self.update_target()
        self.keep_dis = keep_dis
        self.pre_box_center = th.zeros((self.num_envs, 3), dtype=th.float32, device=self.device)
        self.pre_rebuild_local_targets = th.zeros((self.num_envs, 3), dtype=th.float32, device=self.device)
        # self.pre_box_v = th.zeros((self.num_envs, 3), dtype=th.float32, device=self.device)
        self.pre_dis = th.zeros((self.num_envs,), dtype=th.float32, device=self.device)
        self.smooth_factor = 0.5
        self.box_noise = box_noise
        self.FOV = th.pi/2
        self.semantic_id = semantic_id if isinstance(semantic_id, int) else self.semantic_alias.get(semantic_id, None)

    def reset(self, *args, **kwargs) -> Union[TensorDict, Tuple[TensorDict, Dict]]:
        res = super().reset( *args, **kwargs)

        self.update_target()
        return res

    def _reset_attr(self, indices=None):
        super()._reset_attr(indices)

        indices = indices if indices is not None else th.arange(self.num_envs, device=self.device)
        box_center_cache = get_batch_mask_centers_torch(th.tensor(self.sensor_obs["semantic"] == self.semantic_id).squeeze(dim=1))
        for i in indices:
            center = box_center_cache[i]
            if center is not None:
                self.pre_box_center[i, 0] = (center[0] - self._intrinsic.cx) / self._intrinsic.fx * 2
                self.pre_box_center[i, 1] = (center[1] - self._intrinsic.cy) / self._intrinsic.fy * 2
                self.pre_box_center[i, 2] = th.tensor(self.sensor_obs["depth"][i, 0, int(center[1]), int(center[0])])  # Normalize depth

    def update_target(self):
        if not hasattr(self, "_intrinsic"):
            h, w = self.sensor_obs["depth"].shape[-2:]
            s = max(h, w)
            self._intrinsic = intrinsic(fx=s, fy=s, cx=w / 2, cy=h / 2).to(self.device)

        # update target position and velocity
        self.target = th.stack([p[0] for p in self.envs.dynamic_object_position])
        self.target_v = th.stack([v[0] for v in self.envs.dynamic_object_velocity])

        # update target position and velocity in camera frame
        h, w = self.sensor_obs["semantic"].shape[-2:]
        box_center_cache = get_batch_mask_centers_torch(th.tensor(self.sensor_obs["semantic"] == self.semantic_id).squeeze(dim=1))
        for i, center in enumerate(box_center_cache):
            if center is not None:
                self.box_center[i, 0] = (center[0] - self._intrinsic.cx) / self._intrinsic.fx * 2
                self.box_center[i, 1] = (center[1] - self._intrinsic.cy) / self._intrinsic.fy * 2
                self.box_center[i, 2] = th.tensor(self.sensor_obs["depth"][i, 0, int(center[1]), int(center[0])])

        self.box_velocity = (self.box_center - self.pre_box_center) / self.envs.dynamics.ctrl_dt
        if not hasattr(self, "pre_box_v"):
            self.pre_box_v = self.box_velocity.clone()
        self.box_acc = (self.box_velocity - self.pre_box_v) / self.envs.dynamics.ctrl_dt
        self.pre_box_center = self.box_center.clone()
        self.pre_box_v = self.box_velocity.clone()

        # update rebuilt position and velocity in local frame from camera frame
        self.rebuild_local_targets = self.box_center[:, 2:] * th.stack([
                    th.ones_like(self.box_center[:,1]),
                    -self.box_center[:,0],
                    -self.box_center[:,1]
                ]).T
        self.rebuild_local_targets_v = (self.rebuild_local_targets - self.pre_rebuild_local_targets) / self.envs.dynamics.ctrl_dt
        self.pre_rebuild_local_targets = self.rebuild_local_targets.clone()

        # update target position and velocity in local frame
        rela_tar = self.target - self.position
        orientation = self.envs.dynamics._orientation.clone()
        self.local_targets = orientation.inv_rotate(rela_tar.T).T
        add_local_target_v = th.cross(self.angular_velocity-0, self.local_targets-0, dim=1)
        rela_v = self.target_v - self.velocity
        self.local_targets_v = orientation.inv_rotate(rela_v.T).T - add_local_target_v
        self.rebuild_local_targets_v = self.rebuild_local_targets_v
        self.local_v = orientation.inv_rotate(self.velocity.T-0).T-0

        self.head_targets = orientation.world_to_head(rela_tar.T).T
        if not hasattr(self, "pre_head_targets"):
            self.pre_head_targets = self.head_targets.clone()
        self.head_targets_v = orientation.world_to_head((rela_v.T-0)).T
        # cali_head_vel = th.cross(self.angular_velocity * th.tensor([[0, 0, 1]]), self.head_targets)
        # self.head_targets_v = self.head_targets_v #- cali_head_vel
        self.head_v = orientation.world_to_head((self.velocity.T-0)).T
        test = 1

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        self.update_target()

        state = th.hstack([
            # self.local_targets+th.randn_like(self.box_center) * th.tensor([0.01,0.01, 0.01]) * self.box_noise,
            # self.rebuild_local_targets + th.randn_like(self.box_center) * th.tensor([0.02,0.02, 0.02]) * self.box_noise,
            self.head_targets+th.randn_like(self.box_center) * th.tensor([0.01,0.01, 0.01]) * self.box_noise,
            # self.local_targets_v+th.randn_like(self.box_center) * th.tensor([0.01,0.01, 0.01]) * 5 * self.box_noise,
            self.head_targets_v+th.randn_like(self.box_center) * th.tensor([0.01,0.01, 0.01]) * 5 * self.box_noise,
            # self.rebuild_local_targets_v + th.randn_like(self.box_center) * th.tensor([0.02, 0.02, 0.02]) * 5 * self.box_noise,
            # self.box_velocity+th.randn_like(self.box_center) * th.tensor([0.01,0.01, 0.03]) *10* self.box_noise,
            # self.box_velocity,
            self.orientation,
            self.head_v / 10,
            self.angular_velocity / 10,
        ]).to(self.device)
        return TensorDict({
            "state": state,
        })

        # obs = TensorDict({
        #     "state": state,
        #     "depth": th.as_tensor(self.sensor_obs["depth"]).clamp(0.2, 10),
        #     "semantic": th.as_tensor(self.sensor_obs["semantic"].astype(np.float32)),
        # })

        # if "color" in self.sensor_obs:
        #     obs["color"] = th.as_tensor(self.sensor_obs["color"].astype(np.float32))

        # return obs

    def get_success(self) -> th.Tensor:
        return th.full((self.num_envs,), False)

    def get_reward(self) -> th.Tensor:
        base_r = 0.1 * th.ones((self.num_envs,), dtype=th.float32)
        target_vector = self.target - self.position
        normal_target_vector = target_vector / target_vector.norm(dim=1, keepdim=True) - 0
        proj = ((self.direction.clone() - 0) * normal_target_vector - 0).sum(dim=1)
        aware_r = proj * 0.05
        # aware_r = proj * 0.05
        pos_factor = -0.1 * 1 / 9
        pos_r = (self.position - self.target).norm(dim=1) * pos_factor
        keep_pos_r = ((self.position - self.target).norm(dim=1) - self.keep_dis).abs() * -0.02
        vel_r = (self.velocity - 0).norm(dim=1) * -0.002
        ang_vel_r = (self.angular_velocity - 0).norm(dim=1) * -0.004
        acc_r = (self.envs.acceleration - 0).norm(dim=1) * -0.001
        ang_acc_r = (self.envs.angular_acceleration - 0).norm(dim=1) * -0.001
        act_r = self._action[:,1:].norm(dim=1).to(vel_r.device) * -0.003
        # act_change_r = (self.envs.dynamics._pre_action[0].to(self.device).T-
        #                 self._action.to(self.device)
        #                 ).norm(dim=-1) * -0.002
        act_change_r = (self.envs.dynamics._pre_action[-1].to(self.device)-
                        self.envs.dynamics._pre_action[-2].to(self.device)
                        ).T.norm(dim=-1) * -0.001
        diff_r = vel_r + ang_vel_r + aware_r + keep_pos_r + acc_r+ act_r + act_change_r # + acc_r + ang_acc_r
        disc_r = base_r

        reward = diff_r + disc_r

        return {"reward":reward,
                "keep_pos_r":keep_pos_r.clone().detach(),
                "aware_r":aware_r.clone().detach(),
                "ang_acc_r":ang_acc_r.clone().detach(),}
