import numpy as np
import torch.nn.functional as F
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch as th
from VisFly.utils.randomization import TargetUniformRandomizer, UniformStateRandomizer
from VisFly.utils.type import TensorDict

dl = lambda  x: x.clone().detach()


def get_along_vertical_vector(base, obj):
    base_norm = base.norm(dim=1, keepdim=True)
    obj_norm = obj.norm(dim=1, keepdim=True)
    base_normal = base / (base_norm + 1e-8)
    along_obj_norm = (obj * base_normal).sum(dim=1, keepdim=True)
    along_vector = base_normal * along_obj_norm
    vertical_vector = obj - along_vector
    vertical_obj_norm = vertical_vector.norm(dim=1)
    return along_obj_norm.squeeze(), vertical_obj_norm, base_norm.squeeze()

def smooth_l1_loss_per_row(pred, target, beta: float = 1.0, reduction: str = "mean"):
    """
    pred, target: tensors of shape (m, n)
    beta: transition point for Smooth L1
    reduction: "mean" (default) or "sum" over dim=1, or "none" to return (m,n)
    returns: tensor of shape (m,) when reduction is "mean" or "sum"
    """
    # import torch as th
    diff = pred - target
    abs_diff = diff.abs()
    # if beta <= 0:
    #     loss = abs_diff
    # else:
    mask = abs_diff < beta
    loss = th.where(mask, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
    # if reduction == "mean":
    #     return loss.mean(dim=1)
    # if reduction == "sum":
    #     return loss.sum(dim=1)
    return loss
class NavigationEnv(DroneGymEnvsBase):
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
            sensor_kwargs: list = {},
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
            tensor_output: bool = True,
            max_rand_velocity: float = 7.0,
            target_random: bool = True,
            pos_target: Optional[th.Tensor] = None,
            *args,
            **kwargs
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
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,
            *args,
            **kwargs
        )

        self.pre_define_target = target is not None
        if target is not None:
            if not isinstance(target[0], list):
                self.target = th.ones((self.num_envs, 1)) @ th.as_tensor(target)
            else:
                self.target = th.as_tensor(target)
        else:
            self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([[15, 0., 1]])
        self.max_rand_velocity = max_rand_velocity
        self.success_radius = 0.5

        self.target_randomizers = [
            TargetUniformRandomizer(
            # UniformStateRandomizer(
                    position={
                    # "mean":[15,0,1.5],"half":[14, 14, 1.0]
                    "mean":[0,0,0],"half":[5, 5, 0.0]
                    # "mean":[0,0,0],"half":[max_target_dis, max_target_dis, 1.0]
                },
                min_dis=1.0,
                max_dis=6.0,
                # is_collision_func=self.envs.sceneManager.get_point_is_collision,
                # scene_id=i // self.num_agent_per_scene
            ) for i in range(self.num_envs)
        ]

        if pos_target is not None:
            self.pos_target = th.tensor(pos_target)

    def _reset_attr(self, indices=None,reset_latent=False):
        super()._reset_attr(indices)

        if not self.pre_define_target:
            indices = np.arange(self.num_envs) if indices is None else indices
            for i in indices:
                # pos, _, _, _ = self.target_randomizers[i].safe_generate(1)
                # now consider pos generation as vel
                # pos, _, _, _ = self.target_randomizers[i].safe_generate(1, position=th.zeros_like(self.position[i]))
                # vel
                # self.target[i] = pos[0]
                vel = th.tensor([[30,0,2]])-dl(self.position[i:i+1])
                # vel[:,2] = 0
                vel_unit = vel / (vel.norm(dim=1, keepdim=True)+1e-6)
                self.target[i] = vel_unit * th.rand(1) * self.max_rand_velocity

    def detach(self):
        super().detach()
        if hasattr(self, "_pre_acc"):
            self._pre_acc = self._pre_acc.detach()
    
    def get_observation(
            self,
            indices=None
    ) -> Dict:

        # target cat
        # if self.max_target_dis:
        #     rela = self.target - self.position
        #     unit_rela = rela / (rela.norm(dim=1, keepdim=True)+1e-6)
        #     distance = rela.norm(dim=1, keepdim=True).clip(0, self.max_target_dis)
        #     new_target = (unit_rela * distance+self.position).detach()
        # else:
        #     new_target = self.target
        if hasattr(self, "pos_target"):
            pos_target = self.pos_target.repeat(self.num_scene, 1)
            rela_target = (pos_target - self.position)
            self.target = ((rela_target
                           / rela_target.norm(dim=1, keepdim=True))
                           * (rela_target.norm(dim=1, keepdim=True)/1).clamp_max(self.max_rand_velocity)
                           # * (rela_target.norm(dim=1, keepdim=True)/1).clamp_max(self.max_rand_velocity)
                           )
            # scale = ((1 + self.velocity.norm(dim=1) / (self.target.norm(dim=1)+1e-6)) / 2).clamp_min(1.)
            # self.target = self.target * scale.unsqueeze(1)

        orientation = self.envs.dynamics._orientation.clone()
        # rela = new_target - self.position
        # rela_dis = rela.norm(dim=1, keepdim=True)
        # normal_rela = rela #/ rela_dis.clamp_min(1.0).detach()
        # head_target = orientation.world_to_head(normal_rela.T).T
        head_velocity = orientation.world_to_head((self.velocity-0).T).T
        head_target_velocity = dl(orientation.world_to_head(self.target.T).T)
        state = th.hstack([
            # rela / 10,
            head_target_velocity / 10,
            self.orientation,
            # self.velocity / 10,
            head_velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        max_dis = 24.
        min_dis = 0.1
        scale = 3.

        preprocess = lambda x: 1 / (1 + th.as_tensor(x).clamp(min_dis, max_dis) / scale)

        obs = TensorDict({
            "state": state,
            "depth": preprocess(self.sensor_obs["depth"])
            # "depth": 1 / (1 + th.tensor(self.sensor_obs["depth"]).clamp(min_dis, max_dis) / scale)
            # "depth": th.tensor(self.sensor_obs["depth"]).clamp(min_dis, max_dis) / max_dis
        })

        if "depth2" in list(self.observation_space.keys()):
            obs["depth2"] = th.tensor(self.sensor_obs["depth2"]).clamp(min_dis, max_dis)
            max_pool2 = lambda x: F.max_pool2d(x, kernel_size=2, stride=2)
            avg_pool2 = lambda x: F.avg_pool2d(x, kernel_size=2, stride=2)
            max_pool4 = lambda x: F.max_pool2d(x, kernel_size=4, stride=4)
            avg_pool4 = lambda x: F.avg_pool2d(x, kernel_size=4, stride=4)
            # obs["depth"] = max_pool2(avg_pool2(1/(1+obs["depth2"]/scale)))
            obs["depth"] = max_pool2(1/(1+avg_pool2(obs["depth2"]/scale)))

        #     obs["depth"] = 1 / (1 + f2(obs["depth2"]) / scale)
        return obs

    def get_success(self) -> th.Tensor:
        # return th.zeros((self.num_envs,), dtype=th.bool, device=self.device)
        # return ((self.position - self.target).norm(dim=1) <= self.success_radius) & \
        #         (self.velocity.norm(dim=1) <= 0.05)
        reach_bound = (self.position[:,0]<1) | (self.position[:,0]>=59.) | \
                        (self.position[:,1]>29.) | (self.position[:,1]<=-29.)
        return reach_bound

    def get_reward(self,predicted_obs=None) -> th.Tensor:
        # precise and stable target flight
        base_r = 0.1

        # pos_r = (self.position - self.target).norm(dim=1) * -0.01

        # scale = (self.position - self.target).norm(dim=1).detach().clamp_min(0.3)
        # pos_r = pos_r / scale

        vel_r = (self.velocity - self.target).norm(dim=1)
        adaptive_beta = (self.velocity.norm(dim=1)/6).clamp_min(1.0)
        vel_r = smooth_l1_loss_per_row(vel_r, adaptive_beta) * -0.025
        ang_r = (self.angular_velocity - 0).norm(dim=1) * -0.005

        acc_r = (self.envs.acceleration-0).norm(dim=1).pow(1.3) * -0.002
        # acc_r = smooth_l1_loss_per_row(acc_r, th.zeros_like(acc_r)) * -0.003
        if not hasattr(self, "_pre_acc"):
            self._pre_acc = self.envs.acceleration.clone()
        acc_change_r = ((self.envs.acceleration - self._pre_acc).norm(dim=1)/ self.envs.dynamics.dt).pow(2) \
                       * -0.0001
        self._pre_acc = self.envs.acceleration.clone()
        # act_r = self._action.norm(dim=1).cpu() * -0.001
        act_change_r = (self.envs.dynamics._pre_action[-2].to(self.device).T -
                        self._action.to(self.device)
                        ).norm(dim=-1) / self.envs.dynamics.dt * -0.006

        #  heading alignment
        unit_velocity = self.velocity / (self.velocity.norm(dim=1, keepdim=True)+1e-6)
        align = (unit_velocity * self.direction).sum(dim=1)
        align_r = align * self.velocity.norm(dim=1) * 0.005

        share_factor_collision = -0.6
        # share_factor_collision = 0.0
        # collision penalty
        collision_dis = self.collision_vector.norm(dim=1).clamp_min(0.)
        collision_dir = self.collision_vector / (collision_dis.unsqueeze(1)+1e-6)
        collision_dis = (collision_dis - 0.1).abs()
        # approaching_point = self.envs.approaching_point
        # velocity
        thre_vel = 1.0
        weight = ((thre_vel-collision_dis.detach()).clamp(min=0, )/thre_vel).pow(1)
        # weight = 1 / (1 + ((thre_vel-collision_dis) * 0.3).clamp(min=0,))
        col_approach_velocity = (self.velocity * collision_dir.detach()).sum(dim=1).clamp_min(0.)
        col_vel_r = col_approach_velocity * weight * share_factor_collision

        # position
        k = 0.005
        func = lambda x: 6 * k / (x+k)
        func3 = lambda x: 2.5 * th.log(1+th.exp(-32*x))
        func2 = lambda x: -x
        col_dis_r = func(collision_dis) * share_factor_collision

        reward = {
            "reward": base_r + vel_r + ang_r + align_r
                    + act_change_r + acc_r
                    # + acc_change_r
                    + col_vel_r + col_dis_r
            ,
            # "pos_r": dl(pos_r),
            "vel_r": dl(vel_r),
            "ang_r": dl(ang_r),
            "acc_r": dl(acc_r),
            "align_r": dl(align_r),
            "col_vel_r": dl(col_vel_r),
            "col_dis_r": dl(col_dis_r),
            "acc_change_r": dl(acc_change_r),
            # "act_r": dl(act_r),
            "act_change_r": dl(act_change_r),
        }
        return reward