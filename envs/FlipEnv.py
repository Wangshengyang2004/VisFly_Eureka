import os
import sys
import math
sys.path.append(os.getcwd())
import numpy as np
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from VisFly.utils.maths import Quaternion
from typing import Optional, Dict, Tuple
import torch as th
from gymnasium import spaces
from VisFly.utils.type import TensorDict


def smooth_l1_loss_per_row(pred, target, beta: float = 1.0, reduction: str = "mean"):
    """Smooth L1 Loss"""
    diff = pred - target
    abs_diff = diff.abs()
    if beta <= 0:
        loss = abs_diff
    else:
        mask = abs_diff < beta
        loss = th.where(mask, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
    return loss


class FlipEnv(DroneGymEnvsBase):
    """Simplified flip environment: HOVER -> FLIP -> HOVER trajectory."""
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
            debug: bool = False,
            debug_interval: int = 50,
            # Trajectory parameters
            hover_before_steps: int = 50,
            flip_steps: int = 80,
            hover_after_steps: int = 126,
    ):
        random_kwargs = {
            "state_generator": {
                "class": "Uniform",
                "kwargs": [{"position": {"mean": [15., 0., 4], "half": [0.5, 0.5, 0.5]}}]
            }
        } if random_kwargs is None else random_kwargs

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

        # 轨迹配置
        self.hover_before_steps = hover_before_steps
        self.flip_steps = flip_steps
        self.hover_after_steps = hover_after_steps
        self.total_steps = hover_before_steps + flip_steps + hover_after_steps

        # 翻滚参数: 目标角度从0到2π
        # flip_steps=0 表示纯悬停模式
        self.target_angle_total = 2 * math.pi  # 总翻滚角度

        # 目标位置 - ensure it's a 1D tensor of shape [3]
        target_value = [15., 0., 4.] if target is None else target
        self.target = th.as_tensor(target_value, device=self.device).flatten()
        if self.target.shape[0] != 3:
            raise ValueError(f"target must have 3 elements, got shape {self.target.shape}")

        # 状态缓存
        self.progress_buf = th.zeros(self.num_envs, dtype=th.long, device=self.device)

        # 目标姿态缓存（四元数 & 旋转向量）
        self.command_quat = Quaternion(num=self.num_envs, device=self.device)
        self.command_quat_diff = th.zeros((self.num_envs, 4), device=self.device)
        self._aligned_current_quat = Quaternion(num=self.num_envs, device=self.device)

        # 相对位置缓存
        self.relative_pos = th.zeros((self.num_envs, 3), device=self.device)
        self.last_action = th.zeros((self.num_envs, 4), device=self.device)

        # Debug
        self.debug = debug
        self.debug_interval = debug_interval

        # 观测维度 (60 dims total): rel_pos(3)+linvel(3)+angvel(3)+orientation(4)+target_pos(3)+quat_diff(4)+future_quat_diff(40)
        self.obs_dim = 60
        self.obs_buf = th.zeros((self.num_envs, self.obs_dim), device=self.device)

        self.observation_space["state"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    def _get_phase_and_angle(self, step: int) -> Tuple[str, float]:
        """获取当前阶段和目标角度"""
        if self.flip_steps == 0:
            return "hover", 0.0

        if step < self.hover_before_steps:
            return "hover_before", 0.0
        elif step < self.hover_before_steps + self.flip_steps:
            # 翻滚阶段: 使用平滑的角度轨迹 (基于余弦插值)
            flip_progress = (step - self.hover_before_steps) / self.flip_steps
            # 使用 (1 - cos(π*p))/2 实现平滑的 0->1 过渡
            smooth_progress = 0.5 * (1 - math.cos(math.pi * flip_progress))
            angle = self.target_angle_total * smooth_progress
            return "flip", angle
        else:
            return "hover_after", self.target_angle_total

    def _compute_target_angle(self, steps: th.Tensor) -> th.Tensor:
        """批量计算目标角度 (roll angle in radians)"""
        angles = th.zeros(len(steps), device=self.device)

        if self.flip_steps == 0:
            return angles

        flip_start = self.hover_before_steps
        transition_width = 10

        # Use tanh for smooth S-curve transition
        for i in range(len(steps)):
            step_val = steps[i].item() if isinstance(steps[i], th.Tensor) else steps[i]
            shift = (step_val - (flip_start + self.flip_steps / 2.0)) / transition_width
            smooth_progress = 0.5 * (1 + math.tanh(shift))
            angles[i] = self.target_angle_total * smooth_progress

        return angles

    def _compute_target_quat(self, steps: th.Tensor) -> Quaternion:
        """
        生成目标姿态四元数（绕 X 轴平滑翻转）。

        Returns raw quaternion WITHOUT sign normalization to avoid discontinuity at θ=π.
        Current quaternion is aligned to this target in get_observation() for bounded error.
        """
        if self.flip_steps == 0:
            # Pure hover: return identity quaternion
            ones = th.ones_like(steps, dtype=th.float32)
            zeros = th.zeros_like(steps, dtype=th.float32)
            return Quaternion(ones, zeros, zeros, zeros)

        flip_start = self.hover_before_steps
        transition_width = 20
        shift = (steps - (flip_start + self.flip_steps / 2.0)) / transition_width
        smooth_progress = 0.5 * (1 + th.tanh(shift))
        angles = self.target_angle_total * smooth_progress

        half_angle = 0.5 * angles
        w = th.cos(half_angle)
        x = th.sin(half_angle)
        zeros = th.zeros_like(w)
        return Quaternion(w, x, zeros, zeros)  # Raw quaternion for continuity

    @staticmethod
    def _quat_to_rotvec(q: Quaternion) -> th.Tensor:
        """
        将四元数转换为旋转向量，保持批量维度。
        """
        v = th.stack([q.x, q.y, q.z], dim=1)
        norm_v = v.norm(dim=1, keepdim=True)
        w = q.w.unsqueeze(1)
        angle = 2.0 * th.atan2(norm_v, w)
        safe_norm = norm_v.clamp(min=1e-8)
        axis = v / safe_norm
        rotvec = axis * angle
        rotvec = th.where(norm_v > 1e-8, rotvec, th.zeros_like(rotvec))
        return rotvec

    @staticmethod
    def _ensure_positive_w(q: Quaternion) -> Quaternion:
        """
        Ensure quaternion has w >= 0 by negating if necessary.
        This gives a unique canonical representation for each rotation.

        WARNING: This causes discontinuity at θ=π for full 2π flips.
        Prefer _align_quat_to_reference() for continuous trajectories.
        """
        sign = th.where(q.w >= 0, 1.0, -1.0)
        return Quaternion(
            q.w * sign,
            q.x * sign,
            q.y * sign,
            q.z * sign
        )

    @staticmethod
    def _align_quat_to_reference(q: Quaternion, ref: Quaternion) -> Quaternion:
        """
        Align quaternion q to the same hemisphere as reference quaternion ref.

        Since q and -q represent the same rotation, we choose the sign that
        minimizes the distance to ref (i.e., dot product >= 0).

        This ensures temporal continuity when ref is the previous timestep's quaternion,
        and proper error computation when ref is the target quaternion.

        Args:
            q: Quaternion to align
            ref: Reference quaternion (previous timestep or target)

        Returns:
            Aligned quaternion (same rotation as q, but in ref's hemisphere)
        """
        dot = q.w * ref.w + q.x * ref.x + q.y * ref.y + q.z * ref.z
        sign = th.where(dot >= 0, 1.0, -1.0)
        return Quaternion(
            q.w * sign,
            q.x * sign,
            q.y * sign,
            q.z * sign
        )


    def _reset_attr(self, indices=None, reset_latent=True):
        """重置环境属性"""
        super()._reset_attr(indices, reset_latent=reset_latent)

        if indices is None:
            indices = th.arange(self.num_envs, device=self.device)
        else:
            indices = th.atleast_1d(indices)

        # Clone to avoid in-place modification of existing graph
        new_progress = self.progress_buf.clone()
        new_progress[indices] = -1  # Will become 0 after get_observation increments
        self.progress_buf = new_progress

        # Reset angular velocity
        self.envs.dynamics._angular_velocity[:, indices] = 0.0

        # Reset action history
        if hasattr(self.envs.dynamics, "_pre_action") and isinstance(self.envs.dynamics._pre_action, th.Tensor):
            self.envs.dynamics._pre_action[:, indices] = 0.0

        # Clone and reset action/command caches
        new_last_action = self.last_action.clone()
        new_last_action[indices] = 0.0
        self.last_action = new_last_action

        new_command_quat_diff = self.command_quat_diff.clone()
        new_command_quat_diff[indices] = 0.0
        self.command_quat_diff = new_command_quat_diff

        # Reallocate quaternion caches
        self.command_quat = Quaternion(num=self.num_envs, device=self.device)
        self._aligned_current_quat = Quaternion(num=self.num_envs, device=self.device)

    def get_observation(self, indices=None) -> Dict:
        """获取观测. Target uses raw quaternion; current is aligned to target's hemisphere."""
        self.progress_buf = self.progress_buf + 1

        # Clone dynamics to avoid in-place modifications
        pos = self.position.clone()
        vel = self.velocity.clone()
        angvel = self.angular_velocity.clone()
        current_quat = self.envs.dynamics._orientation.clone()

        self.relative_pos = self.target.unsqueeze(0) - pos

        # Target: raw quaternion for continuous trajectory
        target_quat = self._compute_target_quat(self.progress_buf)

        # Align current to target's hemisphere for bounded error
        aligned_current = self._align_quat_to_reference(current_quat, target_quat)

        # Quaternion difference: target - aligned_current
        quat_diff = th.stack([
            target_quat.w - aligned_current.w,
            target_quat.x - aligned_current.x,
            target_quat.y - aligned_current.y,
            target_quat.z - aligned_current.z,
        ], dim=1)
        self.command_quat = target_quat
        self.command_quat_diff = quat_diff
        self._aligned_current_quat = aligned_current

        if self.debug:
            step = int(self.progress_buf[0].item())
            if step % self.debug_interval == 0:
                diff_norm = quat_diff[0].norm().item()
                print(f"[Flip] step={step}")
                print(f"  target  = ({target_quat.w[0].item():.3f}, {target_quat.x[0].item():.3f}, {target_quat.y[0].item():.3f}, {target_quat.z[0].item():.3f})")
                print(f"  current = ({aligned_current.w[0].item():.3f}, {aligned_current.x[0].item():.3f}, {aligned_current.y[0].item():.3f}, {aligned_current.z[0].item():.3f})")
                print(f"  |diff|={diff_norm:.3f}")

        # Future 10-step target quaternions
        future_quats = []
        for future_step in range(1, 11):
            future_steps = self.progress_buf + future_step
            future_quat = self._compute_target_quat(future_steps)
            future_quats.append(th.stack([
                future_quat.w,
                future_quat.x,
                future_quat.y,
                future_quat.z,
            ], dim=1))
        future_quats = th.stack(future_quats, dim=1)
        future_quats_flat = future_quats.reshape(self.num_envs, -1)

        # Assemble observation (60 dims): rel_pos(3) + linvel(3) + angvel(3) + quat(4)
        # + target_pos(3) + quat_diff(4) + future_quat(40)
        aligned_orientation = th.stack([
            aligned_current.w,
            aligned_current.x,
            aligned_current.y,
            aligned_current.z,
        ], dim=1)
        obs_state = th.cat([
            self.relative_pos,
            vel,
            angvel,
            aligned_orientation,  # Aligned to target's hemisphere
            self.target.unsqueeze(0).expand(self.num_envs, -1),
            quat_diff,
            future_quats_flat,
        ], dim=1)

        self.obs_buf = obs_state
        return TensorDict({"state": self.obs_buf})

    def get_success(self) -> th.Tensor:
        """Return False - flipping is too hard to define success condition."""
        return th.zeros(self.num_envs, dtype=th.bool, device=self.device)

    def get_reward(self) -> Dict:
        """四元数姿态跟踪 + 惯性系位置奖励"""
        pos = self.position.clone()

        # Rotation error (quat_diff uses aligned current, so bounded)
        rot_norm = self.command_quat_diff.norm(dim=1)
        rotation_reward = 1.0 * th.exp(-rot_norm ** 2)

        # Position error with safe region
        pos_error = pos - self.target.unsqueeze(0)
        pos_norm = pos_error.norm(dim=1)
        safe_radius = 1.0
        dist_outside = th.nn.functional.softplus(pos_norm - safe_radius, beta=10.0)
        pos_loss = dist_outside ** 2
        position_reward = 1.0 * th.exp(-pos_loss)

        reward = rotation_reward + position_reward

        return {
            "reward": reward,
            "rotation_reward": rotation_reward.detach(),
            "position_reward": position_reward.detach(),
        }

    def get_failure(self) -> th.Tensor:
        return self.is_collision 

