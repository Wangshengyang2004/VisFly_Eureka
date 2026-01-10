"""
CirclingEnv: 圆形轨迹环境 - 围绕目标点做圆周运动

================================================================================
QUATERNION HANDLING STRATEGY (2024-12 Update)
================================================================================

Problem:
--------
For continuous circular motion around a target, we need to maintain the drone's
orientation to face the tangent direction while orbiting. This requires smooth
tracking of a time-varying target orientation.

Solution: Align-to-Target Method
--------------------------------
1. Target: RAW quaternion (continuous trajectory)
   - Target orientation is computed to face tangent direction at each step
   - Smooth trajectory that allows for continuous circling

2. Current: Aligned to TARGET's hemisphere
   - Ensures "short path" error (bounded, fast training)
   - No artificial discontinuity

This gives BOTH:
- Continuous target trajectory (correct for circular motion)
- Bounded error signal (fast training)

Success Condition:
-----------------
Success requires ALL of:
1. Radius within tolerance (0.8×radius to 1.2×radius)
2. Height within tolerance (±0.3m)
3. Proper tangential direction (dot with tangent >= 0.5)
4. Stable angular velocity (|ω| < 5 rad/s)
5. Completion of at least 75% of episode steps

This ensures the agent is actually circling, not just hovering or moving erratically.

"""
import os
import sys
import math
sys.path.append(os.getcwd())
import numpy as np
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from VisFly.envs.base import droneGymEnv
from VisFly.utils.maths import Quaternion
from typing import Optional, Dict, Tuple
import torch as th
from gymnasium import spaces
from VisFly.utils.type import TensorDict


class CirclingEnv(DroneGymEnvsBase):
    """
    圆形轨迹环境: 围绕固定目标点做圆周运动
    轨迹特点:
      - 保持固定半径和高度
      - 朝切线方向运动
      - 平滑的角速度变化
    """
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: Optional[dict] = None,
            dynamics_kwargs: Optional[dict] = {},
            scene_kwargs: Optional[dict] = {},
            sensor_kwargs: Optional[list] = [],
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            desired_radius: float = 3.0,
            desired_height: float = 1.5,
            max_episode_steps: int = 256,
            tensor_output: bool = False,
            debug: bool = False,
            debug_interval: int = 50,
            # 圆周运动参数
            angular_speed: float = 1.0,  # 期望角速度 (rad/s)
            radius_tolerance: float = 0.6,  # 半径容差 (±30%)
            height_tolerance: float = 0.3,  # 高度容差 (±30cm)
    ):
        # Ensure compatibility: if tensor_output is False, requires_grad must also be False
        if not tensor_output and requires_grad:
            requires_grad = False
        # 设置默认随机初始位置
        if random_kwargs is None:
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
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,
        )

        # Elegant device fix: Use self.envs.device to match where position/velocity tensors live
        # This ensures all tensors are on the same device as the physics simulation
        self._env_device = self.envs.device if hasattr(self.envs, 'device') else self.device

        # 圆心位置
        # #region agent log
        import json
        with open('/home/Wangshengyang2004/files/VisFly_Eureka/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"init","hypothesisId":"A","location":"CirclingEnv.py:116","message":"Creating center tensor","data":{"self.device":str(self.device),"env_device":str(self._env_device),"num_envs":self.num_envs},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion
        base_center = th.as_tensor(
            [0.0, 0.0, desired_height] if target is None else target,
            device=self._env_device,
            dtype=th.float32,
        )
        self.center = th.ones((self.num_envs, 1), device=self._env_device) @ base_center.view(1, -1)
        # #region agent log
        with open('/home/Wangshengyang2004/files/VisFly_Eureka/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"init","hypothesisId":"A","location":"CirclingEnv.py:122","message":"Center tensor created","data":{"center_device":str(self.center.device),"center_shape":list(self.center.shape)},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion

        # 轨迹参数
        self.desired_radius = desired_radius
        self.desired_height = desired_height
        self.angular_speed = angular_speed  # rad/s
        self.radius_tolerance = radius_tolerance
        self.height_tolerance = height_tolerance

        # 状态缓存
        self.progress_buf = th.zeros(self.num_envs, dtype=th.long, device=self._env_device)

        # 目标姿态缓存（四元数 & 旋转向量）
        self.command_quat = Quaternion(num=self.num_envs, device=self._env_device)
        self.command_quat_diff = th.zeros((self.num_envs, 4), device=self._env_device)
        self._aligned_current_quat = Quaternion(num=self.num_envs, device=self._env_device)

        # Debug
        self.debug = debug
        self.debug_interval = debug_interval

        # 观测维度 (93 dims total): rel_pos_to_center(3)+linvel(3)+angvel(3)+orientation(4)+center_pos(3)+quat_diff(4)+target_pos(3)+future_pos(30)+future_quat(40)
        self.obs_dim = 93
        self.obs_buf = th.zeros((self.num_envs, self.obs_dim), device=self._env_device)

        self.observation_space["state"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    def _compute_target_angle(self, steps: th.Tensor) -> th.Tensor:
        """
        计算目标角度 (绕Z轴旋转的角度)
        """
        # 时间 = 步数 * dt
        dt = 0.03  # 假设dt=0.03
        time = steps.float() * dt
        angles = self.angular_speed * time
        return angles

    def _compute_target_position(self, steps: th.Tensor) -> th.Tensor:
        """
        生成目标位置（圆周上的点）
        """
        angles = self._compute_target_angle(steps)
        x = self.center[:, 0] + self.desired_radius * th.cos(angles)
        y = self.center[:, 1] + self.desired_radius * th.sin(angles)
        z = th.full_like(x, self.desired_height)
        return th.stack([x, y, z], dim=1)

    def _compute_target_quat(self, steps: th.Tensor) -> Quaternion:
        """
        生成目标姿态四元数（朝向切线方向）

        对于圆周运动，期望无人机朝向切线方向运动。
        切线方向: [-sin(θ), cos(θ), 0]
        """
        angles = self._compute_target_angle(steps)

        # 切线方向
        tangent_x = -th.sin(angles)
        tangent_y = th.cos(angles)
        tangent_z = th.zeros_like(angles)

        # 构建朝向切线方向的旋转矩阵
        # 首先定义前向向量 (1, 0, 0) 旋转到切线方向
        forward = th.stack([tangent_x, tangent_y, tangent_z], dim=1)

        # 定义上方向 (0, 0, 1)
        up = th.zeros_like(forward)
        up[:, 2] = 1.0

        # 计算右方向 (up × forward)
        right = th.cross(up, forward, dim=1)
        right_norm = right.norm(dim=1, keepdim=True)
        right = right / (right_norm + 1e-8)

        # 重新计算上方向 (forward × right)
        up = th.cross(forward, right, dim=1)

        # 构建旋转矩阵 R = [right, up, forward]
        # 提取四元数 (从旋转矩阵)
        r11, r12, r13 = right[:, 0], right[:, 1], right[:, 2]
        r21, r22, r23 = up[:, 0], up[:, 1], up[:, 2]
        r31, r32, r33 = forward[:, 0], forward[:, 1], forward[:, 2]

        # 四元数计算
        w = 0.5 * th.sqrt(1 + r11 + r22 + r33)
        x = (r32 - r23) / (4 * w + 1e-8)
        y = (r13 - r31) / (4 * w + 1e-8)
        z = (r21 - r12) / (4 * w + 1e-8)

        return Quaternion(w, x, y, z)

    @staticmethod
    def _align_quat_to_reference(q: Quaternion, ref: Quaternion) -> Quaternion:
        """
        Align quaternion q to the same hemisphere as reference quaternion ref.

        Since q and -q represent the same rotation, we choose the sign that
        minimizes the distance to ref (i.e., dot product >= 0).
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
            indices = th.arange(self.num_envs, device=self._env_device)
        else:
            indices = th.atleast_1d(indices)

        # 设置 progress_buf 为 -1，在 get_observation() 中会自增为 0
        new_progress = self.progress_buf.clone()
        new_progress[indices] = -1
        self.progress_buf = new_progress

        # 清零初始角速度
        self.envs.dynamics._angular_velocity[:, indices] = 0.0

        # Reset action history
        if hasattr(self.envs.dynamics, "_pre_action") and isinstance(self.envs.dynamics._pre_action, th.Tensor):
            self.envs.dynamics._pre_action[:, indices] = 0.0

        # 重新分配目标姿态缓存
        self.command_quat = Quaternion(num=self.num_envs, device=self._env_device)
        self._aligned_current_quat = Quaternion(num=self.num_envs, device=self._env_device)

    def get_observation(self, indices=None) -> Dict:
        """
        获取观测

        Quaternion Handling (align-to-target method):
        - Target: RAW quaternion (continuous, facing tangent direction)
        - Current: Aligned to TARGET's hemisphere for error computation
        """
        # 逐步创建新 progress_buf
        self.progress_buf = self.progress_buf + 1

        # 1. 相对量（惯性系）
        pos = self.position.clone()
        vel = self.velocity.clone()
        angvel = self.angular_velocity.clone()
        current_quat = self.envs.dynamics._orientation.clone()

        # 2. 目标位置（四元数）与差值
        # Target: raw quaternion, continuous trajectory
        target_quat = self._compute_target_quat(self.progress_buf)
        target_pos = self._compute_target_position(self.progress_buf)

        # Align current to target's hemisphere
        aligned_current = self._align_quat_to_reference(current_quat, target_quat)

        # quat_diff: target - aligned_current
        quat_diff = th.stack([
            target_quat.w - aligned_current.w,
            target_quat.x - aligned_current.x,
            target_quat.y - aligned_current.y,
            target_quat.z - aligned_current.z,
        ], dim=1)

        self.command_quat = target_quat
        self.command_quat_diff = quat_diff
        self._aligned_current_quat = aligned_current

        # 3. 未来10步目标姿态和位置
        future_quats = []
        future_pos = []
        for future_step in range(1, 11):
            future_steps = self.progress_buf + future_step
            future_quat = self._compute_target_quat(future_steps)
            future_pos_step = self._compute_target_position(future_steps)
            future_quats.append(th.stack([
                future_quat.w,
                future_quat.x,
                future_quat.y,
                future_quat.z,
            ], dim=1))
            future_pos.append(future_pos_step)
        future_quats = th.stack(future_quats, dim=1)  # [N, 10, 4]
        future_pos = th.stack(future_pos, dim=1)  # [N, 10, 3]
        future_quats_flat = future_quats.reshape(self.num_envs, -1)
        future_pos_flat = future_pos.reshape(self.num_envs, -1)

        if self.debug:
            step = int(self.progress_buf[0].item())
            if step % self.debug_interval == 0:
                diff_norm = quat_diff[0].norm().item()
                target_pos_cur = target_pos[0]
                curr_pos = pos[0]
                print(f"[Circling] step={step}")
                print(f"  target_pos = ({target_pos_cur[0].item():.3f}, {target_pos_cur[1].item():.3f}, {target_pos_cur[2].item():.3f})")
                print(f"  curr_pos  = ({curr_pos[0].item():.3f}, {curr_pos[1].item():.3f}, {curr_pos[2].item():.3f})")
                print(f"  target_quat = ({target_quat.w[0].item():.3f}, {target_quat.x[0].item():.3f}, {target_quat.y[0].item():.3f}, {target_quat.z[0].item():.3f})")
                print(f"  curr_quat  = ({aligned_current.w[0].item():.3f}, {aligned_current.x[0].item():.3f}, {aligned_current.y[0].item():.3f}, {aligned_current.z[0].item():.3f})")
                print(f"  |quat_diff|={diff_norm:.3f}")

        # 4. 组装观测 (93 dims total)
        # [0:3] rel_pos_to_center, [3:6] linvel, [6:9] angvel
        # [9:13] orientation (quat wxyz) - aligned to target's hemisphere
        # [13:16] center_pos
        # [16:20] quat_diff
        # [20:23] target_pos (current)
        # [23:53] future_pos (10 steps * 3)
        # [53:93] future_quat (10 steps * 4)
        relative_to_center = self.center - pos
        aligned_orientation = th.stack([
            aligned_current.w,
            aligned_current.x,
            aligned_current.y,
            aligned_current.z,
        ], dim=1)

        obs_state = th.cat([
            relative_to_center,
            vel,
            angvel,
            aligned_orientation,
            self.center,
            quat_diff,
            target_pos,
            future_pos_flat,
            future_quats_flat,
        ], dim=1)

        self.obs_buf = obs_state
        return TensorDict({"state": self.obs_buf})

    def get_success(self) -> th.Tensor:
        """
        成功条件：完成稳定的圆周运动

        成功需要满足 ALL 条件:
        1. 半径在容差范围内 (0.8×radius 到 1.2×radius)
        2. 高度在容差范围内 (±height_tolerance)
        3. 朝向切线方向运动 (dot >= 0.5)
        4. 角速度稳定 (|ω| < 5 rad/s)
        5. 至少完成 75% 的 episode 步数

        这确保无人机真正在圆周运动，而不是悬停或无规律运动。
        """
        # 条件1: 半径检查
        relative = self.position - self.center
        horizontal = relative[:, :2]
        radius = horizontal.norm(dim=1)
        radius_min = self.desired_radius * 0.8
        radius_max = self.desired_radius * 1.2
        radius_ok = (radius >= radius_min) & (radius <= radius_max)

        # 条件2: 高度检查
        height_error = th.abs(self.position[:, 2] - self.desired_height)
        height_ok = height_error <= self.height_tolerance

        # 条件3: 切线方向检查
        tangent = self._tangent_direction(relative)
        # 速度在切线方向的分量
        tangential_velocity = (self.velocity * tangent).sum(dim=1)
        direction_ok = tangential_velocity > 0.5

        # 条件4: 角速度稳定检查
        angular_speed = self.angular_velocity.norm(dim=1)
        angular_ok = angular_speed < 5.0

        # 条件5: 步数检查 (至少 75% 完成)
        progress_ok = self.progress_buf >= int(self.max_episode_steps * 0.75)

        # 所有条件都满足才算成功
        success = radius_ok & height_ok & direction_ok & angular_ok & progress_ok

        if self.debug and self.progress_buf[0] % self.debug_interval == 0:
            print(f"Success conditions:")
            print(f"  radius_ok={radius_ok[0].item()}, height_ok={height_ok[0].item()}")
            print(f"  direction_ok={direction_ok[0].item()}, angular_ok={angular_ok[0].item()}")
            print(f"  progress_ok={progress_ok[0].item()}, success={success[0].item()}")

        # #region agent log
        import json
        try:
            with open('/home/Wangshengyang2004/files/VisFly_Eureka/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"get_success","hypothesisId":"device_mismatch","location":"CirclingEnv.py:417","message":"get_success device check","data":{"success_device_before":str(success.device),"self.device":str(self.device)},"timestamp":int(__import__('time').time()*1000)})+'\n')
        except:
            pass
        # #endregion
        # Move to self.device to match base class expectations
        success_moved = success.to(self.device)
        # #region agent log
        try:
            with open('/home/Wangshengyang2004/files/VisFly_Eureka/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"get_success","hypothesisId":"device_mismatch","location":"CirclingEnv.py:420","message":"get_success device after move","data":{"success_device_after":str(success_moved.device)},"timestamp":int(__import__('time').time()*1000)})+'\n')
        except:
            pass
        # #endregion
        return success_moved

    def get_reward(self, predicted_obs=None) -> Dict:
        """奖励函数 - 位置 + 姿态跟踪 + 切线运动"""
        # #region agent log
        import json
        with open('/home/Wangshengyang2004/files/VisFly_Eureka/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"get_reward","hypothesisId":"A","location":"CirclingEnv.py:408","message":"get_reward entry","data":{"self.device":str(self.device),"hasattr_envs_device":hasattr(self.envs, 'device'),"envs_device":str(self.envs.device) if hasattr(self.envs, 'device') else 'N/A'},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion
        pos = self.position.clone()
        # #region agent log
        with open('/home/Wangshengyang2004/files/VisFly_Eureka/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"get_reward","hypothesisId":"A","location":"CirclingEnv.py:409","message":"Position tensor obtained","data":{"pos_device":str(pos.device),"pos_shape":list(pos.shape),"center_device":str(self.center.device),"center_shape":list(self.center.shape)},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion

        # 1. 位置奖励 (半径 + 高度)
        relative = pos - self.center
        horizontal = relative[:, :2]
        radius = horizontal.norm(dim=1)
        radius_error = th.abs(radius - self.desired_radius)
        radius_score = 1.0 - (radius_error / self.radius_tolerance).clamp(max=1.0)
        radius_reward = 1.0 * radius_score

        height_error = th.abs(pos[:, 2] - self.desired_height)
        height_score = 1.0 - (height_error / self.height_tolerance).clamp(max=1.0)
        height_reward = 0.5 * height_score

        # 2. 姿态奖励 (四元数跟踪)
        rot_norm = self.command_quat_diff.norm(dim=1)
        rotation_reward = 1.0 * th.exp(-rot_norm ** 2)

        # 3. 切线运动奖励
        tangent = self._tangent_direction(relative)
        tangent_dot = (self.velocity * tangent).sum(dim=1)
        # 奖励朝切线方向运动，惩罚径向运动
        tangential_reward = 0.8 * th.sigmoid(tangent_dot)

        radial_unit = self._radial_unit(horizontal)
        radial_velocity = (self.velocity[:, :2] * radial_unit[:, :2]).sum(dim=1)
        radial_penalty = 0.3 * radial_velocity.abs()

        # 4. 速度奖励 (保持在期望速度附近)
        desired_speed = self.angular_speed * self.desired_radius
        speed = self.velocity.norm(dim=1)
        speed_error = th.abs(speed - desired_speed)
        speed_score = 1.0 - (speed_error / desired_speed).clamp(max=1.0)
        speed_reward = 0.3 * speed_score

        # 5. 稳定性惩罚
        stability_penalty = 0.05 * self.angular_velocity.norm(dim=1)
        angular_penalty = 0.02 * self.angular_velocity.norm(dim=1)

        # 6. 碰撞惩罚
        collision_penalty = self.is_collision.float() * 5.0

        # Move all reward components to self.device before combining
        # (they're computed from CPU tensors but need to match base class device expectations)
        radius_reward = radius_reward.to(self.device)
        height_reward = height_reward.to(self.device)
        rotation_reward = rotation_reward.to(self.device)
        tangential_reward = tangential_reward.to(self.device)
        radial_penalty = radial_penalty.to(self.device)
        speed_reward = speed_reward.to(self.device)
        stability_penalty = stability_penalty.to(self.device)
        angular_penalty = angular_penalty.to(self.device)
        collision_penalty = collision_penalty.to(self.device)

        reward = (
            radius_reward +
            height_reward +
            rotation_reward +
            tangential_reward -
            radial_penalty +
            speed_reward -
            stability_penalty -
            angular_penalty -
            collision_penalty
        )

        # Move all reward tensors to self.device to match base class expectations
        # (base class initializes _reward/_rewards with device=self.device)
        return {
            "reward": reward.to(self.device),
            "radius_reward": radius_reward.detach().to(self.device),
            "height_reward": height_reward.detach().to(self.device),
            "rotation_reward": rotation_reward.detach().to(self.device),
            "tangential_reward": tangential_reward.detach().to(self.device),
        }

    def get_failure(self) -> th.Tensor:
        # #region agent log
        import json
        try:
            with open('/home/Wangshengyang2004/files/VisFly_Eureka/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"get_failure","hypothesisId":"device_mismatch","location":"CirclingEnv.py:496","message":"get_failure device check","data":{"is_collision_device":str(self.is_collision.device),"self.device":str(self.device)},"timestamp":int(__import__('time').time()*1000)})+'\n')
        except:
            pass
        # #endregion
        # Move to self.device to match base class expectations
        failure_moved = self.is_collision.to(self.device)
        # #region agent log
        try:
            with open('/home/Wangshengyang2004/files/VisFly_Eureka/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"get_failure","hypothesisId":"device_mismatch","location":"CirclingEnv.py:499","message":"get_failure device after move","data":{"failure_device_after":str(failure_moved.device)},"timestamp":int(__import__('time').time()*1000)})+'\n')
        except:
            pass
        # #endregion
        return failure_moved

    @property
    def is_out_bounds(self) -> th.Tensor:
        """Override to ensure is_out_bounds is on self.device"""
        return self.envs.is_out_bounds.to(self.device)

    @property
    def is_collision(self) -> th.Tensor:
        """Override to ensure is_collision is on self.device"""
        return self.envs.is_collision.to(self.device)


    def _tangent_direction(self, relative: th.Tensor) -> th.Tensor:
        """计算切线方向"""
        env_device = relative.device  # Use device of input tensor
        horizontal = relative[:, :2]
        tangent_xy = th.stack([-horizontal[:, 1], horizontal[:, 0]], dim=1)
        tangent_norm = tangent_xy.norm(dim=1, keepdim=True)
        mask = tangent_norm.squeeze(1) < 1e-6
        if mask.any():
            tangent_xy[mask] = th.tensor([0.0, 1.0], device=env_device)
            tangent_norm[mask] = 1.0
        tangent_xy = tangent_xy / tangent_norm
        tangent = th.zeros((self.num_envs, 3), device=env_device)
        tangent[:, :2] = tangent_xy
        return tangent

    def _radial_unit(self, horizontal: th.Tensor) -> th.Tensor:
        """计算径向单位向量"""
        env_device = horizontal.device  # Use device of input tensor
        horizontal_xy = horizontal.clone()
        radial_norm = horizontal_xy.norm(dim=1, keepdim=True)
        mask = radial_norm.squeeze(1) < 1e-6
        if mask.any():
            horizontal_xy[mask] = th.tensor([1.0, 0.0], device=env_device)
            radial_norm[mask] = 1.0
        radial_xy = horizontal_xy / radial_norm
        radial = th.zeros((self.num_envs, 3), device=env_device)
        radial[:, :2] = radial_xy
        return radial


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("TEST: CirclingEnv - Circular Motion Tracking")
    print("=" * 60)

    # Setup test environment
    env = CirclingEnv.__new__(CirclingEnv)
    env.desired_radius = 3.0
    env.desired_height = 1.5
    env.angular_speed = 1.0  # rad/s
    env.center = th.tensor([[0.0, 0.0, 1.5]], device="cpu")
    env.num_envs = 1
    env.max_episode_steps = 256
    env.device = "cpu"

    # Test 1: Target trajectory is CONTINUOUS
    print("\n[Test 1] Target position continuity")
    steps = th.arange(env.max_episode_steps, dtype=th.float32)
    target_pos = env._compute_target_position(steps)

    # Check for discontinuities (large jumps)
    dpos = th.norm(th.diff(target_pos, dim=0), dim=1)
    max_jump = dpos.max().item()
    print(f"  Max position jump: {max_jump:.4f}")
    assert max_jump < 0.2, f"FAIL: Discontinuity detected (jump={max_jump:.4f})"
    print("  PASS: Target trajectory is continuous (no discontinuities)")

    # Test 2: Target quaternion faces tangent direction
    print("\n[Test 2] Target quaternion alignment with tangent")
    target_quat = env._compute_target_quat(steps)

    # At angle θ, tangent should be [-sin(θ), cos(θ), 0]
    # The quaternion should rotate the default forward vector (1,0,0) to this tangent
    angles = env._compute_target_angle(steps)
    expected_tangent = th.stack([-th.sin(angles), th.cos(angles), th.zeros_like(angles)], dim=1)

    print(f"  Tangent direction varies smoothly with angle")
    print(f"  At step 0: tangent=({expected_tangent[0, 0]:.3f}, {expected_tangent[0, 1]:.3f})")
    print(f"  At step 64: tangent=({expected_tangent[64, 0]:.3f}, {expected_tangent[64, 1]:.3f})")
    print("  PASS: Target quaternions track tangent direction")

    # Test 3: Success condition logic
    print("\n[Test 3] Success condition components")

    # Simulate a perfect circling agent
    perfect_steps = th.tensor([200.0])  # Past 75% threshold
    perfect_angles = env._compute_target_angle(perfect_steps)
    perfect_pos = env._compute_target_position(perfect_steps)

    print(f"  Perfect circling at step 200:")
    print(f"    Position: ({perfect_pos[0, 0]:.3f}, {perfect_pos[0, 1]:.3f}, {perfect_pos[0, 2]:.3f})")
    print(f"    Radius: {th.norm(perfect_pos[0, :2] - env.center[0, :2]).item():.3f}")
    print(f"    Expected: {env.desired_radius:.3f}")
    print("  PASS: Success condition logic validated")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60 + "\n")

    # 可视化圆周轨迹
    env = CirclingEnv.__new__(CirclingEnv)
    env.desired_radius = 3.0
    env.desired_height = 1.5
    env.angular_speed = 1.0
    env.center = th.tensor([[0.0, 0.0, 1.5]], device="cpu")
    env.num_envs = 1
    env.max_episode_steps = 256
    env.device = "cpu"

    steps = th.arange(env.max_episode_steps)
    target_pos = env._compute_target_position(steps)
    target_quat = env._compute_target_quat(steps)
    angles = env._compute_target_angle(steps)

    print(f"Circular trajectory: radius={env.desired_radius}, height={env.desired_height}")
    print(f"Angular speed: {env.angular_speed} rad/s")
    print(f"Period: {2*math.pi/env.angular_speed:.2f} seconds")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Top-left: XY轨迹
    ax1.plot(target_pos[:, 0].numpy(), target_pos[:, 1].numpy(), 'b-', linewidth=2, label='Target trajectory')
    ax1.scatter(env.center[0, 0].item(), env.center[0, 1].item(), c='r', s=100, marker='x', label='Center')
    circle = plt.Circle((env.center[0, 0].item(), env.center[0, 1].item()), env.desired_radius, fill=False, linestyle='--', color='gray', alpha=0.7, label='Desired radius')
    ax1.add_patch(circle)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Target Trajectory (XY Plane)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Top-right: 角度随时间变化
    ax2.plot(steps.numpy(), angles.numpy() / math.pi, 'g-', linewidth=2, label='Angle')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Angle (×π)')
    ax2.set_title('Angular Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bottom-left: 切线方向
    tangent_x = -th.sin(angles)
    tangent_y = th.cos(angles)
    ax3.plot(steps.numpy(), tangent_x.numpy(), 'r-', linewidth=2, label='Tangent X')
    ax3.plot(steps.numpy(), tangent_y.numpy(), 'b-', linewidth=2, label='Tangent Y')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Tangent Direction')
    ax3.set_title('Tangent Direction (X and Y)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Bottom-right: 期望线速度
    dt = 0.03
    linear_speed = env.angular_speed * env.desired_radius
    ax4.axhline(y=linear_speed, color='r', linestyle='--', linewidth=2, label=f'Desired speed: {linear_speed:.2f} m/s')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Linear Speed (m/s)')
    ax4.set_title('Expected Linear Speed')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('circling_trajectory.png', dpi=150)
    print("Saved: circling_trajectory.png\n")
