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


class FlipEnv(DroneGymEnvsBase):
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
            flip_command: Optional[float] = None,  # Default flip angle (1 full rotation)
    ):
        # Random initialization for hovering position with some variation
        random_kwargs = {
            "state_generator":
                {
                    "class": "Uniform",
                    "kwargs": [
                        {"position": {"mean": [0., 0., 1.5], "half": [0.2, 0.2, 0.1]}},
                    ]
                }
        }

        self.flip_command = flip_command if flip_command is not None else 2 * th.pi
        self.target_position = None  # Will be initialized after device is set
        self.accumulated_rotation = None  # Track cumulative rotation
        self.last_pitch_angle = None  # Track last pitch angle for rotation accumulation
        
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
        
        # Initialize after parent init (so device is set)
        # Ensure we use the correct device from parent class
        actual_device = self.position.device if hasattr(self, 'position') else self.device
        if self.target_position is None:
            self.target_position = th.tensor([0., 0., 1.5], device=actual_device)
        if self.accumulated_rotation is None:
            self.accumulated_rotation = th.zeros(self.num_envs, device=actual_device)
        if self.last_pitch_angle is None:
            self.last_pitch_angle = th.zeros(self.num_envs, device=actual_device)

    def reset(self, indices=None):
        # Call parent reset without indices parameter
        result = super().reset()
        
        # Get actual device from position tensor
        actual_device = self.position.device
        
        # Reset accumulated rotation for specified indices
        if indices is not None:
            if isinstance(indices, (list, np.ndarray)):
                indices = th.tensor(indices, device=actual_device)
            if self.accumulated_rotation.device != actual_device:
                self.accumulated_rotation = self.accumulated_rotation.to(actual_device)
            if self.last_pitch_angle.device != actual_device:
                self.last_pitch_angle = self.last_pitch_angle.to(actual_device)
            self.accumulated_rotation[indices] = 0
            self.last_pitch_angle[indices] = 0
        else:
            self.accumulated_rotation = th.zeros(self.num_envs, device=actual_device)
            self.last_pitch_angle = th.zeros(self.num_envs, device=actual_device)
            
        return result

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        # Ensure accumulated_rotation is on the correct device
        actual_device = self.state.device
        if self.accumulated_rotation.device != actual_device:
            self.accumulated_rotation = self.accumulated_rotation.to(actual_device)
        
        obs = TensorDict({
            "state": self.state,
            "flip_progress": self.accumulated_rotation.clone().unsqueeze(-1) / self.flip_command,  # Normalized flip progress
        })

        return obs

    def get_success(self) -> th.Tensor:
        # Success when drone completes flip and returns to stable hover
        actual_device = self.position.device
        if self.target_position is None:
            self.target_position = th.tensor([0., 0., 1.5], device=actual_device)
        elif self.target_position.device != actual_device:
            self.target_position = self.target_position.to(actual_device)
        
        # Create new tensors to avoid in-place operations
        pos = (self.position - 0)
        vel = (self.velocity - 0)
        
        pos_dist = (pos - self.target_position.unsqueeze(0)).norm(dim=1)
        vel_dist = vel.norm(dim=1)
        
        # Check if flip is completed
        if self.accumulated_rotation.device != actual_device:
            self.accumulated_rotation = self.accumulated_rotation.to(actual_device)
        
        # Require significant rotation (at least 90% of flip_command) and not just starting
        flip_completed = (self.accumulated_rotation >= (self.flip_command * 0.9)) & (self.accumulated_rotation > 1.0)
        
        # Success if flip completed, position error < 0.3m and velocity < 0.5 m/s
        success = flip_completed & (pos_dist < 0.3) & (vel_dist < 0.5)
        
        # Ensure success is on the same device as position
        return success

    def get_reward(self) -> th.Tensor:
        # Initialize target position if needed
        actual_device = self.position.device
        if self.target_position is None:
            self.target_position = th.tensor([0., 0., 1.5], device=actual_device)
        elif self.target_position.device != actual_device:
            self.target_position = self.target_position.to(actual_device)
            
        # Create new tensors to avoid in-place operations
        pos = (self.position - 0)
        vel = (self.velocity - 0)
        ang_vel = (self.angular_velocity - 0) if hasattr(self, 'angular_velocity') else th.zeros((self.num_envs, 3), device=self.device)
        
        # Get current rotation axes from dynamics
        xz_axis = self.envs.dynamics.xz_axis
        # Handle different tensor dimensions
        if xz_axis.dim() == 3:
            x_axis = xz_axis[0, :, :].T  # Extract x-axis (drone's forward direction)
            z_axis = xz_axis[1, :, :].T  # Extract z-axis (drone's up direction)
        else:
            x_axis = xz_axis[:, 0]  # Extract x-axis from 2D tensor
            z_axis = xz_axis[:, 1]  # Extract z-axis from 2D tensor
        
        # Update accumulated rotation based on angular velocity
        pitch_angular_vel = ang_vel[:, 1] # Pitch is rotation around y-axis
        dt = 0.03  # Time step
        # Ensure accumulated_rotation is on the same device
        if self.accumulated_rotation.device != ang_vel.device:
            self.accumulated_rotation = self.accumulated_rotation.to(ang_vel.device)
        self.accumulated_rotation = self.accumulated_rotation.detach() + th.abs(pitch_angular_vel) * dt
        
        # Dense Position Reward - maintain hover position
        relative_pos = pos - self.target_position.unsqueeze(0)
        pos_dist = relative_pos.norm(dim=1)
        
        # Multi-level position rewards for dense feedback
        pos_reward_l0 = th.exp(-0.5 * pos_dist)  # Exponential decay
        pos_reward_l1 = th.exp(-2.0 * pos_dist)  # Faster decay for closer proximity
        pos_reward_l2 = th.exp(-5.0 * pos_dist)  # Even faster decay for very close
        pos_reward = (pos_reward_l0 + pos_reward_l1 + pos_reward_l2) / 3.0
        
        # Dense Stability Reward - penalize high velocities
        vel_magnitude = vel.norm(dim=1)
        ang_vel_magnitude = ang_vel.norm(dim=1)
        
        stability_reward = th.exp(-0.3 * vel_magnitude) * th.exp(-0.1 * ang_vel_magnitude)
        
        # Simplified orientation reward - use built-in orientation quaternion
        # For simplicity, just encourage upright stance when not flipping
        is_flipping = th.abs(pitch_angular_vel) > 2.0  # rad/s threshold
        orientation_reward = th.where(
            is_flipping,
            th.ones(self.num_envs, device=actual_device),  # No orientation penalty during flip
            th.exp(-0.5 * (self.orientation[:, [0, 1]]).norm(dim=1))  # Penalize roll/pitch when not flipping
        )
        
        # Dense Flip Progress Reward
        flip_remaining = th.clamp(self.flip_command - self.accumulated_rotation, min=0)
        flip_progress_ratio = self.accumulated_rotation / (self.flip_command + 1e-6)
        
        # Encourage flipping when not done, encourage stability when done
        flip_not_done = flip_remaining > 0.1
        
        # If flip not done, reward angular velocity in pitch direction
        flip_speed_reward = th.where(
            flip_not_done,
            th.tanh(th.abs(pitch_angular_vel) / 5.0),  # Reward pitch rotation up to 5 rad/s
            th.exp(-2.0 * th.abs(pitch_angular_vel))  # Penalize rotation after flip is done
        )
        
        # Progress reward increases as we get closer to target rotation
        flip_progress_reward = th.sigmoid(10 * (flip_progress_ratio - 0.5))  # Sigmoid centered at 50% progress
        
        # Height maintenance reward (dense)
        height_reward = th.exp(-2.0 * (pos[:, 2] - 1.5).abs())
        
        # Combine all dense rewards
        total_reward = (
            0.2 * pos_reward +
            0.2 * stability_reward +
            0.1 * orientation_reward +
            0.2 * flip_speed_reward +
            0.2 * flip_progress_reward +
            0.1 * height_reward
        )
        
        # Add small alive bonus
        alive_reward = 0.01 * th.ones_like(total_reward)
        
        # Crash penalty (still sparse but necessary for safety)
        crash_penalty = th.where(
            pos[:, 2] < 0.1,
            -10 * th.ones_like(total_reward),
            th.zeros_like(total_reward)
        )
        
        # Large bonus for completing flip (sparse but motivating)
        # Only give bonus if actually rotated enough (at least 90% of target)
        flip_complete_bonus = th.where(
            (self.accumulated_rotation >= (self.flip_command * 0.9)) & (self.accumulated_rotation > 1.0),
            5.0 * th.ones_like(total_reward),
            th.zeros_like(total_reward)
        )
        
        reward = total_reward + alive_reward + crash_penalty + flip_complete_bonus
        
        # Return reward on the same device as position (which is what base class expects)
        return reward