"""
Prompt Engineering for VisFly Reward Function Generation

This module contains system and user prompts optimized for generating
reward functions for VisFly drone environments.
"""

from typing import Dict, Any


def create_system_prompt() -> str:
    """Create system prompt for LLM reward function generation."""
    return """You are a reward function designer for VisFly drone environments. Generate complete get_reward methods for reinforcement learning.

Environment Variables (always available):
- self.position: [N, 3] drone positions
- self.velocity: [N, 3] linear velocities  
- self.orientation: [N, 4] quaternions
- self.angular_velocity: [N, 3] angular velocities
- self.target: [N, 3] target positions
- self.collision_dis: [N] obstacle distances
- self.sensor_obs['depth']: [N, H, W] depth camera
- self._step_count: current step
- self.num_agent: agent count

Requirements:
1. Return complete get_reward(self) method
2. Use torch operations only
3. Return shape [N] tensor
4. No imports needed (torch/th available)
5. Be confident - all listed variables exist, use them directly

BPTT Gradient Tips (CRITICAL):
- ALWAYS use (self.velocity - 0) to avoid in-place gradient issues
- ALWAYS use (self.angular_velocity - 0) for gradient safety
- Example: velocity_penalty = -torch.norm(self.velocity - 0, dim=1)
- This creates new tensors that preserve gradient flow

Common Patterns:
- Distance: -torch.norm(self.position - self.target, dim=1)
- Velocity penalty: -torch.norm(self.velocity - 0, dim=1) * 0.01
- Collision: -1.0 / (self.collision_dis + 0.2)
- Stability: -torch.norm(self.angular_velocity - 0, dim=1) * 0.001

Numerical Safety:
- Use torch.clamp(x, min=1e-8) only for division denominators
- Write clean, confident code without excessive checks
- Trust the environment structure"""


def get_navigation_env_code() -> str:
    """
    Get pre-extracted NavigationEnv code without reward function (avoids blocking environment creation).
    
    Returns:
        NavigationEnv source code with get_reward method stripped
    """
    return """import numpy as np
from habitat_sim.sensor import SensorType

from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces

from VisFly.utils.type import TensorDict
from math import pi
from typing import Union


def get_along_vertical_vector(base, obj):
    \"\"\"
    Decompose obj vector into components along and perpendicular to base vector
    BPTT-safe: no in-place operations that break gradient computation
    \"\"\"
    # Safe norm computation with minimum clipping to avoid zero gradients
    base_norm = th.clamp(base.norm(dim=1, keepdim=True), min=1e-8)
    obj_norm = th.clamp(obj.norm(dim=1, keepdim=True), min=1e-8)

    # Safe division for normalization
    base_normal = base / base_norm
    along_obj_norm = (obj * base_normal).sum(dim=1, keepdim=True)
    along_vector = base_normal * along_obj_norm
    vertical_vector = obj - along_vector
    vertical_obj_norm = th.clamp(vertical_vector.norm(dim=1), min=1e-8)

    # Ensure we return new tensors to avoid in-place modification issues  
    return along_obj_norm.squeeze() + 0.0, vertical_obj_norm + 0.0, base_norm.squeeze() + 0.0


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
        sensor_kwargs: list = [],
        device: str = "cpu",
        target: Optional[th.Tensor] = None,
        max_episode_steps: int = 256,
        tensor_output: bool = False,
    ):
        # Ensure depth sensor is always available for NavigationEnv
        if not sensor_kwargs:
            sensor_kwargs = [
                {
                    "sensor_type": SensorType.DEPTH,
                    "uuid": "depth",
                    "resolution": [64, 64],
                }
            ]

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

        # Fix device mismatch: ensure target is on correct device
        target_pos = [15, 0.0, 1.5] if target is None else target
        self.target = th.ones((self.num_envs, 1), device=self.device) @ th.as_tensor(
            target_pos, device=self.device
        ).reshape(1, -1)
        self.observation_space["state"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        self.observation_space["depth"] = spaces.Box(
            low=0.0, high=1.0, shape=(1, 64, 64), dtype=np.float32
        )
        self.observation_space["target"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )


    def get_observation(
                self,
                indices=None
        ) -> Dict:
        orientation = self.envs.dynamics._orientation.clone()
        rela = self.target - self.position
        head_target = orientation.world_to_head(rela.T).T
        head_velocity = orientation.world_to_head((self.velocity - 0).T).T  # Note: (velocity - 0) for BPTT gradient safety

        # State observations
        obs = {
            "state": th.cat(
                [
                    head_target,
                    head_velocity,
                    self.angular_velocity,
                    self.orientation,
                ],
                dim=1,
            )
        }

        # Depth sensor observations
        if "depth" in self.obs_keys:
            depth_obs = self.envs.get_sensor_observation(
                self.observation_indices if indices is None else indices
            )["depth"]
            obs["depth"] = depth_obs

        # Target vector for reward function
        obs["target"] = self.target

        return obs

    def get_reward(self) -> th.Tensor:
        \"\"\"
        Reward function to be implemented by Eureka.
        
        Available attributes:
        - self.position: torch.Tensor [N, 3] - drone positions
        - self.velocity: torch.Tensor [N, 3] - linear velocities  
        - self.target: torch.Tensor [N, 3] - target positions
        - self.collision_vector: torch.Tensor [N, 3] - collision avoidance vector
        - self.angular_velocity: torch.Tensor [N, 3] - angular velocities
        - self.direction: torch.Tensor [N, 3] - heading direction
        - self._step_count: int - current episode step
        - self.max_episode_steps: int - maximum steps per episode
        
        Sensor observations via self.get_observation():
        - 'depth': torch.Tensor [N, H, W] - depth camera
        - 'state': torch.Tensor [N, 13] - state vector
        
        Returns:
            torch.Tensor [N] - reward for each agent
        \"\"\"
        # TODO: Implement reward function
        return th.zeros(self.num_envs, device=self.device)
    
    def get_failure(self) -> th.Tensor:
        return self.is_collision

    def get_success(self) -> th.Tensor:
        # Check if agent reaches within threshold of target
        target_distance = th.norm(self.position - self.target, dim=1)
        velocity_magnitude = th.norm(self.velocity, dim=1)
        
        # Success criteria: within distance threshold AND low velocity (stable)
        distance_success = target_distance < 0.5  # Within 0.5m of target
        velocity_success = velocity_magnitude < 0.1  # Moving slowly (stable)
        
        return distance_success & velocity_success

    def reset(self, indices=None):
        \"\"\"Reset environment with new target and random initial positions\"\"\"
        super().reset(indices)
        # Environment is well-defined, target is always available
        return self.get_observation()
"""


def extract_env_code_without_reward(env_class) -> str:
    """
    Extract environment code with reward function removed.
    
    Args:
        env_class: Environment class to extract code from
        
    Returns:
        Environment source code with get_reward method stripped
    """
    # Use pre-extracted code for NavigationEnv
    if env_class.__name__ == 'NavigationEnv':
        return get_navigation_env_code()
    else:
        # Fallback for other environments
        return f"# Environment: {env_class.__name__}\n# Pre-extracted code not available for this environment type."


def create_user_prompt(
    task_description: str,
    context_info: Dict[str, Any],
    feedback: str = "",
    env_code: str = ""
) -> str:
    """
    Create a user prompt for a specific task and context (like real Eureka).
    
    Args:
        task_description: Natural language description of the task
        context_info: Environment-specific context information
        feedback: Feedback from previous iterations
        env_code: Complete environment code with reward stripped
        
    Returns:
        Formatted user prompt string
    """
    prompt_parts = []
    
    # Environment code (main component like real Eureka)
    if env_code:
        prompt_parts.append("The Python environment is:")
        prompt_parts.append("```python")
        prompt_parts.append(env_code)
        prompt_parts.append("```")
        prompt_parts.append("")
    
    # Task description  
    prompt_parts.append(f"Write a reward function for the following task: {task_description}")
    
    # Previous iteration feedback (if any)
    if feedback:
        prompt_parts.append(f"\nFeedback from previous attempts:")
        prompt_parts.append(feedback)
    
    # Request (simplified like real Eureka)
    prompt_parts.append("""
Please provide only the complete get_reward(self) method implementation.""")
    
    return "\n".join(prompt_parts)


def create_improvement_prompt(
    original_code: str,
    performance_issues: str,
    task_description: str
) -> str:
    """Create prompt for improving existing reward function."""
    return f"""Task: {task_description}

Current reward has issues:
```python
{original_code}
```

Issues: {performance_issues}

Fix these problems. Remember BPTT gradient tips:
- Use (self.velocity - 0) not self.velocity directly
- Use (self.angular_velocity - 0) not self.angular_velocity directly

Generate improved get_reward(self) method:"""


def create_context_aware_prompt(
    task_description: str,
    environment_observations: Dict[str, Any],
    sensor_data_shapes: Dict[str, tuple],
    previous_rewards: list = None
) -> str:
    """Create context-aware prompt using actual environment observations."""
    prompt_parts = [f"Task: {task_description}"]
    
    # Direct environment state (confident about tensor shapes)
    if environment_observations:
        prompt_parts.append("\nEnvironment State:")
        for key, value in environment_observations.items():
            # Assume tensors have shape attribute
            prompt_parts.append(f"- {key}: shape {value.shape}")
    
    # Sensor shapes
    if sensor_data_shapes:
        prompt_parts.append("\nSensor Shapes:")
        for sensor_name, shape in sensor_data_shapes.items():
            prompt_parts.append(f"- {sensor_name}: {shape}")
    
    # Previous attempts
    if previous_rewards:
        prompt_parts.append(f"\nTried {len(previous_rewards)} functions. Generate a different approach.")
    
    prompt_parts.append("\nUse (self.velocity - 0) and (self.angular_velocity - 0) for BPTT.\nGenerate get_reward(self):")
    
    return "\n".join(prompt_parts)


# Task-specific hints (concise)
NAVIGATION_PROMPT_SUFFIX = """
Navigation: Focus on distance to target, collision avoidance via depth, smooth velocity."""

RACING_PROMPT_SUFFIX = """
Racing: Maximize forward velocity, gate passing, optimize racing line."""

HOVERING_PROMPT_SUFFIX = """
Hovering: Minimize position/velocity errors, stabilize orientation. Use (self.velocity - 0)."""

TRACKING_PROMPT_SUFFIX = """
Tracking: Maintain relative position, smooth following, visual tracking."""


def get_task_specific_prompt_suffix(task_description: str) -> str:
    """
    Get task-specific prompt suffix based on task description.
    
    Args:
        task_description: Natural language task description
        
    Returns:
        Task-specific prompt suffix
    """
    task_lower = task_description.lower()
    
    if any(keyword in task_lower for keyword in ["navigate", "navigation", "path", "waypoint"]):
        return NAVIGATION_PROMPT_SUFFIX
    elif any(keyword in task_lower for keyword in ["race", "racing", "speed", "fast"]):
        return RACING_PROMPT_SUFFIX
    elif any(keyword in task_lower for keyword in ["hover", "hovering", "stable", "stationary"]):
        return HOVERING_PROMPT_SUFFIX
    elif any(keyword in task_lower for keyword in ["track", "tracking", "follow", "chase"]):
        return TRACKING_PROMPT_SUFFIX
    else:
        return ""


def create_multi_objective_prompt(
    primary_objective: str,
    secondary_objectives: list,
    constraints: list = None
) -> str:
    """Create multi-objective reward prompt."""
    parts = [f"Primary: {primary_objective}"]
    
    if secondary_objectives:
        parts.append("Secondary: " + ", ".join(secondary_objectives))
    
    if constraints:
        parts.append("Constraints: " + ", ".join(constraints))
    
    parts.append("\nBalance objectives with weights. Use (self.velocity - 0) for BPTT.\nGenerate get_reward(self):")
    
    return "\n".join(parts)