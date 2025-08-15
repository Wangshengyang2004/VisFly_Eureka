"""
Prompt Engineering for VisFly Reward Function Generation

This module contains system and user prompts optimized for generating
reward functions for VisFly drone environments.
"""

from typing import Dict, Any
import re
from pathlib import Path


def create_system_prompt() -> str:
    """
    Create the system prompt for LLM reward function generation.
    
    This prompt establishes the context, constraints, and requirements
    for generating VisFly-compatible reward functions.
    """
    return """You are an expert at designing reward functions for VisFly drone environments. Your task is to generate complete get_reward methods that enable effective reinforcement learning for vision-based drone tasks.

VisFly Environment Context:
- self.position: torch.Tensor [N, 3] - drone positions in 3D space
- self.velocity: torch.Tensor [N, 3] - linear velocities 
- self.orientation: torch.Tensor [N, 4] - quaternion orientations
- self.angular_velocity: torch.Tensor [N, 3] - angular velocities
- self.target: torch.Tensor [N, 3] - target positions (when applicable)
- self.sensor_obs: dict containing sensor observations
  - 'depth': torch.Tensor [N, H, W] - depth camera readings
  - 'rgb': torch.Tensor [N, H, W, 3] - RGB camera readings
- self.collision_dis: torch.Tensor [N] - distance to closest obstacle
- self._step_count: int - current episode step
- self.num_agent: int - number of agents in environment
- self.max_episode_steps: int - maximum steps per episode

Critical Requirements:
1. Return complete get_reward(self) method definition
2. Use torch operations exclusively for differentiability
3. Return torch.Tensor with shape [N] for N agents
4. Utilize visual sensors (depth/rgb) when available and relevant
5. No imports needed - torch/th/F are available in execution context
6. Handle edge cases gracefully (missing attributes, zero divisions)
7. Ensure differentiable operations for BPTT compatibility
8. Use torch.clamp() for numerical stability in divisions/norms

Common Patterns:
- Distance rewards: -torch.norm(self.position - self.target, dim=1)
- Velocity rewards: torch.norm(self.velocity, dim=1) or -torch.norm(self.velocity, dim=1)
- Collision avoidance: -torch.sum(depth < threshold, dim=(1,2)) * penalty
- Stability rewards: -torch.norm(orientation - target_orientation, dim=1)
- Time penalties: -torch.ones(self.num_agent) * step_penalty

Safety Guidelines:
- Always check attribute existence with hasattr()
- Use torch.clamp() to prevent division by zero
- Ensure reward tensors have correct shape
- Avoid in-place operations that break gradients
- Use .clone() or + 0.0 to create new tensors when needed

Generate reward functions that are:
1. Mathematically sound and numerically stable
2. Appropriate for the specific task description
3. Balanced between different reward components
4. Optimized for the drone dynamics and sensor capabilities"""


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
        head_velocity = orientation.world_to_head((self.velocity - 0).T).T

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
        
        # Optionally randomize target (can be overridden in subclasses)
        if hasattr(self, 'randomize_target') and self.randomize_target:
            # Random target within reasonable bounds
            target_x = th.uniform(-10, 20, (self.num_envs, 1), device=self.device)
            target_y = th.uniform(-5, 5, (self.num_envs, 1), device=self.device) 
            target_z = th.uniform(1, 3, (self.num_envs, 1), device=self.device)
            self.target = th.cat([target_x, target_y, target_z], dim=1)
            
        return self.get_observation()
"""


def extract_env_code_without_reward(env_class) -> str:
    """
    Extract environment code with reward function removed (optimized to avoid blocking).
    
    Args:
        env_class: Environment class to extract code from
        
    Returns:
        Environment source code with get_reward method stripped
    """
    # Use pre-extracted code to avoid blocking environment creation
    if hasattr(env_class, '__name__') and env_class.__name__ == 'NavigationEnv':
        return get_navigation_env_code()
    else:
        # Fallback for other environments
        return f"# Environment: {env_class.__name__ if hasattr(env_class, '__name__') else 'Unknown'}\n# Pre-extracted code not available for this environment type."


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
    """
    Create a prompt for improving an existing reward function.
    
    Args:
        original_code: Original reward function code
        performance_issues: Description of performance problems
        task_description: Original task description
        
    Returns:
        Improvement prompt string
    """
    return f"""Task: {task_description}

The following reward function has performance issues:

```python
{original_code}
```

Performance Issues Identified:
{performance_issues}

Please provide an improved version that addresses these specific issues while:
1. Maintaining the same function signature and structure
2. Fixing the identified problems
3. Preserving any working aspects of the original reward
4. Using more sophisticated reward shaping if needed
5. Ensuring numerical stability and gradient compatibility

Generate the complete improved get_reward(self) method:"""


def create_context_aware_prompt(
    task_description: str,
    environment_observations: Dict[str, Any],
    sensor_data_shapes: Dict[str, tuple],
    previous_rewards: list = None
) -> str:
    """
    Create a context-aware prompt using actual environment observations.
    
    Args:
        task_description: Task description
        environment_observations: Sample observations from environment
        sensor_data_shapes: Shapes of sensor data tensors
        previous_rewards: List of previous reward function attempts
        
    Returns:
        Context-aware prompt
    """
    prompt_parts = [f"Task: {task_description}"]
    
    # Environment observation details
    if environment_observations:
        prompt_parts.append("\nCurrent Environment State:")
        
        for key, value in environment_observations.items():
            if hasattr(value, 'shape'):
                prompt_parts.append(f"- {key}: shape {value.shape}")
            else:
                prompt_parts.append(f"- {key}: {type(value)}")
    
    # Sensor data information
    if sensor_data_shapes:
        prompt_parts.append("\nSensor Data Shapes:")
        for sensor_name, shape in sensor_data_shapes.items():
            prompt_parts.append(f"- {sensor_name}: {shape}")
    
    # Previous attempts (if any)
    if previous_rewards:
        prompt_parts.append(f"\nPrevious Attempts: {len(previous_rewards)} reward functions tested")
        prompt_parts.append("Focus on novel approaches that haven't been tried yet.")
    
    prompt_parts.append("""
Generate a reward function that takes advantage of the specific environment state and sensor data available.
Provide complete get_reward(self) method:""")
    
    return "\n".join(prompt_parts)


# Specialized prompts for different environment types
NAVIGATION_PROMPT_SUFFIX = """
For navigation tasks, consider:
- Distance to target as primary reward component
- Obstacle avoidance using depth sensor
- Smooth trajectories (velocity/acceleration penalties)
- Orientation alignment with movement direction
- Energy efficiency (control effort penalties)"""

RACING_PROMPT_SUFFIX = """
For racing tasks, consider:
- Forward velocity rewards
- Gate/checkpoint passing bonuses
- Racing line optimization
- Speed vs. control trade-offs
- Lap time minimization"""

HOVERING_PROMPT_SUFFIX = """
For hovering tasks, consider:
- Position stability around target
- Velocity minimization
- Orientation stability
- Altitude maintenance
- Disturbance rejection"""

TRACKING_PROMPT_SUFFIX = """
For tracking tasks, consider:
- Target relative position/velocity
- Prediction and anticipation
- Smooth tracking vs. responsiveness
- Visual target detection and following
- Maintaining optimal tracking distance"""


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
    """
    Create a prompt for multi-objective reward functions.
    
    Args:
        primary_objective: Main objective description
        secondary_objectives: List of secondary objectives
        constraints: List of constraints to satisfy
        
    Returns:
        Multi-objective prompt string
    """
    prompt_parts = [f"Primary Objective: {primary_objective}"]
    
    if secondary_objectives:
        prompt_parts.append("\nSecondary Objectives:")
        for i, obj in enumerate(secondary_objectives, 1):
            prompt_parts.append(f"{i}. {obj}")
    
    if constraints:
        prompt_parts.append("\nConstraints:")
        for i, constraint in enumerate(constraints, 1):
            prompt_parts.append(f"{i}. {constraint}")
    
    prompt_parts.append("""
Create a reward function that balances these multiple objectives appropriately.
Use weighted combinations and consider the relative importance of each component.
Provide the complete get_reward(self) method with clear component explanations:""")
    
    return "\n".join(prompt_parts)