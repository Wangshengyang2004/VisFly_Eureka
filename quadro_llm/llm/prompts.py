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
- self.position: torch.Tensor [N, 3] drone positions
- self.velocity: torch.Tensor [N, 3] linear velocities  
- self.orientation: torch.Tensor [N, 4] quaternions
- self.angular_velocity: torch.Tensor [N, 3] angular velocities
- self.target: torch.Tensor [N, 3] target positions
- self.collision_dis: torch.Tensor [N] obstacle distances
- self.sensor_obs['depth']: numpy.ndarray [N, H, W] depth camera (from habitat)
- self._step_count: int - current step
- self.num_agent: int - agent count
- self.device: torch device for tensors

Requirements:
1. Return complete get_reward(self) method  
2. Use torch operations only (imported as 'torch')
3. Function signature: def get_reward(self) -> torch.Tensor:
4. Return shape [N] tensor
5. IMPORTANT: self.sensor_obs['depth'] is numpy array - convert ONCE at the start:
   depth_tensor = torch.from_numpy(self.sensor_obs['depth']).to(self.device)
6. Do NOT convert other self.* attributes - they are already torch tensors
7. Be confident - all listed variables exist, use them directly

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
- Step penalty: -0.01 * torch.ones(self.num_agent, device=self.device)
- Depth sensor usage:
  depth_tensor = torch.from_numpy(self.sensor_obs['depth']).to(self.device)
  min_depth, _ = torch.min(depth_tensor.view(depth_tensor.size(0), -1), dim=1)
  obstacle_penalty = -1.0 / (min_depth + 0.1)

Numerical Safety:
- Use torch.clamp(x, min=1e-8) only for division denominators
- Write clean, confident code without excessive checks
- Trust the environment structure"""


def load_environment_code(env_path: str) -> str:
    """
    Load environment code dynamically from the actual file.

    Args:
        env_path: Path to the environment file

    Returns:
        Environment source code with get_reward method stripped
    """
    import re
    from pathlib import Path
    
    # Read the environment file
    env_file = Path(env_path)
    if not env_file.exists():
        return f"# Could not load environment from {env_path}"
    
    with open(env_file, 'r') as f:
        code = f.read()
    
    # Remove the get_reward method implementation but keep the signature
    # Pattern to match the entire get_reward method
    pattern = r'(def get_reward\(self[^)]*\)[^:]*:)(.*?)(?=\n    def |\n\nclass |\Z)'
    
    def replace_reward(match):
        signature = match.group(1)
        return f"{signature}\n        # TODO: Implement reward function\n        return torch.zeros(self.num_envs, device=self.device)"
    
    # Replace the get_reward implementation
    code = re.sub(pattern, replace_reward, code, flags=re.DOTALL)
    
    return code


def extract_env_code_without_reward(env_class) -> str:
    """
    Extract environment code with reward function removed.

    Args:
        env_class: Environment class to extract code from

    Returns:
        Environment source code with get_reward method stripped
    """
    import inspect
    from pathlib import Path
    
    # Get the source file of the environment class
    try:
        source_file = inspect.getfile(env_class)
        return load_environment_code(source_file)
    except Exception as e:
        # Fallback if we can't get the source
        return f"# Environment: {env_class.__name__}\n# Could not extract source: {e}"


def create_user_prompt(
    task_description: str,
    context_info: Dict[str, Any],
    feedback: str = "",
    env_code: str = "",
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
    prompt_parts.append(
        f"Write a reward function for the following task: {task_description}"
    )

    # Previous iteration feedback (if any)
    if feedback:
        prompt_parts.append(f"\nFeedback from previous attempts:")
        prompt_parts.append(feedback)

    # Request (simplified like real Eureka)
    prompt_parts.append("""
Please provide only the complete get_reward(self) method implementation.""")

    return "\n".join(prompt_parts)


def create_improvement_prompt(
    original_code: str, performance_issues: str, task_description: str
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
    previous_rewards: list = None,
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
        prompt_parts.append(
            f"\nTried {len(previous_rewards)} functions. Generate a different approach."
        )

    prompt_parts.append(
        "\nUse (self.velocity - 0) and (self.angular_velocity - 0) for BPTT.\nGenerate get_reward(self):"
    )

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

    if any(
        keyword in task_lower
        for keyword in ["navigate", "navigation", "path", "waypoint"]
    ):
        return NAVIGATION_PROMPT_SUFFIX
    elif any(keyword in task_lower for keyword in ["race", "racing", "speed", "fast"]):
        return RACING_PROMPT_SUFFIX
    elif any(
        keyword in task_lower
        for keyword in ["hover", "hovering", "stable", "stationary"]
    ):
        return HOVERING_PROMPT_SUFFIX
    elif any(
        keyword in task_lower for keyword in ["track", "tracking", "follow", "chase"]
    ):
        return TRACKING_PROMPT_SUFFIX
    else:
        return ""


def create_multi_objective_prompt(
    primary_objective: str, secondary_objectives: list, constraints: list = None
) -> str:
    """Create multi-objective reward prompt."""
    parts = [f"Primary: {primary_objective}"]

    if secondary_objectives:
        parts.append("Secondary: " + ", ".join(secondary_objectives))

    if constraints:
        parts.append("Constraints: " + ", ".join(constraints))

    parts.append(
        "\nBalance objectives with weights. Use (self.velocity - 0) for BPTT.\nGenerate get_reward(self):"
    )

    return "\n".join(parts)
