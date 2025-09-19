"""
Prompt Engineering for VisFly Reward Function Generation

This module contains system and user prompts optimized for generating
reward functions for VisFly drone environments.
"""

from typing import Dict, Any, Optional


def create_system_prompt() -> str:
    """Create system prompt for LLM reward function generation."""
    return (
        "You are a reward engineer creating reinforcement-learning reward functions. "
        "Write a complete `def get_reward(self) -> torch.Tensor` implementation that helps the policy master "
        "each VisFly task.\n\n"
        "Authoritative guidance:\n"
        "- Operate entirely in PyTorch; never fall back to NumPy or Python math.\n"
        "- Return a 1-D tensor of length `self.num_agent` on `self.device`.\n"
        "- SHAC/BPTT REQUIRE you to clone dynamics tensors: always write `(self.velocity - 0)` and `(self.angular_velocity - 0)` (and apply the same `- 0` trick to every tensor whose name contains `vel` or `ang_vel`) before using them in calculations.\n"
        "- Avoid in-place edits, boolean indexing assignment, or constructing new tensors with mismatched devices.\n"
        "- Use `torch.where`, `torch.clamp`, vector norms, and smooth penalties to combine reward terms.\n"
        "- Make collision handling robust by reading `self.collision_dis` / `self.collision_vector` rather than raw sensor pixels.\n\n"
        "Common pitfalls to avoid:\n"
        "- Do not call Torch APIs with invalid signatures (e.g., `torch.min(tensor, dim=int)`).\n"
        "- Do not instantiate tensors inside `torch.where` (use scalar literals).\n"
        "- Do not mix `_step_count` scalars directly with tensors without broadcasting helpers.\n"
        "- Do not rely on TorchScript; focus on readable, differentiable PyTorch.\n\n"
        "Deliver confident, well-structured code that balances progress, stability, and safety signals."
    )


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
    source_file = inspect.getfile(env_class)
    return load_environment_code(source_file)


def create_user_prompt(
    task_description: str,
    context_info: Dict[str, Any],
    feedback: str = "",
    env_code: str = "",
    api_doc: Optional[str] = None,
) -> str:
    """Create the user-facing reward prompt following the Eureka structure.

    Template layout: environment stub -> optional API reference -> task briefing ->
    structured context -> prior feedback -> final instructions.
    """
    prompt_parts = []

    if env_code:
        prompt_parts.append("The Python environment is:")
        prompt_parts.append("```python")
        prompt_parts.append(env_code.strip())
        prompt_parts.append("```")
        prompt_parts.append("")

    if api_doc:
        prompt_parts.append("Environment API reference (read once, no need to repeat in output):")
        prompt_parts.append("```text")
        prompt_parts.append(api_doc.strip())
        prompt_parts.append("```")
        prompt_parts.append("")

    prompt_parts.append(f"Task: {task_description}")

    if context_info:
        prompt_parts.append("\nKey environment details:")
        for key in sorted(context_info.keys()):
            value = context_info[key]
            prompt_parts.append(f"- {key}: {value}")

    if feedback:
        prompt_parts.append("\nFeedback from previous attempts (address every point):")
        prompt_parts.append(feedback.strip())

    prompt_parts.append(
        "\nReturn only the complete `def get_reward(self) -> torch.Tensor` implementation."
    )

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
