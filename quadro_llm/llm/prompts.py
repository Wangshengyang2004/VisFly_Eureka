"""
Prompt Engineering for VisFly Reward Function Generation

This module contains system and user prompts optimized for generating
reward functions for VisFly drone environments.
"""

import logging
from typing import Dict, Any, Optional

_logger = logging.getLogger(__name__)


def create_system_prompt(algorithm: str = "bptt") -> str:
    """Create system prompt for LLM reward function generation."""
    dense_reward_note = ""
    if algorithm.lower() in ("bptt", "shac"):
        dense_reward_note = (
            "Reward shape (BPTT/SHAC): Use dense, continuous rewards every step. "
            "Avoid sparse rewards in any form "
            "gradient-based methods need step-wise differentiable signal.\n\n"
        )
    return (
        "You are a reward engineer creating reinforcement-learning reward functions. "
        "Write a complete `def get_reward(self, predicted_obs=None) -> torch.Tensor` implementation that helps the policy master "
        "each VisFly task. You MUST strictly follow ALL rules in the API reference, especially gradient safety patterns like `(tensor - 0)` for dynamics-related tensors.\n\n"
        + dense_reward_note
        + "Guidance:\n"
        "- Scale rewards to reasonable values for stable training.\n\n"
        "CRITICAL CONSTRAINTS (violations cause training failures):\n"
        "1. NEVER use self.is_collision, self.success, self.failure in reward computation - these break gradient flow.\n"
        "2. ALWAYS use (tensor - 0) pattern for velocity/acceleration tensors to preserve gradients.\n\n"
        "Pitfalls to avoid:\n"
        "- Do not call Torch APIs with invalid signatures (e.g., `torch.min(tensor, dim=int)`).\n"
        "- Do not instantiate tensors inside `torch.where` (use scalar literals).\n"
        "- Do not mix `_step_count` scalars directly with tensors without broadcasting helpers.\n"
        "- Only call .clamp(), .clamp_min(), .clamp_max() on torch tensors; Python floats have no .clamp attribute. Keep intermediate values as tensors (do not use .item() before clamping).\n"
        "- Do not rely on TorchScript; focus on readable, differentiable PyTorch.\n\n"
        "Deliver confident, well-structured code with minimal comments. Avoid redundant comments that simply restate the code."
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


def extract_human_reward(env_class) -> Optional[str]:
    """
    Extract the human-designed reward function from environment class.

    Args:
        env_class: Environment class to extract reward function from

    Returns:
        Human reward function code as string, or None if extraction fails
    """
    import re
    import inspect
    from pathlib import Path
    
    try:
        # Get the source file of the environment class
        source_file = inspect.getfile(env_class)
        env_file = Path(source_file)
        
        if not env_file.exists():
            return None
        
        with open(env_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Pattern to match the entire get_reward method
        pattern = r'(def get_reward\(self[^)]*\)[^:]*:)(.*?)(?=\n    def |\n\nclass |\Z)'
        
        match = re.search(pattern, code, flags=re.DOTALL)
        if match:
            # Return the full method (signature + body)
            return match.group(0).strip()
        
        return None
    except Exception as e:
        _logger.warning("Failed to extract human reward from %s: %s", env_class, e)
        return None


def create_user_prompt(
    task_description: str,
    context_info: Dict[str, Any],
    feedback: str = "",
    env_code: str = "",
    api_doc: Optional[str] = None,
    human_reward_code: Optional[str] = None,
    elite_reward_code: Optional[str] = None,
    include_static_info: bool = True,
) -> str:
    """Create the user-facing reward prompt following the Eureka structure.

    Template layout: environment stub -> optional API reference -> task briefing ->
    structured context -> optional human reward -> prior feedback -> final instructions.
    
    Args:
        include_static_info: If False, skip env_code, api_doc, task_description, and context_info
            (useful when these are already in conversation history)
    """
    prompt_parts = []

    if include_static_info:
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

    if human_reward_code:
        prompt_parts.append("\nReference reward function:")
        prompt_parts.append("```python")
        prompt_parts.append(human_reward_code.strip())
        prompt_parts.append("```")
        prompt_parts.append("(Optional reference; your implementation may differ in structure and weights.)")

    if elite_reward_code:
        prompt_parts.append("\nPrevious elite reward function (selected from last iteration):")
        prompt_parts.append("```python")
        prompt_parts.append(elite_reward_code.strip())
        prompt_parts.append("```")

        if feedback:
            prompt_parts.append("\n## Required Modifications (from analysis):")
            prompt_parts.append(feedback.strip())
            prompt_parts.append("\n**IMPORTANT**: You MUST apply the specific fixes listed above.")
            prompt_parts.append(
                "Use the reference as inspiration, but feel free to explore different structures."
            )
        else:
            prompt_parts.append("\nImprove this function based on general reward design principles.")

    elif feedback:
        prompt_parts.append(
            "\n## Required Modifications to the previous elite reward function (your last response):"
        )
        prompt_parts.append(feedback.strip())
        prompt_parts.append("\n**IMPORTANT**: You MUST apply the specific fixes listed above.")
        prompt_parts.append(
            "Use the previous elite as baseline, but feel free to explore different structures."
        )

    prompt_parts.append(
        "\nReturn only the complete `def get_reward(self, predicted_obs=None) -> torch.Tensor` implementation. "
        "The function MUST return a 1-D torch.Tensor of shape (num_agent,). Do NOT return a dict. "
        "Keep the code concise with minimal comments."
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

Fix these problems. Follow the API reference for gradient and safety rules.

Keep the code concise with minimal comments.

Generate improved get_reward(self, predicted_obs=None) method:"""


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
        "\nFollow the API reference for gradient and safety rules. Keep code concise.\n"
        "Generate get_reward(self, predicted_obs=None):"
    )

    return "\n".join(prompt_parts)


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
        "\nBalance objectives with weights. Follow the API reference for gradient and safety rules. Keep code concise.\n"
        "Generate get_reward(self, predicted_obs=None):"
    )

    return "\n".join(parts)


def create_coefficient_tuning_system_prompt() -> str:
    """Create system prompt for coefficient tuning mode."""
    return (
        "You are a reward coefficient tuner. Your task is to optimize the coefficients "
        "of a fixed reward function structure for a drone navigation task.\n\n"
        "IMPORTANT: You must ONLY output coefficient values. Do NOT modify the reward function structure.\n"
        "The reward function structure is fixed and cannot be changed.\n\n"
        "Output format: Return a JSON object with coefficient names and values, or a simple key-value list."
    )


def create_coefficient_tuning_prompt(
    task_description: str,
    feedback: str = "",
    current_coefficients: Optional[Dict[str, float]] = None,
) -> str:
    """
    Create prompt for coefficient tuning mode.
    
    Args:
        task_description: Task description
        feedback: Feedback from previous iterations
        current_coefficients: Current coefficient values (for reference)
        
    Returns:
        Prompt string for LLM
    """
    from ..utils.coefficient_tuner import get_coefficient_names, get_default_coefficients
    
    prompt_parts = []
    
    prompt_parts.append(f"Task: {task_description}")
    prompt_parts.append("")
    prompt_parts.append("You are tuning coefficients for a FIXED reward function structure.")
    prompt_parts.append("The reward function has the following components with tunable coefficients:")
    prompt_parts.append("")
    
    # List all coefficients
    default_coeffs = get_default_coefficients()
    for name, default_value in default_coeffs.items():
        prompt_parts.append(f"- {name}: default = {default_value}")
    
    prompt_parts.append("")
    prompt_parts.append("The reward function structure is:")
    prompt_parts.append("- base_r: base reward (constant)")
    prompt_parts.append("- vel_r_coef: velocity matching target coefficient")
    prompt_parts.append("- ang_r_coef: angular velocity penalty coefficient")
    prompt_parts.append("- acc_r_coef: acceleration penalty coefficient")
    prompt_parts.append("- acc_change_r_coef: acceleration change penalty coefficient")
    prompt_parts.append("- act_change_r_coef: action change penalty coefficient")
    prompt_parts.append("- align_r_coef: heading alignment reward coefficient")
    prompt_parts.append("- col_vel_r_coef: collision velocity penalty coefficient")
    prompt_parts.append("- col_dis_r_coef: collision distance penalty coefficient")
    prompt_parts.append("- share_factor_collision: collision term sharing factor (0-1)")
    prompt_parts.append("")
    
    if current_coefficients:
        prompt_parts.append("Current coefficient values:")
        for name, value in current_coefficients.items():
            prompt_parts.append(f"  {name} = {value}")
        prompt_parts.append("")
    
    if feedback:
        prompt_parts.append("Feedback from previous attempts:")
        prompt_parts.append(feedback.strip())
        prompt_parts.append("")
    
    prompt_parts.append(
        "Output ONLY the coefficient values in JSON format, for example:\n"
        '{"base_r": 0.1, "vel_r_coef": -0.04, "ang_r_coef": -0.02, ...}\n\n'
        "Or as a simple list: base_r=0.1, vel_r_coef=-0.04, ang_r_coef=-0.02, ...\n\n"
        "Do NOT output any code or explanation, only the coefficient values."
    )
    
    return "\n".join(prompt_parts)
