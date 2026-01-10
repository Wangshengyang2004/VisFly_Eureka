"""
Coefficient tuning utilities for TempTuner mode.

This module provides functions to:
1. Extract reward template from NavigationEnv
2. Generate reward function code from coefficients
3. Parse coefficient values from LLM output
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any
import torch as th

logger = logging.getLogger(__name__)


# Fixed reward template based on envs/NavigationEnv.py
NAVIGATION_REWARD_TEMPLATE = """
def get_reward(self, predicted_obs=None) -> th.Tensor:
    # precise and stable target flight
    base_r = {base_r}

    # Velocity matching target
    vel_r = (self.velocity - self.target).norm(dim=1)
    vel_r = smooth_l1_loss_per_row(vel_r, th.zeros_like(vel_r)) * {vel_r_coef}
    
    # Angular velocity penalty
    ang_r = (self.angular_velocity - 0).norm(dim=1) * {ang_r_coef}

    # Acceleration penalty
    acc_r = (self.envs.acceleration - 0).norm(dim=1).pow(1)
    acc_r = smooth_l1_loss_per_row(acc_r, th.zeros_like(acc_r)) * {acc_r_coef}
    
    if not hasattr(self, "_pre_acc"):
        self._pre_acc = self.envs.acceleration.clone()
    acc_change_r = (self.envs.acceleration - self._pre_acc).norm(dim=1).pow(2) * {acc_change_r_coef}
    
    # Action change penalty
    act_change_r = (self.envs.dynamics._pre_action[-2].to(self.device).T -
                    self._action.to(self.device)
                    ).norm(dim=-1) * {act_change_r_coef}

    # Heading alignment reward
    unit_velocity = self.velocity / (self.velocity.norm(dim=1, keepdim=True)+1e-6)
    align = (unit_velocity * self.direction).sum(dim=1)
    align_r = align * self.velocity.norm(dim=1) * {align_r_coef}

    # Collision handling
    share_factor_collision = {share_factor_collision}
    collision_dis = self.collision_vector.norm(dim=1).clamp_min(0.)
    collision_dir = self.collision_vector / (collision_dis.unsqueeze(1)+1e-6)
    
    thre_vel = 1.5
    weight = ((thre_vel-collision_dis.detach()).clamp(min=0, )/thre_vel).pow(1)
    col_approach_velocity = (self.velocity * collision_dir.detach()).sum(dim=1).clamp_min(0.)
    col_vel_r = col_approach_velocity * weight * {col_vel_r_coef} * share_factor_collision * 0.5

    thre_vel = 0.5
    weight = ((thre_vel - collision_dis.detach()).clamp(min=0, ) / thre_vel).pow(1)
    
    # Collision distance penalty
    k = 0.015
    func = lambda x: 12 * k / (x+k)
    col_dis_r = func(collision_dis) * {col_dis_r_coef} * share_factor_collision

    reward = {{
        "reward": base_r + vel_r + ang_r + align_r
                + act_change_r + acc_r
                + acc_change_r
                + col_vel_r + col_dis_r
        ,
        "vel_r": dl(vel_r),
        "ang_r": dl(ang_r),
        "acc_r": dl(acc_r),
        "align_r": dl(align_r),
        "col_vel_r": dl(col_vel_r),
        "col_dis_r": dl(col_dis_r),
        "acc_change_r": dl(acc_change_r),
        "act_change_r": dl(act_change_r),
    }}
    return reward
"""

# Helper function needed in the template
SMOOTH_L1_HELPER = """
def smooth_l1_loss_per_row(pred, target, beta: float = 1.0, reduction: str = "mean"):
    diff = pred - target
    abs_diff = diff.abs()
    if beta <= 0:
        loss = abs_diff
    else:
        mask = abs_diff < beta
        loss = th.where(mask, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
    return loss
"""

DL_HELPER = """
dl = lambda x: x.clone().detach()
"""


# Default coefficients from NavigationEnv.py
DEFAULT_COEFFICIENTS = {
    "base_r": 0.1,
    "vel_r_coef": -0.04,
    "ang_r_coef": -0.02,
    "acc_r_coef": -0.003,
    "acc_change_r_coef": -0.005,
    "act_change_r_coef": -0.004,
    "align_r_coef": 0.005,
    "col_vel_r_coef": -1.0,  # Original: -1 * share_factor_collision * 0.5
    "col_dis_r_coef": -2.0,  # -2 * share_factor_collision
    "share_factor_collision": 0.50,
}


def generate_reward_from_coefficients(coefficients: Dict[str, float]) -> str:
    """
    Generate complete reward function code from coefficient values.
    
    Args:
        coefficients: Dictionary of coefficient values
        
    Returns:
        Complete reward function code string
    """
    # Merge with defaults
    coeffs = {**DEFAULT_COEFFICIENTS, **coefficients}
    
    # Format the template
    reward_code = NAVIGATION_REWARD_TEMPLATE.format(**coeffs)
    
    # Prepend helper functions
    full_code = SMOOTH_L1_HELPER + "\n" + DL_HELPER + "\n" + reward_code
    
    return full_code


def parse_coefficients_from_llm_output(text: str) -> Optional[Dict[str, float]]:
    """
    Parse coefficient values from LLM output.
    
    Supports multiple formats:
    1. JSON: {"base_r": 0.1, "vel_r_coef": -0.04, ...}
    2. Simple key-value pairs: base_r=0.1, vel_r_coef=-0.04, ...
    3. List format: [0.1, -0.04, -0.02, ...] (in order of DEFAULT_COEFFICIENTS keys)
    
    Args:
        text: LLM output text
        
    Returns:
        Dictionary of coefficients or None if parsing failed
    """
    text = text.strip()
    
    # Try JSON first
    try:
        # Extract JSON from code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{[^{}]*"[^"]*"\s*:\s*[^,}]+\s*(?:,\s*"[^"]*"\s*:\s*[^,}]+\s*)*\}', text)
            if json_match:
                text = json_match.group(0)
        
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            # Validate and convert to float
            result = {}
            for key, value in parsed.items():
                if key in DEFAULT_COEFFICIENTS:
                    try:
                        result[key] = float(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid coefficient value for {key}: {value}")
            return result if result else None
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Try key-value pairs: base_r=0.1, vel_r_coef=-0.04, ...
    try:
        pattern = r'(\w+)\s*[:=]\s*([+-]?\d+\.?\d*)'
        matches = re.findall(pattern, text)
        if matches:
            result = {}
            for key, value_str in matches:
                if key in DEFAULT_COEFFICIENTS:
                    try:
                        result[key] = float(value_str)
                    except ValueError:
                        continue
            return result if result else None
    except Exception:
        pass
    
    # Try list format: [0.1, -0.04, ...]
    try:
        list_match = re.search(r'\[([^\]]+)\]', text)
        if list_match:
            values_str = list_match.group(1)
            values = [float(x.strip()) for x in values_str.split(',')]
            keys = list(DEFAULT_COEFFICIENTS.keys())
            if len(values) == len(keys):
                return dict(zip(keys, values))
    except (ValueError, AttributeError):
        pass
    
    logger.warning(f"Failed to parse coefficients from LLM output: {text[:200]}")
    return None


def get_coefficient_names() -> List[str]:
    """Get list of all tunable coefficient names."""
    return list(DEFAULT_COEFFICIENTS.keys())


def get_default_coefficients() -> Dict[str, float]:
    """Get default coefficient values."""
    return DEFAULT_COEFFICIENTS.copy()

