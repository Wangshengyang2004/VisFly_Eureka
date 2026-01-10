"""
Direct Reward Function Injection for VisFly Environments

This module provides functionality to directly inject LLM-generated reward functions
into existing VisFly environment instances without wrapper layers.
"""

import ast
import logging
import textwrap
import time
import torch
import types
from typing import Any, Dict, Optional, Callable

from ..constants import MAX_REASONABLE_REWARD_TENSOR_SIZE


def inject_generated_reward(env_instance: Any, reward_code: str) -> bool:
    """
    Directly inject a reward function into a VisFly environment instance.

    This function compiles and injects the reward code directly into the environment's
    get_reward method, enabling seamless integration with existing VisFly training loops.

    Args:
        env_instance: VisFly environment instance (e.g., NavigationEnv)
        reward_code: String containing the complete get_reward function definition

    Returns:
        bool: True if injection succeeded, False otherwise
    """
    logger = logging.getLogger(__name__)

    # Store original reward function for fallback
    original_get_reward = getattr(env_instance, "get_reward", None)
    env_instance._original_get_reward = original_get_reward

    try:
        # Create execution environment with necessary imports
        exec_globals = {
            "torch": torch,
            "th": torch,  # Common abbreviation in VisFly
            "F": torch.nn.functional,
            "__builtins__": __builtins__,
        }

        # Execute the reward function code
        exec(reward_code, exec_globals)

        # Extract the new reward function
        if "get_reward" not in exec_globals:
            logger.error("No get_reward function found in generated code")
            return False

        new_reward_func = exec_globals["get_reward"]

        # Inject the reward function into the environment
        env_instance.get_reward = types.MethodType(new_reward_func, env_instance)

        # Test the injected reward function
        if not _test_injected_reward(env_instance, original_get_reward):
            logger.error("Injected reward function failed validation")
            return False

        logger.info("Successfully injected reward function")
        return True

    except Exception as e:
        logger.error(f"Failed to inject reward function: {e}")

        # Restore original reward function if available
        if original_get_reward:
            env_instance.get_reward = original_get_reward

        return False


def _parse_reward_function(reward_code: str) -> Optional[Callable]:
    """
    Parse reward function code and return callable function.

    Args:
        reward_code: String containing the reward function definition

    Returns:
        Parsed function if valid, None otherwise
    """
    try:
        # Parse the AST to validate syntax
        tree = ast.parse(reward_code)

        # Find the get_reward function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_reward":
                # Check if function has self parameter
                if not node.args.args or node.args.args[0].arg != "self":
                    return None

                # Compile and extract the function
                code_obj = compile(tree, "<reward_function>", "exec")
                namespace = {
                    "torch": torch,
                    "th": torch,
                    "F": torch.nn.functional,
                    "__builtins__": __builtins__,
                }
                exec(code_obj, namespace)

                if "get_reward" in namespace:
                    return namespace["get_reward"]

        return None

    except (SyntaxError, ValueError) as e:
        return None


def _validate_reward_function(reward_func: Callable, env_instance: Any) -> bool:
    """
    Validate a reward function by testing it on the environment.

    Args:
        reward_func: Function to validate
        env_instance: Environment instance to test on

    Returns:
        True if function is valid, False otherwise
    """
    try:
        # Test the function
        result = reward_func(env_instance)

        # Check if result is a tensor
        if not isinstance(result, torch.Tensor):
            return False

        # Check if result has reasonable shape and values
        if result.numel() == 0:
            return False

        # Check for NaN or infinite values
        if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
            return False

        return True

    except Exception as e:
        return False


def _create_method_wrapper(reward_func: Callable) -> Callable:
    """
    Create a method wrapper for the reward function.

    Args:
        reward_func: Function to wrap as method

    Returns:
        Wrapped method function
    """

    def wrapper(self):
        return reward_func(self)

    return wrapper


def _test_injected_reward(
    env_instance: Any, original_get_reward: Optional[Any]
) -> bool:
    """
    Test the injected reward function to ensure it works correctly.

    Args:
        env_instance: Environment with injected reward function
        original_get_reward: Original reward function for fallback

    Returns:
        bool: True if reward function works correctly
    """
    logger = logging.getLogger(__name__)

    def _restore_original():
        if original_get_reward is None:
            return
        try:
            env_instance.get_reward = original_get_reward
            logger.info("Restored original reward function after validation failure")
        except Exception as restore_error:
            logger.error(
                f"Failed to restore original reward function: {restore_error}"
            )

    try:
        env_instance.reset()
        reward = env_instance.get_reward()
    except Exception as exc:
        logger.error(f"Injected reward evaluation failed: {exc}")
        _restore_original()
        return False

    if not isinstance(reward, torch.Tensor):
        logger.error(
            f"Reward function returned {type(reward)}, expected torch.Tensor"
        )
        _restore_original()
        return False

    if reward.dim() > 2 or reward.numel() == 0:
        logger.error(f"Invalid reward shape: {reward.shape}")
        _restore_original()
        return False

    if torch.isnan(reward).any() or torch.isinf(reward).any():
        logger.error("Reward function returned NaN or infinite values")
        _restore_original()
        return False

    if reward.numel() > MAX_REASONABLE_REWARD_TENSOR_SIZE:
        logger.warning(f"Large reward tensor: {reward.shape}")

    if torch.abs(reward).max() > 1e6:
        logger.warning(
            f"Very large reward values detected: max={torch.abs(reward).max()}"
        )

    logger.debug(
        f"Reward function test passed: shape={reward.shape}, range=[{reward.min():.3f}, {reward.max():.3f}]"
    )
    return True


def extract_reward_function(llm_response: str) -> Optional[str]:
    """
    Extract the get_reward function from an LLM response.

    Args:
        llm_response: Raw response from LLM containing reward function

    Returns:
        str: Extracted reward function code or None if not found
    """
    logger = logging.getLogger(__name__)

    try:
        # Remove thinking/reasoning tags if present
        # Minimax uses <think>...</think> tags
        # Other models may use <think>...</think> or similar
        # This ensures we don't accidentally include thinking content in extracted code
        import re
        cleaned_response = llm_response
        # Remove <think> tags (Minimax)
        cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
        # Remove <think> tags (various formats)
        cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
        cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
        
        lines = cleaned_response.strip().split("\n")
        func_start = None

        # Find the start of the get_reward function
        # Match any get_reward signature: get_reward(self), get_reward(self, predicted_obs=None), etc.
        for i, line in enumerate(lines):
            if line.strip().startswith("def get_reward(self"):
                func_start = i
                break

        if func_start is None:
            logger.error("No get_reward function found in LLM response")
            return None

        # Extract complete function with proper indentation
        func_lines = []
        indent_level = len(lines[func_start]) - len(lines[func_start].lstrip())

        for line in lines[func_start:]:
            if line.strip() == "":
                # Include empty lines within function
                func_lines.append(line)
            elif (
                len(line) - len(line.lstrip()) <= indent_level
                and line.strip()
                and func_lines
            ):
                # Break when we reach code at same or lower indentation level
                break
            else:
                func_lines.append(line)

        extracted_code = "\n".join(func_lines)
        # Normalize indentation so the function is valid at module scope
        extracted_code = textwrap.dedent(extracted_code).lstrip()
        if not extracted_code.endswith("\n"):
            extracted_code += "\n"

        if not extracted_code.startswith("def get_reward(self"):
            logger.error("Extracted reward function does not start with the expected signature")
            return None

        logger.debug(f"Extracted reward function:\n{extracted_code}")

        return extracted_code

    except Exception as e:
        logger.error(f"Failed to extract reward function: {e}")
        return None


def safe_reward_injection(env_instance: Any, reward_code: str) -> bool:
    """
    Attempt to inject reward function into environment.

    This function attempts reward injection and returns success status.
    No fallback mechanism - if injection fails, return False and let
    training naturally fail (which the pipeline will detect and mark as failed).

    Args:
        env_instance: VisFly environment instance
        reward_code: Reward function code to inject

    Returns:
        bool: True if injection succeeded, False otherwise
    """
    logger = logging.getLogger(__name__)

    if inject_generated_reward(env_instance, reward_code):
        logger.info("Successfully injected generated reward function")
        return True

    # Injection failed - return False and let training fail naturally
    logger.warning("Reward function injection failed - training will likely fail")
    return False


class RewardInjector:
    """
    Static helper class for reward injection operations.

    This class provides a cleaner interface for reward injection operations
    and maintains injection history for debugging purposes.
    """

    injection_history = []

    @staticmethod
    def inject_reward_function(env_instance: Any, reward_code: str) -> bool:
        """
        Static method for reward function injection.

        Args:
            env_instance: VisFly environment instance
            reward_code: Reward function code to inject

        Returns:
            bool: True if injection succeeded, False otherwise
        """
        success = safe_reward_injection(env_instance, reward_code)

        # Record injection attempt
        RewardInjector.injection_history.append(
            {
                "timestamp": time.time(),
                "env_class": env_instance.__class__.__name__,
                "success": success,
                "reward_code_hash": hash(reward_code),
            }
        )

        return success


# Convenience function matching the specification
inject_reward_function = RewardInjector.inject_reward_function
