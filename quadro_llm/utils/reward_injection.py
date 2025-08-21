"""
Direct Reward Function Injection for VisFly Environments

This module provides functionality to directly inject LLM-generated reward functions
into existing VisFly environment instances without wrapper layers.
"""

import torch
import types
import logging
import ast
import inspect
from typing import Any, Dict, Optional, Callable


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
    original_get_reward = getattr(env_instance, 'get_reward', None)
    env_instance._original_get_reward = original_get_reward
    
    try:
        # Create execution environment with necessary imports
        exec_globals = {
            'torch': torch,
            'th': torch,  # Common abbreviation in VisFly
            'F': torch.nn.functional,
            '__builtins__': __builtins__
        }
        
        # Execute the reward function code
        exec(reward_code, exec_globals)
        
        # Extract the new reward function
        if 'get_reward' not in exec_globals:
            logger.error("No get_reward function found in generated code")
            return False
            
        new_reward_func = exec_globals['get_reward']
        
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
            if isinstance(node, ast.FunctionDef) and node.name == 'get_reward':
                # Check if function has self parameter
                if not node.args.args or node.args.args[0].arg != 'self':
                    return None
                
                # Compile and extract the function
                code_obj = compile(tree, '<reward_function>', 'exec')
                namespace = {
                    'torch': torch,
                    'th': torch,
                    'F': torch.nn.functional,
                    '__builtins__': __builtins__
                }
                exec(code_obj, namespace)
                
                if 'get_reward' in namespace:
                    return namespace['get_reward']
        
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


def _test_injected_reward(env_instance: Any, original_get_reward: Optional[Any]) -> bool:
    """
    Test the injected reward function to ensure it works correctly.
    
    Args:
        env_instance: Environment with injected reward function
        original_get_reward: Original reward function for fallback
        
    Returns:
        bool: True if reward function works correctly
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Try multiple times in case of transient issues
        for attempt in range(3):
            try:
                # Reset environment to get valid state
                env_instance.reset()
                
                # Call the reward function
                reward = env_instance.get_reward()
                
                # Validate reward output
                if not isinstance(reward, torch.Tensor):
                    logger.error(f"Reward function returned {type(reward)}, expected torch.Tensor")
                    continue
                    
                # Check reward shape is reasonable (allow flexibility for different batch sizes)
                if reward.dim() > 2 or reward.numel() == 0:
                    logger.error(f"Invalid reward shape: {reward.shape}")
                    continue
                    
                # Check for valid values (no NaN or infinite values)
                if torch.isnan(reward).any() or torch.isinf(reward).any():
                    logger.error("Reward function returned NaN or infinite values")
                    continue
                
                # Additional sanity checks
                if reward.numel() > 1000:  # Suspiciously large reward tensor
                    logger.warning(f"Large reward tensor: {reward.shape}")
                
                # Check if reward values are reasonable (not extremely large)
                if torch.abs(reward).max() > 1e6:
                    logger.warning(f"Very large reward values detected: max={torch.abs(reward).max()}")
                
                logger.debug(f"Reward function test passed: shape={reward.shape}, range=[{reward.min():.3f}, {reward.max():.3f}]")
                return True
                
            except Exception as e:
                logger.warning(f"Reward test attempt {attempt + 1} failed: {e}")
                if attempt == 2:  # Last attempt
                    raise
                
        return False
        
    except Exception as e:
        logger.error(f"Reward function test failed after all attempts: {e}")
        
        # Restore original if test fails
        if original_get_reward:
            try:
                env_instance.get_reward = original_get_reward
                logger.info("Restored original reward function after test failure")
            except Exception as restore_error:
                logger.error(f"Failed to restore original reward function: {restore_error}")
            
        return False


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
        lines = llm_response.strip().split('\n')
        func_start = None
        
        # Find the start of the get_reward function
        for i, line in enumerate(lines):
            if line.strip().startswith('def get_reward(self)'):
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
            elif len(line) - len(line.lstrip()) <= indent_level and line.strip() and func_lines:
                # Break when we reach code at same or lower indentation level
                break
            else:
                func_lines.append(line)
        
        extracted_code = '\n'.join(func_lines)
        logger.debug(f"Extracted reward function:\n{extracted_code}")
        
        return extracted_code
        
    except Exception as e:
        logger.error(f"Failed to extract reward function: {e}")
        return None


def create_fallback_reward(env_instance: Any) -> str:
    """
    Create a basic fallback reward function for the given environment.
    
    Args:
        env_instance: VisFly environment instance
        
    Returns:
        str: Fallback reward function code
    """
    # Determine environment type and create appropriate fallback
    env_name = env_instance.__class__.__name__
    
    if "Navigation" in env_name:
        return """
def get_reward(self):
    \"\"\"Fallback reward for navigation tasks\"\"\"
    if hasattr(self, 'target') and hasattr(self, 'position'):
        # Simple distance-based reward
        distance_to_target = torch.norm(self.position - self.target, dim=1)
        return -distance_to_target * 0.1
    else:
        # Default survival reward
        return torch.ones(getattr(self, 'num_agent', 1)) * 0.01
"""
    elif "Racing" in env_name:
        return """
def get_reward(self):
    \"\"\"Fallback reward for racing tasks\"\"\"
    if hasattr(self, 'velocity'):
        # Reward forward movement
        return torch.norm(self.velocity, dim=1) * 0.1
    else:
        return torch.ones(getattr(self, 'num_agent', 1)) * 0.01
"""
    elif "Hover" in env_name:
        return """
def get_reward(self):
    \"\"\"Fallback reward for hovering tasks\"\"\"
    if hasattr(self, 'velocity'):
        # Reward staying still
        return -torch.norm(self.velocity, dim=1) * 0.1
    else:
        return torch.ones(getattr(self, 'num_agent', 1)) * 0.01
"""
    else:
        # Generic fallback
        return """
def get_reward(self):
    \"\"\"Generic fallback reward\"\"\"
    return torch.ones(getattr(self, 'num_agent', 1)) * 0.01
"""


def safe_reward_injection(env_instance: Any, reward_code: str) -> bool:
    """
    Safely inject reward function with comprehensive error handling.
    
    This function attempts reward injection and falls back to a safe default
    if the injection fails at any point.
    
    Args:
        env_instance: VisFly environment instance
        reward_code: Reward function code to inject
        
    Returns:
        bool: True if any valid reward function is active (original or fallback)
    """
    logger = logging.getLogger(__name__)
    
    # Store the original reward function
    original_get_reward = getattr(env_instance, 'get_reward', None)
    
    # Try to inject the generated reward function
    if inject_generated_reward(env_instance, reward_code):
        logger.info("Successfully injected generated reward function")
        return True
    
    # If injection failed, try fallback reward
    logger.warning("Generated reward injection failed, trying fallback reward")
    
    try:
        fallback_code = create_fallback_reward(env_instance)
        if inject_generated_reward(env_instance, fallback_code):
            logger.info("Successfully injected fallback reward function")
            return True
    except Exception as e:
        logger.error(f"Fallback reward injection failed: {e}")
    
    # If everything failed, ensure original reward is restored
    if original_get_reward:
        env_instance.get_reward = original_get_reward
        logger.info("Restored original reward function")
        return True
    
    logger.error("No valid reward function available")
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
            bool: True if injection succeeded
        """
        success = safe_reward_injection(env_instance, reward_code)
        
        # Record injection attempt
        RewardInjector.injection_history.append({
            'timestamp': torch.tensor(0.0).cpu(),  # Placeholder for timestamp
            'env_class': env_instance.__class__.__name__,
            'success': success,
            'reward_code_hash': hash(reward_code)
        })
        
        return success
    
    @staticmethod
    def get_injection_history():
        """Get history of all reward injection attempts."""
        return RewardInjector.injection_history.copy()
    
    @staticmethod
    def clear_history():
        """Clear injection history."""
        RewardInjector.injection_history.clear()


# Convenience function matching the specification
inject_reward_function = RewardInjector.inject_reward_function