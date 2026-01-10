"""
Reward function injection into VisFly environments.
"""

import logging
import types
import torch
import torch.nn.functional as F
from typing import Any, Optional


class RewardInjector:
    """Handles injection of reward functions into VisFly environments."""
    
    def __init__(self):
        """Initialize reward injector."""
        self.logger = logging.getLogger(__name__)
        
    def inject_reward_function(
        self,
        env_instance: Any,
        reward_code: str,
        validate: bool = True
    ) -> bool:
        """
        Inject a reward function into an environment instance.
        
        Args:
            env_instance: VisFly environment instance
            reward_code: Python code defining get_reward(self) method
            validate: Whether to validate the reward function
            
        Returns:
            True if injection successful, False otherwise
        """
        try:
            # Create execution namespace with required imports
            exec_namespace = {
                'torch': torch,
                'th': torch,  # Alias used in VisFly
                'F': F,  # Functional operations
                '__builtins__': {},  # Restrict builtins for safety
                # Add safe math functions
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'len': len,
                'float': float,
                'int': int,
                'bool': bool,
            }
            
            # Execute the reward function code
            exec(reward_code, exec_namespace)
            
            # Extract the get_reward function
            if 'get_reward' not in exec_namespace:
                self.logger.error("No get_reward function found in code")
                return False
            
            reward_func = exec_namespace['get_reward']
            
            # Bind the function to the environment instance
            env_instance.get_reward = types.MethodType(reward_func, env_instance)
            
            # Validate if requested
            if validate:
                success = self._validate_reward_function(env_instance)
                if not success:
                    self.logger.error("Reward function validation failed")
                    return False
            
            self.logger.debug("Successfully injected reward function")
            return True
            
        except SyntaxError as e:
            self.logger.error(f"Syntax error in reward code: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to inject reward function: {e}")
            return False
    
    def _validate_reward_function(self, env_instance: Any) -> bool:
        """
        Validate that the injected reward function works correctly.
        
        Args:
            env_instance: Environment with injected reward function
            
        Returns:
            True if validation passes
        """
        try:
            # Reset environment to get initial state
            env_instance.reset()
            
            # Try to get reward
            reward = env_instance.get_reward()
            
            # Check reward shape and type
            if not isinstance(reward, torch.Tensor):
                self.logger.error(f"Reward is not a tensor: {type(reward)}")
                return False
            
            # Check shape matches number of agents
            expected_shape = (env_instance.num_agent,)
            if reward.shape != expected_shape:
                self.logger.error(
                    f"Reward shape {reward.shape} doesn't match expected {expected_shape}"
                )
                return False
            
            # Check for NaN or Inf values
            if torch.isnan(reward).any():
                self.logger.error("Reward contains NaN values")
                return False
            
            if torch.isinf(reward).any():
                self.logger.error("Reward contains infinite values")
                return False
            
            # Take a step to ensure reward function works during episode
            if hasattr(env_instance, 'action_space'):
                action = env_instance.action_space.sample()
                env_instance.step(action)
                reward_after_step = env_instance.get_reward()
                
                # Same checks after step
                if not isinstance(reward_after_step, torch.Tensor):
                    self.logger.error("Reward after step is not a tensor")
                    return False
                
                if reward_after_step.shape != expected_shape:
                    self.logger.error("Reward shape changed after step")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False
    
    def create_safe_wrapper(
        self,
        env_instance: Any,
        reward_code: str
    ) -> Optional[Any]:
        """
        Create a wrapped environment with safe reward function execution.
        
        Args:
            env_instance: Original environment instance
            reward_code: Reward function code
            
        Returns:
            Wrapped environment or None if wrapping fails
        """
        class SafeRewardWrapper:
            """Wrapper that safely executes reward functions."""
            
            def __init__(self, env, reward_func):
                self.env = env
                self.reward_func = reward_func
                self.logger = logging.getLogger(__name__)
                
                # Proxy all attributes to wrapped environment
                self._update_attributes()
            
            def _update_attributes(self):
                """Update wrapper attributes from wrapped environment."""
                for attr in dir(self.env):
                    if not attr.startswith('_') and attr != 'get_reward':
                        setattr(self, attr, getattr(self.env, attr))
            
            def get_reward(self):
                """Safe reward function execution."""
                try:
                    return self.reward_func()
                except Exception as e:
                    self.logger.error(f"Reward function error: {e}")
                    # Return zero rewards as fallback
                    return torch.zeros(self.env.num_agent, device=self.env.device)
            
            def __getattr__(self, name):
                """Proxy attribute access to wrapped environment."""
                return getattr(self.env, name)
        
        try:
            # Create execution namespace
            exec_namespace = {
                'torch': torch,
                'th': torch,
                'F': F,
                'self': env_instance,  # Bind self to environment
            }
            
            # Create reward function
            exec(reward_code, exec_namespace)
            reward_func = exec_namespace.get('get_reward')
            
            if not reward_func:
                return None
            
            # Create wrapped environment
            wrapped = SafeRewardWrapper(env_instance, lambda: reward_func(env_instance))
            return wrapped
            
        except Exception as e:
            self.logger.error(f"Failed to create safe wrapper: {e}")
            return None