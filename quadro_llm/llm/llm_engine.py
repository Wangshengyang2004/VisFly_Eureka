"""
LLM Engine for VisFly Reward Function Generation

This module handles communication with LLM APIs (OpenAI) to generate reward functions
specifically designed for VisFly drone environments.
"""

import logging
import time
from typing import List, Dict, Any, Optional
import json

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None

from .prompts import (
    create_system_prompt,
    create_user_prompt,
    extract_env_code_without_reward,
)
from ..utils.reward_injection import extract_reward_function


class LLMEngine:
    """
    Engine for generating reward functions using Large Language Models.

    This class handles API communication, prompt engineering, and response parsing
    for generating VisFly-compatible reward functions.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 1500,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """
        Initialize the LLM engine.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (if None, will try to get from environment)
            base_url: Base URL for API calls (for custom endpoints)
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client
        if OpenAI is None:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        try:
            client_kwargs = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if base_url:
                client_kwargs["base_url"] = base_url

            self.client = OpenAI(**client_kwargs)
            self.logger.info(f"Initialized LLM engine with model: {model}")

        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def generate_reward_functions(
        self,
        task_description: str,
        context_info: Dict[str, Any],
        feedback: str = "",
        samples: int = 16,
        env_class=None,
    ) -> List[str]:
        """
        Generate multiple reward function candidates for a given task.

        Args:
            task_description: Natural language description of the task
            context_info: Environment context (sensors, dimensions, etc.)
            feedback: Feedback from previous iterations
            samples: Number of reward functions to generate
            env_class: Environment class to extract code from

        Returns:
            List of reward function code strings
        """
        # Generate reward functions

        # Extract environment code without reward (like real Eureka)
        env_code = ""
        if env_class:
            env_code = extract_env_code_without_reward(env_class)

        # Create prompts
        system_prompt = create_system_prompt()
        user_prompt = create_user_prompt(
            task_description, context_info, feedback, env_code
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        reward_functions = []

        # Generate samples in batches to respect API limits
        batch_size = min(samples, 10)  # OpenAI allows max 10 choices per request
        remaining = samples

        while remaining > 0:
            current_batch = min(remaining, batch_size)

            try:
                batch_functions = self._generate_batch(messages, current_batch)
                reward_functions.extend(batch_functions)
                remaining -= len(batch_functions)

                if remaining > 0:
                    time.sleep(1)  # Brief pause between batches

            except Exception as e:
                self.logger.error(f"Failed to generate batch: {e}")
                break

        # Successfully generated reward functions
        return reward_functions

    def _generate_batch(self, messages: List[Dict], n_samples: int) -> List[str]:
        """
        Generate a batch of reward functions.

        Args:
            messages: Chat messages for the API call
            n_samples: Number of samples to generate

        Returns:
            List of extracted reward function code strings
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=n_samples,
                    timeout=self.timeout,
                )

                # Extract reward functions from responses
                reward_functions = []
                for choice in response.choices:
                    content = choice.message.content
                    reward_code = extract_reward_function(content)

                    if reward_code:
                        reward_functions.append(reward_code)
                    else:
                        self.logger.warning(
                            "Could not extract reward function from response"
                        )

                return reward_functions

            except Exception as e:
                self.logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    self.logger.error(f"All {self.max_retries} attempts failed")
                    raise

        return []

    def generate_single_reward_function(
        self, task_description: str, context_info: Dict[str, Any], feedback: str = ""
    ) -> Optional[str]:
        """
        Generate a single reward function.

        Args:
            task_description: Natural language description of the task
            context_info: Environment context
            feedback: Feedback from previous iterations

        Returns:
            Single reward function code string or None if generation failed
        """
        reward_functions = self.generate_reward_functions(
            task_description, context_info, feedback, samples=1
        )

        return reward_functions[0] if reward_functions else None

    def test_api_connection(self) -> bool:
        """
        Test the API connection with a simple request.

        Returns:
            bool: True if connection works, False otherwise
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                timeout=30,
            )

            self.logger.info("API connection test successful")
            return True

        except Exception as e:
            self.logger.error(f"API connection test failed: {e}")
            return False

    def validate_reward_function(self, reward_code: str) -> bool:
        """
        Validate that a reward function has correct syntax and structure.

        Args:
            reward_code: Reward function code to validate

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check for basic structure
            if "def get_reward(self)" not in reward_code:
                self.logger.error("Reward function missing proper signature")
                return False

            # Try to compile the code
            exec_globals = {"torch": None, "th": None, "F": None}
            exec(reward_code, exec_globals)

            # Check that get_reward function was defined
            if "get_reward" not in exec_globals:
                self.logger.error("get_reward function not found after compilation")
                return False

            self.logger.debug("Reward function validation passed")
            return True

        except SyntaxError as e:
            self.logger.error(f"Syntax error in reward function: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False

    def improve_reward_function(
        self,
        original_code: str,
        performance_feedback: str,
        task_description: str,
        context_info: Dict[str, Any],
    ) -> Optional[str]:
        """
        Generate an improved version of a reward function based on performance feedback.

        Args:
            original_code: Original reward function code
            performance_feedback: Feedback on performance issues
            task_description: Task description
            context_info: Environment context

        Returns:
            Improved reward function code or None if improvement failed
        """
        improvement_prompt = f"""
        The following reward function has performance issues:
        
        ```python
        {original_code}
        ```
        
        Performance feedback: {performance_feedback}
        
        Please provide an improved version that addresses these issues while maintaining the same structure.
        """

        return self.generate_single_reward_function(
            task_description, context_info, improvement_prompt
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
