"""
LLM Engine for VisFly Reward Function Generation

This module handles communication with LLM APIs (OpenAI) to generate reward functions
specifically designed for VisFly drone environments.
"""

import logging
import time
import asyncio
import concurrent.futures
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI

# Note: LLM configuration comes from config files, not constants
from ..core.exceptions import (
    LLMError,
    APIConnectionError,
    ValidationError,
    handle_and_log_error,
    ErrorContext,
)
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
        model: str,
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        max_retries: int,
        thinking_enabled: bool = True,
        batching_strategy: str = "n_parameter",
        supports_n_parameter: bool = True,
        max_concurrent: int = 10,
    ):
        """
        Initialize the LLM engine.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo", "glm-4.5")
            api_key: API key (if None, will try to get from environment)
            base_url: Base URL for API calls (for custom endpoints)
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            thinking_enabled: Whether to enable thinking chain for supported models (GLM-4.5+)
            batching_strategy: Strategy for batching multiple requests ("n_parameter", "sequential", "async", "multiprocessing")
            supports_n_parameter: Whether the API supports the n parameter for multiple completions
            max_concurrent: Maximum concurrent requests for async/multiprocessing strategies
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.thinking_enabled = thinking_enabled
        self.batching_strategy = batching_strategy
        self.supports_n_parameter = supports_n_parameter
        self.max_concurrent = max_concurrent

        self.logger = logging.getLogger(__name__)

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.logger.info(f"Initialized LLM engine with model: {model}")
        
        # Initialize conversation logging
        self.conversations: List[Dict[str, Any]] = []

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

        # Use appropriate batching strategy based on configuration
        if self.batching_strategy == "n_parameter" and self.supports_n_parameter:
            results = self._generate_with_n_parameter(messages, samples)
        elif self.batching_strategy == "sequential":
            results = self._generate_sequential(messages, samples)
        elif self.batching_strategy == "async":
            results = self._generate_async(messages, samples)
        elif self.batching_strategy == "multiprocessing":
            results = self._generate_multiprocessing(messages, samples)
        else:
            # Fallback to sequential if strategy is unknown
            self.logger.warning(f"Unknown batching strategy '{self.batching_strategy}', falling back to sequential")
            results = self._generate_sequential(messages, samples)
        
        # Log the conversation
        self._log_conversation(messages, results, task_description, feedback, samples)
        
        return results

    def _generate_with_n_parameter(self, messages: List[Dict], samples: int) -> List[str]:
        """Generate multiple samples using API's n parameter (OpenAI style)."""
        reward_functions = []
        batch_size = min(samples, 10)  # OpenAI allows max 10 choices per request
        remaining = samples

        while remaining > 0:
            current_batch = min(remaining, batch_size)
            batch_functions = self._generate_single_batch_with_n(messages, current_batch)
            reward_functions.extend(batch_functions)
            remaining -= len(batch_functions)

            if remaining > 0:
                time.sleep(1)  # Brief pause between batches

        return reward_functions

    def _generate_single_batch_with_n(self, messages: List[Dict], n_samples: int) -> List[str]:
        """
        Generate a single batch using n parameter.

        Args:
            messages: Chat messages for the API call
            n_samples: Number of samples to generate in this batch

        Returns:
            List of extracted reward function code strings
        """
        for attempt in range(self.max_retries):
            try:
                # Prepare request parameters
                request_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "n": n_samples,
                    "timeout": self.timeout,
                }
                
                # Add thinking configuration based on config setting
                if hasattr(self, 'thinking_enabled') and not self.thinking_enabled:
                    request_params["extra_body"] = {"thinking": {"type": "disabled"}}
                
                response = self.client.chat.completions.create(**request_params)

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

    def _generate_sequential(self, messages: List[Dict], samples: int) -> List[str]:
        """Generate samples sequentially (one API call per sample)."""
        reward_functions = []
        
        for i in range(samples):
            self.logger.debug(f"Generating sample {i+1}/{samples}")
            
            for attempt in range(self.max_retries):
                try:
                    request_params = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "timeout": self.timeout,
                    }
                    
                    # Add thinking configuration based on config setting
                    if hasattr(self, 'thinking_enabled') and not self.thinking_enabled:
                        request_params["extra_body"] = {"thinking": {"type": "disabled"}}
                    
                    response = self.client.chat.completions.create(**request_params)
                    
                    # Extract reward function from single response
                    if response.choices:
                        content = response.choices[0].message.content
                        reward_code = extract_reward_function(content)
                        if reward_code:
                            reward_functions.append(reward_code)
                        else:
                            self.logger.warning(f"Could not extract reward function from sample {i+1}")
                    
                    break  # Success, move to next sample
                    
                except Exception as e:
                    self.logger.warning(f"Sample {i+1} attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2**attempt)  # Exponential backoff
                    else:
                        self.logger.error(f"Sample {i+1} failed after all retries")
            
            # Brief pause between samples to respect rate limits
            if i < samples - 1:
                time.sleep(0.5)
        
        return reward_functions

    def _generate_async(self, messages: List[Dict], samples: int) -> List[str]:
        """Generate samples asynchronously."""
        import asyncio
        
        async def generate_single_async(session_id: int) -> Optional[str]:
            try:
                request_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "timeout": self.timeout,
                }
                
                # Add thinking configuration based on config setting
                if hasattr(self, 'thinking_enabled') and not self.thinking_enabled:
                    request_params["extra_body"] = {"thinking": {"type": "disabled"}}
                
                response = self.client.chat.completions.create(**request_params)
                
                if response.choices:
                    content = response.choices[0].message.content
                    return extract_reward_function(content)
                
            except Exception as e:
                self.logger.warning(f"Async sample {session_id} failed: {e}")
            
            return None
        
        async def run_async_batch():
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def bounded_generate(session_id):
                async with semaphore:
                    return await generate_single_async(session_id)
            
            tasks = [bounded_generate(i) for i in range(samples)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return [r for r in results if isinstance(r, str) and r]
        
        # Run async batch
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(run_async_batch())

    def _generate_multiprocessing(self, messages: List[Dict], samples: int) -> List[str]:
        """Generate samples using multiprocessing."""
        def generate_single_mp(session_id: int) -> Optional[str]:
            try:
                # Create new client instance for this process
                client_kwargs = {}
                if hasattr(self, 'client'):
                    client_kwargs = self.client._client._default_headers  # Copy headers if needed
                
                # Note: This is simplified - in practice you'd need to properly recreate the client
                # with the same configuration in each process
                request_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "timeout": self.timeout,
                }
                
                # Add thinking configuration based on config setting
                if hasattr(self, 'thinking_enabled') and not self.thinking_enabled:
                    request_params["extra_body"] = {"thinking": {"type": "disabled"}}
                
                response = self.client.chat.completions.create(**request_params)
                
                if response.choices:
                    content = response.choices[0].message.content
                    return extract_reward_function(content)
                    
            except Exception as e:
                self.logger.warning(f"Multiprocessing sample {session_id} failed: {e}")
            
            return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = [executor.submit(generate_single_mp, i) for i in range(samples)]
            results = []
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.warning(f"Multiprocessing task failed: {e}")
            
            return results

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

    def test_api_connection(self) -> None:
        """
        Test the API connection with a simple request.
        Raises exception if connection fails.
        """
        # Prepare request parameters
        request_params = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "timeout": 30,  # API timeout - should come from config
        }
        
        # Add thinking configuration based on config setting
        if hasattr(self, 'thinking_enabled') and not self.thinking_enabled:
            request_params["extra_body"] = {"thinking": {"type": "disabled"}}
        
        response = self.client.chat.completions.create(**request_params)

        self.logger.info("API connection test successful")

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
            validation_error = ValidationError(
                "Reward function has syntax errors",
                details={"syntax_error": str(e)},
                cause=e
            )
            handle_and_log_error(
                self.logger, validation_error, "reward function validation", reraise=False
            )
            return False
        except Exception as e:
            handle_and_log_error(
                self.logger, e, "reward function validation", reraise=False
            )
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

    def _log_conversation(
        self,
        messages: List[Dict[str, Any]],
        results: List[str],
        task_description: str,
        feedback: str,
        samples: int,
    ) -> None:
        """Log the conversation for debugging and analysis."""
        conversation = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "task_description": task_description,
            "feedback": feedback,
            "samples_requested": samples,
            "samples_generated": len(results),
            "messages": messages,
            "results": results,
            "model_config": {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        }
        self.conversations.append(conversation)

    def save_conversations(self, output_dir: str, iteration: int) -> None:
        """Save all conversations to artifacts directory."""
        output_path = Path(output_dir)
        artifacts_dir = output_path / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        conversation_file = artifacts_dir / f"llm_conversations_iteration_{iteration}.json"
        
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(self.conversations)} conversations to {conversation_file}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
