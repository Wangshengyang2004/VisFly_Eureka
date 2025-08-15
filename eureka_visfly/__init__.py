"""
VisFly-Eureka Native Integration

This package provides native integration between VisFly drone simulation environments
and Eureka's LLM-powered reward function optimization.

Key Components:
- EurekaVisFly: Main orchestrator class
- LLMEngine: Interface for reward function generation
- inject_generated_reward: Direct reward injection into environments
- TrainingResult: Performance tracking and evaluation

Usage:
    from eureka_visfly import EurekaVisFly
    from VisFly.envs.NavigationEnv import NavigationEnv
    
    eureka = EurekaVisFly(
        env_class=NavigationEnv,
        task_description="Navigate to target avoiding obstacles",
        llm_config={"model": "gpt-4", "api_key": "your-key"}
    )
    
    best_rewards = eureka.optimize_rewards(iterations=5, samples=16)
"""

from .eureka_visfly import EurekaVisFly, OptimizationConfig
from .llm_engine import LLMEngine
from .reward_injection import (
    inject_generated_reward, 
    RewardInjector,
    safe_reward_injection,
    extract_reward_function
)
from .training_utils import (
    TrainingResult, 
    train_with_generated_reward,
    evaluate_model_performance,
    TrainingMonitor
)
from .prompts import (
    create_system_prompt,
    create_user_prompt,
    create_improvement_prompt,
    get_task_specific_prompt_suffix
)
from .production_pipeline import (
    EurekaNavigationPipeline,
    OptimizationReport,
    IterationSummary
)

__version__ = "0.1.0"
__author__ = "VisFly-Eureka Team"

__all__ = [
    # Main classes
    "EurekaVisFly",
    "OptimizationConfig",
    "LLMEngine",
    "RewardInjector",
    "TrainingResult",
    "TrainingMonitor",
    
    # Core functions
    "inject_generated_reward",
    "safe_reward_injection", 
    "extract_reward_function",
    "train_with_generated_reward",
    "evaluate_model_performance",
    
    # Prompt utilities
    "create_system_prompt",
    "create_user_prompt", 
    "create_improvement_prompt",
    "get_task_specific_prompt_suffix",
]