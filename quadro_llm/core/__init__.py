"""
Core modules for VisFly-Eureka integration.
"""

from .models import RewardFunctionResult, IterationSummary, OptimizationReport
from .reward_injector import RewardInjector
from .training_manager import TrainingManager
from .reward_evaluator import RewardEvaluator

__all__ = [
    'RewardFunctionResult',
    'IterationSummary', 
    'OptimizationReport',
    'RewardInjector',
    'TrainingManager',
    'RewardEvaluator',
]