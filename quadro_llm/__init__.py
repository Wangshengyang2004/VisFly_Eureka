"""
Quadro-LLM: LLM-powered reward optimization for autonomous drone navigation.

A modular system that combines VisFly drone simulation with LLM-generated 
reward functions for autonomous navigation tasks.
"""

from .core.models import RewardFunctionResult, IterationSummary, OptimizationReport, OptimizationConfig
from .pipeline import EurekaNavigationPipeline
from .eureka_visfly import EurekaVisFly

__version__ = "1.0.0"

__all__ = [
    'RewardFunctionResult',
    'IterationSummary', 
    'OptimizationReport',
    'OptimizationConfig',
    'EurekaNavigationPipeline',
    'EurekaVisFly',
]
