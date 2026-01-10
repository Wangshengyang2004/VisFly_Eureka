"""
Quadro-LLM: LLM-powered reward optimization for autonomous drone control.

A modular system that combines VisFly drone simulation with LLM-generated
reward functions for diverse autonomous flight tasks.
"""

from .core.models import (
    RewardFunctionResult,
    IterationSummary,
    OptimizationReport,
    OptimizationConfig,
    TrainingResult,
)
from .pipeline import EurekaPipeline
from .eureka_visfly import EurekaVisFly

__version__ = "1.0.0"

__all__ = [
    "RewardFunctionResult",
    "IterationSummary",
    "OptimizationReport",
    "OptimizationConfig",
    "TrainingResult",
    "EurekaPipeline",
    "EurekaVisFly",
]
