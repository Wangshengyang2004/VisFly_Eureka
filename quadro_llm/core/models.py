"""
Data models for VisFly-Eureka pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class TrainingResult:
    """Results from training a reward function."""
    success_rate: float
    episode_length: float
    training_time: float
    final_reward: float
    convergence_step: int
    reward_code: str = ""
    identifier: str = ""
    log_dir: str = None  # Path to tensorboard logs


@dataclass
class OptimizationConfig:
    """Configuration for reward optimization process"""
    iterations: int = 5
    samples: int = 16
    algorithm: str = "bptt"  # "bptt", "ppo", or "shac"
    evaluation_episodes: int = 10
    success_threshold: float = 0.8
    record_video: bool = False
    gpu_memory_requirement_mb: int = 2048
    # History pruning: number of recent iterations to keep in LLM context
    # -1 = keep all history (no pruning)
    # 0 = Eureka-style: only keep current iteration (aggressive pruning)
    # 1 = keep last iteration (similar to LaRes)
    # 2+ = keep last N iterations (balanced approach)
    history_window_size: int = 2


@dataclass
class RewardFunctionResult:
    """Complete result for a single reward function evaluation"""
    reward_code: str
    identifier: str
    training_successful: bool
    success_rate: float
    episode_length: float
    training_time: float
    final_reward: float
    convergence_step: int
    tensorboard_logs: Dict[str, List[float]] = None
    error_message: str = ""
    log_dir: Optional[str] = None  # Path to tensorboard logs
    peak_memory_mb: float = 0.0  # Peak GPU memory usage in MB
    evaluation_summary: Optional[Dict[str, Any]] = None
    episode_statistics: List[Dict[str, Any]] = field(default_factory=list)
    video_paths: List[str] = field(default_factory=list)
    
    def score(self) -> float:
        """Calculate composite score for ranking"""
        if not self.training_successful:
            return -10000.0  # DUMMY_FAILURE
        
        # Primary: success rate, Secondary: episode efficiency
        efficiency_bonus = max(0, (256 - self.episode_length) / 256) * 0.3
        return self.success_rate * 0.7 + efficiency_bonus
    
    @classmethod
    def failed(cls, reward_code: str, identifier: str, error: str = ""):
        """Create failed result"""
        return cls(
            reward_code=reward_code,
            identifier=identifier,
            training_successful=False,
            success_rate=-1.0,
            episode_length=0.0,
            training_time=0.0,
            final_reward=0.0,
            convergence_step=0,
            error_message=error,
            log_dir=None,
            evaluation_summary=None,
            episode_statistics=[],
            video_paths=[],
        )


@dataclass
class IterationSummary:
    """Summary of one optimization iteration"""
    iteration: int
    samples: List[RewardFunctionResult]
    best_sample_idx: int
    best_success_rate: float
    best_correlation: float
    execution_rate: float
    generation_time: float
    total_training_time: float


@dataclass
class OptimizationReport:
    """Final report from the optimization pipeline."""
    
    total_samples: int
    successful_samples: int
    best_performance: Dict[str, float]
    improvement_metrics: Dict[str, float]
    execution_time: float
    output_directory: str
    iteration_history: List[IterationSummary]
    best_reward_code: Optional[str] = None
    baseline_performance: Optional[Dict[str, float]] = None
    best_artifacts_dir: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            'total_samples': self.total_samples,
            'successful_samples': self.successful_samples,
            'best_performance': self.best_performance,
            'improvement_metrics': self.improvement_metrics,
            'execution_time': self.execution_time,
            'output_directory': self.output_directory,
            'iteration_history': [
                {
                    'iteration': s.iteration,
                    'num_samples': len(s.samples),
                    'successful_samples': sum(1 for r in s.samples if r.training_successful),
                    'best_success_rate': s.best_success_rate,
                    'execution_rate': s.execution_rate,
                }
                for s in self.iteration_history
            ],
            'best_reward_code': self.best_reward_code,
            'baseline_performance': self.baseline_performance,
            'best_artifacts_dir': self.best_artifacts_dir,
        }
