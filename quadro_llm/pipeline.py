"""
Main optimization pipeline for VisFly-Eureka.

This module contains the EurekaNavigationPipeline class that orchestrates
the complete reward optimization workflow.
"""

import os
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List

from .eureka_visfly import EurekaVisFly
from .core.models import OptimizationConfig, OptimizationReport
from .training.parallel_training import ParallelTrainingManager
from .utils.gpu_monitor import GPUMonitor, DynamicGPUResourceManager


class EurekaNavigationPipeline:
    """
    Production pipeline for NavigationEnv reward optimization using Eureka methodology.
    
    This pipeline implements the complete Eureka workflow:
    1. Iterative reward function generation with LLM
    2. Direct injection into VisFly NavigationEnv
    3. Training with comprehensive logging
    4. Result analysis and ranking
    5. Best function selection
    """
    
    def __init__(
        self,
        task_description: str,
        llm_config: Dict[str, Any],
        env_config: Dict[str, Any] = None,
        optimization_config: Dict[str, Any] = None,
        logging_config: Dict[str, Any] = None,
        output_dir: str = "./eureka_output",
        env_class=None
    ):
        """
        Initialize the production pipeline.
        
        Args:
            task_description: Natural language description of navigation task
            llm_config: LLM configuration (API key, model, etc.)
            env_config: NavigationEnv configuration
            optimization_config: Optimization parameters (iterations, samples, etc.)
            logging_config: Logging and output configuration
            output_dir: Directory for saving results
        """
        self.task_description = task_description
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self.setup_logging(logging_config or {})
        
        # Configuration
        self.env_config = env_config or self._default_env_config()
        self.opt_config = optimization_config or self._default_optimization_config()
        self.llm_config = llm_config
        self.env_class = env_class
        
        # Setup artifacts directory
        self.artifacts_dir = self.output_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize GPU monitoring and resource management
        self.gpu_monitor = GPUMonitor()
        self.gpu_resource_manager = DynamicGPUResourceManager(self.gpu_monitor)
        
        # Initialize parallel training manager
        self.training_manager = ParallelTrainingManager(
            results_dir=str(self.output_dir),
            gpu_resource_manager=self.gpu_resource_manager
        )
        
        # Initialize components
        self._initialize_pipeline()
        
        # Pipeline initialized
    
    def setup_logging(self, logging_config: Dict[str, Any]):
        """Setup comprehensive logging"""
        log_level = getattr(logging, logging_config.get("level", "INFO"))
        log_file = self.output_dir / "pipeline.log"
        
        # Create logger
        self.logger = logging.getLogger("EurekaNavigationPipeline")
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Disable propagation to prevent duplicate console output
        self.logger.propagate = False
    
    def _default_env_config(self) -> Dict[str, Any]:
        """Default NavigationEnv configuration"""
        return {
            "num_agent_per_scene": 2,
            "num_scene": 1,
            "visual": True,
            "device": "cpu",
            "requires_grad": True,
            "max_episode_steps": 16,
            "sensor_kwargs": [{
                "sensor_type": "DEPTH",
                "uuid": "depth",
                "resolution": [64, 64],
            }],
            "target": [[15.0, 0.0, 1.5]],
        }
    
    def _default_optimization_config(self) -> Dict[str, Any]:
        """Default optimization configuration"""
        return {
            "iterations": 5,
            "samples": 15, 
            "training_steps": 50,
            "algorithm": "bptt",
            "evaluation_episodes": 20,
            "success_threshold": 0.8,
        }
    
    def _initialize_pipeline(self):
        """Initialize pipeline components"""
        try:
            # Use provided environment class or default to NavigationEnv
            if self.env_class is None:
                from VisFly.envs.NavigationEnv import NavigationEnv
                self.env_class = NavigationEnv
            
            # Create Eureka controller
            self.eureka = EurekaVisFly(
                env_class=self.env_class,
                task_description=self.task_description,
                llm_config=self.llm_config,
                env_kwargs=self.env_config,
                optimization_config=OptimizationConfig(**self.opt_config),
                device=self.env_config["device"]
            )
            
            # Initialization complete - log at debug level only
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def run_optimization(self) -> OptimizationReport:
        """
        Run the complete optimization pipeline.
        
        Returns:
            OptimizationReport with comprehensive results and analysis
        """
        start_time = time.time()
        
        # Start optimization
        
        try:
            # Start parallel training manager
            self.training_manager.start()
            
            # Run iterative optimization
            results = self.eureka.optimize_rewards(
                iterations=self.opt_config["iterations"],
                samples=self.opt_config["samples"]
            )
            
            # Analyze results and create report
            final_report = self._create_optimization_report(results, time.time() - start_time)
            
            # Save outputs
            self._save_outputs(final_report)
            
            self.logger.info(f"Optimization completed in {final_report.execution_time:.1f}s")
            self.logger.info("="*60)
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise
        finally:
            # Stop parallel training manager
            self.training_manager.stop()
    
    def _create_optimization_report(self, results: List, execution_time: float) -> OptimizationReport:
        """Create optimization report from results"""
        
        # Calculate metrics
        successful_results = [r for r in results if r.success_rate >= 0]
        total_samples = len(results)
        successful_samples = len(successful_results)
        
        # Best performance
        best_performance = {}
        if successful_results:
            best = successful_results[0]  # Results are already sorted by score
            best_performance = {
                "success_rate": best.success_rate,
                "episode_length": best.episode_length,
                "training_time": best.training_time,
                "final_reward": best.final_reward,
                "score": best.score()
            }
        
        # Create iteration history from eureka optimization history
        iteration_history = []
        for i, iter_results in enumerate(self.eureka.optimization_history):
            if iter_results:
                best_in_iter = max(iter_results, key=lambda x: x.score())
                iteration_history.append({
                    "iteration": i + 1,
                    "best_success_rate": best_in_iter.success_rate,
                    "num_samples": len(iter_results),
                    "successful_samples": len([r for r in iter_results if r.success_rate >= 0])
                })
        
        return OptimizationReport(
            total_samples=total_samples,
            successful_samples=successful_samples,
            best_performance=best_performance,
            improvement_metrics={"baseline_comparison": "not_implemented"},
            execution_time=execution_time,
            output_directory=str(self.output_dir),
            iteration_history=iteration_history,
            best_reward_code=successful_results[0].reward_code if successful_results else None
        )
    
    def _save_outputs(self, report: OptimizationReport):
        """Save optimization results"""
        
        # Save optimization report
        report_file = self.output_dir / "optimization_report.json"
        with open(report_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        # Save best reward function
        if report.best_reward_code:
            best_function_file = self.output_dir / "best_reward_function.py"
            with open(best_function_file, 'w') as f:
                f.write(report.best_reward_code)
        
        self.logger.info(f"Results saved to {self.output_dir}")


def run_production_pipeline():
    """Run the complete production pipeline with default configuration"""
    
    # Task description for NavigationEnv
    TASK_DESCRIPTION = """
    Navigate drone to target position [15.0, 0.0, 1.5] efficiently while avoiding obstacles.

    The drone must:
    1. Reach the target position with high precision (within 0.5m)
    2. Avoid collisions using depth sensor data 
    3. Maintain stable flight dynamics
    4. Minimize episode length for efficiency
    5. Handle multiple agents simultaneously

    Reward function should:
    - Use distance-based rewards for navigation
    - Incorporate depth sensor for obstacle avoidance
    - Penalize unstable flight (high angular velocities)
    - Provide shaping rewards for smooth trajectories
    - Balance multiple objectives appropriately
    """
    
    # LLM configuration
    LLM_CONFIG = {
        "model": "gpt-4o",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.8,
        "max_tokens": 1500,
        "timeout": 120,
        "max_retries": 3
    }
    
    # Initialize and run pipeline
    pipeline = EurekaNavigationPipeline(
        task_description=TASK_DESCRIPTION,
        llm_config=LLM_CONFIG,
        output_dir="./eureka_navigation_output"
    )
    
    # Run optimization
    results = pipeline.run_optimization()
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"Total Samples Evaluated: {results.total_samples}")
    print(f"Successful Samples: {results.successful_samples}")
    print(f"Best Success Rate: {results.best_performance.get('success_rate', 0):.3f}")
    print(f"Execution Time: {results.execution_time:.1f}s")
    print(f"Results saved to: ./eureka_navigation_output/")
    
    return results


if __name__ == "__main__":
    # Run the production pipeline
    results = run_production_pipeline()