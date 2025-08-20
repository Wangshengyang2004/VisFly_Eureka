"""
VisFly-Eureka Production Pipeline

Complete implementation of the Eureka-style optimization pipeline for VisFly
NavigationEnv with BPTT training and comprehensive result processing.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import torch
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import shutil
from collections import defaultdict
import matplotlib.pyplot as plt

# Add project paths
sys.path.insert(0, '/home/simonwsy/VisFly_Eureka')
sys.path.insert(0, '/home/simonwsy/VisFly_Eureka/VisFly')

from eureka_visfly import EurekaVisFly, OptimizationConfig, TrainingResult
from eureka_visfly.parallel_training import ParallelTrainingManager, TrainingJob
from eureka_visfly.gpu_monitor import GPUMonitor, DynamicGPUResourceManager
from eureka_visfly.tensorboard_utils import (
    load_tensorboard_logs, 
    generate_eureka_style_feedback,
    compute_reward_correlation,
    find_tensorboard_logdir,
    extract_success_metric
)

# LLM Configuration (moved from config.py)
# Load API keys from environment or config file
import yaml
import os

def load_api_config():
    """Load API configuration from file or environment"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'api_keys.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            api_config = yaml.safe_load(f)
            return api_config.get('yunwu', {}).get('api_key')
    return os.getenv('YUNWU_API_KEY')

LLM_CONFIG = {
    "model": "gpt-4o",
    "api_key": load_api_config(),  # Loaded from config file or environment
    "base_url": "https://yunwu.ai/v1",
    "temperature": 0.8,
    "max_tokens": 1500,
    "timeout": 120,
    "max_retries": 3
}


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
            error_message=error
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
    """Complete optimization report"""
    task_description: str
    total_iterations: int
    total_samples: int
    successful_samples: int
    best_reward_function: str
    best_performance: Dict[str, float]
    iteration_history: List[Dict[str, Any]]
    baseline_performance: Dict[str, float]
    improvement_metrics: Dict[str, float]
    execution_time: float


class EurekaNavigationPipeline:
    """
    Production pipeline for NavigationEnv reward optimization using Eureka methodology.
    
    This pipeline implements the complete Eureka workflow:
    1. Iterative reward function generation with GPT-4o
    2. Direct injection into VisFly NavigationEnv
    3. BPTT training with comprehensive logging
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
        
        # Pipeline state
        self.iteration_history: List[IterationSummary] = []
        self.all_results: List[RewardFunctionResult] = []
        self.baseline_performance: Optional[Dict[str, float]] = None
        self.baseline_logs: Optional[Dict[str, List[float]]] = None
        self.best_overall: Optional[RewardFunctionResult] = None
        
        # Initialize GPU monitoring and resource management
        self.gpu_monitor = GPUMonitor()
        self.gpu_resource_manager = DynamicGPUResourceManager(self.gpu_monitor)
        
        # Initialize parallel training manager with GPU resource manager
        self.training_manager = ParallelTrainingManager(
            results_dir=str(self.output_dir),
            gpu_resource_manager=self.gpu_resource_manager
        )
        
        # Conversation and artifact tracking
        self.conversation_history = []
        
        # Initialize components
        self._initialize_pipeline()
        
        self.logger.info(f"Initialized EurekaNavigationPipeline")
        self.logger.info(f"Task: {task_description}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Configuration: {self.opt_config['iterations']} iterations × {self.opt_config['samples']} samples")
    
    def _default_env_config(self) -> Dict[str, Any]:
        """Default NavigationEnv configuration"""
        return {
            "num_agent_per_scene": 4,
            "num_scene": 1,
            "visual": True,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "requires_grad": True,  # Enable BPTT
            "max_episode_steps": 256,
            "sensor_kwargs": [{
                "sensor_type": "DEPTH",
                "uuid": "depth",
                "resolution": [64, 64],
            }],
            "target": torch.tensor([[15.0, 0.0, 1.5]]),
        }
    
    def _default_optimization_config(self) -> Dict[str, Any]:
        """Default optimization configuration"""
        return {
            "iterations": 5,
            "samples": 15, 
            "training_steps": 10000,
            "algorithm": "bptt",
            "evaluation_episodes": 20,
            "success_threshold": 0.8,
        }
    
    def setup_logging(self, logging_config: Dict[str, Any]):
        """Setup comprehensive logging"""
        log_level = logging_config.get("level", logging.INFO)
        log_file = self.output_dir / "pipeline.log"
        
        # Create logger
        self.logger = logging.getLogger("EurekaNavigationPipeline")
        self.logger.setLevel(log_level)
        
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
            
            self.logger.info(f"Pipeline components initialized successfully with {self.env_class.__name__}")
            
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
        
        self.logger.info("="*60)
        self.logger.info("Starting VisFly-Eureka Production Pipeline")
        self.logger.info("="*60)
        
        try:
            # Start parallel training manager
            self.training_manager.start()
            
            # Step 1 & 2: Run baseline and reward generation concurrently
            self.logger.info("Step 1 & 2: Running baseline training and reward generation in parallel...")
            self._run_concurrent_baseline_and_optimization()
            
            # Step 3: Analyze results and select best
            self.logger.info("Step 3: Analyzing results...")
            final_report = self._analyze_final_results()
            
            # Step 4: Save outputs
            self.logger.info("Step 4: Saving outputs...")
            self._save_outputs(final_report)
            
            execution_time = time.time() - start_time
            final_report.execution_time = execution_time
            
            self.logger.info(f"Optimization completed in {execution_time:.1f}s")
            self.logger.info("="*60)
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise
        finally:
            # Stop parallel training manager
            self.training_manager.stop()
    
    def _evaluate_baseline(self):
        """Evaluate baseline reward function performance using parallel training"""
        try:
            # Create baseline training job
            baseline_job = self._create_baseline_training_job()
            
            # Submit and run baseline job
            self.logger.info("Running baseline training in parallel...")
            results = self.training_manager.run_parallel_jobs([baseline_job])
            
            baseline_result = results[baseline_job.job_id]
            
            if baseline_result.success:
                # Extract baseline performance metrics
                self.baseline_performance = baseline_result.metrics
                
                # Load baseline tensorboard logs for detailed analysis
                tensorboard_dir = find_tensorboard_logdir(baseline_result.output_files[0] if baseline_result.output_files else baseline_job.output_dir)
                if tensorboard_dir:
                    self.baseline_logs = load_tensorboard_logs(tensorboard_dir)
                    self.logger.info(f"Loaded baseline tensorboard logs: {list(self.baseline_logs.keys())}")
                else:
                    self.logger.warning("No tensorboard logs found for baseline")
                    self.baseline_logs = {}
                
                self.logger.info(f"Baseline performance: {self.baseline_performance}")
            else:
                self.logger.warning(f"Baseline training failed: {baseline_result.error_message}")
                # Set default baseline performance
                self.baseline_performance = {
                    "success_rate": 0.0,
                    "episode_length": 256.0,
                    "training_time": 0.0,
                    "final_reward": 0.0
                }
                self.baseline_logs = {}
            
        except Exception as e:
            self.logger.warning(f"Baseline evaluation failed: {e}")
            self.baseline_performance = {
                "success_rate": 0.0,
                "episode_length": 256.0,
                "training_time": 0.0,
                "final_reward": 0.0
            }
            self.baseline_logs = {}
    
    def _run_concurrent_baseline_and_optimization(self):
        """Run proper evolution algorithm: generate first, then baseline + training in parallel"""
        self.logger.info("🚀 Starting Evolution Algorithm: Generate First, Then Parallel Training...")
        
        # Step 1: Start iterative evolution algorithm (generate rewards first)
        self.logger.info("🧬 Starting iterative evolution optimization...")
        self._run_iterative_evolution()
        
        self.logger.info("✅ Evolution algorithm completed")
        
    def _run_iterative_evolution(self):
        """Run proper iterative evolution algorithm like real Eureka"""
        for iteration in range(self.opt_config["iterations"]):
            self.logger.info(f"🧬 === Evolution Iteration {iteration + 1}/{self.opt_config['iterations']} ===")
            
            iteration_start = time.time()
            
            # Step 1: Generate feedback from previous iterations (evolution context)
            feedback = self._generate_evolution_feedback(iteration)
            self.logger.info(f"📝 Evolution feedback: {feedback[:200]}...")
            
            # Step 2: Generate 15 reward function candidates using evolutionary feedback
            generation_start = time.time()
            self.logger.info(f"🎯 Generating {self.opt_config['samples']} reward candidates...")
            reward_functions = self._generate_reward_functions(iteration, feedback)
            generation_time = time.time() - generation_start
            
            if not reward_functions:
                self.logger.warning(f"No reward functions generated in iteration {iteration + 1}")
                continue
            
            # Step 3: For first iteration, run baseline + 15 candidates in parallel
            # For subsequent iterations, run 15 candidates in parallel
            training_start = time.time()
            if iteration == 0:
                self.logger.info(f"🏃 Training baseline + {len(reward_functions)} candidates in parallel...")
                iteration_results = self._run_baseline_and_rewards_parallel(reward_functions, iteration)
            else:
                self.logger.info(f"🏃 Training {len(reward_functions)} candidates in parallel...")
                iteration_results = self._run_parallel_reward_training(reward_functions, iteration)
            total_training_time = time.time() - training_start
            
            # Step 4: Process results and select best (evolution selection)
            iteration_summary = self._process_iteration_results(
                iteration, iteration_results, generation_time, total_training_time
            )
            self.iteration_history.append(iteration_summary)
            
            # Step 5: Update global best (evolution)
            best_in_iteration = iteration_results[iteration_summary.best_sample_idx] if iteration_results else None
            if (best_in_iteration and 
                best_in_iteration.score() > (self.best_overall.score() if self.best_overall else -10000)):
                self.best_overall = best_in_iteration
                self.logger.info(f"🎉 New global best in iteration {iteration + 1}: score={self.best_overall.score():.4f}, success_rate={self.best_overall.success_rate:.3f}")
            
            # Step 6: Log evolution progress
            self._log_evolution_summary(iteration_summary)
            
            # Step 7: Save iteration artifacts
            self._save_iteration_artifacts(iteration, iteration_summary)
            
            self.logger.info(f"⏱️ Iteration {iteration + 1} completed in {time.time() - iteration_start:.1f}s")
            
        self.logger.info("🏁 Evolution algorithm completed successfully")
        
    def _save_baseline_artifacts(self, baseline_result, baseline_job):
        """Save baseline results to structured baseline folder"""
        try:
            # Create baseline folder parallel to iter_n folders
            baseline_dir = self.output_dir / "baseline"
            baseline_dir.mkdir(exist_ok=True)
            
            # Save baseline result metadata
            result_file = baseline_dir / "result.json"
            baseline_data = {
                "job_id": baseline_job.job_id,
                "success": baseline_result.success,
                "metrics": baseline_result.metrics if baseline_result.success else {},
                "runtime_seconds": baseline_result.runtime_seconds,
                "error_message": baseline_result.error_message,
                "timestamp": time.time()
            }
            
            with open(result_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            
            # Copy any output files to baseline folder
            if baseline_result.output_files:
                for output_file in baseline_result.output_files:
                    if os.path.exists(output_file):
                        dest_file = baseline_dir / os.path.basename(output_file)
                        shutil.copy2(output_file, dest_file)
            
            # Copy job logs
            job_dir = self.output_dir / "jobs" / baseline_job.job_id
            if job_dir.exists():
                for log_file in ["stdout.log", "stderr.log"]:
                    log_path = job_dir / log_file
                    if log_path.exists():
                        dest_log = baseline_dir / log_file
                        shutil.copy2(log_path, dest_log)
            
            self.logger.info(f"Baseline artifacts saved to {baseline_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save baseline artifacts: {e}")
        
    def _run_baseline_and_rewards_parallel(self, reward_functions: List[str], iteration: int) -> List[RewardFunctionResult]:
        """Run baseline + reward functions in parallel for first iteration"""
        training_jobs = []
        
        # Create baseline job
        baseline_job = self._create_baseline_training_job()
        training_jobs.append(baseline_job)
        
        # Create reward training jobs
        reward_jobs = self._create_reward_training_jobs(reward_functions, iteration)
        training_jobs.extend(reward_jobs)
        
        self.logger.info(f"🚀 Running {len(training_jobs)} training jobs (1 baseline + {len(reward_jobs)} rewards)...")
        
        # Run all jobs in parallel
        job_results = self.training_manager.run_parallel_jobs(training_jobs)
        
        # Process baseline result and save to structured folder
        baseline_result = job_results[baseline_job.job_id]
        self._save_baseline_artifacts(baseline_result, baseline_job)
        
        if baseline_result.success:
            self.baseline_performance = baseline_result.metrics
            # Load baseline tensorboard logs for detailed analysis
            tensorboard_dir = find_tensorboard_logdir(baseline_result.output_files[0] if baseline_result.output_files else baseline_job.output_dir)
            if tensorboard_dir:
                self.baseline_logs = load_tensorboard_logs(tensorboard_dir)
                self.logger.info(f"Loaded baseline tensorboard logs: {list(self.baseline_logs.keys())}")
            else:
                self.logger.warning("No tensorboard logs found for baseline")
                self.baseline_logs = {}
            self.logger.info(f"Baseline performance: {self.baseline_performance}")
        else:
            self.logger.warning(f"Baseline training failed: {baseline_result.error_message}")
            self.baseline_performance = {"success_rate": 0.0, "episode_length": 256.0, "training_time": 0.0, "final_reward": 0.0}
            self.baseline_logs = {}
        
        # Process reward function results
        iteration_results = []
        for job in reward_jobs:
            job_result = job_results[job.job_id]
            reward_result = self._convert_job_to_reward_result(job_result, job)
            iteration_results.append(reward_result)
            self.all_results.append(reward_result)
        
        self.logger.info(f"✅ Completed baseline + reward training for iteration {iteration + 1}")
        return iteration_results
        
    def _generate_evolution_feedback(self, iteration: int) -> str:
        """Generate evolution feedback from previous iterations (like real Eureka)"""
        if iteration == 0:
            return "This is the first iteration. Focus on basic navigation to target position [15.0, 0.0, 1.5] while avoiding obstacles using depth sensor data. Design a reward function that balances distance-to-target, obstacle avoidance, and flight stability."
        
        # Use the sophisticated feedback from previous iterations
        return self._generate_iteration_feedback(iteration)
        
    def _log_evolution_summary(self, iteration_summary: IterationSummary):
        """Log evolution iteration summary with detailed metrics"""
        iteration = iteration_summary.iteration
        
        self.logger.info(f"🧬 Evolution Iteration {iteration + 1} Summary:")
        self.logger.info(f"  📊 Best Success Rate: {iteration_summary.best_success_rate:.3f}")
        self.logger.info(f"  ⚡ Execution Rate: {iteration_summary.execution_rate:.3f}")
        self.logger.info(f"  ⏱️ Generation Time: {iteration_summary.generation_time:.1f}s")
        self.logger.info(f"  🏃 Training Time: {iteration_summary.total_training_time:.1f}s")
        
        if iteration_summary.samples:
            best_sample = iteration_summary.samples[iteration_summary.best_sample_idx]
            self.logger.info(f"  🏆 Best Sample: {best_sample.identifier}")
            self.logger.info(f"  📈 Episode Length: {best_sample.episode_length:.1f}")
            self.logger.info(f"  🎯 Final Reward: {best_sample.final_reward:.3f}")
            
        # Evolution progress comparison
        if len(self.iteration_history) > 1:
            prev_best = self.iteration_history[-2].best_success_rate
            current_best = iteration_summary.best_success_rate
            improvement = current_best - prev_best
            
            if improvement > 0.05:
                self.logger.info(f"  📈 Strong improvement: +{improvement:.3f}")
            elif improvement > 0.01:
                self.logger.info(f"  📈 Moderate improvement: +{improvement:.3f}")
            elif improvement < -0.01:
                self.logger.info(f"  📉 Performance decline: {improvement:.3f}")
            else:
                self.logger.info(f"  ➡️ Similar performance: {improvement:+.3f}")
                
        self.logger.info(f"🧬 Iteration {iteration + 1} evolution cycle completed")
    
    def _run_parallel_reward_training(self, reward_functions: List[str], iteration: int) -> List[RewardFunctionResult]:
        """Run training for reward functions in parallel and return results"""
        training_jobs = self._create_reward_training_jobs(reward_functions, iteration)
        
        if training_jobs:
            self.logger.info(f"🚀 Running {len(training_jobs)} training jobs for iteration {iteration + 1}...")
            job_results = self.training_manager.run_parallel_jobs(training_jobs)
            
            # Process results
            iteration_results = []
            for job in training_jobs:
                job_result = job_results[job.job_id]
                reward_result = self._convert_job_to_reward_result(job_result, job)
                iteration_results.append(reward_result)
                self.all_results.append(reward_result)
            
            self.logger.info(f"✅ Completed parallel training for iteration {iteration + 1}")
            return iteration_results
        else:
            self.logger.warning("No training jobs created")
            return []
    
    def _create_baseline_training_job(self) -> TrainingJob:
        """Create a training job for baseline performance evaluation"""
        # Use our VisFly training wrapper that can configure learning steps
        script_path = "/home/simonwsy/VisFly_Eureka/eureka_visfly/visfly_training_wrapper.py"
        
        # Setup environment variables for baseline training
        env_vars = {
            "PYTHONPATH": "/home/simonwsy/VisFly_Eureka:/home/simonwsy/VisFly_Eureka/VisFly"
        }
        
        # Training arguments with configurable learning steps
        args = [
            "--train", "1",  # Training mode
            "--comment", "baseline_evaluation",  # Comment
            "--seed", "42",
            "--learning_steps", str(self.opt_config["training_steps"]),  # Use configured training steps!
            "--num_agents", str(self.env_config.get("num_agent_per_scene", 160)),
            "--max_episode_steps", str(self.env_config.get("max_episode_steps", 256)),
        ]
        
        job = self.training_manager.create_training_job(
            job_type="baseline",
            script_path=script_path,
            arguments=args,
            environment_vars=env_vars,
            memory_requirement_mb=6000,  # Reduced for baseline
            max_runtime_seconds=1800  # 30 minutes max
        )
        
        return job
    
    def _create_reward_training_jobs(self, reward_functions: List[str], iteration: int) -> List[TrainingJob]:
        """Create training jobs for reward function experiments"""
        jobs = []
        
        for i, reward_code in enumerate(reward_functions):
            # Create job identifier
            job_identifier = f"iter_{iteration}_func_{i}"
            
            # Save conversation and reward function artifacts
            conversation_data = {
                "iteration": iteration,
                "function_index": i,
                "reward_code": reward_code,
                "task_description": self.task_description,
                "conversation_history": self.conversation_history[-10:]  # Last 10 exchanges
            }
            
            # Save reward function to temporary file for injection
            reward_func_file = self.artifacts_dir / "reward_functions" / f"{job_identifier}_reward_function.py"
            reward_func_file.parent.mkdir(exist_ok=True, parents=True)
            with open(reward_func_file, 'w') as f:
                f.write(reward_code)

            # Create training job using our wrapper
            job = self.training_manager.create_training_job(
                job_type="reward_experiment",
                script_path="/home/simonwsy/VisFly_Eureka/eureka_visfly/visfly_training_wrapper.py",
                arguments=[
                    "--train", "1",  # Training mode
                    "--comment", job_identifier,
                    "--seed", str(42 + i),  # Different seed per function
                    "--learning_steps", str(self.opt_config["training_steps"]),  # Use configured training steps!
                    "--num_agents", str(self.env_config.get("num_agent_per_scene", 160)),
                    "--max_episode_steps", str(self.env_config.get("max_episode_steps", 256)),
                    "--reward_function_path", str(reward_func_file)  # Path to custom reward function
                ],
                environment_vars={
                    "PYTHONPATH": "/home/simonwsy/VisFly_Eureka:/home/simonwsy/VisFly_Eureka/VisFly",
                },
                reward_function_code=reward_code,
                conversation_log=json.dumps(conversation_data),
                generation_metadata={
                    "iteration": iteration,
                    "function_index": i,
                    "generation_timestamp": time.time()
                },
                memory_requirement_mb=8000,
                max_runtime_seconds=3600  # 1 hour max per experiment
            )
            
            jobs.append(job)
        
        return jobs
    
    def _convert_job_to_reward_result(self, job_result, training_job) -> RewardFunctionResult:
        """Convert a training job result to a reward function result with tensorboard analysis"""
        
        # Extract metadata
        metadata = training_job.generation_metadata or {}
        
        # Load tensorboard logs if available
        tensorboard_logs = {}
        if job_result.success and job_result.output_files:
            tensorboard_dir = find_tensorboard_logdir(job_result.output_files[0] if job_result.output_files else training_job.output_dir)
            if tensorboard_dir:
                tensorboard_logs = load_tensorboard_logs(tensorboard_dir)
                self.logger.debug(f"Loaded tensorboard logs for {training_job.job_id}: {list(tensorboard_logs.keys())}")
        
        if job_result.success:
            # Extract metrics from successful training or tensorboard logs
            metrics = job_result.metrics
            
            # Prefer tensorboard logs for accuracy
            if tensorboard_logs:
                success_rate = extract_success_metric(tensorboard_logs)
                episode_length = np.mean(tensorboard_logs.get('ep_len_mean', [256.0])) if 'ep_len_mean' in tensorboard_logs else 256.0
                final_reward = tensorboard_logs.get('ep_rew_mean', [0.0])[-1] if 'ep_rew_mean' in tensorboard_logs and tensorboard_logs['ep_rew_mean'] else 0.0
                
                # Estimate convergence step (when success rate stabilizes)
                if 'success_rate' in tensorboard_logs or 'rollout/success_rate' in tensorboard_logs:
                    success_curve = tensorboard_logs.get('success_rate', tensorboard_logs.get('rollout/success_rate', []))
                    convergence_step = self._estimate_convergence_step(success_curve)
                else:
                    convergence_step = len(tensorboard_logs.get('ep_rew_mean', [])) // 2
            else:
                # Fallback to job metrics
                success_rate = metrics.get('success_rate', 0.0)
                episode_length = metrics.get('episode_length', 256.0)
                final_reward = metrics.get('final_reward', 0.0)
                convergence_step = metrics.get('convergence_step', 0)
            
            training_time = job_result.runtime_seconds
            
        else:
            # Failed training
            success_rate = 0.0
            episode_length = 256.0
            final_reward = 0.0
            training_time = job_result.runtime_seconds
            convergence_step = 0
        
        return RewardFunctionResult(
            reward_code=training_job.reward_function_code or "",
            identifier=training_job.job_id,
            training_successful=job_result.success,
            success_rate=success_rate,
            episode_length=episode_length,
            training_time=training_time,
            final_reward=final_reward,
            convergence_step=convergence_step,
            tensorboard_logs=tensorboard_logs,
            error_message=job_result.error_message or ""
        )
        
    def _estimate_convergence_step(self, metric_curve: List[float]) -> int:
        """Estimate when the metric converged (stopped improving significantly)"""
        if len(metric_curve) < 10:
            return len(metric_curve) // 2
        
        # Look for point where improvement rate drops below threshold
        window_size = min(10, len(metric_curve) // 4)
        improvement_threshold = 0.01
        
        for i in range(window_size, len(metric_curve) - window_size):
            recent_mean = np.mean(metric_curve[i:i+window_size])
            previous_mean = np.mean(metric_curve[i-window_size:i])
            
            if abs(recent_mean - previous_mean) < improvement_threshold:
                return i
        
        return len(metric_curve) // 2
    
    
    def _generate_reward_functions(self, iteration: int, feedback: str) -> List[str]:
        """Generate reward function candidates for this iteration with conversation tracking"""
        try:
            # Log the conversation exchange
            conversation_entry = {
                "timestamp": time.time(),
                "iteration": iteration,
                "feedback": feedback,
                "request_type": "reward_generation"
            }
            
            # Generate reward functions using the LLM
            reward_functions = self.eureka.generate_reward_candidates(
                samples=self.opt_config["samples"],
                iteration=iteration,
                feedback=feedback
            )
            
            # Track the LLM response
            conversation_entry["response"] = {
                "num_functions": len(reward_functions),
                "functions": reward_functions
            }
            
            self.conversation_history.append(conversation_entry)
            
            # Save conversation to artifacts
            for i, reward_code in enumerate(reward_functions):
                conversation_data = {
                    "iteration": iteration,
                    "function_index": i,
                    "request": feedback,
                    "response": reward_code,
                    "timestamp": time.time()
                }
                self.training_manager.save_conversation(
                    conversation_data, 
                    f"iter_{iteration}_func_{i}_generation"
                )
            
            self.logger.info(f"Generated {len(reward_functions)} reward function candidates")
            return reward_functions
            
        except Exception as e:
            self.logger.error(f"Reward function generation failed: {e}")
            return []
    
    def _generate_iteration_feedback(self, iteration: int) -> str:
        """Generate simple feedback like real Eureka: best vs baseline + truncated tensorboard"""
        if iteration == 0 or not self.iteration_history:
            return "This is the first iteration. Focus on basic navigation to target position [15.0, 0.0, 1.5] while avoiding obstacles using depth sensor data. Design a reward function that balances distance-to-target, obstacle avoidance, and flight stability."
        
        prev_summary = self.iteration_history[-1]
        
        # Get best sample from previous iteration
        if not prev_summary.samples or prev_summary.best_sample_idx >= len(prev_summary.samples):
            return "Previous iteration had no valid samples. Try simpler reward designs with proper tensor operations."
        
        best_sample = prev_summary.samples[prev_summary.best_sample_idx]
        
        # Simple feedback like real Eureka
        feedback_parts = []
        
        # 1. Tensorboard feedback from best sample (truncated to ~10 points)
        if hasattr(best_sample, 'tensorboard_logs') and best_sample.tensorboard_logs:
            eureka_feedback = generate_eureka_style_feedback(
                best_sample.tensorboard_logs, 
                self.baseline_logs
            )
            feedback_parts.append(eureka_feedback)
        
        # 2. Simple performance comparison with baseline
        if self.baseline_performance:
            baseline_success = self.baseline_performance.get('success_rate', 0.0)
            if best_sample.success_rate > baseline_success:
                feedback_parts.append(f"Performance improved over baseline: {best_sample.success_rate:.3f} vs {baseline_success:.3f}")
            else:
                feedback_parts.append(f"Performance below baseline: {best_sample.success_rate:.3f} vs {baseline_success:.3f}")
        
        return "\n".join(feedback_parts)
    
    def _analyze_tensorboard_logs(self, iteration: int) -> str:
        """Analyze tensorboard logs from previous iteration for additional feedback"""
        if iteration == 0 or not self.iteration_history:
            return ""
        
        try:
            # Look for tensorboard logs in the previous iteration's best result
            prev_summary = self.iteration_history[-1]
            if prev_summary.samples and prev_summary.best_sample_idx < len(prev_summary.samples):
                best_sample = prev_summary.samples[prev_summary.best_sample_idx]
                
                # Parse key training metrics (placeholder - would parse actual tensorboard logs)
                analysis_parts = []
                
                # Training stability analysis
                if best_sample.convergence_step > 0:
                    if best_sample.convergence_step < 1000:
                        analysis_parts.append("Quick convergence suggests well-designed reward.")
                    elif best_sample.convergence_step > 5000:
                        analysis_parts.append("Slow convergence indicates potential reward scaling issues.")
                
                # Reward magnitude analysis
                if abs(best_sample.final_reward) > 1000:
                    analysis_parts.append("High reward magnitude may cause training instability.")
                elif abs(best_sample.final_reward) < 0.1:
                    analysis_parts.append("Low reward magnitude may provide insufficient learning signal.")
                
                return " ".join(analysis_parts)
                
        except Exception as e:
            self.logger.warning(f"Tensorboard analysis failed: {e}")
        
        return ""
    
    def _save_iteration_artifacts(self, iteration: int, summary: IterationSummary):
        """Save artifacts for this iteration with structured folder layout"""
        try:
            # Create iteration folder: iter_0, iter_1, etc.
            iter_dir = self.output_dir / f"iter_{iteration}"
            iter_dir.mkdir(exist_ok=True)
            
            # Save iteration summary
            summary_file = iter_dir / "summary.json"
            summary_data = {
                "iteration": iteration,
                "best_sample_idx": summary.best_sample_idx,
                "best_success_rate": summary.best_success_rate,
                "execution_rate": summary.execution_rate,
                "generation_time": summary.generation_time,
                "total_training_time": summary.total_training_time,
                "timestamp": time.time()
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            # Save each sample in structured variant folders
            for sample_idx, result in enumerate(summary.samples):
                variant_dir = iter_dir / f"variant_{sample_idx + 1:02d}"
                variant_dir.mkdir(exist_ok=True)
                
                # Save reward function code
                reward_file = variant_dir / "reward_function.py"
                with open(reward_file, 'w') as f:
                    f.write(result.reward_code)
                
                # Save result metadata (convert numpy types to native Python types)
                result_file = variant_dir / "result.json"
                result_dict = asdict(result)
                # Convert any numpy types to native Python types for JSON serialization
                for key, value in result_dict.items():
                    if hasattr(value, 'item'):  # numpy scalar
                        result_dict[key] = value.item()
                    elif hasattr(value, 'tolist'):  # numpy array
                        result_dict[key] = value.tolist()
                with open(result_file, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                
                # Copy any training logs/artifacts if available
                if hasattr(result, 'tensorboard_logs') and result.tensorboard_logs:
                    tensorboard_file = variant_dir / "tensorboard_logs.json"
                    with open(tensorboard_file, 'w') as f:
                        json.dump(result.tensorboard_logs, f, indent=2)
            
            # Save best sample details in iteration root
            if summary.samples and summary.best_sample_idx < len(summary.samples):
                best_sample = summary.samples[summary.best_sample_idx]
                best_file = iter_dir / "best_variant.json"
                best_data = {
                    "best_variant_index": summary.best_sample_idx + 1,
                    "best_variant_folder": f"variant_{summary.best_sample_idx + 1:02d}",
                    "performance": asdict(best_sample)
                }
                with open(best_file, 'w') as f:
                    json.dump(best_data, f, indent=2)
            
            self.logger.info(f"Iteration {iteration} artifacts saved to {iter_dir} with {len(summary.samples)} variants")
            
        except Exception as e:
            self.logger.warning(f"Failed to save iteration artifacts: {e}")
    
    def _evaluate_reward_function(self, reward_code: str, identifier: str) -> RewardFunctionResult:
        """Evaluate a single reward function"""
        try:
            # Evaluate using eureka controller
            result = self.eureka.evaluate_reward_function(reward_code, identifier)
            
            if result:
                return RewardFunctionResult(
                    reward_code=reward_code,
                    identifier=identifier,
                    training_successful=True,
                    success_rate=result.success_rate,
                    episode_length=result.episode_length,
                    training_time=result.training_time,
                    final_reward=result.final_reward,
                    convergence_step=result.convergence_step,
                    tensorboard_logs={}  # Would be populated with actual tensorboard data
                )
            else:
                return RewardFunctionResult.failed(reward_code, identifier, "Training failed")
                
        except Exception as e:
            self.logger.error(f"Evaluation failed for {identifier}: {e}")
            return RewardFunctionResult.failed(reward_code, identifier, str(e))
    
    def _process_iteration_results(
        self, 
        iteration: int, 
        results: List[RewardFunctionResult],
        generation_time: float,
        training_time: float
    ) -> IterationSummary:
        """Process results from one iteration"""
        
        # Calculate metrics
        successes = [r.success_rate if r.training_successful else -1.0 for r in results]
        successful_results = [r for r in results if r.training_successful]
        
        # Find best sample
        best_idx = np.argmax(successes) if successes else 0
        best_success_rate = successes[best_idx] if successes else -1.0
        
        # Calculate execution rate
        execution_rate = len(successful_results) / len(results) if results else 0.0
        
        # Calculate correlation (placeholder - would use actual baseline comparison)
        best_correlation = 0.5 if successful_results else -1.0
        
        return IterationSummary(
            iteration=iteration,
            samples=results,
            best_sample_idx=best_idx,
            best_success_rate=best_success_rate,
            best_correlation=best_correlation,
            execution_rate=execution_rate,
            generation_time=generation_time,
            total_training_time=training_time
        )
    
    def _log_iteration_summary(self, summary: IterationSummary):
        """Log comprehensive iteration summary"""
        self.logger.info(f"Iteration {summary.iteration + 1} Summary:")
        self.logger.info(f"  Best Success Rate: {summary.best_success_rate:.3f}")
        self.logger.info(f"  Execution Rate: {summary.execution_rate:.1%} ({len([r for r in summary.samples if r.training_successful])}/{len(summary.samples)})")
        self.logger.info(f"  Generation Time: {summary.generation_time:.1f}s")
        self.logger.info(f"  Training Time: {summary.total_training_time:.1f}s")
        
        if summary.samples and summary.best_sample_idx < len(summary.samples):
            best_sample = summary.samples[summary.best_sample_idx]
            self.logger.info(f"  Best Sample: {best_sample.identifier}")
            self.logger.info(f"    Episode Length: {best_sample.episode_length:.1f}")
            self.logger.info(f"    Final Reward: {best_sample.final_reward:.3f}")
    
    def _analyze_final_results(self) -> OptimizationReport:
        """Analyze final results and create comprehensive report"""
        
        # Sort all results by performance
        successful_results = [r for r in self.all_results if r.training_successful]
        successful_results.sort(key=lambda x: x.score(), reverse=True)
        
        # Calculate improvement metrics
        improvement_metrics = {}
        if self.baseline_performance and successful_results:
            best_result = successful_results[0]
            improvement_metrics = {
                "success_rate_improvement": best_result.success_rate - self.baseline_performance["success_rate"],
                "episode_length_improvement": self.baseline_performance["episode_length"] - best_result.episode_length,
                "relative_improvement": (best_result.success_rate - self.baseline_performance["success_rate"]) / max(self.baseline_performance["success_rate"], 0.01)
            }
        
        # Create iteration history summary
        iteration_history = []
        for summary in self.iteration_history:
            iteration_history.append({
                "iteration": summary.iteration + 1,
                "best_success_rate": summary.best_success_rate,
                "execution_rate": summary.execution_rate,
                "generation_time": summary.generation_time,
                "training_time": summary.total_training_time
            })
        
        # Best performance
        best_performance = {}
        if successful_results:
            best = successful_results[0]
            best_performance = {
                "success_rate": best.success_rate,
                "episode_length": best.episode_length,
                "training_time": best.training_time,
                "final_reward": best.final_reward,
                "score": best.score()
            }
        
        return OptimizationReport(
            task_description=self.task_description,
            total_iterations=self.opt_config["iterations"],
            total_samples=len(self.all_results),
            successful_samples=len(successful_results),
            best_reward_function=successful_results[0].reward_code if successful_results else "",
            best_performance=best_performance,
            iteration_history=iteration_history,
            baseline_performance=self.baseline_performance or {},
            improvement_metrics=improvement_metrics,
            execution_time=0.0  # Will be set by caller
        )
    
    def _save_outputs(self, report: OptimizationReport):
        """Save all outputs and results in structured format"""
        
        # Save optimization report at root level
        report_file = self.output_dir / "optimization_report.json"
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        
        # Save best reward function at root level
        if report.best_reward_function:
            best_function_file = self.output_dir / "best_reward_function.py"
            with open(best_function_file, 'w') as f:
                f.write(report.best_reward_function)
        
        # Create summary of folder structure
        structure_summary = {
            "folder_structure": {
                "baseline/": "Baseline training results and logs",
                "iter_0/": "First iteration with 15 variants (variant_01/ to variant_15/)",
                "iter_1/": "Second iteration with 15 variants",
                "iter_n/": "Additional iterations as configured",
                "optimization_report.json": "Overall optimization summary",
                "best_reward_function.py": "Best performing reward function code"
            },
            "variant_structure": {
                "variant_XX/": {
                    "reward_function.py": "Generated reward function code",
                    "result.json": "Training results and performance metrics",
                    "tensorboard_logs.json": "Training curves and detailed metrics (if available)"
                }
            },
            "iteration_structure": {
                "summary.json": "Iteration summary with best variant info",
                "best_variant.json": "Details about the best performing variant in this iteration"
            }
        }
        
        structure_file = self.output_dir / "folder_structure.json"
        with open(structure_file, 'w') as f:
            json.dump(structure_summary, f, indent=2)
        
        # Create visualization (basic)
        self._create_visualizations(report)
        
        self.logger.info(f"Structured outputs saved to {self.output_dir}")
        self.logger.info(f"Folder structure: baseline/ + iter_0/ to iter_{len(self.iteration_history)-1}/ with variant_01/ to variant_15/ subfolders")
    
    def _create_visualizations(self, report: OptimizationReport):
        """Create basic visualization of results"""
        if not report.iteration_history:
            return
            
        try:
            # Success rate progression
            iterations = [h["iteration"] for h in report.iteration_history]
            success_rates = [h["best_success_rate"] for h in report.iteration_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, success_rates, 'b-o', linewidth=2, markersize=6)
            plt.xlabel('Iteration')
            plt.ylabel('Best Success Rate')
            plt.title('Success Rate Progression')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / "success_rate_progression.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Execution rate progression  
            execution_rates = [h["execution_rate"] for h in report.iteration_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, execution_rates, 'g-o', linewidth=2, markersize=6)
            plt.xlabel('Iteration')
            plt.ylabel('Execution Rate')
            plt.title('Execution Rate Progression')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.savefig(self.output_dir / "execution_rate_progression.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Visualizations created successfully")
            
        except Exception as e:
            self.logger.warning(f"Visualization creation failed: {e}")


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
    
    # Environment configuration
    env_config = {
        "num_agent_per_scene": 4,
        "num_scene": 1,
        "visual": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "requires_grad": True,  # Enable BPTT
        "max_episode_steps": 256,
        "sensor_kwargs": [{
            "sensor_type": "DEPTH",
            "uuid": "depth",
            "resolution": [64, 64],
        }],
        "target": torch.tensor([[15.0, 0.0, 1.5]]),
    }
    
    # Optimization configuration
    optimization_config = {
        "iterations": 5,
        "samples": 15,
        "training_steps": 10000,
        "algorithm": "bptt",
        "evaluation_episodes": 20,
        "success_threshold": 0.8,
    }
    
    # Initialize and run pipeline
    pipeline = EurekaNavigationPipeline(
        task_description=TASK_DESCRIPTION,
        llm_config=LLM_CONFIG,
        env_config=env_config,
        optimization_config=optimization_config,
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