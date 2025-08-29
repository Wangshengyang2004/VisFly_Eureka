"""
Parallel Training Manager for VisFly-Eureka

This module manages parallel execution of training jobs across multiple GPUs,
including baseline evaluation and reward function experiments.
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
import queue
import uuid
from datetime import datetime

from ..utils.gpu_monitor import GPUMonitor, DynamicGPUResourceManager


@dataclass
class TrainingJob:
    """Definition of a training job"""

    job_id: str
    job_type: str  # 'baseline', 'reward_experiment'
    script_path: str
    environment_vars: Dict[str, str]
    arguments: List[str]
    output_dir: str
    gpu_id: Optional[int] = None
    memory_requirement_mb: int = 8000
    max_runtime_seconds: int = 3600
    reward_function_code: Optional[str] = None
    conversation_log: Optional[str] = None
    generation_metadata: Optional[Dict] = None


@dataclass
class JobResult:
    """Result of a completed training job"""

    job_id: str
    success: bool
    exit_code: int
    runtime_seconds: float
    stdout: str
    stderr: str
    metrics: Dict[str, Any]
    output_files: List[str]
    error_message: Optional[str] = None


class ParallelTrainingManager:
    """Manages parallel execution of training jobs with GPU monitoring"""

    def __init__(
        self,
        results_dir: str = "results",
        max_workers: int = 4,
        gpu_resource_manager: Optional[DynamicGPUResourceManager] = None,
    ):
        """
        Initialize parallel training manager.

        Args:
            results_dir: Directory to save all results and artifacts
            max_workers: Maximum number of concurrent jobs
            gpu_resource_manager: Optional external GPU resource manager
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize GPU monitoring and management
        if gpu_resource_manager is not None:
            self.gpu_resource_manager = gpu_resource_manager
            self.gpu_monitor = gpu_resource_manager.monitor
        else:
            self.gpu_monitor = GPUMonitor()
            self.gpu_resource_manager = DynamicGPUResourceManager(self.gpu_monitor)

        # Note: Now using gpu_resource_manager for all GPU operations

        # Job management
        self.active_jobs = {}  # job_id -> subprocess.Popen
        self.completed_jobs = {}  # job_id -> JobResult
        self.job_queue = queue.Queue()
        self.results_queue = queue.Queue()

        # Use results directory directly (no session subdirectory)
        self.session_dir = self.results_dir
        self.session_dir.mkdir(exist_ok=True)

        # Artifacts storage
        self.artifacts_dir = self.session_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)
        self.conversations_dir = self.artifacts_dir / "conversations"
        self.conversations_dir.mkdir(exist_ok=True)
        self.reward_functions_dir = self.artifacts_dir / "reward_functions"
        self.reward_functions_dir.mkdir(exist_ok=True)
        self.injected_code_dir = self.artifacts_dir / "injected_code"
        self.injected_code_dir.mkdir(exist_ok=True)

        # ParallelTrainingManager initialized

    def start(self):
        """Start the parallel training system"""
        self.gpu_monitor.start_monitoring()
        self.gpu_monitor.log_gpu_status()

        # Calculate GPU capabilities
        estimates = self.gpu_monitor.estimate_max_parallel_jobs()
        total_capacity = sum(estimates.values())

    def stop(self):
        """Stop the parallel training system"""
        # Import configuration
        try:
            from config import GPU_CONFIG

            termination_timeout = GPU_CONFIG["process_termination_timeout"]
        except ImportError:
            termination_timeout = 10

        # Stop all active jobs
        for job_id, process in list(self.active_jobs.items()):
            self.logger.info(f"Terminating job {job_id}")
            process.terminate()
            try:
                process.wait(timeout=termination_timeout)
            except subprocess.TimeoutExpired:
                process.kill()

        # Release all allocated GPU resources
        for job_id in list(self.gpu_resource_manager.allocated_gpus.keys()):
            self.gpu_resource_manager.release_gpu(job_id)
            self.logger.debug(f"Released GPU resources for job {job_id}")

        # Clear any remaining GPU memory
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

        self.gpu_monitor.stop_monitoring()
        self.logger.info("Stopped ParallelTrainingManager")

    def save_conversation(self, conversation_data: Dict[str, Any], job_id: str) -> str:
        """
        Save LLM conversation data to artifacts.

        Args:
            conversation_data: Dictionary with conversation history
            job_id: Associated job ID

        Returns:
            Path to saved conversation file
        """
        conversation_file = self.conversations_dir / f"{job_id}_conversation.json"

        with open(conversation_file, "w") as f:
            json.dump(conversation_data, f, indent=2, default=str)

        self.logger.info(f"Saved conversation for job {job_id} to {conversation_file}")
        return str(conversation_file)

    def save_reward_function(
        self, reward_code: str, job_id: str, metadata: Optional[Dict] = None
    ) -> str:
        """
        Save generated reward function code.

        Args:
            reward_code: Generated reward function source code
            job_id: Associated job ID
            metadata: Optional metadata about generation

        Returns:
            Path to saved reward function file
        """
        reward_file = self.reward_functions_dir / f"{job_id}_reward_function.py"

        # Save the raw reward function code
        with open(reward_file, "w") as f:
            f.write(reward_code)

        # Save metadata if provided
        if metadata:
            metadata_file = self.reward_functions_dir / f"{job_id}_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        self.logger.info(f"Saved reward function for job {job_id} to {reward_file}")
        return str(reward_file)

    def save_injected_code(
        self, injected_code: str, job_id: str, injection_result: bool
    ) -> str:
        """
        Save code after injection into environment.

        Args:
            injected_code: Code that was injected
            job_id: Associated job ID
            injection_result: Whether injection was successful

        Returns:
            Path to saved injected code file
        """
        injected_file = self.injected_code_dir / f"{job_id}_injected.py"

        # Create injection report
        injection_report = {
            "job_id": job_id,
            "injection_successful": injection_result,
            "timestamp": datetime.now().isoformat(),
            "injected_code": injected_code,
        }

        with open(injected_file.with_suffix(".json"), "w") as f:
            json.dump(injection_report, f, indent=2)

        # Save raw injected code
        with open(injected_file, "w") as f:
            f.write(injected_code)

        self.logger.info(f"Saved injected code for job {job_id} to {injected_file}")
        return str(injected_file)

    def create_training_job(
        self,
        job_type: str,
        script_path: str,
        arguments: List[str] = None,
        environment_vars: Dict[str, str] = None,
        reward_function_code: str = None,
        conversation_log: str = None,
        generation_metadata: Dict = None,
        memory_requirement_mb: int = None,
        max_runtime_seconds: int = None,
    ) -> TrainingJob:
        """
        Create a new training job.

        Args:
            job_type: Type of job ('baseline', 'reward_experiment')
            script_path: Path to training script
            arguments: Command line arguments for the script
            environment_vars: Environment variables to set
            reward_function_code: Generated reward function code (if applicable)
            conversation_log: LLM conversation log
            generation_metadata: Metadata about reward generation
            memory_requirement_mb: GPU memory requirement
            max_runtime_seconds: Maximum runtime before timeout

        Returns:
            TrainingJob instance
        """
        # Default values (removed config.py dependency)
        if memory_requirement_mb is None:
            memory_requirement_mb = 8000
        if max_runtime_seconds is None:
            max_runtime_seconds = 3600
        job_id = f"{job_type}_{uuid.uuid4().hex[:8]}"

        # Create job-specific output directory
        job_output_dir = self.session_dir / "jobs" / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)

        # Set default environment variables
        env_vars = {
            "CUDA_VISIBLE_DEVICES": "",  # Will be set when device is allocated
            "PYTHONPATH": str(Path.cwd()),
            "VISFLY_DEVICE": "auto",  # Will be overridden during execution
        }

        if environment_vars:
            env_vars.update(environment_vars)

        job = TrainingJob(
            job_id=job_id,
            job_type=job_type,
            script_path=script_path,
            environment_vars=env_vars,
            arguments=arguments or [],
            output_dir=str(job_output_dir),
            memory_requirement_mb=memory_requirement_mb,
            max_runtime_seconds=max_runtime_seconds,
            reward_function_code=reward_function_code,
            conversation_log=conversation_log,
            generation_metadata=generation_metadata,
        )

        # Save artifacts if provided
        if conversation_log:
            self.save_conversation({"log": conversation_log}, job_id)

        if reward_function_code:
            self.save_reward_function(reward_function_code, job_id, generation_metadata)

        self.logger.info(f"Created {job_type} job {job_id}")
        return job

    def submit_job(self, job: TrainingJob) -> bool:
        """
        Submit a job for execution using dynamic GPU resource management.

        Args:
            job: TrainingJob to execute

        Returns:
            True if job was submitted successfully
        """
        # Try to allocate device using dynamic GPU resource manager
        device_id = self.gpu_resource_manager.allocate_gpu_with_queue(
            job.job_id, job.memory_requirement_mb, allow_cpu_fallback=True
        )

        if device_id is None:
            self.logger.warning(
                f"Cannot submit job {job.job_id} - no device available, added to queue"
            )
            return False

        job.gpu_id = device_id

        # Set CUDA_VISIBLE_DEVICES and VisFly device appropriately
        if device_id == "cpu":
            job.environment_vars["CUDA_VISIBLE_DEVICES"] = (
                ""  # Hide all GPUs for CPU-only execution
            )
            job.environment_vars["VISFLY_DEVICE"] = "cpu"
        else:
            # Extract GPU ID from "cuda:X" format
            gpu_num = device_id.split(":")[-1] if ":" in device_id else device_id
            job.environment_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
            job.environment_vars["VISFLY_DEVICE"] = "cuda"

        # Submit to job queue
        self.job_queue.put(job)
        device_name = "CPU" if device_id == "cpu" else f"GPU {device_id}"
        self.logger.info(f"Submitted job {job.job_id} to {device_name}")
        return True

    def _execute_job(self, job: TrainingJob) -> JobResult:
        """
        Execute a single training job.

        Args:
            job: TrainingJob to execute

        Returns:
            JobResult with execution results
        """
        start_time = time.time()

        # Validate job parameters
        if not os.path.exists(job.script_path):
            self.logger.error(f"Script not found: {job.script_path}")
            return JobResult(
                job_id=job.job_id,
                success=False,
                exit_code=-5,
                runtime_seconds=0,
                stdout="",
                stderr=f"Script not found: {job.script_path}",
                metrics={},
                output_files=[],
                error_message=f"Script not found: {job.script_path}",
            )

        # Ensure output directory exists
        output_dir = Path(job.output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.logger.error(f"Cannot create output directory {output_dir}: {e}")
            return JobResult(
                job_id=job.job_id,
                success=False,
                exit_code=-6,
                runtime_seconds=0,
                stdout="",
                stderr=f"Cannot create output directory: {e}",
                metrics={},
                output_files=[],
                error_message=f"Cannot create output directory: {e}",
            )

        try:
            # Prepare command
            cmd = [sys.executable, job.script_path] + job.arguments

            # Setup environment
            env = os.environ.copy()
            env.update(job.environment_vars)

            # Create output files
            stdout_file = Path(job.output_dir) / "stdout.log"
            stderr_file = Path(job.output_dir) / "stderr.log"

            device_name = "CPU" if job.gpu_id == "cpu" else f"GPU {job.gpu_id}"
            self.logger.info(f"Starting job {job.job_id} on {device_name}")
            self.logger.debug(f"Command: {' '.join(cmd)}")

            # Execute job with memory monitoring
            with open(stdout_file, "w") as stdout, open(stderr_file, "w") as stderr:
                process = subprocess.Popen(
                    cmd,
                    stdout=stdout,
                    stderr=stderr,
                    env=env,
                    cwd=str(Path(job.script_path).parent),
                )

                self.active_jobs[job.job_id] = process

                # Start memory monitoring thread
                if job.gpu_id != "cpu":
                    monitor_thread = threading.Thread(
                        target=self._monitor_job_memory,
                        args=(job.job_id, process),
                        daemon=True,
                    )
                    monitor_thread.start()

                try:
                    exit_code = process.wait(timeout=job.max_runtime_seconds)
                except subprocess.TimeoutExpired:
                    self.logger.warning(
                        f"Job {job.job_id} timed out after {job.max_runtime_seconds}s"
                    )
                    process.kill()
                    exit_code = -1
                finally:
                    if job.job_id in self.active_jobs:
                        del self.active_jobs[job.job_id]

                    # Explicit cleanup - terminate any child processes and clear GPU memory
                    try:
                        import psutil

                        parent = psutil.Process(process.pid)
                        for child in parent.children(recursive=True):
                            child.terminate()
                    except (psutil.NoSuchProcess, ImportError):
                        # psutil not available or process already dead
                        pass

                    # Clear GPU memory cache if using CUDA
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    except ImportError:
                        pass

            # Read outputs
            stdout_content = stdout_file.read_text() if stdout_file.exists() else ""
            stderr_content = stderr_file.read_text() if stderr_file.exists() else ""

            # Check for CUDA OOM errors
            if (
                "CUDA out of memory" in stderr_content
                or "OutOfMemoryError" in stderr_content
            ):
                should_retry = self.gpu_resource_manager.handle_cuda_oom_error(
                    job.job_id
                )
                if should_retry:
                    self.logger.info(f"Job {job.job_id} had CUDA OOM, queued for retry")
                    # Mark as special retry failure code
                    exit_code = -100

            # Extract metrics from output
            metrics = self._extract_metrics(stdout_content, stderr_content)

            # Find output files
            output_files = list(Path(job.output_dir).glob("**/*"))
            output_files = [str(f) for f in output_files if f.is_file()]

            runtime = time.time() - start_time
            success = exit_code == 0

            result = JobResult(
                job_id=job.job_id,
                success=success,
                exit_code=exit_code,
                runtime_seconds=runtime,
                stdout=stdout_content,
                stderr=stderr_content,
                metrics=metrics,
                output_files=output_files,
                error_message=None if success else f"Exit code {exit_code}",
            )

            self.logger.info(
                f"Job {job.job_id} completed: {'SUCCESS' if success else 'FAILED'} "
                f"in {runtime:.1f}s"
            )

            return result

        except Exception as e:
            runtime = time.time() - start_time
            error_msg = f"Job execution failed: {e}"
            self.logger.error(f"Job {job.job_id} error: {error_msg}")

            return JobResult(
                job_id=job.job_id,
                success=False,
                exit_code=-2,
                runtime_seconds=runtime,
                stdout="",
                stderr=str(e),
                metrics={},
                output_files=[],
                error_message=error_msg,
            )

        finally:
            # Release GPU
            self.gpu_resource_manager.release_gpu(job.job_id)

    def _extract_metrics(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Extract training metrics from output logs"""
        metrics = {}

        # Look for common metrics patterns in stdout
        lines = stdout.split("\n")
        for line in lines:
            # Success rate pattern
            if "success_rate" in line.lower():
                try:
                    # Extract number after 'success_rate'
                    parts = line.split("success_rate")
                    if len(parts) > 1:
                        import re

                        numbers = re.findall(r"[\d.]+", parts[1])
                        if numbers:
                            metrics["success_rate"] = float(numbers[0])
                except:
                    pass

            # Episode length pattern
            if "episode_length" in line.lower():
                try:
                    import re

                    numbers = re.findall(r"[\d.]+", line)
                    if numbers:
                        metrics["episode_length"] = float(numbers[-1])
                except:
                    pass

            # Loss patterns
            if "loss" in line.lower():
                try:
                    import re

                    numbers = re.findall(r"[\d.]+", line)
                    if numbers:
                        metrics["final_loss"] = float(numbers[-1])
                except:
                    pass

        return metrics

    def run_parallel_jobs(
        self, jobs: List[TrainingJob], max_concurrent: Optional[int] = None
    ) -> Dict[str, JobResult]:
        """
        Run multiple jobs in parallel with GPU management.

        Args:
            jobs: List of TrainingJob instances to execute
            max_concurrent: Maximum concurrent jobs (defaults to max_workers)

        Returns:
            Dictionary mapping job_id to JobResult
        """
        if max_concurrent is None:
            max_concurrent = self.max_workers

        results = {}

        # Submit all jobs
        submitted_jobs = []
        for job in jobs:
            if self.submit_job(job):
                submitted_jobs.append(job)
            else:
                # Create a failed result for jobs that couldn't be submitted
                results[job.job_id] = JobResult(
                    job_id=job.job_id,
                    success=False,
                    exit_code=-3,
                    runtime_seconds=0,
                    stdout="",
                    stderr="",
                    metrics={},
                    output_files=[],
                    error_message="Could not allocate GPU",
                )

        self.logger.info(
            f"Running {len(submitted_jobs)} jobs in parallel (max {max_concurrent} concurrent)"
        )

        # Execute jobs with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all jobs to executor
            future_to_job = {}

            for job in submitted_jobs:
                if not self.job_queue.empty():
                    job = self.job_queue.get()
                    future = executor.submit(self._execute_job, job)
                    future_to_job[future] = job

            # Collect results as they complete
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    results[result.job_id] = result
                    self.completed_jobs[result.job_id] = result
                except Exception as e:
                    self.logger.error(f"Job {job.job_id} failed with exception: {e}")
                    results[job.job_id] = JobResult(
                        job_id=job.job_id,
                        success=False,
                        exit_code=-4,
                        runtime_seconds=0,
                        stdout="",
                        stderr=str(e),
                        metrics={},
                        output_files=[],
                        error_message=f"Execution exception: {e}",
                    )
                finally:
                    # Always clean up GPU resources when job completes
                    self.gpu_resource_manager.release_gpu(job.job_id)
                    self.logger.debug(f"Released GPU resources for job {job.job_id}")

        # Save session summary
        self._save_session_summary(results)

        return results

    def _save_session_summary(self, results: Dict[str, JobResult]):
        """Save summary of the training session"""
        summary = {
            "run_directory": str(self.session_dir),
            "timestamp": datetime.now().isoformat(),
            "total_jobs": len(results),
            "successful_jobs": sum(1 for r in results.values() if r.success),
            "failed_jobs": sum(1 for r in results.values() if not r.success),
            "total_runtime": sum(r.runtime_seconds for r in results.values()),
            "jobs": {job_id: asdict(result) for job_id, result in results.items()},
        }

        summary_file = self.session_dir / "run_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Saved session summary to {summary_file}")
        self.logger.info(
            f"Session results: {summary['successful_jobs']}/{summary['total_jobs']} jobs succeeded"
        )

    def _monitor_job_memory(self, job_id: str, process: subprocess.Popen):
        """Monitor memory usage for a job and update the resource manager"""
        training_step = 0
        last_stdout_lines = 0

        while process.poll() is None:  # While process is running
            try:
                # Check for training progress in stdout
                stdout_file = (
                    Path(
                        self.active_jobs[job_id] if job_id in self.active_jobs else ""
                    ).parent
                    / "stdout.log"
                )
                if stdout_file.exists():
                    with open(stdout_file, "r") as f:
                        lines = f.readlines()
                        new_lines = lines[last_stdout_lines:]
                        last_stdout_lines = len(lines)

                        # Look for training step indicators
                        for line in new_lines:
                            if "Training" in line and "/" in line:
                                # Try to extract step number like "Training 1000/110000"
                                try:
                                    parts = line.split()
                                    for part in parts:
                                        if "/" in part:
                                            step_str = part.split("/")[0]
                                            step_num = int(step_str)
                                            if step_num > training_step:
                                                training_step = step_num
                                                self.gpu_resource_manager.update_job_memory_usage(
                                                    job_id, training_step
                                                )
                                                break
                                except (ValueError, IndexError):
                                    continue

                # Sleep before next check
                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.debug(f"Memory monitoring error for job {job_id}: {e}")
                break


def main():
    """Test parallel training manager"""
    logging.basicConfig(level=logging.INFO)

    manager = ParallelTrainingManager()
    manager.start()

    try:
        # Create test jobs
        jobs = []
        for i in range(3):
            job = manager.create_training_job(
                job_type="test",
                script_path="/usr/bin/sleep",
                arguments=["5"],  # Sleep for 5 seconds
                environment_vars={"TEST_VAR": f"test_{i}"},
            )
            jobs.append(job)

        # Run jobs in parallel
        results = manager.run_parallel_jobs(jobs)

        # Print results
        for job_id, result in results.items():
            print(f"Job {job_id}: {'SUCCESS' if result.success else 'FAILED'}")

    finally:
        manager.stop()


if __name__ == "__main__":
    main()
