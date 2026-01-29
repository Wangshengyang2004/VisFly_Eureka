"""
Subprocess-based reward function evaluation for VisFly-Eureka.

Provides isolated, parallel execution of reward function training
using VisFly environments and algorithms.
"""

import os
import sys
import json
import logging
import subprocess
import signal
import tempfile
import threading
import time
import yaml
import shutil
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch

from .models import RewardFunctionResult
from ..utils.gpu_monitor import EnhancedGPUMonitor
from ..utils.gpu_monitor_dynamic import DynamicGPUResourceManager
from ..utils.tensorboard_utils import find_tensorboard_logdir


class SubprocessRewardEvaluator:
    """
    Evaluates reward functions in isolated subprocesses.
    
    This class provides safe, parallel execution of reward function training
    by running each evaluation in a separate subprocess. This prevents
    memory leaks, crashes, and enables true parallel execution.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, gpu_monitor: Optional[EnhancedGPUMonitor] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="eureka_eval_"))
        self.temp_dir.mkdir(exist_ok=True)

        # Resolve evaluation worker script path (standalone file)
        self.eval_script = (Path(__file__).parent / "evaluation_worker.py").resolve()
        if not self.eval_script.exists():
            raise FileNotFoundError(f"Evaluation worker script not found at {self.eval_script}")
        
        # Use provided GPU monitor or create new one
        if gpu_monitor:
            self.gpu_monitor = gpu_monitor
            self.owns_gpu_monitor = False
        else:
            self.gpu_monitor = EnhancedGPUMonitor(utilization_threshold=80)
            self.owns_gpu_monitor = True
            
        self.gpu_allocator = DynamicGPUResourceManager(self.gpu_monitor)

    def _read_tail(self, p: Path, max_bytes: int = 8192) -> str:
        """Return last part of a text file decoded as utf-8 for diagnostics."""
        try:
            if not p.exists():
                return ""
            size = p.stat().st_size
            with open(p, 'rb') as fh:
                if size > max_bytes:
                    fh.seek(size - max_bytes)
                data = fh.read()
            text = data.decode('utf-8', errors='replace')
            return text[-1200:]
        except Exception:
            return ""
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert torch tensors and other non-serializable objects to JSON-safe formats"""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, '__dict__'):  # custom objects
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj

    def _create_failed_result(
        self,
        reward_code: str,
        identifier: str,
        error: str,
        log_dir: Optional[str] = None
    ) -> RewardFunctionResult:
        """Create a failed RewardFunctionResult with standard error values."""
        return RewardFunctionResult(
            reward_code=reward_code,
            identifier=identifier,
            training_successful=False,
            success_rate=-1.0,
            episode_length=0.0,
            training_time=0.0,
            final_reward=0.0,
            convergence_step=0,
            error_message=error,
            log_dir=log_dir,
            peak_memory_mb=0.0,
        )

    def evaluate_reward_function(
        self,
        reward_code: str,
        identifier: str,
        env_config: Dict[str, Any],
        optimization_config: Dict[str, Any],
        env_class_path: str = "VisFly.envs.NavigationEnv.NavigationEnv",
        base_output_dir: Optional[str] = None,
        eval_env_config: Optional[Dict[str, Any]] = None
    ) -> RewardFunctionResult:
        """
        Evaluate a reward function in a subprocess using simplified approach.
        """

        # Initialize variables early to avoid locals() checks
        stdout_file = None
        stderr_file = None

        # Create configuration file for subprocess
        config_file = self.temp_dir / f"{identifier}_config.json"
        output_file = self.temp_dir / f"{identifier}_result.json"

        # Convert torch tensors to lists for JSON serialization
        serializable_env_config = self._make_json_serializable(env_config)
        allocated_device: Optional[str] = None
        peak_memory_mb = 0.0

        # Determine iteration folder; prefer explicit field in optimization_config
        iteration_value = optimization_config.get('iteration')
        if iteration_value is None:
            # Fallback: parse prefix like iter0_sample0
            if identifier.startswith('iter') and '_sample' in identifier:
                iteration_value = identifier.split('_')[0].replace('iter', '')
            else:
                iteration_value = 0
        iteration_folder = f"iter{iteration_value}"

        # Create output directory for this evaluation (train/iterX/sampleY)
        if base_output_dir:
            eval_output_dir = Path(base_output_dir) / "train" / iteration_folder / identifier
        else:
            eval_output_dir = self.temp_dir / f"{identifier}_outputs"
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute project root dynamically (repo root)
        project_root = Path(__file__).resolve().parents[2]

        config = {
            'reward_code': reward_code,
            'identifier': identifier,
            'env_config': serializable_env_config,
            'eval_env_config': self._make_json_serializable(eval_env_config) if eval_env_config else serializable_env_config,
            'optimization_config': optimization_config,
            'env_class': env_class_path,
            'output_dir': str(eval_output_dir),
            'project_root': str(project_root),
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        try:
            # Load algorithm config to determine GPU requirements
            if "algorithm" not in optimization_config:
                raise ValueError("Algorithm must be specified in optimization_config")
            algorithm = optimization_config["algorithm"].lower()
            env_name = env_class_path.split('.')[-2].lower().replace('env', '')
            alg_cfg_path = project_root / "configs" / "algs" / env_name / f"{algorithm}.yaml"

            with open(alg_cfg_path, 'r') as f:
                alg_cfg = yaml.safe_load(f)
            if "algorithm" not in alg_cfg:
                raise ValueError(f"Algorithm config missing 'algorithm' section in {alg_cfg_path}")
            alg_params = alg_cfg["algorithm"]
            if "device" not in alg_params:
                raise ValueError(f"Device must be specified in algorithm config for {algorithm} in {alg_cfg_path}")
            alg_device = alg_params["device"]
            
            # Run subprocess
            self.logger.debug(f"Starting subprocess for {identifier}")
            
            cmd = [sys.executable, str(self.eval_script), str(config_file), str(output_file)]
            
            # Create log files for stdout/stderr
            stdout_file = eval_output_dir / "training_stdout.log"
            stderr_file = eval_output_dir / "training_stderr.log"
            
            # Setup environment with GPU allocation
            subprocess_env = os.environ.copy()
            # Prepend dynamic project paths to PYTHONPATH
            py_paths = [str(project_root), str(project_root / "VisFly")]
            existing = subprocess_env.get("PYTHONPATH", "")
            subprocess_env["PYTHONPATH"] = os.pathsep.join(py_paths + ([existing] if existing else []))
            
            # Allocate GPU for this job if algorithm requires CUDA
            if alg_device != "cpu":
                allocated_device = self.gpu_allocator.allocate_gpu_with_queue(
                    job_id=identifier,
                    memory_requirement_mb=optimization_config.get(
                        "gpu_memory_requirement_mb", 2048
                    ),
                    allow_cpu_fallback=True,
                )

                if allocated_device and allocated_device.startswith("cuda:"):
                    gpu_id = allocated_device.split(":")[1]
                    subprocess_env["CUDA_VISIBLE_DEVICES"] = gpu_id
                    self.logger.debug(f"[{identifier}] Allocated GPU {gpu_id}")
                else:
                    subprocess_env["CUDA_VISIBLE_DEVICES"] = ""
                    self.logger.debug(f"[{identifier}] Using CPU (no GPU available)")
            else:
                subprocess_env["CUDA_VISIBLE_DEVICES"] = ""
                self.logger.debug(f"[{identifier}] Algorithm configured for CPU")

            # When GPU allocation was used (including fallback to CPU), tell worker the effective device
            if allocated_device is not None:
                with open(config_file, "r") as f:
                    config = json.load(f)
                config.setdefault("optimization_config", {})["effective_algorithm_device"] = allocated_device
                with open(config_file, "w") as f:
                    json.dump(config, f, indent=2)
            
            subprocess_env.update({
                "OMP_NUM_THREADS": str(torch.get_num_threads()),
                "MKL_NUM_THREADS": str(torch.get_num_threads()), 
                "NUMBA_NUM_THREADS": str(torch.get_num_threads()),
                "NUMEXPR_MAX_THREADS": str(os.cpu_count() or 1),  # Use all available cores
                "OMP_PROC_BIND": "false",
                "OMP_PLACES": "cores",
                "CUDA_LAUNCH_BLOCKING": "0",
                "PYTHONUNBUFFERED": "1",
            })
            
            # Use Popen for live streaming
            with open(stdout_file, 'w') as stdout_f, open(stderr_file, 'w') as stderr_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=subprocess_env,
                    preexec_fn=os.setsid  # start new process group (POSIX)
                )

                # Track progress for heartbeat
                last_heartbeat = time.time()
                heartbeat_interval = 30  # seconds
                
                # Save stdout/stderr to files, selectively forward important messages to console
                def stream_output(pipe, file_handle, prefix):
                    nonlocal last_heartbeat
                    for line in iter(pipe.readline, ''):
                        file_handle.write(line)
                        file_handle.flush()
                        
                        line_stripped = line.strip()
                        upper = line_stripped.upper()
                        
                        # Log only key milestones (not all [stage] messages)
                        if any(k in upper for k in [
                            'STARTING TRAINING', 'TRAINING COMPLETED', 
                            'EVALUATION COMPLETED'
                        ]):
                            self.logger.info(f"[{identifier}] {line_stripped}")
                        # Log critical errors (but skip metric names containing "error")
                        elif any(k in upper for k in ["EXCEPTION", "TRACEBACK", "FAILED"]):
                            self.logger.warning(f"[{identifier}] {line_stripped}")
                        elif "ERROR" in upper:
                            # Skip common non-critical "error" mentions in metric names and internal VisFly errors
                            if not any(skip in upper for skip in [
                                "ERROR_MEAN", "ERROR_R", "POS_ERROR", "_ERROR",
                                "JSONSTLTYPES", "NAVMESH_INSTANCES"
                            ]):
                                self.logger.warning(f"[{identifier}] {line_stripped}")

                        # Periodic heartbeat to show progress (debug level)
                        current_time = time.time()
                        if current_time - last_heartbeat > heartbeat_interval:
                            self.logger.debug(f"[{identifier}] Training in progress...")
                            last_heartbeat = current_time
                
                # Start streaming threads
                stdout_thread = threading.Thread(
                    target=stream_output, 
                    args=(process.stdout, stdout_f, "OUT"),
                    daemon=True
                )
                stderr_thread = threading.Thread(
                    target=stream_output, 
                    args=(process.stderr, stderr_f, "ERR"),
                    daemon=True
                )
                
                stdout_thread.start()
                stderr_thread.start()
                
                # Wait for process completion
                try:
                    exit_code = process.wait()
                finally:
                    stdout_thread.join(timeout=5)
                    stderr_thread.join(timeout=5)
                    if process.stdout:
                        process.stdout.close()
                    if process.stderr:
                        process.stderr.close()
            
            # Check if output file was created
            if not output_file.exists():
                with open(eval_output_dir / "reward_function.py", 'w') as f:
                    f.write(reward_code)
                # Attach stderr tail for diagnostics
                err_tail = self._read_tail(stderr_file) or self._read_tail(stdout_file)
                # Detect empty stdout (possible crash before logging started)
                empty_stdout = not stdout_file.exists() or stdout_file.stat().st_size == 0
                detail = f"No output file created. Return code: {exit_code}"
                if empty_stdout:
                    detail += "\nStdout was empty; process may have failed before training initialization (check stderr tail)."
                if err_tail:
                    detail += f"\n--- stderr tail ---\n{err_tail}"
                return self._create_failed_result(reward_code, identifier, detail, str(eval_output_dir))
            
            # Load result
            try:
                with open(output_file, 'r') as f:
                    result_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                detail = f"Failed to parse output file: {e}\nFile: {output_file}"
                self.logger.error(detail)
                return self._create_failed_result(reward_code, identifier, detail, str(eval_output_dir))
            
            # Persist raw result for debugging
            try:
                with open(eval_output_dir / "result.json", 'w') as rf:
                    json.dump(result_data, rf, indent=2)
            except Exception as persist_err:
                self.logger.debug(f"Could not persist result.json: {persist_err}")

            if result_data['success']:
                # Save the reward function
                with open(eval_output_dir / "reward_function.py", 'w') as f:
                    f.write(reward_code)

                self.logger.info(f"Saved training outputs to {eval_output_dir}")

                # Find the correct tensorboard log directory (handles nested structures)
                log_dir_path = find_tensorboard_logdir(str(eval_output_dir))
                if not log_dir_path:
                    log_dir_path = str(eval_output_dir)  # Fallback to output dir
                peak_memory_mb = result_data.get('peak_memory_mb', 0.0)

                return RewardFunctionResult(
                    reward_code=result_data['reward_code'],
                    identifier=result_data['identifier'],
                    training_successful=True,
                    success_rate=result_data['success_rate'],
                    episode_length=result_data['episode_length'],
                    training_time=result_data['training_time'],
                    final_reward=result_data['final_reward'],
                    convergence_step=result_data['convergence_step'],
                    log_dir=log_dir_path,
                    peak_memory_mb=peak_memory_mb,
                    evaluation_summary=result_data.get('aggregate_statistics')
                    or result_data.get('evaluation_summary'),
                    episode_statistics=result_data.get('episode_statistics', []),
                    video_paths=result_data.get('video_paths', []),
                )
            else:
                # Include log_dir and stderr tail on failure
                err_tail = self._read_tail(stderr_file) or self._read_tail(stdout_file)
                detail = result_data.get('error', 'Unknown error')
                tb = result_data.get('traceback')
                if tb:
                    # append a small traceback snippet
                    tb_tail = str(tb)[-1200:]
                    detail += f"\n--- traceback tail ---\n{tb_tail}"
                if err_tail:
                    detail += f"\n--- stderr tail ---\n{err_tail}"
                # Persist failure summary
                try:
                    with open(eval_output_dir / "failure_summary.txt", 'w') as ff:
                        ff.write(detail)
                except Exception:
                    pass

                peak_memory_mb = result_data.get('peak_memory_mb', 0.0)

                return self._create_failed_result(reward_code, identifier, detail, str(eval_output_dir))
                
        except Exception as e:
            self.logger.error(f"Subprocess evaluation failed for {identifier}: {e}")
            # Try to attach stderr tail if available
            try:
                err_tail = self._read_tail(stderr_file) if stderr_file else ""
                detail = f"{e}\n--- stderr tail ---\n{err_tail}" if err_tail else str(e)
            except Exception:
                detail = str(e)
            return self._create_failed_result(
                reward_code, identifier, detail,
                str(eval_output_dir) if eval_output_dir else None
            )
            
        finally:
            # Release GPU allocation if it was allocated
            try:
                if allocated_device is not None:
                    self.gpu_allocator.release_gpu(identifier, peak_memory_mb)
            except Exception as e:
                self.logger.debug(f"GPU cleanup failed for {identifier}: {e}")
            
            # Cleanup temporary files
            try:
                if config_file.exists():
                    config_file.unlink()
                if output_file.exists():
                    output_file.unlink()
            except OSError as e:
                self.logger.debug(f"Cleanup temp files failed: {e}")
            except Exception as e:
                self.logger.debug(f"Cleanup encountered non-OS error: {e}")
    
    def evaluate_multiple_parallel(
        self,
        reward_functions: List[str],
        identifiers: List[str],
        env_config: Dict[str, Any],
        optimization_config: Dict[str, Any],
        env_class_path: str = "VisFly.envs.NavigationEnv.NavigationEnv",
        max_concurrent: int = 4,
        base_output_dir: Optional[str] = None,
        eval_env_config: Optional[Dict[str, Any]] = None
    ) -> List[RewardFunctionResult]:
        """
        Evaluate multiple reward functions in parallel subprocesses.
        """
        results_map: dict[int, RewardFunctionResult] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_idx: Dict[concurrent.futures.Future, int] = {}
            for i in range(len(reward_functions)):
                submit_args = (
                    reward_functions[i],
                    identifiers[i],
                    env_config,
                    optimization_config,
                    env_class_path,
                )
                submit_kwargs = {
                    "base_output_dir": base_output_dir,
                    "eval_env_config": eval_env_config,
                }

                future = executor.submit(
                    self.evaluate_reward_function, *submit_args, **submit_kwargs
                )
                future_to_idx[future] = i
            
            # Collect results then order by index
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results_map[idx] = result
                    self.logger.debug(f"Completed evaluation {idx + 1}/{len(reward_functions)}: {identifiers[idx]}")
                except Exception as e:
                    self.logger.error(f"Failed evaluation {idx + 1}: {e}")
                    results_map[idx] = RewardFunctionResult.failed(
                        reward_functions[idx], identifiers[idx], str(e)
                    )
        
        # Build ordered list of results
        ordered_results: List[RewardFunctionResult] = [results_map[i] for i in range(len(reward_functions))]
        return ordered_results
    
    def cleanup(self):
        """Clean up temporary directory and GPU monitoring"""
        # EnhancedGPUMonitor doesn't need explicit stop
        pass

        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp directory: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()
