"""
DynamicGPUResourceManager adapted for EnhancedGPUMonitor
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from threading import Lock

from .gpu_monitor import EnhancedGPUMonitor
from ..constants import (
    GPU_MEMORY_SAFETY_MARGIN,
    MEMORY_GROWTH_ESTIMATION_PERIOD,
    MEMORY_GROWTH_CALCULATION_DIVISOR,
    MEMORY_GROWTH_FACTOR_MAX,
    MEMORY_ESTIMATION_THRESHOLD_STEPS,
    MEMORY_ESTIMATION_BUFFER_FACTOR,
    OOM_RETRY_MEMORY_INCREASE_FACTOR,
)


@dataclass
class JobMemoryProfile:
    """Memory usage profile for a training job"""
    job_id: str
    gpu_id: int
    initial_memory_mb: int
    peak_memory_mb: int
    current_memory_mb: int
    start_time: float
    last_update: float
    training_step: int
    estimated_final_memory_mb: Optional[int] = None


class DynamicGPUResourceManager:
    """Advanced GPU resource manager with dynamic memory monitoring and queuing"""

    def __init__(self, monitor: Optional[EnhancedGPUMonitor] = None, utilization_threshold: int = 80):
        """
        Initialize the resource manager.

        Args:
            monitor: Optional EnhancedGPUMonitor instance. If None, creates one.
            utilization_threshold: Maximum GPU utilization % to consider available (default 80%)
        """
        self.monitor = monitor or EnhancedGPUMonitor(utilization_threshold=utilization_threshold)
        self.logger = logging.getLogger(__name__)
        self.allocated_gpus = {}  # job_id -> gpu_id mapping
        self.gpu_assignments = {}  # gpu_id -> [job_ids] mapping
        self.job_profiles = {}  # job_id -> JobMemoryProfile
        self.job_queue = []  # List of (job_id, memory_requirement_mb) waiting for GPU
        self.memory_estimates = {}  # job_type -> estimated_memory_mb
        self._lock = Lock()  # Thread-safe access to shared state

        # Memory monitoring settings
        self.memory_check_interval = 10.0  # seconds
        self.memory_safety_margin = GPU_MEMORY_SAFETY_MARGIN
        self.max_memory_growth_factor = MEMORY_GROWTH_FACTOR_MAX

        # Load balancing
        self.round_robin_counter = 0  # For better GPU distribution

    def allocate_gpu_with_queue(
        self,
        job_id: str,
        memory_requirement_mb: int = 8000,  # Default 8GB per job
        allow_cpu_fallback: bool = True,
    ) -> Optional[str]:
        """
        Smart GPU allocation with dynamic memory monitoring and queuing.

        Args:
            job_id: Unique identifier for the job
            memory_requirement_mb: Initial memory requirement estimate for the job
            allow_cpu_fallback: Whether to fall back to CPU if no GPU available

        Returns:
            Device identifier ("cuda:X" for GPU X, "cpu" for CPU fallback, None if queued)
        """
        with self._lock:
            # Step 1: Check if we can fit this job on any GPU considering current usage + projections
            device = self._try_immediate_allocation(job_id, memory_requirement_mb)
            if device:
                return device

            # Step 2: If no immediate allocation possible, add to queue unless CPU fallback allowed
            if allow_cpu_fallback:
                self.logger.warning(
                    f"No GPU available for job {job_id} (need {memory_requirement_mb}MB), falling back to CPU"
                )
                self.allocated_gpus[job_id] = "cpu"
                return "cpu"
            else:
                # Add to queue and return None (caller should wait)
                self.job_queue.append((job_id, memory_requirement_mb))
                self.logger.info(
                    f"Job {job_id} queued (need {memory_requirement_mb}MB), queue length: {len(self.job_queue)}"
                )
                return None

    def _try_immediate_allocation(
        self, job_id: str, memory_requirement_mb: int
    ) -> Optional[str]:
        """Try to allocate GPU immediately based on current memory usage and projections"""

        # Use the EnhancedGPUMonitor to get available GPUs (already filtered by utilization)
        available_gpus = self.monitor.get_available_gpus(min_memory_mb=memory_requirement_mb)

        if not available_gpus:
            return None

        # Get GPU info for load balancing
        gpu_infos = self.monitor.get_gpu_info()
        gpu_info_dict = {gpu.device_id: gpu for gpu in gpu_infos}

        # Sort available GPUs by current job count (load balancing), then by free memory
        gpu_candidates = []
        for gpu_id in available_gpus:
            if gpu_id in gpu_info_dict:
                gpu = gpu_info_dict[gpu_id]
                current_jobs = len(self.gpu_assignments.get(gpu_id, []))

                # Calculate projected memory with running jobs
                projected_used = self._calculate_projected_memory_usage(gpu_id, gpu.memory_used)
                available_memory = gpu.memory_total - projected_used - self.memory_safety_margin

                if available_memory >= memory_requirement_mb:
                    gpu_candidates.append({
                        'gpu_id': gpu_id,
                        'available_memory': available_memory,
                        'current_jobs': current_jobs,
                        'utilization': gpu.utilization,
                        'free_memory': gpu.memory_free
                    })

        if not gpu_candidates:
            return None

        # Sort by current job count (load balancing), then by available memory
        gpu_candidates.sort(key=lambda x: (x['current_jobs'], -x['available_memory']))

        # Use round-robin among GPUs with same job count for better distribution
        min_jobs = gpu_candidates[0]['current_jobs']
        least_loaded_gpus = [gpu for gpu in gpu_candidates if gpu['current_jobs'] == min_jobs]

        if len(least_loaded_gpus) > 1:
            # Round-robin among least loaded GPUs
            selected_idx = self.round_robin_counter % len(least_loaded_gpus)
            self.round_robin_counter += 1
            selected = least_loaded_gpus[selected_idx]
        else:
            selected = gpu_candidates[0]

        best_gpu = selected['gpu_id']

        # Allocate the GPU
        self.allocated_gpus[job_id] = best_gpu
        if best_gpu not in self.gpu_assignments:
            self.gpu_assignments[best_gpu] = []
        self.gpu_assignments[best_gpu].append(job_id)

        # Create initial memory profile
        gpu_info = gpu_info_dict[best_gpu]
        self.job_profiles[job_id] = JobMemoryProfile(
            job_id=job_id,
            gpu_id=best_gpu,
            initial_memory_mb=gpu_info.memory_used,
            peak_memory_mb=gpu_info.memory_used,
            current_memory_mb=gpu_info.memory_used,
            start_time=time.time(),
            last_update=time.time(),
            training_step=0,
        )

        self.logger.info(
            f"Allocated GPU {best_gpu} to job {job_id} "
            f"({selected['available_memory']}MB available, {selected['utilization']}% util, "
            f"{len(self.gpu_assignments[best_gpu])} jobs)"
        )
        return f"cuda:{best_gpu}"

    def _calculate_projected_memory_usage(self, gpu_id: int, current_used_mb: int) -> int:
        """Calculate projected memory usage for a GPU based on running jobs"""

        # Add projected growth for running jobs
        projected_growth = 0
        for job_id in self.gpu_assignments.get(gpu_id, []):
            if job_id in self.job_profiles:
                profile = self.job_profiles[job_id]

                # Estimate how much more memory this job might use
                time_running = time.time() - profile.start_time
                if time_running < MEMORY_GROWTH_ESTIMATION_PERIOD:  # First 5 minutes - expect growth
                    growth_factor = min(
                        self.max_memory_growth_factor,
                        1.0 + (MEMORY_GROWTH_ESTIMATION_PERIOD - time_running) / MEMORY_GROWTH_CALCULATION_DIVISOR
                    )
                    estimated_peak = profile.current_memory_mb * growth_factor
                    projected_growth += max(0, estimated_peak - profile.current_memory_mb)

        return current_used_mb + projected_growth

    def update_job_memory_usage(self, job_id: str, training_step: int):
        """Update memory usage for a running job (call this from training loop)"""
        if job_id not in self.job_profiles:
            return

        with self._lock:
            profile = self.job_profiles[job_id]

            # Get current memory usage for this GPU
            total, used, free = self.monitor.get_memory_usage(profile.gpu_id)

            # Update profile
            profile.current_memory_mb = used
            profile.peak_memory_mb = max(profile.peak_memory_mb, used)
            profile.training_step = training_step
            profile.last_update = time.time()

            # After threshold steps, estimate final memory usage
            if training_step >= MEMORY_ESTIMATION_THRESHOLD_STEPS and profile.estimated_final_memory_mb is None:
                profile.estimated_final_memory_mb = int(
                    profile.peak_memory_mb * MEMORY_ESTIMATION_BUFFER_FACTOR
                )
                self.logger.info(
                    f"Job {job_id} estimated final memory: {profile.estimated_final_memory_mb}MB "
                    f"(current: {used}MB, peak: {profile.peak_memory_mb}MB)"
                )

    def handle_cuda_oom_error(self, job_id: str) -> bool:
        """
        Handle CUDA OOM error for a job. Return True if job should be retried, False if failed.
        """
        with self._lock:
            if job_id in self.job_profiles:
                profile = self.job_profiles[job_id]
                gpu_id = profile.gpu_id

                self.logger.error(f"CUDA OOM error for job {job_id} on GPU {gpu_id}")

                # Check if this GPU is overloaded
                current_jobs = len(self.gpu_assignments.get(gpu_id, []))
                if current_jobs > 1:
                    # Multiple jobs on this GPU - queue this job for retry when space available
                    self.logger.info(
                        f"GPU {gpu_id} overloaded with {current_jobs} jobs, queueing {job_id} for retry"
                    )
                    self.job_queue.append(
                        (job_id, int(profile.peak_memory_mb * OOM_RETRY_MEMORY_INCREASE_FACTOR))
                    )  # Request more memory for retry
                    return True
                else:
                    # Only job on GPU but still OOM - probably job is too large for this GPU
                    self.logger.error(
                        f"Job {job_id} too large for GPU {gpu_id} ({profile.peak_memory_mb}MB), failing"
                    )
                    return False
            return False

    def process_queue(self):
        """Process waiting jobs in queue when resources become available"""
        if not self.job_queue:
            return

        with self._lock:
            processed = []
            for i, (job_id, memory_requirement_mb) in enumerate(self.job_queue):
                device = self._try_immediate_allocation(job_id, memory_requirement_mb)
                if device:
                    processed.append(i)
                    self.logger.info(f"Allocated {device} to queued job {job_id}")
                    # Note: Caller needs to restart the job with this device

            # Remove processed jobs from queue (in reverse order to maintain indices)
            for i in reversed(processed):
                del self.job_queue[i]

    def release_gpu(self, job_id: str, actual_peak_memory_mb: float = 0.0):
        """Release GPU/CPU allocation for a completed job"""
        with self._lock:
            if job_id in self.allocated_gpus:
                device_id = self.allocated_gpus[job_id]
                del self.allocated_gpus[job_id]

                # Only manage GPU assignments for actual GPUs, not CPU
                if (
                    device_id != "cpu"
                    and device_id in self.gpu_assignments
                    and job_id in self.gpu_assignments[device_id]
                ):
                    self.gpu_assignments[device_id].remove(job_id)
                    if not self.gpu_assignments[device_id]:
                        del self.gpu_assignments[device_id]

                # Clean up job profile
                if job_id in self.job_profiles:
                    profile = self.job_profiles[job_id]
                    final_estimate = getattr(profile, 'estimated_final_memory_mb', None)
                    # Use actual peak memory from training results if provided
                    if actual_peak_memory_mb > 0:
                        actual_peak = f"{actual_peak_memory_mb:.0f}"
                    else:
                        tracked_peak = getattr(profile, 'peak_memory_mb', 0)
                        actual_peak = f"{tracked_peak:.0f}" if tracked_peak > 0 else "Unknown"

                    self.logger.info(
                        f"Job {job_id} completed - Peak memory: {actual_peak}MB, "
                        f"Final memory estimate: {final_estimate}MB"
                    )
                    del self.job_profiles[job_id]

                device_name = f"GPU {device_id}" if device_id != "cpu" else "CPU"
                self.logger.info(f"Released {device_name} from job {job_id}")

                # Process queue to assign waiting jobs
                self.process_queue()
            else:
                self.logger.warning(f"Attempted to release unknown job {job_id}")

    def get_gpu_load_balancing(self) -> Dict[int, int]:
        """Get current job count per GPU for load balancing"""
        with self._lock:
            load = {}
            for gpu_id, job_list in self.gpu_assignments.items():
                load[gpu_id] = len(job_list)
            return load