"""
GPU Monitoring and Resource Management for Parallel Training

This module provides GPU monitoring capabilities and intelligent resource
allocation for running multiple training processes in parallel.
"""

import torch
import subprocess
import json
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from threading import Thread, Event, Lock
import psutil
import os


@dataclass
class GPUInfo:
    """Information about a GPU device"""
    device_id: int
    name: str
    memory_total: int  # MB
    memory_used: int   # MB
    memory_free: int   # MB
    utilization: int   # percentage
    temperature: int   # Celsius
    power_usage: int   # Watts
    processes: List[Dict]


class GPUMonitor:
    """Monitor GPU usage and manage resource allocation"""
    
    def __init__(self, update_interval: float = None):
        """
        Initialize GPU monitor.
        
        Args:
            update_interval: How often to update GPU stats (seconds)
        """
        if update_interval is None:
            # Import here to avoid circular imports
            try:
                from config import GPU_CONFIG
                update_interval = GPU_CONFIG["gpu_monitor_update_interval"]
            except ImportError:
                update_interval = 5.0  # fallback default
        
        self.update_interval = update_interval
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.monitor_thread = None
        self.stop_event = Event()
        self.gpu_stats = {}
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available - GPU monitoring disabled")
            self.num_gpus = 0
        else:
            self.num_gpus = torch.cuda.device_count()
            self.logger.info(f"Found {self.num_gpus} GPU(s)")
            
            # Initialize fast memory monitoring
            self._torch_memory_cache = {}
            self._torch_cache_time = 0
    
    def get_gpu_info(self) -> List[GPUInfo]:
        """Get current information for all GPUs"""
        if self.num_gpus == 0:
            return []
        
        try:
            # Use nvidia-smi to get detailed GPU information
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,'
                'utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True, timeout=30)
            
            gpu_infos = []
            lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            for line in lines:
                if not line.strip():
                    continue
                    
                parts = [p.strip() for p in line.split(',')]
                if len(parts) != 8:
                    continue
                
                try:
                    device_id = int(parts[0])
                    name = parts[1]
                    memory_total = int(parts[2])
                    memory_used = int(parts[3])
                    memory_free = int(parts[4])
                    utilization = int(parts[5])
                    temperature = int(parts[6])
                    power_usage = int(float(parts[7]))  # Power can be float
                    
                    # Get processes for this GPU
                    processes = self._get_gpu_processes(device_id)
                    
                    gpu_infos.append(GPUInfo(
                        device_id=device_id,
                        name=name,
                        memory_total=memory_total,
                        memory_used=memory_used,
                        memory_free=memory_free,
                        utilization=utilization,
                        temperature=temperature,
                        power_usage=power_usage,
                        processes=processes
                    ))
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Failed to parse GPU info line: {line} - {e}")
                    continue
            
            return gpu_infos
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get GPU info: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error getting GPU info: {e}")
            return []
    
    def get_fast_memory_info(self) -> Dict[int, Dict[str, int]]:
        """
        Get GPU memory info using PyTorch (faster than nvidia-smi).
        
        Returns:
            Dict mapping GPU ID to memory info (total, used, free in MB)
        """
        current_time = time.time()
        
        # Use cache if recent (within 2 seconds)
        if current_time - self._torch_cache_time < 2.0:
            return self._torch_memory_cache
        
        memory_info = {}
        
        try:
            for gpu_id in range(self.num_gpus):
                with torch.cuda.device(gpu_id):
                    # Get memory info in bytes, convert to MB
                    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory // (1024 * 1024)
                    allocated_memory = torch.cuda.memory_allocated() // (1024 * 1024)
                    cached_memory = torch.cuda.memory_reserved() // (1024 * 1024)
                    
                    # Estimated free memory (approximation)
                    used_memory = max(allocated_memory, cached_memory)
                    free_memory = max(0, total_memory - used_memory)
                    
                    memory_info[gpu_id] = {
                        'total': int(total_memory),
                        'used': int(used_memory),
                        'free': int(free_memory)
                    }
        except Exception as e:
            self.logger.debug(f"Fast memory check failed: {e}")
            return {}
        
        self._torch_memory_cache = memory_info
        self._torch_cache_time = current_time
        return memory_info
    
    def _get_gpu_processes(self, gpu_id: int) -> List[Dict]:
        """Get processes running on a specific GPU"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-compute-apps=gpu_bus_id,pid,process_name,used_memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            processes = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                    
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    try:
                        processes.append({
                            'pid': int(parts[1]),
                            'name': parts[2],
                            'memory_mb': int(parts[3])
                        })
                    except (ValueError, IndexError):
                        continue
            
            return processes
            
        except subprocess.CalledProcessError:
            return []
    
    def start_monitoring(self):
        """Start continuous GPU monitoring"""
        if self.monitoring or self.num_gpus == 0:
            return
        
        self.monitoring = True
        self.stop_event.clear()
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        # Started GPU monitoring
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Stopped GPU monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring and not self.stop_event.wait(self.update_interval):
            try:
                gpu_infos = self.get_gpu_info()
                self.gpu_stats = {gpu.device_id: gpu for gpu in gpu_infos}
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def get_available_gpus(self, min_memory_mb: int = None, max_utilization: int = None) -> List[int]:
        """
        Get list of GPU IDs that are available for new training jobs.
        
        Args:
            min_memory_mb: Minimum free memory required (MB)
            max_utilization: Maximum utilization percentage for available GPU
            
        Returns:
            List of available GPU device IDs
        """
        if min_memory_mb is None or max_utilization is None:
            try:
                from config import GPU_CONFIG
                min_memory_mb = min_memory_mb or GPU_CONFIG["min_memory_mb"]
                max_utilization = max_utilization or GPU_CONFIG["max_utilization_percent"]
            except ImportError:
                min_memory_mb = min_memory_mb or 4000
                max_utilization = max_utilization or 30
        
        # Use cached stats if monitoring is active and recent
        current_time = time.time()
        if not self.monitoring:
            # Get current stats if not monitoring
            gpu_infos = self.get_gpu_info()
            self.gpu_stats = {gpu.device_id: gpu for gpu in gpu_infos}
            self._last_update = current_time
        elif not hasattr(self, '_last_update') or (current_time - self._last_update) > self.update_interval * 2:
            # Force refresh if data is too old
            gpu_infos = self.get_gpu_info()
            self.gpu_stats = {gpu.device_id: gpu for gpu in gpu_infos}
            self._last_update = current_time
        
        available = []
        for gpu_id, gpu_info in self.gpu_stats.items():
            if (gpu_info.memory_free >= min_memory_mb and 
                gpu_info.utilization <= max_utilization):
                available.append(gpu_id)
        
        # Only log if there's a change in available GPUs or every 5 calls
        if not hasattr(self, '_last_available') or self._last_available != available:
            self.logger.info(f"Available GPUs: {available} (mem>={min_memory_mb}MB, util<={max_utilization}%)")
            self._last_available = available
        
        return available
    
    def estimate_max_parallel_jobs(self, memory_per_job_mb: int = 8000) -> Dict[int, int]:
        """
        Estimate maximum parallel jobs per GPU based on memory requirements.
        
        Args:
            memory_per_job_mb: Estimated memory per training job (MB)
            
        Returns:
            Dict mapping GPU ID to max parallel jobs
        """
        estimates = {}
        
        if not self.monitoring:
            gpu_infos = self.get_gpu_info()
            self.gpu_stats = {gpu.device_id: gpu for gpu in gpu_infos}
        
        for gpu_id, gpu_info in self.gpu_stats.items():
            # Reserve some memory for system overhead
            usable_memory = max(0, gpu_info.memory_free - 1000)  # Reserve 1GB
            max_jobs = max(0, usable_memory // memory_per_job_mb)
            estimates[gpu_id] = max_jobs
            
            self.logger.info(f"GPU {gpu_id}: {usable_memory}MB free -> max {max_jobs} jobs "
                           f"({memory_per_job_mb}MB each)")
        
        return estimates
    
    def log_gpu_status(self):
        """Log current GPU status"""
        if self.num_gpus == 0:
            self.logger.info("No GPUs available")
            return
        
        if not self.monitoring:
            gpu_infos = self.get_gpu_info()
        else:
            gpu_infos = list(self.gpu_stats.values())
        
        # GPU status available but not logging to reduce verbosity


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
    
    def __init__(self, monitor: GPUMonitor):
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
        self.allocated_gpus = {}  # job_id -> gpu_id mapping
        self.gpu_assignments = {}  # gpu_id -> [job_ids] mapping
        self.job_profiles = {}  # job_id -> JobMemoryProfile
        self.job_queue = []  # List of (job_id, memory_requirement_mb) waiting for GPU
        self.memory_estimates = {}  # job_type -> estimated_memory_mb
        self._lock = Lock()  # Thread-safe access to shared state
        
        # Memory monitoring settings
        self.memory_check_interval = 10.0  # seconds
        self.memory_safety_margin = 1000  # MB reserve for safety
        self.max_memory_growth_factor = 1.5  # Allow 50% growth from initial estimate
    
    def allocate_gpu_with_queue(self, job_id: str, memory_requirement_mb: int = 8000, allow_cpu_fallback: bool = True) -> Optional[str]:
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
                self.logger.warning(f"No GPU available for job {job_id} (need {memory_requirement_mb}MB), falling back to CPU")
                self.allocated_gpus[job_id] = "cpu"
                return "cpu"
            else:
                # Add to queue and return None (caller should wait)
                self.job_queue.append((job_id, memory_requirement_mb))
                self.logger.info(f"Job {job_id} queued (need {memory_requirement_mb}MB), queue length: {len(self.job_queue)}")
                return None
    
    def _try_immediate_allocation(self, job_id: str, memory_requirement_mb: int) -> Optional[str]:
        """Try to allocate GPU immediately based on current memory usage and projections"""
        # Get fresh memory info
        memory_info = self.monitor.get_fast_memory_info()
        if not memory_info:
            return None
        
        best_gpu = None
        max_available_memory = 0
        
        for gpu_id, mem_info in memory_info.items():
            # Calculate projected memory usage for this GPU
            projected_usage = self._calculate_projected_memory_usage(gpu_id, mem_info)
            available_memory = mem_info['total'] - projected_usage - self.memory_safety_margin
            
            # Check if this GPU can fit the new job
            if available_memory >= memory_requirement_mb and available_memory > max_available_memory:
                max_available_memory = available_memory
                best_gpu = gpu_id
        
        if best_gpu is not None:
            # Allocate the GPU
            self.allocated_gpus[job_id] = best_gpu
            if best_gpu not in self.gpu_assignments:
                self.gpu_assignments[best_gpu] = []
            self.gpu_assignments[best_gpu].append(job_id)
            
            # Create initial memory profile
            initial_memory = memory_info[best_gpu]['used']
            self.job_profiles[job_id] = JobMemoryProfile(
                job_id=job_id,
                gpu_id=best_gpu,
                initial_memory_mb=initial_memory,
                peak_memory_mb=initial_memory,
                current_memory_mb=initial_memory,
                start_time=time.time(),
                last_update=time.time(),
                training_step=0
            )
            
            self.logger.info(f"Allocated GPU {best_gpu} to job {job_id} "
                           f"({max_available_memory}MB available, {len(self.gpu_assignments[best_gpu])} jobs)")
            return f"cuda:{best_gpu}"
        
        return None
    
    def _calculate_projected_memory_usage(self, gpu_id: int, mem_info: Dict[str, int]) -> int:
        """Calculate projected memory usage for a GPU based on running jobs"""
        current_used = mem_info['used']
        
        # Add projected growth for running jobs
        projected_growth = 0
        for job_id in self.gpu_assignments.get(gpu_id, []):
            if job_id in self.job_profiles:
                profile = self.job_profiles[job_id]
                
                # Estimate how much more memory this job might use
                time_running = time.time() - profile.start_time
                if time_running < 300:  # First 5 minutes - expect growth
                    growth_factor = min(self.max_memory_growth_factor, 1.0 + (300 - time_running) / 600)
                    estimated_peak = profile.current_memory_mb * growth_factor
                    projected_growth += max(0, estimated_peak - profile.current_memory_mb)
        
        return current_used + projected_growth
    
    def update_job_memory_usage(self, job_id: str, training_step: int):
        """Update memory usage for a running job (call this from training loop)"""
        if job_id not in self.job_profiles:
            return
        
        with self._lock:
            profile = self.job_profiles[job_id]
            
            # Get current memory usage for this GPU
            memory_info = self.monitor.get_fast_memory_info()
            if profile.gpu_id in memory_info:
                current_memory = memory_info[profile.gpu_id]['used']
                
                # Update profile
                profile.current_memory_mb = current_memory
                profile.peak_memory_mb = max(profile.peak_memory_mb, current_memory)
                profile.training_step = training_step
                profile.last_update = time.time()
                
                # After 1000 steps, estimate final memory usage
                if training_step >= 1000 and profile.estimated_final_memory_mb is None:
                    profile.estimated_final_memory_mb = int(profile.peak_memory_mb * 1.1)  # Add 10% buffer
                    self.logger.info(f"Job {job_id} estimated final memory: {profile.estimated_final_memory_mb}MB "
                                   f"(current: {current_memory}MB, peak: {profile.peak_memory_mb}MB)")
    
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
                    self.logger.info(f"GPU {gpu_id} overloaded with {current_jobs} jobs, queueing {job_id} for retry")
                    self.job_queue.append((job_id, profile.peak_memory_mb * 1.2))  # Request 20% more memory
                    return True
                else:
                    # Only job on GPU but still OOM - probably job is too large for this GPU
                    self.logger.error(f"Job {job_id} too large for GPU {gpu_id} ({profile.peak_memory_mb}MB), failing")
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
    
    def release_gpu(self, job_id: str):
        """Release GPU/CPU allocation for a completed job"""
        with self._lock:
            if job_id in self.allocated_gpus:
                device_id = self.allocated_gpus[job_id]
                del self.allocated_gpus[job_id]
                
                # Only manage GPU assignments for actual GPUs, not CPU
                if device_id != "cpu" and device_id in self.gpu_assignments and job_id in self.gpu_assignments[device_id]:
                    self.gpu_assignments[device_id].remove(job_id)
                    if not self.gpu_assignments[device_id]:
                        del self.gpu_assignments[device_id]
                
                # Clean up job profile
                if job_id in self.job_profiles:
                    profile = self.job_profiles[job_id]
                    self.logger.info(f"Job {job_id} completed - Peak memory: {profile.peak_memory_mb}MB, "
                                   f"Final memory estimate: {profile.estimated_final_memory_mb}MB")
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


def main():
    """Test GPU monitoring functionality"""
    logging.basicConfig(level=logging.INFO)
    
    monitor = GPUMonitor()
    monitor.start_monitoring()
    
    try:
        monitor.log_gpu_status()
        
        available = monitor.get_available_gpus()
        print(f"Available GPUs: {available}")
        
        estimates = monitor.estimate_max_parallel_jobs()
        print(f"Max parallel jobs: {estimates}")
        
        # Test for 10 seconds
        time.sleep(10)
        
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    main()