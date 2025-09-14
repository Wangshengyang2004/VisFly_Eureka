"""
Enhanced GPU Monitoring with nvitop support

This module provides GPU monitoring using nvitop when available,
falling back to nvidia-smi if needed.
"""

import torch
import subprocess
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Try to import nvitop
try:
    from nvitop import Device
    NVITOP_AVAILABLE = True
except ImportError:
    NVITOP_AVAILABLE = False

from ..constants import (
    GPU_MEMORY_CONVERSION_FACTOR,
    bytes_to_mb,
)


@dataclass
class GPUInfo:
    """Information about a GPU device"""
    device_id: int
    name: str
    memory_total: int  # MB
    memory_used: int  # MB
    memory_free: int  # MB
    utilization: int  # percentage
    temperature: int  # Celsius
    power_usage: int  # Watts
    processes: List[Dict]


class EnhancedGPUMonitor:
    """Enhanced GPU monitor with nvitop support"""

    def __init__(self, utilization_threshold: int = 80):
        """
        Initialize GPU monitor.

        Args:
            utilization_threshold: Maximum GPU utilization % to consider available (default 80%)
        """
        self.logger = logging.getLogger(__name__)
        self.utilization_threshold = utilization_threshold

        # Check if CUDA is available
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available - GPU monitoring disabled")
            self.num_gpus = 0
            self.use_nvitop = False
        else:
            self.num_gpus = torch.cuda.device_count()
            self.use_nvitop = NVITOP_AVAILABLE

            if self.use_nvitop:
                self.logger.info(f"Using nvitop for GPU monitoring ({self.num_gpus} GPUs)")
                # Initialize nvitop devices
                self.devices = [Device(i) for i in range(self.num_gpus)]
            else:
                self.logger.info(f"Using nvidia-smi for GPU monitoring ({self.num_gpus} GPUs)")

    def get_gpu_info_nvitop(self) -> List[GPUInfo]:
        """Get GPU info using nvitop (faster and more reliable)"""
        gpu_infos = []

        try:
            for device in self.devices:
                # Get memory info in MB
                memory_total_mb = bytes_to_mb(device.memory_total())
                memory_used_mb = bytes_to_mb(device.memory_used())
                memory_free_mb = bytes_to_mb(device.memory_free())

                # Get utilization %
                utilization = device.gpu_utilization()

                # Get temperature
                temperature = device.temperature()

                # Get power usage
                power_draw = device.power_draw()
                power_usage = power_draw // 1000 if power_draw else 0  # Convert mW to W

                # Get processes
                processes = []
                try:
                    proc_list = device.processes()
                    # Handle both list of process objects and list of PIDs
                    if proc_list and hasattr(proc_list[0], 'pid'):
                        # Process objects
                        for proc in proc_list:
                            processes.append({
                                'pid': proc.pid,
                                'name': proc.name() if hasattr(proc, 'name') else 'unknown',
                                'memory_mb': bytes_to_mb(proc.gpu_memory()) if hasattr(proc, 'gpu_memory') else 0
                            })
                    elif proc_list and isinstance(proc_list[0], int):
                        # Just PIDs
                        for pid in proc_list:
                            processes.append({
                                'pid': pid,
                                'name': f'pid_{pid}',
                                'memory_mb': 0
                            })
                except Exception:
                    # If process enumeration fails, continue without processes
                    pass

                gpu_infos.append(GPUInfo(
                    device_id=device.index,
                    name=device.name(),
                    memory_total=memory_total_mb,
                    memory_used=memory_used_mb,
                    memory_free=memory_free_mb,
                    utilization=utilization,
                    temperature=temperature,
                    power_usage=power_usage,
                    processes=processes
                ))

        except Exception as e:
            self.logger.error(f"Error getting GPU info with nvitop: {e}")
            # Fall back to nvidia-smi
            return self.get_gpu_info_nvidia_smi()

        return gpu_infos

    def get_gpu_info_nvidia_smi(self) -> List[GPUInfo]:
        """Get GPU info using nvidia-smi (fallback)"""
        if self.num_gpus == 0:
            return []

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free,"
                    "utilization.gpu,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )

            gpu_infos = []
            lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

            for line in lines:
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 8:
                    continue

                try:
                    gpu_infos.append(GPUInfo(
                        device_id=int(parts[0]),
                        name=parts[1],
                        memory_total=int(parts[2]),
                        memory_used=int(parts[3]),
                        memory_free=int(parts[4]),
                        utilization=int(parts[5]) if parts[5] != '[N/A]' else 0,
                        temperature=int(parts[6]) if parts[6] != '[N/A]' else 0,
                        power_usage=int(float(parts[7])) if parts[7] != '[N/A]' else 0,
                        processes=[]  # nvidia-smi doesn't provide process info easily
                    ))
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Failed to parse GPU info line: {line} - {e}")
                    continue

            return gpu_infos

        except Exception as e:
            self.logger.error(f"Failed to get GPU info with nvidia-smi: {e}")
            return []

    def get_gpu_info(self) -> List[GPUInfo]:
        """Get current information for all GPUs"""
        if self.num_gpus == 0:
            return []

        if self.use_nvitop:
            return self.get_gpu_info_nvitop()
        else:
            return self.get_gpu_info_nvidia_smi()

    def get_available_gpus(self, min_memory_mb: int = 1000) -> List[int]:
        """
        Get list of available GPUs based on utilization threshold and memory.

        Args:
            min_memory_mb: Minimum free memory required (MB)

        Returns:
            List of available GPU IDs
        """
        gpu_infos = self.get_gpu_info()
        available = []

        for gpu in gpu_infos:
            if gpu.utilization <= self.utilization_threshold and gpu.memory_free >= min_memory_mb:
                available.append(gpu.device_id)
                self.logger.debug(
                    f"GPU {gpu.device_id} available: {gpu.utilization}% util, "
                    f"{gpu.memory_free}MB free"
                )
            else:
                reason = []
                if gpu.utilization > self.utilization_threshold:
                    reason.append(f"high util ({gpu.utilization}%)")
                if gpu.memory_free < min_memory_mb:
                    reason.append(f"low memory ({gpu.memory_free}MB)")
                self.logger.debug(
                    f"GPU {gpu.device_id} unavailable: {', '.join(reason)}"
                )

        return available

    def select_best_gpu(self, min_memory_mb: int = 1000) -> Optional[int]:
        """
        Select the best available GPU based on lowest utilization and most free memory.

        Args:
            min_memory_mb: Minimum free memory required

        Returns:
            GPU ID of best available GPU, or None if none available
        """
        gpu_infos = self.get_gpu_info()

        # Filter available GPUs
        available_gpus = []
        for gpu in gpu_infos:
            if gpu.utilization <= self.utilization_threshold and gpu.memory_free >= min_memory_mb:
                available_gpus.append(gpu)

        if not available_gpus:
            self.logger.warning(
                f"No GPUs available (need <={self.utilization_threshold}% util, "
                f">={min_memory_mb}MB free)"
            )
            return None

        # Sort by utilization (ascending), then by free memory (descending)
        available_gpus.sort(key=lambda g: (g.utilization, -g.memory_free))

        best_gpu = available_gpus[0]
        self.logger.info(
            f"Selected GPU {best_gpu.device_id}: {best_gpu.utilization}% util, "
            f"{best_gpu.memory_free}MB free"
        )

        return best_gpu.device_id

    def get_memory_usage(self, gpu_id: int) -> Tuple[int, int, int]:
        """
        Get memory usage for a specific GPU.

        Returns:
            Tuple of (total_mb, used_mb, free_mb)
        """
        if self.use_nvitop and gpu_id < len(self.devices):
            device = self.devices[gpu_id]
            return (
                bytes_to_mb(device.memory_total()),
                bytes_to_mb(device.memory_used()),
                bytes_to_mb(device.memory_free())
            )
        else:
            # Use PyTorch as fallback
            with torch.cuda.device(gpu_id):
                total = bytes_to_mb(torch.cuda.get_device_properties(gpu_id).total_memory)
                allocated = bytes_to_mb(torch.cuda.memory_allocated())
                cached = bytes_to_mb(torch.cuda.memory_reserved())
                used = max(allocated, cached)
                free = max(0, total - used)
                return (total, used, free)


def test_enhanced_monitor():
    """Test the enhanced GPU monitor"""
    logging.basicConfig(level=logging.DEBUG)

    monitor = EnhancedGPUMonitor(utilization_threshold=80)

    # Get all GPU info
    gpu_infos = monitor.get_gpu_info()
    for gpu in gpu_infos:
        print(f"GPU {gpu.device_id} ({gpu.name}):")
        print(f"  Utilization: {gpu.utilization}%")
        print(f"  Memory: {gpu.memory_used}/{gpu.memory_total} MB")
        print(f"  Temperature: {gpu.temperature}°C")
        print(f"  Power: {gpu.power_usage}W")
        print(f"  Processes: {len(gpu.processes)}")

    # Get available GPUs
    available = monitor.get_available_gpus(min_memory_mb=1000)
    print(f"\nAvailable GPUs (util<=80%, mem>=1000MB): {available}")

    # Select best GPU
    best_gpu = monitor.select_best_gpu(min_memory_mb=1000)
    print(f"Best GPU: {best_gpu}")


if __name__ == "__main__":
    test_enhanced_monitor()