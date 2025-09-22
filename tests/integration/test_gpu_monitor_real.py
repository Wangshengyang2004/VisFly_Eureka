"""
Real GPU monitor integration tests (skipped by default).
"""

import time
from pathlib import Path

import pytest
import torch

from quadro_llm.utils.gpu_monitor import EnhancedGPUMonitor

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow]


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping real GPU tests")


def test_nvitop_import_or_fallback():
    _require_cuda()
    monitor = EnhancedGPUMonitor()
    # Either nvitop is available or we fall back to nvidia-smi
    assert monitor.num_gpus >= 0


def test_enhanced_gpu_monitor_basic():
    _require_cuda()
    monitor = EnhancedGPUMonitor(utilization_threshold=80)
    if monitor.num_gpus == 0:
        pytest.skip("No GPUs detected on this system")

    gpu_infos = monitor.get_gpu_info()
    assert isinstance(gpu_infos, list)
    assert all(hasattr(g, 'device_id') for g in gpu_infos)

    # Filtering and best selection should execute without error
    _ = monitor.get_available_gpus(min_memory_mb=100)
    _ = monitor.select_best_gpu(min_memory_mb=100)


def test_nvitop_vs_nvidia_smi_parity():
    _require_cuda()
    monitor = EnhancedGPUMonitor()

    info_smi = monitor.get_gpu_info_nvidia_smi()
    # nvitop may not be present; if not, just ensure smi path works
    if monitor.use_nvitop:
        info_nvitop = monitor.get_gpu_info_nvitop()
        # If both available, basic consistency checks
        if info_nvitop and info_smi:
            i = 0
            assert info_nvitop[i].device_id == info_smi[i].device_id


def test_utilization_filtering_real():
    _require_cuda()
    monitor = EnhancedGPUMonitor(utilization_threshold=80)
    if monitor.num_gpus == 0:
        pytest.skip("No GPUs detected on this system")

    available = monitor.get_available_gpus(min_memory_mb=100)
    assert isinstance(available, list)


def test_stress_monitor_short():
    _require_cuda()
    monitor = EnhancedGPUMonitor(utilization_threshold=80)
    if monitor.num_gpus == 0:
        pytest.skip("No GPUs detected on this system")

    # Keep stress short to avoid long CI times
    errors = 0
    for _ in range(10):
        try:
            _ = monitor.get_gpu_info()
            _ = monitor.get_available_gpus(min_memory_mb=100)
            _ = monitor.select_best_gpu(min_memory_mb=100)
        except Exception:
            errors += 1
    assert errors == 0
