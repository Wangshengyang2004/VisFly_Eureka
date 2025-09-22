"""
Unit tests for EnhancedGPUMonitor with nvitop support (pytest style)
"""

import pytest
from unittest.mock import Mock, patch

from quadro_llm.utils.gpu_monitor import (
    EnhancedGPUMonitor,
    GPUInfo,
)


@pytest.fixture
def mock_gpu_infos():
    """Sample GPU states for testing."""
    return [
        GPUInfo(
            device_id=0,
            name="NVIDIA GeForce RTX 3090",
            memory_total=24576,
            memory_used=8192,
            memory_free=16384,
            utilization=30,
            temperature=55,
            power_usage=150,
            processes=[{'pid': 1234, 'name': 'python', 'memory_mb': 4096}],
        ),
        GPUInfo(
            device_id=1,
            name="NVIDIA GeForce RTX 3090",
            memory_total=24576,
            memory_used=20480,
            memory_free=4096,
            utilization=85,
            temperature=75,
            power_usage=300,
            processes=[{'pid': 5678, 'name': 'training.py', 'memory_mb': 18432}],
        ),
        GPUInfo(
            device_id=2,
            name="NVIDIA GeForce RTX 3090",
            memory_total=24576,
            memory_used=2048,
            memory_free=22528,
            utilization=10,
            temperature=45,
            power_usage=100,
            processes=[],
        ),
    ]


@patch('quadro_llm.utils.gpu_monitor.NVITOP_AVAILABLE', False)
@patch('torch.cuda.is_available')
@patch('torch.cuda.device_count')
def test_initialization_with_cuda(mock_device_count, mock_is_available):
    mock_is_available.return_value = True
    mock_device_count.return_value = 3

    monitor = EnhancedGPUMonitor(utilization_threshold=80)

    assert monitor.num_gpus == 3
    assert monitor.utilization_threshold == 80


@patch('torch.cuda.is_available')
def test_initialization_without_cuda(mock_is_available):
    mock_is_available.return_value = False

    monitor = EnhancedGPUMonitor()

    assert monitor.num_gpus == 0
    assert not monitor.use_nvitop


@patch('quadro_llm.utils.gpu_monitor.NVITOP_AVAILABLE', False)
@patch('torch.cuda.is_available')
@patch('torch.cuda.device_count')
def test_get_available_gpus_with_utilization_filter(mock_device_count, mock_is_available, mock_gpu_infos):
    mock_is_available.return_value = True
    mock_device_count.return_value = 3

    monitor = EnhancedGPUMonitor(utilization_threshold=80)

    with patch.object(monitor, 'get_gpu_info', return_value=mock_gpu_infos):
        available = monitor.get_available_gpus(min_memory_mb=5000)
        assert available == [0, 2]


@patch('quadro_llm.utils.gpu_monitor.Device', Mock())
@patch('quadro_llm.utils.gpu_monitor.NVITOP_AVAILABLE', False)
@patch('torch.cuda.is_available')
@patch('torch.cuda.device_count')
def test_get_available_gpus_with_memory_filter(mock_device_count, mock_is_available, mock_gpu_infos):
    mock_is_available.return_value = True
    mock_device_count.return_value = 3

    monitor = EnhancedGPUMonitor(utilization_threshold=90)

    with patch.object(monitor, 'get_gpu_info', return_value=mock_gpu_infos):
        available = monitor.get_available_gpus(min_memory_mb=20000)
        assert available == [2]


@patch('torch.cuda.is_available')
@patch('torch.cuda.device_count')
def test_select_best_gpu(mock_device_count, mock_is_available, mock_gpu_infos):
    mock_is_available.return_value = True
    mock_device_count.return_value = 3

    monitor = EnhancedGPUMonitor(utilization_threshold=80)

    with patch.object(monitor, 'get_gpu_info', return_value=mock_gpu_infos):
        best_gpu = monitor.select_best_gpu(min_memory_mb=1000)
        assert best_gpu == 2


@patch('torch.cuda.is_available')
@patch('torch.cuda.device_count')
def test_select_best_gpu_no_available(mock_device_count, mock_is_available, mock_gpu_infos):
    mock_is_available.return_value = True
    mock_device_count.return_value = 3

    monitor = EnhancedGPUMonitor(utilization_threshold=5)

    with patch.object(monitor, 'get_gpu_info', return_value=mock_gpu_infos):
        best_gpu = monitor.select_best_gpu(min_memory_mb=1000)
        assert best_gpu is None


@patch('quadro_llm.utils.gpu_monitor.NVITOP_AVAILABLE', True)
@patch('torch.cuda.is_available')
@patch('torch.cuda.device_count')
def test_nvitop_integration(mock_device_count, mock_is_available):
    mock_is_available.return_value = True
    mock_device_count.return_value = 1

    # Mock nvitop Device
    mock_device = Mock()
    mock_device.index = 0
    mock_device.name.return_value = "NVIDIA GeForce RTX 3090"
    mock_device.memory_total.return_value = 24576 * 1024 * 1024
    mock_device.memory_used.return_value = 8192 * 1024 * 1024
    mock_device.memory_free.return_value = 16384 * 1024 * 1024
    mock_device.gpu_utilization.return_value = 30
    mock_device.temperature.return_value = 55
    mock_device.power_draw.return_value = 150000
    mock_device.processes.return_value = []

    with patch('quadro_llm.utils.gpu_monitor.Device', return_value=mock_device):
        monitor = EnhancedGPUMonitor()
        if monitor.use_nvitop:
            gpu_infos = monitor.get_gpu_info_nvitop()
            assert len(gpu_infos) == 1
            assert gpu_infos[0].device_id == 0
            assert gpu_infos[0].utilization == 30
            assert gpu_infos[0].memory_free == 16384


@patch('quadro_llm.utils.gpu_monitor.Device', Mock())
@patch('quadro_llm.utils.gpu_monitor.NVITOP_AVAILABLE', False)
@patch('torch.cuda.is_available')
@patch('torch.cuda.device_count')
@patch('subprocess.run')
def test_nvidia_smi_fallback(mock_subprocess, mock_device_count, mock_is_available):
    mock_is_available.return_value = True
    mock_device_count.return_value = 1

    mock_result = Mock()
    mock_result.stdout = "0, NVIDIA GeForce RTX 3090, 24576, 8192, 16384, 30, 55, 150.5"
    mock_subprocess.return_value = mock_result

    monitor = EnhancedGPUMonitor()
    monitor.use_nvitop = False
    gpu_infos = monitor.get_gpu_info_nvidia_smi()

    assert len(gpu_infos) == 1
    assert gpu_infos[0].device_id == 0
    assert gpu_infos[0].utilization == 30
    assert gpu_infos[0].memory_free == 16384


@patch('quadro_llm.utils.gpu_monitor.Device', Mock())
@patch('quadro_llm.utils.gpu_monitor.NVITOP_AVAILABLE', False)
@patch('torch.cuda.is_available')
@patch('torch.cuda.device_count')
@patch('subprocess.run')
def test_nvidia_smi_parse_error_handling(mock_subprocess, mock_device_count, mock_is_available):
    mock_is_available.return_value = True
    mock_device_count.return_value = 1

    mock_result = Mock()
    mock_result.stdout = "0, NVIDIA GeForce RTX 3090, 24576, 8192, 16384, [N/A], [N/A], [N/A]"
    mock_subprocess.return_value = mock_result

    monitor = EnhancedGPUMonitor()
    monitor.use_nvitop = False
    gpu_infos = monitor.get_gpu_info_nvidia_smi()

    assert len(gpu_infos) == 1
    assert gpu_infos[0].utilization == 0
    assert gpu_infos[0].temperature == 0
    assert gpu_infos[0].power_usage == 0


@patch('quadro_llm.utils.gpu_monitor.Device', Mock())
@patch('quadro_llm.utils.gpu_monitor.NVITOP_AVAILABLE', False)
@patch('torch.cuda.is_available')
@patch('torch.cuda.device_count')
def test_get_memory_usage_with_nvitop(mock_device_count, mock_is_available):
    mock_is_available.return_value = True
    mock_device_count.return_value = 1

    mock_device = Mock()
    mock_device.memory_total.return_value = 24576 * 1024 * 1024
    mock_device.memory_used.return_value = 8192 * 1024 * 1024
    mock_device.memory_free.return_value = 16384 * 1024 * 1024

    monitor = EnhancedGPUMonitor()
    monitor.use_nvitop = True
    monitor.devices = [mock_device]

    total, used, free = monitor.get_memory_usage(0)
    assert total == 24576
    assert used == 8192
    assert free == 16384


@patch('quadro_llm.utils.gpu_monitor.NVITOP_AVAILABLE', False)
@patch('torch.cuda.is_available')
@patch('torch.cuda.device_count')
@patch('torch.cuda.get_device_properties')
@patch('torch.cuda.memory_allocated')
@patch('torch.cuda.memory_reserved')
def test_get_memory_usage_pytorch_fallback(mock_reserved, mock_allocated, mock_props, mock_device_count, mock_is_available):
    mock_is_available.return_value = True
    mock_device_count.return_value = 1

    mock_props.return_value = Mock(total_memory=24576 * 1024 * 1024)
    mock_allocated.return_value = 6144 * 1024 * 1024
    mock_reserved.return_value = 8192 * 1024 * 1024

    monitor = EnhancedGPUMonitor()
    monitor.use_nvitop = False

    total, used, free = monitor.get_memory_usage(0)
    assert total == 24576
    assert used == 8192
    assert free == 16384


def test_no_cuda_available():
    with patch('torch.cuda.is_available', return_value=False):
        monitor = EnhancedGPUMonitor()
        assert monitor.num_gpus == 0
        assert monitor.get_gpu_info() == []
        assert monitor.get_available_gpus() == []
        assert monitor.select_best_gpu() is None


@patch('torch.cuda.is_available')
@patch('torch.cuda.device_count')
def test_multi_gpu_load_balancing(mock_device_count, mock_is_available):
    mock_is_available.return_value = True
    mock_device_count.return_value = 4

    gpu_scenarios = [
        GPUInfo(0, "GPU0", 24576, 20000, 4576, 75, 70, 250, []),
        GPUInfo(1, "GPU1", 24576, 8192, 16384, 85, 65, 200, []),
        GPUInfo(2, "GPU2", 24576, 4096, 20480, 20, 50, 150, []),
        GPUInfo(3, "GPU3", 24576, 12288, 12288, 50, 60, 180, []),
    ]

    monitor = EnhancedGPUMonitor(utilization_threshold=80)

    with patch.object(monitor, 'get_gpu_info', return_value=gpu_scenarios):
        best = monitor.select_best_gpu(min_memory_mb=10000)
        assert best == 2

        available = monitor.get_available_gpus(min_memory_mb=1000)
        assert 1 not in available
        assert 2 in available


@patch('torch.cuda.is_available')
@patch('torch.cuda.device_count')
def test_all_gpus_busy_scenario(mock_device_count, mock_is_available):
    mock_is_available.return_value = True
    mock_device_count.return_value = 2

    busy_gpus = [
        GPUInfo(0, "GPU0", 24576, 23000, 1576, 95, 80, 350, []),
        GPUInfo(1, "GPU1", 24576, 22000, 2576, 90, 78, 340, []),
    ]

    monitor = EnhancedGPUMonitor(utilization_threshold=80)

    with patch.object(monitor, 'get_gpu_info', return_value=busy_gpus):
        available = monitor.get_available_gpus(min_memory_mb=1000)
        assert available == []

        best = monitor.select_best_gpu(min_memory_mb=1000)
        assert best is None
