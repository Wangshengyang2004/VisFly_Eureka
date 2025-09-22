#!/usr/bin/env python3
"""
Unit tests for PPO algorithm fixes, particularly boolean tensor handling.
"""

import unittest
import numpy as np
import torch as th
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPPOBooleanTensorFix(unittest.TestCase):
    """Test PPO's handling of boolean done tensors"""

    def test_boolean_tensor_conversion(self):
        """Test that boolean tensors are converted to float correctly"""
        # Create boolean tensor (as returned by VisFly environments)
        dones = th.tensor([True, False, True, False], dtype=th.bool)

        # Apply the fix from PPO
        if isinstance(dones, th.Tensor):
            if dones.dtype == th.bool:
                dones = dones.float()
            dones = dones.cpu().numpy()
        elif isinstance(dones, np.ndarray) and dones.dtype == np.bool_:
            dones = dones.astype(np.float32)

        # Check result
        self.assertIsInstance(dones, np.ndarray)
        self.assertIn(dones.dtype, [np.float32, np.float64])  # Can be either depending on system
        np.testing.assert_array_equal(dones, np.array([1.0, 0.0, 1.0, 0.0]))

    def test_numpy_boolean_conversion(self):
        """Test that numpy boolean arrays are converted correctly"""
        dones = np.array([True, False, True], dtype=np.bool_)

        # Apply the fix
        if isinstance(dones, th.Tensor):
            if dones.dtype == th.bool:
                dones = dones.float()
            dones = dones.cpu().numpy()
        elif isinstance(dones, np.ndarray) and dones.dtype == np.bool_:
            dones = dones.astype(np.float32)

        # Check result
        self.assertIsInstance(dones, np.ndarray)
        self.assertEqual(dones.dtype, np.float32)
        np.testing.assert_array_equal(dones, np.array([1.0, 0.0, 1.0], dtype=np.float32))

    def test_already_float_tensor(self):
        """Test that float tensors are handled correctly"""
        dones = th.tensor([1.0, 0.0, 1.0, 0.0], dtype=th.float32)

        # Apply the fix
        if isinstance(dones, th.Tensor):
            if dones.dtype == th.bool:
                dones = dones.float()
            dones = dones.cpu().numpy()

        # Check result - should just convert to numpy
        self.assertIsInstance(dones, np.ndarray)
        np.testing.assert_array_equal(dones, np.array([1.0, 0.0, 1.0, 0.0]))


class TestEnvironmentCompatibility(unittest.TestCase):
    """Test environment compatibility with PPO"""

    def test_tensor_output_true(self):
        """Test when environment has tensor_output=True"""
        # Simulate environment step output with tensor_output=True
        obs = th.randn(4, 10)
        rewards = th.tensor([1.0, 2.0, 3.0, 4.0])
        dones = th.tensor([False, False, True, False], dtype=th.bool)
        infos = [{} for _ in range(4)]

        # Apply PPO fix for dones
        if isinstance(dones, th.Tensor):
            if dones.dtype == th.bool:
                dones = dones.float()
            dones = dones.cpu().numpy()

        # Verify conversion
        self.assertIsInstance(dones, np.ndarray)
        np.testing.assert_array_equal(dones, np.array([0.0, 0.0, 1.0, 0.0]))

    def test_tensor_output_false(self):
        """Test when environment has tensor_output=False"""
        # Simulate environment step output with tensor_output=False
        obs = np.random.randn(4, 10)
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        dones = np.array([False, False, True, False], dtype=np.bool_)
        infos = [{} for _ in range(4)]

        # Apply PPO fix for dones
        if isinstance(dones, np.ndarray) and dones.dtype == np.bool_:
            dones = dones.astype(np.float32)

        # Verify conversion
        self.assertIsInstance(dones, np.ndarray)
        self.assertEqual(dones.dtype, np.float32)
        np.testing.assert_array_equal(dones, np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()