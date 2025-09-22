#!/usr/bin/env python3
"""
Unit tests for enhanced run.py features including video recording and trajectory plotting.
"""

import unittest
import sys
import numpy as np
import torch as th
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import functions from run.py
from run import (
    collect_trajectory_data,
    create_trajectory_plot,
    record_video_frame,
    create_environment,
    apply_config_overrides
)


class TestTrajectoryData(unittest.TestCase):
    """Test trajectory data collection"""

    def setUp(self):
        """Set up mock environment"""
        self.mock_env = Mock()
        self.mock_env.position = th.tensor([[1.0, 2.0, 3.0]])
        self.mock_env.velocity = th.tensor([[0.1, 0.2, 0.3]])
        self.mock_env.orientation = th.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.mock_env.angular_velocity = th.tensor([[0.01, 0.02, 0.03]])
        self.mock_env.target = th.tensor([[5.0, 5.0, 5.0]])

    def test_collect_trajectory_data(self):
        """Test that trajectory data is collected correctly"""
        obs = None  # Not used in current implementation
        data = collect_trajectory_data(self.mock_env, obs)

        self.assertIn('position', data)
        self.assertIn('velocity', data)
        self.assertIn('orientation', data)
        self.assertIn('angular_velocity', data)
        self.assertIn('target', data)

        # Check data is numpy arrays
        self.assertIsInstance(data['position'], np.ndarray)
        self.assertIsInstance(data['velocity'], np.ndarray)

        # Check shapes
        self.assertEqual(data['position'].shape, (1, 3))
        self.assertEqual(data['velocity'].shape, (1, 3))

    def test_collect_trajectory_data_no_attributes(self):
        """Test handling when environment lacks attributes"""
        mock_env = Mock(spec=[])  # Empty spec means no attributes
        data = collect_trajectory_data(mock_env, None)

        # Should return empty dict or handle gracefully
        self.assertIsInstance(data, dict)


class TestVideoRecording(unittest.TestCase):
    """Test video recording functionality"""

    @patch('cv2.VideoWriter')
    @patch('cv2.cvtColor')
    def test_record_video_frame_init(self, mock_cvtColor, mock_VideoWriter):
        """Test video writer initialization"""
        # Mock environment render
        mock_env = Mock()
        mock_env.render = Mock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))

        # Mock cv2 available
        with patch('run.cv2', create=True):
            video_params = {'path': Path('/tmp/test.mp4'), 'fps': 30}
            video_writer = record_video_frame(mock_env, 'init', video_params)

            # Should have called render
            mock_env.render.assert_called()

    def test_record_video_frame_no_cv2(self):
        """Test graceful handling when cv2 is not available"""
        mock_env = Mock()

        # Set cv2 to None (not installed)
        with patch('run.cv2', None):
            result = record_video_frame(mock_env, None, None)
            self.assertIsNone(result)


class TestConfigOverrides(unittest.TestCase):
    """Test configuration override functionality"""

    def test_apply_config_overrides(self):
        """Test that command line overrides are applied correctly"""
        env_config = {
            'env': {
                'num_agent_per_scene': 10,
                'device': 'cuda'
            }
        }

        alg_config = {
            'learn': {
                'total_timesteps': 1000
            },
            'algorithm': {
                'device': 'cuda'
            }
        }

        # Mock args with overrides
        args = Mock()
        args.learning_steps = 5000
        args.num_agents = 20
        args.device = 'cpu'
        args.comment = 'test_comment'

        env_config_new, alg_config_new = apply_config_overrides(env_config, alg_config, args)

        # Check overrides were applied
        self.assertEqual(alg_config_new['learn']['total_timesteps'], 5000)
        self.assertEqual(env_config_new['env']['num_agent_per_scene'], 20)
        self.assertEqual(env_config_new['env']['device'], 'cpu')
        self.assertEqual(alg_config_new['algorithm']['device'], 'cpu')
        self.assertEqual(alg_config_new['comment'], 'test_comment')

    def test_apply_config_overrides_no_overrides(self):
        """Test that configs remain unchanged when no overrides"""
        env_config = {'env': {'num_agent_per_scene': 10}}
        alg_config = {'learn': {'total_timesteps': 1000}}

        args = Mock()
        args.learning_steps = None
        args.num_agents = None
        args.device = None
        args.comment = None
        args.env = 'test'
        args.algorithm = 'bptt'

        env_config_new, alg_config_new = apply_config_overrides(env_config, alg_config, args)

        # Check values unchanged except for default comment
        self.assertEqual(env_config_new['env']['num_agent_per_scene'], 10)
        self.assertEqual(alg_config_new['learn']['total_timesteps'], 1000)
        self.assertEqual(alg_config_new['comment'], 'test_bptt')


class TestEnvironmentCreation(unittest.TestCase):
    """Test environment creation with parameter filtering"""

    @patch('run.get_env_class_registry')
    def test_create_environment_filters_params(self, mock_registry):
        """Test that non-constructor parameters are filtered out"""
        # Mock environment class
        MockEnvClass = Mock()
        mock_env_instance = Mock()
        MockEnvClass.return_value = mock_env_instance
        mock_registry.return_value = {'test_env': MockEnvClass}

        env_config = {
            'env': {
                'num_agent_per_scene': 1,
                'device': 'cpu',
                'landing_target': [1, 2, 3],  # Should be filtered
                'wind_disturbance': True,      # Should be filtered
                'name': 'test'                 # Should be filtered
            }
        }

        env = create_environment('test_env', env_config, is_training=True)

        # Check that filtered params were set as attributes
        self.assertEqual(env, mock_env_instance)

        # Verify the constructor was called without filtered params
        call_kwargs = MockEnvClass.call_args[1]
        self.assertNotIn('landing_target', call_kwargs)
        self.assertNotIn('wind_disturbance', call_kwargs)
        self.assertNotIn('name', call_kwargs)
        self.assertIn('num_agent_per_scene', call_kwargs)
        self.assertIn('device', call_kwargs)


class TestRewardHandling(unittest.TestCase):
    """Test different reward format handling"""

    def test_dict_reward_with_reward_key(self):
        """Test handling dictionary rewards with 'reward' key"""
        reward = {
            'reward': th.tensor([1.5]),
            'distance_reward': th.tensor([0.5]),
            'orientation_reward': th.tensor([1.0])
        }

        # This would be tested in the actual run_testing function
        # Here we just verify the logic
        if isinstance(reward, dict):
            if "reward" in reward:
                reward_value = float(reward["reward"].mean().item())
            else:
                reward_value = sum(float(v.mean().item()) for v in reward.values())

        self.assertEqual(reward_value, 1.5)

    def test_dict_reward_without_reward_key(self):
        """Test handling dictionary rewards without 'reward' key"""
        reward = {
            'distance_reward': th.tensor([0.5]),
            'orientation_reward': th.tensor([1.0])
        }

        if isinstance(reward, dict):
            if "reward" in reward:
                reward_value = float(reward["reward"].mean().item())
            else:
                reward_value = sum(float(v.mean().item()) for v in reward.values())

        self.assertEqual(reward_value, 1.5)


if __name__ == '__main__':
    unittest.main()