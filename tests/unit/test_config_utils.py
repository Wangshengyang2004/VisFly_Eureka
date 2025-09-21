"""
Unit tests for configuration utilities.
"""

import pytest
import yaml
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from quadro_llm.utils.config_utils import ConfigManager


class TestConfigManager:
    """Test configuration management utilities."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            
            # Create sample configs
            env_dir = config_dir / 'envs'
            alg_dir = config_dir / 'algs'
            env_dir.mkdir()
            alg_dir.mkdir()
            
            # Sample environment config
            env_config = {
                'env': {
                    'num_agent_per_scene': 100,
                    'max_episode_steps': 256,
                    'visual': True,
                    'requires_grad': True,
                    'tensor_output': True,
                    'device': 'cpu'
                },
                'eval_env': {
                    'num_agent_per_scene': 1,
                    'visual': True
                }
            }
            
            with open(env_dir / 'test_env.yaml', 'w') as f:
                yaml.dump(env_config, f)
            
            # Sample algorithm config
            alg_config = {
                'algorithm': {
                    'policy': 'MultiInputPolicy',
                    'learning_rate': 0.001,
                    'gamma': 0.99
                },
                'env_overrides': {
                    'requires_grad': True,
                    'device': 'cpu'
                },
                'learn': {
                    'total_timesteps': 10000
                }
            }
            
            test_alg_dir = alg_dir / 'test_env'
            test_alg_dir.mkdir()
            with open(test_alg_dir / 'bptt.yaml', 'w') as f:
                yaml.dump(alg_config, f)
            
            yield config_dir
    
    def test_load_env_config(self, temp_config_dir):
        """Test loading environment configuration."""
        manager = ConfigManager(str(temp_config_dir))
        
        config = manager.load_env_config('test_env')
        
        assert config is not None
        assert 'env' in config
        assert 'eval_env' in config
        assert config['env']['num_agent_per_scene'] == 100
    
    def test_load_alg_config(self, temp_config_dir):
        """Test loading algorithm configuration."""
        manager = ConfigManager(str(temp_config_dir))
        
        config = manager.load_alg_config('test_env', 'bptt')
        
        assert config is not None
        assert 'algorithm' in config
        assert 'env_overrides' in config
        assert config['algorithm']['learning_rate'] == 0.001
    
    def test_merge_configs(self, temp_config_dir):
        """Test merging environment and algorithm configs."""
        manager = ConfigManager(str(temp_config_dir))
        
        merged = manager.get_merged_config('test_env', 'bptt')
        
        assert merged is not None
        assert 'env' in merged
        assert 'algorithm' in merged
        
        # Check override was applied
        assert merged['env']['requires_grad'] == True
        assert merged['env']['device'] == 'cpu'
    
    def test_list_available_envs(self, temp_config_dir):
        """Test listing available environments."""
        manager = ConfigManager(str(temp_config_dir))
        
        envs = manager.list_available_envs()
        
        assert 'test_env' in envs
    
    def test_list_available_algs(self, temp_config_dir):
        """Test listing available algorithms for environment."""
        manager = ConfigManager(str(temp_config_dir))
        
        algs = manager.list_available_algs('test_env')
        
        assert 'bptt' in algs
    
    def test_validate_config(self, temp_config_dir):
        """Test configuration validation."""
        manager = ConfigManager(str(temp_config_dir))
        
        config = manager.load_env_config('test_env')
        
        # Should not raise exception for valid config
        manager.validate_env_config(config)
        
        # Test invalid config
        invalid_config = {'env': {}}  # Missing required fields
        
        with pytest.raises(ValueError):
            manager.validate_env_config(invalid_config)
    
    def test_config_not_found(self, temp_config_dir):
        """Test handling of missing configuration files."""
        manager = ConfigManager(str(temp_config_dir))
        
        # Non-existent environment
        config = manager.load_env_config('nonexistent')
        assert config is None
        
        # Non-existent algorithm
        config = manager.load_alg_config('test_env', 'nonexistent')
        assert config is None
    
    def test_config_template_generation(self, temp_config_dir):
        """Test generation of configuration templates."""
        manager = ConfigManager(str(temp_config_dir))
        
        # Generate environment template
        env_template = manager.generate_env_template('new_env')
        
        assert isinstance(env_template, dict)
        assert 'env' in env_template
        assert 'eval_env' in env_template
        
        # Generate algorithm template
        alg_template = manager.generate_alg_template('new_env', 'ppo')
        
        assert isinstance(alg_template, dict)
        assert 'algorithm' in alg_template
        assert 'env_overrides' in alg_template


class TestConfigValidation:
    """Test configuration validation logic."""
    
    def test_env_config_required_fields(self):
        """Test environment configuration required field validation."""
        manager = ConfigManager('/tmp')  # Path doesn't matter for validation
        
        # Valid config
        valid_config = {
            'env': {
                'num_agent_per_scene': 100,
                'max_episode_steps': 256,
                'visual': True,
                'requires_grad': True,
                'tensor_output': True,
                'device': 'cpu',
                'dynamics_kwargs': {'dt': 0.03}
            },
            'eval_env': {
                'num_agent_per_scene': 1
            }
        }
        
        # Should not raise
        manager.validate_env_config(valid_config)
        
        # Missing required field
        invalid_config = dict(valid_config)
        del invalid_config['env']['num_agent_per_scene']
        
        with pytest.raises(ValueError, match="Missing required field"):
            manager.validate_env_config(invalid_config)
    
    def test_alg_config_validation(self):
        """Test algorithm configuration validation."""
        manager = ConfigManager('/tmp')
        
        # Valid BPTT config
        valid_bptt = {
            'algorithm': {
                'policy': 'MultiInputPolicy',
                'learning_rate': 0.001,
                'horizon': 96
            },
            'env_overrides': {
                'requires_grad': True,
                'tensor_output': False,
                'device': 'cpu'
            },
            'learn': {
                'total_timesteps': 10000
            }
        }
        
        manager.validate_alg_config(valid_bptt, 'bptt')
        
        # Invalid BPTT config (wrong requires_grad)
        invalid_bptt = dict(valid_bptt)
        invalid_bptt['env_overrides']['requires_grad'] = False
        
        with pytest.raises(ValueError, match="BPTT requires requires_grad=True"):
            manager.validate_alg_config(invalid_bptt, 'bptt')
    
    def test_device_validation(self):
        """Test device configuration validation."""
        manager = ConfigManager('/tmp')
        
        valid_config = {
            'env': {
                'device': 'cpu',
                'num_agent_per_scene': 100,
                'max_episode_steps': 256,
                'visual': True,
                'requires_grad': True,
                'tensor_output': True,
                'dynamics_kwargs': {'dt': 0.03}
            },
            'eval_env': {'num_agent_per_scene': 1}
        }
        
        # Valid devices
        for device in ['cpu', 'cuda', 'cuda:0']:
            config = dict(valid_config)
            config['env']['device'] = device
            manager.validate_env_config(config)
        
        # Invalid device
        invalid_config = dict(valid_config)
        invalid_config['env']['device'] = 'invalid_device'
        
        with pytest.raises(ValueError, match="Invalid device"):
            manager.validate_env_config(invalid_config)


class TestConfigCompatibility:
    """Test configuration compatibility checks."""
    
    def test_algorithm_env_compatibility(self):
        """Test algorithm-environment compatibility checks."""
        manager = ConfigManager('/tmp')
        
        # BPTT with tensor environment
        bptt_config = {
            'algorithm': {'policy': 'MultiInputPolicy'},
            'env_overrides': {
                'requires_grad': True,
                'tensor_output': False,
                'device': 'cpu'
            }
        }
        
        env_config = {
            'env': {
                'num_agent_per_scene': 100,
                'tensor_output': True,  # Will be overridden
                'requires_grad': False  # Will be overridden
            }
        }
        
        merged = manager._apply_overrides(env_config, bptt_config)
        
        # Check overrides were applied
        assert merged['env']['requires_grad'] == True
        assert merged['env']['tensor_output'] == False
    
    def test_sensor_config_validation(self):
        """Test sensor configuration validation."""
        manager = ConfigManager('/tmp')
        
        # Valid sensor config
        valid_config = {
            'env': {
                'num_agent_per_scene': 100,
                'max_episode_steps': 256,
                'visual': True,
                'requires_grad': True,
                'tensor_output': True,
                'device': 'cpu',
                'dynamics_kwargs': {'dt': 0.03},
                'sensor_kwargs': [
                    {
                        'sensor_type': 'DEPTH',
                        'uuid': 'depth',
                        'resolution': [64, 64],
                        'position': [0, 0.2, 0]
                    }
                ]
            },
            'eval_env': {'num_agent_per_scene': 1}
        }
        
        manager.validate_env_config(valid_config)
        
        # Invalid sensor config (missing required fields)
        invalid_config = dict(valid_config)
        invalid_config['env']['sensor_kwargs'] = [
            {'sensor_type': 'DEPTH'}  # Missing uuid, resolution
        ]
        
        with pytest.raises(ValueError, match="Invalid sensor configuration"):
            manager.validate_env_config(invalid_config)
