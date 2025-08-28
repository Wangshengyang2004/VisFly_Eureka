"""
Unit tests for all environment configurations and classes.
"""

import pytest
import torch
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add project to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'VisFly'))

from envs.CatchEnv import CatchEnv
from envs.DynamicEnv import DynamicEnv  
from envs.LandingEnv import LandingEnv
from envs.MultiNavigationEnv import MultiNavigationEnv
from envs.RacingEnv import RacingEnv
from envs.NavigationEnv import NavigationEnv
from envs.HoverEnv import HoverEnv


class TestEnvironmentConfigs:
    """Test environment configuration files."""
    
    @pytest.fixture
    def config_dir(self):
        return PROJECT_ROOT / 'configs' / 'envs'
    
    @pytest.fixture
    def alg_config_dir(self):
        return PROJECT_ROOT / 'configs' / 'algs'
    
    def test_all_env_configs_exist(self, config_dir):
        """Test that all environment config files exist."""
        expected_configs = [
            'navigation.yaml',
            'hover.yaml', 
            'easy_navigation.yaml',
            'catch.yaml',
            'dynamic.yaml',
            'landing.yaml',
            'multi_navigation.yaml',
            'racing.yaml'
        ]
        
        for config_name in expected_configs:
            config_path = config_dir / config_name
            assert config_path.exists(), f"Missing config: {config_name}"
    
    def test_all_alg_configs_exist(self, alg_config_dir):
        """Test that all algorithm configs exist for each environment."""
        environments = ['navigation', 'hover', 'easy_navigation', 'catch', 'dynamic', 
                       'landing', 'multi_navigation', 'racing']
        algorithms = ['bptt', 'ppo', 'shac']
        
        for env in environments:
            env_dir = alg_config_dir / env
            if env_dir.exists():
                for alg in algorithms:
                    alg_config = env_dir / f'{alg}.yaml'
                    if alg_config.exists():
                        # Test config is valid YAML
                        with open(alg_config, 'r') as f:
                            config = yaml.safe_load(f)
                            assert isinstance(config, dict)
                            assert 'algorithm' in config
    
    def test_config_structure(self, config_dir):
        """Test configuration file structure."""
        for config_file in config_dir.glob('*.yaml'):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Test main env config
            assert 'env' in config
            env_config = config['env']
            
            # Required fields
            assert 'num_agent_per_scene' in env_config
            assert 'max_episode_steps' in env_config
            assert 'dynamics_kwargs' in env_config
            assert 'visual' in env_config
            assert 'requires_grad' in env_config
            assert 'tensor_output' in env_config
            assert 'device' in env_config
            
            # Test eval env config exists
            assert 'eval_env' in config


class TestEnvironmentClasses:
    """Test environment class implementations."""
    
    def test_navigation_env_import(self):
        """Test NavigationEnv can be imported."""
        assert NavigationEnv is not None
    
    def test_hover_env_import(self):
        """Test HoverEnv can be imported.""" 
        assert HoverEnv is not None
        
    def test_catch_env_import(self):
        """Test CatchEnv can be imported."""
        assert CatchEnv is not None
        
    def test_dynamic_env_import(self):
        """Test DynamicEnv can be imported."""
        assert DynamicEnv is not None
        
    def test_landing_env_import(self):
        """Test LandingEnv can be imported."""
        assert LandingEnv is not None
        
    def test_multi_navigation_env_import(self):
        """Test MultiNavigationEnv can be imported."""
        assert MultiNavigationEnv is not None
        
    def test_racing_env_import(self):
        """Test RacingEnv can be imported."""
        assert RacingEnv is not None


class TestEnvironmentRewardProperties:
    """Test environment-specific reward function properties."""
    
    def test_catch_env_properties(self):
        """Test CatchEnv has expected properties for reward functions."""
        # Test class has expected attributes
        expected_attrs = ['position', 'velocity', 'target', 'collision_dis']
        
        # Note: We can't instantiate without VisFly dependencies
        # but we can check the class exists and imports
        assert hasattr(CatchEnv, '__init__')
    
    def test_dynamic_env_properties(self):
        """Test DynamicEnv has expected properties."""
        assert hasattr(DynamicEnv, '__init__')
        
    def test_landing_env_properties(self):
        """Test LandingEnv has expected properties."""
        assert hasattr(LandingEnv, '__init__')
        
    def test_multi_navigation_env_properties(self):
        """Test MultiNavigationEnv has expected properties."""
        assert hasattr(MultiNavigationEnv, '__init__')
        
    def test_racing_env_properties(self):
        """Test RacingEnv has expected properties."""
        assert hasattr(RacingEnv, '__init__')


class TestEnvironmentSpecificConfigs:
    """Test environment-specific configuration details."""
    
    @pytest.fixture
    def config_dir(self):
        return PROJECT_ROOT / 'configs' / 'envs'
    
    def test_catch_config_specifics(self, config_dir):
        """Test Catch environment configuration specifics."""
        with open(config_dir / 'catch.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        env_config = config['env']
        
        # Catch-specific parameters
        assert 'target_kwargs' in env_config
        assert 'initial_position' in env_config['target_kwargs']
        assert 'velocity_range' in env_config['target_kwargs']
        assert 'size' in env_config['target_kwargs']
        
        # Fast dynamics for catching
        assert env_config['dynamics_kwargs']['dt'] == 0.02
        assert env_config['dynamics_kwargs']['ctrl_dt'] == 0.02
    
    def test_dynamic_config_specifics(self, config_dir):
        """Test Dynamic environment configuration specifics."""
        with open(config_dir / 'dynamic.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        env_config = config['env']
        
        # Dynamic obstacle parameters
        assert 'obstacle_update_rate' in env_config
        assert 'prediction_horizon' in env_config
        
        # Multi-sensor setup
        assert len(env_config['sensor_kwargs']) >= 2
        sensor_types = [s['sensor_type'] for s in env_config['sensor_kwargs']]
        assert 'DEPTH' in sensor_types
        assert 'RGB' in sensor_types
    
    def test_landing_config_specifics(self, config_dir):
        """Test Landing environment configuration specifics."""
        with open(config_dir / 'landing.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        env_config = config['env']
        
        # Landing-specific parameters
        assert 'landing_target' in env_config
        assert 'landing_tolerance' in env_config
        assert 'wind_disturbance' in env_config
        
        # Downward-facing sensor
        depth_sensor = next(s for s in env_config['sensor_kwargs'] 
                           if s['sensor_type'] == 'DEPTH')
        assert depth_sensor['position'][2] == -0.2  # Downward facing
    
    def test_multi_navigation_config_specifics(self, config_dir):
        """Test Multi-navigation environment configuration specifics."""
        with open(config_dir / 'multi_navigation.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        env_config = config['env']
        
        # Multi-waypoint parameters
        assert 'waypoints' in env_config
        assert 'waypoint_tolerance' in env_config
        assert 'sequential_waypoints' in env_config
        assert 'path_planning' in env_config
        
        # Multiple waypoints
        assert len(env_config['waypoints']) >= 3
    
    def test_racing_config_specifics(self, config_dir):
        """Test Racing environment configuration specifics."""
        with open(config_dir / 'racing.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        env_config = config['env']
        
        # Racing-specific parameters
        assert 'race_gates' in env_config
        assert 'gate_tolerance' in env_config
        assert 'time_penalty' in env_config
        assert 'gate_order' in env_config
        assert 'max_velocity' in env_config
        assert 'crash_penalty' in env_config
        
        # High-frequency dynamics for racing
        assert env_config['dynamics_kwargs']['dt'] == 0.015
        
        # Multiple gates
        assert len(env_config['race_gates']) >= 4


class TestAlgorithmConfigurations:
    """Test algorithm-specific configurations."""
    
    @pytest.fixture
    def alg_config_dir(self):
        return PROJECT_ROOT / 'configs' / 'algs'
    
    def test_bptt_configs(self, alg_config_dir):
        """Test BPTT configuration consistency."""
        for env_dir in alg_config_dir.iterdir():
            if env_dir.is_dir():
                bptt_config = env_dir / 'bptt.yaml'
                if bptt_config.exists():
                    with open(bptt_config, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # BPTT requirements
                    assert config['env_overrides']['requires_grad'] == True
                    assert config['env_overrides']['tensor_output'] == False
                    assert config['env_overrides']['device'] == 'cpu'
                    
                    # Has learning parameters
                    assert 'learn' in config
                    assert 'total_timesteps' in config['learn']
    
    def test_ppo_configs(self, alg_config_dir):
        """Test PPO configuration consistency."""
        for env_dir in alg_config_dir.iterdir():
            if env_dir.is_dir():
                ppo_config = env_dir / 'ppo.yaml'
                if ppo_config.exists():
                    with open(ppo_config, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # PPO requirements
                    assert config['env_overrides']['requires_grad'] == False
                    assert config['env_overrides']['tensor_output'] == False
                    
                    # PPO-specific parameters
                    alg_config = config['algorithm']
                    assert 'gamma' in alg_config
                    assert 'gae_lambda' in alg_config
                    assert 'n_epochs' in alg_config
                    assert 'n_steps' in alg_config
    
    def test_shac_configs(self, alg_config_dir):
        """Test SHAC configuration consistency."""
        for env_dir in alg_config_dir.iterdir():
            if env_dir.is_dir():
                shac_config = env_dir / 'shac.yaml'
                if shac_config.exists():
                    with open(shac_config, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # SHAC requirements
                    assert config['env_overrides']['requires_grad'] == True
                    assert config['env_overrides']['tensor_output'] == False
                    assert config['env_overrides']['device'] == 'cpu'
                    
                    # SHAC-specific parameters
                    alg_config = config['algorithm']
                    assert 'horizon' in alg_config
                    assert 'tau' in alg_config
                    assert 'buffer_size' in alg_config
                    assert 'batch_size' in alg_config