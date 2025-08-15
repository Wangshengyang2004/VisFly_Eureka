"""
Navigation Environment Eureka Optimization Example

This example demonstrates the complete VisFly-Eureka integration pipeline
using the NavigationEnv for basic testing and validation.
"""

import sys
import os
import torch
import logging

# Add the project root to Python path
sys.path.insert(0, '/home/simonwsy/VisFly_Eureka')
sys.path.insert(0, '/home/simonwsy/VisFly_Eureka/VisFly')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_reward_injection():
    """Test basic reward injection functionality."""
    print("=== Testing Reward Injection ===")
    
    try:
        from VisFly.envs.NavigationEnv import NavigationEnv
        from eureka_visfly.reward_injection import inject_generated_reward
        from habitat_sim.sensor import SensorType
        
        # Create a minimal NavigationEnv with required sensors
        env = NavigationEnv(
            num_agent_per_scene=2,
            num_scene=1,
            visual=True,  # NavigationEnv requires visual=True
            max_episode_steps=32,
            device="cpu",
            sensor_kwargs=[
                {
                    "sensor_type": SensorType.DEPTH,
                    "uuid": "depth",
                    "resolution": [32, 32],
                }
            ]
        )
        
        # Test reward injection with a simple reward function
        test_reward_code = """
def get_reward(self):
    \"\"\"Simple test reward function\"\"\"
    # Return constant reward for all agents
    return torch.ones(self.num_agent) * 0.5
"""
        
        print(f"Environment created with {env.num_agent} agents")
        
        # Test injection
        success = inject_generated_reward(env, test_reward_code)
        print(f"Reward injection success: {success}")
        
        if success:
            # Test the injected reward
            env.reset()
            reward = env.get_reward()
            print(f"Reward output: {reward}")
            print(f"Reward shape: {reward.shape}")
            
            expected_shape = (env.num_agent,)
            if reward.shape == expected_shape:
                print("✓ Reward shape matches expected")
            else:
                print(f"✗ Reward shape mismatch: got {reward.shape}, expected {expected_shape}")
        
        return success
        
    except Exception as e:
        print(f"Reward injection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_llm_engine():
    """Test the mock LLM engine for development."""
    print("\n=== Testing Mock LLM Engine ===")
    
    try:
        from eureka_visfly.llm_engine import MockLLMEngine
        
        # Create mock LLM engine
        llm = MockLLMEngine(model="mock-gpt-4")
        
        # Test API connection
        connection_ok = llm.test_api_connection()
        print(f"Mock API connection: {connection_ok}")
        
        # Test reward generation
        task_description = "Navigate drone to target position while avoiding obstacles"
        context_info = {
            "environment_class": "NavigationEnv",
            "num_agents": 2,
            "max_episode_steps": 32,
            "sensors": [{"type": "depth", "uuid": "depth", "resolution": [64, 64]}]
        }
        
        reward_functions = llm.generate_reward_functions(
            task_description=task_description,
            context_info=context_info,
            samples=3
        )
        
        print(f"Generated {len(reward_functions)} mock reward functions")
        
        if reward_functions:
            print("First generated reward function:")
            print(reward_functions[0])
            return True
        else:
            print("No reward functions generated")
            return False
            
    except Exception as e:
        print(f"Mock LLM engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_navigation_env_creation():
    """Test NavigationEnv creation with different configurations."""
    print("\n=== Testing NavigationEnv Creation ===")
    
    try:
        from VisFly.envs.NavigationEnv import NavigationEnv
        from habitat_sim.sensor import SensorType
        
        # Test minimal configuration - NavigationEnv requires visual=True and depth sensors
        env_config = {
            "num_agent_per_scene": 1,
            "num_scene": 1,
            "visual": True,  # NavigationEnv requires visual=True
            "max_episode_steps": 16,
            "device": "cpu",
            "target": torch.tensor([[5.0, 0.0, 1.5]]),
            "sensor_kwargs": [
                {
                    "sensor_type": SensorType.DEPTH,
                    "uuid": "depth",
                    "resolution": [32, 32],  # Smaller resolution for faster testing
                }
            ],
            "scene_kwargs": {
                "path": "VisFly/datasets/visfly-beta/configs/scenes/empty_stage"  # Use empty scene
            }
        }
        
        env = NavigationEnv(**env_config)
        print(f"Created NavigationEnv with config: {env_config}")
        
        # Test environment reset
        obs = env.reset()
        print(f"Reset successful, observation keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
        
        # Test original reward function
        original_reward = env.get_reward()
        print(f"Original reward: {original_reward}")
        
        # Test environment step
        action = env.action_space.sample() if hasattr(env, 'action_space') else torch.zeros(4)
        obs, reward, done, info = env.step(action)
        print(f"Step successful, reward: {reward}")
        
        return True
        
    except Exception as e:
        print(f"NavigationEnv creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_pipeline():
    """Test the complete end-to-end pipeline with mock components."""
    print("\n=== Testing End-to-End Pipeline ===")
    
    try:
        from VisFly.envs.NavigationEnv import NavigationEnv
        from eureka_visfly import EurekaVisFly, OptimizationConfig, MockLLMEngine
        
        # Create environment configuration
        env_kwargs = {
            "num_agent_per_scene": 1,
            "num_scene": 1,
            "visual": True,  # NavigationEnv requires visual=True
            "max_episode_steps": 16,
            "target": torch.tensor([[3.0, 0.0, 1.0]]),
            "sensor_kwargs": [
                {
                    "sensor_type": "DEPTH",  # Use string instead of enum for simplicity
                    "uuid": "depth", 
                    "resolution": [32, 32],
                }
            ]
        }
        
        # Create optimization configuration
        opt_config = OptimizationConfig(
            iterations=1,  # Just one iteration for testing
            samples=2,     # Just two samples for testing
            training_steps=100,  # Minimal training for testing
            algorithm="bptt"
        )
        
        # Create Eureka controller with mock LLM
        eureka = EurekaVisFly(
            env_class=NavigationEnv,
            task_description="Navigate to target position efficiently",
            llm_config={"model": "mock-gpt-4"},  # This will use MockLLMEngine
            env_kwargs=env_kwargs,
            optimization_config=opt_config,
            device="cpu"
        )
        
        # Replace with mock LLM engine
        eureka.llm = MockLLMEngine()
        
        print("Created EurekaVisFly controller with mock LLM")
        
        # Test environment creation
        test_env = eureka.create_environment()
        print(f"Test environment created: {test_env.__class__.__name__}")
        
        # Test reward generation
        reward_functions = eureka.generate_reward_candidates(samples=2, iteration=0)
        print(f"Generated {len(reward_functions)} reward functions")
        
        if reward_functions:
            print("Testing first reward function:")
            test_env = eureka.create_environment()
            
            # Test reward injection
            from eureka_visfly.reward_injection import inject_generated_reward
            success = inject_generated_reward(test_env, reward_functions[0])
            
            if success:
                test_env.reset()
                reward = test_env.get_reward()
                print(f"✓ Injected reward works: {reward}")
                return True
            else:
                print("✗ Reward injection failed")
                return False
        else:
            print("✗ No reward functions generated")
            return False
            
    except Exception as e:
        print(f"End-to-end pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("VisFly-Eureka Integration Test")
    print("=" * 50)
    
    tests = [
        ("NavigationEnv Creation", test_navigation_env_creation),
        ("Reward Injection", test_reward_injection), 
        ("Mock LLM Engine", test_mock_llm_engine),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 30)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✓ {test_name}: PASSED")
            else:
                print(f"✗ {test_name}: FAILED")
                
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL" 
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Phase 1 implementation complete.")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    main()