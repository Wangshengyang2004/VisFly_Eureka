"""
Real Navigation Environment Optimization Example

This example demonstrates the complete VisFly-Eureka integration pipeline
using the real API and NavigationEnv for actual reward function optimization.
"""

import sys
import os
import torch
import logging

# Add project paths
sys.path.insert(0, '/home/simonwsy/VisFly_Eureka')
sys.path.insert(0, '/home/simonwsy/VisFly_Eureka/VisFly')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_simple_nav_env():
    """Create a simple navigation environment for testing."""
    
    class SimpleNavigationEnv:
        """Simplified navigation environment for real API testing."""
        
        def __init__(self, num_agent_per_scene=1, device="cpu", target=None, **kwargs):
            self.num_agent = num_agent_per_scene
            self.device = device
            self.target = target if target is not None else torch.tensor([[5.0, 0.0, 1.5]])
            self.position = torch.zeros(self.num_agent, 3)
            self.velocity = torch.zeros(self.num_agent, 3)
            self.orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * self.num_agent)  # Quaternion
            self.angular_velocity = torch.zeros(self.num_agent, 3)
            self.max_episode_steps = kwargs.get("max_episode_steps", 256)
            self._step_count = 0
            
            # Simulate sensor data
            self.sensor_obs = {
                "depth": torch.ones(self.num_agent, 64, 64) * 2.0,  # 2 meters depth
                "rgb": torch.zeros(self.num_agent, 64, 64, 3)  # Empty RGB
            }
            
            # Collision simulation
            self.collision_dis = torch.ones(self.num_agent) * 2.0
            
        def reset(self):
            """Reset environment to initial state."""
            self.position = torch.randn(self.num_agent, 3) * 0.5  # Random start position
            self.velocity = torch.zeros(self.num_agent, 3)
            self.orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * self.num_agent)
            self.angular_velocity = torch.zeros(self.num_agent, 3)
            self._step_count = 0
            
            # Update sensor simulation based on position
            distance_to_obstacles = torch.norm(self.position, dim=1, keepdim=True)
            self.sensor_obs["depth"] = torch.clamp(distance_to_obstacles.unsqueeze(-1).expand(-1, 64, 64), min=0.1, max=5.0)
            self.collision_dis = distance_to_obstacles.squeeze()
            
            state = torch.cat([
                self.target - self.position,  # Relative target position
                self.orientation,             # Quaternion orientation  
                self.velocity,                # Velocity
                self.angular_velocity         # Angular velocity
            ], dim=1)
            
            return {"state": state, "depth": self.sensor_obs["depth"]}
            
        def step(self, action):
            """Simulate one environment step."""
            # Simple physics simulation
            self.velocity += action * 0.1
            self.position += self.velocity * 0.02
            self._step_count += 1
            
            # Update sensor simulation
            distance_to_obstacles = torch.norm(self.position, dim=1, keepdim=True)
            self.sensor_obs["depth"] = torch.clamp(distance_to_obstacles.unsqueeze(-1).expand(-1, 64, 64), min=0.1, max=5.0)
            self.collision_dis = distance_to_obstacles.squeeze()
            
            # Get observation
            state = torch.cat([
                self.target - self.position,
                self.orientation,
                self.velocity, 
                self.angular_velocity
            ], dim=1)
            
            obs = {"state": state, "depth": self.sensor_obs["depth"]}
            
            # Get reward from current reward function
            reward = self.get_reward()
            
            # Simple termination condition
            distance_to_target = torch.norm(self.position - self.target, dim=1)
            done = (distance_to_target < 0.5) | (self._step_count >= self.max_episode_steps)
            
            return obs, reward, done.any(), {}
            
        def get_reward(self):
            """Default reward function (will be replaced by generated ones)."""
            # Simple distance-based reward
            distance = torch.norm(self.position - self.target, dim=1)
            return -distance * 0.1
            
        def get_success(self):
            """Check if navigation was successful."""
            distance = torch.norm(self.position - self.target, dim=1)
            return distance < 0.5
    
    return SimpleNavigationEnv


def test_real_api_connection():
    """Test connection to the real API."""
    print("=== Testing Real API Connection ===")
    
    try:
        from eureka_visfly.llm_engine import LLMEngine
        from config import LLM_CONFIG
        
        # Create LLM engine with real API config
        llm = LLMEngine(**LLM_CONFIG)
        
        # Test connection
        success = llm.test_api_connection()
        print(f"API connection test: {'✓ SUCCESS' if success else '✗ FAILED'}")
        
        return success
        
    except Exception as e:
        print(f"✗ API connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_reward_generation():
    """Test real reward function generation."""
    print("\n=== Testing Real Reward Generation ===")
    
    try:
        from eureka_visfly.llm_engine import LLMEngine
        from config import LLM_CONFIG
        
        # Create LLM engine
        llm = LLMEngine(**LLM_CONFIG)
        
        # Test reward generation
        task_description = "Navigate drone to target position while avoiding obstacles using depth sensor"
        context_info = {
            "environment_class": "NavigationEnv",
            "num_agents": 1,
            "max_episode_steps": 256,
            "sensors": [{"type": "depth", "uuid": "depth", "resolution": [64, 64]}]
        }
        
        print(f"Generating reward functions for: {task_description}")
        print("This may take a few moments...")
        
        reward_functions = llm.generate_reward_functions(
            task_description=task_description,
            context_info=context_info,
            samples=3
        )
        
        print(f"✓ Generated {len(reward_functions)} reward functions")
        
        if reward_functions:
            print("\nFirst generated reward function:")
            print("-" * 60)
            print(reward_functions[0])
            print("-" * 60)
            
            # Validate the reward function
            for i, reward_code in enumerate(reward_functions):
                valid = llm.validate_reward_function(reward_code)
                print(f"Reward function {i+1} validation: {'✓ VALID' if valid else '✗ INVALID'}")
        
        return len(reward_functions) > 0
        
    except Exception as e:
        print(f"✗ Reward generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_integration():
    """Test the complete integration with real API."""
    print("\n=== Testing Real Integration ===")
    
    try:
        from eureka_visfly import EurekaVisFly, OptimizationConfig
        from config import LLM_CONFIG
        
        # Create environment class
        SimpleNavEnv = create_simple_nav_env()
        
        # Create optimization configuration
        opt_config = OptimizationConfig(
            iterations=1,     # Just one iteration for testing
            samples=2,        # Just two samples for testing  
            training_steps=50, # Minimal training for testing
            algorithm="bptt"
        )
        
        # Environment configuration
        env_kwargs = {
            "num_agent_per_scene": 1,
            "device": "cpu",
            "target": torch.tensor([[3.0, 1.0, 1.5]]),
            "max_episode_steps": 50
        }
        
        print("Creating EurekaVisFly controller with real API...")
        
        # Create Eureka controller
        eureka = EurekaVisFly(
            env_class=SimpleNavEnv,
            task_description="Navigate drone to target position [3, 1, 1.5] efficiently while avoiding obstacles using depth sensor data",
            llm_config=LLM_CONFIG,
            env_kwargs=env_kwargs,
            optimization_config=opt_config,
            device="cpu"
        )
        
        print(f"✓ Controller created with {type(eureka.llm).__name__}")
        
        # Test environment creation
        test_env = eureka.create_environment()
        print(f"✓ Environment created: {test_env.__class__.__name__}")
        
        # Test reward generation
        print("Generating reward function candidates...")
        reward_functions = eureka.generate_reward_candidates(samples=2, iteration=0)
        print(f"✓ Generated {len(reward_functions)} reward function candidates")
        
        if reward_functions:
            print("\nTesting reward injection...")
            
            # Test reward injection
            from eureka_visfly.reward_injection import inject_generated_reward
            
            test_env2 = eureka.create_environment()
            test_env2.reset()
            
            original_reward = test_env2.get_reward()
            print(f"Original reward: {original_reward}")
            
            success = inject_generated_reward(test_env2, reward_functions[0])
            
            if success:
                new_reward = test_env2.get_reward()
                print(f"✓ Reward injection successful! New reward: {new_reward}")
                
                # Test that reward function actually changed
                if not torch.allclose(new_reward, original_reward, atol=1e-6):
                    print("✓ Reward function successfully replaced")
                    return True
                else:
                    print("! Reward function appears unchanged")
                    return False
            else:
                print("✗ Reward injection failed")
                return False
        else:
            print("✗ No reward functions generated")
            return False
            
    except Exception as e:
        print(f"✗ Real integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_mini_optimization():
    """Run a mini optimization to demonstrate the full pipeline."""
    print("\n=== Running Mini Optimization ===")
    
    try:
        from eureka_visfly import EurekaVisFly, OptimizationConfig
        from config import LLM_CONFIG
        
        # Create environment class
        SimpleNavEnv = create_simple_nav_env()
        
        # Create optimization configuration for quick test
        opt_config = OptimizationConfig(
            iterations=2,        # 2 iterations
            samples=3,           # 3 reward functions per iteration
            training_steps=20,   # Very minimal training
            algorithm="bptt",
            evaluation_episodes=3  # Quick evaluation
        )
        
        # Environment configuration
        env_kwargs = {
            "num_agent_per_scene": 1,
            "device": "cpu",
            "target": torch.tensor([[4.0, 0.0, 1.0]]),
            "max_episode_steps": 30
        }
        
        print("Starting mini optimization...")
        
        # Create Eureka controller
        eureka = EurekaVisFly(
            env_class=SimpleNavEnv,
            task_description="Navigate drone efficiently to target position [4, 0, 1] while avoiding obstacles and maintaining stability",
            llm_config=LLM_CONFIG,
            env_kwargs=env_kwargs,
            optimization_config=opt_config,
            device="cpu"
        )
        
        # Run optimization
        print("Running optimization (this may take several minutes)...")
        results = eureka.optimize_rewards()
        
        if results:
            print(f"\n✓ Optimization complete! Generated {len(results)} results")
            
            # Show best results
            best_result = results[0]
            print(f"Best reward score: {best_result.score():.4f}")
            print(f"Success rate: {best_result.success_rate:.3f}")
            print(f"Episode length: {best_result.episode_length:.1f}")
            print(f"Training time: {best_result.training_time:.1f}s")
            
            if len(results) > 1:
                print(f"\nTop 3 results:")
                for i, result in enumerate(results[:3]):
                    print(f"{i+1}. Score: {result.score():.4f}, Success: {result.success_rate:.3f}, Length: {result.episode_length:.1f}")
            
            return True
        else:
            print("✗ No results generated")
            return False
            
    except Exception as e:
        print(f"✗ Mini optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run real API integration tests."""
    print("VisFly-Eureka Real API Integration Test")
    print("=" * 60)
    
    tests = [
        ("Real API Connection", test_real_api_connection),
        ("Real Reward Generation", test_real_reward_generation),
        ("Real Integration", test_real_integration),
        ("Mini Optimization", run_mini_optimization),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("REAL API TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL" 
        print(f"{test_name:.<35} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All real API tests passed! Ready for production use.")
        return True
    elif passed > 0:
        print("⚠️  Some tests passed. Partial functionality working.")
        return True
    else:
        print("❌ All tests failed. Check API configuration and network connection.")
        return False


if __name__ == "__main__":
    main()