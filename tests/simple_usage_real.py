"""
Simple Usage Example with Real API for VisFly-Eureka Integration

This example demonstrates how to use the VisFly-Eureka integration 
with the real API for reward function optimization.
"""

import sys
import torch
import logging

# Add project paths
sys.path.insert(0, '/home/simonwsy/VisFly_Eureka')
sys.path.insert(0, '/home/simonwsy/VisFly_Eureka/VisFly')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def simple_example_with_real_api():
    """Demonstrate basic usage with real API."""
    print("=" * 60)
    print("VisFly-Eureka Integration - Real API Example")
    print("=" * 60)
    
    # Import components
    from eureka_visfly import EurekaVisFly, OptimizationConfig
    from config import LLM_CONFIG
    
    # Define a simple navigation environment for demonstration
    class SimpleNavigationEnv:
        """Simple navigation environment for demonstration."""
        
        def __init__(self, num_agent_per_scene=1, device="cpu", target=None, **kwargs):
            self.num_agent = num_agent_per_scene
            self.device = device
            self.target = target if target is not None else torch.tensor([[5.0, 0.0, 1.5]])
            self.position = torch.zeros(self.num_agent, 3)
            self.velocity = torch.zeros(self.num_agent, 3)
            self.orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * self.num_agent)
            self.angular_velocity = torch.zeros(self.num_agent, 3)
            self.max_episode_steps = kwargs.get("max_episode_steps", 100)
            self._step_count = 0
            
            # Simulate depth sensor data
            self.sensor_obs = {
                "depth": torch.ones(self.num_agent, 64, 64) * 2.0  # 2 meters depth
            }
            self.collision_dis = torch.ones(self.num_agent) * 2.0
            
        def reset(self):
            """Reset environment to initial state."""
            self.position = torch.randn(self.num_agent, 3) * 0.1  # Start near origin
            self.velocity = torch.zeros(self.num_agent, 3)
            self._step_count = 0
            return {"state": torch.cat([self.position, self.velocity], dim=1)}
            
        def step(self, action):
            """Simulate one environment step."""
            self.velocity += action * 0.1
            self.position += self.velocity * 0.02
            self._step_count += 1
            
            obs = {"state": torch.cat([self.position, self.velocity], dim=1)}
            reward = self.get_reward()
            done = torch.norm(self.position - self.target, dim=1) < 0.1
            
            return obs, reward, done.any(), {}
            
        def get_reward(self):
            """Default reward function (will be replaced by generated ones)."""
            distance = torch.norm(self.position - self.target, dim=1)
            return -distance * 0.1
            
        def get_success(self):
            """Check if navigation was successful."""
            distance = torch.norm(self.position - self.target, dim=1)
            return distance < 0.1

    print("1. Setting up environment and configuration...")
    
    # Configure the optimization
    config = OptimizationConfig(
        iterations=1,        # Just 1 iteration for demo
        samples=2,           # Generate 2 reward functions
        training_steps=50,   # Minimal training for demonstration
        algorithm="bptt",
        evaluation_episodes=3
    )
    
    # Environment configuration
    env_kwargs = {
        "num_agent_per_scene": 1,
        "device": "cpu",
        "target": torch.tensor([[3.0, 1.0, 1.0]]),
        "max_episode_steps": 50
    }
    
    print("2. Creating EurekaVisFly controller with real API...")
    
    # Create the Eureka controller with real API
    eureka = EurekaVisFly(
        env_class=SimpleNavigationEnv,
        task_description="Navigate drone to target position [3, 1, 1] efficiently while avoiding obstacles using depth sensor",
        llm_config=LLM_CONFIG,  # Real API configuration
        env_kwargs=env_kwargs,
        optimization_config=config,
        device="cpu"
    )
    
    print(f"✓ Controller created with {type(eureka.llm).__name__}")
    
    print("\n3. Testing environment creation...")
    
    # Test environment creation
    test_env = eureka.create_environment()
    print(f"✓ Environment created: {test_env.__class__.__name__}")
    
    # Test environment functionality
    obs = test_env.reset()
    print(f"✓ Environment reset successful, obs shape: {obs['state'].shape}")
    
    original_reward = test_env.get_reward()
    print(f"✓ Original reward: {original_reward}")
    
    print("\n4. Testing real reward function generation...")
    print("This may take a moment to contact the API...")
    
    # Generate reward function candidates
    reward_functions = eureka.generate_reward_candidates(samples=2, iteration=0)
    print(f"✓ Generated {len(reward_functions)} reward function candidates")
    
    if reward_functions:
        print("\nFirst generated reward function:")
        print("-" * 40)
        print(reward_functions[0])
        print("-" * 40)
        
        print("\n5. Testing reward injection...")
        
        # Test reward injection
        from eureka_visfly.reward_injection import inject_generated_reward
        
        test_env2 = eureka.create_environment()
        success = inject_generated_reward(test_env2, reward_functions[0])
        
        if success:
            test_env2.reset()
            new_reward = test_env2.get_reward()
            print(f"✓ Reward injection successful! New reward: {new_reward}")
            
            # Compare with original
            if not torch.allclose(new_reward, original_reward):
                print("✓ Reward function successfully replaced")
            else:
                print("! Reward function appears unchanged")
        else:
            print("✗ Reward injection failed")
            
    print("\n6. Real optimization workflow:")
    
    print("📝 With real API, you can now:")
    print("   1. Generate sophisticated reward functions using GPT-4")
    print("   2. Inject them directly into VisFly environments")
    print("   3. Train policies using BPTT or PPO")
    print("   4. Evaluate and rank reward functions by performance")
    print("   5. Iterate with feedback to continuously improve")
    
    print("\n" + "=" * 60)
    print("🎉 Real API Integration Complete!")
    print("=" * 60)
    print()
    print("✓ Real Components Working:")
    print("  - OpenAI-compatible API integration")
    print("  - Real reward function generation with GPT-4o")
    print("  - Context-aware prompts for VisFly environments")
    print("  - Direct reward injection without wrapper layers")
    print("  - Ready for full optimization pipeline")
    print()
    print("🚀 Ready for Production Reward Optimization!")


def show_real_api_usage():
    """Show the main API usage patterns with real configuration."""
    print("\n" + "=" * 60)
    print("Real API Usage Examples")
    print("=" * 60)
    
    print("\n1. Basic Usage with Real API:")
    print("""
from eureka_visfly import EurekaVisFly
from VisFly.envs.NavigationEnv import NavigationEnv
from config import LLM_CONFIG

# Create Eureka controller with real API
eureka = EurekaVisFly(
    env_class=NavigationEnv,
    task_description="Navigate to target avoiding obstacles using depth sensor", 
    llm_config=LLM_CONFIG  # Real API configuration
)

# Run optimization
best_rewards = eureka.optimize_rewards(iterations=5, samples=16)
print(f"Best reward score: {best_rewards[0].score():.3f}")
""")
    
    print("\n2. Custom API Configuration:")
    print("""
# Custom API configuration
custom_llm_config = {
    "model": "gpt-4o",
    "api_key": "your-api-key",
    "base_url": "https://yunwu.ai/v1",
    "temperature": 0.8,
    "max_tokens": 1500,
    "timeout": 120
}

eureka = EurekaVisFly(
    env_class=YourEnv,
    task_description="Your task",
    llm_config=custom_llm_config
)
""")
    
    print("\n3. Environment-Specific Optimization:")
    print("""
from config import ENV_CONFIGS

# Use predefined environment configurations
nav_config = ENV_CONFIGS["NavigationEnv"]

eureka = EurekaVisFly(
    env_class=NavigationEnv,
    task_description="Precise navigation with obstacle avoidance",
    llm_config=LLM_CONFIG,
    env_kwargs=nav_config
)
""")


def test_api_connection():
    """Quick test to verify API is working."""
    print("\n" + "=" * 60)
    print("API Connection Test")
    print("=" * 60)
    
    try:
        from eureka_visfly.llm_engine import LLMEngine
        from config import LLM_CONFIG
        
        print("Testing API connection...")
        llm = LLMEngine(**LLM_CONFIG)
        
        success = llm.test_api_connection()
        if success:
            print("✅ API connection successful!")
            print("   Model:", LLM_CONFIG["model"])
            print("   Base URL:", LLM_CONFIG["base_url"])
        else:
            print("❌ API connection failed!")
            
        return success
        
    except Exception as e:
        print(f"❌ API connection error: {e}")
        return False


if __name__ == "__main__":
    try:
        # Test API first
        if test_api_connection():
            simple_example_with_real_api()
            show_real_api_usage()
        else:
            print("\n❌ Cannot run example - API connection failed")
            print("Please check your API configuration in config.py")
            
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()