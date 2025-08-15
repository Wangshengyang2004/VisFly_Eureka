# VisFly-Eureka Native Integration

**LLM-Powered Reward Function Optimization for VisFly Drone Environments**

This project provides native integration between VisFly drone simulation environments and GPT-4o powered reward function optimization, enabling automatic generation and optimization of reward functions for vision-based UAV tasks.

## 🚀 Features

- **Direct Integration**: Reward functions inject directly into VisFly environments without wrapper layers
- **GPT-4o Powered**: Sophisticated reward function generation using state-of-the-art language models
- **Context-Aware**: Environment-specific prompts that understand VisFly's sensor data and dynamics
- **Differentiable-First**: Native support for VisFly's BPTT (Back-Propagation Through Time) training
- **Visual Intelligence**: Leverages depth/RGB sensors for sophisticated reward design
- **Multi-Environment**: Support for Navigation, Racing, Hovering, and Tracking environments

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Eureka-VisFly Core                      │
│              (Single Orchestrator)                     │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
│     GPT-4o   │ │ Reward   │ │ Training   │
│   Engine     │ │Injection │ │  Monitor   │
└──────────────┘ └────┬─────┘ └────────────┘
                      │
            ┌─────────▼─────────┐
            │ VisFly Environment │
            │ (Direct Integration)│
            └───────────────────┘
```

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd VisFly_Eureka
   ```

2. **Install dependencies**:
   ```bash
   pip install openai torch
   # Install VisFly dependencies as per VisFly documentation
   ```

3. **Configure API** (edit `config.py`):
   ```python
   LLM_CONFIG = {
       "model": "gpt-4o",
       "api_key": "your-api-key",
       "base_url": "https://yunwu.ai/v1"  # or "https://api.openai.com/v1"
   }
   ```

## 🔧 Quick Start

### Basic Usage

```python
from eureka_visfly import EurekaVisFly
from VisFly.envs.NavigationEnv import NavigationEnv
from config import LLM_CONFIG

# Create Eureka controller
eureka = EurekaVisFly(
    env_class=NavigationEnv,
    task_description="Navigate to target avoiding obstacles using depth sensor",
    llm_config=LLM_CONFIG
)

# Run optimization
best_rewards = eureka.optimize_rewards(iterations=5, samples=16)
print(f"Best reward score: {best_rewards[0].score():.3f}")
```

### Direct Reward Injection

```python
from eureka_visfly import inject_generated_reward
from VisFly.envs.NavigationEnv import NavigationEnv

# Create environment
env = NavigationEnv(num_agent_per_scene=4, device="cuda")

# Generate custom reward with GPT-4o
reward_code = '''
def get_reward(self):
    """Navigate to target while avoiding obstacles"""
    # Distance reward
    distance = torch.norm(self.position - self.target, dim=1)
    distance_reward = -distance * 0.1
    
    # Collision avoidance using depth sensor
    if 'depth' in self.sensor_obs:
        depth_data = self.sensor_obs['depth']
        collision_penalty = -torch.sum(depth_data < 0.5, dim=(1,2)) * 0.001
    else:
        collision_penalty = torch.zeros_like(distance_reward)
    
    return distance_reward + collision_penalty
'''

# Inject reward function
inject_generated_reward(env, reward_code)
```

## 🎯 Environment Support

### Navigation Environment
```python
from config import ENV_CONFIGS

eureka = EurekaVisFly(
    env_class=NavigationEnv,
    task_description="Precise navigation with obstacle avoidance using depth sensor",
    llm_config=LLM_CONFIG,
    env_kwargs=ENV_CONFIGS["NavigationEnv"]
)
```

### Custom Environment Configuration
```python
env_config = {
    "num_agent_per_scene": 4,
    "device": "cuda",
    "visual": True,
    "sensor_kwargs": [{
        "sensor_type": "DEPTH",
        "uuid": "depth",
        "resolution": [64, 64]
    }]
}

eureka = EurekaVisFly(
    env_class=YourCustomEnv,
    task_description="Your task description",
    llm_config=LLM_CONFIG,
    env_kwargs=env_config
)
```

## 📊 Advanced Usage

### Optimization Configuration
```python
from eureka_visfly import OptimizationConfig

config = OptimizationConfig(
    iterations=10,           # Number of optimization iterations
    samples=32,              # Reward functions per iteration
    training_steps=50000,    # Training steps per evaluation
    algorithm="bptt",        # "bptt" or "ppo"
    evaluation_episodes=20,  # Episodes for evaluation
    success_threshold=0.9    # Success rate threshold
)

eureka = EurekaVisFly(
    env_class=NavigationEnv,
    task_description="High-precision navigation task",
    llm_config=LLM_CONFIG,
    optimization_config=config
)
```

### Multi-Agent Coordination
```python
eureka = EurekaVisFly(
    env_class=MultiNavigationEnv,
    task_description="Coordinate multiple drones to reach targets while avoiding collisions",
    llm_config=LLM_CONFIG,
    env_kwargs={"num_agent_per_scene": 8}
)

results = eureka.optimize_rewards()
```

## 🧪 Testing

Run the test suite to verify your installation:

```bash
# Test real API integration
python examples/simple_usage_real.py

# Comprehensive integration test
python examples/real_navigation_example.py

# Basic functionality test
python test_basic_integration.py
```

## 📁 Project Structure

```
eureka_visfly/
├── __init__.py              # Package exports
├── eureka_visfly.py         # Main controller
├── llm_engine.py            # GPT-4o API integration
├── reward_injection.py      # Direct reward injection
├── training_utils.py        # Training and evaluation
└── prompts.py              # LLM prompt engineering

examples/
├── simple_usage_real.py     # Quick start example
├── real_navigation_example.py # Comprehensive example
└── navigation_example.py    # NavigationEnv specific

config.py                    # API configuration
```

## 🔍 Example Generated Reward Functions

### Navigation with Obstacle Avoidance
```python
def get_reward(self):
    """Navigate to target while avoiding obstacles using depth sensor"""
    reward = torch.zeros(self.num_agent)
    
    if hasattr(self, 'position') and hasattr(self, 'target'):
        # Distance-based reward
        distance_to_target = torch.norm(self.position - self.target, dim=1)
        distance_reward = -distance_to_target
        reward += distance_reward
    
    # Obstacle avoidance using depth sensor
    if 'depth' in self.sensor_obs:
        depth_data = self.sensor_obs['depth']
        collision_threshold = 0.5
        collision_penalty = -torch.sum(depth_data < collision_threshold, dim=(1, 2))
        reward += collision_penalty * 0.001
    
    # Time penalty for efficiency
    step_penalty = -0.01
    reward += step_penalty
    
    return reward
```

## 🎯 Key Benefits

1. **No Manual Reward Engineering**: GPT-4o automatically generates sophisticated reward functions
2. **Context-Aware Generation**: Understands VisFly environment structure and sensor data
3. **Native Integration**: Direct injection without performance overhead
4. **Iterative Improvement**: Automatically refines rewards based on training performance
5. **Multi-Objective Support**: Handles complex tasks with multiple objectives
6. **Differentiable Compatible**: Works seamlessly with VisFly's BPTT training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **VisFly Team**: For the excellent drone simulation framework
- **OpenAI**: For providing the GPT-4o API
- **Original Eureka Authors**: For the reward optimization concept

## 🆘 Support

- **Documentation**: Check the `examples/` directory for usage examples
- **Issues**: Report bugs and feature requests via GitHub issues
- **API Support**: Ensure your yunwu.ai API key is properly configured

---

**Ready to revolutionize drone reward engineering with AI! 🚁🤖**