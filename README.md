# QuadroLLM: VisFly-Eureka Integration

**Advanced LLM-Powered Reward Function Optimization for Vision-Based Quadrotor Control**

QuadroLLM integrates VisFly (visual-based quadrotor simulator) with Eureka-style LLM-powered reward optimization, enabling automatic generation and iterative refinement of reward functions for complex drone control tasks. The system uses large language models to generate sophisticated reward functions that are automatically optimized through parallel training and evaluation.

## 🚀 Key Features

### Core Capabilities
- **🧠 LLM-Powered Optimization**: Uses GPT-4o/GLM-4.6 for intelligent reward function generation
- **🔄 Iterative Refinement**: Automatically improves reward functions based on training performance and TensorBoard feedback
- **⚡ Parallel Evaluation**: Multi-GPU training with intelligent resource allocation and load balancing  
- **📊 Comprehensive Analytics**: TensorBoard integration with DataFrame analysis for data-driven optimization
- **💾 Full Artifact Logging**: Complete conversation logs, generated rewards, and training metrics preservation

### Technical Excellence
- **🎯 Direct Integration**: Native reward injection into VisFly environments without performance overhead
- **🔍 Context-Aware Generation**: Environment-specific prompts understanding VisFly sensor data and dynamics
- **🧮 Differentiable-First**: Full support for BPTT (Back-Propagation Through Time) and gradient-based optimization
- **👁️ Visual Intelligence**: Leverages depth/RGB/IMU sensors for sophisticated multi-modal reward design
- **🌐 Multi-Environment Support**: Navigation, Racing, Hovering, Object Tracking, Landing, and custom environments

### Production-Ready
- **⚙️ Hydra Configuration**: Composable config system with environment, algorithm, and LLM provider flexibility
- **🔧 Algorithm Agnostic**: Support for BPTT, PPO, SHAC with automatic hyperparameter adaptation
- **📈 GPU Monitoring**: Real-time memory tracking, OOM handling, and dynamic resource management
- **🛡️ Robust Error Handling**: Comprehensive failure recovery, timeout management, and subprocess isolation

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      QuadroLLM Pipeline                        │
│            (Hydra-Configured Orchestration)                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
      ┌───────────────┼───────────────┐
      │               │               │
┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
│    LLM    │  │ Parallel  │  │   GPU     │
│  Engine   │  │Evaluation │  │ Resource  │
│(GPT/GLM)  │  │ Manager   │  │ Manager   │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │              │              │
      ▼              ▼              ▼
┌─────────────────────────────────────────────────┐
│            Reward Injection System              │
├─────────────────────────────────────────────────┤
│              VisFly Environments                │
│   Navigation │ Racing │ Hover │ Track │ Land    │
└─────────────────┬───────────────────────────────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐
│   BPTT    │ │  PPO  │ │   SHAC    │
│(Diff Sim)│ │ (GPU) │ │ (ActorCr) │
└───────────┘ └───────┘ └───────────┘
```

## 📦 Installation & Setup

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-org/QuadroLLM.git
cd QuadroLLM

# Install in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

### 2. VisFly Integration
Ensure VisFly is properly installed and accessible:
```bash
# VisFly should be in the project directory or Python path
ls VisFly/  # Should contain envs/, algorithms/, etc.
```

### 3. API Configuration
Set up your LLM API credentials:
```bash
# Option 1: Environment variables (recommended)
export OPENAI_API_KEY="your-openai-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"

# Option 2: For custom providers (e.g., GLM-4.6)
export OPENAI_API_KEY="your-glm-api-key" 
export LLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"
```

### 4. Verify Installation
```bash
# Test basic functionality
python -c "import quadro_llm; print('✅ QuadroLLM installed successfully')"

# Test with minimal config
python main.py --help

# Run a quick test (dry-run mode)
python main.py optimization.dry_run=true
```

## 🔧 Usage Guide

### 1. Hydra-Based Pipeline (Recommended)
The main entry point uses Hydra for configuration management:

```bash
# Basic usage with defaults
python main.py

# Customize LLM and environment
python main.py llm=gpt-4o env=navigation_env

# Full customization
python main.py \
  llm.model=gpt-4o \
  llm.temperature=0.8 \
  env=racing_env \
  optimization.iterations=10 \
  optimization.samples=32
```

### 2. Direct Training (Skip LLM Generation)
Train with existing reward functions:

```bash
# Train with BPTT algorithm
python run.py --env navigation --algorithm bptt --num_envs 160

# Train with PPO on GPU
python run.py --env hover --algorithm ppo --num_envs 48 --device cuda

# Use custom reward function
python run.py --env racing --algorithm shac \
  --reward_function_path rewards/custom_racing_reward.py
```

### 3. Python API Usage
```python
from quadro_llm import Pipeline, EurekaVisFly
from VisFly.envs.NavigationEnv import NavigationEnv

# Option A: Production Pipeline
pipeline = Pipeline.from_config("configs/config.yaml")
results = pipeline.run()

# Option B: Direct API
eureka = EurekaVisFly(
    env_class=NavigationEnv,
    task_description="Navigate to target avoiding obstacles using depth sensor",
    llm_config={"model": "gpt-4o", "api_key": "your-key"},
    optimization_config={"iterations": 5, "samples": 16}
)
best_rewards = eureka.optimize_rewards()
```

## 🎛️ Configuration System

QuadroLLM uses Hydra for flexible, composable configuration:

### Environment Configs (`configs/envs/`)
```yaml
# navigation_env.yaml
_target_: VisFly.envs.NavigationEnv.NavigationEnv
num_agent_per_scene: 160  # BPTT optimized
device: cpu
requires_grad: true
max_episode_steps: 500
sensor_kwargs:
  - sensor_type: DEPTH
    uuid: depth
    resolution: [64, 64]
```

### Algorithm Configs (`configs/algs/navigation/`)
```yaml
# bptt.yaml - For gradient-based optimization
algorithm: BPTT
device: cpu
requires_grad: true
learning_rate: 0.001
batch_size: 160

# ppo.yaml - For reinforcement learning
algorithm: PPO
device: cuda
requires_grad: false
n_steps: 2048
batch_size: 48
```

### LLM Configs (`configs/llm/`)
```yaml
# gpt-4o.yaml
model: gpt-4o
api_key: ${oc.env:OPENAI_API_KEY}
base_url: https://api.openai.com/v1
temperature: 0.7
max_tokens: 4000
batching_strategy: n_parameter

# glm-4.6.yaml
model: glm-4.6
api_key: ${oc.env:OPENAI_API_KEY}
base_url: https://open.bigmodel.cn/api/paas/v4/
batching_strategy: sequential
thinking_enabled: false
```

## 🎯 Supported Environments & Algorithms

### Environment Capabilities

| Environment | Description | Key Features | Recommended Algorithm |
|-------------|-------------|--------------|----------------------|
| **NavigationEnv** | Point-to-point navigation with obstacles | Depth sensors, collision avoidance | BPTT (160 agents, CPU) |
| **HoverEnv** | Stability and position holding | IMU feedback, wind disturbance | PPO (48 agents, GPU) |
| **RacingEnv** | High-speed racing through gates | RGB vision, trajectory optimization | SHAC (160 agents, CPU) |
| **ObjectTrackingEnv** | Visual object following | RGB+depth, moving targets | PPO (48 agents, GPU) |
| **LandingEnv** | Precision landing maneuvers | Depth+IMU, landing pad detection | BPTT (160 agents, CPU) |
| **CatchEnv** | Dynamic object interception | Multi-modal sensing | SHAC (160 agents, CPU) |
| **FlipEnv** | Aerobatic maneuver execution | IMU-heavy, attitude control | BPTT (160 agents, CPU) |

### Algorithm Characteristics

#### BPTT (Back-Propagation Through Time)
- **Best for**: Precise control, gradient-based optimization
- **Setup**: CPU-optimized, 160 agents, `requires_grad=true`
- **Strengths**: Analytical gradients, fast convergence
- **Use cases**: Navigation, landing, aerobatics

#### PPO (Proximal Policy Optimization)
- **Best for**: Exploration-heavy tasks, robust learning
- **Setup**: GPU-accelerated, 48 agents, `requires_grad=false`
- **Strengths**: Sample efficiency, stable training
- **Use cases**: Hovering, tracking, dynamic environments

#### SHAC (Soft Hierarchical Actor-Critic)
- **Best for**: Complex multi-objective tasks
- **Setup**: CPU-optimized, 160 agents, `requires_grad=true`
- **Strengths**: Hierarchical control, multi-task learning
- **Use cases**: Racing, catching, complex maneuvers

## 📊 Monitoring & Analytics

### TensorBoard Integration
QuadroLLM automatically generates comprehensive training analytics:

```bash
# Start TensorBoard to monitor training
tensorboard --logdir outputs/

# Monitor specific experiment
tensorboard --logdir outputs/2025-01-09_14-30-15/tensorboard/
```

**Available Metrics:**
- Episode reward curves and success rates
- Training loss (actor/critic for RL algorithms)
- GPU memory usage and resource utilization
- Algorithm-specific metrics (policy entropy, value estimates)

### Output Structure
```
outputs/2025-01-09_14-30-15/
├── config.yaml                    # Complete experiment configuration
├── generated_rewards/             # All generated reward functions
│   ├── iter0/                   # Reward functions by iteration
│   └── iter1/                   
├── training_outputs/             # Training results per sample
│   ├── sample0/                 # Individual training logs
│   └── sample1/                 
├── artifacts/                    # Analysis artifacts
│   ├── llm_conversations_iteration_0.json  # Complete LLM interactions
│   └── optimization_summary.json # Final results summary
└── tensorboard/                  # TensorBoard logs
```

### Performance Analysis
```python
# Load optimization results for analysis
from quadro_llm.utils.tensorboard_utils import load_tensorboard_logs
from pathlib import Path

# Load training logs
logs = load_tensorboard_logs("outputs/latest/training_outputs/sample0")

# Analyze performance trends
if "rollout/success_rate" in logs:
    success_rates = logs["rollout/success_rate"]
    print(f"Final success rate: {success_rates[-1]:.3f}")
    print(f"Convergence at step: {len(success_rates) * 100}")
```

## 🧪 Testing & Validation

### Test Suite
```bash
# Run all tests
pytest tests/ -v

# CPU-only tests (no GPU required)
pytest tests/unit/ -m "not gpu"

# Integration tests (without LLM API calls)
pytest tests/integration/ -m "not llm"

# Full integration with LLM APIs
pytest tests/integration/ -m "llm" --api-key="your-key"

# Coverage report
pytest --cov=quadro_llm --cov-report=html
```

### Validation Examples
```bash
# Quick validation (uses mock LLM)
python -m quadro_llm.examples.mock_validation

# Real API test (requires API key)
python -m quadro_llm.examples.api_validation

# End-to-end pipeline test
python main.py env=navigation_env optimization.samples=2 optimization.iterations=1
```

### Debugging Tools
```bash
# Verbose mode with detailed logging
python main.py hydra.verbose=true optimization.dry_run=false

# Test specific components
python -c "from quadro_llm import Pipeline; Pipeline.test_components()"

# Validate configuration
python -c "from hydra import initialize, compose; initialize('configs'); print('✅ Config valid')"
```

## 📁 Project Structure

```
QuadroLLM/
├── main.py                          # Hydra-based main entry point
├── run.py                           # Direct training entry point
├── quadro_llm/                      # Core package
│   ├── __init__.py                  # Main exports
│   ├── pipeline.py                  # Production pipeline
│   ├── eureka_visfly.py            # Core optimization controller
│   ├── core/                        # Core components
│   │   ├── models.py               # Data models and types
│   │   ├── evaluation.py           # Training evaluation
│   │   ├── evaluation_worker.py    # Subprocess evaluation worker
│   │   └── subprocess_evaluator.py # Parallel evaluation manager
│   ├── llm/                        # LLM integration
│   │   ├── llm_engine.py          # LLM API client
│   │   └── prompts.py             # Prompt engineering
│   ├── training/                   # Training components
│   │   ├── parallel_training.py   # Multi-process training
│   │   └── visfly_training_wrapper.py # VisFly integration
│   └── utils/                      # Utilities
│       ├── config_utils.py        # Configuration helpers
│       ├── gpu_monitor.py         # GPU resource management
│       ├── tensorboard_utils.py   # TensorBoard integration
│       └── reward_injection.py    # Reward function injection
├── configs/                        # Hydra configurations
│   ├── config.yaml                # Main config
│   ├── envs/                      # Environment configs
│   ├── algs/                      # Algorithm configs per environment
│   ├── llm/                       # LLM provider configs
│   └── system/                    # System performance profiles
├── tests/                          # Test suite
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── conftest.py               # Test fixtures
├── VisFly/                        # VisFly simulator (submodule/dependency)
│   ├── envs/                     # Environment implementations
│   └── algorithms/               # Algorithm implementations
└── requirements.txt               # Python dependencies
```

## 🔍 Generated Reward Examples

### Navigation with Multi-Modal Sensing
```python
def get_reward(self):
    """Advanced navigation reward leveraging depth sensor and collision detection"""
    import torch
    
    reward = torch.zeros(self.num_agent, device=self.device)
    
    # Target approach reward (progressive)
    if hasattr(self, 'position') and hasattr(self, 'target'):
        distance = torch.norm(self.position - self.target, dim=1)
        progress_reward = -distance * 0.2
        
        # Success bonus for reaching target
        success_bonus = torch.where(distance < 0.5, 10.0, 0.0)
        reward += progress_reward + success_bonus
    
    # Multi-layered collision avoidance
    if 'depth' in self.sensor_obs:
        depth = self.sensor_obs['depth']  # Shape: [N, H, W]
        
        # Critical collision zone (immediate danger)
        critical_pixels = (depth < 0.3).float()
        critical_penalty = -torch.sum(critical_pixels, dim=(1,2)) * 0.1
        
        # Warning zone (future planning)
        warning_pixels = ((depth >= 0.3) & (depth < 0.8)).float()
        warning_penalty = -torch.sum(warning_pixels, dim=(1,2)) * 0.01
        
        reward += critical_penalty + warning_penalty
    
    # Collision detection backup
    if hasattr(self, 'collision_dis'):
        collision_penalty = -torch.clamp((1.0 - self.collision_dis), min=0) * 5.0
        reward += collision_penalty
    
    # Efficiency incentive
    velocity_magnitude = torch.norm(self.velocity, dim=1)
    efficiency_reward = torch.clamp(velocity_magnitude - 0.1, min=0) * 0.05
    reward += efficiency_reward
    
    return reward
```

### Racing Environment Reward
```python  
def get_reward(self):
    """High-speed racing reward with trajectory optimization"""
    import torch
    
    reward = torch.zeros(self.num_agent, device=self.device)
    
    # Speed incentive for racing
    speed = torch.norm(self.velocity, dim=1)
    speed_reward = torch.clamp(speed, max=15.0) * 0.3
    reward += speed_reward
    
    # Gate passage detection and ordering
    if hasattr(self, 'gate_progress') and hasattr(self, '_step_count'):
        # Reward for advancing through gates in order
        gate_bonus = self.gate_progress * 20.0
        
        # Time-based efficiency bonus
        time_bonus = torch.where(
            self.gate_progress > 0.8, 
            torch.clamp(500 - self._step_count, min=0) * 0.02,
            0.0
        )
        reward += gate_bonus + time_bonus
    
    # Racing-specific collision handling (less punitive)
    if 'depth' in self.sensor_obs:
        depth = self.sensor_obs['depth']
        collision_risk = (depth < 0.2).float().sum(dim=(1,2))
        collision_penalty = -collision_risk * 0.05  # Light penalty for racing
        reward += collision_penalty
        
    return reward
```

## 🎯 Key Research Contributions

### 1. **LLM-Native Reward Optimization**
- First system to integrate Eureka-style optimization directly with differentiable simulation
- Context-aware prompt engineering that understands multi-modal sensor data (depth, RGB, IMU)
- Automatic reward function refinement based on TensorBoard training curves and DataFrame analytics

### 2. **Production-Grade Parallel Training**
- Intelligent GPU resource management with dynamic memory monitoring and OOM recovery
- Multi-algorithm support (BPTT, PPO, SHAC) with automatic hyperparameter adaptation
- Subprocess isolation preventing memory leaks and ensuring reproducible results

### 3. **Comprehensive Evaluation Framework** 
- Full conversation logging for LLM interaction analysis and reproducibility
- TensorBoard integration with automatic DataFrame creation for next-iteration feedback
- Hydra-based configuration system enabling systematic hyperparameter studies

### 4. **Vision-Based Quadrotor Control Innovation**
- Native support for VisFly's visual navigation environments and sensor modalities
- Multi-environment evaluation across navigation, racing, hovering, and tracking tasks
- Direct reward injection enabling real-time policy adaptation during training

### 5. **Scalability & Robustness**
- Handles large-scale parallel training (160+ agents on multi-GPU systems)
- Comprehensive error handling with timeout management and graceful failure recovery
- Extensive test coverage including unit, integration, and end-to-end validation

## 📖 Citation

If you use QuadroLLM in your research, please cite:

```bibtex
@software{quadro_llm_2025,
  title={QuadroLLM: Advanced LLM-Powered Reward Function Optimization for Vision-Based Quadrotor Control},
  author={Your Name and Team},
  year={2025},
  url={https://github.com/your-org/QuadroLLM},
  note={Integrating VisFly simulation with Eureka-style LLM optimization}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Development installation
git clone https://github.com/your-org/QuadroLLM.git
cd QuadroLLM
pip install -e ".[dev]"

# Pre-commit hooks
pre-commit install

# Run tests before submitting
pytest tests/ --cov=quadro_llm
```

### Areas for Contribution
- **New Environments**: Add support for additional VisFly environments
- **Algorithm Integration**: Implement new RL/optimization algorithms  
- **LLM Providers**: Add support for additional LLM APIs
- **Performance Optimizations**: GPU memory efficiency, training speed improvements
- **Analysis Tools**: Enhanced visualization and analysis capabilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **VisFly Development Team**: For creating the outstanding visual-based quadrotor simulation framework
- **OpenAI & BigModel Teams**: For providing powerful LLM APIs (GPT-4o, GLM-4.6)
- **Original Eureka Authors** (Ma et al.): For pioneering LLM-powered reward optimization methodology  
- **Hydra Team**: For the excellent configuration management framework
- **PyTorch Community**: For the foundation enabling differentiable simulation

## 🆘 Support & Community

- **📚 Documentation**: Comprehensive guides in the `/docs` directory
- **🐛 Bug Reports**: Submit issues via [GitHub Issues](https://github.com/your-org/QuadroLLM/issues)
- **💬 Discussions**: Join our [GitHub Discussions](https://github.com/your-org/QuadroLLM/discussions)
- **🔧 API Support**: Ensure your LLM API credentials are properly configured
- **📧 Contact**: [your-email@institution.edu](mailto:your-email@institution.edu) for research collaboration

---

**🚁 Revolutionizing quadrotor control through the power of AI and differentiable simulation! 🤖**
