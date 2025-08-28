# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

VisFly_Eureka (renamed to quadro-llm) integrates VisFly (visual-based quadrotor simulator) with Eureka (LLM-powered reward optimization). The system generates and optimizes reward functions automatically using GPT-4o for drone control tasks.

## Commands

### Setup and Installation

```bash
# Install dependencies
pip install -e .

# Run tests
pytest tests/ -v
pytest tests/unit/ -m "not gpu"  # CPU-only tests
pytest tests/integration/ -m "not llm"  # Without LLM API calls

# Run with coverage
pytest --cov=quadro_llm --cov-report=html
```

### Training and Optimization

```bash
# LLM-powered reward optimization (main entry)
python main.py  # Uses default Hydra config
python main.py llm=gpt-4o env=navigation_env
python main.py optimization.iterations=10 system=high_performance

# Direct training with existing rewards
python run.py --env navigation --algorithm bptt
python run.py --env hover --algorithm ppo --num_envs 48
python run.py --env racing --algorithm shac --reward_function_path rewards/custom.py

# Console commands (after installation)
quadro-llm --config configs/config.yaml
visfly-train --env navigation --algorithm bptt
```

### Configuration Overrides

```bash
# Hydra-style overrides for main.py
python main.py llm.model=gpt-4o llm.temperature=0.8
python main.py env=racing_env optimization.samples=32
python main.py hydra.run.dir=outputs/custom_experiment

# Command-line args for run.py
python run.py --env navigation --algorithm bptt --num_envs 160 --device cpu
python run.py --env hover --algorithm ppo --timesteps 100000 --save_freq 10000
```

### Monitoring

```bash
# TensorBoard for training metrics
tensorboard --logdir outputs/
tensorboard --logdir outputs/*/tensorboard/

# GPU monitoring (built-in)
# Automatically monitors during training with GPUMonitor class
```

## High-Level Architecture

### Core Module Structure

```
quadro_llm/
├── __init__.py                 # Main exports (EurekaVisFly, Pipeline, models)
├── pipeline.py                 # Production pipeline orchestrator
├── eureka_visfly.py           # Core Eureka-VisFly controller
├── core/
│   ├── models.py              # Data models (RewardFunction, OptimizationResult)
│   ├── evaluation.py          # Training evaluation metrics
│   └── reward_injection.py    # Runtime reward function injection
├── llm/
│   ├── llm_engine.py          # LLM API integration
│   └── prompts.py             # Prompt templates and engineering
├── training/
│   ├── parallel_training.py   # Multi-process training manager
│   └── visfly_training_wrapper.py  # VisFly integration wrapper
└── utils/
    ├── config_utils.py        # Configuration management
    ├── gpu_monitor.py         # GPU resource monitoring
    ├── training_utils.py      # Training helpers
    └── reward_injection.py    # Injection utilities
```

### Configuration System

Uses Hydra for composable configs:
- `configs/config.yaml`: Main config with defaults
- `configs/envs/*.yaml`: Environment-specific settings
- `configs/algs/*/*yaml`: Algorithm configs per environment
- `configs/llm/*.yaml`: LLM provider configurations
- `configs/system/*.yaml`: System performance profiles

### Environment Configurations

Key parameters per environment:
- **BPTT**: `requires_grad: true`, CPU-optimized, 160 agents
- **PPO**: `requires_grad: false`, GPU-accelerated, 48 agents  
- **SHAC**: `requires_grad: true`, CPU-optimized, 160 agents

### Direct Reward Injection Pattern

```python
# Runtime injection without modifying environment
def inject_reward(env_instance, reward_code):
    exec_globals = {'torch': torch, 'th': torch}
    exec(reward_code, exec_globals)
    env_instance.get_reward = types.MethodType(
        exec_globals['get_reward'], env_instance
    )
```

### Environment State Access

Available in reward functions via `self`:
- `position`: [N, 3] drone positions
- `velocity`: [N, 3] linear velocities
- `orientation`: [N, 4] quaternions
- `angular_velocity`: [N, 3]
- `target`: [N, 3] target positions
- `sensor_obs`: dict with 'depth'/'rgb' data
- `collision_dis`: [N] obstacle distances
- `_step_count`: current episode step
- `max_episode_steps`: episode limit
- `_success`: success flag

### Workflow Patterns

1. **Eureka Pipeline** (main.py):
   ```python
   Pipeline → Generate rewards → Inject → Train → Evaluate → Iterate
   ```

2. **Direct Training** (run.py):
   ```python
   Load env config → Setup algorithm → Inject reward → Train → Save
   ```

3. **Custom Reward Development**:
   ```python
   Write reward → Test with run.py → Integrate into pipeline
   ```

### Testing Strategy

- **Unit tests**: Core functions, reward injection, models
- **Integration tests**: Full pipeline, training loops
- **Fixtures**: Mock VisFly environments for isolated testing
- **Markers**: `@pytest.mark.gpu`, `@pytest.mark.llm`, `@pytest.mark.slow`

### Performance Optimization

- **Batch sizes**: BPTT/SHAC use 160 agents on CPU, PPO uses 48 on GPU
- **Memory management**: Automatic GPU memory monitoring and adjustment
- **Parallel evaluation**: Subprocess isolation for concurrent training
- **Algorithm selection**: BPTT for precise control, PPO for exploration

### Key Files and Entry Points

- `main.py`: Hydra-based Eureka pipeline entry
- `run.py`: Direct training with algorithm selection
- `quadro_llm/pipeline.py`: Production pipeline implementation
- `quadro_llm/eureka_visfly.py`: Core optimization logic
- `configs/config.yaml`: Main configuration hub
- `tests/conftest.py`: Test fixtures and mocks

### Environment Implementations

Located in `envs/`:
- `NavigationEnv.py`: Navigate to target with obstacles
- `HoverEnv.py`: Stability control
- `ObjectTrackingEnv.py`: Visual tracking
- `RacingEnv.py`: Racing scenarios
- `LandingEnv.py`: Precision landing
- `CatchEnv.py`: Object catching
- `FlipEnv.py`: Aerobatic maneuvers

### Algorithm Implementations

Located in `algorithms/`:
- `BPTT.py`: Differentiable simulation with analytical gradients
- `PPO.py`: Proximal Policy Optimization for RL
- `SHAC.py`: Soft Hierarchical Actor-Critic

### Debugging and Development

```bash
# Test specific components
pytest tests/unit/test_reward_injection.py -v
pytest tests/integration/test_pipeline.py::test_navigation_pipeline

# Debug mode with verbose output
python main.py hydra.verbose=true

# Dry run without training
python main.py optimization.dry_run=true

# Custom output directory
python main.py hydra.run.dir=./debug_output
```

### API Configuration

Set in configs or environment:
```bash
export OPENAI_API_KEY="your-key"
export LLM_BASE_URL="https://api.openai.com/v1"  # or custom endpoint
```

Or in config files:
```yaml
llm:
  api_key: ${oc.env:OPENAI_API_KEY}
  base_url: ${oc.env:LLM_BASE_URL,https://api.openai.com/v1}
```