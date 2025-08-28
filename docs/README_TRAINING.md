# Config-Based Training System Guide

The new training wrapper uses a clean config file structure similar to the obj_track project. This makes algorithm and environment management much simpler and more maintainable.

## Directory Structure

```
configs/
├── envs/
│   └── navigation/
│       └── env.yaml          # Navigation environment config
├── algs/
│   └── navigation/
│       ├── bptt.yaml         # BPTT algorithm config for navigation
│       ├── ppo.yaml          # PPO algorithm config for navigation
│       └── shac.yaml         # SHAC algorithm config for navigation
└── config.yaml               # Main pipeline config
```

## Quick Usage

### Basic Training Commands

```bash
# Train navigation with BPTT (default)
python quadro_llm/training/visfly_training_wrapper_v2.py

# Train navigation with PPO
python quadro_llm/training/visfly_training_wrapper_v2.py --algorithm ppo

# Train navigation with SHAC
python quadro_llm/training/visfly_training_wrapper_v2.py --algorithm shac

# Train with custom comment and save directory
python quadro_llm/training/visfly_training_wrapper_v2.py --algorithm ppo --comment "my_experiment" --save_dir "./my_results"
```

### Parameter Overrides

```bash
# Override specific parameters
python quadro_llm/training/visfly_training_wrapper_v2.py \
  --algorithm ppo \
  --learning_steps 50000 \
  --num_agents 32 \
  --device cpu

# Use custom reward function
python quadro_llm/training/visfly_training_wrapper_v2.py \
  --algorithm bptt \
  --reward_function_path generated_rewards/best_reward.py
```

## Configuration Files

### Environment Config (`configs/envs/navigation/env.yaml`)

Defines environment parameters:
- Number of agents
- Episode length
- Sensor configuration
- Scene settings
- Target position
- Random initialization

### Algorithm Configs (`configs/algs/navigation/{algorithm}.yaml`)

Each algorithm has its own config with:
- **algorithm**: Core algorithm parameters
- **env_overrides**: Environment-specific overrides
- **learn**: Training parameters
- **test**: Evaluation parameters

## Algorithm-Specific Settings

### BPTT Configuration
- **Device**: CPU (optimal for differentiable simulation)
- **Agents**: 160 (can handle larger batches)
- **Gradients**: Required (`requires_grad: true`)
- **Learning Steps**: 10,000 (fast convergence)

### PPO Configuration
- **Device**: CUDA (benefits from GPU parallelization)
- **Agents**: 48 (balanced for GPU memory)
- **Gradients**: Not required (`requires_grad: false`)
- **Learning Steps**: 10,000,000 (needs more samples)

### SHAC Configuration
- **Device**: CPU (temporal difference learning)
- **Agents**: 160 (similar to BPTT)
- **Gradients**: Required (`requires_grad: true`)
- **Learning Steps**: 50,000 (moderate sample efficiency)

## Adding New Environments

To add a new environment (e.g., "racing"):

1. **Create environment config**:
   ```bash
   mkdir -p configs/envs/racing
   # Create configs/envs/racing/env.yaml
   ```

2. **Create algorithm configs**:
   ```bash
   mkdir -p configs/algs/racing
   # Create configs/algs/racing/bptt.yaml
   # Create configs/algs/racing/ppo.yaml
   # Create configs/algs/racing/shac.yaml
   ```

3. **Use the new environment**:
   ```bash
   python quadro_llm/training/visfly_training_wrapper_v2.py --env racing --algorithm bptt
   ```

## Integration with Eureka Pipeline

The config system integrates seamlessly with Eureka:

```python
# In eureka pipeline
training_command = [
    "python", "quadro_llm/training/visfly_training_wrapper_v2.py",
    "--env", "navigation",
    "--algorithm", "bptt",
    "--reward_function_path", reward_file_path,
    "--learning_steps", "10000",
    "--comment", f"iter_{iteration}_func_{func_id}"
]
```

## Benefits of Config-Based System

1. **Clear Separation**: Environment vs algorithm settings
2. **Easy Maintenance**: No hardcoded parameters in code
3. **Experiment Reproducibility**: All settings in version-controlled files
4. **Algorithm Comparison**: Easy to switch between algorithms
5. **Parameter Tuning**: Modify configs without code changes
6. **Extensibility**: Add new environments/algorithms by adding config files

## Config File Examples

### Minimal Algorithm Override

```yaml
# configs/algs/navigation/my_bptt.yaml
algorithm:
  learning_rate: 0.01  # Custom learning rate
  horizon: 64         # Custom horizon

env_overrides:
  num_agent_per_scene: 32  # Fewer agents

learn:
  total_timesteps: 5000    # Shorter training
```

### Custom Environment

```yaml
# configs/envs/navigation/fast_env.yaml
env:
  num_agent_per_scene: 16    # Fewer agents for faster training
  max_episode_steps: 128     # Shorter episodes
  visual: false              # Disable rendering
  target: [10.0, 0.0, 1.0]   # Closer target
```

## Troubleshooting

### Config File Not Found
```
FileNotFoundError: Environment config not found: configs/envs/navigation/env.yaml
```
**Solution**: Ensure config files exist in the correct directory structure.

### Parameter Override Not Working
**Solution**: Check that parameter names match exactly those in the config files.

### Algorithm Creation Error
**Solution**: Verify that all required parameters are present in the algorithm config.

This config-based approach makes the training system much more maintainable and follows established patterns from successful RL projects!