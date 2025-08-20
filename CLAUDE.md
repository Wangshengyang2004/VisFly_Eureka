# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

VisFly_Eureka integrates VisFly (a high-performance visual-based quadrotor simulator) with Eureka (LLM-powered reward optimization). The project implements native Eureka functionality directly into VisFly environments for automated reward function generation and optimization.

### Project Structure

- **VisFly/**: Core simulator with differentiable dynamics and visual rendering
  - `envs/`: Environment implementations (NavigationEnv, RacingEnv, HoverEnv, etc.)
  - `utils/algorithms/`: BPTT, PPO, SAC implementations
  - `utils/policies/`: Feature extractors and policy networks
  - `examples/`: Training scripts and demos
- **backup/Eureka/**: Reference Eureka implementation
- **eureka_visfly/**: Native Eureka-VisFly integration (to be implemented)
- **SPEC.md**: Native integration specification

## Development Commands

### Setup and Installation

```bash
# Install system dependencies
sudo apt-get install libcgal-dev

# Create conda environment (Python 3.9 or 3.10)
conda env create -f VisFly/environment.yml
# or
conda env create -f VisFly/environment-ng.yaml

# Install modified habitat-sim from source
git clone https://github.com/Fanxing-LI/habitat-sim
cd habitat-sim
# Follow habitat-sim build instructions

# Download visfly-beta dataset
cd VisFly/datasets
git clone https://YourUsername:YourAccessToken@huggingface.co/datasets/LiFanxing/visfly-beta.git

# Install project
pip install -e .
```

### Training and Testing

```bash
# BPTT training (differentiable simulation)
python VisFly/examples/navigation/bptt.py -t 1 -c "experiment_name"
python VisFly/examples/diff_hovering/bptt.py -t 1

# PPO/SAC training
python VisFly/examples/cluttered_flight/rl.py -t 1 -c "ppo_experiment"

# Test trained models
python VisFly/examples/navigation/bptt.py -t 0 -w "MODEL_NAME"

# Common arguments:
# -t: 1=train, 0=test
# -w: model weights path
# -c: comment/identifier for saved models
```

### Monitoring

```bash
# Launch tensorboard for training monitoring
tensorboard --logdir VisFly/examples/*/saved/
```

## High-Level Architecture

### Native Eureka-VisFly Integration

The native integration allows direct reward function injection without adapter layers:

```python
# Direct reward injection pattern
def inject_generated_reward(env_instance, reward_code):
    exec(reward_code, {'torch': torch, 'th': torch})
    env_instance.get_reward = types.MethodType(exec_globals['get_reward'], env_instance)

# Usage
nav_env = NavigationEnv(...)
inject_generated_reward(nav_env, llm_generated_code)
```

### Key VisFly Environment Properties

Accessible in reward functions via `self`:
- `position`: torch.Tensor [N, 3] - drone positions
- `velocity`: torch.Tensor [N, 3] - linear velocities
- `orientation`: torch.Tensor [N, 4] - quaternions
- `angular_velocity`: torch.Tensor [N, 3]
- `target`: torch.Tensor [N, 3] - target positions
- `sensor_obs`: dict with 'depth'/'rgb' sensor data
- `collision_dis`: torch.Tensor [N] - obstacle distances
- `collision_point`, `collision_vector`: collision info
- `_step_count`: int - current episode step
- `max_episode_steps`: int - episode limit
- `_success`: bool - success flag

### Training Algorithms

1. **BPTT** (Back-propagation through time)
   - Requires `requires_grad=True` in env setup
   - Uses analytical gradients through differentiable simulation
   - Best for precise control tasks

2. **PPO/SAC** (Standard RL)
   - Works with any environment configuration
   - Better for exploration-heavy tasks

### Custom Environment Template

```python
from VisFly.envs.droneGymEnv import DroneGymEnvsBase
import torch as th
from gym import spaces

class CustomEnv(DroneGymEnvsBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom observation spaces
        self.observation_space["custom"] = spaces.Box(...)
        
    def get_observation(self, indices=None):
        return {
            "state": self.sensor_obs["IMU"].cpu().numpy(),
            "depth": self.sensor_obs["depth"],
            "custom": self.custom_obs,
        }
    
    def get_reward(self) -> th.Tensor:
        # Return tensor of shape [num_agents]
        return reward_tensor
    
    def get_success(self) -> th.Tensor:
        # Return boolean tensor of shape [num_agents]
        return success_tensor
```

## Eureka Integration Workflow

1. **Generate Rewards**: LLM creates reward functions based on task description
2. **Inject Functions**: Direct injection into VisFly environment instances
3. **Train & Evaluate**: Use BPTT or PPO to train with generated rewards
4. **Iterate**: Refine rewards based on training performance

### Example Eureka Usage (once implemented)

```python
from eureka_visfly import EurekaVisFly
from VisFly.envs.NavigationEnv import NavigationEnv

eureka = EurekaVisFly(
    env_class=NavigationEnv,
    task_description="Navigate to target avoiding obstacles",
    llm_config={"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}
)

best_rewards = eureka.optimize_rewards(iterations=5, samples=16)
```

## Performance Tips

- **GPU vs CPU**: GPU recommended for visual environments; CPU may be faster for <1000 envs
- **Batch Size**: Use `num_agent_per_scene` to control parallel environments
- **Memory**: Reduce batch size if encountering GPU OOM errors
- **Differentiable Mode**: Set `torch.autograd.set_detect_anomaly(False)` for performance

## Testing

```bash
# Run unit tests (once implemented)
pytest tests/

# Test specific components
pytest tests/test_reward_injection.py -v
pytest tests/test_training.py -v
```

## Important File Locations

- Training scripts: `VisFly/examples/*/`
- Environment definitions: `VisFly/envs/`
- Feature extractors: `VisFly/utils/policies/extractors.py`
- BPTT implementation: `VisFly/utils/algorithms/BPTT.py`
- Scene configs: `VisFly/datasets/visfly-beta/configs/`
- Saved models: `VisFly/examples/*/saved/`