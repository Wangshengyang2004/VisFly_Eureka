# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

VisFly_Eureka is a project that combines VisFly (a versatile quadrotor simulator for visual-based flight) with Eureka (an LLM-powered reward function optimization framework). The goal is to reproduce Eureka functionality in a professional UAV simulator environment.

### Key Components

- **VisFly/**: Main simulator package for visual-based drone flight
- **backup/Eureka/**: Original Eureka codebase for reference
- **SPEC.md**: Migration specification document

## Environment Setup

### Prerequisites

The project uses conda environments with specific versions. Two environment files are available:

```bash
# For Python 3.9 (primary)
conda env create -f VisFly/environment.yml

# For Python 3.10 (alternative)
conda env create -f VisFly/environment-ng.yaml
```

### Additional Dependencies

- Install CGAL (Computational Geometry Algorithms Library):
  ```bash
  sudo apt-get install libcgal-dev
  ```
- Modified habitat-sim must be built from source following the instructions in the VisFly README

### Dataset Setup

Download the visfly-beta dataset from Hugging Face:
```bash
cd VisFly/datasets
git clone https://YourUsername:YourAccessToken@huggingface.co/datasets/LiFanxing/visfly-beta.git
```

## Development Commands

### Training Examples

```bash
# Train a navigation agent
python examples/navigation/bptt.py -t 1

# Test a trained model
python examples/navigation/bptt.py -t 0 -w MODEL_NAME

# Train with custom comment/identifier
python examples/navigation/bptt.py -t 1 -c "custom_experiment"
```

### Common Arguments

- `-t`: Training mode (1) or testing mode (0)
- `-w`: Model weight path to load
- `-c`: Comment/identifier for saving models

## Architecture Overview

### Core Components

1. **Environment System** (`VisFly/envs/`)
   - Base classes: `droneEnv.py`, `droneGymEnv.py`
   - Specific environments: `NavigationEnv.py`, `RacingEnv.py`, `HoverEnv.py`, etc.
   - Multi-agent support: `multiDroneGymEnv.py`

2. **Dynamics and Control** (`VisFly/envs/base/`)
   - `dynamics.py`: Physics simulation
   - `controller.py`: Control algorithms
   
3. **Policy Networks** (`VisFly/utils/policies/`)
   - Feature extractors for various sensor inputs
   - Support for CNN, MLP, and recurrent architectures
   - Custom extractors can be added in `extractors.py`

4. **Algorithms** (`VisFly/utils/algorithms/`)
   - `BPTT.py`: Back-propagation through time for differentiable simulation
   - `PPO.py`, `SAC.py`: Standard RL algorithms

5. **Scene Management** (`VisFly/utils/`)
   - `SceneManager.py`: Environment scene handling
   - `ObjectManger.py`: Dynamic object management

### Key Features

- **Differentiable Simulation**: Supports BPTT training with analytical gradients
- **Visual Rendering**: High FPS rendering (up to 10kHz) with habitat-sim backend  
- **Flexible Environments**: Easy customization of reward functions and observations
- **Multi-sensor Support**: Depth cameras, RGB cameras, IMU data
- **Scene Datasets**: Rich 3D environments from Replica and custom scenes

## Environment Configuration

### Basic Environment Setup

```python
env = NavigationEnv(
    num_agent_per_scene=1,
    num_scene=1,
    visual=True,
    device="cuda",
    max_episode_steps=256,
    dynamics_kwargs={
        "action_type": "bodyrate",  # ["bodyrate", "thrust", "velocity", "position"]
        "dt": 0.0025,  # simulation timestep
        "ctrl_dt": 0.02,  # control timestep
    },
    scene_kwargs={
        "path": "datasets/visfly-beta/configs/scenes/SCENE_NAME"
    },
    sensor_kwargs=[{
        "sensor_type": habitat_sim.SensorType.DEPTH,
        "uuid": "depth",
        "resolution": [64, 64],
    }],
    target=torch.tensor([9, 0., 1]),  # target position
)
```

### Custom Environment Creation

To create new environments:
1. Inherit from `DroneGymEnvsBase`
2. Implement `get_observation()`, `get_reward()`, `get_success()`
3. Add custom observation spaces as needed

## Development Workflow

1. **Environment Development**: Create/modify environments in `VisFly/envs/`
2. **Policy Design**: Customize feature extractors in `VisFly/utils/policies/extractors.py`
3. **Training**: Use BPTT for differentiable simulation or standard RL algorithms
4. **Evaluation**: Test trained models with evaluation scripts

## Important Notes

- GPU acceleration is recommended for visual environments
- For fewer than 1000 environments, CPU may be faster than GPU
- Differentiable simulation requires `requires_grad=True` in environment setup
- Models are saved with format: `ALGORITHM_COMMENT_INDEX.zip`
- Use tensorboard for training monitoring

## Migration from Eureka

The project aims to integrate Eureka's reward function optimization with VisFly's simulation capabilities. The original Eureka code is preserved in `backup/Eureka/` for reference during implementation.