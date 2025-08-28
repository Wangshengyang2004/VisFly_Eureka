# Reward Injection Workflow Documentation

## Overview

This document explains how the VisFly-Eureka system is designed to run after reward function injection. The system integrates Eureka's LLM-generated reward functions directly into VisFly environments for training and evaluation.

## System Architecture

### 1. **Project Structure**
```
VisFly_Eureka/
├── envs/                           # Project-specific environments (obj_track quality)
├── VisFly/envs/                    # Core VisFly environments (updated with obj_track)
├── configs/
│   ├── envs/                       # Environment configs (navigation.yaml, hover.yaml, etc.)
│   └── algs/                       # Algorithm configs by environment
├── run.py                          # Unified training/testing entry point
├── quadro_llm/training/            # Eureka pipeline integration
└── generated_rewards/              # LLM-generated reward functions
```

### 2. **Available Environments**

| Environment | Description | Use Case |
|-------------|-------------|----------|
| **NavigationEnv** | Navigate to target avoiding obstacles | Complex navigation tasks |
| **EasyNavigationEnv** | Navigate in open space | Simple navigation learning |
| **HoverEnv** | Maintain stable hovering | Stability and control tasks |
| **ObjectTrackingEnv** | Track moving objects | Visual tracking tasks |

## Reward Injection Process

### Step 1: **Eureka Generation**
```python
# Eureka pipeline generates reward functions
eureka = EurekaVisFly(
    env_class=NavigationEnv,
    task_description="Navigate to target avoiding obstacles",
    llm_config=llm_config
)

best_rewards = eureka.optimize_rewards(iterations=5, samples=15)
```

### Step 2: **Reward Function Files**
Generated reward functions are saved as Python files:
```python
# generated_rewards/iteration_1/reward_function_03.py
def get_reward(self) -> th.Tensor:
    """LLM-generated reward function"""
    # Distance to target reward
    distance_reward = -0.1 * (self.position - self.target).norm(dim=1)
    
    # Collision penalty using depth sensor
    collision_penalty = -5.0 * (self.sensor_obs["depth"] < 0.3).float().sum(dim=[1,2])
    
    # Velocity smoothness
    velocity_penalty = -0.01 * self.velocity.norm(dim=1)
    
    return distance_reward + collision_penalty + velocity_penalty
```

### Step 3: **Dynamic Injection**
The system injects reward functions at runtime:
```python
def inject_reward_function(reward_function_path: str, env_class):
    """Inject custom reward function into environment class"""
    with open(reward_function_path, 'r') as f:
        reward_code = f.read()
    
    exec_globals = {'torch': th, 'th': th, 'np': np}
    exec(reward_code, exec_globals)
    
    # Replace the class method
    env_class.get_reward = exec_globals['get_reward']
```

## Running the System After Injection

### Method 1: **Using run.py (Recommended)**

```bash
# Train with injected reward function
python run.py \
  --env navigation \
  --algorithm bptt \
  --train 1 \
  --reward_function_path generated_rewards/iteration_1/reward_function_03.py \
  --comment "eureka_iter_1_func_3"

# Test trained model
python run.py \
  --env navigation \
  --algorithm bptt \
  --train 0 \
  --weight "BPTT_eureka_iter_1_func_3_1.pth" \
  --comment "test_eureka_func_3"
```

### Method 2: **Using Training Wrapper**

```bash
# Alternative training method
python quadro_llm/training/visfly_training_wrapper.py \
  --env navigation \
  --algorithm bptt \
  --reward_function_path generated_rewards/best_reward.py \
  --learning_steps 10000
```

### Method 3: **Programmatic Usage**

```python
from run import run_training, inject_reward_function
from envs.NavigationEnv import NavigationEnv
import argparse

# Inject reward function
inject_reward_function("generated_rewards/best_reward.py", NavigationEnv)

# Create args object
args = argparse.Namespace(
    env='navigation',
    algorithm='bptt',
    train=1,
    comment='programmatic_run',
    reward_function_path='generated_rewards/best_reward.py'
)

# Run training
success = run_training(args)
```

## Workflow Comparison with obj_track

### **obj_track Workflow**
```bash
# obj_track approach
cd exps/navi
python run.py --algorithm BPTT --comment experiment_1
```

### **VisFly-Eureka Workflow (Post-Injection)**
```bash
# Our approach after Eureka generates rewards
python run.py \
  --env navigation \
  --algorithm bptt \
  --reward_function_path generated_rewards/iteration_2/reward_function_07.py \
  --comment "eureka_iter_2_func_7"
```

## Key Differences and Advantages

### **1. Dynamic Reward Injection**
- **obj_track**: Fixed reward functions in environment classes
- **VisFly-Eureka**: Runtime injection of LLM-generated rewards

### **2. Unified Entry Point**
- **obj_track**: Separate run.py per experiment
- **VisFly-Eureka**: Single run.py for all environments/algorithms

### **3. Config-Driven**
- Both systems use YAML configs
- VisFly-Eureka has flattened structure: `configs/envs/navigation.yaml`

### **4. Eureka Integration**
- Seamless integration with reward optimization pipeline
- Automatic reward function management
- Training metadata tracking

## Complete Training Pipeline

### **1. Initial Setup**
```python
# Pipeline creates environments and baseline
pipeline = EurekaNavigationPipeline(
    task_description="Navigate to target avoiding obstacles",
    llm_config=llm_config
)
```

### **2. Eureka Optimization**
```python
# Generate and evaluate reward functions
results = pipeline.run_optimization()  # 5 iterations × 15 samples = 75 functions tested
```

### **3. Best Reward Selection**
```python
# System selects best performing reward function
best_reward = results.best_reward_code
best_score = results.best_performance["score"]
```

### **4. Production Training**
```bash
# Train final model with best reward
python run.py \
  --env navigation \
  --algorithm bptt \
  --reward_function_path generated_rewards/best_reward.py \
  --learning_steps 50000 \
  --comment "production_model"
```

### **5. Evaluation**
```bash
# Test trained model
python run.py \
  --env navigation \
  --algorithm bptt \
  --train 0 \
  --weight "BPTT_production_model_1.pth"
```

## Environment-Specific Considerations

### **Navigation Environments**
- **Sensors**: Depth camera for obstacle detection
- **Rewards**: Distance, collision avoidance, trajectory smoothness
- **Config**: `configs/envs/navigation.yaml`

### **Hover Environments** 
- **Sensors**: IMU only (no visual sensors)
- **Rewards**: Position stability, velocity minimization
- **Config**: `configs/envs/hover.yaml`

### **Tracking Environments**
- **Sensors**: RGB + depth cameras
- **Rewards**: Object tracking accuracy, smooth following
- **Config**: More complex sensor setup

## Error Handling and Debugging

### **Common Issues**
1. **Reward Function Syntax Errors**
   - System validates reward function before injection
   - Provides clear error messages

2. **Device Mismatches**
   - Configs specify device (CPU/GPU) per algorithm
   - Automatic tensor device alignment

3. **Missing Dependencies**
   - System checks for required sensors/observations
   - Falls back to default configurations

### **Debug Mode**
```bash
# Run with debug output
python run.py --env navigation --algorithm bptt --train 1 --verbose
```

## Performance Monitoring

### **Training Metrics**
- TensorBoard logs in results directory
- JSON metadata files with training statistics
- Reward function performance tracking

### **Evaluation Metrics**
- Success rates across environments
- Episode lengths and completion times
- Trajectory quality metrics

## Integration Points

### **1. Eureka Pipeline Integration**
- Direct reward injection into training wrapper
- Automated evaluation and ranking
- Best reward selection and deployment

### **2. VisFly Integration**
- Uses native VisFly environments and algorithms
- Preserves all VisFly functionality
- Compatible with existing VisFly workflows

### **3. Config System Integration**
- Environment-specific parameter tuning
- Algorithm-specific optimizations
- Easy parameter override system

This architecture provides a clean separation between reward generation (Eureka) and reward execution (VisFly), while maintaining the flexibility and performance of both systems.