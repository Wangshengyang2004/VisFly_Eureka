#!/usr/bin/env python3
"""
VisFly Training Wrapper for Eureka Pipeline

This wrapper allows configuring VisFly training parameters (like learning steps)
without modifying the original VisFly codebase. It imports VisFly components
and runs training with Eureka-specific configurations.
"""

import sys
import os
import time
import torch
import torch as th
import numpy as np
import traceback
import argparse

# Add VisFly to path
sys.path.append('/home/simonwsy/VisFly_Eureka/VisFly')
sys.path.append('/home/simonwsy/VisFly_Eureka')

# Import VisFly components
from VisFly.utils.launcher import training_params
from VisFly.utils.policies import extractors
from VisFly.utils.algorithms.BPTT import BPTT
from VisFly.utils import savers
from VisFly.envs.NavigationEnv import NavigationEnv
from VisFly.utils.type import Uniform

# Disable gradient anomaly detection for physics-based BPTT
torch.autograd.set_detect_anomaly(False)

def parse_args():
    """Parse command line arguments for Eureka training"""
    parser = argparse.ArgumentParser(description='VisFly Training Wrapper for Eureka')
    parser.add_argument('--train', '-t', type=int, default=1, help='Training mode (1) or test mode (0)')
    parser.add_argument('--comment', '-c', type=str, default='eureka_experiment', help='Experiment comment')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
    parser.add_argument('--learning_steps', type=int, default=10000, help='Number of learning steps')
    parser.add_argument('--num_agents', type=int, default=160, help='Number of parallel agents')
    parser.add_argument('--max_episode_steps', type=int, default=256, help='Maximum episode steps')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--horizon', type=int, default=96, help='BPTT horizon')
    parser.add_argument('--reward_function_path', type=str, default=None, help='Path to custom reward function')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save results')
    
    return parser.parse_args()

def inject_reward_function(reward_function_code: str, env_class):
    """
    Inject custom reward function into the NavigationEnv class.
    
    Args:
        reward_function_code: Python code for the get_reward method
        env_class: The NavigationEnv class to modify
    """
    if reward_function_code:
        try:
            # Create a new get_reward method from the provided code
            exec_globals = {
                'torch': torch,
                'th': th,
                'np': np,
                'numpy': np
            }
            
            # Execute the reward function code to create the method
            exec(f"def get_reward(self):\n{reward_function_code}", exec_globals)
            
            # Replace the method in the class
            env_class.get_reward = exec_globals['get_reward']
            
            print(f"✅ Successfully injected custom reward function")
            
        except Exception as e:
            print(f"❌ Failed to inject reward function: {e}")
            traceback.print_exc()
            raise

def setup_training_environment(args):
    """Setup the training environment with Eureka-specific parameters"""
    
    # Configure training parameters based on arguments
    training_params["num_agent_per_scene"] = args.num_agents
    training_params["learning_step"] = args.learning_steps  # This is the key override!
    training_params["comment"] = args.comment
    training_params["seed"] = args.seed
    training_params["max_episode_steps"] = args.max_episode_steps
    training_params["learning_rate"] = args.learning_rate
    training_params["horizon"] = args.horizon
    training_params["dump_step"] = 50
    
    print(f"🎯 Training Configuration:")
    print(f"  Learning steps: {training_params['learning_step']}")
    print(f"  Agents: {training_params['num_agent_per_scene']}")
    print(f"  Episode steps: {training_params['max_episode_steps']}")
    print(f"  Learning rate: {training_params['learning_rate']}")
    print(f"  Comment: {training_params['comment']}")
    
    # Setup save directory
    if args.save_dir:
        save_folder = args.save_dir
    else:
        save_folder = os.path.dirname(os.path.abspath(__file__)) + "/saved/"
    
    os.makedirs(save_folder, exist_ok=True)
    
    return save_folder

def run_training(args):
    """Run the VisFly training with configured parameters"""
    
    print(f"🚀 Starting VisFly training with Eureka configuration...")
    
    # Setup training environment
    save_folder = setup_training_environment(args)
    
    # Load custom reward function if provided
    if args.reward_function_path and os.path.exists(args.reward_function_path):
        with open(args.reward_function_path, 'r') as f:
            reward_code = f.read()
        inject_reward_function(reward_code, NavigationEnv)
    
    # Random initialization for environment resets (from original bptt.py)
    random_kwargs = {
        "state_generator": {
            "class": "Uniform",
            "kwargs": [
                {
                    "position": {"mean": [6., -2., 2.], "half": [.50, .50, .50]},
                }
            ]
        }
    }
    
    # Scene configuration for visual rendering
    scene_kwargs = {
        "path": "datasets/visfly-beta/configs/scenes/box15_wall_box15_wall"
    }
    
    # Environment configuration
    env_config = {
        "num_agent_per_scene": training_params["num_agent_per_scene"],
        "num_scene": 1,
        "visual": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "requires_grad": True,  # Enable BPTT
        "max_episode_steps": training_params["max_episode_steps"],
        "sensor_kwargs": [{
            "sensor_type": "DEPTH",
            "uuid": "depth",
            "resolution": [64, 64],
        }],
        "target": torch.tensor([[15.0, 0.0, 1.5]]),
        "random_kwargs": random_kwargs,
        "scene_kwargs": scene_kwargs
    }
    
    try:
        # Create environment
        print("🌍 Creating NavigationEnv...")
        env = NavigationEnv(**env_config)
        
        # Create BPTT algorithm (following original bptt.py configuration)
        print("🧠 Creating BPTT algorithm...")
        model = BPTT(
            env=env,
            policy="MultiInputPolicy",
            policy_kwargs=dict(
                features_extractor_class=extractors.FlexibleExtractor,
                features_extractor_kwargs=dict(
                    net_arch=dict(
                        depth=dict(layer=[128]),
                        state=dict(layer=[128, 64]),
                    ),
                    activation_fn=torch.nn.ReLU,
                ),
                net_arch=dict(pi=[64, 64], qf=[64, 64]),
                activation_fn=torch.nn.ReLU,
                optimizer_kwargs=dict(weight_decay=1e-5),
            ),
            learning_rate=training_params["learning_rate"],
            comment=training_params["comment"],
            save_path=save_folder,
            horizon=int(training_params["horizon"]),
            gamma=training_params.get("gamma", 0.99),
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=int(training_params["seed"]),
            dump_step=int(training_params.get("dump_step", 50)),
        )
        
        if args.train:
            # Train the model
            print(f"🏃 Starting training for {training_params['learning_step']} steps...")
            start_time = time.time()
            
            model.learn(int(training_params["learning_step"]))
            model.save()
            
            training_time = time.time() - start_time
            training_params["time"] = training_time
            
            print(f"✅ Training completed in {training_time:.2f} seconds")
            
            # Save training parameters
            import json
            params_file = os.path.join(save_folder, "training_params.json")
            with open(params_file, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_params = {}
                for k, v in training_params.items():
                    if isinstance(v, (np.integer, np.floating)):
                        json_params[k] = v.item()
                    else:
                        json_params[k] = v
                json.dump(json_params, f, indent=2)
            
            print(f"💾 Saved training parameters to {params_file}")
        else:
            print("🧪 Test mode - skipping training")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    print("=" * 60)
    print("🚁 VisFly Training Wrapper for Eureka Pipeline")
    print("=" * 60)
    
    args = parse_args()
    
    try:
        success = run_training(args)
        if success:
            print("✅ Training completed successfully!")
            return 0
        else:
            print("❌ Training failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Wrapper failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())