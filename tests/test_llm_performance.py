#!/usr/bin/env python3
"""
Quick performance test for LLM configurations.
Supports testing different models via CLI arguments.
"""

import time
import logging
import yaml
import argparse
from pathlib import Path
import sys

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()  # Go up one level from tests/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "VisFly"))

from quadro_llm.llm.llm_engine import LLMEngine

# Import environment for testing
try:
    from envs.NavigationEnv import NavigationEnv
    TEST_ENV_CLASS = NavigationEnv
except ImportError:
    TEST_ENV_CLASS = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_llm_config(model_name="glm-4.5"):
    """Load LLM configuration for specified model"""
    config_path = PROJECT_ROOT / f"configs/llm/{model_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        llm_cfg = yaml.safe_load(f)
    
    # Load API keys
    api_keys_path = PROJECT_ROOT / "configs/api_keys.yaml"
    with open(api_keys_path, 'r') as f:
        api_config = yaml.safe_load(f)
        vendor = llm_cfg["llm"]["vendor"]
        vendor_config = api_config["providers"][vendor]
    
    # Combine configuration
    config = {
        "model": llm_cfg["llm"]["model"],
        "api_key": vendor_config["api_key"],
        "base_url": vendor_config["base_url"],
        "temperature": llm_cfg["llm"]["temperature"],
        "max_tokens": llm_cfg["llm"]["max_tokens"],
        "timeout": llm_cfg["llm"]["timeout"],
        "max_retries": llm_cfg["llm"]["max_retries"],
    }
    
    # Add optional configurations
    if "thinking" in llm_cfg["llm"] and "enabled" in llm_cfg["llm"]["thinking"]:
        config["thinking_enabled"] = llm_cfg["llm"]["thinking"]["enabled"]
    
    if "batching" in llm_cfg["llm"]:
        batching = llm_cfg["llm"]["batching"]
        if "strategy" in batching:
            config["batching_strategy"] = batching["strategy"]
        if "supports_n_parameter" in batching:
            config["supports_n_parameter"] = batching["supports_n_parameter"]
        if "max_concurrent" in batching:
            config["max_concurrent"] = batching["max_concurrent"]
    
    return config


def test_api_connection(model_name="glm-4.5"):
    """Test API connection for specified model"""
    logger.info(f"Testing {model_name} API connection...")
    
    try:
        config = load_llm_config(model_name)
        llm = LLMEngine(**config)
        
        start_time = time.time()
        llm.test_api_connection()
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"✓ Connection test: {duration:.2f}s")
        return True, duration
        
    except Exception as e:
        logger.error(f"✗ Connection test failed: {e}")
        return False, 0


def test_reward_generation(model_name="glm-4.5", samples=3):
    """Test reward function generation performance"""
    logger.info(f"Testing {model_name} reward generation with {samples} samples...")
    
    try:
        config = load_llm_config(model_name)
        llm = LLMEngine(**config)
        
        task_description = "Navigate to target position while avoiding obstacles"
        context_info = {
            "observation_space": "position, velocity, orientation",
            "action_space": "thrust, roll, pitch, yaw",
            "environment": "3D quadrotor simulation"
        }
        
        start_time = time.time()
        response = llm.generate_reward_functions(
            task_description=task_description,
            context_info=context_info,
            feedback="",
            samples=samples,
            env_class=TEST_ENV_CLASS
        )
        end_time = time.time()
        
        duration = end_time - start_time
        success_rate = len(response) / samples * 100 if samples > 0 else 0
        
        logger.info(f"✓ Generation: {duration:.2f}s, {len(response)}/{samples} samples ({success_rate:.1f}%)")
        
        if response:
            avg_length = sum(len(r) for r in response) / len(response)
            logger.info(f"  Average function length: {avg_length:.0f} characters")
        
        return True, duration, len(response)
        
    except Exception as e:
        logger.error(f"✗ Generation failed: {e}")
        return False, 0, 0


def benchmark_model(model_name, samples=3):
    """Benchmark a model's performance"""
    logger.info("="*60)
    logger.info(f"BENCHMARKING MODEL: {model_name.upper()}")
    logger.info("="*60)
    
    # Test 1: API Connection
    logger.info("\n" + "-"*30)
    logger.info("1. API Connection Test")
    logger.info("-"*30)
    conn_success, conn_time = test_api_connection(model_name)
    
    if not conn_success:
        logger.error("Skipping generation test due to connection failure")
        return
    
    # Test 2: Reward Generation
    logger.info("\n" + "-"*30)
    logger.info("2. Reward Generation Test")
    logger.info("-"*30)
    gen_success, gen_time, gen_count = test_reward_generation(model_name, samples)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK RESULTS")
    logger.info("="*60)
    
    if conn_success:
        logger.info(f"Connection Time: {conn_time:.2f}s")
    
    if gen_success:
        avg_time_per_sample = gen_time / samples if samples > 0 else 0
        logger.info(f"Generation Time: {gen_time:.2f}s ({avg_time_per_sample:.2f}s per sample)")
        logger.info(f"Success Rate: {gen_count}/{samples} ({gen_count/samples*100:.1f}%)")
        
        # Performance rating
        if avg_time_per_sample < 5:
            logger.info("⚡ Performance: EXCELLENT")
        elif avg_time_per_sample < 15:
            logger.info("✅ Performance: GOOD")
        elif avg_time_per_sample < 60:
            logger.info("⚠️  Performance: ACCEPTABLE")
        else:
            logger.info("❌ Performance: POOR")
    
    # Show configuration
    try:
        config = load_llm_config(model_name)
        logger.info(f"\nConfiguration:")
        logger.info(f"- Model: {config.get('model')}")
        logger.info(f"- Temperature: {config.get('temperature')}")
        logger.info(f"- Max Tokens: {config.get('max_tokens')}")
        logger.info(f"- Batching: {config.get('batching_strategy', 'n_parameter')}")
        logger.info(f"- Thinking: {'disabled' if not config.get('thinking_enabled', True) else 'enabled'}")
    except Exception:
        pass
    
    logger.info("="*60)


def main():
    """Main entry point with CLI argument support"""
    parser = argparse.ArgumentParser(
        description="Test LLM performance",
        epilog="Examples:\n"
               "  python test_llm_performance.py --model glm-4.5\n"
               "  python test_llm_performance.py --model gpt-4o --samples 5\n"
               "  python test_llm_performance.py --list-models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", "-m", 
        default="glm-4.5",
        help="Model configuration to test (default: glm-4.5)"
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=3,
        help="Number of reward function samples to generate (default: 3)"
    )
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="List available model configurations"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        configs_dir = PROJECT_ROOT / "configs/llm"
        if configs_dir.exists():
            logger.info("Available model configurations:")
            for config_file in configs_dir.glob("*.yaml"):
                logger.info(f"  - {config_file.stem}")
        else:
            logger.error(f"No model configurations found in {configs_dir}")
        return
    
    # Validate model exists
    config_path = PROJECT_ROOT / f"configs/llm/{args.model}.yaml"
    if not config_path.exists():
        logger.error(f"Model configuration not found: {args.model}")
        logger.info("Use --list-models to see available configurations")
        sys.exit(1)
    
    # Run benchmark
    try:
        benchmark_model(args.model, args.samples)
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()