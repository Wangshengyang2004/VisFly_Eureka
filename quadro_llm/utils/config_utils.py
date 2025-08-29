"""
Configuration utilities for VisFly-Eureka.
"""

import yaml
import os
from typing import Optional


def load_api_config() -> Optional[str]:
    """Load API configuration from file or environment"""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "configs",
        "api_keys.yaml",
    )
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            api_config = yaml.safe_load(f)
            return api_config.get("yunwu", {}).get("api_key")
    return os.getenv("YUNWU_API_KEY")


def get_default_llm_config() -> dict:
    """Get default LLM configuration"""
    return {
        "model": "gpt-4o",
        "api_key": load_api_config(),
        "base_url": "https://yunwu.ai/v1",
        "temperature": 0.8,
        "max_tokens": 1500,
        "timeout": 120,
        "max_retries": 3,
    }
