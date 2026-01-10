"""
Configuration utilities for VisFly-Eureka.
"""

import yaml
import os
from pathlib import Path
from typing import Optional, Any, Dict, List


def load_api_config():
    """Load API configuration from file"""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "configs",
        "api_keys.yaml",
    )

    with open(config_path, "r") as f:
        api_config = yaml.safe_load(f)

    default_provider = api_config["providers"]["default"]
    provider_config = api_config["providers"][default_provider]

    api_key = provider_config["api_key"]
    base_url = provider_config["base_url"]

    if not api_key:
        raise ValueError(f"API key is required but not configured for provider {default_provider}")
    if not base_url:
        raise ValueError(f"Base URL is required but not configured for provider {default_provider}")

    return {"api_key": api_key, "base_url": base_url, "provider": default_provider}


def get_default_llm_config() -> dict:
    """Get default LLM configuration"""
    api_config = load_api_config()
    return {
        "model": "gpt-4o",
        "api_key": api_config["api_key"],
        "base_url": api_config["base_url"],
        "temperature": 0.8,
        "max_tokens": 1500,
        "timeout": 120,
        "max_retries": 3,
    }


class ConfigManager:
    """
    Backwards-compatible configuration manager used by unit/integration tests.

    The project primarily relies on Hydra for configuration, but this helper
    provides lightweight file-based loading/merging for scripts and tests.
    """

    def __init__(self, config_root: str):
        self.config_root = Path(config_root)

    def _read_yaml(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def load_env_config(self, env_name: str) -> Optional[Dict[str, Any]]:
        return self._read_yaml(self.config_root / "envs" / f"{env_name}.yaml")

    def load_alg_config(self, env_name: str, algorithm: str) -> Optional[Dict[str, Any]]:
        return self._read_yaml(self.config_root / "algs" / env_name / f"{algorithm}.yaml")

    def get_merged_config(self, env_name: str, algorithm: str) -> Optional[Dict[str, Any]]:
        env_cfg = self.load_env_config(env_name)
        alg_cfg = self.load_alg_config(env_name, algorithm)
        if env_cfg is None or alg_cfg is None:
            return None
        return self.merge_configs(env_cfg, alg_cfg)

    def merge_configs(self, env_config: Dict[str, Any], alg_config: Dict[str, Any]) -> Dict[str, Any]:
        merged = yaml.safe_load(yaml.dump(env_config))  # simple deep copy
        merged.setdefault("algorithm", {})
        merged["algorithm"] = alg_config.get("algorithm", {})

        overrides = alg_config.get("env_overrides", {})
        merged.setdefault("env", {})
        merged["env"].update(overrides)
        return merged

    def list_available_envs(self) -> List[str]:
        env_dir = self.config_root / "envs"
        if not env_dir.exists():
            return []
        return sorted(p.stem for p in env_dir.glob("*.yaml"))

    def list_available_algs(self, env_name: str) -> List[str]:
        alg_dir = self.config_root / "algs" / env_name
        if not alg_dir.exists():
            return []
        return sorted(p.stem for p in alg_dir.glob("*.yaml"))

    def validate_env_config(self, config: Dict[str, Any]) -> None:
        if "env" not in config or "eval_env" not in config:
            raise ValueError("Environment config must contain both 'env' and 'eval_env'")
        required_fields = ["num_agent_per_scene", "device"]
        for field in required_fields:
            if field not in config.get("env", {}):
                raise ValueError(f"Missing required env field: {field}")

    def generate_env_template(self, env_name: str) -> Dict[str, Any]:
        # env_name is unused but kept for API compatibility
        return {
            "env": {
                "num_agent_per_scene": 160,
                "max_episode_steps": 256,
                "visual": False,
                "requires_grad": True,
                "tensor_output": True,
                "device": "cpu",
            },
            "eval_env": {"num_agent_per_scene": 1, "visual": False},
        }
