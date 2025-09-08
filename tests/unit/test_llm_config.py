"""
Unit tests for LLM configuration loading and validation.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
import yaml

from quadro_llm.llm.llm_engine import LLMEngine


class TestLLMConfiguration:
    """Test LLM configuration handling."""

    def test_llm_engine_initialization_minimal(self):
        """Test LLM engine initialization with minimal config."""
        config = {
            "model": "test-model",
            "api_key": "test-key",
            "base_url": "https://test.api.com",
            "temperature": 0.8,
            "max_tokens": 4096,
            "timeout": 120,
            "max_retries": 3,
        }
        
        with patch('quadro_llm.llm.llm_engine.OpenAI'):
            llm = LLMEngine(**config)
            
            assert llm.model == "test-model"
            assert llm.temperature == 0.8
            assert llm.batching_strategy == "n_parameter"  # default
            assert llm.thinking_enabled == True  # default

    def test_llm_engine_initialization_full(self):
        """Test LLM engine initialization with full config."""
        config = {
            "model": "glm-4.5",
            "api_key": "test-key",
            "base_url": "https://test.api.com",
            "temperature": 1.0,
            "max_tokens": 8192,
            "timeout": 520,
            "max_retries": 3,
            "thinking_enabled": False,
            "batching_strategy": "async",
            "supports_n_parameter": False,
            "max_concurrent": 10,
        }
        
        with patch('quadro_llm.llm.llm_engine.OpenAI'):
            llm = LLMEngine(**config)
            
            assert llm.model == "glm-4.5"
            assert llm.thinking_enabled == False
            assert llm.batching_strategy == "async"
            assert llm.supports_n_parameter == False
            assert llm.max_concurrent == 10

    def test_batching_strategy_selection(self):
        """Test batching strategy selection logic."""
        base_config = {
            "model": "test-model",
            "api_key": "test-key", 
            "base_url": "https://test.api.com",
            "temperature": 0.8,
            "max_tokens": 4096,
            "timeout": 120,
            "max_retries": 3,
        }
        
        # Test n_parameter strategy
        config = {**base_config, "batching_strategy": "n_parameter", "supports_n_parameter": True}
        with patch('quadro_llm.llm.llm_engine.OpenAI'):
            llm = LLMEngine(**config)
            with patch.object(llm, '_generate_with_n_parameter', return_value=[]) as mock_n:
                llm.generate_reward_functions("test", {}, "", 3, None)
                mock_n.assert_called_once()

        # Test sequential strategy  
        config = {**base_config, "batching_strategy": "sequential", "supports_n_parameter": False}
        with patch('quadro_llm.llm.llm_engine.OpenAI'):
            llm = LLMEngine(**config)
            with patch.object(llm, '_generate_sequential', return_value=[]) as mock_seq:
                llm.generate_reward_functions("test", {}, "", 3, None)
                mock_seq.assert_called_once()

    def test_thinking_parameter_application(self):
        """Test thinking parameter is applied correctly."""
        config = {
            "model": "glm-4.5",
            "api_key": "test-key",
            "base_url": "https://test.api.com", 
            "temperature": 0.8,
            "max_tokens": 4096,
            "timeout": 120,
            "max_retries": 3,
            "thinking_enabled": False,
        }
        
        with patch('quadro_llm.llm.llm_engine.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = []
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            llm = LLMEngine(**config)
            llm.test_api_connection()
            
            # Check that thinking config was applied
            call_args = mock_client.chat.completions.create.call_args[1]
            assert "extra_body" in call_args
            assert call_args["extra_body"]["thinking"]["type"] == "disabled"

    def test_fallback_strategy(self):
        """Test fallback to sequential for unknown strategies.""" 
        config = {
            "model": "test-model",
            "api_key": "test-key",
            "base_url": "https://test.api.com",
            "temperature": 0.8,
            "max_tokens": 4096,
            "timeout": 120,
            "max_retries": 3,
            "batching_strategy": "unknown_strategy",
        }
        
        with patch('quadro_llm.llm.llm_engine.OpenAI'):
            llm = LLMEngine(**config)
            with patch.object(llm, '_generate_sequential', return_value=[]) as mock_seq:
                with patch.object(llm.logger, 'warning') as mock_warn:
                    llm.generate_reward_functions("test", {}, "", 3, None)
                    
                    mock_seq.assert_called_once()
                    mock_warn.assert_called_once()
                    assert "Unknown batching strategy" in str(mock_warn.call_args)


class TestConfigurationLoading:
    """Test configuration file loading (integration-like tests)."""

    def test_yaml_config_structure(self):
        """Test expected YAML configuration structure."""
        sample_config = {
            "llm": {
                "vendor": "bigmodel",
                "model": "glm-4.5",
                "temperature": 1.0,
                "max_tokens": 8192,
                "timeout": 520,
                "max_retries": 3,
                "thinking": {
                    "enabled": False
                },
                "batching": {
                    "supports_n_parameter": False,
                    "strategy": "async", 
                    "max_concurrent": 10
                }
            }
        }
        
        # Test structure validation
        assert "llm" in sample_config
        llm_config = sample_config["llm"]
        
        # Required fields
        required_fields = ["vendor", "model", "temperature", "max_tokens", "timeout", "max_retries"]
        for field in required_fields:
            assert field in llm_config
        
        # Optional structured fields
        if "thinking" in llm_config:
            assert "enabled" in llm_config["thinking"]
            assert isinstance(llm_config["thinking"]["enabled"], bool)
        
        if "batching" in llm_config:
            batching = llm_config["batching"]
            if "supports_n_parameter" in batching:
                assert isinstance(batching["supports_n_parameter"], bool)
            if "strategy" in batching:
                assert batching["strategy"] in ["n_parameter", "sequential", "async", "multiprocessing"]
            if "max_concurrent" in batching:
                assert isinstance(batching["max_concurrent"], int)
                assert batching["max_concurrent"] > 0