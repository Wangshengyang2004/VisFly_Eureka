"""
Integration tests for LLM batching strategies.
"""

import pytest
import time
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from quadro_llm.llm.llm_engine import LLMEngine


@pytest.fixture
def mock_llm_config():
    """Mock LLM configuration for testing."""
    return {
        "model": "test-model",
        "api_key": "test-key",
        "base_url": "https://test.api.com",
        "temperature": 0.8,
        "max_tokens": 4096,
        "timeout": 120,
        "max_retries": 3,
        "thinking_enabled": False,
        "batching_strategy": "sequential",
        "supports_n_parameter": False,
        "max_concurrent": 5,
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = """
def get_reward(self):
    import torch
    return torch.zeros(self.num_envs)
    """
    return mock_response


class TestLLMBatching:
    """Test LLM batching strategies."""

    def test_sequential_batching(self, mock_llm_config, mock_openai_response):
        """Test sequential batching strategy."""
        mock_llm_config["batching_strategy"] = "sequential"
        
        with patch('quadro_llm.llm.llm_engine.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai.return_value = mock_client
            
            llm = LLMEngine(**mock_llm_config)
            
            # Test sequential generation
            with patch('quadro_llm.llm.llm_engine.extract_reward_function') as mock_extract:
                mock_extract.return_value = "def get_reward(self): return torch.zeros(1)"
                
                results = llm._generate_sequential([{"role": "user", "content": "test"}], 3)
                
                assert len(results) == 3
                assert mock_client.chat.completions.create.call_count == 3
                assert all("get_reward" in result for result in results)

    def test_n_parameter_batching(self, mock_llm_config, mock_openai_response):
        """Test n-parameter batching strategy."""
        mock_llm_config["batching_strategy"] = "n_parameter"
        mock_llm_config["supports_n_parameter"] = True
        
        # Mock multiple choices in response
        mock_openai_response.choices = [Mock() for _ in range(3)]
        for choice in mock_openai_response.choices:
            choice.message.content = "def get_reward(self): return torch.zeros(1)"
        
        with patch('quadro_llm.llm.llm_engine.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai.return_value = mock_client
            
            llm = LLMEngine(**mock_llm_config)
            
            # Test n-parameter generation
            with patch('quadro_llm.llm.llm_engine.extract_reward_function') as mock_extract:
                mock_extract.return_value = "def get_reward(self): return torch.zeros(1)"
                
                results = llm._generate_with_n_parameter([{"role": "user", "content": "test"}], 3)
                
                assert len(results) == 3
                # Should use single API call with n=3
                mock_client.chat.completions.create.assert_called_once()
                call_args = mock_client.chat.completions.create.call_args[1]
                assert call_args['n'] == 3

    def test_thinking_configuration(self, mock_llm_config, mock_openai_response):
        """Test thinking configuration is properly applied."""
        mock_llm_config["thinking_enabled"] = False
        
        with patch('quadro_llm.llm.llm_engine.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai.return_value = mock_client
            
            llm = LLMEngine(**mock_llm_config)
            
            # Test API connection (should add thinking config)
            llm.test_api_connection()
            
            call_args = mock_client.chat.completions.create.call_args[1]
            assert "extra_body" in call_args
            assert call_args["extra_body"]["thinking"]["type"] == "disabled"

    def test_batch_size_handling(self, mock_llm_config, mock_openai_response):
        """Test batch size handling for large requests."""
        mock_llm_config["batching_strategy"] = "n_parameter" 
        mock_llm_config["supports_n_parameter"] = True
        
        with patch('quadro_llm.llm.llm_engine.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai.return_value = mock_client
            
            llm = LLMEngine(**mock_llm_config)
            
            with patch('quadro_llm.llm.llm_engine.extract_reward_function') as mock_extract:
                mock_extract.return_value = "def get_reward(self): return torch.zeros(1)"
                with patch.object(llm, '_generate_single_batch_with_n', return_value=["test"]) as mock_batch:
                    
                    # Test large batch (should split into multiple calls)
                    results = llm._generate_with_n_parameter([{"role": "user", "content": "test"}], 25)
                    
                    # Should make 3 calls: 10 + 10 + 5
                    assert mock_batch.call_count == 3
                    call_args = [call.args[1] for call in mock_batch.call_args_list]
                    assert call_args == [10, 10, 5]


@pytest.mark.integration
class TestLLMIntegration:
    """Integration tests requiring real API configuration."""

    def test_config_loading_from_file(self):
        """Test loading configuration from actual config files."""
        # This test can be run if config files exist
        config_path = Path("configs/llm/glm-4.5.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify expected structure
        assert "llm" in config
        assert "batching" in config["llm"]
        assert "thinking" in config["llm"]
        
        # Verify batching config
        batching = config["llm"]["batching"]
        assert "strategy" in batching
        assert "supports_n_parameter" in batching
        assert "max_concurrent" in batching
        
        # Verify thinking config
        thinking = config["llm"]["thinking"]
        assert "enabled" in thinking