"""
Unit tests for LLM interface.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from quadro_llm.core.llm_interface import LLMInterface


class TestLLMInterface:
    """Test LLM interface functionality."""
    
    @pytest.fixture
    def llm_config(self):
        """Create test LLM configuration."""
        return {
            'api_key': 'test-key',
            'model': 'gpt-4o',
            'base_url': 'https://api.test.com',
            'temperature': 0.8,
            'max_tokens': 2000,
            'timeout': 30,
            'max_retries': 2
        }
    
    @pytest.fixture
    def mock_llm(self, llm_config):
        """Create LLM interface with mocked OpenAI client."""
        with patch('quadro_llm.core.llm_interface.openai.OpenAI'):
            llm = LLMInterface(llm_config)
            llm.client = Mock()
            return llm
    
    def test_initialization(self, llm_config):
        """Test LLM interface initialization."""
        with patch('quadro_llm.core.llm_interface.openai.OpenAI') as mock_openai:
            llm = LLMInterface(llm_config)
            
            # Check OpenAI client was created with correct params
            mock_openai.assert_called_once_with(
                api_key='test-key',
                base_url='https://api.test.com',
                timeout=30,
                max_retries=2
            )
            
            assert llm.model == 'gpt-4o'
            assert llm.temperature == 0.8
            assert llm.max_tokens == 2000
    
    def test_extract_reward_code_valid(self, mock_llm):
        """Test extraction of valid reward code."""
        # Test with code block
        response = """
        Here's the reward function:
        ```python
        def get_reward(self):
            distance = torch.norm(self.position - self.target, dim=1)
            return -distance
        ```
        """
        
        code = mock_llm._extract_reward_code(response)
        assert code is not None
        assert 'def get_reward(self)' in code
        assert 'return' in code
    
    def test_extract_reward_code_no_block(self, mock_llm):
        """Test extraction without code block markers."""
        response = """
def get_reward(self):
    distance = torch.norm(self.position - self.target, dim=1)
    return -distance
        """
        
        code = mock_llm._extract_reward_code(response)
        assert code is not None
        assert 'def get_reward(self)' in code
    
    def test_extract_reward_code_invalid(self, mock_llm):
        """Test extraction fails for invalid code."""
        # No function definition
        response = "This is just text without any code"
        code = mock_llm._extract_reward_code(response)
        assert code is None
        
        # Function without return
        response = """
        def get_reward(self):
            distance = torch.norm(self.position - self.target, dim=1)
        """
        code = mock_llm._extract_reward_code(response)
        assert code is None
    
    def test_generate_reward_functions(self, mock_llm):
        """Test reward function generation."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_choice1 = Mock()
        mock_choice1.message.content = """
        def get_reward(self):
            return torch.zeros(self.num_agent)
        """
        mock_choice2 = Mock()
        mock_choice2.message.content = """
        def get_reward(self):
            return -torch.norm(self.velocity - 0, dim=1)
        """
        mock_response.choices = [mock_choice1, mock_choice2]
        
        mock_llm.client.chat.completions.create.return_value = mock_response
        
        # Generate functions
        functions = mock_llm.generate_reward_functions(
            task_description="Navigate to target",
            num_samples=2,
            env_code="# Environment code",
            iteration=1
        )
        
        assert len(functions) == 2
        assert all('def get_reward(self)' in f for f in functions)
        
        # Check API was called correctly
        mock_llm.client.chat.completions.create.assert_called_once()
        call_args = mock_llm.client.chat.completions.create.call_args
        assert call_args.kwargs['model'] == 'gpt-4o'
        assert call_args.kwargs['n'] == 2
    
    def test_generate_reward_functions_batch(self, mock_llm):
        """Test batch generation for large sample counts."""
        # Mock responses for multiple batches
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = """
        def get_reward(self):
            return torch.zeros(self.num_agent)
        """
        mock_response.choices = [mock_choice] * 10
        
        mock_llm.client.chat.completions.create.return_value = mock_response
        
        # Request more than batch size
        functions = mock_llm.generate_reward_functions(
            task_description="Test task",
            num_samples=15,  # > 10 batch size
            iteration=0
        )
        
        # Should make 2 API calls (10 + 5)
        assert mock_llm.client.chat.completions.create.call_count == 2
    
    def test_generate_reward_functions_error_handling(self, mock_llm):
        """Test error handling in generation."""
        # Mock API error
        mock_llm.client.chat.completions.create.side_effect = Exception("API Error")
        
        functions = mock_llm.generate_reward_functions(
            task_description="Test task",
            num_samples=2,
            iteration=0
        )
        
        # Should return empty list on error
        assert functions == []
    
    def test_improve_reward_function(self, mock_llm):
        """Test reward function improvement."""
        original_code = """
        def get_reward(self):
            return torch.zeros(self.num_agent)
        """
        
        # Mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = """
        def get_reward(self):
            distance = -torch.norm(self.position - self.target, dim=1)
            velocity_penalty = -torch.norm(self.velocity - 0, dim=1) * 0.01
            return distance + velocity_penalty
        """
        mock_response.choices = [mock_choice]
        
        mock_llm.client.chat.completions.create.return_value = mock_response
        
        improved = mock_llm.improve_reward_function(
            original_code=original_code,
            performance_issues="No progress toward target",
            task_description="Navigate to target"
        )
        
        assert improved is not None
        assert 'distance' in improved
        assert 'velocity_penalty' in improved