# Tests

This directory contains all tests for the VisFly-Eureka project.

## Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   └── test_llm_config.py   # LLM configuration and engine tests
├── integration/             # Integration tests for system interactions  
│   └── test_llm_batching.py # LLM batching strategy tests
├── test_llm_performance.py  # Performance benchmarking tool
├── run_llm_tests.py         # Test runner script
└── README.md               # This file
```

## Running Tests

### Unit Tests
```bash
python -m pytest tests/unit/ -v
```

### Integration Tests  
```bash
python -m pytest tests/integration/ -v
```

### All Pytest Tests
```bash
python -m pytest tests/ -v
```

### Performance Benchmarks
```bash
# Test default model (glm-4.5)
cd tests/
python test_llm_performance.py

# Test specific model
python test_llm_performance.py --model gpt-4o

# Test with more samples
python test_llm_performance.py --model glm-4.5 --samples 5

# List available models
python test_llm_performance.py --list-models
```

### Test Runner (All Tests)
```bash
cd tests/
python run_llm_tests.py --all

# Or specific categories
python run_llm_tests.py --unit
python run_llm_tests.py --integration  
python run_llm_tests.py --performance --model glm-4.5
```

## Test Categories

### Unit Tests
- Test individual components in isolation
- Use mocks for external dependencies
- Fast execution
- Focus on logic correctness

### Integration Tests  
- Test component interactions
- May use real configurations
- Test API integrations with mocks
- Focus on system behavior

### Performance Tests
- Benchmark real API performance
- Measure response times
- Test different batching strategies
- Compare model configurations

## Adding New Tests

### Unit Tests
Add to `tests/unit/` with the pattern `test_*.py`:
```python
import pytest
from unittest.mock import Mock, patch

def test_your_function():
    # Test implementation
    assert True
```

### Integration Tests
Add to `tests/integration/` with the pattern `test_*.py`:
```python
import pytest

@pytest.mark.integration
def test_system_interaction():
    # Test implementation
    assert True
```

### Performance Tests
Extend `test_llm_performance.py` or create similar standalone scripts for different benchmarks.

## Test Markers

Use pytest markers to categorize tests:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.gpu` - Tests requiring GPU
- `@pytest.mark.llm` - Tests requiring LLM API access

## Configuration

Tests use the same configuration system as the main application:
- `configs/llm/*.yaml` - LLM model configurations
- `configs/api_keys.yaml` - API credentials

Performance tests automatically load configurations for the specified model.