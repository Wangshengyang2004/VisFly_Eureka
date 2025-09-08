"""
System constants for VisFly-Eureka

IMPORTANT: This module contains only constants that are NOT defined in configuration files.

PRINCIPLE: Config is always correct - no defensive defaults needed.
- Timeouts, episode steps, batch sizes, etc. → Use config values directly
- System limits, conversion factors, thresholds → Keep as constants here

All configurable values should be in configs/ and accessed through config objects:
- configs/config.yaml - Global pipeline settings
- configs/envs/*.yaml - Environment parameters (episode steps, batch size, etc.)
- configs/algs/*/*.yaml - Algorithm parameters (timeouts, learning rates, etc.)
- configs/llm/*.yaml - LLM parameters (temperature, max_tokens, timeout, etc.)
"""

# =============================================================================
# SYSTEM LIMITS AND THRESHOLDS (Not configurable)
# =============================================================================

# Memory and resource management
GPU_MEMORY_CONVERSION_FACTOR = 1024 * 1024  # Bytes to MB conversion
MEMORY_CACHE_TTL_SECONDS = 2.0              # Cache GPU memory info for 2 seconds
GPU_MEMORY_SAFETY_MARGIN = 1000             # 1GB safety reserve for GPU allocation

# Memory growth estimation during training
MEMORY_GROWTH_ESTIMATION_PERIOD = 300       # First 5 minutes - expect growth
MEMORY_GROWTH_CALCULATION_DIVISOR = 600     # For growth factor calculation  
MEMORY_GROWTH_FACTOR_MAX = 1.5              # Maximum allowed memory growth (50%)
MEMORY_ESTIMATION_BUFFER_FACTOR = 1.1       # 10% buffer for final memory estimation
OOM_RETRY_MEMORY_INCREASE_FACTOR = 1.2      # Request 20% more memory after OOM

# Training monitoring thresholds
MEMORY_ESTIMATION_THRESHOLD_STEPS = 1000    # When to estimate final memory usage

# =============================================================================
# VALIDATION CONSTANTS (Not configurable)
# =============================================================================

# Input validation limits
MAX_CONFIG_DEPTH = 10              # Maximum nesting depth for configs
MAX_ENV_NAME_LENGTH = 50           # Maximum environment name length
MAX_ALGORITHM_NAME_LENGTH = 20     # Maximum algorithm name length

# Reward function validation
MAX_REASONABLE_REWARD_TENSOR_SIZE = 1000  # Elements in reward tensor
REWARD_VALIDATION_ENABLED = True

# =============================================================================
# ERROR HANDLING CONSTANTS (Not configurable)
# =============================================================================

# Retry behavior
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_FACTOR = 2           # Exponential backoff multiplier
BASE_RETRY_DELAY = 1.0             # Base delay in seconds

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def bytes_to_mb(bytes_value: int) -> int:
    """Convert bytes to MB"""
    return bytes_value // GPU_MEMORY_CONVERSION_FACTOR


def mb_to_bytes(mb_value: int) -> int:
    """Convert MB to bytes"""  
    return mb_value * GPU_MEMORY_CONVERSION_FACTOR


def get_gpu_memory_with_safety_margin(available_mb: int) -> int:
    """Calculate usable GPU memory with safety margin"""
    return max(0, available_mb - GPU_MEMORY_SAFETY_MARGIN)