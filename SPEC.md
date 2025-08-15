# VisFly-Eureka Native Integration Specification

## Project Overview

This project builds **Eureka** natively for **VisFly**, creating a seamless LLM-powered reward optimization system for vision-based UAV tasks. Rather than adapting existing Eureka for VisFly, we build from scratch to leverage VisFly's unique capabilities including differentiable simulation, high-performance visual rendering, and native tensor operations.

### Core Objectives

1. **Native VisFly Integration**: Direct reward injection into VisFly environments without adapter layers
2. **LLM-Powered Optimization**: Automated reward function generation and iterative improvement
3. **Differentiable-First Design**: Prioritize BPTT training with gradient-aware reward functions
4. **Visual Intelligence**: Leverage depth/RGB sensors for sophisticated reward design
5. **Multi-Agent Coordination**: Support coordinated multi-drone reward optimization

## Native Architecture

### 1. Direct Integration Design

```
┌─────────────────────────────────────────────────────────┐
│                 Eureka-VisFly Core                      │
│              (Single Orchestrator)                     │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
│     LLM      │ │ Reward   │ │ Training   │
│   Engine     │ │Injection │ │  Monitor   │
└──────────────┘ └────┬─────┘ └────────────┘
                      │
            ┌─────────▼─────────┐
            │ VisFly Environment │
            │ (Direct Integration)│
            └───────────────────┘
```

### 2. Core Components

#### **Eureka-VisFly Controller** (`eureka_visfly.py`)
```python
class EurekaVisFly:
    def __init__(self, env_class, task_description, llm_config):
        self.env_class = env_class
        self.task_description = task_description
        self.llm = LLMEngine(llm_config)
        self.reward_injector = RewardInjector()
        self.training_monitor = TrainingMonitor()
    
    def optimize_rewards(self, iterations=5, samples=16):
        # Main optimization loop with direct environment control
        pass
```

#### **Direct Reward Injection** (`reward_injection.py`)
```python
class RewardInjector:
    @staticmethod 
    def inject_reward_function(env_instance, reward_code):
        # Direct function replacement in environment instance
        reward_func = compile_reward_function(reward_code)
        env_instance.get_reward = types.MethodType(reward_func, env_instance)
        return env_instance
```

#### **LLM Engine** (`llm_engine.py`) 
```python
class LLMEngine:
    def generate_reward_functions(self, prompt, samples=16):
        # Clean API calls with structured responses
        responses = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=self.temperature,
            n=samples
        )
        return [self.parse_reward_code(r.message.content) for r in responses.choices]
```

## Technical Requirements

### 1. Native Reward Function Interface

**Direct VisFly Integration** - No wrapper classes needed:
```python
def inject_generated_reward(env_instance, reward_code_string):
    """Direct injection into existing VisFly environment"""
    exec_globals = {'torch': torch, 'th': torch, 'F': torch.nn.functional}
    exec(reward_code_string, exec_globals)
    
    # Replace environment's get_reward method directly
    new_reward_func = exec_globals['get_reward']
    env_instance.get_reward = types.MethodType(new_reward_func, env_instance)

# Usage with any VisFly environment
nav_env = NavigationEnv(...)
inject_generated_reward(nav_env, llm_generated_code)
```

**Native Reward Function Template**:
```python
def get_reward(self) -> torch.Tensor:
    """
    Native VisFly reward function - direct access to all environment state
    LLM generates this entire function body
    """
    # Direct access to VisFly environment properties
    pos_reward = -torch.norm(self.position - self.target, dim=1)
    
    # Visual sensor utilization
    if hasattr(self, 'sensor_obs') and 'depth' in self.sensor_obs:
        depth_data = self.sensor_obs['depth']
        collision_penalty = -torch.sum(depth_data < 0.5, dim=(1,2)) * 0.001
    else:
        collision_penalty = torch.zeros_like(pos_reward)
        
    # Stability reward using differentiable operations
    orientation_penalty = -torch.norm(self.orientation - torch.tensor([1,0,0,0]), dim=1) * 0.1
    
    return pos_reward + collision_penalty + orientation_penalty
```

### 2. Native Training Integration

**Direct VisFly Training** - Use existing VisFly training patterns:
```python
def train_with_generated_reward(env, reward_code, algorithm="bptt", steps=10000):
    """Train directly with VisFly's native training systems"""
    # Inject reward function
    inject_generated_reward(env, reward_code)
    
    if algorithm == "bptt":
        from VisFly.utils.algorithms.BPTT import BPTT
        model = BPTT(env=env, policy="MultiInputPolicy")
    else:
        from stable_baselines3 import PPO
        model = PPO("MultiInputPolicy", env)
        
    model.learn(total_timesteps=steps)
    return model

# No wrapper layers - direct integration with VisFly patterns
```

**Training Evaluation**:
```python
@dataclass 
class TrainingResult:
    success_rate: float
    episode_length: float 
    training_time: float
    final_reward: float
    convergence_step: int
    
    def score(self) -> float:
        """Simple scoring for reward function comparison"""
        return self.success_rate * 0.6 + (1.0 / max(self.episode_length, 1)) * 0.4
```

### 3. Native LLM Prompt Engineering

**System Prompt for VisFly**:
```
You are designing reward functions for VisFly drone environments. Generate complete get_reward methods.

VisFly Environment Context:
- self.position: torch.Tensor [N, 3] - drone positions
- self.velocity: torch.Tensor [N, 3] - linear velocities  
- self.orientation: torch.Tensor [N, 4] - quaternions
- self.angular_velocity: torch.Tensor [N, 3] - angular velocities
- self.target: torch.Tensor [N, 3] - target positions
- self.sensor_obs: dict with 'depth', 'rgb' sensors
- self.collision_dis: torch.Tensor [N] - distance to closest obstacle
- self._step_count: int - current episode step

Requirements:
1. Return complete get_reward(self) method
2. Use torch operations for differentiability  
3. Return torch.Tensor [N] for N agents
4. Utilize visual sensors when available
5. No imports needed - torch/th available

Task: {task_description}
```

**Clean Response Parsing**:
```python
def extract_reward_function(llm_response: str) -> str:
    """Extract reward function from LLM response"""
    # Find function definition
    lines = llm_response.strip().split('\n')
    func_start = None
    
    for i, line in enumerate(lines):
        if line.strip().startswith('def get_reward(self)'):
            func_start = i
            break
            
    if func_start is None:
        raise ValueError("No get_reward function found")
        
    # Extract complete function
    func_lines = []
    indent_level = len(lines[func_start]) - len(lines[func_start].lstrip())
    
    for line in lines[func_start:]:
        if line.strip() == "":
            func_lines.append(line)
        elif len(line) - len(line.lstrip()) <= indent_level and line.strip() and func_lines:
            break
        else:
            func_lines.append(line)
            
    return '\n'.join(func_lines)
```

## Implementation Roadmap

### Phase 1: Core Native Integration (Week 1)
**Deliverables**:
- [ ] `eureka_visfly.py` - Main controller
- [ ] `reward_injection.py` - Direct reward injection 
- [ ] `llm_engine.py` - OpenAI API integration
- [ ] Basic NavigationEnv test

**Success Criteria**:
- Inject LLM-generated reward into NavigationEnv instance
- Complete end-to-end: prompt → LLM → code → injection → training

### Phase 2: Training & Evaluation (Week 2)
**Deliverables**:
- [ ] Native BPTT training integration
- [ ] Training result evaluation and scoring
- [ ] Iterative improvement loop
- [ ] Multi-sample evaluation

**Success Criteria**:
- Train with both BPTT and PPO using injected rewards
- Rank reward functions by training performance
- Demonstrate iterative improvement

### Phase 3: Advanced Features (Week 3)
**Deliverables**:
- [ ] Visual sensor utilization in rewards
- [ ] Multi-environment support (Navigation, Racing, Tracking)
- [ ] Multi-agent coordination rewards
- [ ] Comprehensive unit tests

**Success Criteria**:
- Generate rewards using depth/RGB sensor data
- Support multiple VisFly environment types
- Handle multi-agent scenarios effectively

## Comprehensive Unit Testing Strategy

### 1. Core Component Tests

**LLM Integration Tests** (`tests/test_llm_engine.py`):
```python
class TestLLMEngine(unittest.TestCase):
    def setUp(self):
        self.llm_engine = LLMEngine(model="gpt-4", api_key="test-key")
        
    @patch('openai.ChatCompletion.create')
    def test_generate_reward_functions(self, mock_openai):
        # Test successful API call and response parsing
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="def get_reward(self):\n    return torch.zeros(1)"))]
        mock_openai.return_value = mock_response
        
        result = self.llm_engine.generate_reward_functions("test prompt")
        self.assertEqual(len(result), 1)
        
    @patch('openai.ChatCompletion.create')  
    def test_api_failure_handling(self, mock_openai):
        # Test API failures result in empty list, not crash
        mock_openai.side_effect = openai.APIError("Test error")
        result = self.llm_engine.generate_reward_functions("test prompt")
        self.assertEqual(result, [])
```

**Reward Injection Tests** (`tests/test_reward_injection.py`):
```python
class TestRewardInjection(unittest.TestCase):
    def test_inject_reward_function(self):
        # Test direct reward injection into VisFly environment
        env = NavigationEnv(num_agent_per_scene=2, max_episode_steps=10)
        reward_code = """
def get_reward(self):
    return torch.ones(self.num_agent) * 5.0
"""
        inject_generated_reward(env, reward_code)
        env.reset()
        
        reward = env.get_reward()
        self.assertEqual(reward.shape, (2,))
        self.assertTrue(torch.allclose(reward, torch.tensor([5.0, 5.0])))
        
    def test_invalid_reward_code(self):
        # Test that invalid code doesn't crash the system
        env = NavigationEnv(num_agent_per_scene=1)
        invalid_code = "def invalid_function(): syntax error here"
        
        # Should fail gracefully without crashing
        with self.assertLogs() as log:
            inject_generated_reward(env, invalid_code)
            # Environment should retain original reward function
            env.reset()
            original_reward = env.get_reward()
            self.assertIsInstance(original_reward, torch.Tensor)
```

**Training Integration Tests** (`tests/test_training.py`):
```python
class TestTrainingIntegration(unittest.TestCase):
    def test_bptt_training_with_injected_reward(self):
        # Test BPTT training works with injected reward functions
        env = NavigationEnv(num_agent_per_scene=4, requires_grad=True, max_episode_steps=32)
        reward_code = """
def get_reward(self):
    return -torch.norm(self.position - self.target, dim=1)
"""
        inject_generated_reward(env, reward_code)
        
        model = train_with_generated_reward(env, reward_code, algorithm="bptt", steps=100)
        self.assertIsNotNone(model)
        
    def test_ppo_training_with_injected_reward(self):
        # Test PPO training works with injected rewards
        env = NavigationEnv(num_agent_per_scene=2, max_episode_steps=16)
        reward_code = """
def get_reward(self):
    return torch.randn(self.num_agent)
"""
        model = train_with_generated_reward(env, reward_code, algorithm="ppo", steps=50)
        self.assertIsNotNone(model)
```

### 2. End-to-End System Tests

**Complete Pipeline Test** (`tests/test_pipeline.py`):
```python
class TestEurekaVisFlyPipeline(unittest.TestCase):
    @patch('openai.ChatCompletion.create')
    def test_complete_optimization_pipeline(self, mock_openai):
        # Mock LLM response
        mock_openai.return_value = Mock(choices=[
            Mock(message=Mock(content="""
def get_reward(self):
    return -torch.norm(self.position - self.target, dim=1) * 0.1
"""))
        ])
        
        # Test complete pipeline from LLM to training
        eureka = EurekaVisFly(
            env_class=NavigationEnv,
            task_description="Navigate to target",
            llm_config={"model": "gpt-4", "api_key": "test"}
        )
        
        results = eureka.optimize_rewards(iterations=1, samples=1)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], TrainingResult)
```

### 3. Critical Error Isolation

**Error Boundaries** - Critical components should fail without propagating:
```python
def safe_llm_call(llm_engine, prompt):
    """LLM calls should never crash the system"""
    result = llm_engine.generate_reward_functions(prompt)
    return result if result else ["# Fallback reward function\ndef get_reward(self):\n    return torch.zeros(self.num_agent)"]

def safe_reward_injection(env, reward_code):
    """Reward injection should never crash the environment"""  
    original_get_reward = env.get_reward
    inject_generated_reward(env, reward_code)
    
    # Test injected reward function
    env.reset()
    test_reward = env.get_reward()
    if not isinstance(test_reward, torch.Tensor) or test_reward.shape != (env.num_agent,):
        # Restore original reward function
        env.get_reward = original_get_reward
        
def safe_training(env, reward_code, algorithm="bptt", steps=1000):
    """Training should complete or fail gracefully"""
    model = train_with_generated_reward(env, reward_code, algorithm, steps)
    return model if model else None
```

## Native VisFly Code Organization

### 1. Minimal File Structure
```
eureka_visfly/
├── eureka_visfly.py          # Main controller class
├── llm_engine.py             # OpenAI integration  
├── reward_injection.py       # Direct reward injection
├── training_utils.py         # Training helpers
└── prompts.py               # LLM prompt templates

tests/
├── test_llm_engine.py        # LLM API tests
├── test_reward_injection.py  # Injection tests  
├── test_training.py          # Training integration tests
└── test_pipeline.py          # End-to-end tests

examples/
├── navigation_example.py     # NavigationEnv optimization
├── racing_example.py         # RacingEnv optimization  
└── multi_agent_example.py    # Multi-agent coordination
```

### 2. Native Integration Benefits

**No Adapter Layers**:
- Direct method injection into VisFly environments
- Use existing VisFly training patterns (BPTT, PPO)
- Leverage VisFly's native tensor operations and device management

**Simplified Architecture**:
- Single controller class orchestrates everything
- Direct reward function replacement via `types.MethodType`
- Native support for VisFly's differentiable simulation

**Clean Error Handling**:
- Critical failures (LLM API, reward injection) are contained
- System continues with fallback reward functions
- No complex error propagation chains

## Success Criteria & Validation

### 1. Core Functionality  
- [ ] Direct reward injection into any VisFly environment class
- [ ] LLM-generated rewards train successfully with BPTT and PPO
- [ ] Multi-iteration improvement shows reward function evolution
- [ ] Visual sensor integration (depth/RGB) in generated rewards

### 2. Performance Targets
- [ ] Training converges within 2x manual reward design time
- [ ] Generated rewards achieve ≥90% of manual reward performance
- [ ] System handles 5+ simultaneous optimization runs
- [ ] Unit tests achieve >95% coverage

### 3. Robustness Requirements
- [ ] LLM API failures don't crash the system
- [ ] Invalid reward code gracefully falls back to defaults
- [ ] Training failures are logged and don't propagate
- [ ] Multi-environment support (Navigation, Racing, Tracking, Landing)

## Example Usage

```python
# Simple usage - optimize NavigationEnv rewards
from eureka_visfly import EurekaVisFly
from VisFly.envs.NavigationEnv import NavigationEnv

eureka = EurekaVisFly(
    env_class=NavigationEnv,
    task_description="Navigate drone to target while avoiding obstacles using depth sensor",
    llm_config={"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}
)

# Run optimization - generates, tests, and ranks reward functions
best_rewards = eureka.optimize_rewards(iterations=5, samples=16)
print(f"Best reward achieved {best_rewards[0].success_rate:.2f} success rate")

# Use best reward function in production
best_env = NavigationEnv(...)
inject_generated_reward(best_env, best_rewards[0].code)
```

## Conclusion

This specification defines a **native VisFly-Eureka integration** that builds Eureka functionality directly into VisFly without adapter layers. The approach prioritizes:

1. **Simplicity**: Direct reward injection, minimal file structure, clean APIs
2. **Performance**: Native tensor operations, differentiable-first design
3. **Robustness**: Comprehensive testing, error boundaries, graceful failures
4. **Usability**: Clear examples, straightforward usage patterns

By building Eureka natively for VisFly, we create a seamless system that leverages VisFly's unique capabilities while maintaining the simplicity and effectiveness of LLM-powered reward optimization.