# VisFly-Eureka (quadro-llm) - Code Quality Improvement TODO

## 🚨 CRITICAL ISSUES (Must Fix - Blocking)

### 1. Fix Bare Exception Handling
- [ ] `/quadro_llm/eureka_visfly.py:312-313` - Replace bare except with specific exceptions
- [ ] `/quadro_llm/training/parallel_training.py:535,545,555` - Add specific exception types
- [ ] `/quadro_llm/core/subprocess_evaluator.py:470-472` - Handle OSError explicitly

### 2. Remove Hardcoded Paths
- [ ] `/quadro_llm/core/subprocess_evaluator.py:76-77` - Replace hardcoded `/home/simonwsy/` paths with dynamic resolution
- [ ] Pass project root as configuration parameter
- [ ] Use `Path(__file__).parent` for relative path resolution

### 3. Extract Large Embedded Script
- [ ] Create `/quadro_llm/core/evaluation_worker.py` as standalone script
- [ ] Move 200+ lines of embedded code from `subprocess_evaluator.py`
- [ ] Update subprocess calls to use external script

## ⚠️ WARNING ISSUES (Should Fix)

### 4. Replace Print Statements with Logging
- [ ] `/quadro_llm/training/visfly_training_wrapper.py` - Replace 35+ print statements
- [ ] Remove emoji decorations unless specifically requested
- [ ] Use appropriate log levels (INFO, DEBUG, WARNING)

### 5. Fix Redundant Imports
- [ ] `/quadro_llm/training/parallel_training.py:531,541` - Move imports to module level
- [ ] Remove duplicate `import re` statements in loops
- [ ] Consolidate all imports at file top

### 6. Improve Error Propagation
- [ ] Define clear error hierarchy (CriticalError, RecoverableError)
- [ ] Ensure consistent error handling across modules
- [ ] Add proper error context and messages

## 💡 SUGGESTIONS (Nice to Have)

### 7. Add Configuration Validation
- [ ] Restore minimal `validate_critical_config()` in `main.py`
- [ ] Check for API keys existence
- [ ] Validate algorithm selection
- [ ] Verify CUDA availability when requested

### 8. Implement Resource Management
- [ ] Add context managers to `SubprocessRewardEvaluator`
- [ ] Ensure proper cleanup of temp directories
- [ ] Handle subprocess termination gracefully

### 9. Add Type Hints
- [ ] Add type hints to all public methods in `eureka_visfly.py`
- [ ] Type hint core functions in `training_utils.py`
- [ ] Include return type annotations

### 10. Performance Optimizations
- [ ] Cache evaluation worker script path
- [ ] Implement connection pooling for parallel evaluations
- [ ] Add memory usage monitoring

### 11. Documentation
- [ ] Add comprehensive docstrings to public APIs
- [ ] Document expected exceptions and error codes
- [ ] Create architecture diagram

## 📋 Implementation Priority

### Phase 1: Critical (Immediate)
1. Fix hardcoded paths
2. Fix bare exceptions
3. Extract embedded script

### Phase 2: High Priority (This Week)
4. Replace prints with logging
5. Move imports to module level
6. Add critical config validation

### Phase 3: Medium Priority (Next Sprint)
7. Add context managers
8. Add type hints
9. Improve error propagation

### Phase 4: Low Priority (As Available)
10. Performance optimizations
11. Complete documentation
12. Add comprehensive tests

## ✅ Completed Items
- [x] Removed unnecessary wrapper functions in `main.py`
- [x] Simplified configuration to trust YAML files directly
- [x] Fixed max_workers configuration issue
- [x] Added eval_env_config support
- [x] Updated prompts for numpy/torch conversion
- [x] Reduced subprocess logging verbosity

## 🐛 Known Issues
- Reward function generation has tensor compatibility issues with numpy arrays from habitat
- Video rendering and trajectory plotting need completion
- Some generated reward functions fail due to incorrect tensor operations

## 📝 Notes
- Always test changes with 2x5 configuration before committing
- Ensure no API keys are exposed in commits
- Keep backward compatibility when possible
- Document breaking changes clearly