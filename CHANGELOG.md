# CHANGELOG

## Recent Changes

### Elite Voter Implementation
- **Added**: `quadro_llm/core/elite_voter.py` - LLM agent for selecting best reward function from candidates
- **Added**: Elite voter analyzes evaluation results and trajectory data to select elite policy
- **Added**: Trajectory data formatting with full timesteps (no sampling/truncating) and .3f precision
- **Added**: Simplified prompts focusing on why selected candidate is elite (not describing all candidates)
- **Changed**: Elite voter reasoning focuses on selected candidate's strengths, not comparison of all candidates

### Trajectory Data Management
- **Added**: Save trajectory data (positions, velocities, orientations, angular_velocities, target) as .npz files during evaluation
- **Added**: Trajectory data saved in `trajectories/` subdirectory within output directory
- **Added**: Trajectory path stored in `episode_statistics` for each episode
- **Changed**: Full trajectory data (all timesteps) sent to elite voter, not sampled/truncated
- **Changed**: Euler angles (Roll, Pitch, Yaw) included for flip tasks with orientation analysis

### Plotting Refactoring
- **Changed**: `plot_trajectory` refactored to 3x3 subplot layout using `VisFly.utils.FigFashion.FigFon`
- **Changed**: 9 distinct subplots: 3D trajectory, Position (X/Y/Z), Velocity (Vx/Vy/Vz), Angular Velocity (ωx/ωy/ωz), Orientation (Euler angles), Distance to Target, Speed Magnitude, Angular Velocity Magnitude, Position Error
- **Removed**: Defensive code and try-except blocks for FigFashion import
- **Changed**: Direct import and usage of `FigFon` from VisFly

### Elite Voter Integration
- **Added**: `EliteVoter` initialized in `EurekaVisFly.__init__`
- **Changed**: Replaced heuristic selection (`max(success_rate)`) with `elite_voter.vote()` call
- **Changed**: Elite voter selection stored in `self.elite_vote_results` list
- **Changed**: Elite reward function code stored in `self.best_reward_functions` list
- **Changed**: Previous elite reward code passed to LLM via `previous_elite_reward` parameter

### Feedback Generation
- **Renamed**: `_generate_feedback_with_tensorboard` → `_generate_feedback`
- **Removed**: TensorBoard data from main feedback (handled by elite voter instead)
- **Added**: Elite voter reasoning included in feedback for next iteration
- **Changed**: Feedback uses elite voter's selected result, not heuristic selection
- **Changed**: Feedback format simplified to LaRes-style (evaluation metrics, per-episode stats)
- **Removed**: TensorBoard log loading and dataframe appending from feedback

### Conversation History Management
- **Changed**: History updated with elite reward function only (not first candidate `results[0]`)
- **Changed**: History update happens after elite voter selection, not during `generate_reward_functions`
- **Changed**: History contains only elite reward functions, not all candidates
- **Added**: Static info (env_code, api_doc, task_description, context_info) check in history
- **Changed**: Static info only included if not present in current history window (avoids repetition)
- **Added**: Protection against losing static info when history is pruned (re-includes if missing)

### Prompt Engineering
- **Added**: `include_static_info` parameter to `create_user_prompt()` function
- **Changed**: Static info (env_code, api_doc, task_description, context_info) conditionally included
- **Changed**: Static info included if: no history exists, history_window_size=0, or static info not in current history
- **Changed**: Elite voter prompt focuses on explaining why selected candidate is elite
- **Removed**: Complex diagnostic metrics (trembling, smoothness, flip phase detection) from trajectory formatting

### LLM Configuration
- **Changed**: Elite voter uses same LLM config as main LLM (including `thinking_enabled`)
- **Changed**: Elite voter uses `LLMEngine._build_request_params()` to ensure consistent config
- **Removed**: `max_tokens` limit for elite voter (allows full response)
- **Changed**: JSON parsing enhanced to extract from markdown code blocks (````json`)

### Evaluation Changes
- **Removed**: Hardcoded `eval_runs = max(10, requested_eval_episodes)` limit
- **Changed**: `eval_runs` directly uses `requested_eval_episodes` from config
- **Changed**: Evaluation episodes no longer have minimum limit

### Configuration
- **Removed**: `api_doc_path` from config.yaml (now hardcoded in code)
- **Changed**: API doc path hardcoded to `quadro_llm/llm/prompts/api-doc.txt`

### Code Quality
- **Removed**: Excessive defensive code and try-except blocks where not needed
- **Changed**: Direct usage of imports without defensive checks
- **Added**: Clear, concise code without excessive defensive logic

## Previous Changes (Context)

### Environment and Task Selection
- **Fixed**: Task/environment selection when `task=flip_task` is passed via CLI
- **Changed**: `create_eureka_controller` prioritizes task category when task is explicitly overridden
- **Changed**: CLI logging correctly reflects task and environment based on overrides

### FlipEnv Fixes
- **Fixed**: `self.target` shape issues (now always flattened to [3])
- **Fixed**: `_reset_attr()` signature to match base class (`reset_latent=True` parameter)
- **Changed**: Environment context extraction updated for FlipEnv attributes

### Reward Function Injection
- **Added**: `math` module to `exec_globals` in reward injector and evaluation worker
- **Fixed**: `NameError: name 'math' is not defined` in generated reward functions

### Human Reward Inclusion
- **Added**: `include_human_reward` feature to include human-designed reward in LLM prompt
- **Added**: `extract_human_reward()` function to extract reward code from environment class
- **Changed**: Human reward code conditionally included in prompt based on config

## Potential Issues to Check

1. **History Management**: Verify that static info is correctly detected in history and not lost during pruning
2. **Elite Voter**: Check if trajectory data loading from .npz files works correctly for all episode types
3. **Feedback Generation**: Ensure elite voter's selected result is correctly used when constructing feedback
4. **Static Info Detection**: Verify the string matching logic ("The Python environment is:" or "Task:") correctly identifies static info
5. **History Update Timing**: Check that history is updated with correct user_prompt that matches the iteration's actual prompt
6. **Elite Reward Code**: Verify that `previous_elite_reward` passed to LLM matches the actual elite selection
7. **Trajectory Data**: Check if trajectory .npz files are correctly saved and loaded, especially for multiple agents
8. **Plotting**: Verify 3x3 subplot layout works correctly with FigFashion for all trajectory types
9. **Euler Angle Conversion**: Check quaternion to Euler conversion for orientation plotting
10. **JSON Parsing**: Verify elite voter's JSON response parsing handles edge cases (markdown blocks, malformed JSON)



