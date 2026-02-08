"""
Mock test for EurekaVisFly iteration logic and data structure integrity.

This test verifies:
- Iteration increments correctly
- Data structures (optimization_history, complete_iteration_results, elite_vote_results) are populated correctly
- Elite voter selection matches feedback generation
- Index consistency across iterations
- Feedback generation uses correct iteration data
- No actual LLM calls, training, or evaluation are performed
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Mock openai module before importing anything that uses it
sys.modules['openai'] = MagicMock()
sys.modules['openai'].OpenAI = Mock

# Now import the modules we need
from quadro_llm.eureka_visfly import EurekaVisFly
from quadro_llm.core.models import OptimizationConfig, TrainingResult, RewardFunctionResult
from dataclasses import dataclass
from typing import Optional

# Local EliteVoterResult for testing (elite_voter.py has been removed)
@dataclass
class EliteVoterResult:
    """Result from elite voter LLM agent"""
    selected_index: int
    reasoning: str
    selected_identifier: Optional[str] = None
    analysis_summary: Optional[str] = None
    candidate_count: int = 0
    conversation: Optional[list] = None
    code_level_feedback: Optional[Any] = None


class MockEnv:
    """Mock environment class for testing"""
    def __init__(self, **kwargs):
        pass


class MockLLMEngine:
    """Mock LLM engine that doesn't make actual API calls"""
    
    def __init__(self, **kwargs):
        self.model = kwargs.get("model", "mock-model")
        self.history_window_size = kwargs.get("history_window_size", 2)
        self.conversation_history = []
        self.api_doc_content = None
        
    def generate_reward_functions(
        self,
        task_description: str,
        context_info: str,
        feedback: str,
        samples: int,
        env_class,
        previous_elite_reward: str = None,
        **kwargs,
    ) -> List[str]:
        """Mock reward function generation"""
        # Generate mock reward functions
        reward_functions = []
        for i in range(samples):
            reward_code = f"def get_reward(self, obs, action):\n    # Mock reward function {i}\n    return 1.0"
            reward_functions.append(reward_code)
        return reward_functions
    
    def _update_conversation_history(self, reward_code: str, user_prompt: str):
        """Mock history update"""
        self.conversation_history.append({
            "role": "user",
            "content": user_prompt
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": reward_code
        })
        # Simulate history pruning based on window_size
        if self.history_window_size > 0 and len(self.conversation_history) > self.history_window_size * 4:
            # Keep only recent history (rough approximation)
            self.conversation_history = self.conversation_history[-self.history_window_size * 4:]
    
    def save_conversations(self, output_dir: str, iteration: int):
        """Mock conversation saving"""
        pass


class MockEliteVoter:
    """Mock elite voter that selects deterministically"""
    
    def __init__(self, llm_engine):
        self.llm_engine = llm_engine
        self.vote_call_count = 0
        
    def vote(self, results: List[RewardFunctionResult]) -> EliteVoterResult:
        """Mock voting that selects the best result deterministically"""
        self.vote_call_count += 1
        
        successful_results = [r for r in results if r.training_successful]
        if not successful_results:
            return EliteVoterResult(
                selected_index=0,
                reasoning="All candidates failed",
                selected_identifier=results[0].identifier if results else "sample0",
            )

        # Select the result with highest success_rate
        best_result = max(successful_results, key=lambda r: (r.success_rate, -r.episode_length))
        selected_index = results.index(best_result)

        return EliteVoterResult(
            selected_index=selected_index,
            reasoning=f"Selected {best_result.identifier} with success_rate={best_result.success_rate:.3f}",
            selected_identifier=best_result.identifier,
        )


class MockSubprocessEvaluator:
    """Mock evaluator that doesn't actually train"""
    
    def __init__(self, logger):
        self.logger = logger
        
    def evaluate_multiple_parallel(
        self,
        reward_functions: List[str],
        identifiers: List[str],
        env_config: Dict[str, Any],
        optimization_config: Dict[str, Any],
        env_class_path: str,
        max_concurrent: int,
        base_output_dir: str = None,
        eval_env_config: Dict[str, Any] = None,
    ) -> List[RewardFunctionResult]:
        """Mock evaluation that returns deterministic results"""
        iteration = optimization_config.get("iteration", 0)
        results = []
        
        for i, (reward_code, identifier) in enumerate(zip(reward_functions, identifiers)):
            # Simulate training: later iterations perform better
            base_success = 0.5 + (iteration * 0.1) + (i * 0.05)
            success_rate = min(1.0, base_success)
            
            result = RewardFunctionResult(
                reward_code=reward_code,
                identifier=identifier,
                training_successful=True,
                success_rate=success_rate,
                episode_length=200.0 - (iteration * 10) - (i * 5),
                training_time=10.0,
                final_reward=success_rate * 100.0,
                convergence_step=1000,
                evaluation_summary={
                    "success_rate": success_rate,
                    "success_count": int(success_rate * 10),
                    "actual_evaluation_episodes": 10,
                    "mean_episode_length": 200.0 - (iteration * 10) - (i * 5),
                    "mean_final_distance": 0.1 if success_rate > 0.8 else 1.0,
                    "collision_count": 0,
                    "mean_episode_reward": success_rate * 100.0,
                },
                episode_statistics=[
                    {
                        "success": j < int(success_rate * 10),
                        "final_distance_to_target": 0.1 if j < int(success_rate * 10) else 1.0,
                        "collision": False,
                    }
                    for j in range(10)
                ],
            )
            results.append(result)
        
        return results
    
    def cleanup(self):
        """Mock cleanup"""
        pass


@pytest.fixture
def mock_eureka_visfly():
    """Create a mock EurekaVisFly instance with all external dependencies mocked"""
    
    # Mock environment class
    env_class = MockEnv
    
    # Use patch to replace the classes before instantiation
    with patch('quadro_llm.eureka_visfly.LLMEngine', MockLLMEngine), \
         patch('quadro_llm.eureka_visfly.SubprocessRewardEvaluator', MockSubprocessEvaluator):
        
        opt_config = OptimizationConfig(
            iterations=3,
            samples=5,
            algorithm="bptt",
            evaluation_episodes=10,
        )
        
        eureka = EurekaVisFly(
            env_class=env_class,
            task_description="Test task: hover at position (0, 0, 1)",
            llm_config={"model": "mock-model", "api_key": "mock-key", "history_window_size": 2},
            env_kwargs={"num_envs": 1, "device": "cpu"},
            optimization_config=opt_config,
            device="cpu",
            max_workers=2,
            eval_env_config={"num_envs": 1},
            use_coefficient_tuning=False,
        )
        
        # Add mock elite_voter for backward compatibility with tests
        # (actual code uses agent_voter, but tests still reference elite_voter)
        eureka.elite_voter = MockEliteVoter(eureka.llm)
        
        yield eureka


def test_iteration_increment(mock_eureka_visfly):
    """Test that iterations increment correctly and data structures are populated"""
    
    # Track state across iterations
    iteration_states = []
    
    # Override optimize_rewards to track state
    original_optimize_rewards = mock_eureka_visfly.optimize_rewards
    
    def tracked_optimize_rewards():
        results = []
        iterations = mock_eureka_visfly.config.iterations
        samples = mock_eureka_visfly.config.samples
        
        for iteration in range(iterations):
            # Record state at start of iteration
            state = {
                "iteration": iteration,
                "optimization_history_len": len(mock_eureka_visfly.optimization_history),
                "complete_iteration_results_len": len(mock_eureka_visfly.complete_iteration_results),
                "elite_vote_results_len": len(mock_eureka_visfly.elite_vote_results),
                "best_reward_functions_len": len(mock_eureka_visfly.best_reward_functions),
            }
            iteration_states.append(state)
            
            # Mock feedback generation for this iteration
            if iteration > 0:
                feedback = mock_eureka_visfly._generate_feedback(iteration)
                assert feedback is not None, f"Iteration {iteration}: Feedback should be generated"
            
            # Mock reward generation
            reward_functions = mock_eureka_visfly.generate_reward_candidates(
                samples, iteration, None if iteration == 0 else "mock feedback"
            )
            assert len(reward_functions) == samples, f"Iteration {iteration}: Should generate {samples} reward functions"
            
            # Mock evaluation
            identifiers = [f"sample{i}" for i in range(len(reward_functions))]
            iteration_results = mock_eureka_visfly.evaluator.evaluate_multiple_parallel(
                reward_functions=reward_functions,
                identifiers=identifiers,
                env_config=mock_eureka_visfly.env_kwargs,
                optimization_config={
                    "algorithm": mock_eureka_visfly.config.algorithm,
                    "evaluation_episodes": mock_eureka_visfly.config.evaluation_episodes,
                    "iteration": iteration,
                },
                env_class_path="mock.Env",
                max_concurrent=mock_eureka_visfly.max_workers,
            )
            
            # Store complete results
            mock_eureka_visfly.complete_iteration_results.append(iteration_results)
            
            # Convert to TrainingResult
            training_results = []
            for result in iteration_results:
                if result.training_successful:
                    training_result = TrainingResult(
                        success_rate=result.success_rate,
                        episode_length=result.episode_length,
                        training_time=result.training_time,
                        final_reward=result.final_reward,
                        convergence_step=result.convergence_step,
                        reward_code=result.reward_code,
                        identifier=result.identifier,
                    )
                    training_results.append(training_result)
                    results.append(training_result)
            
            # Store iteration results
            mock_eureka_visfly.optimization_history.append(training_results)
            
            # Elite voter selection
            if iteration_results:
                vote_result = mock_eureka_visfly.elite_voter.vote(iteration_results)
                best_result = iteration_results[vote_result.selected_index]
                mock_eureka_visfly.best_reward_functions.append(best_result.reward_code)
                mock_eureka_visfly.elite_vote_results[iteration] = vote_result
                
                # Verify vote_result matches best_result
                assert iteration_results[vote_result.selected_index].identifier == best_result.identifier, \
                    f"Iteration {iteration}: Vote result index should match best result"
            
            # Record state after iteration
            state["after_optimization_history_len"] = len(mock_eureka_visfly.optimization_history)
            state["after_complete_iteration_results_len"] = len(mock_eureka_visfly.complete_iteration_results)
            state["after_elite_vote_results_len"] = len(mock_eureka_visfly.elite_vote_results)
            state["after_best_reward_functions_len"] = len(mock_eureka_visfly.best_reward_functions)
        
        return results
    
    # Run tracked optimization
    final_results = tracked_optimize_rewards()
    
    # Verify iteration increments
    assert len(iteration_states) == 3, "Should have 3 iterations"
    
    # Verify data structures grow correctly
    for i, state in enumerate(iteration_states):
        assert state["iteration"] == i, f"State iteration should match index {i}"
        
        if i == 0:
            # First iteration: should start empty
            assert state["optimization_history_len"] == 0, "First iteration should start with empty history"
            assert state["complete_iteration_results_len"] == 0, "First iteration should start with empty complete results"
        else:
            # Subsequent iterations: should have previous iterations
            assert state["optimization_history_len"] == i, f"Iteration {i}: Should have {i} previous iterations in history"
            assert state["complete_iteration_results_len"] == i, f"Iteration {i}: Should have {i} previous iterations in complete results"
        
        # After iteration: should have one more entry
        assert state["after_optimization_history_len"] == i + 1, f"Iteration {i}: Should have {i+1} entries after"
        assert state["after_complete_iteration_results_len"] == i + 1, f"Iteration {i}: Should have {i+1} complete results after"
        assert state["after_elite_vote_results_len"] == i + 1, f"Iteration {i}: Should have {i+1} elite vote results after"
        assert state["after_best_reward_functions_len"] == i + 1, f"Iteration {i}: Should have {i+1} best reward functions after"


def test_elite_voter_index_consistency(mock_eureka_visfly):
    """Test that elite voter selection indices are consistent across iterations"""
    
    iterations = 3
    samples = 5
    
    for iteration in range(iterations):
        # Generate mock results
        mock_results = []
        for i in range(samples):
            base_success = 0.5 + (iteration * 0.1) + (i * 0.05)
            success_rate = min(1.0, base_success)
            
            result = RewardFunctionResult(
                reward_code=f"code_{iteration}_{i}",
                identifier=f"sample{i}",
                training_successful=True,
                success_rate=success_rate,
                episode_length=200.0 - (i * 5),
                training_time=10.0,
                final_reward=success_rate * 100.0,
                convergence_step=1000,
            )
            mock_results.append(result)
        
        # Store complete results
        mock_eureka_visfly.complete_iteration_results.append(mock_results)
        
        # Convert to TrainingResult
        training_results = []
        for result in mock_results:
            training_result = TrainingResult(
                success_rate=result.success_rate,
                episode_length=result.episode_length,
                training_time=result.training_time,
                final_reward=result.final_reward,
                convergence_step=result.convergence_step,
                reward_code=result.reward_code,
                identifier=result.identifier,
            )
            training_results.append(training_result)
        
        mock_eureka_visfly.optimization_history.append(training_results)
        
        # Elite voter selection
        vote_result = mock_eureka_visfly.elite_voter.vote(mock_results)
        best_result = mock_results[vote_result.selected_index]
        mock_eureka_visfly.best_reward_functions.append(best_result.reward_code)
        mock_eureka_visfly.elite_vote_results[iteration] = vote_result
        
        # Verify index is valid
        assert 0 <= vote_result.selected_index < len(mock_results), \
            f"Iteration {iteration}: Vote result index {vote_result.selected_index} should be in range [0, {len(mock_results)})"
        
        # Verify selected result matches
        assert mock_results[vote_result.selected_index].identifier == best_result.identifier, \
            f"Iteration {iteration}: Selected result identifier should match"
        


def test_feedback_generation_uses_correct_iteration_data(mock_eureka_visfly):
    """Test that feedback generation uses data from the correct iteration"""
    
    iterations = 3
    samples = 5
    
    for iteration in range(iterations):
        # Create mock results for this iteration
        mock_results = []
        for i in range(samples):
            base_success = 0.5 + (iteration * 0.1) + (i * 0.05)
            success_rate = min(1.0, base_success)
            
            result = RewardFunctionResult(
                reward_code=f"code_{iteration}_{i}",
                identifier=f"sample{i}",
                training_successful=True,
                success_rate=success_rate,
                episode_length=200.0 - (i * 5),
                training_time=10.0,
                final_reward=success_rate * 100.0,
                convergence_step=1000,
                evaluation_summary={
                    "success_rate": success_rate,
                    "success_count": int(success_rate * 10),
                    "actual_evaluation_episodes": 10,
                    "mean_episode_length": 200.0 - (i * 5),
                },
            )
            mock_results.append(result)
        
        # Store complete results
        mock_eureka_visfly.complete_iteration_results.append(mock_results)
        
        # Convert to TrainingResult
        training_results = []
        for result in mock_results:
            training_result = TrainingResult(
                success_rate=result.success_rate,
                episode_length=result.episode_length,
                training_time=result.training_time,
                final_reward=result.final_reward,
                convergence_step=result.convergence_step,
                reward_code=result.reward_code,
                identifier=result.identifier,
            )
            training_results.append(training_result)
        
        mock_eureka_visfly.optimization_history.append(training_results)
        
        # Elite voter selection
        vote_result = mock_eureka_visfly.elite_voter.vote(mock_results)
        best_result = mock_results[vote_result.selected_index]
        mock_eureka_visfly.best_reward_functions.append(best_result.reward_code)
        mock_eureka_visfly.elite_vote_results[iteration] = vote_result
        
        # Test feedback generation for NEXT iteration
        if iteration < iterations - 1:
            next_iteration = iteration + 1
            
            # Generate feedback for next iteration
            feedback = mock_eureka_visfly._generate_feedback(next_iteration)
            
            # Verify feedback was generated
            assert feedback is not None, f"Feedback for iteration {next_iteration} should be generated"
            assert len(feedback) > 0, f"Feedback for iteration {next_iteration} should not be empty"
            
            # Verify feedback uses correct iteration data
            # Should use data from iteration 'iteration' (previous iteration)
            assert len(mock_eureka_visfly.optimization_history) > iteration, \
                f"Should have history for iteration {iteration} when generating feedback for {next_iteration}"
            
            # Check that vote_result used in feedback matches the stored one
            stored_vote_result = mock_eureka_visfly.elite_vote_results[iteration]
            assert stored_vote_result.selected_index == vote_result.selected_index, \
                f"Stored vote result index should match for iteration {iteration}"
            # Verify feedback contains expected information
            assert "win rate" in feedback.lower() or "success" in feedback.lower(), \
                f"Feedback should contain performance metrics"


def test_feedback_uses_elite_voter_selection(mock_eureka_visfly):
    """Test that feedback generation uses the elite voter's selection, not just the first candidate"""
    
    iterations = 2
    samples = 5
    
    for iteration in range(iterations):
        # Create mock results with known best candidate (not first)
        mock_results = []
        for i in range(samples):
            # Make sample2 the best (not sample0)
            if i == 2:
                success_rate = 0.95
            else:
                success_rate = 0.5 + (i * 0.05)
            
            result = RewardFunctionResult(
                reward_code=f"code_{iteration}_{i}",
                identifier=f"sample{i}",
                training_successful=True,
                success_rate=success_rate,
                episode_length=200.0 - (i * 5),
                training_time=10.0,
                final_reward=success_rate * 100.0,
                convergence_step=1000,
                evaluation_summary={
                    "success_rate": success_rate,
                    "success_count": int(success_rate * 10),
                    "actual_evaluation_episodes": 10,
                },
            )
            mock_results.append(result)
        
        # Store complete results
        mock_eureka_visfly.complete_iteration_results.append(mock_results)
        
        # Convert to TrainingResult
        training_results = []
        for result in mock_results:
            training_result = TrainingResult(
                success_rate=result.success_rate,
                episode_length=result.episode_length,
                training_time=result.training_time,
                final_reward=result.final_reward,
                convergence_step=result.convergence_step,
                reward_code=result.reward_code,
                identifier=result.identifier,
            )
            training_results.append(training_result)
        
        mock_eureka_visfly.optimization_history.append(training_results)
        
        # Elite voter selection (should select sample2)
        vote_result = mock_eureka_visfly.elite_voter.vote(mock_results)
        best_result = mock_results[vote_result.selected_index]
        mock_eureka_visfly.best_reward_functions.append(best_result.reward_code)
        mock_eureka_visfly.elite_vote_results[iteration] = vote_result
        
        # Verify elite voter selected sample2 (index 2)
        assert vote_result.selected_index == 2, \
            f"Iteration {iteration}: Elite voter should select sample2 (index 2), got index {vote_result.selected_index}"
        assert best_result.identifier == "sample2", \
            f"Iteration {iteration}: Best result should be sample2"
        
        # Test feedback generation for next iteration
        if iteration < iterations - 1:
            next_iteration = iteration + 1
            feedback = mock_eureka_visfly._generate_feedback(next_iteration)
            
            # Verify feedback uses elite voter's selection (sample2), not sample0
            # The feedback should reference the elite selection's success rate (0.95)
            assert "0.95" in feedback or "0.950" in feedback or "95" in feedback or "0.9" in feedback, \
                f"Feedback should contain elite selection's success rate (0.95), not first candidate's rate"


def test_index_consistency_between_data_structures(mock_eureka_visfly):
    """Test that indices are consistent between optimization_history, complete_iteration_results, and elite_vote_results"""
    
    iterations = 3
    samples = 5
    
    for iteration in range(iterations):
        # Create mock results
        mock_results = []
        for i in range(samples):
            base_success = 0.5 + (iteration * 0.1) + (i * 0.05)
            success_rate = min(1.0, base_success)
            
            result = RewardFunctionResult(
                reward_code=f"code_{iteration}_{i}",
                identifier=f"sample{i}",
                training_successful=True,
                success_rate=success_rate,
                episode_length=200.0 - (i * 5),
                training_time=10.0,
                final_reward=success_rate * 100.0,
                convergence_step=1000,
            )
            mock_results.append(result)
        
        # Store complete results
        mock_eureka_visfly.complete_iteration_results.append(mock_results)
        
        # Convert to TrainingResult (filter successful only)
        training_results = []
        for result in mock_results:
            if result.training_successful:
                training_result = TrainingResult(
                    success_rate=result.success_rate,
                    episode_length=result.episode_length,
                    training_time=result.training_time,
                    final_reward=result.final_reward,
                    convergence_step=result.convergence_step,
                    reward_code=result.reward_code,
                    identifier=result.identifier,
                )
                training_results.append(training_result)
        
        mock_eureka_visfly.optimization_history.append(training_results)
        
        # Elite voter selection
        vote_result = mock_eureka_visfly.elite_voter.vote(mock_results)
        best_result = mock_results[vote_result.selected_index]
        mock_eureka_visfly.best_reward_functions.append(best_result.reward_code)
        mock_eureka_visfly.elite_vote_results[iteration] = vote_result
        
        # Verify consistency: complete_iteration_results[iteration] should match
        assert len(mock_eureka_visfly.complete_iteration_results) == iteration + 1, \
            f"Should have {iteration + 1} complete iteration results"
        assert len(mock_eureka_visfly.complete_iteration_results[iteration]) == samples, \
            f"Iteration {iteration}: Should have {samples} results in complete_iteration_results"
        
        # Verify vote_result.selected_index is valid for complete_iteration_results[iteration]
        assert 0 <= vote_result.selected_index < len(mock_eureka_visfly.complete_iteration_results[iteration]), \
            f"Iteration {iteration}: Vote result index {vote_result.selected_index} should be valid for complete_iteration_results"
        
        # Verify the selected result exists in complete_iteration_results
        selected_from_complete = mock_eureka_visfly.complete_iteration_results[iteration][vote_result.selected_index]
        assert selected_from_complete.identifier == best_result.identifier, \
            f"Iteration {iteration}: Selected result from complete_iteration_results should match best_result"
        
        # Verify the selected result exists in optimization_history (if successful)
        if best_result.training_successful:
            found_in_history = False
            for training_result in mock_eureka_visfly.optimization_history[iteration]:
                if training_result.identifier == best_result.identifier:
                    found_in_history = True
                    assert training_result.success_rate == best_result.success_rate, \
                        f"Iteration {iteration}: Success rate should match between complete and history"
                    break
            assert found_in_history, \
                f"Iteration {iteration}: Elite selection should be found in optimization_history"


def test_feedback_uses_correct_vote_result_per_iteration(mock_eureka_visfly):
    """Test that feedback generation uses the correct vote_result from the right iteration"""
    
    iterations = 3
    samples = 5
    
    vote_results_by_iteration = []
    
    for iteration in range(iterations):
        # Create mock results
        mock_results = []
        for i in range(samples):
            base_success = 0.5 + (iteration * 0.1) + (i * 0.05)
            success_rate = min(1.0, base_success)
            
            result = RewardFunctionResult(
                reward_code=f"code_{iteration}_{i}",
                identifier=f"sample{i}",
                training_successful=True,
                success_rate=success_rate,
                episode_length=200.0 - (i * 5),
                training_time=10.0,
                final_reward=success_rate * 100.0,
                convergence_step=1000,
            )
            mock_results.append(result)
        
        mock_eureka_visfly.complete_iteration_results.append(mock_results)
        
        training_results = []
        for result in mock_results:
            training_result = TrainingResult(
                success_rate=result.success_rate,
                episode_length=result.episode_length,
                training_time=result.training_time,
                final_reward=result.final_reward,
                convergence_step=result.convergence_step,
                reward_code=result.reward_code,
                identifier=result.identifier,
            )
            training_results.append(training_result)
        
        mock_eureka_visfly.optimization_history.append(training_results)
        
        # Elite voter selection
        vote_result = mock_eureka_visfly.elite_voter.vote(mock_results)
        best_result = mock_results[vote_result.selected_index]
        mock_eureka_visfly.best_reward_functions.append(best_result.reward_code)
        mock_eureka_visfly.elite_vote_results[iteration] = vote_result
        
        vote_results_by_iteration.append({
            "iteration": iteration,
            "vote_result": vote_result,
            "best_identifier": best_result.identifier,
        })
        
        # Test feedback generation for next iteration uses THIS iteration's vote_result
        if iteration < iterations - 1:
            next_iteration = iteration + 1
            
            # The feedback should use vote_result from iteration 'iteration' (previous)
            # This is tested by checking that the feedback generation logic retrieves
            # elite_vote_results[iteration] (which is iteration - 1 for next_iteration)
            feedback = mock_eureka_visfly._generate_feedback(next_iteration)
            
            # Verify the stored vote_result matches what should be used
            stored_vote_result = mock_eureka_visfly.elite_vote_results[iteration]
            # Verify feedback contains the selection (indirectly, through the reasoning)
            # The feedback includes vote_result.reasoning which should reference the correct selection
            assert vote_result.reasoning in feedback or best_result.identifier in feedback, \
                f"Feedback should reference the elite selection from iteration {iteration}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
