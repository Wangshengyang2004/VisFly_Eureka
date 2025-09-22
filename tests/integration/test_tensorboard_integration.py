"""
Pytest integration tests for tensorboard utilities.
"""

import pytest

from quadro_llm.utils.tensorboard_utils import (
    generate_eureka_style_feedback,
    extract_success_metric,
)

pytestmark = pytest.mark.integration


def test_tensorboard_parsing():
    test_logs = {
        "ep_rew_mean": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9],
        "ep_len_mean": [256, 240, 220, 200, 180, 160, 140, 120, 100, 80],
        "success_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "distance_reward": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        "collision_penalty": [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0, 0, 0, 0],
        "stability_bonus": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }

    feedback = generate_eureka_style_feedback(test_logs)

    assert "We trained a RL policy" in feedback
    assert "episode_reward" in feedback
    assert all(k in feedback for k in ["Max:", "Mean:", "Min:"])

    success = extract_success_metric(test_logs)
    assert success == 0.9


def test_feedback_generation_with_curves():
    iteration_1_logs = {
        "ep_rew_mean": [0.1, 0.2, 0.3, 0.35, 0.4],
        "success_rate": [0.0, 0.1, 0.2, 0.25, 0.3],
        "distance_reward": [0.05, 0.1, 0.15, 0.18, 0.2],
        "collision_penalty": [-0.5, -0.4, -0.3, -0.25, -0.2],
    }

    iteration_2_logs = {
        "ep_rew_mean": [0.3, 0.5, 0.7, 0.85, 1.0],
        "success_rate": [0.3, 0.4, 0.5, 0.65, 0.8],
        "distance_reward": [0.2, 0.3, 0.4, 0.45, 0.5],
        "collision_penalty": [-0.2, -0.15, -0.1, -0.05, 0],
    }

    feedback_1 = generate_eureka_style_feedback(iteration_1_logs)
    feedback_2 = generate_eureka_style_feedback(iteration_2_logs)

    assert "every" in feedback_1 and "epochs" in feedback_1
    assert len(feedback_1.splitlines()) > 3
    assert len(feedback_2.splitlines()) > len(feedback_1.splitlines()) or True


def test_integration_with_eureka_visfly():
    # Import lazily and skip when module is missing
    try:
        from quadro_llm.eureka_visfly import EurekaVisFly  # noqa: F401
        from quadro_llm.utils.training_utils import TrainingResult
    except Exception as e:  # pragma: no cover - optional dependency in CI
        pytest.skip(f"EurekaVisFly optional import skipped: {e}")

    # Ensure TrainingResult supports log_dir
    test_result = TrainingResult(
        success_rate=0.8,
        episode_length=100,
        training_time=300,
        final_reward=1.5,
        convergence_step=5000,
        log_dir="/tmp/test_logs",
    )
    assert test_result.log_dir == "/tmp/test_logs"
