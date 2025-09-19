import textwrap

from quadro_llm.llm.prompts import create_system_prompt, create_user_prompt


def test_system_prompt_contains_key_guidance():
    prompt = create_system_prompt()
    assert "You are a reward engineer" in prompt
    assert "Return a 1-D tensor" in prompt
    assert "Avoid in-place" in prompt
    assert "SHAC/BPTT REQUIRE" in prompt


def test_navigation_prompt_snapshot():
    env_code = textwrap.dedent(
        """
        class NavigationEnv:
            def get_reward(self):
                pass
        """
    ).strip()
    api_doc = "field_a, field_b"
    context = {"action_space": "Box(4,)", "observation_space": "Dict(state, depth)"}
    feedback = "Prior reward oscillated and ignored collisions."

    prompt = create_user_prompt(
        task_description="Navigate to the goal while avoiding obstacles.",
        context_info=context,
        feedback=feedback,
        env_code=env_code,
        api_doc=api_doc,
    )

    expected = (
        "The Python environment is:\n"
        "```python\n"
        f"{env_code}\n"
        "```\n\n"
        "Environment API reference (read once, no need to repeat in output):\n"
        "```text\n"
        "field_a, field_b\n"
        "```\n\n"
        "Task: Navigate to the goal while avoiding obstacles.\n\n"
        "Key environment details:\n"
        "- action_space: Box(4,)\n"
        "- observation_space: Dict(state, depth)\n\n"
        "Feedback from previous attempts (address every point):\n"
        "Prior reward oscillated and ignored collisions.\n\n"
        "Return only the complete `def get_reward(self) -> torch.Tensor` implementation."
    )

    assert prompt == expected


def test_landing_prompt_snapshot():
    env_code = "class LandingEnv: ..."
    api_doc = "altitude, touchdown_speed"
    context = {"max_episode_steps": 256}

    prompt = create_user_prompt(
        task_description="Land smoothly on the pad.",
        context_info=context,
        env_code=env_code,
        api_doc=api_doc,
    )

    expected = (
        "The Python environment is:\n"
        "```python\n"
        "class LandingEnv: ...\n"
        "```\n\n"
        "Environment API reference (read once, no need to repeat in output):\n"
        "```text\n"
        "altitude, touchdown_speed\n"
        "```\n\n"
        "Task: Land smoothly on the pad.\n\n"
        "Key environment details:\n"
        "- max_episode_steps: 256\n\n"
        "Return only the complete `def get_reward(self) -> torch.Tensor` implementation."
    )

    assert prompt == expected
