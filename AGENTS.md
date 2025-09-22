# Repository Guidelines

## Project Structure & Module Organization
Core automation lives in `quadro_llm/` (Hydra pipeline, LLM interfaces, utilities) with executable entry points in `main.py` and `run.py`. Training algorithms sit in `algorithms/` while simulator sources stay in `VisFly/`, mirroring the upstream VisFly layout. Hydra configuration trees are under `configs/` (environment, algorithm, and provider folders). Tests reside in `tests/` with `unit/` and `integration/` subpackages, and coverage artifacts land in `htmlcov/`. Generated models, rollouts, and logs are stored in `saved/` and `results/`; keep large artifacts out of version control.

## Build, Test, and Development Commands
Install in editable mode and pull dependencies with `pip install -e .` followed by `pip install -r requirements.txt`. Run the Hydra-driven workflow via `python main.py` or target a task, for example `python main.py env=racing_env llm.model=gpt-4o`. To bypass LLM generation, use `python run.py --env navigation --algorithm ppo --num_envs 48`. TensorBoard utilities live in `monitor_tensorboard.py`; launch with `python monitor_tensorboard.py --logdir results/latest` to inspect runs.

## Coding Style & Naming Conventions
Python code follows 4-space indentation, type-hinted functions, and descriptive docstrings. Format with `black .` and lint with `flake8`, both declared in `requirements.txt`. Use snake_case for modules, functions, and configuration keys, PascalCase for classes, and keep Hydra config filenames lowercase with dashes (e.g., `configs/envs/navigation.yaml`).

## Testing Guidelines
Pytest is configured in `pytest.ini` to collect from `tests/` and enforce 20% minimum coverage. Run the full suite with `pytest` or focus on fast checks via `pytest -m "not slow and not gpu"`. GPU or LLM-dependent tests are explicitly marked; skip them locally unless hardware and credentials are available. Coverage HTML reports render to `htmlcov/index.html` for review before submitting changes.

## Commit & Pull Request Guidelines
Recent history mixes Conventional Commit prefixes (`feat:`, `fix:`) with sentence-style summaries. Prefer `type: imperative summary` (e.g., `feat: add hydra config sweeper`) to keep the log consistent. For pull requests, include the problem statement, the configuration overrides used, links to relevant issues, and screenshots or TensorBoard snapshots when behaviour changes. Note any required secrets or hardware so reviewers can reproduce results.

## Security & Configuration Tips
Never commit API keys: `configs/api_keys.yaml` is gitignored; populate it by copying `configs/api_keys.example.yaml` or export environment variables. Treat `VisFly/environment*.yml` as read-only templates—create overrides instead of editing in place. Scrub logs for LLM prompts or credentials before attaching them to issues.
