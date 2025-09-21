# Refactor & Expansion Master Plan

## Guiding Principles
- Preserve current CLI/API surfaces (`EurekaVisFly.optimize_rewards`, `main.py`, `run.py`) while the new architecture incubates behind feature flags.
- Favour clear module boundaries and dependency injection so agents, evaluators, and VLM services can be swapped or composed without touching orchestration code.
- Build small, testable helpers; defer heavy refactors until unit coverage and integration harnesses are in place.

## Target Package Layout
```
quadro_llm/
  bootstrap/              # CLI & Hydra wiring helpers
    __init__.py, loaders.py
  orchestration/          # Pipelines and run controllers
    pipeline.py, iteration.py, result_reporter.py
  agents/
    base.py               # Protocols: TaskPlanner, RewardEngineer, ResultReviewer
    single_agent.py       # Long-context agent implementation
    multi_agent.py        # Manager orchestrating task analyzer / reward engineer / evaluator
    memory.py             # Conversation buffers & summarizers
  llm/
    engine.py             # Thin transport layer around provider SDKs
    prompts/
      __init__.py
      builders.py         # Prompt templates assembled from dataclasses
  evaluators/
    subprocess.py         # Existing evaluator split into launcher, worker, collector
    metrics.py            # Shared analysis utilities
  artifacts/
    paths.py              # GeneratedArtifactPaths, artifact validators
    writers.py            # Save reward code, conversations, videos
  vlm/
    client_base.py        # VisionLanguageClient protocol
    openai.py, custom.py  # Concrete adapters
    video_summarizer.py   # Pipeline that produces behaviour commentary
  data_models/
    __init__.py
    specs.py              # TaskSpec, RewardProposal, EvaluationSnapshot, ConversationLog
```

## Workstreams
### 1. Core Orchestration Layer
- Break `EurekaVisFly.optimize_rewards` into `_plan_iteration`, `_evaluate_candidates`, and `_aggregate_results` housed in `orchestration/iteration.py`.
- Move tensorboard feedback logic into `orchestration/result_reporter.py`; expose a pure `build_feedback(iteration: IterationData) -> FeedbackBundle` for reuse.
- Introduce `OrchestrationContext` (env metadata, hydra dirs, logger, device) shared across strategies.

### 2. Agent System (Multi & Single)
- Define `TaskPlanner`, `RewardEngineer`, and `ResultReviewer` protocols in `agents/base.py` with minimal method sets (`prepare_task`, `propose_rewards`, `assess_iteration`).
- Implement `MultiAgentController` that sequences planner → engineer → reviewer and manages inter-agent memory via `ConversationState` objects.
- Provide `SingleAgentController` that wraps a solo LLM with long-context buffer; reuse same protocol so orchestration code just toggles controller selection based on `cfg.agent.mode`.
- Ensure both controllers accept dependency-injected LLM/VLM clients and persistence hooks.

### 3. Long-Context & Memory Infrastructure
- Add `memory.py` with ring buffers, semantic summarizers, and optional vector store adapters (pluggable, default in-memory) to keep context windows under provider limits.
- Expose streaming API for appending evaluator metrics so agents can “think” across iterations.

### 4. Vision-Language Module
- Implement `VisionLanguageClient` protocol supporting `summarize_video(path, metadata) -> SummaryReport` and optional screenshot commentary.
- Add transport wrappers: `OpenAIVLMClient`, `HTTPVLMClient` (config-driven endpoint).
- Build `video_summarizer.VideoSummarizer` service that coordinates video encoding (FFmpeg hooks), uploads via client, and emits markdown/JSON artefacts stored via `artifacts.writers`.

### 5. Evaluator & Artifact Management
- Extract existing `_save_reward_functions` logic into `artifacts/writers.py` using `GeneratedArtifactPaths` dataclass for consistent naming (`iter{n}`, `sample{idx}`).
- Split `SubprocessRewardEvaluator.evaluate_multiple_parallel` into `Plan` (determine work items), `Dispatch` (start workers, manage timeouts), and `Collect` (normalize outputs) for finer-grained testing.
- Centralize artifact discovery (reward functions, tensorboard logs, conversation dumps, VLM summaries) in `artifacts/paths.py` so downstream consumers need no string concatenation.

### 6. Configuration & Bootstrap
- Create `bootstrap/loaders.py` to translate Hydra configs into strongly typed `AgentOrchestrationConfig`, `LLMClientConfig`, `VLMClientConfig` objects before orchestrator instantiation.
- Update CLI flags to expose multi-agent vs single-agent selection (`--agent-mode`, `--memory-policy`, `--vlm-endpoint`) with sensible defaults tied to current behaviour.

### 7. Testing & Tooling
- Add unit suites for new dataclasses, prompt builders, memory buffers, and artifact path logic.
- Provide contract tests ensuring both agent controllers honour the protocol (shared fixtures mocking LLM responses).
- Extend integration test to simulate one iteration with multi-agent controller and mocked VLM client, confirming artifact layout.

## Sequencing & Dependencies
1. **Stabilize data models & artifact paths** (Workstreams 5 & 6) to minimize churn for later steps.
2. **Refactor orchestration iteration flow** while keeping old APIs (Workstream 1).
3. **Introduce agent protocols and single-agent controller** behind config flag; migrate existing logic to new controller (Workstream 2 baseline).
4. **Add multi-agent manager** and update pipeline to switch controllers via config.
5. **Layer in memory utilities** and extend single-agent controller to support long contexts.
6. **Integrate VLM module** once orchestration and controllers expose hooks for evaluation media.

## Risks & Mitigations
- **Regression risk**: Maintain shadow mode where new controllers log outputs without affecting reward selection until validated; compare metrics with golden runs.
- **Complexity creep**: Enforce linting + type checks on new modules; add concise module-level READMEs documenting responsibilities.
- **Provider limits**: Design clients to accept rate-limiter injectors; include retry/backoff policies configurable via Hydra.

## Verification Plan
- Incrementally expand `pytest --cov=quadro_llm` coverage thresholds as modules land.
- Add golden-file tests for artifact naming and VLM summaries (mocked responses) to prevent regressions.
- Document new architecture in `AGENTS.md` after each major milestone and ensure README diagrams mirror the final layout.
