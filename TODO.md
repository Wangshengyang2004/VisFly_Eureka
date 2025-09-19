# Prompt Design TODO

- [x] Rework `quadro_llm/llm/prompts.py` system prompt to match original Eureka tone while keeping VisFly-specific gradient safety tips; verify wording against `backup/Eureka/eureka/utils/prompts/initial_system.txt`.
- [x] Introduce an explicit environment API appendix (see `api-doc.txt`) into prompt assembly so agents learn VisFly state names without over-inflating token count; gate inclusion behind a config flag.
- [x] Normalize user prompt structure toward Eureka format (task description, env code stub, feedback) and document the template so future tasks can swap environments cleanly.
- [x] Audit feedback strings produced in `quadro_llm/utils/tensorboard_utils.py` and `EurekaVisFly.get_iteration_feedback` to ensure terminology matches the updated prompt language (e.g., "task_score" vs "consecutive_successes").
- [x] Build regression tests (golden prompt snapshots) for at least one navigation and one landing task to catch accidental prompt drift.
