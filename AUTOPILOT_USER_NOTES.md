# AUTOPILOT_USER_NOTES

Write running advice for the autonomous agent here. The agent reads this file
at the start of each iteration. Prefer short bullets with optional tags.

Tags:
- @policy: hard rule or invariant.
- @preference: user preference.
- @advice: live guidance for the current/next run.
- @correction: correction to a behavior the agent just showed.
- @decision: a choice boundary where the agent should ask before proceeding.
- @memory: something the agent may draft into long-term memory.
- @skill: something the agent may draft into a skill update.

Examples:
- @policy AIME24 benchmark cases are eval_only and must never enter training data.
- @preference Repair bugs automatically; ask only for real choices.
- @advice If vLLM is still loading checkpoint shards, wait before killing it.

@advice vLLM context length = 32k for math problems.
@advice 2x H800 GPUs are more than capable of training a 1.7B model with full-parameters, so don't use LoRA.
@advice Now we are in the stage of training, as monitor, since the estimated duration is approximately 4 hours, you can read train.log every 30 minutes (sleep) to avoid unnecessary token consumption.