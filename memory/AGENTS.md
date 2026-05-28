# Always-on agent memory

- Autonomous mode uses only atomic actions: `run_tool` and `stop`.
- Atomic execution tools are `bash`, `cat`, `grep`, `web_search`, and `browser`; the communication tool is `answer_human`.
- Use `answer_human` for acknowledgements and real decision requests. Do not use `ask_human`.
- `AUTOPILOT_USER_NOTES.md` and `.autopilot/human/INBOX.md` are live user guidance; read `world_state.human_channel.notes` every iteration.
- Skills are instruction files. Read the relevant `SKILL.md` with `cat`; do not call skills as tools.
- Do not run hidden workflow wrappers or macro actions. Compose the work from visible atomic tool calls.
- Ask the user only for real choices: safety policy, contamination tradeoffs, resource allocation conflicts, or objective changes.
- Do not ask the user to fix JSON, schema, path, timeout, command, or missing-file bugs; repair those with tools.
