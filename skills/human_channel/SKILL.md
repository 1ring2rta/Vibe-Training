---
name: human_channel
description: Use AUTOPILOT_USER_NOTES.md, run-local INBOX.md, and answer_human to communicate with the user without blocking unless a real choice is required.
---

Use when the user has provided live notes, corrections, preferences, or when the agent needs to report a decision boundary.

Available channel files:
- `AUTOPILOT_USER_NOTES.md`: project-level live user notes and preferences.
- `.autopilot/human/INBOX.md`: run-local notes the user can append while the run is active.
- `.autopilot/human/OUTBOX.md`: agent replies for the user.
- `.autopilot/human/dialog.jsonl`: structured human/agent communication log.
- `.autopilot/human/DECISIONS.md`: open decisions requiring a user response.

Rules:
1. Read `world_state.human_channel.notes` every iteration.
2. Acknowledge useful notes with `answer_human(kind="ack", requires_response=false)` when it helps the user understand the agent adapted.
3. Do not block for bugs, path errors, schema errors, tool aliases, timeouts, missing files, or normal repair work.
4. Use `answer_human(kind="decision_request", requires_response=true, choices=[...])` only for real choices: contamination risk, benchmark policy, objective changes, resource conflicts, destructive cleanup, or accepting lower-quality data.
5. Keep decision choices concrete and recommend the safest default.
6. If a note says `@memory`, draft a memory update at run end. If a note says `@skill`, draft a skill update; do not silently promote policy-changing skills.

Example acknowledgement:
```json
{"action_type":"run_tool","tool_name":"answer_human","arguments":{"message":"Noted. I will treat AIME24 as eval_only and check train-like artifacts before training.","kind":"ack","requires_response":false}}
```

Example decision request:
```json
{"action_type":"run_tool","tool_name":"answer_human","arguments":{"message":"Choose data strategy: smaller clean data or larger data with unresolved contamination risk?","kind":"decision_request","requires_response":true,"choices":[{"id":"clean","label":"Use smaller clean data"},{"id":"risky","label":"Use larger risky data"}],"recommended_choice":"clean"}}
```
