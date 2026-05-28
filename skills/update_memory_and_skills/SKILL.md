---
name: update_memory_and_skills
description: At the end of a run or after repeated failures, draft durable memory and skill updates.
---

Allowed tools: `bash`, `cat`, `grep`.

Procedure:
1. Inspect event logs and autonomous reports.
2. Extract stable facts: environment paths, timeouts, startup behavior, recurring bugs, and successful commands.
3. Write low-risk facts to `.autopilot/memory/drafts/`.
4. Write skill improvement drafts to `.autopilot/skills_drafts/`.
5. Do not promote policy relaxations, benchmark contamination, or risky source choices automatically.
6. Ask human only before changing safety policy or trusted skills.
