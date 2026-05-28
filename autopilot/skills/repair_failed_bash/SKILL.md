---
name: repair_failed_bash
description: Repair failed commands, path mistakes, schema mismatches, and timeouts without asking the human.
---

Allowed tools: `bash`, `cat`, `grep`.

Rules:
- Do not ask human for bugs.
- Do ask human for policy choices, data-risk choices, objective changes, or irreversible resource decisions.

Procedure:
1. Read stderr/stdout and relevant logs.
2. Classify failure: path, missing dependency, timeout, process conflict, YAML/schema, data contamination, or tool misuse.
3. Use `grep`/`cat`/`bash ls/find` to discover the correct path or state.
4. Make the smallest correction in the run workspace.
5. Re-run validation before continuing.
