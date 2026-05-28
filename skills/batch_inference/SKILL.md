---
name: batch_inference
description: Generate predictions for a JSONL benchmark or validation set through an OpenAI-compatible local endpoint.
---

Allowed tools: `bash`, `cat`, `grep`.

Procedure:
1. Use `cat` to inspect input JSONL schema.
2. Write a small run-local Python script with `bash`; do not modify core repo source.
3. Call the endpoint with explicit model, temperature, top_p, max_tokens, and timeout.
4. Write predictions JSONL with stable `id` and `response` fields.
5. Log request failures per case; do not silently drop cases.
6. Verify prediction count equals case count before scoring.

For math exact-answer tasks, prefer a final answer prompt that requests a single integer or boxed integer.
