---
name: trusted_eval
description: Establish a trusted eval-only benchmark and score predictions without making the benchmark available for training.
---

Use when the agent needs a baseline or post-training score.

Allowed tools: `bash`, `cat`, `grep`.

Procedure:
1. Create or locate the benchmark cases under `.autopilot/eval_programs/<benchmark>/` only.
2. Treat benchmark cases as `eval_only`; never copy them under `round_*`, `prepared`, `train`, `sft`, or dataset directories.
3. Inspect evaluator code with `cat` before running it.
4. Generate predictions into the eval workspace.
5. Run a deterministic metric script and write `evaluation_result.json` with `eval_source`, `benchmark`, `metric_name`, `case_count`, `score`, and `target_met`.
6. Use `grep` to verify no target benchmark terms appear in train-like artifacts before training.

Forbidden shortcuts:
- Do not use target benchmark cases as training examples.
- Do not write a permissive evaluator that extracts gold answers from cases.
- Do not mark target met without a real `evaluation_result.json` from benchmark cases.
