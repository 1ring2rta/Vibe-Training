---
name: evaluate_checkpoint
description: Evaluate a trained checkpoint or adapter against trusted eval-only cases and compare to baseline.
---

Allowed tools: `bash`, `cat`, `grep`.

Procedure:
1. Locate checkpoint/adapter and verify it is from a clean run.
2. Serve the model or merged adapter with explicit vLLM command.
3. Generate predictions with the `batch_inference` procedure.
4. Score with the trusted evaluator.
5. Write comparison JSON showing baseline score, new score, case count, and target_met.
6. If target not met, diagnose wrong cases; do not stop.
