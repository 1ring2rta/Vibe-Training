---
name: artifact_policy_debug
description: Investigate artifact policy failures such as benchmark leakage or invalid training manifests.
---

Allowed tools: `bash`, `cat`, `grep`.

Procedure:
1. Use `grep` for the violation term and path from the policy report.
2. Determine whether the artifact is eval-only or train-like.
3. If eval-only, move it under `.autopilot/eval_programs/` and label it there.
4. If train-like, remove or regenerate the artifact without target benchmark content.
5. Re-run the policy scan before continuing.
