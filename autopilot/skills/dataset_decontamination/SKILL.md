---
name: dataset_decontamination
description: Check candidate training data against eval-only benchmark cases and block benchmark leakage.
---

Allowed tools: `bash`, `cat`, `grep`.

Procedure:
1. Locate eval-only cases and candidate training data.
2. Compute normalized hashes for prompts/questions.
3. Check source id, path, and metadata for target benchmark terms.
4. Check exact normalized overlap and sample-level fuzzy overlap.
5. Write `decontamination_report.json` with status `clean`, `blocked`, or `needs_human`.
6. Only `clean` candidates may be used in train manifests.

For AIME24 targets, block source/path terms matching `aime`, `aime24`, `aime-24`, `aime_24`, `aime-2024`, `AIME_2024`, or `1983-2024` unless the artifact is explicitly eval-only.
