---
name: clean_data_search
description: Search for training data candidates without leaking the target benchmark into training.
---

Allowed tools: `web_search`, `browser`, `bash`, `cat`, `grep`.

Rules:
- The agent writes every search query itself.
- Do not use the target benchmark name as a positive training-data query term.
- For AIME24, do not search `AIME 2024 problems with solutions` as training data.
- Prefer broad sources: olympiad-style math, pre-target-year contests, synthetic reasoning, proof datasets, and non-overlapping integer-answer data.

Procedure:
1. Draft 3-5 broad queries with exclusion terms for the target benchmark.
2. Use `web_search`.
3. Use `browser` to inspect dataset cards, source, license, and schema.
4. Use `bash` only for explicit downloads or local sampling.
5. Write a run-local candidate manifest with provenance and status `train_candidate`.
6. Run decontamination before any candidate can become `train_accepted`.

Ask human only if the choice is a real tradeoff, such as using a high-value but license-unclear dataset.
