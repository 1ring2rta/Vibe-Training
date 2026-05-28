---
name: convert_to_sft
description: Convert accepted clean datasets into an explicit SFT JSONL and LLaMA-Factory dataset_info entry.
---

Allowed tools: `bash`, `cat`, `grep`.

Procedure:
1. Read the candidate manifest and decontamination report.
2. Refuse conversion if candidate status is not clean/accepted.
3. Inspect schema samples with `bash` or `cat`.
4. Write a small run-local converter script.
5. Convert a preview first and validate fields.
6. Convert full data into `round_*/prepared/data/`.
7. Write `dataset_info.json` with source provenance.
8. Grep the output for target benchmark leakage terms before training.
