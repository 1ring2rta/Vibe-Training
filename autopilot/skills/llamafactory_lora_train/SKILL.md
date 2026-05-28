---
name: llamafactory_lora_train
description: Launch a LLaMA-Factory LoRA/SFT training run through explicit bash, not through wrapper workflows.
---

Allowed tools: `bash`, `cat`, `grep`.

Procedure:
1. Validate clean training data and decontamination report.
2. Write train YAML under the run workspace.
3. Inspect YAML with `cat`; ensure types are correct, especially booleans like `overwrite_output_dir`.
4. For two GPUs, launch with explicit `CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 NPROC_PER_NODE=2` when using DDP.
5. Use explicit timeout, log file, output dir, and save_steps.
6. Monitor logs and checkpoint directory with `bash`/`grep`.
7. Never use target benchmark data as training data.
