---
name: serve_vllm
description: Start and verify a local vLLM server with explicit bash commands, PID/log tracking, and readiness checks.
---

Allowed tools: `bash`, `cat`, `grep`.

Procedure:
1. Check GPU status with `nvidia-smi` via `bash`.
2. Start vLLM with one explicit command using `bash` and `detached=true`.
3. Include `CUDA_VISIBLE_DEVICES`, `--tensor-parallel-size`, `--port`, `--max-model-len`, and a log file.
4. Verify readiness by checking the log and opening the local port or `/v1/models`.
5. If startup is slow but logs show model loading, wait and re-check; do not restart blindly.
6. Record PID/log paths in the run workspace.

Failure recovery:
- If port is busy, inspect existing processes and pick another port or kill only a tracked matching service.
- If model loads on only one GPU unintentionally, restart with explicit `CUDA_VISIBLE_DEVICES=0,1` and `--tensor-parallel-size 2`.
