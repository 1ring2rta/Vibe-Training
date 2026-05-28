# Known bug memories

- A previous wrapper-based training path used a default 600s timeout and killed long LLaMA-Factory runs; prefer explicit bash commands with explicit timeout/logging.
- vLLM startup for multi-GPU Qwen-style models can take several minutes. Check logs and port readiness instead of assuming a 30s timeout means failure.
- Dataset searches for a benchmark target must not use the benchmark name as a positive training-data query term.
