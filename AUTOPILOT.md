Development rules:

- Keep deterministic rules testable before adding LLM judgement.
- Avoid hiding irreversible actions inside agents; training commands should be explicit in manifests and context logs.
- Always keep 2x H800 GPUs fully utilized.