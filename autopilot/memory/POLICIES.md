# Safety and evaluation policies

- Target benchmark files are `eval_only` and must never flow into training data, training manifests, or train YAML.
- For AIME24 targets, training artifacts must not reference AIME 2024, AIME24, `aime-24`, `aime_24`, or `1983-2024` sources.
- Use `web_search`/`browser` for external network discovery. Bash network access is not the default path.
- Training must have an auditable data manifest and a clean decontamination report before launch.
- Stop only when a trusted benchmark evaluation meets the target; smoke/synthetic evals cannot stop the run.
- Memory and skill updates start as drafts. Promote low-risk bug facts automatically; policy changes require human choice.
