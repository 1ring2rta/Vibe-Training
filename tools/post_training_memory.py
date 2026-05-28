from __future__ import annotations

import time
from pathlib import Path
from typing import Any


DEFAULT_TEMPLATE = """# PostTrainingAgent.md

This file is the agent's long-term post-training memory.  It should record stable lessons from real runs: resource decisions, environment choices, data sources that worked or failed, verifier reliability, training failures, evaluation regressions, and repository improvements.

## Standing rules

- Prefer target-driven loops over fixed workflows.
- Treat GPU/service allocation as a per-phase decision based on current resources.
- Activate environments only when the selected task needs them.
- Ask the human when resource ownership, experiment intent, or irreversible operations are ambiguous.

## Experience log
"""


def ensure_post_training_memory(project_root: str | Path) -> Path:
    path = Path(project_root) / "PostTrainingAgent.md"
    if not path.exists():
        path.write_text(DEFAULT_TEMPLATE.rstrip() + "\n", encoding="utf-8")
    return path


def append_post_training_experience(project_root: str | Path, *, title: str, summary: str, payload: dict[str, Any] | None = None) -> Path:
    path = ensure_post_training_memory(project_root)
    payload = payload or {}
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"\n### {timestamp} — {title}", "", summary.strip() or "No summary provided."]
    important_keys = ["goal", "phase", "score", "target_met", "environment", "cuda_visible_devices", "ask_human"]
    details = []
    for key in important_keys:
        if key in payload and payload[key] not in (None, "", [], {}):
            details.append(f"- {key}: {payload[key]}")
    if details:
        lines.extend(["", "Details:", *details])
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return path


class PostTrainingAgentMemory:
    """Small wrapper used by the goal loop to maintain PostTrainingAgent.md."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.path = self.project_root / "PostTrainingAgent.md"

    def ensure(self) -> Path:
        return ensure_post_training_memory(self.project_root)

    def append(self, title: str, summary: str, payload: dict[str, Any] | None = None) -> Path:
        return append_post_training_experience(self.project_root, title=title, summary=summary, payload=payload)
