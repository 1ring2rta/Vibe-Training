from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

from autopilot.models import to_jsonable


@dataclass
class MemoryIndex:
    always_files: list[str]
    draft_files: list[str]
    snippets: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


class AgentMemory:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.memory_dir = self.root / ".autopilot" / "memory"
        self.drafts_dir = self.memory_dir / "drafts"

    def materialize_builtin_memory(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        try:
            package_root = resources.files("autopilot") / "memory"
            for src in package_root.iterdir():
                if not src.is_file() or src.suffix.lower() != ".md":
                    continue
                dst = self.memory_dir / src.name
                if not dst.exists():
                    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            return

    def index(self, *, max_chars_per_file: int = 2500) -> MemoryIndex:
        always: list[str] = []
        snippets: list[dict[str, str]] = []
        for path in sorted(self.memory_dir.glob("*.md")):
            always.append(str(path.relative_to(self.root)))
            text = path.read_text(encoding="utf-8", errors="replace")[:max_chars_per_file]
            snippets.append({"path": str(path.relative_to(self.root)), "content": text})
        drafts = [str(p.relative_to(self.root)) for p in sorted(self.drafts_dir.glob("*.md"))]
        return MemoryIndex(always_files=always, draft_files=drafts, snippets=snippets)

    def write_run_maintenance_drafts(self, *, report: dict[str, Any], event_log_path: Path | None = None) -> dict[str, str]:
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        status = report.get("status") or "UNKNOWN"
        actions = report.get("actions") or []
        failures = []
        for row in actions:
            outcome = row.get("outcome") or {}
            action = row.get("action") or {}
            if not outcome.get("ok", True):
                failures.append(f"- {action.get('action_type')} / {action.get('tool_name')}: {outcome.get('error') or outcome.get('result')}")
        summary = [
            "# Draft run memory update",
            "",
            f"- Status: {status}",
            f"- Actions: {len(actions)}",
            f"- Event log: {event_log_path or ''}",
            "",
            "## Failures or repair candidates",
            *(failures[:20] or ["- None recorded."]),
            "",
            "## Promotion rule",
            "Promote stable bug fixes and environment facts automatically; ask human before changing safety, contamination, or benchmark policy.",
        ]
        run_path = self.drafts_dir / f"run_summary_{now}.md"
        run_path.write_text("\n".join(summary).rstrip() + "\n", encoding="utf-8")

        skill_path = self.root / ".autopilot" / "skills_drafts" / f"skill_update_{now}.md"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(
            "# Draft skill update\n\n"
            "Review whether this run exposed a repeatable procedure that should become a skill change.\n\n"
            "Do not promote contaminated-data behavior, benchmark leakage, or policy relaxation.\n",
            encoding="utf-8",
        )
        return {"memory_draft": str(run_path.relative_to(self.root)), "skill_draft": str(skill_path.relative_to(self.root))}
