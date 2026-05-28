from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ContextEvent:
    event_id: str
    timestamp: float
    kind: str
    title: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    importance: int = 1

    @property
    def estimated_chars(self) -> int:
        return len(self.title) + len(self.summary) + len(json.dumps(self.payload, ensure_ascii=False, default=str))


@dataclass
class ContextState:
    session_id: str
    created_at: float
    updated_at: float
    project_memory: str = ""
    auto_memory: str = ""
    rolling_summary: str = ""
    compression_count: int = 0
    events: list[ContextEvent] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def estimated_chars(self) -> int:
        return (
            len(self.project_memory)
            + len(self.auto_memory)
            + len(self.rolling_summary)
            + sum(event.estimated_chars for event in self.events)
        )


class ContextManager:
    """Progressive context manager inspired by coding-agent workflows.

    It keeps three layers:
    1. project memory: human-written persistent instructions;
    2. auto memory: learned notes and stable preferences;
    3. session state: rolling summary + recent events.

    When the session grows beyond ``max_chars``, older events are compressed into
    the rolling summary while recent events stay verbatim. This gives later agent
    steps a compact but still useful context packet.
    """

    def __init__(
        self,
        state_path: str | Path,
        *,
        project_root: str | Path | None = None,
        max_chars: int = 24000,
        keep_recent_events: int = 12,
    ) -> None:
        self.state_path = Path(state_path)
        self.project_root = Path(project_root or Path.cwd())
        self.max_chars = max_chars
        self.keep_recent_events = keep_recent_events
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_or_create()

    def _load_or_create(self) -> ContextState:
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            events = [ContextEvent(**event) for event in data.get("events", [])]
            return ContextState(
                session_id=data.get("session_id") or str(uuid.uuid4()),
                created_at=float(data.get("created_at") or time.time()),
                updated_at=float(data.get("updated_at") or time.time()),
                project_memory=data.get("project_memory") or self.read_project_memory(),
                auto_memory=data.get("auto_memory") or self.read_auto_memory(),
                rolling_summary=data.get("rolling_summary") or "",
                compression_count=int(data.get("compression_count") or 0),
                events=events,
                artifacts=data.get("artifacts") or [],
            )
        return ContextState(
            session_id=str(uuid.uuid4()),
            created_at=time.time(),
            updated_at=time.time(),
            project_memory=self.read_project_memory(),
            auto_memory=self.read_auto_memory(),
        )

    def save(self) -> Path:
        self.state.updated_at = time.time()
        data = asdict(self.state)
        self.state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return self.state_path

    def _read_memory_files(self, paths: list[Path], *, max_chars: int = 25000) -> str:
        blocks: list[str] = []
        seen: set[str] = set()
        for path in paths:
            try:
                resolved = str(path.expanduser().resolve())
            except Exception:
                resolved = str(path)
            if resolved in seen or not path.exists() or not path.is_file():
                continue
            seen.add(resolved)
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            if text.strip():
                blocks.append(f"## {path}\n" + text.strip())
        return "\n\n".join(blocks)[:max_chars]

    def read_project_memory(self) -> str:
        # Claude Code conventions are supported alongside Autopilot memory.
        # This lets the same repo-level CLAUDE.md guide both Claude and KIMI-run
        # Autopilot loops without duplicating instructions.
        return self._read_memory_files([
            self.project_root / "AUTOPILOT.md",
            self.project_root / "CLAUDE.md",
            self.project_root / ".claude" / "CLAUDE.md",
            self.project_root / ".autopilot" / "project.md",
        ])

    def read_auto_memory(self) -> str:
        return self._read_memory_files([
            self.project_root / "PostTrainingAgent.md",
            self.project_root / ".autopilot" / "memory.md",
            self.project_root / ".claude" / "memory.md",
            self.project_root / ".claude" / "autopilot_memory.md",
            Path.home() / ".claude" / "CLAUDE.md",
        ])

    def write_project_memory(self, text: str) -> Path:
        path = self.project_root / "AUTOPILOT.md"
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        self.state.project_memory = text.rstrip() + "\n"
        self.save()
        return path

    def append_auto_memory(self, note: str) -> Path:
        mem_dir = self.project_root / ".autopilot"
        mem_dir.mkdir(parents=True, exist_ok=True)
        path = mem_dir / "memory.md"
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if note.strip() and note.strip() not in old:
            new = (old.rstrip() + "\n- " + note.strip() + "\n").lstrip()
            path.write_text(new, encoding="utf-8")
            claude_dir = self.project_root / ".claude"
            claude_dir.mkdir(parents=True, exist_ok=True)
            claude_path = claude_dir / "autopilot_memory.md"
            claude_old = claude_path.read_text(encoding="utf-8") if claude_path.exists() else ""
            if note.strip() not in claude_old:
                claude_path.write_text((claude_old.rstrip() + "\n- " + note.strip() + "\n").lstrip(), encoding="utf-8")
            self.state.auto_memory = self.read_auto_memory()[:25000]
            self.save()
        return path

    def append_post_training_memory(
        self,
        note: str,
        *,
        title: str = "Autopilot Experience",
        path: str | Path = "PostTrainingAgent.md",
    ) -> Path:
        out = (self.project_root / path) if not Path(path).is_absolute() else Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        old = out.read_text(encoding="utf-8") if out.exists() else "# PostTrainingAgent Memory\n\n"
        clean = note.strip()
        if clean and clean not in old:
            import time as _time
            block = f"## {title} — {_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n{clean}\n"
            out.write_text(old.rstrip() + "\n\n" + block, encoding="utf-8")
            self.state.auto_memory = self.read_auto_memory()[:25000]
            self.save()
        return out

    def add_event(
        self,
        kind: str,
        title: str,
        summary: str,
        payload: dict[str, Any] | None = None,
        *,
        importance: int = 1,
        auto_compact: bool = True,
    ) -> ContextEvent:
        event = ContextEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            kind=kind,
            title=title,
            summary=summary,
            payload=payload or {},
            importance=importance,
        )
        self.state.events.append(event)
        if auto_compact and self.state.estimated_chars > self.max_chars:
            self.compact()
        self.save()
        return event

    def add_artifact(self, path: str | Path, kind: str, description: str) -> None:
        self.state.artifacts.append({"path": str(path), "kind": kind, "description": description, "timestamp": time.time()})
        self.save()

    def compact(self, *, keep_recent_events: int | None = None) -> None:
        keep = keep_recent_events if keep_recent_events is not None else self.keep_recent_events
        if len(self.state.events) <= keep:
            return
        old_events = self.state.events[:-keep]
        self.state.events = self.state.events[-keep:]
        block = self._summarize_events(old_events)
        self.state.compression_count += 1
        header = f"\n## Compression {self.state.compression_count} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        self.state.rolling_summary = (self.state.rolling_summary.rstrip() + header + block).strip() + "\n"
        # Keep rolling summary bounded by dropping oldest text first.
        if len(self.state.rolling_summary) > self.max_chars:
            self.state.rolling_summary = self.state.rolling_summary[-self.max_chars :]
        self.save()

    def _summarize_events(self, events: list[ContextEvent]) -> str:
        lines: list[str] = []
        for event in events:
            important = " !" if event.importance >= 3 else ""
            payload_hint = ""
            if event.payload:
                keys = ",".join(list(event.payload.keys())[:5])
                payload_hint = f" [{keys}]"
            lines.append(f"- ({event.kind}{important}) {event.title}: {event.summary}{payload_hint}")
        return "\n".join(lines) + "\n"

    def render_context(self, max_chars: int | None = None) -> str:
        max_chars = max_chars or self.max_chars
        sections: list[str] = []
        if self.state.project_memory.strip():
            sections.append("# Project Memory\n" + self.state.project_memory.strip())
        if self.state.auto_memory.strip():
            sections.append("# Auto Memory\n" + self.state.auto_memory.strip())
        if self.state.rolling_summary.strip():
            sections.append("# Rolling Summary\n" + self.state.rolling_summary.strip())
        if self.state.events:
            lines = ["# Recent Events"]
            for event in self.state.events:
                lines.append(f"- {event.kind} | {event.title}: {event.summary}")
            sections.append("\n".join(lines))
        if self.state.artifacts:
            lines = ["# Artifacts"]
            for artifact in self.state.artifacts[-20:]:
                lines.append(f"- {artifact.get('kind')}: {artifact.get('path')} — {artifact.get('description')}")
            sections.append("\n".join(lines))
        text = "\n\n".join(sections).strip()
        if len(text) > max_chars:
            return text[-max_chars:]
        return text
