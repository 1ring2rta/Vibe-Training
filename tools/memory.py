from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable


CLAUDE_MEMORY_CANDIDATES = [
    "CLAUDE.md",
    ".claude/CLAUDE.md",
    ".claude/memory.md",
    ".claude/autopilot_memory.md",
    ".autopilot/memory.md",
    Path.home() / ".claude" / "CLAUDE.md",
    Path.home() / ".claude" / "memory.md",
]


def _read_files(root: Path, rels: Iterable[str], *, max_chars: int = 25000) -> str:
    chunks: list[str] = []
    seen: set[Path] = set()
    for rel in rels:
        path = (root / rel).expanduser()
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen or not path.exists() or not path.is_file():
            continue
        seen.add(resolved)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if text.strip():
            chunks.append(f"## {rel}\n{text.strip()}")
    return "\n\n".join(chunks)[:max_chars]


def read_claude_memory(root: str | Path = ".", *, max_chars: int = 25000) -> str:
    return _read_files(Path(root), CLAUDE_MEMORY_CANDIDATES, max_chars=max_chars)


def ensure_post_training_agent(root: str | Path = ".", filename: str = "PostTrainingAgent.md") -> Path:
    path = Path(root) / filename
    if not path.exists():
        path.write_text(
            "# PostTrainingAgent Memory\n\n"
            "This file is maintained by Autopilot/KIMI during post-training loops.\n"
            "It should accumulate stable lessons about resources, environments, training failures, evals, and recovery decisions.\n",
            encoding="utf-8",
        )
    return path


def append_post_training_agent_memory(root: str | Path, notes: Iterable[str], *, section: str = "## Autopilot Experience", filename: str = "PostTrainingAgent.md") -> Path:
    path = ensure_post_training_agent(root, filename=filename)
    clean = [str(note).strip() for note in notes if str(note).strip()]
    if not clean:
        return path
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else "# PostTrainingAgent Memory\n\n"
    body = "\n".join(f"- {note}" if not note.lstrip().startswith(('-', '#')) else note for note in clean)
    block = f"\n\n{section}\n\n_Time: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n\n{body}\n"
    if body not in old:
        path.write_text(old.rstrip() + block, encoding="utf-8")
    return path


def append_claude_memory(root: str | Path, notes: Iterable[str]) -> Path:
    root = Path(root)
    claude_dir = root / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    path = claude_dir / "autopilot_memory.md"
    clean = [str(note).strip() for note in notes if str(note).strip()]
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else "# Autopilot Memory\n\n"
    body = "\n".join(f"- {note}" if not note.lstrip().startswith(('-', '#')) else note for note in clean)
    if body and body not in old:
        path.write_text(old.rstrip() + "\n\n" + body + "\n", encoding="utf-8")
    return path
