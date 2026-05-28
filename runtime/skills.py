from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from autopilot.models import to_jsonable


@dataclass
class SkillInfo:
    name: str
    description: str
    path: str

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def _parse_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    block = text[3:end].strip()
    data: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip().strip('"')
    return data


class SkillLibrary:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.skills_dir = self.root / ".autopilot" / "skills"
        self.drafts_dir = self.root / ".autopilot" / "skills_drafts"

    def materialize_builtin_skills(self) -> None:
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        try:
            package_root = resources.files("autopilot") / "skills"
            for child in package_root.iterdir():
                if not child.is_dir():
                    continue
                src = child / "SKILL.md"
                if not src.is_file():
                    continue
                dst_dir = self.skills_dir / child.name
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / "SKILL.md"
                if not dst.exists():
                    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            return

    def list(self) -> list[SkillInfo]:
        out: list[SkillInfo] = []
        if not self.skills_dir.exists():
            return out
        for skill_md in sorted(self.skills_dir.glob("*/SKILL.md")):
            text = skill_md.read_text(encoding="utf-8", errors="replace")
            meta = _parse_frontmatter(text)
            name = meta.get("name") or skill_md.parent.name
            desc = meta.get("description") or self._first_sentence(text)
            out.append(SkillInfo(name=name, description=desc[:500], path=str(skill_md.relative_to(self.root))))
        return out

    def prompt_index(self, *, max_chars: int = 8000) -> list[dict[str, Any]]:
        items = [s.to_dict() for s in self.list()]
        text = ""
        selected: list[dict[str, Any]] = []
        for item in items:
            row = f"{item['name']}: {item['description']} ({item['path']})\n"
            if len(text) + len(row) > max_chars:
                break
            text += row
            selected.append(item)
        return selected

    def write_draft(self, name: str, text: str) -> Path:
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "skill_update"
        path = self.drafts_dir / f"{safe}.md"
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        return path

    @staticmethod
    def _first_sentence(text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("---")]
        return lines[0] if lines else "No description."
