from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopilot.models import to_jsonable
from autopilot.tools.bash import BashRunner


@dataclass
class RepoSnapshot:
    path: str
    git_root: str | None
    git_status_short: str
    git_diff_stat: str
    branch: str | None
    files: list[str] = field(default_factory=list)
    important_files: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self.__dict__)


def _safe_read(path: Path, max_chars: int = 12000) -> str:
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
        return text[:max_chars]
    except Exception as exc:
        return f"<read_error {type(exc).__name__}: {exc}>"


def _list_repo_files(repo: Path, max_files: int = 300) -> list[str]:
    preferred_dirs = ['autopilot', 'tests', 'scripts', 'configs']
    files: list[str] = []
    for dirname in preferred_dirs:
        base = repo / dirname
        if not base.exists():
            continue
        for path in sorted(base.rglob('*')):
            if path.is_file() and not any(part in {'.git', '__pycache__', '.pytest_cache'} for part in path.parts):
                files.append(str(path.relative_to(repo)))
                if len(files) >= max_files:
                    return files
    for name in ['pyproject.toml', 'README.md', 'autopilot.example.yaml', 'AUTOPILOT.md', 'CLAUDE.md', 'PostTrainingAgent.md']:
        if (repo / name).exists() and name not in files:
            files.append(name)
    return files[:max_files]


def collect_repo_snapshot(repo_path: str | Path | None = None, max_files: int = 300) -> RepoSnapshot:
    repo = Path(repo_path or os.getcwd()).expanduser().resolve()
    runner = BashRunner(cwd=repo, timeout=30)
    errors: list[str] = []

    root_res = runner.run(['git', 'rev-parse', '--show-toplevel'])
    git_root: str | None = None
    if root_res.ok:
        git_root = root_res.stdout.strip() or None
        repo = Path(git_root).resolve() if git_root else repo
        runner = BashRunner(cwd=repo, timeout=30)
    else:
        errors.append((root_res.stderr or root_res.stdout or 'not a git repo').strip()[:1000])

    status = runner.run(['git', 'status', '--short']) if git_root else None
    diff = runner.run(['git', 'diff', '--stat']) if git_root else None
    branch = runner.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD']) if git_root else None

    files = _list_repo_files(repo, max_files=max_files)
    important: dict[str, str] = {}
    for rel in ['pyproject.toml', 'README.md', 'autopilot.example.yaml', 'AUTOPILOT.md', 'CLAUDE.md', 'PostTrainingAgent.md', 'autopilot/goal/loop.py', 'autopilot/cli/goal.py', 'autopilot/cli/run.py', 'autopilot/config.py', 'autopilot/tools/bash.py']:
        path = repo / rel
        if path.exists() and path.is_file():
            important[rel] = _safe_read(path, max_chars=8000)

    return RepoSnapshot(
        path=str(repo),
        git_root=git_root,
        git_status_short=(status.stdout if status and status.ok else ''),
        git_diff_stat=(diff.stdout if diff and diff.ok else ''),
        branch=(branch.stdout.strip() if branch and branch.ok else None),
        files=files,
        important_files=important,
        errors=errors,
    )


def write_repo_snapshot(path: str | Path, snapshot: RepoSnapshot) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return out
