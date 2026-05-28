from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from autopilot.llm.kimi import KimiClient
from autopilot.tools.bash import BashRunner

_TEXT_EXTS = {".py", ".toml", ".yaml", ".yml", ".md", ".txt", ".json", ".sh"}
_SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "runs", "reports", "prepared", "saves", "dist", "build"}


@dataclass
class RepoFileSnippet:
    path: str
    chars: int
    excerpt: str


@dataclass
class RepoSnapshot:
    root: str
    files: list[str]
    snippets: list[RepoFileSnippet]
    test_files: list[str]
    pyproject_present: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_rel(path: Path, root: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except Exception:
        return None
    parts = rel.parts
    if any(part in _SKIP_DIRS for part in parts):
        return None
    return rel.as_posix()


def iter_repo_files(root: str | Path, max_files: int = 500) -> list[str]:
    root = Path(root).resolve()
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = _safe_rel(path, root)
        if rel is None:
            continue
        files.append(rel)
        if len(files) >= max_files:
            break
    return files


def inspect_repo(root: str | Path, *, max_files: int = 500, max_snippets: int = 18, max_chars_per_file: int = 4000) -> RepoSnapshot:
    root = Path(root).resolve()
    files = iter_repo_files(root, max_files=max_files)
    priority_prefixes = (
        "autopilot/goal/",
        "autopilot/cli/goal.py",
        "autopilot/cli/run.py",
        "autopilot/tools/bash.py",
        "autopilot/config.py",
        "tests/",
        "README.md",
        "pyproject.toml",
    )
    selected: list[str] = []
    for rel in files:
        if rel.endswith(tuple(_TEXT_EXTS)) and (rel.startswith(priority_prefixes) or rel in {"README.md", "pyproject.toml", "autopilot.example.yaml"}):
            selected.append(rel)
        if len(selected) >= max_snippets:
            break
    snippets: list[RepoFileSnippet] = []
    for rel in selected:
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        snippets.append(RepoFileSnippet(path=rel, chars=len(text), excerpt=text[:max_chars_per_file]))
    return RepoSnapshot(
        root=str(root),
        files=files,
        snippets=snippets,
        test_files=[f for f in files if f.startswith("tests/") and f.endswith(".py")],
        pyproject_present=(root / "pyproject.toml").exists(),
    )


def plan_repo_improvements(
    kimi: KimiClient,
    *,
    goal: str,
    snapshot: RepoSnapshot,
    compute_resources: Mapping[str, Any] | None = None,
    loop_state: Mapping[str, Any] | None = None,
    max_tokens: int = 4000,
) -> dict[str, Any]:
    payload = {
        "goal": goal,
        "compute_resources": compute_resources or {},
        "loop_state": loop_state or {},
        "repo_snapshot": snapshot.to_dict(),
    }
    system = "你是这个 LLM Training Autopilot 仓库的 coding agent。你可以阅读仓库、提出修改、运行测试。只输出 JSON。"
    user = f"""
用户希望模型在训练循环中也能改进 Autopilot 仓库本身。请根据目标、计算资源、当前 loop 状态和仓库片段，提出下一步仓库改进。

输出严格 JSON：
{{
  "summary": "一句话说明这轮应该改什么",
  "patches": [
    {{"path": "相对仓库路径", "content": "完整文件内容。只在确实需要直接覆盖文件时提供"}}
  ],
  "commands": ["可以运行的测试或诊断命令，例如 python -m pytest -q"],
  "risks": ["可能的风险或需要观察的现象"],
  "notes": "其它说明"
}}

约束：
1. patches 必须只写 Autopilot 仓库内的文本文件；不要写 secrets；不要修改用户数据或模型权重目录。
2. 优先改 agent loop、bash runner、resource/vLLM management、tests。
3. 如果信息不足，可以只给 commands 和 summary，不要编造大改动。

信息：
{json.dumps(payload, ensure_ascii=False)[:50000]}
""".strip()
    data = kimi._chat_json(system, user, purpose="repo_agent_plan", max_tokens=max_tokens, temperature=0.3)
    return data if isinstance(data, dict) else {"raw": str(data)[:4000]}


def _validate_patch_path(root: Path, rel: str) -> Path:
    if not rel or rel.startswith(("/", "~")):
        raise ValueError(f"Patch path must be relative: {rel!r}")
    path = (root / rel).resolve()
    path.relative_to(root.resolve())
    if any(part in _SKIP_DIRS for part in path.relative_to(root.resolve()).parts):
        raise ValueError(f"Patch path is in a skipped directory: {rel!r}")
    if path.suffix and path.suffix not in _TEXT_EXTS:
        raise ValueError(f"Patch path must be a text/config file: {rel!r}")
    return path


def apply_repo_patches(root: str | Path, patches: Iterable[Mapping[str, Any]], *, backup_dir: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    backup_root = Path(backup_dir).resolve() if backup_dir else root / ".autopilot" / "repo_backups"
    results: list[dict[str, Any]] = []
    for patch in patches:
        rel = str(patch.get("path", ""))
        content = patch.get("content")
        if not isinstance(content, str):
            results.append({"path": rel, "ok": False, "error": "missing string content"})
            continue
        try:
            path = _validate_patch_path(root, rel)
            if path.exists():
                backup_path = backup_root / rel
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                backup_path.write_text(path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            else:
                backup_path = None
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            results.append({"path": rel, "ok": True, "backup": str(backup_path) if backup_path else None, "bytes": len(content.encode("utf-8"))})
        except Exception as exc:
            results.append({"path": rel, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return results


def run_repo_commands(root: str | Path, commands: Iterable[str], *, timeout: float = 600.0, limit: int = 4) -> list[dict[str, Any]]:
    runner = BashRunner(cwd=root, timeout=timeout)
    results: list[dict[str, Any]] = []
    for command in list(commands)[:limit]:
        result = runner.run(str(command), shell=True, timeout=timeout)
        results.append({
            "command": str(command),
            "returncode": result.returncode,
            "ok": result.ok,
            "stdout_tail": result.stdout[-3000:],
            "stderr_tail": result.stderr[-3000:],
            "timed_out": result.timed_out,
        })
    return results
