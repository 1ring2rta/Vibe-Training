from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _print_task(task: dict[str, Any], indent: int = 0) -> None:
    prefix = "  " * indent
    status = task.get("status", "?")
    name = task.get("name", "<unnamed>")
    summary = task.get("summary", "")
    result_path = task.get("result_path", "")
    print(f"{prefix}- [{status}] {name}: {summary}")
    if result_path:
        rp = Path(result_path)
        if rp.exists():
            try:
                result = _load_json(rp)
                for child in result.get("child_tasks", []) or []:
                    _print_task(child, indent + 1)
            except Exception:
                pass


def command_tree(args: argparse.Namespace) -> int:
    root = Path(args.root)
    index_path = root / "loop_index.json" if root.is_dir() else root
    if not index_path.exists():
        print(f"[error] Missing loop index: {index_path}")
        return 2
    data = _load_json(index_path)
    print(f"Agent loop: {data.get('name')} — {data.get('objective')}")
    print(f"workspace: {data.get('workspace_dir')}")
    for task in data.get("child_tasks", []) or []:
        _print_task(task, 0)
    return 0


def command_show(args: argparse.Namespace) -> int:
    path = Path(args.result)
    if path.is_dir():
        path = path / "result.json"
    if not path.exists():
        print(f"[error] Missing result file: {path}")
        return 2
    data = _load_json(path)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    print(f"Task: {data.get('name')}")
    print(f"Status: {data.get('status')}")
    print(f"Objective: {data.get('objective')}")
    print(f"Summary: {data.get('summary')}")
    if data.get("error"):
        print(f"Error: {data.get('error')}")
    if data.get("artifacts"):
        print("Artifacts:")
        for artifact in data.get("artifacts", []):
            print(f"  - {artifact.get('kind')}: {artifact.get('path')} — {artifact.get('description', '')}")
    if data.get("child_tasks"):
        print("Child tasks:")
        for child in data.get("child_tasks", []):
            _print_task(child, 1)
    return 0


def _normalize_agent_root(path: Path) -> Path:
    if (path / ".autopilot" / "agent").exists():
        return path / ".autopilot" / "agent"
    return path


def _iter_task_dirs(root: Path) -> list[Path]:
    root = _normalize_agent_root(root)
    if not root.exists():
        return []
    return sorted({p.parent for p in root.rglob("task.json")}, key=lambda p: p.stat().st_mtime)


def _load_task_meta(task_dir: Path) -> dict[str, Any]:
    try:
        return _load_json(task_dir / "task.json")
    except Exception:
        return {"name": task_dir.name}


def command_status(args: argparse.Namespace) -> int:
    root = _normalize_agent_root(Path(args.root))
    task_dirs = _iter_task_dirs(root)
    if not task_dirs:
        print(f"[warn] No task.json files found under: {root}")
        return 1
    running = [d for d in task_dirs if not (d / "result.json").exists()]
    done = [d for d in task_dirs if (d / "result.json").exists()]
    failed = []
    for d in done:
        try:
            if _load_json(d / "result.json").get("status") == "failed":
                failed.append(d)
        except Exception:
            pass
    print(f"agent_root: {root}")
    print(f"tasks: total={len(task_dirs)} done={len(done)} running={len(running)} failed={len(failed)}")
    if running:
        print("\nCurrent/pending tasks without result.json:")
        for d in running[-10:]:
            meta = _load_task_meta(d)
            age = time.time() - float(meta.get("created_at") or d.stat().st_mtime)
            print(f"- {meta.get('name')} | age={age:.1f}s | {d}")
            print(f"  objective: {meta.get('objective') or ''}")
    print("\nLatest tasks:")
    for d in task_dirs[-int(args.last):]:
        meta = _load_task_meta(d)
        result_path = d / "result.json"
        if result_path.exists():
            try:
                result = _load_json(result_path)
                status = result.get("status", "done")
                summary = result.get("summary", "")
                duration = result.get("duration_seconds", "")
            except Exception:
                status, summary, duration = "done", "", ""
        else:
            status, summary, duration = "running", "", ""
        print(f"- [{status}] {meta.get('name')} duration={duration} :: {summary}")
        print(f"  {d}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect nested agent-loop task trees and results.")
    sub = parser.add_subparsers(dest="command", required=True)

    tree = sub.add_parser("tree", help="Print a loop tree from .autopilot/agent/loop_index.json.")
    tree.add_argument("--root", default=".autopilot/agent", help="Agent root directory or loop_index.json path.")
    tree.set_defaults(func=command_tree)

    show = sub.add_parser("show", help="Show one task result.json or task directory.")
    show.add_argument("result", help="Path to result.json or a task directory.")
    show.add_argument("--json", action="store_true", help="Print raw JSON.")
    show.set_defaults(func=command_show)

    status = sub.add_parser("status", help="Show live progress by scanning task directories, even before loop_index.json is written.")
    status.add_argument("--root", default=".autopilot/agent", help="Agent root directory, run output directory, or .autopilot/agent path.")
    status.add_argument("--last", type=int, default=12, help="Show the last N tasks.")
    status.set_defaults(func=command_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
