from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autopilot.config import load_settings, validate_settings
from autopilot.runtime.agent_turn import AgentTurnRunner
from autopilot.runtime.clients import LLMClientRegistry
from autopilot.runtime.state import RunStateStore
from autopilot.runtime.tools import build_default_model_tool_registry
from autopilot.runtime.processes import ProcessRegistry
from autopilot.eval.programs import EvalProgramWorkspace


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Agent runtime utilities: client registry, tool-enabled turns, trajectory, and resume state.")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="Show run_state/task_graph resume status.")
    s.add_argument("--root", default="runs/goal_loop")

    s = sub.add_parser("resume-plan", help="Mark stale RUNNING tasks interrupted and print runnable/waiting/failed tasks.")
    s.add_argument("--root", default="runs/goal_loop")

    s = sub.add_parser("clients", help="Show configured OpenAI-compatible clients and role bindings.")

    s = sub.add_parser("trajectory", help="Show frontier trajectory paths and basic counts.")
    s.add_argument("--root", default=None, help="Trajectory root. Default from config or .autopilot/frontier_trajectory.")

    s = sub.add_parser("processes", help="List Autopilot-tracked processes for a run.")
    s.add_argument("--root", default="runs/runtime_debug")
    s.add_argument("--active-only", action="store_true")
    s.add_argument("--no-exited", action="store_true")

    s = sub.add_parser("kill-process", help="Kill a tracked process by process_id/pid/name/kind/pattern.")
    s.add_argument("--root", default="runs/runtime_debug")
    s.add_argument("--process-id")
    s.add_argument("--pid", type=int)
    s.add_argument("--name")
    s.add_argument("--kind")
    s.add_argument("--pattern")
    s.add_argument("--signal", default="TERM")
    s.add_argument("--no-process-group", action="store_true")
    s.add_argument("--force-after-seconds", type=float, default=5.0)
    s.add_argument("--allow-untracked", action="store_true")

    s = sub.add_parser("eval-programs", help="List run-local evaluator programs.")
    s.add_argument("--root", default="runs/runtime_debug")

    s = sub.add_parser("turn", help="Run one tool-enabled model turn for debugging.")
    s.add_argument("--root", default="runs/runtime_debug")
    s.add_argument("--role", default="director")
    s.add_argument("--objective", default="Debug the generic agent runtime.")
    s.add_argument("--prompt", required=True)
    s.add_argument("--allow-bash", action="store_true")

    return p


def _count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(config_file=args.config)
    for warning in validate_settings(settings):
        print(f"[warn] {warning}")

    if args.cmd == "status":
        store = RunStateStore(args.root)
        _print_json({"state": store.state(), "task_graph": store.task_graph(), "artifacts": store._read_json(store.artifacts_path, {})})
        return 0

    if args.cmd == "resume-plan":
        store = RunStateStore(args.root)
        _print_json(store.resume_plan())
        return 0

    if args.cmd == "clients":
        reg = LLMClientRegistry.from_settings(settings)
        _print_json(reg.to_dict())
        return 0

    if args.cmd == "trajectory":
        reg = LLMClientRegistry.from_settings(settings, trajectory_root=args.root)
        rec = reg.trajectory_recorder
        if rec is None:
            _print_json({"enabled": False})
            return 0
        _print_json({"enabled": True, "paths": rec.paths(), "counts": {name: _count_lines(Path(path)) for name, path in rec.paths().items() if path.endswith(".jsonl")}, "audit": rec.audit()})
        return 0

    if args.cmd == "processes":
        reg = ProcessRegistry(args.root)
        _print_json({"registry_path": str(reg.registry_path), "processes": reg.list(active_only=bool(args.active_only), include_exited=not bool(args.no_exited))})
        return 0

    if args.cmd == "kill-process":
        reg = ProcessRegistry(args.root)
        _print_json(reg.kill(process_id=args.process_id, pid=args.pid, name=args.name, kind=args.kind, pattern=args.pattern, sig=args.signal, kill_process_group=not bool(args.no_process_group), force_after_seconds=args.force_after_seconds, allow_untracked=bool(args.allow_untracked), reason="autopilot-runtime kill-process"))
        return 0

    if args.cmd == "eval-programs":
        ws = EvalProgramWorkspace(args.root)
        _print_json(ws.to_dict())
        return 0

    if args.cmd == "turn":
        root = Path(args.root)
        state = RunStateStore(root)
        clients = LLMClientRegistry.from_settings(settings, trajectory_root=root / ".autopilot" / "frontier_trajectory")
        tools = build_default_model_tool_registry(workspace=root, run_state=state, allow_bash=bool(args.allow_bash), settings=settings)
        runner = AgentTurnRunner(clients, tools, state)
        result = runner.run(role=args.role, objective=args.objective, purpose="runtime_debug_turn", messages=[{"role": "user", "content": args.prompt}])
        _print_json(result.__dict__)
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
