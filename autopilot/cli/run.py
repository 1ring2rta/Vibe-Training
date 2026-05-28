from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from autopilot.agent import AgentLoop
from autopilot.config import apply_config_defaults, load_settings
from autopilot.context import ContextManager
from autopilot.runtime.processes import ProcessRegistry
from autopilot.tools.bash import BashRunner
from autopilot.tools.long_task_monitor import LongTaskSupervisor
from autopilot.tools.resource_allocation import choose_environment_for_stage


def _parse_env_overrides(values: list[str] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"--env must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--env has empty key: {item}")
        env[key] = value
    return env




def _infer_stage_from_train_yaml(path: str | Path) -> str:
    p = Path(path)
    stem = p.stem.lower()
    if stem.startswith("train_"):
        return stem.replace("train_", "", 1) or "sft"
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("stage:"):
                return stripped.split(":", 1)[1].strip().strip("'\"").lower() or "sft"
    except Exception:
        pass
    return "sft"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bash command or LLaMA-Factory train YAML and capture output as an agent-loop task.")
    parser.add_argument("--config", default=None, help="YAML config path. Default: AUTOPILOT_CONFIG or ./autopilot.yaml if present.")
    parser.add_argument("--env-file", default=None, help="Optional .env path for backwards compatibility.")
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--context-state", default=None)
    parser.add_argument("--agent-root", default=None, help="Optional .autopilot/agent workspace. Default: <cwd>/.autopilot/agent")
    parser.add_argument("--train-yaml", default=None, help="Run: llamafactory-cli train <yaml>.")
    parser.add_argument("--environment", default=None, help="Name/id of a configured runtime environment to activate for this command, e.g. llamaf.")
    parser.add_argument("--env-setup", default=None, help="Explicit shell snippet to run before the command. Overrides --environment.")
    parser.add_argument("--no-env-setup", action="store_true", help="Do not activate any environment, even if --environment/defaults.run.environment is set.")
    parser.add_argument("--env", action="append", default=[], help="Environment override KEY=VALUE for the command. Can be repeated.")
    parser.add_argument("--monitor-interval", type=float, default=30.0, help="Heartbeat interval in seconds for long training commands. Default: 30.")
    parser.add_argument("--monitor-dir", default=None, help="Directory for long_task_status.json and long_task_heartbeats.jsonl. Default: beside the train YAML.")
    parser.add_argument("--no-monitor", action="store_true", help="Disable long-task heartbeat/GPU/log monitoring.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --, e.g. autopilot-run -- python -m pytest -q")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    parse_argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(parse_argv)
    settings = load_settings(config_file=args.config, env_file=args.env_file)
    apply_config_defaults(args, parser, settings, "run", parse_argv, aliases={"environment": ["environment", "runtime_environment"]})
    if settings.config_path:
        print(f"[info] Loaded config: {settings.config_path}")
    if args.train_yaml:
        command = ["llamafactory-cli", "train", args.train_yaml]
        title = f"train {Path(args.train_yaml).name}"
    else:
        command = args.command
        if command and command[0] == "--":
            command = command[1:]
        title = "manual command"
    if not command:
        print("[error] Provide --train-yaml or a command after --.", file=sys.stderr)
        return 2

    if args.train_yaml and not args.no_env_setup and not args.environment and not args.env_setup:
        inferred_stage = _infer_stage_from_train_yaml(args.train_yaml)
        env = choose_environment_for_stage(settings, inferred_stage)
        if env is not None:
            args.environment = env.name
            print(f"[info] auto-selected environment: {env.name} for stage={inferred_stage}")

    env_setup = None
    env_name = None if args.no_env_setup else args.environment
    if not args.no_env_setup:
        if args.env_setup:
            env_setup = args.env_setup
        elif env_name:
            env_setup = settings.environment_activation(env_name)
            if not env_setup:
                known = ", ".join(sorted(settings.environments_as_dict())) or "none"
                print(f"[error] Unknown or non-activatable environment {env_name!r}. Known environments: {known}", file=sys.stderr)
                return 2
    try:
        env_overrides = _parse_env_overrides(args.env)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    display = BashRunner.quote_command(command)
    print(f"[info] command: {display}")
    if env_name:
        print(f"[info] environment: {env_name}")
    if env_setup:
        print(f"[info] env_setup: {env_setup}")
    if env_overrides:
        print(f"[info] env_overrides: {env_overrides}")
    if args.dry_run:
        return 0

    cwd = Path(args.cwd).resolve()
    registry_root = Path(os.getenv("AUTOPILOT_PROCESS_REGISTRY_ROOT") or cwd).resolve()
    process_registry = ProcessRegistry(registry_root)
    context_path = Path(args.context_state) if args.context_state else cwd / ".autopilot" / "context" / "session.json"
    context = ContextManager(context_path, project_root=cwd)
    agent_root = Path(args.agent_root) if args.agent_root else cwd / ".autopilot" / "agent"
    agent = AgentLoop.root(
        name="run",
        objective=f"Execute bash command: {display}",
        context=context,
        workspace_dir=agent_root,
    )
    runner = BashRunner(cwd=cwd, timeout=args.timeout, setup_command=env_setup)

    def bash_task(loop: AgentLoop) -> dict:
        stream_output = str(os.getenv("AUTOPILOT_STREAM_OUTPUT", "1")).lower() not in {"0", "false", "no", "off"}
        supervisor = None
        if args.train_yaml and not args.no_monitor:
            if args.monitor_dir:
                monitor_dir = Path(args.monitor_dir)
            else:
                monitor_dir = Path(args.train_yaml).resolve().parent.parent / "monitor"
            supervisor = LongTaskSupervisor(monitor_dir, label=Path(args.train_yaml).stem, interval_seconds=float(args.monitor_interval or 30.0))
            print(f"[info] monitor: {monitor_dir / 'long_task_status.json'}")
        result = runner.run(
            command,
            env=dict(env_overrides, AUTOPILOT_PROCESS_REGISTRY_ROOT=str(registry_root), AUTOPILOT_ACTION_ID=os.getenv("AUTOPILOT_ACTION_ID", "")),
            stream_output=stream_output,
            heartbeat_interval=float(args.monitor_interval or 0) if supervisor else None,
            heartbeat_callback=supervisor.heartbeat if supervisor else None,
            process_registry=process_registry,
            process_label=f"autopilot-run:{Path(args.train_yaml).stem if args.train_yaml else 'manual'}",
            process_kind="training" if args.train_yaml else "manual_command",
            action_id=os.getenv("AUTOPILOT_ACTION_ID") or None,
            environment_name=env_name,
            process_metadata={"train_yaml": args.train_yaml, "monitor_dir": str(supervisor.root_dir) if supervisor else None},
        )
        if supervisor is not None:
            supervisor.finish(returncode=result.returncode, timed_out=result.timed_out, stdout_tail=result.stdout[-12000:], stderr_tail=result.stderr[-12000:])
        if not stream_output:
            print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, file=sys.stderr, end="")
        context.add_event(
            "bash",
            title,
            f"returncode={result.returncode}, ok={result.ok}",
            {"command": result.command, "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:], "env_setup": result.env_setup, "environment": env_name, "env_overrides": env_overrides, "monitor_dir": str(supervisor.root_dir) if supervisor else None, "pid": result.pid, "process_group_id": result.process_group_id, "process_id": result.process_id, "process_registry": str(process_registry.registry_path)},
            importance=2,
        )
        loop.record_tool_call(
            "bash.run",
            inputs={"command": result.command, "cwd": result.cwd, "timeout": args.timeout, "env_setup": result.env_setup, "environment": env_name, "env_overrides": env_overrides},
            output_summary=f"returncode={result.returncode}, ok={result.ok}",
            output={"stdout_tail": result.stdout[-4000:], "stderr_tail": result.stderr[-4000:], "timed_out": result.timed_out, "pid": result.pid, "process_group_id": result.process_group_id, "process_id": result.process_id, "process_registry": str(process_registry.registry_path)},
            importance=2,
        )
        loop.set_result("Bash command completed", {"returncode": result.returncode, "ok": result.ok, "timed_out": result.timed_out, "env_setup": result.env_setup, "environment": env_name, "monitor_dir": str(supervisor.root_dir) if supervisor else None, "pid": result.pid, "process_group_id": result.process_group_id, "process_id": result.process_id, "process_registry": str(process_registry.registry_path)})
        if not result.ok:
            raise RuntimeError(f"Command failed: returncode={result.returncode}, timed_out={result.timed_out}")
        return {"returncode": result.returncode, "pid": result.pid, "process_group_id": result.process_group_id, "process_id": result.process_id, "process_registry": str(process_registry.registry_path)}

    task_result = agent.run_task(title, f"Run command: {display}", bash_task, task_type="bash", raise_on_error=False)
    agent.set_result("Run command finished", {"task_status": task_result.status, "result_path": task_result.result_path})
    agent.save_loop_index()
    context.save()
    print(f"[done] context: {context_path}")
    print(f"[done] agent loop: {agent_root / 'loop_index.json'}")
    return 0 if task_result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
