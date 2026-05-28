from __future__ import annotations

import argparse
import json
from pathlib import Path

from autopilot.models import to_jsonable
from autopilot.runtime.processes import ProcessRegistry


def _print(data: object) -> None:
    print(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect/start/kill Autopilot-managed processes.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--root", required=True)
    p.add_argument("--active-only", action="store_true")
    p.add_argument("--no-exited", action="store_true")
    p = sub.add_parser("kill")
    p.add_argument("--root", required=True)
    p.add_argument("--process-id")
    p.add_argument("--pid", type=int)
    p.add_argument("--name")
    p.add_argument("--kind")
    p.add_argument("--pattern")
    p.add_argument("--signal", default="TERM")
    p.add_argument("--no-process-group", action="store_true")
    p.add_argument("--force-after-seconds", type=float, default=5.0)
    p.add_argument("--allow-untracked", action="store_true")
    p = sub.add_parser("start")
    p.add_argument("--root", required=True)
    p.add_argument("--command", required=True)
    p.add_argument("--name", default="manual_process")
    p.add_argument("--kind", default="manual")
    p.add_argument("--cwd", default=".")
    p.add_argument("--environment")
    p.add_argument("--setup-command")
    p.add_argument("--log-file")
    p.add_argument("--pid-file")
    args = parser.parse_args(argv)
    store = ProcessRegistry(args.root)
    if args.cmd == "list":
        _print({"registry_path": str(store.registry_path), "processes": store.list(active_only=bool(args.active_only), include_exited=not bool(args.no_exited))})
        return 0
    if args.cmd == "kill":
        _print(store.kill(process_id=args.process_id, pid=args.pid, name=args.name, kind=args.kind, pattern=args.pattern, sig=args.signal, kill_process_group=not args.no_process_group, force_after_seconds=args.force_after_seconds, allow_untracked=args.allow_untracked, reason="autopilot-processes kill"))
        return 0
    if args.cmd == "start":
        _print(store.start_background(args.command, cwd=Path(args.cwd), setup_command=args.setup_command, label=args.name, kind=args.kind, environment=args.environment, log_file=args.log_file, pid_file=args.pid_file))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
