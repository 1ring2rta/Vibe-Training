from __future__ import annotations

import argparse
from pathlib import Path

from autopilot.context import ContextManager


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Autopilot session context and progressive compaction.")
    parser.add_argument("--state", default=".autopilot/context/session.json", help="Context state JSON path.")
    parser.add_argument("--project-root", default=".", help="Project root for AUTOPILOT.md and .autopilot/memory.md.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="Create or load a context state.")
    init.add_argument("--project-memory", default=None, help="Optional text to write into AUTOPILOT.md.")

    add = sub.add_parser("add", help="Add an event.")
    add.add_argument("--kind", required=True)
    add.add_argument("--title", required=True)
    add.add_argument("--summary", required=True)
    add.add_argument("--importance", type=int, default=1)

    mem = sub.add_parser("remember", help="Append a stable note to .autopilot/memory.md or Claude memory.")
    mem.add_argument("note")
    mem.add_argument("--claude", action="store_true", help="Write to .claude/autopilot_memory.md instead of .autopilot/memory.md.")

    compact = sub.add_parser("compact", help="Compress older events into the rolling summary.")
    compact.add_argument("--keep", type=int, default=12)

    show = sub.add_parser("show", help="Render current context packet.")
    show.add_argument("--max-chars", type=int, default=24000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manager = ContextManager(args.state, project_root=args.project_root)
    if args.cmd == "init":
        if args.project_memory:
            manager.write_project_memory(args.project_memory)
        manager.save()
        print(f"[done] context state: {Path(args.state).resolve()}")
    elif args.cmd == "add":
        event = manager.add_event(args.kind, args.title, args.summary, importance=args.importance)
        print(f"[done] event: {event.event_id}")
    elif args.cmd == "remember":
        path = manager.append_auto_memory(args.note, claude=args.claude)
        print(f"[done] memory: {path}")
    elif args.cmd == "compact":
        manager.compact(keep_recent_events=args.keep)
        print(f"[done] compacted: {Path(args.state).resolve()}")
    elif args.cmd == "show":
        print(manager.render_context(max_chars=args.max_chars))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
