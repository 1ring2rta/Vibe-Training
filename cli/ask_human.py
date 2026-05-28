from __future__ import annotations

import argparse
import json
from pathlib import Path

from autopilot.tools.ask_human import AskHumanTool


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ask/answer human questions queued by the agent.")
    p.add_argument("--root", default=".", help="Project/run root containing .autopilot/human.")
    sub = p.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--urgency", default="normal")
    ask.add_argument("--option", action="append", default=[])
    ask.add_argument("--blocking", action="store_true")
    ls = sub.add_parser("list")
    ls.add_argument("--status", default=None)
    ans = sub.add_parser("answer")
    ans.add_argument("question_id")
    ans.add_argument("response")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    tool = AskHumanTool(Path(args.root), mode="blocking" if getattr(args, "blocking", False) else "queue")
    if args.cmd == "ask":
        q = tool.ask(args.question, urgency=args.urgency, suggested_options=args.option, blocking=args.blocking)
        print(json.dumps(q.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "list":
        data = [q.to_dict() for q in tool.list_questions(status=args.status)]
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "answer":
        q = tool.answer(args.question_id, args.response)
        print(json.dumps(q.to_dict(), ensure_ascii=False, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
