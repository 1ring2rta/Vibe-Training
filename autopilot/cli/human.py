from __future__ import annotations

import argparse
from pathlib import Path

from autopilot.tools.ask_human import AskHumanTool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or answer Autopilot ask_human questions.")
    parser.add_argument("--workspace", default=".autopilot/human", help="Human queue workspace. Goal runs use <output>/.autopilot/human.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="List pending/answered questions.")
    reply = sub.add_parser("reply", help="Answer a question by id.")
    reply.add_argument("question_id")
    reply.add_argument("--answer", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    tool = AskHumanTool(Path(args.workspace))
    if args.cmd == "list":
        questions = tool.list_questions()
        if not questions:
            print("[info] no ask_human questions found")
            return 0
        for q in questions:
            print(f"{q.question_id}\t{q.status}\t{q.question}")
            if q.response:
                print(f"  answer: {q.response}")
        return 0
    if args.cmd == "reply":
        path = tool.reply(args.question_id, args.answer)
        print(f"[done] wrote reply: {path}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
