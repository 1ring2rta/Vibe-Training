from __future__ import annotations

import argparse
import json

from autopilot.eval.programs import EvalProgramWorkspace
from autopilot.models import to_jsonable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare/refine/list run-local benchmark evaluator programs.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--root", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--goal", default="")
    p.add_argument("--target", default="")
    p.add_argument("--metric")
    p = sub.add_parser("write")
    p.add_argument("--root", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--benchmark", default="custom")
    p.add_argument("--metric")
    p.add_argument("--notes", default="")
    p.add_argument("--evaluator-py")
    p = sub.add_parser("refine")
    p.add_argument("--root", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--instructions", required=True)
    p.add_argument("--patch")
    p = sub.add_parser("list")
    p.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    ws = EvalProgramWorkspace(args.root)
    if args.cmd == "plan":
        spec = ws.plan_from_benchmark(args.benchmark, goal=args.goal, target=args.target, metric=args.metric)
        print(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "write":
        files = {"eval.py": args.evaluator_py} if args.evaluator_py else {}
        spec = ws.write_generated_program(name=args.name, benchmark=args.benchmark, metric=args.metric, files=files, notes=args.notes)
        print(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "refine":
        out = ws.record_refinement(name=args.name, instructions=args.instructions, patch=args.patch)
        print(json.dumps(to_jsonable(out), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "list":
        print(json.dumps(to_jsonable({"ok": True, "programs": ws.list(), "registry_path": str(ws.registry_path)}), ensure_ascii=False, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
