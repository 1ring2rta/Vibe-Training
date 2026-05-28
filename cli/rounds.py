from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autopilot.goal.round_trace import load_round_metrics_history


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _round_dir(root: Path, round_idx: int) -> Path:
    return root / f"round_{round_idx}"


def command_summary(args: argparse.Namespace) -> int:
    root = Path(args.root)
    history = load_round_metrics_history(root)
    trajectories = []
    for manifest_path in sorted(root.glob("round_*/kimi_trajectory/combined/manifest.json")):
        data = _load_json(manifest_path, {})
        if isinstance(data, dict):
            trajectories.append(data)
    data = {"root": str(root), "round_metrics": history, "round_kimi_trajectories": trajectories}
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    print(f"run_root: {root}")
    if not history:
        print("round metrics: none")
    else:
        print("round metrics:")
        for item in history:
            delta = item.get("metric_delta") or {}
            pre = delta.get("pre") or {}
            post = delta.get("post") or {}
            print(f"- round {item.get('round')}: {pre.get('score')} -> {post.get('score')} delta={delta.get('score_delta')} stage={item.get('train_stage')} training_ok={item.get('training_ok')}")
            paths = item.get("paths") or {}
            if paths.get("json"):
                print(f"  metrics_json: {paths['json']}")
    if trajectories:
        print("round KIMI trajectories:")
        for item in trajectories:
            counts = item.get("counts") or {}
            print(f"- round {item.get('round')}: {item.get('output_dir')} raw_calls={counts.get('kimi_raw_calls.jsonl', 0)} messages={counts.get('kimi_messages.jsonl', 0)}")
    return 0


def command_metrics(args: argparse.Namespace) -> int:
    root = Path(args.root)
    path = _round_dir(root, args.round) / "metrics" / "round_metrics.json"
    data = _load_json(path, {})
    if not data:
        print(f"[error] missing round metrics: {path}")
        return 2
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    delta = data.get("metric_delta") or {}
    pre = delta.get("pre") or {}
    post = delta.get("post") or {}
    print(f"round: {data.get('round')}")
    print(f"metric: {data.get('metric_name')} target={data.get('target_value')}")
    print(f"before: score={pre.get('score')} target_met={pre.get('target_met')} failures={pre.get('failure_count')}")
    print(f"after:  score={post.get('score')} target_met={post.get('target_met')} failures={post.get('failure_count')}")
    print(f"delta:  score_delta={delta.get('score_delta')} failure_delta={delta.get('failure_delta')}")
    paths = data.get("paths") or {}
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


def command_trajectory(args: argparse.Namespace) -> int:
    root = Path(args.root)
    path = _round_dir(root, args.round) / "kimi_trajectory" / "combined" / "manifest.json"
    data = _load_json(path, {})
    if not data:
        print(f"[error] missing round KIMI trajectory manifest: {path}")
        return 2
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    print(f"round: {data.get('round')}")
    print(f"output_dir: {data.get('output_dir')}")
    print(f"dataset_info: {data.get('dataset_info')}")
    print("counts:")
    for k, v in (data.get("counts") or {}).items():
        print(f"- {k}: {v}")
    print("merged files:")
    for k, v in (data.get("merged_files") or {}).items():
        print(f"- {k}: {v}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect per-round metrics and KIMI trajectories for an Autopilot goal run.")
    sub = parser.add_subparsers(dest="command", required=True)

    summary = sub.add_parser("summary", help="Show round metric history and trajectory locations.")
    summary.add_argument("--root", required=True, help="Goal run root, e.g. runs/coding_goal_v3")
    summary.add_argument("--json", action="store_true", help="Print JSON")
    summary.set_defaults(func=command_summary)

    metrics = sub.add_parser("metrics", help="Show one round's before/after metrics.")
    metrics.add_argument("--root", required=True, help="Goal run root")
    metrics.add_argument("--round", type=int, required=True, help="Round number")
    metrics.add_argument("--json", action="store_true", help="Print JSON")
    metrics.set_defaults(func=command_metrics)

    trajectory = sub.add_parser("trajectory", help="Show one round's merged KIMI trajectory bundle.")
    trajectory.add_argument("--root", required=True, help="Goal run root")
    trajectory.add_argument("--round", type=int, required=True, help="Round number")
    trajectory.add_argument("--json", action="store_true", help="Print JSON")
    trajectory.set_defaults(func=command_trajectory)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
