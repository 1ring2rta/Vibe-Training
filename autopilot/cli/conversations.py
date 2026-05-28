from __future__ import annotations

import argparse
import json
from pathlib import Path

from autopilot.config import load_settings
from autopilot.llm.conversation_recorder import export_conversation_logs, write_llamafactory_dataset_info


def _count_jsonl(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except Exception:
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect/export trainable KIMI conversation logs.")
    parser.add_argument("command", choices=["paths", "stats", "export"], help="paths: print expected paths; stats: count records; export: copy logs and write dataset_info.json")
    parser.add_argument("--config", default=None, help="YAML config path. Default: AUTOPILOT_CONFIG or ./autopilot.yaml if present.")
    parser.add_argument("--env-file", default=None, help="Optional .env file path for backwards compatibility.")
    parser.add_argument("--root", default=None, help="Conversation log root. Default from conversation_logging.root or .autopilot/conversations")
    parser.add_argument("--output-dir", default=None, help="Export destination. Default: same as --root")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    settings = load_settings(config_file=args.config, env_file=args.env_file)
    raw = settings.raw_config.get("conversation_logging") if isinstance(settings.raw_config, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    root = Path(args.root or raw.get("root") or ".autopilot/conversations")
    paths = {
        "root": str(root),
        "raw_calls": str(root / "kimi_raw_calls.jsonl"),
        "messages": str(root / "kimi_messages.jsonl"),
        "sharegpt": str(root / "kimi_sharegpt.jsonl"),
        "multiturn_messages": str(root / "kimi_multiturn_messages.jsonl"),
        "multiturn_sharegpt": str(root / "kimi_multiturn_sharegpt.jsonl"),
        "dataset_info": str(root / "dataset_info.json"),
        "state": str(root / "kimi_session_state.json"),
    }
    if args.command == "paths":
        if args.json:
            print(json.dumps(paths, ensure_ascii=False, indent=2))
        else:
            for k, v in paths.items():
                print(f"{k}: {v}")
        return 0
    if args.command == "stats":
        write_llamafactory_dataset_info(root)
        stats = {k: _count_jsonl(Path(v)) for k, v in paths.items() if k not in {"root", "dataset_info", "state"}}
        stats["root"] = str(root)
        stats["dataset_info"] = paths["dataset_info"]
        if args.json:
            print(json.dumps(stats, ensure_ascii=False, indent=2))
        else:
            for k, v in stats.items():
                print(f"{k}: {v}")
        return 0
    copied = export_conversation_logs(root, args.output_dir)
    if args.json:
        print(json.dumps(copied, ensure_ascii=False, indent=2))
    else:
        for k, v in copied.items():
            print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
