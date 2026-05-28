from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from autopilot.config import load_settings, validate_settings
from autopilot.kernel.agent_kernel import AutonomousAgentKernel


def _parse_target(target: str | None) -> str:
    return target or ""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the de-workflowed autonomous post-training kernel. The teacher composes all work from atomic tools: bash, cat, grep, web_search, browser.")
    p.add_argument("--config", default=None, help="YAML config. Prefer generic clients/roles, e.g. clients.teacher + roles.director=teacher.")
    p.add_argument("--env-file", default=None)
    p.add_argument("--goal", required=True)
    p.add_argument("--target", default="", help="Target expression, e.g. aime24_all.exact_match_accuracy>=0.80 or swe_bench_lite.resolved_rate>=0.20")
    p.add_argument("--output-dir", default="runs/autonomous_goal")
    p.add_argument("--role", default="director")
    p.add_argument("--max-iterations", type=int, default=96)
    p.add_argument("--max-seconds", type=float, default=None, help="Wall-clock budget for this invocation, measured with time.time().")
    p.add_argument("--max-hours", type=float, default=None, help="Convenience wall-clock budget in hours; overrides --max-seconds when set.")
    p.add_argument("--execute", dest="execute", action="store_true", default=True)
    p.add_argument("--plan", "--plan-only", "--no-execute", dest="execute", action="store_false")
    p.add_argument("--no-bash", action="store_true", help="Hide/deny bash tool for this run.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(config_file=args.config, env_file=args.env_file)
    if settings.config_path:
        print(f"[info] Loaded config: {settings.config_path}")
    for w in validate_settings(settings):
        print(f"[warn] Config: {w}", file=sys.stderr)
    max_seconds = args.max_seconds
    if args.max_hours is not None:
        max_seconds = float(args.max_hours) * 3600.0
    kernel = AutonomousAgentKernel(
        settings=settings,
        root=Path(args.output_dir),
        goal=args.goal,
        target=_parse_target(args.target),
        execute=bool(args.execute),
        role=args.role,
        max_iterations=args.max_iterations,
        max_seconds=max_seconds,
        allow_bash=not args.no_bash,
    )
    report = kernel.run()
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    print(f"[done] autonomous kernel report: {Path(args.output_dir) / 'autonomous_kernel_report.json'}")
    print(f"[done] frontier trajectory: {Path(args.output_dir) / '.autopilot' / 'frontier_trajectory'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
