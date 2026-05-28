from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from autopilot.config import load_settings, validate_settings
from autopilot.goal.loop import GoalLoopRunner
from autopilot.kernel.agent_kernel import AutonomousAgentKernel
from autopilot.goal.spec import build_goal_spec
from autopilot.models import to_jsonable


def _parse_target_expr(target: str | None) -> tuple[str | None, float | None]:
    if not target:
        return None, None
    match = re.match(r"\s*([A-Za-z0-9_.-]+)\s*(?:>=|=|:)?\s*([0-9]*\.?[0-9]+)\s*$", target)
    if not match:
        raise ValueError("--target should look like accuracy>=0.8 or pass_rate:0.75")
    return match.group(1), float(match.group(2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a target-driven nested agent loop: eval target -> collect/train/eval -> diagnose -> remediate.")
    parser.add_argument("--config", default=None, help="YAML config path. Default: AUTOPILOT_CONFIG or ./autopilot.yaml if present.")
    parser.add_argument("--env-file", default=None, help="Optional .env path for backwards compatibility.")
    parser.add_argument("--goal", default=None, help="High-level target, e.g. '提升模型 coding 能力'.")
    parser.add_argument("--target", default=None, help="Compact target expression, e.g. accuracy>=0.80.")
    parser.add_argument("--target-metric", default=None, help="Metric name, e.g. accuracy, pass_rate, win_rate.")
    parser.add_argument("--target-score", type=float, default=None, help="Target threshold, e.g. 0.80.")
    parser.add_argument("--eval-set", default=None, help="JSON/JSONL eval set with prompt/expected/tags/verifier fields.")
    parser.add_argument("--base-model", default=None, help="Base model name/path for the goal spec.")
    parser.add_argument("--max-rounds", type=int, default=None, help="Maximum loop rounds.")
    parser.add_argument("--output-dir", default="runs/goal_loop", help="Output directory for reports, contexts, and round artifacts.")
    exec_group = parser.add_mutually_exclusive_group()
    exec_group.add_argument("--execute", dest="execute", action="store_true", default=None, help="Run collect/prepare/train/bash subtasks. This is the default for real runs.")
    exec_group.add_argument("--plan-only", "--plan", "--no-execute", dest="execute", action="store_false", help="Plan commands and write loop state without executing collect/prepare/train.")
    parser.add_argument("--no-evaluate", action="store_true", help="Skip local evaluation even if vLLM is configured.")
    parser.add_argument("--no-web-search", action="store_true", help="Do not use web search for tool/verifier discovery.")
    parser.add_argument("--no-teacher-samples", "--no-kimi-samples", dest="no_teacher_samples", action="store_true", help="Do not ask the remote teacher model to generate eval/remediation samples.")
    parser.add_argument("--discover-tools", action=argparse.BooleanOptionalAction, default=True, help="Infer/discover tool candidates. Currently this controls web-search use for tool discovery.")
    parser.add_argument("--discover-verifiers", action=argparse.BooleanOptionalAction, default=True, help="Infer/discover verifier candidates. Currently this controls web-search use for verifier discovery.")
    parser.add_argument("--max-generated-tests", type=int, default=5, help="Max fallback/teacher-generated eval samples.")
    parser.add_argument("--agent-max-iterations", type=int, default=512)
    parser.add_argument("--max-seconds", type=float, default=None, help="Wall-clock budget for autonomous kernel invocations, measured with time.time().")
    parser.add_argument("--max-hours", type=float, default=None, help="Wall-clock budget in hours for autonomous kernel invocations; overrides --max-seconds.")
    parser.add_argument("--no-resource-discovery", action="store_true", help="Skip nvidia-smi / local resource inspection.")
    parser.add_argument("--no-self-improve-repo", action="store_true", help="Skip teacher repository self-improvement / model-director tasks.")
    parser.add_argument("--no-manage-vllm", action="store_true", help="Do not prepare vLLM service start/kill commands for the teacher agent to choose from.")
    parser.add_argument("--interactive-human", action=argparse.BooleanOptionalAction, default=None, help="When ask_human is used, prompt on stdin if interactive; otherwise write a pending question artifact.")
    parser.add_argument("--legacy-workflow", action="store_true", help="Use the pre-v0.7 GoalLoopRunner instead of the autonomous agent kernel.")
    parser.add_argument("--autonomous", action=argparse.BooleanOptionalAction, default=None, help="Use the autonomous agent kernel. Default: enabled when a director/teacher client is configured; otherwise legacy plan mode is used.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    parse_argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(parse_argv)
    if args.target:
        try:
            args.target_metric, args.target_score = _parse_target_expr(args.target)
        except ValueError as exc:
            parser.error(str(exc))

    settings = load_settings(config_file=args.config, env_file=args.env_file)
    execute = True if args.execute is None else bool(args.execute)
    if settings.config_path:
        print(f"[info] Loaded config: {settings.config_path}")
    for warning in validate_settings(settings):
        print(f"[warn] Config: {warning}", file=sys.stderr)

    spec = build_goal_spec(
        raw_config=settings.raw_config,
        goal=args.goal,
        target_metric=args.target_metric,
        target_score=args.target_score,
        eval_set=args.eval_set,
        base_model=args.base_model,
        max_rounds=args.max_rounds,
        use_kimi_generated_tests=not args.no_teacher_samples,
    )
    if not spec.description:
        parser.error("--goal is required unless goal.description is set in YAML.")

    has_director = False
    try:
        reg = settings.client_registry(trajectory_root=Path(args.output_dir) / ".autopilot" / "frontier_trajectory")
        has_director = bool(reg.role_client_name("director"))
    except Exception:
        has_director = False
    use_autonomous = (args.autonomous if args.autonomous is not None else has_director) and not args.legacy_workflow
    max_seconds = args.max_seconds
    if args.max_hours is not None:
        max_seconds = float(args.max_hours) * 3600.0

    if use_autonomous:
        target_expr = args.target or (f"{spec.target.name}>={spec.target.target}" if spec.target.name else "")
        kernel = AutonomousAgentKernel(
            settings=settings,
            root=Path(args.output_dir),
            goal=spec.description,
            target=target_expr,
            execute=execute,
            role="director",
            max_iterations=args.agent_max_iterations,
            max_seconds=max_seconds,
            allow_bash=True,
        )
        kernel_report = kernel.run()
        result = {
            "report_json": str(Path(args.output_dir) / "autonomous_kernel_report.json"),
            "report_markdown": str(Path(args.output_dir) / "autonomous_kernel_report.md"),
            "agent_root": str(Path(args.output_dir) / ".autopilot"),
        }
        # The autonomous CLI writes JSON; create a compact Markdown peer for compatibility.
        md = Path(result["report_markdown"])
        if not md.exists():
            md.write_text(f"# Autonomous Kernel Report\n\nstatus: {kernel_report.status}\n\nelapsed_seconds: {kernel_report.elapsed_seconds}\n", encoding="utf-8")
    else:
        runner = GoalLoopRunner(
            spec=spec,
            settings=settings,
            output_dir=Path(args.output_dir),
            config_path=args.config or settings.config_path,
            execute=execute,
            use_web_search=(not args.no_web_search and (args.discover_tools or args.discover_verifiers)),
            use_kimi_samples=not args.no_teacher_samples,
            evaluate=not args.no_evaluate,
            max_generated_tests=args.max_generated_tests,
            agent_max_iterations=args.agent_max_iterations,
            discover_resources=not args.no_resource_discovery,
            self_improve_repo=not args.no_self_improve_repo,
            manage_vllm=not args.no_manage_vllm,
            interactive_human=args.interactive_human,
        )
        result = runner.run()

    # Friendly compatibility files with short names for quick inspection.
    out_dir = Path(args.output_dir)
    final_score = 0.0
    status = "planned"
    try:
        data = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))
        evaluation = data.get("evaluation") or {}
        if evaluation.get("score") is not None:
            final_score = float(evaluation.get("score"))
            status = "target_met" if bool(evaluation.get("target_met")) else "needs_more_rounds"
    except Exception:
        pass
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "eval_suite.json").write_text(json.dumps({"cases": to_jsonable(spec.eval_cases)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compat_report = {
        "status": status,
        "final_score": final_score,
        "target_metric": spec.target.name,
        "target_value": spec.target.target,
        "artifacts": {"goal_loop_report": result["report_json"], "goal_loop_markdown": result["report_markdown"], "plan_commands": result["report_json"]},
    }
    (out_dir / "goal_report.json").write_text(json.dumps(compat_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[done] Goal report JSON: {result['report_json']}")
    print(f"[done] Goal report Markdown: {result['report_markdown']}")
    print(f"[done] agent loop: {Path(result['agent_root']) / 'loop_index.json'}")
    if not execute:
        print("[info] collect/prepare/train/model commands were planned but not executed because --plan-only/--no-execute was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
