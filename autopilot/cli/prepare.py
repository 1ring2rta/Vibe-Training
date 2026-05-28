from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from autopilot.agent import AgentLoop
from autopilot.config import apply_config_defaults, load_settings
from autopilot.context import ContextManager
from autopilot.data.decontam import DecontaminationReportBuilder
from autopilot.llamafactory.config import generate_training_configs
from autopilot.llamafactory.converter import LlamaFactoryConverter, choose_stage, write_dataset_info
from autopilot.llamafactory.sources import load_rows_for_item
from autopilot.llamafactory.validate import validate_prepared_dataset_dir, validate_train_yaml
from autopilot.tools.bash import BashRunner
from autopilot.models import to_jsonable


def _get(item: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = item
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _training_types(item: dict[str, Any]) -> list[str]:
    decision_types = _get(item, ["adoption_decision", "training_types"], None)
    if decision_types:
        return [str(x) for x in decision_types]
    class_types = _get(item, ["classification", "recommended_training"], []) or []
    return [str(x) for x in class_types]


def _format_type(item: dict[str, Any]) -> str:
    return str(_get(item, ["classification", "format_type"], "unknown"))


def _final_score(item: dict[str, Any]) -> float:
    val = _get(item, ["adoption_decision", "final_score"], None)
    if val is None:
        val = _get(item, ["score", "suitability_score"], 0.0)
    try:
        return float(val)
    except Exception:
        return 0.0


def _action(item: dict[str, Any]) -> str:
    return str(_get(item, ["adoption_decision", "action"], "review"))


def _item_stage(item: dict[str, Any]) -> str:
    format_type = _format_type(item)
    types = {x.lower() for x in _training_types(item)}
    if format_type in {"preference_pair", "preference_pair_without_explicit_prompt"} or "dpo" in types:
        return "dpo"
    if "kto" in types:
        return "kto"
    if format_type == "raw_text" or "continued_pretraining" in types or "pt" in types or "cpt" in types:
        return "pt"
    stage = choose_stage(_training_types(item), format_type, requested_stage="auto")
    if "rlvr" in types and stage == "sft":
        # LLaMA-Factory does not consume RLVR directly here, but keep a quota
        # bucket so verifiable/test datasets are not always crowded out by SFT.
        return "rlvr"
    return stage


def _parse_stage_quota(value: str | None) -> dict[str, int]:
    quotas: dict[str, int] = {}
    if not value:
        return quotas
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            quotas[item.lower()] = 1
            continue
        key, val = item.split(":", 1)
        try:
            quotas[key.strip().lower()] = max(0, int(val.strip()))
        except Exception:
            continue
    return quotas


def select_items(report: list[dict[str, Any]], actions: set[str], min_score: float, limit: int | None, stage_quota: str | None = None) -> list[dict[str, Any]]:
    selected = [item for item in report if _action(item) in actions and _final_score(item) >= min_score]
    selected.sort(key=_final_score, reverse=True)
    quotas = _parse_stage_quota(stage_quota)
    if not quotas:
        return selected[:limit] if limit is not None else selected

    chosen: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for stage, quota in quotas.items():
        if quota <= 0:
            continue
        stage_items = [item for item in selected if _item_stage(item) == stage]
        # If RLVR-specific items are scarce, let QA/test-like SFT items fill the bucket.
        if stage == "rlvr" and not stage_items:
            stage_items = [item for item in selected if "rlvr" in {x.lower() for x in _training_types(item)}]
        for item in stage_items[:quota]:
            dataset_id = str(item.get("dataset_id"))
            if dataset_id not in seen_ids:
                chosen.append(item)
                seen_ids.add(dataset_id)
    if limit is None or len(chosen) < limit:
        for item in selected:
            dataset_id = str(item.get("dataset_id"))
            if dataset_id in seen_ids:
                continue
            chosen.append(item)
            seen_ids.add(dataset_id)
            if limit is not None and len(chosen) >= limit:
                break
    return chosen[:limit] if limit is not None else chosen


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare accepted dataset candidates as LLaMA-Factory local data, dataset_info.json, and train YAML configs."
    )
    parser.add_argument("--report", default=None, help="Path to collection_report.json from autopilot-collect. Can also be set in config defaults.prepare.report.")
    parser.add_argument("--output-dir", default="prepared", help="Output directory for prepared artifacts.")
    parser.add_argument("--config", default=None, help="YAML config path. Default: ./autopilot.yaml or AUTOPILOT_CONFIG.")
    parser.add_argument("--env-file", default=None, help="Optional .env path for backward compatibility.")
    parser.add_argument("--source", choices=["auto", "hf", "report"], default="auto", help="Rows source. auto tries HF streaming then report samples.")
    parser.add_argument("--actions", default="accept", help="Comma-separated adoption actions to include, e.g. accept,review.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum final/rule score to include.")
    parser.add_argument("--dataset-limit", type=int, default=None, help="Maximum selected datasets to convert.")
    parser.add_argument("--stage-quota", default=None, help="Diversified stage quotas, e.g. sft:3,dpo:2,pt:2,kto:1,rlvr:2. Applied before filling remaining slots by score.")
    parser.add_argument("--max-rows", type=int, default=5000, help="Maximum converted rows per dataset.")
    parser.add_argument("--target", default="", help="Target benchmark/metric string used for data decontamination gates, e.g. aime24_all.exact_match_accuracy>=0.5.")
    parser.add_argument("--eval-cases", default=None, help="Optional JSON/JSONL target eval cases used for exact normalized prompt decontamination.")
    parser.add_argument("--allow-benchmark-contamination", action="store_true", help="Write decontamination findings but do not block benchmark-overlapping datasets.")
    parser.add_argument("--max-zero-row-fraction", type=float, default=0.5, help="Fail preparation when too many selected datasets write zero rows. Set >1 to disable.")
    parser.add_argument("--stage", default="auto", help="Force stage: auto/sft/dpo/kto/pt/rm. Default auto.")
    parser.add_argument("--name-prefix", default="auto", help="Prefix for LLaMA-Factory dataset names.")
    parser.add_argument("--merge-dataset-info", action="store_true", help="Merge with existing dataset_info.json if present.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote HF dataset loading code when source is hf/auto.")

    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct", help="model_name_or_path for generated YAML.")
    parser.add_argument("--template", default="qwen", help="LLaMA-Factory template name.")
    parser.add_argument("--finetuning-type", default="lora", help="lora/full/freeze.")
    parser.add_argument("--cutoff-len", type=int, default=2048)
    parser.add_argument("--yaml-max-samples", type=int, default=None, help="Optional max_samples in generated YAML.")
    parser.add_argument("--train-output-root", default="saves", help="Root output_dir for LLaMA-Factory checkpoints.")
    parser.add_argument("--report-to", default="none", help="LLaMA-Factory report_to value, e.g. none/wandb/tensorboard.")

    parser.add_argument("--run-smoke-test", action="store_true", help="Use bash runner to validate generated dataset files.")
    parser.add_argument("--run-train", action="store_true", help="Run llamafactory-cli train for generated YAML files.")
    parser.add_argument("--train-stage", default=None, help="Only run this stage yaml when --run-train, e.g. sft/dpo.")
    parser.add_argument("--bash-timeout", type=float, default=600.0, help="Timeout for bash commands.")
    parser.add_argument("--context-state", default=None, help="Optional context state path. Default: <output-dir>/.autopilot/context/session.json")
    parser.add_argument("--agent-max-iterations", type=int, default=256, help="Maximum events per agent loop before it raises.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    parse_argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(parse_argv)
    settings = load_settings(config_file=args.config, env_file=args.env_file)
    apply_config_defaults(args, parser, settings, "prepare", parse_argv)
    if not args.report:
        parser.error("--report is required unless defaults.prepare.report is set in the YAML config.")
    if settings.config_path:
        print(f"[info] Loaded config: {settings.config_path}")
    report_path = Path(args.report)
    output_dir = Path(args.output_dir)
    dataset_dir = output_dir / "data"
    config_dir = output_dir / "configs"
    output_dir.mkdir(parents=True, exist_ok=True)

    context_path = Path(args.context_state) if args.context_state else output_dir / ".autopilot" / "context" / "session.json"
    context = ContextManager(context_path, project_root=Path.cwd())
    agent = AgentLoop.root(
        name="prepare",
        objective=f"Prepare LLaMA-Factory artifacts from {report_path}",
        context=context,
        workspace_dir=output_dir / ".autopilot" / "agent",
        max_iterations=args.agent_max_iterations,
    )
    context.add_event("prepare", "start", f"Preparing LLaMA-Factory artifacts from {report_path}", {"args": vars(args)}, importance=2)

    report: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    conversions = []
    source_notes: list[dict[str, Any]] = []
    skipped_datasets: list[dict[str, Any]] = []
    yaml_paths: dict[str, Path] = {}
    manifest_path: Path | None = None
    dataset_info_path: Path | None = None
    decontam_report_path: Path = output_dir / "decontamination_report.json"
    decontam = DecontaminationReportBuilder(
        target=args.target,
        eval_cases_path=args.eval_cases,
        allow_contamination=bool(args.allow_benchmark_contamination),
    )

    def read_report_task(loop: AgentLoop) -> dict:
        nonlocal report
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("collection report must be a JSON array")
            report = data
        except Exception as exc:
            print(f"[error] Could not read report: {exc}", file=sys.stderr)
            context.add_event("prepare", "read_report_failed", str(exc), importance=3)
            raise
        loop.set_result(f"Read {len(report)} report items", {"report": str(report_path), "item_count": len(report)})
        return {"item_count": len(report)}

    try:
        agent.run_task("read_collection_report", "Load collection_report.json as the input to preparation", read_report_task)
    except Exception:
        context.save()
        return 2

    def select_task(loop: AgentLoop) -> dict:
        nonlocal selected
        actions = {x.strip() for x in args.actions.split(",") if x.strip()}
        raw_selected = select_items(report, actions=actions, min_score=args.min_score, limit=args.dataset_limit, stage_quota=args.stage_quota)
        selected = []
        for item in raw_selected:
            dataset_id = str(item.get("dataset_id") or "")
            findings = decontam.inspect_dataset_id(dataset_id)
            if any(f.blocked for f in findings):
                skipped_datasets.append({"dataset_id": dataset_id, "reason": "decontamination_name_block", "findings": [f.to_dict() for f in findings]})
                continue
            selected.append(item)
        decontam.write(decontam_report_path)
        print(f"[info] Selected {len(selected)} datasets from {len(report)} report items ({len(raw_selected) - len(selected)} blocked by decontamination).")
        context.add_event("prepare", "selected_datasets", f"Selected {len(selected)} datasets", {"dataset_ids": [x.get("dataset_id") for x in selected], "blocked_dataset_ids": decontam.report.blocked_dataset_ids})
        loop.decide(
            "selected_datasets",
            f"Selected {len(selected)} datasets from actions={sorted(actions)} and min_score={args.min_score}",
            {"dataset_ids": [x.get("dataset_id") for x in selected], "stages": [_item_stage(x) for x in selected], "stage_quota": args.stage_quota},
        )
        loop.add_artifact(decontam_report_path, "decontamination_report", "Benchmark contamination screening report")
        loop.set_result("Selected datasets for conversion", {"selected_count": len(selected), "dataset_ids": [x.get("dataset_id") for x in selected], "stages": [_item_stage(x) for x in selected], "stage_quota": args.stage_quota, "blocked_dataset_ids": decontam.report.blocked_dataset_ids})
        return {"selected_count": len(selected), "blocked_dataset_ids": decontam.report.blocked_dataset_ids}

    agent.run_task("select_datasets", "Choose accepted/reviewed datasets from report according to action and score filters", select_task)

    converter = LlamaFactoryConverter(dataset_dir, name_prefix=args.name_prefix)
    for idx, item in enumerate(selected, start=1):
        def convert_dataset_task(loop: AgentLoop, item: dict[str, Any] = item, idx: int = idx) -> dict:
            nonlocal conversions, source_notes
            dataset_id = str(item.get("dataset_id"))
            rows_holder: dict[str, Any] = {}

            def load_rows_task(sub: AgentLoop) -> dict:
                print(f"[info] [{idx}/{len(selected)}] Loading rows for {dataset_id} from {args.source}")
                rows, actual_source, error = load_rows_for_item(
                    item,
                    source=args.source,
                    token=settings.hf_token,
                    trust_remote_code=args.trust_remote_code,
                    max_rows=args.max_rows,
                    endpoint=settings.hf_endpoint,
                )
                if error:
                    print(f"[warn] {dataset_id}: {error}", file=sys.stderr)
                    sub.observe("row_loading_warning", error, importance=2)
                rows_holder.update({"rows": rows, "actual_source": actual_source, "error": error})
                source_notes.append({"dataset_id": dataset_id, "source": actual_source, "rows": len(rows), "note": error})
                sub.record_tool_call(
                    "load_rows_for_item",
                    inputs={"dataset_id": dataset_id, "source": args.source, "max_rows": args.max_rows},
                    output_summary=f"loaded {len(rows)} rows from {actual_source}",
                    output={"actual_source": actual_source, "error": error},
                )
                sub.set_result(f"Loaded {len(rows)} rows", {"row_count": len(rows), "source": actual_source, "error": error})
                return {"row_count": len(rows)}

            loop.run_task("load_rows", f"Load rows for {dataset_id} from report or HF streaming", load_rows_task)
            rows = rows_holder.get("rows") or []
            if not rows:
                print(f"[warn] {dataset_id}: no rows available; skipping", file=sys.stderr)
                skipped_datasets.append({"dataset_id": dataset_id, "reason": "no_rows_available"})
                loop.set_result("Skipped dataset because no rows were available", {"dataset_id": dataset_id})
                return {"dataset_id": dataset_id, "skipped": True}

            row_findings = decontam.inspect_rows(dataset_id, rows)
            decontam.write(decontam_report_path)
            if any(f.blocked for f in row_findings):
                print(f"[warn] {dataset_id}: blocked by decontamination overlap", file=sys.stderr)
                skipped_datasets.append({"dataset_id": dataset_id, "reason": "decontamination_row_overlap", "findings": [f.to_dict() for f in row_findings]})
                loop.set_result("Skipped dataset because it overlaps the target eval set", {"dataset_id": dataset_id, "findings": [f.to_dict() for f in row_findings]})
                return {"dataset_id": dataset_id, "skipped": True, "reason": "decontamination_row_overlap"}

            def convert_rows_task(sub: AgentLoop) -> dict:
                result = converter.convert_rows(
                    dataset_id=dataset_id,
                    rows=rows,
                    format_type=_format_type(item),
                    training_types=_training_types(item),
                    requested_stage=args.stage,
                    max_rows=args.max_rows,
                )
                conversions.append(result)
                print(
                    f"[info] Converted {dataset_id} -> {result.dataset_name} | stage={result.stage} | "
                    f"format={result.formatting} | rows={result.rows_written} | skipped={result.rows_skipped}"
                )
                context.add_event(
                    "convert",
                    result.dataset_name,
                    f"{result.rows_written} rows written, {result.rows_skipped} skipped, stage={result.stage}",
                    {"dataset_id": dataset_id, "data_file": result.data_file, "warnings": result.warnings},
                )
                if result.rows_written > 0 and result.data_file:
                    sub.add_artifact(result.data_file, f"llamafactory_data:{result.stage}", f"Converted JSONL for {dataset_id}")
                elif result.rows_written <= 0:
                    skipped_datasets.append({"dataset_id": dataset_id, "reason": "zero_converted_rows", "warnings": result.warnings, "sample_keys": result.sample_keys})
                sub.set_result(
                    "Converted rows to LLaMA-Factory local data",
                    {
                        "dataset_id": dataset_id,
                        "dataset_name": result.dataset_name,
                        "stage": result.stage,
                        "formatting": result.formatting,
                        "rows_written": result.rows_written,
                        "rows_skipped": result.rows_skipped,
                        "rows_seen": result.rows_seen,
                        "sample_keys": result.sample_keys,
                        "warnings": result.warnings,
                    },
                )
                return {"conversion": to_jsonable(result)}

            convert_result = loop.run_task("convert_rows", f"Convert {dataset_id} rows to LLaMA-Factory format", convert_rows_task)
            loop.set_result(
                f"Dataset conversion complete: {dataset_id}",
                {"dataset_id": dataset_id, "conversion_status": convert_result.status},
            )
            return {"dataset_id": dataset_id, "conversion_status": convert_result.status}

        result = agent.run_task(
            f"prepare_dataset:{item.get('dataset_id')}",
            f"Load and convert dataset {item.get('dataset_id')} into LLaMA-Factory local data",
            convert_dataset_task,
            inputs={"dataset_id": item.get("dataset_id"), "index": idx, "total": len(selected)},
            task_type="dataset_conversion",
            raise_on_error=False,
        )
        if not result.ok:
            print(f"[warn] Prepare task failed for {item.get('dataset_id')}: {result.error}", file=sys.stderr)

    def quality_gate_task(loop: AgentLoop) -> dict:
        attempted = len([x for x in conversions if x.rows_seen > 0])
        zero = len([x for x in conversions if x.rows_seen > 0 and x.rows_written <= 0])
        frac = (zero / attempted) if attempted else 0.0
        report_data = decontam.report.to_dict()
        decontam.write(decontam_report_path)
        loop.add_artifact(decontam_report_path, "decontamination_report", "Benchmark contamination screening report")
        loop.set_result("Preparation quality gates checked", {"attempted_conversions": attempted, "zero_row_conversions": zero, "zero_row_fraction": frac, "decontamination": report_data})
        usable = len([x for x in conversions if x.rows_written > 0])
        if not selected and skipped_datasets:
            raise RuntimeError("all candidate datasets were skipped by decontamination or row loading")
        if selected and usable == 0:
            raise RuntimeError("no selected datasets produced usable rows after decontamination/conversion")
        if attempted and args.max_zero_row_fraction <= 1.0 and frac > args.max_zero_row_fraction:
            raise RuntimeError(f"zero-row conversion fraction {frac:.2f} exceeds --max-zero-row-fraction={args.max_zero_row_fraction}")
        return {"ok": True, "usable_conversions": usable, "zero_row_fraction": frac, "decontamination_ok": decontam.report.ok}

    quality_result = agent.run_task("quality_gates", "Validate decontamination and conversion quality before writing train configs", quality_gate_task, raise_on_error=False)
    if not quality_result.ok:
        print(f"[error] Preparation quality gates failed: {quality_result.error}", file=sys.stderr)
        context.save()
        return 5

    def dataset_info_task(loop: AgentLoop) -> dict:
        nonlocal dataset_info_path
        dataset_info_path = write_dataset_info(dataset_dir, conversions, merge_existing=args.merge_dataset_info)
        context.add_artifact(dataset_info_path, "dataset_info", "LLaMA-Factory dataset_info.json")
        loop.add_artifact(dataset_info_path, "dataset_info", "LLaMA-Factory dataset_info.json")
        loop.set_result("Wrote dataset_info.json", {"path": str(dataset_info_path), "conversion_count": len(conversions)})
        return {"dataset_info": str(dataset_info_path)}

    agent.run_task("write_dataset_info", "Write LLaMA-Factory dataset_info.json for converted datasets", dataset_info_task)

    def config_task(loop: AgentLoop) -> dict:
        nonlocal yaml_paths
        yaml_paths = generate_training_configs(
            conversions,
            config_dir=config_dir,
            dataset_dir=dataset_dir,
            model_name_or_path=args.base_model,
            template=args.template,
            output_root=args.train_output_root,
            finetuning_type=args.finetuning_type,
            cutoff_len=args.cutoff_len,
            max_samples=args.yaml_max_samples,
            report_to=args.report_to,
        )
        for stage, path in yaml_paths.items():
            context.add_artifact(path, f"train_yaml:{stage}", f"LLaMA-Factory {stage} training config")
            loop.add_artifact(path, f"train_yaml:{stage}", f"LLaMA-Factory {stage} training config")
        loop.set_result("Generated LLaMA-Factory training YAMLs", {"configs": {stage: str(path) for stage, path in yaml_paths.items()}})
        return {"configs": {stage: str(path) for stage, path in yaml_paths.items()}}

    agent.run_task("generate_training_configs", "Generate train_sft/train_dpo/train_pt YAML files as needed", config_task)

    def manifest_task(loop: AgentLoop) -> dict:
        nonlocal manifest_path
        manifest = {
            "report": str(report_path),
            "output_dir": str(output_dir),
            "dataset_dir": str(dataset_dir),
            "dataset_info": str(dataset_info_path) if dataset_info_path else None,
            "configs": {stage: str(path) for stage, path in yaml_paths.items()},
            "source_notes": source_notes,
            "skipped_datasets": skipped_datasets,
            "decontamination_report": str(decontam_report_path),
            "decontamination": decontam.report.to_dict(),
            "conversions": [
                {
                    "dataset_id": r.dataset_id,
                    "dataset_name": r.dataset_name,
                    "stage": r.stage,
                    "formatting": r.formatting,
                    "data_file": r.data_file,
                    "rows_written": r.rows_written,
                    "rows_skipped": r.rows_skipped,
                    "rows_seen": r.rows_seen,
                    "sample_keys": r.sample_keys,
                    "warnings": r.warnings,
                }
                for r in conversions
            ],
        }
        manifest_path = output_dir / "prepare_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        context.add_artifact(manifest_path, "manifest", "Preparation manifest")
        loop.add_artifact(manifest_path, "manifest", "Preparation manifest")
        loop.set_result("Wrote preparation manifest", manifest)
        return manifest

    agent.run_task("write_prepare_manifest", "Write manifest describing data files, configs, and source notes", manifest_task)

    runner = BashRunner(cwd=Path.cwd(), timeout=args.bash_timeout)

    def smoke_test_task(loop: AgentLoop) -> dict:
        if not args.run_smoke_test:
            loop.set_result("Smoke test skipped", {"run_smoke_test": False})
            return {"skipped": True}
        # Keep this validation in-process. Earlier versions spawned a Python
        # subprocess here; in long agent/test sessions that can leave descendant
        # processes behind if the parent is interrupted. The real training step
        # still uses BashRunner, while the smoke test remains deterministic.
        print(f"[info] Running smoke test: validate_prepared_dataset_dir {dataset_dir}")
        errors = validate_prepared_dataset_dir(dataset_dir)
        for yaml_path in yaml_paths.values():
            errors.extend([f"{yaml_path}: {err}" for err in validate_train_yaml(yaml_path)])
        ok = not errors
        context.add_event(
            "validate",
            "smoke_test",
            f"ok={ok}",
            {"dataset_dir": str(dataset_dir), "errors": errors[:20]},
            importance=2,
        )
        loop.record_tool_call(
            "llamafactory.validate_prepared_dataset_dir",
            inputs={"dataset_dir": str(dataset_dir)},
            output_summary=f"ok={ok}, errors={len(errors)}",
            output={"errors": errors[:20]},
            importance=2,
        )
        loop.set_result("Smoke test completed", {"ok": ok, "errors": errors[:20]})
        if errors:
            for err in errors[:20]:
                print(f"[error] {err}", file=sys.stderr)
            raise RuntimeError("Smoke test failed")
        print(f"[ok] prepared dataset_dir is valid: {dataset_dir}")
        return {"ok": True}

    smoke_result = agent.run_task("run_smoke_test", "Optionally validate generated dataset files with the bash runner", smoke_test_task, task_type="bash", raise_on_error=False)
    if args.run_smoke_test and not smoke_result.ok:
        print("[error] Smoke test failed.", file=sys.stderr)
        context.save()
        return 3

    def train_task(loop: AgentLoop) -> dict:
        if not args.run_train:
            loop.set_result("Training skipped", {"run_train": False})
            return {"skipped": True}
        selected_yamls = yaml_paths
        if args.train_stage:
            selected_yamls = {args.train_stage: yaml_paths[args.train_stage]} if args.train_stage in yaml_paths else {}
        train_results = []
        for stage, yaml_path in selected_yamls.items():
            def one_stage(sub: AgentLoop, stage: str = stage, yaml_path: Path = yaml_path) -> dict:
                cmd = ["llamafactory-cli", "train", str(yaml_path)]
                print(f"[info] Running train command for stage={stage}: {' '.join(cmd)}")
                result = runner.run(cmd, timeout=args.bash_timeout)
                print(result.stdout, end="")
                if result.stderr:
                    print(result.stderr, file=sys.stderr, end="")
                context.add_event(
                    "bash",
                    f"train_{stage}",
                    f"returncode={result.returncode}, ok={result.ok}",
                    {"command": result.command, "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]},
                    importance=3,
                )
                sub.record_tool_call(
                    "bash.run",
                    inputs={"command": result.command, "cwd": result.cwd},
                    output_summary=f"returncode={result.returncode}, ok={result.ok}",
                    output={"stdout_tail": result.stdout[-4000:], "stderr_tail": result.stderr[-4000:], "timed_out": result.timed_out},
                    importance=3,
                )
                sub.set_result(f"Train command completed for {stage}", {"returncode": result.returncode, "ok": result.ok, "stage": stage})
                if not result.ok:
                    raise RuntimeError(f"Training failed for stage={stage}")
                return {"ok": result.ok, "returncode": result.returncode}

            res = loop.run_task(f"train_{stage}", f"Run llamafactory-cli train for {stage}", one_stage, task_type="bash", raise_on_error=False)
            train_results.append({"stage": stage, "ok": res.ok, "result_path": res.result_path, "error": res.error})
            if not res.ok:
                loop.set_result("One or more training stages failed", {"train_results": train_results})
                raise RuntimeError(f"Training failed for stage={stage}")
        loop.set_result("Training tasks completed", {"train_results": train_results})
        return {"train_results": train_results}

    train_result = agent.run_task("run_training", "Optionally run LLaMA-Factory train commands as nested bash tasks", train_task, task_type="bash", raise_on_error=False)
    if args.run_train and not train_result.ok:
        context.save()
        return 4

    final_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path and manifest_path.exists() else {}
    context.add_event("prepare", "done", f"Prepared {len(conversions)} datasets and {len(yaml_paths)} training configs", final_manifest, importance=3)
    agent.set_result("Preparation complete", {"conversion_count": len(conversions), "config_count": len(yaml_paths), "manifest": str(manifest_path) if manifest_path else None})
    agent.save_loop_index()
    context.save()

    print(f"[done] decontamination_report: {decontam_report_path}")
    print(f"[done] dataset_info: {dataset_info_path}")
    for stage, path in yaml_paths.items():
        print(f"[done] train yaml ({stage}): {path}")
    print(f"[done] manifest: {manifest_path}")
    print(f"[done] context: {context_path}")
    print(f"[done] agent loop: {output_dir / '.autopilot' / 'agent' / 'loop_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
