from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autopilot.agent import AgentLoop
from autopilot.config import apply_config_defaults, load_settings, validate_settings
from autopilot.context import ContextManager
from autopilot.data.classifier import classify_dataset
from autopilot.data.decider import decide_dataset_adoption, merge_llm_decision
from autopilot.data.probe import build_probe_examples, probe_model_on_examples
from autopilot.data.reviewer import DeterministicSampleReviewer, with_llm_review
from autopilot.data.scorer import score_dataset
from autopilot.hf.browse import HFDatasetBrowser, features_from_viewer_examples
from autopilot.hf.inspect import DatasetInspector
from autopilot.hf.search import HuggingFaceDatasetSearcher
from autopilot.llm.kimi import KimiClient
from autopilot.llm.vllm import VLLMClient
from autopilot.models import DatasetInspection, DatasetReportItem, DatasetSearchResult, to_jsonable
from autopilot.reports.render import sort_report_items, write_json_report, write_markdown_report
from autopilot.tools.web_search import WebSearchTool, extract_hf_dataset_ids


DEFAULT_QUERY_SUFFIXES = [
    "instruction",
    "sft",
    "preference",
    "dpo",
    "chosen rejected",
    "reward model",
    "kto feedback",
    "rlvr unit tests",
    "grpo",
    "continued pretraining",
    "raw corpus",
    "chat",
    "qa",
]


def fallback_queries(goal: str, max_queries: int = 8) -> list[str]:
    goal = goal.strip()
    queries: list[str] = []
    if goal:
        queries.append(goal)
        for suffix in DEFAULT_QUERY_SUFFIXES:
            queries.append(f"{goal} {suffix}")
    lower = goal.lower()
    if any(term in lower for term in ["中文", "chinese", "zh", "汉语", "華語"]):
        queries.extend(["chinese instruction", "zh sft", "中文 问答"])
    if "法律" in lower or "legal" in lower or "law" in lower:
        queries.extend(["legal qa", "law instruction", "法律 问答"])
    if "代码" in lower or "code" in lower or "program" in lower or "coding" in lower:
        queries.extend([
            "code instruction",
            "python code instruction",
            "code preference dpo",
            "programming chosen rejected preference",
            "code reward model preference",
            "code KTO feedback",
            "code unit tests RLVR",
            "programming problems unit tests",
            "HumanEval MBPP code tests",
            "raw code corpus continued pretraining",
            "github code corpus",
            "StarCoderData",
        ])
    if "数学" in lower or "math" in lower:
        queries.extend(["math reasoning", "gsm8k", "math preference"])

    dedup: list[str] = []
    for q in queries:
        q = q.strip()
        if q and q not in dedup:
            dedup.append(q)
        if len(dedup) >= max_queries:
            break
    return dedup


def _inspection_from_overview_if_needed(inspection: DatasetInspection, overview) -> DatasetInspection:
    if inspection.sample_rows:
        inspection.web_overview = overview
        return inspection
    if overview and overview.example_rows:
        columns, features = features_from_viewer_examples(overview.example_rows)
        return DatasetInspection(
            dataset_id=inspection.dataset_id,
            config_name=overview.selected_config,
            split=overview.selected_split,
            columns=columns,
            features=features,
            sample_rows=overview.example_rows,
            metadata=inspection.metadata,
            load_error=inspection.load_error,
            web_overview=overview,
        )
    inspection.web_overview = overview
    return inspection



def _make_hf_searcher(settings):
    try:
        return HuggingFaceDatasetSearcher(token=settings.hf_token, endpoint=settings.hf_endpoint)
    except TypeError:  # test doubles / older implementations
        return HuggingFaceDatasetSearcher(token=settings.hf_token)


def _make_hf_browser(settings):
    try:
        return HFDatasetBrowser(token=settings.hf_token, endpoint=settings.hf_endpoint)
    except TypeError:  # test doubles / older implementations
        return HFDatasetBrowser(token=settings.hf_token)


def _make_dataset_inspector(settings, trust_remote_code: bool):
    try:
        return DatasetInspector(token=settings.hf_token, trust_remote_code=trust_remote_code, endpoint=settings.hf_endpoint)
    except TypeError:  # test doubles / older implementations
        return DatasetInspector(token=settings.hf_token, trust_remote_code=trust_remote_code)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect HF datasets by browsing overview/examples/files, probing a vLLM endpoint, and deciding adoption/training type."
    )
    parser.add_argument("--goal", default=None, help="Training goal, e.g. '提升中文法律问答能力'. Can also be set in config defaults.collect.goal.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum unique datasets to inspect.")
    parser.add_argument("--per-query-limit", type=int, default=20, help="HF result limit per search query.")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of sample rows to fetch/browse per dataset.")
    parser.add_argument("--max-configs", type=int, default=3, help="Maximum dataset configs to try when streaming fallback is used.")
    parser.add_argument("--max-files", type=int, default=60, help="Maximum visible repository files to list per dataset.")
    parser.add_argument("--sort", default="downloads", help="HF search sort key. Common: downloads, likes, lastModified.")
    parser.add_argument("--output-dir", default="reports", help="Directory for JSON/Markdown reports.")
    parser.add_argument("--config", default=None, help="YAML config path. Default: ./autopilot.yaml or AUTOPILOT_CONFIG.")
    parser.add_argument("--env-file", default=None, help="Optional .env file path for backward compatibility.")
    parser.add_argument("--use-llm-queries", action="store_true", help="Use KIMI to generate HF search queries if configured.")
    parser.add_argument("--use-web-search", action="store_true", help="Use generic web search to discover Hugging Face dataset pages before HF API search.")
    parser.add_argument("--skip-hf-search", action="store_true", help="Skip Hugging Face Hub list_datasets search. Useful when HF is blocked but Serper/Brave/Tavily can discover dataset URLs.")
    parser.add_argument("--web-search-limit", type=int, default=30, help="Maximum web-search results to inspect for HF dataset URLs.")
    parser.add_argument("--web-search-per-query", type=int, default=8, help="Web-search result limit per query.")
    parser.add_argument("--use-llm-decision", action="store_true", help="Use KIMI to make/adjust adoption decisions if configured.")
    parser.add_argument("--use-llm-review", action="store_true", help="Compatibility option: attach KIMI sample review to report.")
    parser.add_argument("--llm-review-top-n", type=int, default=10, help="Only call KIMI review/decision for the top N preliminary candidates.")
    parser.add_argument("--test-vllm", action="store_true", help="Probe VLLM_BASE_URL/VLLM_MODEL on extracted prompts from each dataset.")
    parser.add_argument("--max-probe-samples", type=int, default=3, help="Maximum extracted prompts to send to vLLM per dataset.")
    parser.add_argument("--query", action="append", default=[], help="Extra explicit HF search query. Can be repeated.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote dataset loading code for streaming fallback. Default disabled.")
    parser.add_argument("--context-state", default=None, help="Optional context state path. Default: <output-dir>/.autopilot/context/session.json")
    parser.add_argument("--agent-max-iterations", type=int, default=256, help="Maximum events per agent loop before it raises.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    parse_argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(parse_argv)
    settings = load_settings(config_file=args.config, env_file=args.env_file)
    apply_config_defaults(args, parser, settings, "collect", parse_argv, aliases={"query": ["query", "queries"]})
    if not args.goal:
        parser.error("--goal is required unless defaults.collect.goal is set in the YAML config.")
    if settings.config_path:
        print(f"[info] Loaded config: {settings.config_path}")
    for warning in validate_settings(settings):
        print(f"[warn] Config: {warning}", file=sys.stderr)
    output_dir = Path(args.output_dir)
    context_path = Path(args.context_state) if args.context_state else output_dir / ".autopilot" / "context" / "session.json"
    context = ContextManager(context_path, project_root=Path.cwd())
    agent = AgentLoop.root(
        name="collect",
        objective=f"Collect candidate datasets for: {args.goal}",
        context=context,
        workspace_dir=output_dir / ".autopilot" / "agent",
        max_iterations=args.agent_max_iterations,
    )
    context.add_event("collect", "start", f"Collecting datasets for goal: {args.goal}", {"args": vars(args)}, importance=2)

    kimi: KimiClient | None = None
    vllm: VLLMClient | None = None
    queries: list[str] = []
    web_dataset_ids: list[str] = []
    candidates: list[DatasetSearchResult] = []
    items: list[DatasetReportItem] = []
    json_path: Path | None = None
    md_path: Path | None = None

    def initialize_runtime(loop: AgentLoop) -> dict:
        nonlocal kimi, vllm
        kimi_enabled = False
        vllm_enabled = False
        if args.use_llm_queries or args.use_llm_decision or args.use_llm_review:
            if not settings.kimi_configured:
                print("[warn] KIMI is not configured; LLM query/decision steps will be skipped.", file=sys.stderr)
                loop.observe("kimi_unconfigured", "KIMI requested but not configured")
            else:
                try:
                    kimi = KimiClient(settings, conversation_root=output_dir / ".autopilot" / "conversations", session_id=f"collect:{args.goal}")
                    kimi_enabled = True
                    loop.record_tool_call("kimi_client", inputs={"base_url": settings.kimi_base_url, "model": settings.kimi_model}, output_summary="KIMI client initialized")
                except Exception as exc:
                    print(f"[warn] Could not initialize KIMI client: {exc}", file=sys.stderr)
                    loop.observe("kimi_init_failed", str(exc), importance=2)

        if args.test_vllm:
            if not settings.vllm_configured:
                print("[warn] VLLM_BASE_URL/VLLM_MODEL not configured; vLLM probe will be skipped.", file=sys.stderr)
                loop.observe("vllm_unconfigured", "vLLM probe requested but not configured")
            else:
                try:
                    vllm = VLLMClient.from_settings(settings)
                    vllm_enabled = True
                    loop.record_tool_call("vllm_client", inputs={"base_url": settings.vllm_base_url, "model": settings.vllm_model}, output_summary="vLLM client initialized")
                except Exception as exc:
                    print(f"[warn] Could not initialize vLLM client: {exc}", file=sys.stderr)
                    loop.observe("vllm_init_failed", str(exc), importance=2)
        loop.set_result("Runtime clients initialized", {"kimi_enabled": kimi_enabled, "vllm_enabled": vllm_enabled})
        return {"kimi_enabled": kimi_enabled, "vllm_enabled": vllm_enabled}

    agent.run_task("initialize_runtime", "Initialize KIMI and vLLM clients if requested", initialize_runtime)

    def generate_queries(loop: AgentLoop) -> dict:
        nonlocal queries
        generated: list[str] = []
        if args.use_llm_queries and kimi is not None:
            try:
                generated = kimi.generate_search_queries(args.goal, max_queries=8)
                loop.record_tool_call("kimi.generate_search_queries", inputs={"goal": args.goal}, output_summary=f"Generated {len(generated)} LLM queries", output={"queries": generated})
            except Exception as exc:
                print(f"[warn] KIMI query generation failed: {exc}", file=sys.stderr)
                loop.observe("kimi_query_failed", str(exc), importance=2)
        queries.extend(generated)
        queries.extend(args.query)
        fallback = fallback_queries(args.goal, max_queries=8)
        queries.extend(fallback)
        queries = list(dict.fromkeys(q for q in queries if q.strip()))[:24]
        print(f"[info] Search queries: {queries}")
        context.add_event("collect", "search_queries", f"Generated {len(queries)} search queries", {"queries": queries})
        loop.set_result(f"Generated {len(queries)} search queries", {"queries": queries, "fallback_queries": fallback, "llm_queries": generated})
        return {"queries": queries}

    agent.run_task("generate_search_queries", "Generate HF/web search queries from the user goal", generate_queries)

    def run_web_search(loop: AgentLoop) -> dict:
        nonlocal web_dataset_ids
        if not args.use_web_search:
            loop.set_result("Web search disabled", {"dataset_ids": []})
            return {"dataset_ids": []}
        try:
            web_tool = WebSearchTool(settings)
            web_hits = []
            for query in queries[:12]:
                web_query = query if "site:huggingface.co" in query else f"site:huggingface.co/datasets {query}"
                hits = web_tool.search(web_query, limit=args.web_search_per_query)
                web_hits.extend(hits)
                loop.record_tool_call("web_search.search", inputs={"query": web_query, "limit": args.web_search_per_query}, output_summary=f"{len(hits)} hits")
                if len(web_hits) >= args.web_search_limit:
                    break
            web_dataset_ids = extract_hf_dataset_ids(web_hits[: args.web_search_limit])
            print(f"[info] Web search found {len(web_hits[: args.web_search_limit])} hits and {len(web_dataset_ids)} HF dataset ids.")
            context.add_event("web_search", "hf_dataset_ids", f"Found {len(web_dataset_ids)} HF dataset ids from web search", {"dataset_ids": web_dataset_ids})
            loop.set_result(f"Found {len(web_dataset_ids)} HF dataset ids from web search", {"dataset_ids": web_dataset_ids, "hit_count": len(web_hits[: args.web_search_limit])})
        except Exception as exc:
            print(f"[warn] Web search failed and will be skipped: {exc}", file=sys.stderr)
            loop.observe("web_search_failed", str(exc), importance=2)
            web_dataset_ids = []
        return {"dataset_ids": web_dataset_ids}

    agent.run_task("web_search_hf_dataset_pages", "Use general web search to discover HF dataset pages", run_web_search)

    def hf_search(loop: AgentLoop) -> dict:
        nonlocal candidates
        candidates = []
        hf_errors: list[str] = []
        if args.skip_hf_search:
            loop.observe("hf_search_skipped", "--skip-hf-search was set; using web-discovered dataset IDs only.")
        else:
            try:
                searcher = _make_hf_searcher(settings)
                candidates = searcher.search_many(
                    queries=queries,
                    per_query_limit=args.per_query_limit,
                    max_total=args.limit,
                    sort=args.sort,
                )
                hf_errors = getattr(searcher, "last_errors", []) or []
                loop.record_tool_call(
                    "huggingface_hub.search_many",
                    inputs={"queries": queries, "per_query_limit": args.per_query_limit, "max_total": args.limit, "sort": args.sort, "endpoint": settings.hf_endpoint},
                    output_summary=f"{len(candidates)} candidates from HF API",
                    output={"dataset_ids": [c.dataset_id for c in candidates], "errors": hf_errors[:5]},
                )
                if hf_errors and not candidates:
                    print(f"[warn] HF search returned no results after {len(hf_errors)} query errors; falling back to web-discovered dataset ids.", file=sys.stderr)
            except Exception as exc:
                hf_errors = [f"{type(exc).__name__}: {exc}"]
                print(f"[warn] HF search failed and will be skipped: {exc}", file=sys.stderr)
                loop.observe("hf_search_failed", str(exc), importance=2)
        existing_ids = {c.dataset_id for c in candidates}
        for dataset_id in web_dataset_ids:
            if dataset_id not in existing_ids and len(candidates) < args.limit:
                candidates.append(DatasetSearchResult(dataset_id=dataset_id, source="web_search"))
                existing_ids.add(dataset_id)
        print(f"[info] Found {len(candidates)} unique candidate datasets.")
        context.add_event("collect", "candidates", f"Found {len(candidates)} unique candidate datasets", {"dataset_ids": [c.dataset_id for c in candidates], "hf_errors": hf_errors[:5]})
        loop.set_result(f"Found {len(candidates)} unique candidate datasets", {"dataset_ids": [c.dataset_id for c in candidates], "hf_errors": hf_errors[:5]})
        return {"candidate_count": len(candidates), "hf_error_count": len(hf_errors)}

    agent.run_task("hf_search_candidates", "Search Hugging Face Hub and merge web-discovered dataset ids", hf_search)

    browser = _make_hf_browser(settings)
    inspector = _make_dataset_inspector(settings, args.trust_remote_code)
    reviewer = DeterministicSampleReviewer()

    for idx, candidate in enumerate(candidates, start=1):
        def process_dataset(loop: AgentLoop, candidate: DatasetSearchResult = candidate, idx: int = idx) -> dict:
            nonlocal items
            overview_holder: dict[str, object] = {}
            inspection_holder: dict[str, DatasetInspection] = {}

            print(f"[info] [{idx}/{len(candidates)}] Browsing {candidate.dataset_id}")

            def browse_task(sub: AgentLoop) -> dict:
                overview = browser.browse(candidate.dataset_id, sample_size=args.sample_size, max_files=args.max_files)
                overview_holder["overview"] = overview
                sub.record_tool_call(
                    "hf_dataset_browser.browse",
                    inputs={"dataset_id": candidate.dataset_id, "sample_size": args.sample_size, "max_files": args.max_files},
                    output_summary=f"files={len(overview.files)}, examples={len(overview.example_rows)}, error={overview.browse_error}",
                    output={"selected_config": overview.selected_config, "selected_split": overview.selected_split, "files": [f.path for f in overview.files[:20]]},
                )
                sub.set_result("Browsed dataset card/files/examples", {"example_count": len(overview.example_rows), "file_count": len(overview.files), "browse_error": overview.browse_error})
                return {"overview": to_jsonable(overview)}

            loop.run_task("browse_dataset", f"Browse card, examples, and repo files for {candidate.dataset_id}", browse_task)
            overview = overview_holder.get("overview")

            print(f"[info] [{idx}/{len(candidates)}] Inspecting examples/schema {candidate.dataset_id}")

            def inspect_task(sub: AgentLoop) -> dict:
                inspection = inspector.inspect(
                    candidate.dataset_id,
                    sample_size=args.sample_size,
                    search_metadata=candidate,
                    max_configs=args.max_configs,
                )
                inspection = _inspection_from_overview_if_needed(inspection, overview)
                inspection_holder["inspection"] = inspection
                sub.record_tool_call(
                    "dataset_inspector.inspect",
                    inputs={"dataset_id": candidate.dataset_id, "sample_size": args.sample_size, "max_configs": args.max_configs},
                    output_summary=f"columns={inspection.columns}, rows={len(inspection.sample_rows)}, load_error={inspection.load_error}",
                    output={"columns": inspection.columns, "features": inspection.features, "row_count": len(inspection.sample_rows)},
                )
                sub.set_result("Inspected schema and sample rows", {"columns": inspection.columns, "sample_count": len(inspection.sample_rows), "load_error": inspection.load_error})
                return {"inspection": to_jsonable(inspection)}

            loop.run_task("inspect_schema_and_samples", f"Inspect streamable schema and rows for {candidate.dataset_id}", inspect_task)
            inspection = inspection_holder["inspection"]

            analysis_holder: dict[str, object] = {}

            def classify_task(sub: AgentLoop) -> dict:
                classification = classify_dataset(inspection.columns, inspection.sample_rows)
                quality = reviewer.review(inspection, goal=args.goal)
                score = score_dataset(args.goal, inspection, classification, quality)
                analysis_holder.update({"classification": classification, "quality": quality, "score": score})
                sub.decide(
                    "dataset_training_type",
                    f"format={classification.format_type}; training={[t.value for t in classification.recommended_training]}",
                    {"confidence": classification.confidence, "reasons": classification.reasons, "risks": classification.risks},
                )
                sub.set_result(
                    "Classified dataset and computed preliminary score",
                    {"format_type": classification.format_type, "recommended_training": [t.value for t in classification.recommended_training], "score": score.suitability_score},
                )
                return {"classification": to_jsonable(classification), "score": to_jsonable(score), "quality": to_jsonable(quality)}

            loop.run_task("classify_and_score_dataset", f"Classify data format and training fit for {candidate.dataset_id}", classify_task)
            classification = analysis_holder["classification"]
            quality = analysis_holder["quality"]
            score = analysis_holder["score"]

            trials = []

            def probe_task(sub: AgentLoop) -> dict:
                nonlocal trials
                if vllm is None:
                    sub.set_result("vLLM probe skipped", {"reason": "vllm_not_configured"})
                    return {"trials": []}
                examples = build_probe_examples(inspection.sample_rows, max_examples=args.max_probe_samples)
                if examples:
                    print(f"[info] [{idx}/{len(candidates)}] Probing vLLM on {len(examples)} prompts")
                    trials = probe_model_on_examples(vllm, examples)
                    sub.record_tool_call(
                        "vllm.chat_completions",
                        inputs={"example_count": len(examples), "model": settings.vllm_model},
                        output_summary=f"Generated {len(trials)} model trial responses",
                        output={"errors": [t.error for t in trials if t.error]},
                    )
                    sub.set_result("Probed local vLLM model on dataset prompts", {"trial_count": len(trials)})
                else:
                    print(f"[info] [{idx}/{len(candidates)}] No prompt-like examples for vLLM probe")
                    sub.set_result("No prompt-like examples for vLLM probe", {"trial_count": 0})
                return {"trials": to_jsonable(trials)}

            loop.run_task("probe_vllm_on_samples", f"Ask local model to answer sample prompts from {candidate.dataset_id}", probe_task)

            decision_holder = {}

            def decide_task(sub: AgentLoop) -> dict:
                decision = decide_dataset_adoption(args.goal, inspection, classification, overview=overview, trials=trials)
                decision_holder["decision"] = decision
                sub.decide(
                    "adoption_decision",
                    f"action={decision.action}; final={decision.final_score:.3f}; training={[t.value for t in decision.training_types]}",
                    {"reasons": decision.reasons, "notes": decision.notes},
                )
                sub.set_result("Made deterministic adoption decision", {"decision": to_jsonable(decision)})
                return {"decision": to_jsonable(decision)}

            loop.run_task("decide_dataset_adoption", f"Decide whether to adopt and how to train on {candidate.dataset_id}", decide_task)
            decision = decision_holder["decision"]

            item = DatasetReportItem(
                dataset_id=candidate.dataset_id,
                score=score,
                classification=classification,
                risk_assessment=quality,
                inspection=inspection,
                web_overview=overview,
                model_trials=trials,
                adoption_decision=decision,
            )
            items.append(item)
            context.add_event(
                "dataset",
                candidate.dataset_id,
                f"format={classification.format_type}, decision={(decision.action if decision else 'review')}, final={(decision.final_score if decision else score.suitability_score):.3f}",
                {"columns": inspection.columns, "training": [t.value for t in (decision.training_types if decision else classification.recommended_training)]},
            )
            loop.set_result(
                f"Dataset processed: {candidate.dataset_id}",
                {
                    "dataset_id": candidate.dataset_id,
                    "format_type": classification.format_type,
                    "decision": decision.action if decision else "review",
                    "final_score": decision.final_score if decision else score.suitability_score,
                },
            )
            return {"dataset_id": candidate.dataset_id}

        result = agent.run_task(
            f"dataset:{candidate.dataset_id}",
            f"Browse, inspect, classify, probe, and decide adoption for {candidate.dataset_id}",
            process_dataset,
            inputs={"dataset_id": candidate.dataset_id, "index": idx, "total": len(candidates)},
            task_type="dataset",
            raise_on_error=False,
        )
        if not result.ok:
            print(f"[warn] Dataset task failed for {candidate.dataset_id}: {result.error}", file=sys.stderr)

    items = sort_report_items(items)

    def llm_review_task(loop: AgentLoop) -> dict:
        nonlocal items
        if not ((args.use_llm_decision or args.use_llm_review) and kimi is not None and items):
            loop.set_result("LLM review/decision skipped", {"reason": "not_requested_or_unavailable"})
            return {"reviewed_count": 0}
        reviewed: list[DatasetReportItem] = []
        for idx, item in enumerate(items, start=1):
            if idx <= args.llm_review_top_n:
                if args.use_llm_decision:
                    print(f"[info] KIMI deciding adoption for {item.dataset_id}")
                    try:
                        llm_decision = kimi.decide_dataset_adoption(args.goal, item.web_overview, item.inspection, item.classification, item.model_trials)
                        if item.adoption_decision:
                            item.adoption_decision = merge_llm_decision(item.adoption_decision, llm_decision)
                        loop.record_tool_call("kimi.decide_dataset_adoption", inputs={"dataset_id": item.dataset_id}, output_summary="Merged LLM adoption decision", output=llm_decision)
                    except Exception as exc:
                        print(f"[warn] KIMI adoption decision failed for {item.dataset_id}: {exc}", file=sys.stderr)
                        loop.observe(f"kimi_decision_failed:{item.dataset_id}", str(exc), importance=2)
                if args.use_llm_review:
                    print(f"[info] KIMI reviewing samples for {item.dataset_id}")
                    try:
                        llm_review = kimi.review_dataset_samples(args.goal, item.inspection, item.classification)
                        item.risk_assessment = with_llm_review(item.risk_assessment, llm_review)
                        loop.record_tool_call("kimi.review_dataset_samples", inputs={"dataset_id": item.dataset_id}, output_summary="Attached LLM sample review", output=llm_review)
                    except Exception as exc:
                        print(f"[warn] KIMI sample review failed for {item.dataset_id}: {exc}", file=sys.stderr)
                        loop.observe(f"kimi_sample_review_failed:{item.dataset_id}", str(exc), importance=2)
            reviewed.append(item)
        items = sort_report_items(reviewed)
        loop.set_result(f"LLM reviewed top {min(args.llm_review_top_n, len(items))} candidates", {"item_count": len(items)})
        return {"item_count": len(items)}

    agent.run_task("llm_review_and_rerank", "Optionally ask KIMI to refine top candidate decisions", llm_review_task)

    def write_reports(loop: AgentLoop) -> dict:
        nonlocal json_path, md_path
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = write_json_report(items, output_dir / "collection_report.json")
        md_path = write_markdown_report(items, output_dir / "collection_report.md", goal=args.goal)
        context.add_artifact(json_path, "collection_report_json", "Dataset collection JSON report")
        context.add_artifact(md_path, "collection_report_markdown", "Dataset collection Markdown report")
        context.add_event("collect", "done", f"Wrote reports with {len(items)} items", {"json": str(json_path), "markdown": str(md_path)}, importance=3)
        loop.add_artifact(json_path, "collection_report_json", "Dataset collection JSON report")
        loop.add_artifact(md_path, "collection_report_markdown", "Dataset collection Markdown report")
        loop.set_result(f"Wrote reports with {len(items)} items", {"json": str(json_path), "markdown": str(md_path), "item_count": len(items)})
        return {"json": str(json_path), "markdown": str(md_path), "item_count": len(items)}

    agent.run_task("write_collection_reports", "Write JSON/Markdown reports and attach them to context", write_reports)

    agent.set_result(
        f"Collection complete with {len(items)} candidate datasets",
        {"item_count": len(items), "json": str(json_path) if json_path else None, "markdown": str(md_path) if md_path else None},
    )
    agent.save_loop_index()
    context.save()

    print(f"[done] JSON report: {json_path}")
    print(f"[done] Markdown report: {md_path}")
    print(f"[done] context: {context_path}")
    print(f"[done] agent loop: {output_dir / '.autopilot' / 'agent' / 'loop_index.json'}")
    print("\nTop candidates:")
    for idx, item in enumerate(items[:10], start=1):
        decision = item.adoption_decision
        final = decision.final_score if decision else item.score.suitability_score
        action = decision.action if decision else "review"
        training = ",".join(t.value for t in (decision.training_types if decision else item.classification.recommended_training))
        print(f"{idx:>2}. {item.dataset_id} | final={final:.3f} | action={action} | {item.classification.format_type} | {training}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
