from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autopilot.config import apply_config_defaults, load_settings
from autopilot.data.classifier import classify_dataset
from autopilot.data.reviewer import DeterministicSampleReviewer, with_llm_review
from autopilot.data.scorer import score_dataset
from autopilot.hf.inspect import DatasetInspector
from autopilot.hf.search import HuggingFaceDatasetSearcher
from autopilot.llm.kimi import KimiClient
from autopilot.models import DatasetReportItem
from autopilot.reports.render import sort_report_items, write_json_report, write_markdown_report


DEFAULT_QUERY_SUFFIXES = [
    "instruction",
    "sft",
    "preference",
    "dpo",
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
    # Domain-specific expansions.
    lower = goal.lower()
    if any(term in lower for term in ["中文", "chinese", "zh", "汉语", "華語"]):
        queries.extend(["chinese instruction", "zh sft", "中文 问答"])
    if "法律" in lower or "legal" in lower or "law" in lower:
        queries.extend(["legal qa", "law instruction", "法律 问答"])
    if "代码" in lower or "code" in lower or "program" in lower:
        queries.extend(["code instruction", "python code", "code preference"])
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search, classify, review, and score Hugging Face datasets for LLM training.")
    parser.add_argument("--goal", default=None, help="Training goal, e.g. '提升中文法律问答能力'. Can also be set in config defaults.search.goal.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum unique datasets to inspect.")
    parser.add_argument("--per-query-limit", type=int, default=20, help="HF result limit per search query.")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of streaming sample rows per dataset.")
    parser.add_argument("--max-configs", type=int, default=3, help="Maximum dataset configs to try when loading samples.")
    parser.add_argument("--sort", default="downloads", help="HF search sort key. Common: downloads, likes, lastModified.")
    parser.add_argument("--output-dir", default="reports", help="Directory for JSON/Markdown reports.")
    parser.add_argument("--config", default=None, help="YAML config path. Default: ./autopilot.yaml or AUTOPILOT_CONFIG.")
    parser.add_argument("--env-file", default=None, help="Optional .env file path for backward compatibility.")
    parser.add_argument("--use-llm-queries", action="store_true", help="Use KIMI to generate HF search queries if configured.")
    parser.add_argument("--use-llm-review", action="store_true", help="Use KIMI to review small redacted sample previews if configured.")
    parser.add_argument("--llm-review-top-n", type=int, default=10, help="Only call KIMI review for the top N preliminary candidates.")
    parser.add_argument("--query", action="append", default=[], help="Extra explicit HF search query. Can be repeated.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote dataset loading code. Default is disabled for safety.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    parse_argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(parse_argv)
    settings = load_settings(config_file=args.config, env_file=args.env_file)
    apply_config_defaults(args, parser, settings, "search", parse_argv, aliases={"query": ["query", "queries"]})
    if not args.goal:
        parser.error("--goal is required unless defaults.search.goal is set in the YAML config.")
    if settings.config_path:
        print(f"[info] Loaded config: {settings.config_path}")

    kimi: KimiClient | None = None
    if args.use_llm_queries or args.use_llm_review:
        if not settings.kimi_configured:
            print("[warn] KIMI is not configured; falling back to deterministic mode.", file=sys.stderr)
        else:
            try:
                kimi = KimiClient(settings, conversation_root=Path(args.output_dir) / ".autopilot" / "conversations", session_id=f"search:{args.goal}")
            except Exception as exc:
                print(f"[warn] Could not initialize KIMI client: {exc}", file=sys.stderr)
                kimi = None

    queries: list[str] = []
    if args.use_llm_queries and kimi is not None:
        try:
            queries.extend(kimi.generate_search_queries(args.goal, max_queries=8))
        except Exception as exc:
            print(f"[warn] KIMI query generation failed: {exc}", file=sys.stderr)
    queries.extend(args.query)
    queries.extend(fallback_queries(args.goal, max_queries=8))
    # Deduplicate while preserving order.
    queries = list(dict.fromkeys(q for q in queries if q.strip()))[:12]

    print(f"[info] Search queries: {queries}")
    searcher = HuggingFaceDatasetSearcher(token=settings.hf_token, endpoint=settings.hf_endpoint)
    candidates = searcher.search_many(
        queries=queries,
        per_query_limit=args.per_query_limit,
        max_total=args.limit,
        sort=args.sort,
    )
    print(f"[info] Found {len(candidates)} unique candidate datasets.")

    inspector = DatasetInspector(token=settings.hf_token, trust_remote_code=args.trust_remote_code, endpoint=settings.hf_endpoint)
    reviewer = DeterministicSampleReviewer()
    preliminary_items: list[DatasetReportItem] = []

    for idx, candidate in enumerate(candidates, start=1):
        print(f"[info] [{idx}/{len(candidates)}] Inspecting {candidate.dataset_id}")
        inspection = inspector.inspect(
            candidate.dataset_id,
            sample_size=args.sample_size,
            search_metadata=candidate,
            max_configs=args.max_configs,
        )
        classification = classify_dataset(inspection.columns, inspection.sample_rows)
        risk = reviewer.review(inspection, goal=args.goal)
        score = score_dataset(args.goal, inspection, classification, risk)
        preliminary_items.append(
            DatasetReportItem(
                dataset_id=candidate.dataset_id,
                score=score,
                classification=classification,
                risk_assessment=risk,
                inspection=inspection,
            )
        )

    items = sort_report_items(preliminary_items)

    if args.use_llm_review and kimi is not None and items:
        reviewed: list[DatasetReportItem] = []
        for idx, item in enumerate(items, start=1):
            if idx <= args.llm_review_top_n:
                print(f"[info] KIMI reviewing {item.dataset_id}")
                try:
                    llm_review = kimi.review_dataset_samples(args.goal, item.inspection, item.classification)
                    item.risk_assessment = with_llm_review(item.risk_assessment, llm_review)
                    # Adjust score mildly with LLM review without letting it dominate.
                    if isinstance(llm_review, dict):
                        try:
                            domain = float(llm_review.get("domain_match_score", 0.5))
                            quality = float(llm_review.get("quality_score", 0.5))
                            risk_s = float(llm_review.get("risk_score", 0.5))
                            adjusted = item.score.suitability_score * 0.78 + domain * 0.10 + quality * 0.10 - risk_s * 0.05
                            item.score.suitability_score = round(max(0.0, min(1.0, adjusted)), 4)
                            item.score.components["kimi_adjustment"] = round(item.score.suitability_score - sum(v for k, v in item.score.components.items() if k != "kimi_adjustment"), 4)
                        except Exception:
                            pass
                except Exception as exc:
                    print(f"[warn] KIMI review failed for {item.dataset_id}: {exc}", file=sys.stderr)
            reviewed.append(item)
        items = sort_report_items(reviewed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "dataset_report"
    json_path = write_json_report(items, output_dir / f"{safe_name}.json")
    md_path = write_markdown_report(items, output_dir / f"{safe_name}.md", goal=args.goal)

    print(f"[done] JSON report: {json_path}")
    print(f"[done] Markdown report: {md_path}")
    print("\nTop candidates:")
    for idx, item in enumerate(items[:10], start=1):
        training = ",".join(t.value for t in item.classification.recommended_training)
        print(f"{idx:>2}. {item.dataset_id} | score={item.score.suitability_score:.3f} | {item.classification.format_type} | {training}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
