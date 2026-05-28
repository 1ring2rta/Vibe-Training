from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from autopilot.eval.runner import EvaluationResult
from autopilot.models import to_jsonable


@dataclass
class FailureDiagnosis:
    weak_tags: list[dict[str, Any]] = field(default_factory=list)
    failure_count: int = 0
    recommendations: list[str] = field(default_factory=list)
    data_search_queries: list[str] = field(default_factory=list)
    suggested_training: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def diagnose_failures(goal: str, eval_result: EvaluationResult | None) -> FailureDiagnosis:
    if eval_result is None or not eval_result.case_results:
        return FailureDiagnosis(
            recommendations=["No evaluation result yet; first create or load an eval set, then run baseline evaluation."],
            data_search_queries=[goal, f"{goal} instruction data", f"{goal} benchmark"],
            suggested_training=["sft"],
        )
    failures = eval_result.failures
    counter: Counter[str] = Counter()
    for failure in failures:
        for tag in failure.tags or ["untagged"]:
            counter[tag] += 1
    weak_tags = [{"tag": tag, "count": count} for tag, count in counter.most_common()]
    queries: list[str] = []
    for tag, _ in counter.most_common(5):
        queries.append(f"{goal} {tag} instruction")
        queries.append(f"{goal} {tag} benchmark")
    if not queries:
        queries = [f"{goal} instruction", f"{goal} preference", f"{goal} evaluation"]

    suggested = ["sft"]
    if failures and eval_result.score is not None:
        suggested.append("dpo")
    if any(tag in {"coding", "tests", "math", "unit_test", "verifiable"} for tag in counter):
        suggested.append("rlvr")
    recommendations = []
    if failures:
        recommendations.append("Generate or collect more data concentrated on the weak tags.")
        recommendations.append("Ask KIMI to generate targeted evaluation samples for the failure clusters.")
        if "rlvr" in suggested:
            recommendations.append("Attach executable/verifiable rewards before RLVR; do not rely only on subjective judge scores.")
    else:
        recommendations.append("No failed cases found; verify the eval set is strong enough before accepting the checkpoint.")
    return FailureDiagnosis(
        weak_tags=weak_tags,
        failure_count=len(failures),
        recommendations=recommendations,
        data_search_queries=queries,
        suggested_training=list(dict.fromkeys(suggested)),
    )
