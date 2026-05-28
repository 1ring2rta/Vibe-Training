from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autopilot.models import to_jsonable


@dataclass
class BenchmarkSpec:
    name: str
    task_type: str
    metric: str
    min_cases: int
    early_stop_allowed: bool = True
    verifier: str | None = None
    runner: str | None = None
    source: str = "benchmark"
    install: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


class BenchmarkRegistry:
    def __init__(self, benchmarks: dict[str, BenchmarkSpec] | None = None) -> None:
        self.benchmarks = benchmarks or self.default().benchmarks

    @classmethod
    def default(cls) -> "BenchmarkRegistry":
        return cls({
            "aime24_all": BenchmarkSpec(
                name="aime24_all",
                task_type="math_exact",
                metric="exact_match_accuracy",
                min_cases=30,
                verifier="aime_integer_exact",
                notes="2024 AIME I + II, deterministic integer-answer verifier; 80% means at least 24/30.",
            ),
            "swe_bench_lite": BenchmarkSpec(
                name="swe_bench_lite",
                task_type="agentic_repo_repair",
                metric="resolved_rate",
                min_cases=10,
                runner="swebench_harness",
                install={"repo": "https://github.com/SWE-bench/SWE-bench", "pin": "pinned_commit_required"},
                notes="Agentic benchmark: checkout repos, generate/apply patches, run tests, parse resolved_rate.",
            ),
            "spider_test_suite": BenchmarkSpec(
                name="spider_test_suite",
                task_type="text_to_sql",
                metric="test_suite_accuracy",
                min_cases=50,
                runner="taoyds/test-suite-sql-eval",
                install={"repo": "https://github.com/taoyds/test-suite-sql-eval", "pin": "pinned_commit_required"},
                notes="External repo evaluator for Spider-style SQL execution/test-suite metrics.",
            ),
            "humaneval_plus": BenchmarkSpec(
                name="humaneval_plus",
                task_type="function_coding",
                metric="pass_at_1",
                min_cases=50,
                verifier="python_unit_tests",
                notes="Function-level code generation benchmark; useful before expensive SWE-bench.",
            ),
        })

    def infer(self, goal: str, target: str = "") -> list[BenchmarkSpec]:
        text = (goal + " " + target).lower()
        if "aime" in text or "math" in text or "数学" in text:
            return [self.benchmarks["aime24_all"]]
        if "swe" in text or "repo" in text or "github" in text or "仓库" in text:
            return [self.benchmarks["swe_bench_lite"]]
        if "spider" in text or "sql" in text or "text-to-sql" in text:
            return [self.benchmarks["spider_test_suite"]]
        if "code" in text or "coding" in text or "python" in text or "代码" in text:
            return [self.benchmarks["humaneval_plus"], self.benchmarks["swe_bench_lite"]]
        return []

    def to_dict(self) -> dict[str, Any]:
        return {k: v.to_dict() for k, v in self.benchmarks.items()}

