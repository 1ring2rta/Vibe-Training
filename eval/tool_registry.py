from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopilot.models import to_jsonable


@dataclass
class EvalToolSpec:
    name: str
    kind: str  # builtin | external_repo | agentic_eval
    purpose: str
    metrics: list[str] = field(default_factory=list)
    repo_url: str | None = None
    commit: str | None = None
    install_commands: list[str] = field(default_factory=list)
    run_command_template: str | None = None
    parser: str | None = None
    requirements: list[str] = field(default_factory=list)
    status: str = "candidate"
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalToolRegistry:
    def __init__(self, tools: list[EvalToolSpec] | None = None) -> None:
        self.tools: dict[str, EvalToolSpec] = {}
        for tool in tools or []:
            self.add(tool)

    @classmethod
    def default(cls) -> "EvalToolRegistry":
        return cls([
            EvalToolSpec("exact_match", "builtin", "Exact string/numeric answer checking.", metrics=["accuracy"]),
            EvalToolSpec("contains", "builtin", "Check whether the response contains an expected substring.", metrics=["accuracy"]),
            EvalToolSpec("regex", "builtin", "Regex-based response verification.", metrics=["accuracy"]),
            EvalToolSpec("python_unit_tests", "builtin", "Run small pytest/assert snippets for coding eval/RLVR verifier candidates.", metrics=["pass_rate"], requirements=["python"]),
            EvalToolSpec("llm_judge", "builtin", "Use the configured judge client to grade open-ended answers.", metrics=["judge_score"]),
            EvalToolSpec(
                "spider_test_suite_sql_eval",
                "external_repo",
                "Spider/Text-to-SQL test-suite evaluator. Clone/pin before use and parse execution/test-suite accuracy.",
                metrics=["execution_accuracy", "test_suite_accuracy"],
                repo_url="https://github.com/taoyds/test-suite-sql-eval",
                commit=None,
                install_commands=["git clone <repo_url> <tool_dir>", "pip install -r <tool_dir>/requirements.txt || true"],
                run_command_template="python <tool_dir>/evaluation.py --gold <gold> --pred <pred> --db <db_dir> --table <tables_json>",
                parser="parse_stdout_for_sql_eval_metrics",
                requirements=["git", "python", "sqlite/databases"],
            ),
            EvalToolSpec(
                "swe_bench_with_swe_agent",
                "agentic_eval",
                "Repository-level software engineering benchmark runner using SWE-bench/SWE-agent style tool-use environment.",
                metrics=["resolved_rate", "patch_success"],
                repo_url="https://github.com/SWE-bench/SWE-bench",
                commit=None,
                install_commands=["git clone <repo_url> <tool_dir>", "pip install -e <tool_dir>"],
                run_command_template="<agent_runner> --instances <instances> --model <model> --output <output_dir>",
                parser="parse_swe_bench_results_json",
                requirements=["git", "docker_or_sandbox", "agent_framework"],
            ),
        ])

    def add(self, tool: EvalToolSpec) -> None:
        self.tools[tool.name] = tool

    def select_for_goal(self, goal: str) -> list[EvalToolSpec]:
        lower = goal.lower()
        selected: list[EvalToolSpec] = []
        if any(x in lower for x in ["sql", "spider", "text-to-sql", "database", "数据库"]):
            selected.append(self.tools["spider_test_suite_sql_eval"])
        if any(x in lower for x in ["swe", "repo", "github issue", "软件工程", "代码仓库"]):
            selected.append(self.tools["swe_bench_with_swe_agent"])
        if any(x in lower for x in ["code", "coding", "python", "代码", "bug"]):
            selected.append(self.tools["python_unit_tests"])
        if not selected:
            selected.extend([self.tools["exact_match"], self.tools["contains"], self.tools["llm_judge"]])
        return selected

    def to_dict(self) -> dict[str, Any]:
        return {name: to_jsonable(spec) for name, spec in sorted(self.tools.items())}

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return p
