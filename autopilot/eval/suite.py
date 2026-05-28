from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from autopilot.models import to_jsonable
from autopilot.tools.coding_sandbox import PythonSandboxTool


@dataclass
class EvalCase:
    """One model test case.

    Metrics supported by the first implementation:
    - exact_match: normalized full-string match against `expected`/`reference_answer`;
    - contains: response must contain `expected`;
    - llm_judge: KIMI/frontier judge scores the answer, with deterministic fallback;
    - python_unit_tests: append `tests` to the model's Python code and run it in the sandbox.
    """

    id: str
    prompt: str
    metric: str = "llm_judge"
    expected: str | None = None
    reference_answer: str | None = None
    tests: str | None = None
    tags: list[str] = field(default_factory=list)
    weakness_area: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSuite:
    goal: str
    cases: list[EvalCase]
    name: str = "goal_eval"
    generated_by: str = "deterministic"
    notes: str = ""


@dataclass
class EvalResult:
    case_id: str
    prompt: str
    metric: str
    passed: bool
    score: float
    response: str | None = None
    expected: str | None = None
    reference_answer: str | None = None
    error: str | None = None
    judge_feedback: str = ""
    tags: list[str] = field(default_factory=list)
    weakness_area: str = "general"
    latency_seconds: float | None = None


@dataclass
class ModelEvalReport:
    suite_name: str
    goal: str
    metric_name: str
    score: float
    passed_count: int
    total_count: int
    target_value: float | None = None
    target_met: bool | None = None
    results: list[EvalResult] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    notes: str = ""


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _strip_code_fence(text: str | None) -> str:
    text = text or ""
    fenced = re.search(r"```(?:python)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _make_case(case_id: str, prompt: str, *, metric: str = "llm_judge", expected: str | None = None, reference_answer: str | None = None, tests: str | None = None, tags: list[str] | None = None, weakness_area: str = "general") -> EvalCase:
    return EvalCase(
        id=case_id,
        prompt=prompt,
        metric=metric,
        expected=expected,
        reference_answer=reference_answer,
        tests=tests,
        tags=tags or [],
        weakness_area=weakness_area,
    )


def default_eval_cases_for_goal(goal: str, max_cases: int = 8) -> list[EvalCase]:
    """Small deterministic seed eval suite used when no eval file/KIMI is available.

    This is deliberately modest. KIMI-generated evals or user-provided held-out
    tests should replace/augment these cases in real runs.
    """
    lower = goal.lower()
    cases: list[EvalCase] = []
    if any(term in lower for term in ["code", "coding", "代码", "python", "program", "算法"]):
        cases.extend(
            [
                _make_case(
                    "coding_add_two_numbers",
                    "Write a Python function `add(a, b)` that returns the sum of two numbers. Return only code.",
                    metric="python_unit_tests",
                    tests="assert add(2, 3) == 5\nassert add(-1, 1) == 0",
                    tags=["coding", "function_generation"],
                    weakness_area="basic_code_generation",
                ),
                _make_case(
                    "coding_format_duration",
                    "Write a Python function `format_elapsed(seconds)` that returns `12.34s` for values below 60 seconds and `2m03.00s` for 123 seconds. Return only code.",
                    metric="python_unit_tests",
                    tests='assert format_elapsed(12.34) == "12.34s"\nassert format_elapsed(123) == "2m03.00s"',
                    tags=["coding", "edge_cases"],
                    weakness_area="edge_case_handling",
                ),
                _make_case(
                    "coding_explain_bug",
                    "Explain why this Python code is wrong and provide the corrected line: `items = []; print(items[0])`",
                    metric="llm_judge",
                    reference_answer="The list is empty, so indexing position 0 raises IndexError. Guard against empty input or append an item before indexing.",
                    tags=["coding", "debugging"],
                    weakness_area="debugging_explanation",
                ),
            ]
        )
    elif any(term in lower for term in ["数学", "math", "reasoning", "推理"]):
        cases.extend(
            [
                _make_case("math_linear", "Solve: 3x + 5 = 20. Return only the value of x.", metric="exact_match", expected="5", tags=["math"], weakness_area="algebra"),
                _make_case("math_arithmetic", "A box has 7 red balls and 5 blue balls. How many balls are there? Return only the number.", metric="exact_match", expected="12", tags=["math"], weakness_area="arithmetic"),
            ]
        )
    elif any(term in lower for term in ["法律", "legal", "law"]):
        cases.extend(
            [
                _make_case("legal_trial_period", "劳动合同可以约定试用期多久？", metric="llm_judge", reference_answer="试用期长短应与劳动合同期限相匹配，同一用人单位与同一劳动者只能约定一次试用期。", tags=["legal", "zh"], weakness_area="domain_knowledge"),
                _make_case("legal_divorce_agreement", "离婚协议是否签字就立即解除婚姻关系？", metric="llm_judge", reference_answer="通常需要办理离婚登记后才解除婚姻关系，协议中财产等安排也需结合具体情形判断。", tags=["legal", "zh"], weakness_area="domain_knowledge"),
            ]
        )
    else:
        cases.extend(
            [
                _make_case("general_instruction", f"Answer this task carefully: {goal}", metric="llm_judge", reference_answer="The answer should satisfy the stated user goal with concrete, correct reasoning.", tags=["general"], weakness_area="goal_following"),
                _make_case("general_refusal_to_guess", "When you are unsure about a factual claim, what should you do?", metric="contains", expected="verify", tags=["general"], weakness_area="calibration"),
            ]
        )
    return cases[:max_cases]


def _case_from_dict(data: dict[str, Any], idx: int) -> EvalCase:
    return EvalCase(
        id=str(data.get("id") or data.get("name") or f"case_{idx}"),
        prompt=str(data.get("prompt") or data.get("input") or data.get("question") or ""),
        metric=str(data.get("metric") or data.get("scoring") or "llm_judge"),
        expected=data.get("expected") if data.get("expected") is not None else data.get("answer"),
        reference_answer=data.get("reference_answer") if data.get("reference_answer") is not None else data.get("reference"),
        tests=data.get("tests") if data.get("tests") is not None else data.get("unit_tests"),
        tags=[str(x) for x in (data.get("tags") or [])],
        weakness_area=str(data.get("weakness_area") or data.get("area") or "general"),
        metadata=dict(data.get("metadata") or {}),
    )


def load_eval_suite(path: str | Path, *, default_goal: str = "") -> EvalSuite:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        cases_raw = raw
        name = path.stem
        goal = default_goal
        notes = ""
    elif isinstance(raw, dict):
        cases_raw = raw.get("cases") or raw.get("evals") or raw.get("tests") or []
        name = str(raw.get("name") or path.stem)
        goal = str(raw.get("goal") or default_goal)
        notes = str(raw.get("notes") or "")
    else:
        raise ValueError(f"Eval suite must be a list or object: {path}")
    cases = [_case_from_dict(case, idx) for idx, case in enumerate(cases_raw, start=1) if isinstance(case, dict)]
    return EvalSuite(goal=goal, cases=cases, name=name, generated_by="file", notes=notes)


def write_eval_suite(suite: EvalSuite, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(suite), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def evaluate_answer(case: EvalCase, response: str | None, *, judge_client: Any | None = None, sandbox: PythonSandboxTool | None = None) -> EvalResult:
    response = response or ""
    metric = case.metric.lower().strip()
    expected = case.expected or case.reference_answer
    error: str | None = None
    feedback = ""
    score = 0.0
    passed = False

    if metric == "exact_match":
        passed = _norm(response) == _norm(expected)
        score = 1.0 if passed else 0.0
        feedback = "exact_match"
    elif metric == "contains":
        passed = bool(expected) and _norm(expected) in _norm(response)
        score = 1.0 if passed else 0.0
        feedback = "contains"
    elif metric == "python_unit_tests":
        sandbox = sandbox or PythonSandboxTool(timeout_seconds=10)
        code = _strip_code_fence(response)
        tests = case.tests or ""
        if not code.strip():
            error = "empty_model_response"
        elif not tests.strip():
            error = "missing_unit_tests"
        else:
            result = sandbox.run_python(code + "\n\n" + tests + "\n")
            passed = result.returncode == 0
            score = 1.0 if passed else 0.0
            feedback = (result.stdout + "\n" + result.stderr).strip()[:1200]
            if result.timed_out:
                error = "unit_tests_timed_out"
            elif result.returncode != 0:
                error = f"unit_tests_failed:returncode={result.returncode}"
    elif metric == "llm_judge":
        if judge_client is not None and hasattr(judge_client, "judge_eval_answer"):
            try:
                data = judge_client.judge_eval_answer(case=to_jsonable(case), response=response)
                score = float(data.get("score", 0.0))
                passed = bool(data.get("passed", score >= 0.7))
                feedback = str(data.get("feedback") or data.get("reason") or "")
            except Exception as exc:
                error = f"judge_failed:{type(exc).__name__}: {exc}"
        if not feedback and expected:
            # Deterministic fallback: partial credit by overlap.
            expected_terms = set(_norm(expected).split())
            response_terms = set(_norm(response).split())
            if expected_terms:
                overlap = len(expected_terms & response_terms) / max(1, len(expected_terms))
                score = max(score, min(1.0, overlap))
                passed = score >= 0.6
                feedback = "llm_judge_fallback_overlap"
        if not feedback:
            feedback = "llm_judge_unavailable"
    else:
        # Unknown metrics get a conservative deterministic fallback.
        passed = bool(expected) and _norm(expected) in _norm(response)
        score = 1.0 if passed else 0.0
        feedback = f"unknown_metric_fallback:{metric}"

    return EvalResult(
        case_id=case.id,
        prompt=case.prompt,
        metric=case.metric,
        passed=passed,
        score=round(float(score), 4),
        response=response,
        expected=case.expected,
        reference_answer=case.reference_answer,
        error=error,
        judge_feedback=feedback,
        tags=case.tags,
        weakness_area=case.weakness_area,
    )


def run_eval_suite(
    suite: EvalSuite,
    *,
    model_client: Any | None = None,
    judge_client: Any | None = None,
    target_metric: str = "accuracy",
    target_value: float | None = None,
    max_tokens: int = 1024,
) -> ModelEvalReport:
    results: list[EvalResult] = []
    sandbox = PythonSandboxTool(timeout_seconds=10)
    for case in suite.cases:
        started = time.perf_counter()
        error: str | None = None
        if model_client is None:
            response = ""
            error = "model_client_unavailable"
        else:
            try:
                response = model_client.chat(
                    messages=[{"role": "user", "content": case.prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                response = ""
                error = f"model_call_failed:{type(exc).__name__}: {exc}"
        result = evaluate_answer(case, response, judge_client=judge_client, sandbox=sandbox)
        if error and not result.error:
            result.error = error
        result.latency_seconds = round(time.perf_counter() - started, 4)
        results.append(result)

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    score = round((sum(float(r.score) for r in results) / total) if total else 0.0, 4)
    return ModelEvalReport(
        suite_name=suite.name,
        goal=suite.goal,
        metric_name=target_metric,
        score=score,
        passed_count=passed_count,
        total_count=total,
        target_value=target_value,
        target_met=(score >= target_value) if target_value is not None else None,
        results=results,
    )


def failed_areas(report: ModelEvalReport, limit: int = 8) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for result in report.results:
        if result.passed:
            continue
        area = result.weakness_area or "general"
        bucket = buckets.setdefault(area, {"area": area, "failed_count": 0, "case_ids": [], "tags": set(), "examples": []})
        bucket["failed_count"] += 1
        bucket["case_ids"].append(result.case_id)
        bucket["tags"].update(result.tags)
        if len(bucket["examples"]) < 3:
            bucket["examples"].append({"prompt": result.prompt, "response": (result.response or "")[:500], "error": result.error, "feedback": result.judge_feedback})
    rows = []
    for bucket in buckets.values():
        bucket["tags"] = sorted(str(x) for x in bucket["tags"])
        rows.append(bucket)
    rows.sort(key=lambda x: x["failed_count"], reverse=True)
    return rows[:limit]
