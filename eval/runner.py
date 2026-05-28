from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from autopilot.goal.spec import EvalCase
from autopilot.llm.kimi import KimiClient
from autopilot.llm.vllm import VLLMClient
from autopilot.models import to_jsonable
from autopilot.tools.bash import BashRunner


@dataclass
class EvalCaseResult:
    case_index: int
    prompt: str
    expected: str | None
    response: str | None
    passed: bool | None
    score: float | None
    verifier: str
    tags: list[str] = field(default_factory=list)
    reason: str = ""
    error: str | None = None


@dataclass
class EvaluationResult:
    metric_name: str
    score: float | None
    target: float | None = None
    target_met: bool = False
    case_results: list[EvalCaseResult] = field(default_factory=list)
    failures: list[EvalCaseResult] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def score_exact_or_contains(case: EvalCase, response: str | None) -> tuple[bool | None, float | None, str]:
    if response is None:
        return None, None, "no response"
    if case.expected is None or not str(case.expected).strip():
        return None, None, "no reference answer"
    verifier = (case.verifier or "contains").lower()
    expected = normalize_answer(case.expected)
    actual = normalize_answer(response)
    if verifier == "exact_match":
        ok = actual == expected
        return ok, 1.0 if ok else 0.0, "exact match" if ok else "exact mismatch"
    ok = expected in actual
    return ok, 1.0 if ok else 0.0, "expected substring present" if ok else "expected substring missing"


def run_python_unit_test(code: str, tests: str, *, timeout: float = 10.0) -> tuple[bool, str]:
    """Run small unit tests in a temporary directory.

    This is a lightweight verifier hook. It is not a security sandbox; it is
    intended for trusted/local generated code in agentic training experiments.
    """
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate_test.py"
        path.write_text(code.rstrip() + "\n\n" + tests.rstrip() + "\n", encoding="utf-8")
        result = BashRunner(cwd=td, timeout=timeout).run(["python", str(path)])
        output = (result.stdout + "\n" + result.stderr).strip()
        return result.ok, output[-2000:]


def score_case(case: EvalCase, response: str | None, *, kimi: KimiClient | None = None, goal: str = "") -> EvalCaseResult:
    verifier = (case.verifier or "contains").lower()
    if verifier in {"contains", "exact_match", "reference_contains"}:
        passed, score, reason = score_exact_or_contains(case, response)
        return EvalCaseResult(
            case_index=-1,
            prompt=case.prompt,
            expected=case.expected,
            response=response,
            passed=passed,
            score=score,
            verifier=verifier,
            tags=case.tags,
            reason=reason,
        )
    if verifier in {"python_unit_test", "unit_test"}:
        tests = str(case.metadata.get("tests") or case.expected or "")
        if not response or not tests:
            return EvalCaseResult(-1, case.prompt, case.expected, response, None, None, verifier, case.tags, "missing response or tests")
        ok, output = run_python_unit_test(response, tests)
        return EvalCaseResult(-1, case.prompt, case.expected, response, ok, 1.0 if ok else 0.0, verifier, case.tags, output)
    if verifier in {"kimi_judge", "llm_judge", "judge"} and kimi is not None:
        try:
            judgement = kimi.judge_eval_case(goal=goal, prompt=case.prompt, expected=case.expected, response=response or "", rubric=case.rubric)
            score = judgement.get("score")
            if score is not None:
                score = float(score)
            passed_raw = judgement.get("passed")
            passed = bool(passed_raw) if passed_raw is not None else (score is not None and score >= 0.7)
            return EvalCaseResult(-1, case.prompt, case.expected, response, passed, score, verifier, case.tags, str(judgement.get("reason") or judgement))
        except Exception as exc:
            return EvalCaseResult(-1, case.prompt, case.expected, response, None, None, verifier, case.tags, "KIMI judge failed", error=f"{type(exc).__name__}: {exc}")
    # Unknown verifier: fall back to reference scoring if possible.
    passed, score, reason = score_exact_or_contains(case, response)
    return EvalCaseResult(-1, case.prompt, case.expected, response, passed, score, verifier or "unknown", case.tags, reason)


def evaluate_cases(
    cases: Iterable[EvalCase],
    *,
    metric_name: str = "accuracy",
    target: float | None = None,
    vllm: VLLMClient | None = None,
    kimi: KimiClient | None = None,
    goal: str = "",
    provided_responses: list[str] | None = None,
    max_tokens: int = 1024,
) -> EvaluationResult:
    results: list[EvalCaseResult] = []
    provided_responses = provided_responses or []
    for idx, case in enumerate(list(cases)):
        response: str | None = None
        error: str | None = None
        if idx < len(provided_responses):
            response = provided_responses[idx]
        elif vllm is not None:
            try:
                response = vllm.chat([{"role": "user", "content": case.prompt}], temperature=0.0, max_tokens=max_tokens)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        result = score_case(case, response, kimi=kimi, goal=goal)
        result.case_index = idx
        if error and result.error is None:
            result.error = error
            result.reason = error
        results.append(result)
    scored = [r.score for r in results if r.score is not None]
    score = sum(scored) / len(scored) if scored else None
    failures = [r for r in results if r.passed is False or r.error]
    target_met = bool(score is not None and target is not None and score >= target)
    return EvaluationResult(
        metric_name=metric_name,
        score=score,
        target=target,
        target_met=target_met,
        case_results=results,
        failures=failures,
        notes=f"Evaluated {len(results)} cases; scored {len(scored)} cases.",
    )


def write_eval_cases_jsonl(cases: Iterable[EvalCase], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(to_jsonable(case), ensure_ascii=False) + "\n")
    return p
