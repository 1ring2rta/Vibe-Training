from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from autopilot.models import to_jsonable


@dataclass
class MetricTarget:
    """A stopping condition for the training loop."""

    name: str = "accuracy"
    target: float = 0.8
    eval_set: str | None = None
    split: str | None = None
    higher_is_better: bool = True
    tolerance: float = 1e-9

    def met(self, value: float | None) -> bool:
        if value is None:
            return False
        if self.higher_is_better:
            return value + self.tolerance >= self.target
        return value - self.tolerance <= self.target


@dataclass
class EvalCase:
    prompt: str
    expected: str | None = None
    rubric: str | None = None
    tags: list[str] = field(default_factory=list)
    source: str = "inline"
    verifier: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalSpec:
    """A target-driven post-training objective.

    The framework treats this as the root state of a loop, not as a fixed
    workflow. Every round can create subtasks for data search, tool discovery,
    verifier discovery, training, evaluation, and remediation.
    """

    name: str
    description: str
    base_model: str | None = None
    target: MetricTarget = field(default_factory=MetricTarget)
    eval_cases: list[EvalCase] = field(default_factory=list)
    max_rounds: int = 2
    preferred_training: list[str] = field(default_factory=lambda: ["sft", "dpo", "rlvr"])
    use_kimi_generated_tests: bool = True
    use_kimi_judge: bool = True
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _as_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def eval_case_from_mapping(data: Mapping[str, Any], *, source: str = "inline") -> EvalCase:
    prompt = str(data.get("prompt") or data.get("question") or data.get("input") or "").strip()
    expected = data.get("expected", data.get("answer", data.get("output", data.get("reference", data.get("reference_answer")))))
    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    metadata = {k: v for k, v in data.items() if k not in {"prompt", "question", "input", "expected", "answer", "output", "reference", "rubric", "tags", "source", "verifier"}}
    return EvalCase(
        prompt=prompt,
        expected=str(expected) if expected is not None else None,
        rubric=str(data.get("rubric")) if data.get("rubric") is not None else None,
        tags=[str(x) for x in tags],
        source=str(data.get("source") or source),
        verifier=str(data.get("verifier", data.get("metric"))) if data.get("verifier", data.get("metric")) is not None else None,
        metadata=metadata,
    )


def load_eval_cases(path: str | Path | None) -> list[EvalCase]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval set not found: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    cases: list[EvalCase] = []
    if p.suffix.lower() in {".jsonl", ".jsonlines"}:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, Mapping):
                case = eval_case_from_mapping(item, source=str(p))
                if case.prompt:
                    cases.append(case)
        return cases
    data = json.loads(text)
    if isinstance(data, Mapping) and isinstance(data.get("cases"), list):
        data = data["cases"]
    if isinstance(data, list):
        for item in data:
            if isinstance(item, Mapping):
                case = eval_case_from_mapping(item, source=str(p))
                if case.prompt:
                    cases.append(case)
    elif isinstance(data, Mapping):
        case = eval_case_from_mapping(data, source=str(p))
        if case.prompt:
            cases.append(case)
    return cases


def build_goal_spec(
    *,
    raw_config: Mapping[str, Any] | None = None,
    goal: str | None = None,
    target_metric: str | None = None,
    target_score: float | None = None,
    eval_set: str | None = None,
    base_model: str | None = None,
    max_rounds: int | None = None,
    use_kimi_generated_tests: bool | None = None,
    use_kimi_judge: bool | None = None,
) -> GoalSpec:
    cfg = dict(raw_config or {})
    goal_cfg = cfg.get("goal") or cfg.get("target") or {}
    if not isinstance(goal_cfg, Mapping):
        goal_cfg = {}
    target_cfg = goal_cfg.get("target") or goal_cfg.get("metric") or {}
    if not isinstance(target_cfg, Mapping):
        target_cfg = {}

    description = str(goal or goal_cfg.get("description") or goal_cfg.get("goal") or cfg.get("goal") or "").strip()
    name = str(goal_cfg.get("name") or (description[:48] if description else "training-goal")).strip() or "training-goal"
    metric_name = str(target_metric or target_cfg.get("name") or target_cfg.get("metric") or "accuracy")
    score = float(target_score) if target_score is not None else _as_float(target_cfg.get("target", target_cfg.get("threshold")), 0.8)
    eval_path = eval_set or target_cfg.get("eval_set") or target_cfg.get("dataset") or goal_cfg.get("eval_set")

    inline_cases: list[EvalCase] = []
    for raw_case in _as_list(goal_cfg.get("eval_cases")):
        if isinstance(raw_case, Mapping):
            case = eval_case_from_mapping(raw_case, source="config")
            if case.prompt:
                inline_cases.append(case)

    file_cases = load_eval_cases(str(eval_path)) if eval_path else []
    preferred_training = [str(x) for x in _as_list(goal_cfg.get("preferred_training") or goal_cfg.get("training") or ["sft", "dpo", "rlvr"])]

    return GoalSpec(
        name=name,
        description=description,
        base_model=str(base_model or goal_cfg.get("base_model") or goal_cfg.get("model") or "") or None,
        target=MetricTarget(
            name=metric_name,
            target=score,
            eval_set=str(eval_path) if eval_path else None,
            split=str(target_cfg.get("split")) if target_cfg.get("split") else None,
            higher_is_better=_as_bool(target_cfg.get("higher_is_better"), True),
        ),
        eval_cases=file_cases + inline_cases,
        max_rounds=int(max_rounds) if max_rounds is not None else _as_int(goal_cfg.get("max_rounds"), 2),
        preferred_training=preferred_training,
        use_kimi_generated_tests=_as_bool(use_kimi_generated_tests, _as_bool(goal_cfg.get("use_kimi_generated_tests"), True)),
        use_kimi_judge=_as_bool(use_kimi_judge, _as_bool(goal_cfg.get("use_kimi_judge"), True)),
        constraints=dict(goal_cfg.get("constraints") or {}) if isinstance(goal_cfg.get("constraints"), Mapping) else {},
    )


def default_eval_cases_for_goal(goal: str, *, n: int = 5) -> list[EvalCase]:
    """Deterministic fallback tests when no external eval set is provided.

    These are not a benchmark. They are smoke/regression prompts so the loop can
    keep moving until KIMI or a user-provided eval set supplies stronger cases.
    """
    lower = goal.lower()
    if any(x in lower for x in ["code", "coding", "代码", "python", "bug"]):
        cases = [
            EvalCase(prompt="Write a Python function add(a, b) that returns the sum.", expected="def add", tags=["coding", "function"], verifier="contains"),
            EvalCase(prompt="Fix this bug: def f(xs): return xs[0] when xs may be empty.", expected="if", tags=["coding", "bugfix"], verifier="contains"),
            EvalCase(prompt="Explain why binary search is O(log n).", expected="log", tags=["coding", "explain"], verifier="contains"),
            EvalCase(prompt="Write a pytest test for a function that reverses a string.", expected="assert", tags=["coding", "tests"], verifier="contains"),
            EvalCase(prompt="Implement factorial recursively in Python.", expected="factorial", tags=["coding", "algorithm"], verifier="contains"),
        ]
    elif any(x in lower for x in ["math", "数学", "推理"]):
        cases = [
            EvalCase(prompt="Solve: 17 + 25 = ?", expected="42", tags=["math", "arithmetic"], verifier="exact_match"),
            EvalCase(prompt="Solve: if 3x = 12, what is x?", expected="4", tags=["math", "algebra"], verifier="contains"),
            EvalCase(prompt="A rectangle is 3 by 5. What is its area?", expected="15", tags=["math"], verifier="contains"),
        ]
    else:
        cases = [
            EvalCase(prompt=f"Give a concise, correct answer for this target task: {goal}", expected=None, tags=["smoke"], verifier="kimi_judge"),
            EvalCase(prompt=f"Create one high-quality example question for: {goal}", expected=None, tags=["sample_generation"], verifier="kimi_judge"),
        ]
    return cases[: max(1, n)]
