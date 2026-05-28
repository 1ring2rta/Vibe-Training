# New v0.5 goal-loop eval suite API.
from .suite import (
    EvalCase,
    EvalResult,
    EvalSuite,
    ModelEvalReport,
    default_eval_cases_for_goal,
    load_eval_suite,
    run_eval_suite,
    write_eval_suite,
)

# Backward-compatible v0.5 draft API, if these modules are present.
try:  # pragma: no cover - compatibility shim
    from autopilot.eval.runner import EvalCaseResult, EvaluationResult, evaluate_cases, write_eval_cases_jsonl
    from autopilot.eval.diagnose import FailureDiagnosis, diagnose_failures
except Exception:  # pragma: no cover
    EvalCaseResult = EvaluationResult = FailureDiagnosis = None  # type: ignore
    evaluate_cases = write_eval_cases_jsonl = diagnose_failures = None  # type: ignore

__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalSuite",
    "ModelEvalReport",
    "default_eval_cases_for_goal",
    "load_eval_suite",
    "run_eval_suite",
    "write_eval_suite",
    "EvalCaseResult",
    "EvaluationResult",
    "evaluate_cases",
    "write_eval_cases_jsonl",
    "FailureDiagnosis",
    "diagnose_failures",
]
from autopilot.eval.tool_registry import EvalToolRegistry, EvalToolSpec
