from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REAL_EVAL_SOURCES = {"benchmark", "explicit", "explicit_eval_set", "private_eval", "heldout", "real"}
SMOKE_EVAL_SOURCES = {"smoke", "fallback", "synthetic", "teacher_generated", "kimi_generated", "generated"}


@dataclass
class StopDecision:
    stop: bool
    reason: str
    target_met: bool = False
    early_stop_allowed: bool = False
    eval_source: str = "unknown"
    case_count: int = 0
    score: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class EvalPolicy:
    policy: str = "real_required"
    allow_smoke_early_stop: bool = False
    min_cases_for_early_stop: int = 30

    @classmethod
    def from_config(cls, raw_config: dict[str, Any] | None) -> "EvalPolicy":
        raw = (raw_config or {}).get("eval") if isinstance(raw_config, dict) else None
        if not isinstance(raw, dict):
            raw = {}
        return cls(
            policy=str(raw.get("policy") or "real_required"),
            allow_smoke_early_stop=bool(raw.get("allow_smoke_early_stop", False)),
            min_cases_for_early_stop=int(raw.get("min_cases_for_early_stop") or raw.get("min_cases") or 30),
        )

    def _score_threshold_target(self, evaluation: dict[str, Any]) -> tuple[float | None, float | None, bool]:
        score_raw = evaluation.get("score", evaluation.get("accuracy", evaluation.get("metric_value")))
        threshold_raw = evaluation.get("threshold", evaluation.get("target_threshold"))
        score: float | None = None
        threshold: float | None = None
        try:
            if score_raw is not None:
                score = float(score_raw)
        except Exception:
            score = None
        try:
            if threshold_raw is not None:
                threshold = float(threshold_raw)
        except Exception:
            threshold = None
        if "target_met" in evaluation:
            return score, threshold, bool(evaluation.get("target_met"))
        if score is not None and threshold is not None:
            return score, threshold, score >= threshold
        return score, threshold, False

    def decide(self, evaluation: dict[str, Any] | None) -> StopDecision:
        if not evaluation:
            return StopDecision(False, "no evaluation result")

        score, threshold, target_met = self._score_threshold_target(evaluation)
        source = str(
            evaluation.get("eval_source")
            or evaluation.get("source")
            or evaluation.get("metadata", {}).get("eval_source")
            or ("benchmark" if evaluation.get("benchmark") else "unknown")
        )
        details = evaluation.get("case_results") or evaluation.get("details") or []
        case_count = int(evaluation.get("case_count") or evaluation.get("total") or (len(details) if isinstance(details, list) else 0))
        real = source in REAL_EVAL_SOURCES or source.startswith("benchmark:")
        smoke = source in SMOKE_EVAL_SOURCES or source.startswith("smoke")
        contam = evaluation.get("decontamination") or evaluation.get("decontamination_report") or {}
        if isinstance(contam, dict) and contam.get("ok") is False:
            return StopDecision(False, "decontamination report is not OK", target_met=target_met, eval_source=source, case_count=case_count, score=score, threshold=threshold)
        if not target_met:
            return StopDecision(False, "target not met", target_met=False, eval_source=source, case_count=case_count, score=score, threshold=threshold)
        if smoke and not self.allow_smoke_early_stop:
            return StopDecision(False, "smoke/synthetic eval cannot stop the autonomous loop", target_met=True, eval_source=source, case_count=case_count, score=score, threshold=threshold)
        if self.policy == "real_required" and not real:
            return StopDecision(False, "eval policy requires benchmark/explicit/private eval before stopping", target_met=True, eval_source=source, case_count=case_count, score=score, threshold=threshold)
        if case_count < self.min_cases_for_early_stop:
            return StopDecision(False, f"case_count {case_count} < min_cases_for_early_stop {self.min_cases_for_early_stop}", target_met=True, eval_source=source, case_count=case_count, score=score, threshold=threshold)
        return StopDecision(True, "real eval target met", target_met=True, early_stop_allowed=True, eval_source=source, case_count=case_count, score=score, threshold=threshold)
