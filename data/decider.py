from __future__ import annotations

from typing import Any

from autopilot.data.probe import text_similarity
from autopilot.data.scorer import goal_match_score
from autopilot.models import (
    AdoptionDecision,
    DatasetClassification,
    DatasetInspection,
    DatasetWebOverview,
    ModelTrial,
    TrainingType,
)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def model_gap_score(trials: list[ModelTrial]) -> float:
    scored = [t.similarity_to_reference for t in trials if t.similarity_to_reference is not None and not t.error]
    if not scored:
        # Unknown, not bad. The decider will reduce its weight if no trials exist.
        return 0.0
    avg_similarity = sum(float(s) for s in scored) / len(scored)
    return round(_clamp(1.0 - avg_similarity), 4)


def decide_dataset_adoption(
    goal: str,
    inspection: DatasetInspection,
    classification: DatasetClassification,
    overview: DatasetWebOverview | None = None,
    trials: list[ModelTrial] | None = None,
) -> AdoptionDecision:
    trials = trials or []
    metadata = inspection.metadata
    goal_score = goal_match_score(goal, inspection.dataset_id, metadata.tags, inspection.columns)
    visible_sample_score = 1.0 if len(inspection.sample_rows) >= 5 else (0.6 if inspection.sample_rows else 0.0)
    file_signal = 0.2
    if overview and overview.files:
        file_signal = 0.6
        if any(f.path.lower().endswith(('.json', '.jsonl', '.parquet', '.csv', '.arrow')) for f in overview.files):
            file_signal = 1.0

    if TrainingType.UNKNOWN in classification.recommended_training:
        schema_score = 0.15
    else:
        schema_score = classification.confidence

    data_value = _clamp(0.38 * schema_score + 0.32 * goal_score + 0.2 * visible_sample_score + 0.1 * file_signal)
    gap = model_gap_score(trials)

    if trials:
        final = _clamp(0.58 * data_value + 0.32 * gap + 0.10 * visible_sample_score)
    else:
        final = _clamp(0.84 * data_value + 0.16 * visible_sample_score)

    reasons: list[str] = []
    reasons.extend(classification.reasons[:3])
    if overview and overview.files:
        reasons.append(f"Found {len(overview.files)} visible repository files.")
    if inspection.sample_rows:
        reasons.append(f"Loaded {len(inspection.sample_rows)} example rows for inspection.")
    if trials:
        reasons.append(f"Probed local vLLM model on {len(trials)} extracted prompts; model_gap_score={gap:.3f}.")

    if TrainingType.UNKNOWN in classification.recommended_training or not inspection.sample_rows:
        action = "review"
    elif final >= 0.72:
        action = "accept"
    elif final >= 0.42:
        action = "review"
    else:
        action = "reject"

    notes = ""
    if trials and gap >= 0.55:
        notes = "本地模型和参考答案差距较大，这类数据可能有训练价值。"
    elif trials and gap < 0.25:
        notes = "本地模型对抽样问题已接近参考答案，训练增益可能有限。"
    elif not trials:
        notes = "未配置或未启用 vLLM probe，因此决策主要基于数据结构和样本可见性。"

    return AdoptionDecision(
        action=action,
        final_score=round(final, 4),
        data_value_score=round(data_value, 4),
        model_gap_score=gap,
        training_types=classification.recommended_training,
        reasons=reasons,
        notes=notes,
    )


def merge_llm_decision(base: AdoptionDecision, llm_data: dict[str, Any]) -> AdoptionDecision:
    if not isinstance(llm_data, dict):
        return base
    action = str(llm_data.get("action") or llm_data.get("recommended_action") or base.action).lower()
    if action not in {"accept", "review", "reject"}:
        action = base.action

    def get_float(key: str, fallback: float) -> float:
        try:
            return _clamp(float(llm_data.get(key, fallback)))
        except Exception:
            return fallback

    training_types: list[TrainingType] = []
    for raw in llm_data.get("training_types") or llm_data.get("best_training_types") or []:
        try:
            training_types.append(TrainingType(str(raw)))
        except Exception:
            continue
    if not training_types:
        training_types = base.training_types

    reason = llm_data.get("reason") or llm_data.get("notes") or llm_data.get("conversion_notes")
    reasons = list(base.reasons)
    if reason:
        reasons.insert(0, f"LLM decision: {reason}")

    return AdoptionDecision(
        action=action,
        final_score=round(get_float("final_score", base.final_score), 4),
        data_value_score=round(get_float("data_value_score", base.data_value_score), 4),
        model_gap_score=round(get_float("model_gap_score", base.model_gap_score), 4),
        training_types=training_types,
        reasons=reasons[:8],
        notes=str(llm_data.get("conversion_notes") or llm_data.get("notes") or base.notes),
        llm_decision=llm_data,
    )
