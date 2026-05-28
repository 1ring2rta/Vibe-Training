from __future__ import annotations

import math
import re

from autopilot.models import DatasetClassification, DatasetInspection, DatasetScore, RiskAssessment, TrainingType

OPEN_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "bsd-2-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "odc-by",
}

RESTRICTIVE_LICENSE_MARKERS = {"non-commercial", "noncommercial", "cc-by-nc", "research-only", "other"}

DOMAIN_SYNONYMS = {
    "法律": ["law", "legal", "juris", "court", "case", "律师", "法", "合同", "法规"],
    "代码": ["code", "coding", "programming", "python", "java", "leetcode", "bug", "debug", "代码"],
    "数学": ["math", "mathematics", "gsm", "proof", "algebra", "geometry", "数学", "推理"],
    "医疗": ["medical", "medicine", "health", "clinical", "doctor", "医疗", "医学"],
    "金融": ["finance", "financial", "stock", "bank", "经济", "金融"],
    "中文": ["chinese", "zh", "cn", "中文", "汉语", "華語"],
}


def _tokenize(text: str) -> set[str]:
    text = text.lower()
    parts = re.split(r"[^a-z0-9\u4e00-\u9fff]+", text)
    tokens = {p for p in parts if len(p) >= 2}
    for key, synonyms in DOMAIN_SYNONYMS.items():
        if key in text:
            tokens.update(s.lower() for s in synonyms)
    return tokens


def goal_match_score(goal: str, dataset_id: str, tags: list[str], columns: list[str]) -> float:
    goal_tokens = _tokenize(goal)
    dataset_tokens = _tokenize(" ".join([dataset_id, *tags, *columns]))
    if not goal_tokens:
        return 0.08
    overlap = len(goal_tokens & dataset_tokens)
    score = min(1.0, overlap / max(3, len(goal_tokens) * 0.5))
    # Tiny extra for exact Chinese terms appearing in id/tags/columns.
    lower_blob = " ".join([dataset_id, *tags, *columns]).lower()
    for term in ["中文", "法律", "代码", "数学", "医疗", "金融"]:
        if term in goal and term in lower_blob:
            score = min(1.0, score + 0.12)
    return score


def license_score(license_name: str | None) -> float:
    if not license_name:
        return 0.2
    license_name = license_name.lower()
    if license_name in OPEN_LICENSES or any(open_lic in license_name for open_lic in OPEN_LICENSES):
        return 1.0
    if any(marker in license_name for marker in RESTRICTIVE_LICENSE_MARKERS):
        return 0.0
    return 0.45


def community_score(downloads: int | None, likes: int | None) -> float:
    d = max(0, downloads or 0)
    l = max(0, likes or 0)
    # log scale: downloads dominate but likes help.
    return max(0.0, min(1.0, (math.log10(d + 1) / 6.0) * 0.75 + (math.log10(l + 1) / 4.0) * 0.25))


def training_type_score(classification: DatasetClassification) -> float:
    if TrainingType.UNKNOWN in classification.recommended_training:
        return 0.1
    if TrainingType.SFT in classification.recommended_training:
        return 0.95 * classification.confidence
    if TrainingType.DPO in classification.recommended_training:
        return 0.9 * classification.confidence
    if TrainingType.CONTINUED_PRETRAINING in classification.recommended_training:
        return 0.7 * classification.confidence
    if TrainingType.RLVR in classification.recommended_training:
        return 0.75 * classification.confidence
    return classification.confidence


def score_dataset(
    goal: str,
    inspection: DatasetInspection,
    classification: DatasetClassification,
    risk: RiskAssessment,
) -> DatasetScore:
    metadata = inspection.metadata
    components = {
        "goal_match": goal_match_score(goal, inspection.dataset_id, metadata.tags, inspection.columns) * 0.25,
        "schema_training_fit": training_type_score(classification) * 0.28,
        "sample_quality": risk.quality_score * 0.22,
        "license": license_score(metadata.license) * 0.15,
        "community_signal": community_score(metadata.downloads, metadata.likes) * 0.10,
    }
    penalty = risk.risk_score * 0.35
    score = sum(components.values()) - penalty
    score = max(0.0, min(1.0, score))

    reasons = list(classification.reasons)
    if metadata.license:
        reasons.append(f"Detected license: {metadata.license}.")
    if risk.sample_count:
        reasons.append(f"Reviewed {risk.sample_count} sample rows.")

    risks = list(classification.risks)
    risks.extend([f"{flag.name}: {flag.description}" for flag in risk.flags])

    return DatasetScore(
        suitability_score=round(score, 4),
        components={k: round(v, 4) for k, v in components.items()},
        reasons=reasons,
        risks=risks,
    )
