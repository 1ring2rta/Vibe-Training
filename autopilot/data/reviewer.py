from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from typing import Any

from autopilot.data.schema import row_text, value_to_text
from autopilot.models import DatasetInspection, RiskAssessment, RiskFlag, RiskSeverity


def compact_text(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text[:limit] + ("..." if len(text) > limit else "")


# Kept for backward compatibility with older report code. In this version the
# main flow does not perform security/PII screening, so this is a no-op.
def redact_pii_text(text: str) -> str:
    return text


def redact_pii_obj(value: Any) -> Any:
    return value


def _text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()


def _chinese_ratio(texts: list[str]) -> float:
    joined = "".join(texts)
    if not joined:
        return 0.0
    chars = [ch for ch in joined if not ch.isspace()]
    if not chars:
        return 0.0
    zh = sum(1 for ch in chars if "\u4e00" <= ch <= "\u9fff")
    return zh / max(1, len(chars))


def _is_chinese_goal(goal: str | None) -> bool:
    if not goal:
        return False
    zh = sum(1 for ch in goal if "\u4e00" <= ch <= "\u9fff")
    return zh >= 2 or any(term in goal.lower() for term in ["zh", "chinese", "中文", "汉语", "華語"])


def _flag(name: str, severity: RiskSeverity, description: str, count: int = 0, examples: list[str] | None = None) -> RiskFlag:
    return RiskFlag(name=name, severity=severity, description=description, count=count, examples=examples or [])


class DeterministicSampleReviewer:
    """Review whether samples are useful and convertible.

    This intentionally does not perform safety/PII/toxicity checks. It focuses on
    whether the dataset has enough visible examples, whether samples are non-empty,
    whether rows look duplicated, and whether language/domain roughly match the
    user's stated goal.
    """

    def review(self, inspection: DatasetInspection, goal: str | None = None) -> RiskAssessment:
        rows = inspection.sample_rows or []
        texts = [row_text(row) for row in rows]
        nonempty_texts = [t for t in texts if t.strip()]
        sample_count = len(rows)
        flags: list[RiskFlag] = []

        if sample_count == 0:
            flags.append(_flag("no_visible_samples", RiskSeverity.HIGH, "No example rows could be loaded from the dataset viewer or streaming loader."))
        elif sample_count < 3:
            flags.append(_flag("few_visible_samples", RiskSeverity.MEDIUM, "Only a very small number of examples was available for review.", count=sample_count))

        empty_count = sample_count - len(nonempty_texts)
        if empty_count > 0:
            flags.append(_flag("empty_rows", RiskSeverity.MEDIUM, "Some loaded rows are empty or stringify to empty text.", count=empty_count))

        lengths = [len(t) for t in nonempty_texts]
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        if nonempty_texts and avg_len < 20:
            flags.append(_flag("short_rows", RiskSeverity.MEDIUM, "Average sample text is very short, so training value may be limited.", count=len(nonempty_texts)))
        if nonempty_texts and avg_len > 12000:
            flags.append(_flag("very_long_rows", RiskSeverity.LOW, "Average sample text is very long; conversion may need truncation/chunking.", count=len(nonempty_texts)))

        hashes = [_text_hash(t) for t in nonempty_texts]
        duplicate_rate = 0.0
        if hashes:
            duplicate_rate = 1.0 - len(set(hashes)) / len(hashes)
            if duplicate_rate >= 0.25:
                flags.append(_flag("duplicate_examples", RiskSeverity.MEDIUM, "A noticeable fraction of sampled rows are duplicates.", count=int(duplicate_rate * len(hashes))))

        chinese_ratio = _chinese_ratio(nonempty_texts)
        if _is_chinese_goal(goal) and nonempty_texts and chinese_ratio < 0.08:
            flags.append(_flag("language_mismatch", RiskSeverity.MEDIUM, "Goal looks Chinese-oriented but visible samples contain little Chinese text."))

        risk_score = 0.0
        for flag in flags:
            if flag.severity == RiskSeverity.HIGH:
                risk_score += 0.45
            elif flag.severity == RiskSeverity.MEDIUM:
                risk_score += 0.18
            else:
                risk_score += 0.07
        risk_score = max(0.0, min(1.0, risk_score))
        quality_score = max(0.0, min(1.0, 1.0 - risk_score * 0.75))
        if sample_count >= 5 and nonempty_texts:
            quality_score = min(1.0, quality_score + 0.08)

        return RiskAssessment(
            risk_score=round(risk_score, 4),
            quality_score=round(quality_score, 4),
            flags=flags,
            sample_count=sample_count,
            duplicate_rate=round(duplicate_rate, 4),
            avg_text_length=round(avg_len, 2),
            chinese_char_ratio=round(chinese_ratio, 4),
        )


def with_llm_review(assessment: RiskAssessment, llm_review: dict[str, Any]) -> RiskAssessment:
    """Attach LLM review result without letting it redefine deterministic fields."""
    return replace(assessment, llm_review=llm_review)
