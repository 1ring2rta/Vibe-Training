from __future__ import annotations

from typing import Any

from autopilot.data.schema import normalize_col, normalized_columns, looks_like_chat_messages
from autopilot.models import DatasetClassification, TrainingType


INSTRUCTION_COLS = {"instruction", "instruct", "task", "query", "question", "prompt", "input"}
OUTPUT_COLS = {"output", "response", "answer", "completion", "target", "label", "assistant_response"}
PROMPT_COLS = {"prompt", "query", "question", "instruction", "input"}
CHOSEN_COLS = {"chosen", "chosen_response", "accepted", "winner", "positive", "better"}
REJECTED_COLS = {"rejected", "rejected_response", "refused", "loser", "negative", "worse"}
TEXT_COLS = {"text", "content", "document", "article", "passage"}
LABEL_COLS = {"label", "score", "rating", "reward", "preference", "kto_tag"}
TEST_COLS = {"test", "tests", "unit_tests", "assertions", "verifier", "reward_fn", "judge", "expected_output"}


def _has_any(cols: set[str], candidates: set[str]) -> bool:
    return bool(cols & candidates)


def _sample_has_chat_messages(sample_rows: list[dict[str, Any]], column: str) -> bool:
    for row in sample_rows[:10]:
        for key, value in row.items():
            if normalize_col(key) == column and looks_like_chat_messages(value):
                return True
    return False


def classify_dataset(columns: list[str], sample_rows: list[dict[str, Any]] | None = None) -> DatasetClassification:
    """Classify the likely training type from schema and a few samples.

    This intentionally starts with deterministic rules. LLM-based judgement can be
    layered on top later, but the base classifier should stay testable.
    """
    sample_rows = sample_rows or []
    cols = normalized_columns(columns)

    # Strong preference-pair patterns.
    if _has_any(cols, CHOSEN_COLS) and _has_any(cols, REJECTED_COLS) and _has_any(cols, PROMPT_COLS):
        return DatasetClassification(
            format_type="preference_pair",
            recommended_training=[TrainingType.DPO, TrainingType.REWARD_MODEL],
            confidence=0.96,
            reasons=["Found prompt-like + chosen-like + rejected-like columns."],
            risks=["Preference pairs still need quality and alignment review."],
        )

    if _has_any(cols, CHOSEN_COLS) and _has_any(cols, REJECTED_COLS):
        return DatasetClassification(
            format_type="preference_pair_without_explicit_prompt",
            recommended_training=[TrainingType.DPO, TrainingType.REWARD_MODEL],
            confidence=0.86,
            reasons=["Found chosen-like and rejected-like columns."],
            risks=["Prompt may be embedded or missing; conversion needs inspection."],
        )

    # KTO-like binary feedback, especially if it includes prompt/output.
    if "kto_tag" in cols or (_has_any(cols, PROMPT_COLS) and _has_any(cols, LABEL_COLS) and _has_any(cols, OUTPUT_COLS)):
        return DatasetClassification(
            format_type="binary_or_labeled_feedback",
            recommended_training=[TrainingType.KTO, TrainingType.REWARD_MODEL],
            confidence=0.76,
            reasons=["Found prompt/output plus label-like feedback columns."],
            risks=["Need to verify label semantics before KTO/RM training."],
        )

    # Chat formats.
    if "messages" in cols:
        confidence = 0.92 if _sample_has_chat_messages(sample_rows, "messages") else 0.82
        return DatasetClassification(
            format_type="chat_messages",
            recommended_training=[TrainingType.SFT],
            confidence=confidence,
            reasons=["Found messages column."],
            risks=[] if confidence > 0.9 else ["Need to verify messages are role/content chat turns."],
        )

    if "conversations" in cols:
        confidence = 0.9 if _sample_has_chat_messages(sample_rows, "conversations") else 0.82
        return DatasetClassification(
            format_type="sharegpt_conversations",
            recommended_training=[TrainingType.SFT],
            confidence=confidence,
            reasons=["Found conversations column."],
            risks=[] if confidence > 0.88 else ["Need to verify ShareGPT-style conversation schema."],
        )

    # Alpaca / prompt-completion / QA formats.
    if "instruction" in cols and _has_any(cols, OUTPUT_COLS):
        return DatasetClassification(
            format_type="alpaca_instruction",
            recommended_training=[TrainingType.SFT],
            confidence=0.95,
            reasons=["Found instruction plus output/answer/response column."],
            risks=[],
        )

    # Keep explicit question/answer datasets separate from generic prompt-completion
    # so downstream converters can map question -> instruction and answer -> output.
    if {"question", "answer"}.issubset(cols):
        rec = [TrainingType.SFT]
        if _has_any(cols, TEST_COLS):
            rec.append(TrainingType.RLVR)
        return DatasetClassification(
            format_type="qa",
            recommended_training=rec,
            confidence=0.82,
            reasons=["Found question and answer columns."],
            risks=["RLVR is only appropriate if answers can be automatically verified."],
        )

    if _has_any(cols, PROMPT_COLS) and _has_any(cols, {"completion", "response", "output", "answer"}):
        return DatasetClassification(
            format_type="prompt_completion",
            recommended_training=[TrainingType.SFT],
            confidence=0.86,
            reasons=["Found prompt/question/instruction plus completion/response/output/answer columns."],
            risks=["Need to verify target column is a high-quality assistant answer."],
        )

    # Verifiable code/math datasets.
    if _has_any(cols, PROMPT_COLS) and _has_any(cols, TEST_COLS):
        return DatasetClassification(
            format_type="verifiable_task",
            recommended_training=[TrainingType.SFT, TrainingType.RLVR],
            confidence=0.8,
            reasons=["Found prompt-like columns and test/verifier-like columns."],
            risks=["Reward/verifier code must be sandboxed before RL."],
        )

    # Raw text for continued pretraining.
    if _has_any(cols, TEXT_COLS):
        return DatasetClassification(
            format_type="raw_text",
            recommended_training=[TrainingType.CONTINUED_PRETRAINING],
            confidence=0.84,
            reasons=["Found text/content/document-like column."],
            risks=["Raw text is not directly suitable for SFT without transformation."],
        )

    return DatasetClassification(
        format_type="unknown",
        recommended_training=[TrainingType.UNKNOWN],
        confidence=0.2,
        reasons=["Could not infer training type from available columns."],
        risks=["Manual inspection or custom converter is required."],
    )
