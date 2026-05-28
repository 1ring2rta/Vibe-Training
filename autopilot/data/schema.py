from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any


def normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def normalized_columns(columns: list[str]) -> set[str]:
    return {normalize_col(c) for c in columns}


def first_present(cols: set[str], candidates: set[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def value_to_text(value: Any, max_len: int = 8000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:max_len]
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)[:max_len]
    except Exception:
        return str(value)[:max_len]


def row_text(row: dict[str, Any], columns: list[str] | None = None, max_len: int = 12000) -> str:
    if columns:
        parts = [value_to_text(row.get(c), max_len=max_len) for c in columns]
    else:
        parts = [value_to_text(v, max_len=max_len) for v in row.values()]
    return "\n".join(p for p in parts if p).strip()[:max_len]


def infer_column_types(sample_rows: list[dict[str, Any]]) -> dict[str, str]:
    types: dict[str, Counter[str]] = {}
    for row in sample_rows:
        for k, v in row.items():
            types.setdefault(k, Counter())[type(v).__name__] += 1
    return {k: counter.most_common(1)[0][0] for k, counter in types.items()}


def looks_like_chat_messages(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    if not isinstance(first, dict):
        return False
    keys = {normalize_col(k) for k in first.keys()}
    return bool({"role", "content"}.issubset(keys) or {"from", "value"}.issubset(keys))


def column_value_examples(sample_rows: list[dict[str, Any]], column: str, limit: int = 3) -> list[str]:
    examples: list[str] = []
    for row in sample_rows:
        if column in row and row[column] is not None:
            text = value_to_text(row[column], max_len=240).replace("\n", " ")
            if text:
                examples.append(text)
        if len(examples) >= limit:
            break
    return examples


def compact_sample_rows(sample_rows: list[dict[str, Any]], max_rows: int = 5, max_chars_per_value: int = 500) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in sample_rows[:max_rows]:
        new_row: dict[str, Any] = {}
        for k, v in row.items():
            text = value_to_text(v, max_len=max_chars_per_value)
            if len(text) > max_chars_per_value:
                text = text[:max_chars_per_value] + "...[truncated]"
            new_row[k] = text
        compact.append(new_row)
    return compact
