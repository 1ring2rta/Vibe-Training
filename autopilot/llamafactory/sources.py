from __future__ import annotations

import itertools
import os
from typing import Any

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]


def _report_value(item: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = item
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def rows_from_report_item(item: dict[str, Any], max_rows: int | None = None) -> list[dict[str, Any]]:
    rows = _report_value(item, ["inspection", "sample_rows"], []) or []
    rows = [row for row in rows if isinstance(row, dict)]
    return rows[:max_rows] if max_rows is not None else rows


def rows_from_hf_item(
    item: dict[str, Any],
    *,
    token: str | None = None,
    trust_remote_code: bool = False,
    max_rows: int | None = None,
    endpoint: str | None = None,
) -> list[dict[str, Any]]:
    if load_dataset is None:
        raise RuntimeError("The optional 'datasets' package is not installed.")
    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", endpoint.rstrip("/"))
    dataset_id = item.get("dataset_id")
    if not dataset_id:
        raise ValueError("Report item is missing dataset_id.")
    config_name = _report_value(item, ["inspection", "config_name"]) or _report_value(item, ["web_overview", "selected_config"])
    split = _report_value(item, ["inspection", "split"]) or _report_value(item, ["web_overview", "selected_split"]) or "train"
    kwargs: dict[str, Any] = {
        "path": dataset_id,
        "split": split,
        "streaming": True,
        "token": token,
        "trust_remote_code": trust_remote_code,
    }
    if config_name:
        kwargs["name"] = config_name
    ds = load_dataset(**kwargs)
    limit = max_rows if max_rows is not None else 10000
    return [dict(row) for row in itertools.islice(iter(ds), limit)]


def load_rows_for_item(
    item: dict[str, Any],
    *,
    source: str,
    token: str | None = None,
    trust_remote_code: bool = False,
    max_rows: int | None = None,
    endpoint: str | None = None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    """Return (rows, actual_source, error). source is report|hf|auto."""
    source = source.lower()
    if source == "report":
        return rows_from_report_item(item, max_rows=max_rows), "report", None
    if source == "hf":
        try:
            return rows_from_hf_item(item, token=token, trust_remote_code=trust_remote_code, max_rows=max_rows, endpoint=endpoint), "hf", None
        except Exception as exc:
            return [], "hf", f"{type(exc).__name__}: {exc}"
    if source == "auto":
        try:
            rows = rows_from_hf_item(item, token=token, trust_remote_code=trust_remote_code, max_rows=max_rows, endpoint=endpoint)
            if rows:
                return rows, "hf", None
        except Exception as exc:
            fallback = rows_from_report_item(item, max_rows=max_rows)
            return fallback, "report", f"HF load failed; used report samples. {type(exc).__name__}: {exc}"
        return rows_from_report_item(item, max_rows=max_rows), "report", "HF returned no rows; used report samples."
    raise ValueError("source must be one of: auto, hf, report")
