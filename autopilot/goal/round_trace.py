from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from autopilot.models import to_jsonable
from autopilot.llm.conversation_recorder import write_llamafactory_dataset_info

KIMI_JSONL_FILES = [
    "kimi_raw_calls.jsonl",
    "kimi_messages.jsonl",
    "kimi_sharegpt.jsonl",
    "kimi_multiturn_messages.jsonl",
    "kimi_multiturn_sharegpt.jsonl",
]
KIMI_STATE_FILES = ["kimi_session_state.json", "dataset_info.json"]


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _append_jsonl(path: Path, item: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(item), ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                rows.append(data)
        except Exception:
            rows.append({"raw": line})
    return rows


def _count_jsonl(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except Exception:
        return 0


def _metric_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _result_dict(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        try:
            data = result.to_dict()
            return data if isinstance(data, dict) else None
        except Exception:
            pass
    if is_dataclass(result):
        try:
            data = asdict(result)
            return data if isinstance(data, dict) else None
        except Exception:
            pass
    return None


def _case_results(eval_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(eval_data, dict):
        return []
    rows = eval_data.get("case_results") or []
    return [x for x in rows if isinstance(x, dict)]


def _tag_breakdown(eval_data: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _case_results(eval_data):
        tags = row.get("tags") or ["untagged"]
        if isinstance(tags, str):
            tags = [tags]
        score = _metric_number(row.get("score"))
        passed = row.get("passed")
        for tag in tags or ["untagged"]:
            key = str(tag or "untagged")
            bucket = out.setdefault(key, {"count": 0, "scored": 0, "score_sum": 0.0, "passed": 0, "failed": 0, "score": None})
            bucket["count"] += 1
            if score is not None:
                bucket["scored"] += 1
                bucket["score_sum"] += score
            if passed is True:
                bucket["passed"] += 1
            elif passed is False or row.get("error"):
                bucket["failed"] += 1
    for bucket in out.values():
        if bucket["scored"]:
            bucket["score"] = bucket["score_sum"] / bucket["scored"]
        bucket.pop("score_sum", None)
    return out


def summarize_evaluation(result: Any) -> dict[str, Any] | None:
    data = _result_dict(result)
    if data is None:
        return None
    cases = _case_results(data)
    scored = [x for x in cases if _metric_number(x.get("score")) is not None]
    failures = data.get("failures") or []
    return {
        "metric_name": data.get("metric_name"),
        "score": _metric_number(data.get("score")),
        "target": _metric_number(data.get("target")),
        "target_met": bool(data.get("target_met")),
        "case_count": len(cases),
        "scored_case_count": len(scored),
        "failure_count": len(failures) if isinstance(failures, list) else None,
        "notes": data.get("notes") or "",
        "by_tag": _tag_breakdown(data),
    }


def compute_metric_delta(pre_eval: Any, post_eval: Any) -> dict[str, Any]:
    pre = summarize_evaluation(pre_eval)
    post = summarize_evaluation(post_eval)
    pre_score = pre.get("score") if pre else None
    post_score = post.get("score") if post else None
    score_delta = None
    if pre_score is not None and post_score is not None:
        score_delta = post_score - pre_score
    tag_delta: dict[str, Any] = {}
    pre_tags = (pre or {}).get("by_tag") or {}
    post_tags = (post or {}).get("by_tag") or {}
    for tag in sorted(set(pre_tags) | set(post_tags)):
        a = pre_tags.get(tag, {})
        b = post_tags.get(tag, {})
        a_score = _metric_number(a.get("score"))
        b_score = _metric_number(b.get("score"))
        tag_delta[tag] = {
            "pre_score": a_score,
            "post_score": b_score,
            "score_delta": (b_score - a_score) if a_score is not None and b_score is not None else None,
            "pre_failures": a.get("failed"),
            "post_failures": b.get("failed"),
            "failure_delta": (b.get("failed") - a.get("failed")) if isinstance(a.get("failed"), int) and isinstance(b.get("failed"), int) else None,
        }
    return {
        "pre": pre,
        "post": post,
        "score_delta": score_delta,
        "target_met_before": bool(pre.get("target_met")) if pre else False,
        "target_met_after": bool(post.get("target_met")) if post else False,
        "target_met_changed": (bool(pre.get("target_met")) if pre else False) != (bool(post.get("target_met")) if post else False),
        "failure_delta": ((post or {}).get("failure_count") - (pre or {}).get("failure_count")) if isinstance((post or {}).get("failure_count"), int) and isinstance((pre or {}).get("failure_count"), int) else None,
        "by_tag_delta": tag_delta,
    }


def render_round_metrics_markdown(record: Mapping[str, Any]) -> str:
    delta = record.get("metric_delta") or {}
    pre = delta.get("pre") or {}
    post = delta.get("post") or {}
    lines = [
        f"# Round {record.get('round')} Metrics",
        "",
        f"Metric: `{record.get('metric_name')}` target `{record.get('target_value')}`",
        f"Training stage: `{record.get('train_stage')}`; training_ok: `{record.get('training_ok')}`",
        "",
        "## Before / After",
        f"- before score: `{pre.get('score')}`; target_met: `{pre.get('target_met')}`; failures: `{pre.get('failure_count')}`",
        f"- after score: `{post.get('score')}`; target_met: `{post.get('target_met')}`; failures: `{post.get('failure_count')}`",
        f"- score_delta: `{delta.get('score_delta')}`; failure_delta: `{delta.get('failure_delta')}`",
        "",
        "## Tag deltas",
    ]
    tag_delta = delta.get("by_tag_delta") or {}
    if not tag_delta:
        lines.append("No tag-level score deltas available.")
    else:
        for tag, item in sorted(tag_delta.items()):
            lines.append(f"- `{tag}`: {item.get('pre_score')} -> {item.get('post_score')} (delta={item.get('score_delta')}, failures {item.get('pre_failures')} -> {item.get('post_failures')})")
    if record.get("notes"):
        lines += ["", "## Notes", str(record.get("notes"))]
    return "\n".join(lines).rstrip() + "\n"


def write_round_metrics(
    round_dir: str | Path,
    *,
    round_idx: int,
    metric_name: str,
    target_value: float | None,
    pre_eval: Any,
    post_eval: Any,
    train_stage: str | None,
    training_ok: bool | None,
    training_result_path: str | None = None,
    model_under_test: Mapping[str, Any] | None = None,
    checkpoint_candidates: list[str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    root = Path(round_dir) / "metrics"
    delta = compute_metric_delta(pre_eval, post_eval)
    record = {
        "round": round_idx,
        "metric_name": metric_name,
        "target_value": target_value,
        "train_stage": train_stage,
        "training_ok": training_ok,
        "training_result_path": training_result_path,
        "model_under_test": dict(model_under_test or {}),
        "checkpoint_candidates": list(checkpoint_candidates or []),
        "metric_delta": delta,
        "notes": notes or "Post evaluation uses the currently configured evaluation endpoint. If the new checkpoint was not deployed, the after score is a same-endpoint regression check rather than a fine-tuned-checkpoint score.",
    }
    json_path = _write_json(root / "round_metrics.json", record)
    md_path = root / "round_metrics.md"
    md_path.write_text(render_round_metrics_markdown(record), encoding="utf-8")
    record["paths"] = {"json": str(json_path), "markdown": str(md_path)}
    _write_json(root / "round_metrics.json", record)
    return record


def _copy_existing(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def merge_kimi_trajectory_sources(
    *,
    round_idx: int,
    sources: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Merge KIMI conversation logs from goal-controller and child tools into one round bundle.

    The merged files keep trainable LLaMA-Factory-compatible message/sharegpt rows,
    while metadata marks the round and source so bad turns can be filtered later.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_records: list[dict[str, Any]] = []
    counts = {name: 0 for name in KIMI_JSONL_FILES}
    for source in sources:
        label = str(source.get("label") or source.get("name") or "source")
        root = Path(str(source.get("root") or ""))
        exists = root.exists()
        record: dict[str, Any] = {"label": label, "root": str(root), "exists": exists, "files": {}}
        if not exists:
            source_records.append(record)
            continue
        # Keep an exact copy per source for auditability.
        source_copy_dir = out / "sources" / label.replace("/", "_").replace(" ", "_")
        for filename in KIMI_JSONL_FILES + KIMI_STATE_FILES:
            copied = _copy_existing(root / filename, source_copy_dir / filename)
            if copied:
                record["files"][filename] = copied
        # Merge trainable JSONL files and add round/source metadata.
        for filename in KIMI_JSONL_FILES:
            src_file = root / filename
            dst_file = out / filename
            for item in _read_jsonl(src_file):
                meta = item.setdefault("metadata", {})
                if not isinstance(meta, dict):
                    item["metadata"] = {"original_metadata": meta}
                    meta = item["metadata"]
                meta.setdefault("round", round_idx)
                meta.setdefault("source_label", label)
                meta.setdefault("source_root", str(root))
                _append_jsonl(dst_file, item)
                counts[filename] += 1
        source_records.append(record)
    dataset_info = write_llamafactory_dataset_info(out)
    manifest = {
        "round": round_idx,
        "output_dir": str(out),
        "sources": source_records,
        "merged_files": {name: str(out / name) for name in KIMI_JSONL_FILES},
        "counts": {name: _count_jsonl(out / name) for name in KIMI_JSONL_FILES},
        "dataset_info": str(dataset_info),
    }
    manifest_path = _write_json(out / "manifest.json", manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest


def find_round_conversation_roots(round_dir: str | Path, *, include_goal_controller: bool = True) -> list[dict[str, Any]]:
    root = Path(round_dir)
    sources: list[dict[str, Any]] = []
    if include_goal_controller:
        sources.append({"label": "goal_controller", "root": str(root / "kimi_trajectory" / "goal_controller")})
    candidates = [
        ("collect", root / "collection" / ".autopilot" / "conversations"),
        ("prepare", root / "prepared" / ".autopilot" / "conversations"),
        ("training", root / "training" / ".autopilot" / "conversations"),
    ]
    for label, path in candidates:
        sources.append({"label": label, "root": str(path)})
    return sources


def load_round_metrics_history(output_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(output_dir)
    history_path = root / "round_metrics_history.json"
    if history_path.exists():
        try:
            data = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
        except Exception:
            pass
    records = []
    for path in sorted(root.glob("round_*/metrics/round_metrics.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                records.append(data)
        except Exception:
            continue
    return records


def write_round_metrics_history(output_dir: str | Path, records: list[dict[str, Any]]) -> Path:
    path = Path(output_dir) / "round_metrics_history.json"
    return _write_json(path, records)
