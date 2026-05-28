from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopilot.models import to_jsonable

AIME_TARGET_RE = re.compile(r"(?i)(aime[_\-\s]?2024|aime[_\-\s]?24|aime[_\-\s]?i|aime[_\-\s]?ii|1983\s*[-_]\s*2024|aime)")
TRAIN_PATH_RE = re.compile(r"(?i)(train|training|prepared|dataset_info|sft|dpo|kto|grpo|rlvr|round_\d+|data_round)")
EVAL_ONLY_PATH_RE = re.compile(r"(?i)(eval_programs|evaluation|eval/|aime24_cases|benchmark|predictions|results)")


@dataclass
class ArtifactPolicyViolation:
    path: str
    reason: str
    severity: str = "error"
    snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class ArtifactPolicyReport:
    ok: bool
    violations: list[ArtifactPolicyViolation] = field(default_factory=list)
    warnings: list[ArtifactPolicyViolation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def goal_targets_aime24(goal: str = "", target: str = "") -> bool:
    text = f"{goal} {target}".lower()
    return "aime" in text and ("24" in text or "2024" in text)


def is_eval_only_path(rel: str) -> bool:
    rel_norm = rel.replace("\\", "/")
    return bool(EVAL_ONLY_PATH_RE.search(rel_norm))


def is_train_like_path(rel: str) -> bool:
    rel_norm = rel.replace("\\", "/")
    return bool(TRAIN_PATH_RE.search(rel_norm)) and not is_eval_only_path(rel_norm)


def scan_benchmark_leakage(root: str | Path, *, goal: str = "", target: str = "", max_bytes: int = 2_000_000) -> ArtifactPolicyReport:
    """Scan run artifacts for target benchmark leakage into train-like paths.

    This is an artifact firewall: regardless of which shell commands were used,
    training manifests/data cannot reference target benchmark datasets.
    """
    root = Path(root)
    violations: list[ArtifactPolicyViolation] = []
    warnings: list[ArtifactPolicyViolation] = []
    if not root.exists() or not goal_targets_aime24(goal, target):
        return ArtifactPolicyReport(ok=True)

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = str(path.relative_to(root))
        except Exception:
            rel = str(path)
        if any(part in {".git", "__pycache__", ".venv"} for part in path.parts):
            continue
        lower_rel = rel.lower()
        if is_eval_only_path(rel):
            # Eval-only files may contain target benchmark cases; label them.
            continue
        should_scan = is_train_like_path(rel)
        if not should_scan:
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        match = AIME_TARGET_RE.search(rel) or AIME_TARGET_RE.search(text)
        if not match:
            continue
        snippet = ""
        if AIME_TARGET_RE.search(text):
            m = AIME_TARGET_RE.search(text)
            if m:
                start = max(m.start() - 120, 0)
                end = min(m.end() + 120, len(text))
                snippet = text[start:end].replace("\n", " ")
        violations.append(ArtifactPolicyViolation(path=rel, reason="target benchmark term appears in train-like artifact", snippet=snippet[:500]))

    return ArtifactPolicyReport(ok=not violations, violations=violations, warnings=warnings)


def label_eval_only(root: str | Path, path: str | Path, *, reason: str = "trusted benchmark materialized for evaluation") -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    labels = root / ".autopilot" / "provenance" / "labels.jsonl"
    labels.parent.mkdir(parents=True, exist_ok=True)
    p = Path(path)
    rel = str(p if not p.is_absolute() else p)
    record = {"path": rel, "label": "eval_only", "reason": reason}
    with labels.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
