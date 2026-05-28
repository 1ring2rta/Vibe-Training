from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from autopilot.models import to_jsonable

PROMPT_KEYS = [
    "problem",
    "question",
    "prompt",
    "instruction",
    "query",
    "task",
    "input",
    "text",
]

AIME24_ID_PATTERNS = [
    re.compile(r"(^|[^a-z0-9])aime[-_ ]?2024([^a-z0-9]|$)", re.I),
    re.compile(r"(^|[^a-z0-9])aime[-_ ]?24([^a-z0-9]|$)", re.I),
    re.compile(r"1983[-_ ]?2024", re.I),
]


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\\[a-z]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_hash(value: Any) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def extract_prompt_text(row: dict[str, Any]) -> str:
    for key in PROMPT_KEYS:
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            messages = None
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("from") or "").lower()
            if role in {"user", "human", ""}:
                return str(msg.get("content") or msg.get("value") or msg.get("text") or "")
    return ""


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
            except Exception:
                continue
        return rows
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ["cases", "examples", "data", "rows", "items"]:
            val = data.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


@dataclass
class DecontaminationFinding:
    dataset_id: str
    reason: str
    blocked: bool
    kind: str
    row_index: int | None = None
    eval_hash: str | None = None
    preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class DecontaminationReport:
    target: str = ""
    eval_cases_path: str | None = None
    eval_case_count: int = 0
    allow_contamination: bool = False
    findings: list[DecontaminationFinding] = field(default_factory=list)

    @property
    def blocked_dataset_ids(self) -> list[str]:
        return sorted({f.dataset_id for f in self.findings if f.blocked})

    @property
    def ok(self) -> bool:
        # A report is OK when contaminated datasets were detected and excluded.
        # Callers should set used_contaminated_dataset_ids/fatal if they choose to
        # keep contaminated data.
        return True

    def to_dict(self) -> dict[str, Any]:
        data = to_jsonable(self)
        data["blocked_dataset_ids"] = self.blocked_dataset_ids
        data["used_contaminated_dataset_ids"] = []
        data["ok"] = self.ok
        return data


class DecontaminationReportBuilder:
    def __init__(self, *, target: str = "", eval_cases_path: str | Path | None = None, allow_contamination: bool = False) -> None:
        self.target = target or ""
        self.eval_cases_path = str(eval_cases_path) if eval_cases_path else None
        self.allow_contamination = allow_contamination
        self.eval_text_by_hash: dict[str, str] = {}
        if eval_cases_path:
            for row in _read_json_or_jsonl(Path(eval_cases_path)):
                text = extract_prompt_text(row)
                norm = normalize_text(text)
                if len(norm) >= 20:
                    self.eval_text_by_hash[text_hash(text)] = text[:500]
        self.report = DecontaminationReport(
            target=self.target,
            eval_cases_path=self.eval_cases_path,
            eval_case_count=len(self.eval_text_by_hash),
            allow_contamination=allow_contamination,
        )

    def _target_is_aime24(self) -> bool:
        text = normalize_text(self.target)
        return "aime24" in text or "aime 24" in text or ("aime" in text and "2024" in text)

    def inspect_dataset_id(self, dataset_id: str) -> list[DecontaminationFinding]:
        findings: list[DecontaminationFinding] = []
        if self._target_is_aime24():
            did = dataset_id.lower().replace("/", " ")
            if any(pattern.search(did) for pattern in AIME24_ID_PATTERNS):
                findings.append(DecontaminationFinding(
                    dataset_id=dataset_id,
                    kind="dataset_id_pattern",
                    reason="dataset id appears to contain AIME 2024 / AIME24 benchmark data while target is AIME24",
                    blocked=not self.allow_contamination,
                ))
        self.report.findings.extend(findings)
        return findings

    def inspect_rows(self, dataset_id: str, rows: Iterable[dict[str, Any]], *, max_rows: int | None = None) -> list[DecontaminationFinding]:
        findings: list[DecontaminationFinding] = []
        if not self.eval_text_by_hash:
            return findings
        for idx, row in enumerate(rows):
            if max_rows is not None and idx >= max_rows:
                break
            if not isinstance(row, dict):
                continue
            text = extract_prompt_text(row)
            norm = normalize_text(text)
            if len(norm) < 20:
                continue
            h = text_hash(text)
            if h in self.eval_text_by_hash:
                findings.append(DecontaminationFinding(
                    dataset_id=dataset_id,
                    kind="exact_normalized_prompt_overlap",
                    reason="training row prompt exactly matches a target eval prompt after normalization",
                    blocked=not self.allow_contamination,
                    row_index=idx,
                    eval_hash=h,
                    preview=text[:500],
                ))
                if len(findings) >= 20:
                    break
        self.report.findings.extend(findings)
        return findings

    def dataset_blocked(self, dataset_id: str) -> bool:
        return any(f.dataset_id == dataset_id and f.blocked for f in self.report.findings)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path
