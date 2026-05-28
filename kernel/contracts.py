from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopilot.kernel.action_schema import ExpectedArtifact
from autopilot.llamafactory.validate import validate_train_yaml
from autopilot.models import to_jsonable
from autopilot.runtime.paths import normalize_workspace_path, workspace_relative_path


@dataclass
class ContractCheck:
    path: str
    kind: str
    required: bool
    ok: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class ContractReport:
    ok: bool
    checks: list[ContractCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def _resolve(root: Path, rel: str) -> list[Path]:
    normalized = normalize_workspace_path(root, rel)
    pattern = str(normalized)
    if any(ch in rel for ch in "*?["):
        return [Path(x) for x in glob.glob(pattern)]
    return [normalized]


def _load_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"invalid json: {exc}"


def _check_evaluation_result(path: Path) -> tuple[bool, str]:
    data, err = _load_json(path)
    if err:
        return False, err
    if not isinstance(data, dict):
        return False, "evaluation result must be a JSON object"
    score = data.get("score", data.get("accuracy", data.get("metric_value")))
    if score is None:
        return False, "missing score/accuracy/metric_value"
    try:
        float(score)
    except Exception:
        return False, f"score is not numeric: {score!r}"
    case_count = data.get("case_count", data.get("total"))
    if case_count is None:
        details = data.get("details") or data.get("case_results") or []
        case_count = len(details) if isinstance(details, list) else 0
    try:
        if int(case_count) <= 0:
            return False, "case_count/total must be positive"
    except Exception:
        return False, f"case_count/total is not an integer: {case_count!r}"
    source = str(data.get("eval_source") or data.get("source") or "").strip()
    if not source:
        return False, "missing eval_source/source"
    if "target_met" not in data:
        return False, "missing target_met boolean"
    return True, "valid evaluation_result schema"


def _check_decontamination_report(path: Path) -> tuple[bool, str]:
    data, err = _load_json(path)
    if err:
        return False, err
    if not isinstance(data, dict):
        return False, "decontamination report must be a JSON object"
    findings = data.get("findings") or []
    if not isinstance(findings, list):
        return False, "findings must be a list"
    blocked = [f for f in findings if isinstance(f, dict) and f.get("blocked")]
    used = data.get("used_contaminated_dataset_ids") or []
    if used and not data.get("allow_contamination"):
        return False, f"contaminated datasets were used: {', '.join(map(str, used[:5]))}"
    if data.get("fatal") or data.get("ok") is False:
        return False, "decontamination report marked fatal/ok=false"
    return True, f"valid decontamination report; findings={len(findings)}, blocked_from_use={len(blocked)}"


def _check_one(root: Path, contract: ExpectedArtifact) -> ContractCheck:
    paths = _resolve(root, contract.path)
    existing = [p for p in paths if p.exists()]
    kind = contract.kind
    if kind == "file_exists":
        ok = bool(existing and any(p.is_file() for p in existing))
        reason = "file exists" if ok else "missing file"
    elif kind == "directory_exists":
        ok = bool(existing and any(p.is_dir() for p in existing))
        reason = "directory exists" if ok else "missing directory"
    elif kind == "json_exists":
        ok = False
        reason = "missing json"
        for p in existing:
            if p.is_file():
                _, err = _load_json(p)
                ok = err is None
                reason = "valid json" if ok else str(err)
                if ok:
                    break
    elif kind == "jsonl_nonempty":
        ok = False
        reason = "missing/nonempty jsonl"
        for p in existing:
            if p.is_file():
                lines = [ln for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
                ok = len(lines) >= int(contract.min_count or 1)
                reason = f"{len(lines)} nonempty lines"
                if ok:
                    break
    elif kind == "train_yaml_exists":
        files = []
        for p in existing:
            if p.is_dir():
                files.extend(list(p.glob("train_*.yaml")) + list(p.glob("train_*.yml")))
            elif p.name.startswith("train_") and p.suffix in {".yaml", ".yml"}:
                files.append(p)
        ok = len(files) >= int(contract.min_count or 1)
        reason = f"{len(files)} train yaml files"
    elif kind == "train_yaml_valid":
        ok = False
        reason = "missing train yaml"
        files: list[Path] = []
        for p in existing:
            if p.is_dir():
                files.extend(list(p.glob("train_*.yaml")) + list(p.glob("train_*.yml")))
            elif p.suffix in {".yaml", ".yml"}:
                files.append(p)
        errors: list[str] = []
        for p in files:
            errs = validate_train_yaml(p)
            if errs:
                errors.extend([f"{p}: {err}" for err in errs[:5]])
            else:
                ok = True
                break
        reason = "valid train yaml" if ok else "; ".join(errors[:8]) or reason
    elif kind == "evaluation_result_valid":
        ok = False
        reason = "missing evaluation result"
        for p in existing:
            if p.is_file():
                ok, reason = _check_evaluation_result(p)
                if ok:
                    break
    elif kind == "decontamination_report_valid":
        ok = False
        reason = "missing decontamination report"
        for p in existing:
            if p.is_file():
                ok, reason = _check_decontamination_report(p)
                if ok:
                    break
    elif kind == "checkpoint_exists":
        files = []
        for p in existing:
            if p.is_dir():
                files.extend(list(p.rglob("adapter_config.json")) + list(p.rglob("pytorch_model*.bin")) + list(p.rglob("model*.safetensors")))
            elif p.is_file():
                files.append(p)
        ok = len(files) >= int(contract.min_count or 1)
        reason = f"{len(files)} checkpoint-ish files"
    else:
        ok = bool(existing)
        reason = f"generic existence check for {kind}"
    if not contract.required and not ok:
        return ContractCheck(contract.path, kind, contract.required, True, f"optional: {reason}")
    return ContractCheck(contract.path, kind, contract.required, ok, reason)


def normalize_expected_artifacts(root: str | Path, contracts: list[ExpectedArtifact]) -> list[ExpectedArtifact]:
    normalized: list[ExpectedArtifact] = []
    for contract in contracts:
        normalized.append(ExpectedArtifact(
            path=workspace_relative_path(root, contract.path),
            kind=contract.kind,
            required=contract.required,
            description=contract.description,
            min_count=contract.min_count,
        ))
    return normalized


def validate_contracts(root: str | Path, contracts: list[ExpectedArtifact]) -> ContractReport:
    r = Path(root)
    checks = [_check_one(r, c) for c in normalize_expected_artifacts(r, contracts)]
    return ContractReport(ok=all(c.ok for c in checks), checks=checks)
