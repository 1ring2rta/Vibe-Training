from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


def validate_jsonl(path: str | Path, max_rows: int = 1000) -> tuple[int, list[str]]:
    path = Path(path)
    errors: list[str] = []
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(obj, dict):
                errors.append(f"line {line_no}: row is not an object")
            count += 1
            if count >= max_rows:
                break
    return count, errors


def validate_dataset_info(dataset_info_path: str | Path) -> tuple[dict[str, Any], list[str]]:
    path = Path(dataset_info_path)
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, [f"dataset_info.json invalid JSON: {exc}"]
    if not isinstance(data, dict):
        return {}, ["dataset_info.json must be an object"]
    for name, entry in data.items():
        if not isinstance(entry, dict):
            errors.append(f"{name}: entry must be object")
            continue
        if "file_name" not in entry:
            errors.append(f"{name}: missing file_name")
        if "columns" not in entry:
            errors.append(f"{name}: missing columns")
    return data, errors


def _load_yaml_mapping(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    path = Path(path)
    if yaml is None:
        return {}, ["PyYAML is required to validate train YAML files"]
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, [f"invalid YAML: {exc}"]
    if not isinstance(data, dict):
        return {}, ["train YAML must be a mapping/object"]
    return data, []


def _require_type(data: dict[str, Any], key: str, typ: type | tuple[type, ...], errors: list[str]) -> None:
    if key not in data:
        errors.append(f"missing required key: {key}")
        return
    value = data.get(key)
    if not isinstance(value, typ):
        expected = typ.__name__ if isinstance(typ, type) else "/".join(t.__name__ for t in typ)
        errors.append(f"{key} must be {expected}, got {type(value).__name__}: {value!r}")


def _number_like(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def validate_train_yaml(path: str | Path) -> list[str]:
    """Validate the subset of LLaMA-Factory train YAML that Autopilot mutates.

    This catches common agent mistakes such as a broad sed replacement changing
    ``overwrite_output_dir: true`` into a string path.
    """
    path = Path(path)
    data, errors = _load_yaml_mapping(path)
    if errors:
        return errors

    _require_type(data, "model_name_or_path", str, errors)
    _require_type(data, "dataset_dir", str, errors)
    if "dataset" not in data:
        errors.append("missing required key: dataset")
    elif not isinstance(data["dataset"], (str, list)) or not data["dataset"]:
        errors.append("dataset must be a non-empty string or list")
    _require_type(data, "stage", str, errors)
    _require_type(data, "output_dir", str, errors)
    _require_type(data, "overwrite_output_dir", bool, errors)
    _require_type(data, "do_train", bool, errors)
    _require_type(data, "bf16", bool, errors)
    _require_type(data, "per_device_train_batch_size", int, errors)
    if isinstance(data.get("per_device_train_batch_size"), bool):
        errors.append("per_device_train_batch_size must be int, not bool")
    if "learning_rate" not in data or not _number_like(data.get("learning_rate")):
        errors.append(f"learning_rate must be numeric-like, got {data.get('learning_rate')!r}")
    if "num_train_epochs" not in data or not _number_like(data.get("num_train_epochs")):
        errors.append(f"num_train_epochs must be numeric-like, got {data.get('num_train_epochs')!r}")
    if data.get("stage") not in {"sft", "dpo", "kto", "rm", "pt", "ppo", "grpo", "orpo", "simpo"}:
        errors.append(f"unsupported/unknown stage: {data.get('stage')!r}")

    dataset_dir = Path(str(data.get("dataset_dir") or ""))
    if data.get("dataset_dir") and not dataset_dir.exists():
        errors.append(f"dataset_dir does not exist: {dataset_dir}")
    if dataset_dir.exists() and not (dataset_dir / "dataset_info.json").exists():
        errors.append(f"dataset_dir missing dataset_info.json: {dataset_dir}")
    return errors


def validate_prepared_dataset_dir(dataset_dir: str | Path) -> list[str]:
    dataset_dir = Path(dataset_dir)
    info, errors = validate_dataset_info(dataset_dir / "dataset_info.json")
    for name, entry in info.items():
        file_name = entry.get("file_name")
        if not file_name:
            continue
        data_path = dataset_dir / str(file_name)
        if not data_path.exists():
            errors.append(f"{name}: data file missing: {data_path}")
            continue
        if data_path.suffix == ".jsonl":
            count, row_errors = validate_jsonl(data_path)
            if count == 0:
                errors.append(f"{name}: data file has zero rows")
            errors.extend([f"{name}: {err}" for err in row_errors[:10]])
        elif data_path.suffix == ".json":
            try:
                data = json.loads(data_path.read_text(encoding="utf-8"))
                if not isinstance(data, list) or not data:
                    errors.append(f"{name}: json file should contain a non-empty list")
            except Exception as exc:
                errors.append(f"{name}: invalid json file: {exc}")
    return errors
