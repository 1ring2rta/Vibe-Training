from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from autopilot.llamafactory.converter import ConversionResult


STAGE_DEFAULT_LR = {
    "sft": "1.0e-4",
    "dpo": "5.0e-6",
    "rm": "1.0e-5",
    "kto": "5.0e-6",
    "pt": "5.0e-5",
}


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or text.lower() in {"true", "false", "null"} or any(ch in text for ch in [":", "#", "\n", "\t"]):
        return repr(text)
    return text


def write_simple_yaml(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {_yaml_scalar(sub_value)}")
        elif isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {_yaml_scalar(item)}")
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def model_slug(model_name_or_path: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name_or_path).strip("_").lower()
    return slug or "model"


def build_training_config(
    *,
    stage: str,
    dataset_names: list[str],
    dataset_dir: str | Path,
    model_name_or_path: str,
    template: str,
    output_root: str | Path,
    finetuning_type: str = "lora",
    cutoff_len: int = 2048,
    max_samples: int | None = None,
    learning_rate: str | None = None,
    num_train_epochs: float = 3.0,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    bf16: bool = True,
    report_to: str = "none",
) -> dict[str, Any]:
    stage = stage.lower()
    output_dir = Path(output_root) / model_slug(model_name_or_path) / finetuning_type / stage
    cfg: dict[str, Any] = {
        "model_name_or_path": model_name_or_path,
        "trust_remote_code": True,
        "stage": stage,
        "do_train": True,
        "finetuning_type": finetuning_type,
    }
    if finetuning_type == "lora":
        cfg.update({"lora_rank": 8, "lora_target": "all"})
    cfg.update(
        {
            "dataset_dir": str(Path(dataset_dir).resolve()),
            "dataset": ",".join(dataset_names),
            "template": template,
            "cutoff_len": cutoff_len,
            "preprocessing_num_workers": 8,
            "dataloader_num_workers": 4,
            "output_dir": str(output_dir),
            "logging_steps": 10,
            "save_steps": 500,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "save_only_model": False,
            "report_to": report_to,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate or STAGE_DEFAULT_LR.get(stage, "1.0e-4"),
            "num_train_epochs": num_train_epochs,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "bf16": bf16,
            "ddp_timeout": 180000000,
            "resume_from_checkpoint": None,
        }
    )
    if max_samples is not None:
        cfg["max_samples"] = max_samples
    return cfg


def generate_training_configs(
    conversions: list[ConversionResult],
    *,
    config_dir: str | Path,
    dataset_dir: str | Path,
    model_name_or_path: str,
    template: str,
    output_root: str | Path,
    finetuning_type: str = "lora",
    cutoff_len: int = 2048,
    max_samples: int | None = None,
    report_to: str = "none",
) -> dict[str, Path]:
    by_stage: dict[str, list[str]] = {}
    for result in conversions:
        if result.rows_written <= 0:
            continue
        by_stage.setdefault(result.stage, []).append(result.dataset_name)

    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for stage, dataset_names in sorted(by_stage.items()):
        cfg = build_training_config(
            stage=stage,
            dataset_names=dataset_names,
            dataset_dir=dataset_dir,
            model_name_or_path=model_name_or_path,
            template=template,
            output_root=output_root,
            finetuning_type=finetuning_type,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            report_to=report_to,
        )
        path = config_dir / f"train_{stage}.yaml"
        write_simple_yaml(cfg, path)
        paths[stage] = path
    return paths
