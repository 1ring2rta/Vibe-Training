from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopilot.models import to_jsonable


@dataclass
class DataAdapterSpec:
    dataset_id: str
    stage: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_format: str = "llamafactory_jsonl"
    converter_path: str | None = None
    smoke_test_path: str | None = None
    validation_report_path: str | None = None
    status: str = "planned"  # planned | generated | validated | failed
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self.__dict__)


class DataAdapterAgentPlan:
    """Plan dataset-specific adapters without polluting the core converter.

    Generic converters should run first.  When rows_written == 0 or validation
    fails, the agent writes a dataset-specific converter under the run artifact
    directory, validates it on sample rows, and only then promotes it to training.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_spec(self, dataset_id: str, *, stage: str, input_schema: dict[str, Any] | None = None, notes: str = "") -> DataAdapterSpec:
        safe = "".join(ch if ch.isalnum() else "_" for ch in dataset_id)[:120]
        adapter_dir = self.root / safe
        adapter_dir.mkdir(parents=True, exist_ok=True)
        spec = DataAdapterSpec(
            dataset_id=dataset_id,
            stage=stage,
            input_schema=input_schema or {},
            converter_path=str(adapter_dir / "convert.py"),
            smoke_test_path=str(adapter_dir / "test_samples.jsonl"),
            validation_report_path=str(adapter_dir / "validation_report.json"),
            notes=notes,
        )
        (adapter_dir / "adapter_spec.json").write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        readme = adapter_dir / "README.md"
        readme.write_text(
            f"# Data adapter plan: {dataset_id}\n\n"
            "Generic converter failed or was insufficient. A model/client may write `convert.py` here, but it must pass smoke validation before use.\n\n"
            "Required artifacts:\n"
            "- convert.py\n- adapter_spec.json\n- test_samples.jsonl\n- converted_preview.jsonl\n- validation_report.json\n",
            encoding="utf-8",
        )
        return spec
