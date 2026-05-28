from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from autopilot.data.schema import compact_sample_rows
from autopilot.models import DatasetReportItem, to_jsonable


def _item_final_score(item: DatasetReportItem) -> float:
    if item.adoption_decision is not None:
        return item.adoption_decision.final_score
    return item.score.suitability_score


def write_json_report(items: list[DatasetReportItem], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for item in items:
        obj = to_jsonable(item)
        obj["inspection"]["sample_rows"] = compact_sample_rows(item.inspection.sample_rows, max_rows=8, max_chars_per_value=900)
        if obj.get("web_overview") and obj["web_overview"].get("example_rows"):
            obj["web_overview"]["example_rows"] = compact_sample_rows(item.web_overview.example_rows if item.web_overview else [], max_rows=8, max_chars_per_value=900)
        data.append(obj)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_markdown_report(items: list[DatasetReportItem], output_path: str | Path, goal: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Dataset Collection Report\n")
    lines.append(f"**Goal:** {goal}\n")
    lines.append("| Rank | Dataset | Final | Action | Format | Training | vLLM probe | Files | Samples |")
    lines.append("|---:|---|---:|---|---|---|---:|---:|---:|")
    for i, item in enumerate(items, start=1):
        decision = item.adoption_decision
        final = _item_final_score(item)
        action = decision.action if decision else "review"
        training_types = decision.training_types if decision else item.classification.recommended_training
        training = ", ".join(t.value for t in training_types)
        file_count = len(item.web_overview.files) if item.web_overview else 0
        lines.append(
            f"| {i} | `{item.dataset_id}` | {final:.3f} | {action} | "
            f"{item.classification.format_type} | {training} | {len(item.model_trials)} | {file_count} | {len(item.inspection.sample_rows)} |"
        )

    for i, item in enumerate(items, start=1):
        overview = item.web_overview
        decision = item.adoption_decision
        lines.append(f"\n## {i}. {item.dataset_id}\n")
        if overview:
            lines.append(f"- **Hub URL:** {overview.hub_url}")
        if decision:
            lines.append(f"- **Decision:** {decision.action} / final={decision.final_score:.3f} / data_value={decision.data_value_score:.3f} / model_gap={decision.model_gap_score:.3f}")
            lines.append(f"- **Decision notes:** {decision.notes or 'n/a'}")
        lines.append(f"- **Rule score:** {item.score.suitability_score:.3f}")
        lines.append(f"- **Config/Split:** {item.inspection.config_name or (overview.selected_config if overview else None) or 'default'} / {item.inspection.split or (overview.selected_split if overview else None) or 'unknown'}")
        lines.append(f"- **Columns:** {', '.join(item.inspection.columns) if item.inspection.columns else 'unknown'}")
        lines.append(f"- **Format:** {item.classification.format_type}")
        rec = decision.training_types if decision else item.classification.recommended_training
        lines.append(f"- **Recommended training:** {', '.join(t.value for t in rec)}")
        lines.append(f"- **License:** {item.inspection.metadata.license or 'unknown'}")
        lines.append(f"- **Downloads/Likes:** {item.inspection.metadata.downloads or 0} / {item.inspection.metadata.likes or 0}")
        if item.inspection.load_error:
            lines.append(f"- **Load error:** `{item.inspection.load_error}`")
        if overview and overview.browse_error:
            lines.append(f"- **Browse notes:** `{overview.browse_error}`")

        if decision and decision.reasons:
            lines.append("- **Decision reasons:**")
            for reason in decision.reasons[:8]:
                lines.append(f"  - {reason}")
        else:
            lines.append("- **Rule reasons:**")
            for reason in item.score.reasons[:8]:
                lines.append(f"  - {reason}")

        if overview and overview.card_excerpt:
            excerpt = overview.card_excerpt[:1200]
            lines.append("- **Dataset card excerpt:**")
            lines.append("```text")
            lines.append(excerpt)
            lines.append("```")

        if overview and overview.files:
            lines.append("- **Visible files:**")
            for f in overview.files[:12]:
                size = f" ({f.size} bytes)" if f.size is not None else ""
                lines.append(f"  - `{f.path}`{size}")

        if item.model_trials:
            lines.append("- **vLLM probe:**")
            for trial in item.model_trials[:5]:
                lines.append(f"  - row={trial.row_index}, similarity={trial.similarity_to_reference}, error={trial.error or 'none'}")
                lines.append(f"    - prompt: {trial.prompt[:240].replace(chr(10), ' ')}")
                if trial.reference_answer:
                    lines.append(f"    - reference: {trial.reference_answer[:240].replace(chr(10), ' ')}")
                if trial.model_response:
                    lines.append(f"    - model: {trial.model_response[:240].replace(chr(10), ' ')}")

        sample_preview = compact_sample_rows(item.inspection.sample_rows, max_rows=2, max_chars_per_value=400)
        if sample_preview:
            lines.append("- **Sample preview:**")
            lines.append("```json")
            lines.append(json.dumps(sample_preview, ensure_ascii=False, indent=2))
            lines.append("```")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def sort_report_items(items: Iterable[DatasetReportItem]) -> list[DatasetReportItem]:
    return sorted(items, key=_item_final_score, reverse=True)
