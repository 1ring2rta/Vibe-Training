from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from autopilot.data.schema import normalize_col, value_to_text
from autopilot.models import TrainingType


PROMPT_CANDIDATES = ["instruction", "prompt", "question", "query", "task", "input"]
QUERY_CANDIDATES = ["input", "context", "passage", "background"]
OUTPUT_CANDIDATES = ["output", "response", "answer", "completion", "target", "assistant_response"]
TEXT_CANDIDATES = ["text", "content", "document", "article", "passage"]
SYSTEM_CANDIDATES = ["system", "system_prompt"]
CHOSEN_CANDIDATES = ["chosen", "chosen_response", "accepted", "winner", "positive", "better"]
REJECTED_CANDIDATES = ["rejected", "rejected_response", "refused", "loser", "negative", "worse"]
LABEL_CANDIDATES = ["kto_tag", "label", "score", "rating", "reward", "preference"]
MESSAGES_CANDIDATES = ["messages", "conversations"]


@dataclass
class ConversionResult:
    dataset_id: str
    dataset_name: str
    stage: str
    formatting: str
    data_file: str
    rows_written: int
    rows_skipped: int
    dataset_info_entry: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    rows_seen: int = 0
    sample_keys: list[str] = field(default_factory=list)


@dataclass
class ConvertedRow:
    row: dict[str, Any]
    formatting: str
    stage: str


def slugify_dataset_name(dataset_id: str, stage: str, prefix: str = "auto") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", dataset_id).strip("_").lower()
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        slug = "dataset"
    return f"{prefix}_{stage}_{slug}"


def _norm_key_map(row: dict[str, Any]) -> dict[str, str]:
    return {normalize_col(str(k)): str(k) for k in row.keys()}


def _get(row: dict[str, Any], candidates: Iterable[str]) -> tuple[str | None, Any | None]:
    mapping = _norm_key_map(row)
    for candidate in candidates:
        key = mapping.get(normalize_col(candidate))
        if key is not None:
            value = row.get(key)
            if value is not None:
                return key, value
    return None, None


def _text(value: Any, max_len: int = 12000) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ["content", "value", "text", "answer", "response", "output"]:
            if key in value and value[key] is not None:
                return _text(value[key], max_len=max_len)
        return json.dumps(value, ensure_ascii=False)[:max_len]
    if isinstance(value, list):
        # Chat messages are handled elsewhere. For generic fields keep a compact text version.
        return json.dumps(value, ensure_ascii=False)[:max_len]
    return value_to_text(value, max_len=max_len).strip()


def _truthy_label(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value > 0:
            return True
        if value <= 0:
            return False
    text = _text(value, max_len=200).strip().lower()
    if text in {"true", "yes", "y", "1", "positive", "good", "chosen", "accepted", "accept", "win", "winner"}:
        return True
    if text in {"false", "no", "n", "0", "negative", "bad", "rejected", "reject", "lose", "loser"}:
        return False
    return None


def _role_from_message(msg: dict[str, Any]) -> str:
    raw = str(msg.get("role", msg.get("from", ""))).strip().lower()
    if raw in {"user", "human", "customer"}:
        return "human"
    if raw in {"assistant", "gpt", "bot", "model"}:
        return "gpt"
    if raw in {"system"}:
        return "system"
    if raw in {"tool", "observation"}:
        return "observation"
    if raw in {"function", "function_call"}:
        return "function_call"
    return raw or "human"


def _content_from_message(msg: dict[str, Any]) -> str:
    for key in ["content", "value", "text"]:
        if key in msg:
            return _text(msg.get(key))
    return _text(msg)


def _normalize_conversation(value: Any) -> tuple[list[dict[str, str]], str | None]:
    """Normalize OpenAI/ShareGPT-style messages to ShareGPT from/value turns."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return [], None
    if not isinstance(value, list):
        return [], None

    conversations: list[dict[str, str]] = []
    system: str | None = None
    for raw in value:
        if not isinstance(raw, dict):
            continue
        role = _role_from_message(raw)
        content = _content_from_message(raw)
        if not content:
            continue
        if role == "system" and system is None:
            system = content
            continue
        conversations.append({"from": role, "value": content})
    return conversations, system


def _conversation_from_prompt_response(prompt: str, response: str) -> list[dict[str, str]]:
    return [{"from": "human", "value": prompt}, {"from": "gpt", "value": response}]


def _build_instruction_and_input(row: dict[str, Any]) -> tuple[str, str, str | None]:
    prompt_key, prompt_value = _get(row, PROMPT_CANDIDATES)
    system_key, system_value = _get(row, SYSTEM_CANDIDATES)
    instruction = _text(prompt_value)
    query = ""

    # If the primary prompt is instruction/question/prompt, a separate input/context field can be query.
    input_key, input_value = _get(row, QUERY_CANDIDATES)
    if input_key and input_key != prompt_key:
        query = _text(input_value)

    return instruction, query, _text(system_value) if system_key else None


def _convert_sft(row: dict[str, Any], format_type: str) -> ConvertedRow | None:
    msg_key, msg_value = _get(row, MESSAGES_CANDIDATES)
    if msg_key and msg_value is not None and format_type in {"chat_messages", "sharegpt_conversations"}:
        conversations, system = _normalize_conversation(msg_value)
        if len(conversations) >= 2:
            out: dict[str, Any] = {"conversations": conversations, "system": system or ""}
            return ConvertedRow(out, formatting="sharegpt", stage="sft")

    instruction, query, system = _build_instruction_and_input(row)
    _, output_value = _get(row, OUTPUT_CANDIDATES)
    output = _text(output_value)
    if not instruction or not output:
        return None
    out = {"instruction": instruction, "input": query, "output": output, "system": system or ""}
    return ConvertedRow(out, formatting="alpaca", stage="sft")


def _convert_preference(row: dict[str, Any]) -> ConvertedRow | None:
    chosen_key, chosen_value = _get(row, CHOSEN_CANDIDATES)
    rejected_key, rejected_value = _get(row, REJECTED_CANDIDATES)
    if not chosen_key or not rejected_key:
        return None

    msg_key, msg_value = _get(row, MESSAGES_CANDIDATES)
    if msg_key and msg_value is not None:
        conversations, system = _normalize_conversation(msg_value)
        if conversations:
            out: dict[str, Any] = {
                "conversations": conversations,
                "chosen": {"from": "gpt", "value": _text(chosen_value)},
                "rejected": {"from": "gpt", "value": _text(rejected_value)},
                "system": system or "",
            }
            if out["chosen"]["value"] and out["rejected"]["value"]:
                return ConvertedRow(out, formatting="sharegpt", stage="dpo")

    instruction, query, system = _build_instruction_and_input(row)
    chosen = _text(chosen_value)
    rejected = _text(rejected_value)
    if not instruction or not chosen or not rejected:
        return None
    out = {"instruction": instruction, "input": query, "chosen": chosen, "rejected": rejected, "system": system or ""}
    return ConvertedRow(out, formatting="alpaca", stage="dpo")


def _convert_kto(row: dict[str, Any]) -> ConvertedRow | None:
    instruction, query, system = _build_instruction_and_input(row)
    _, output_value = _get(row, OUTPUT_CANDIDATES)
    _, label_value = _get(row, LABEL_CANDIDATES)
    output = _text(output_value)
    label = _truthy_label(label_value)
    if not instruction or not output or label is None:
        return None
    out = {"instruction": instruction, "input": query, "output": output, "kto_tag": label, "system": system or ""}
    return ConvertedRow(out, formatting="alpaca", stage="kto")


def _convert_pretrain(row: dict[str, Any]) -> ConvertedRow | None:
    _, text_value = _get(row, TEXT_CANDIDATES)
    text = _text(text_value)
    if not text:
        return None
    return ConvertedRow({"text": text}, formatting="alpaca", stage="pt")


def choose_stage(training_types: list[str | TrainingType], format_type: str, requested_stage: str = "auto") -> str:
    if requested_stage and requested_stage != "auto":
        return requested_stage.lower()
    values = {t.value if isinstance(t, TrainingType) else str(t) for t in training_types}
    # Structural signals are stronger than an LLM's broad training_types list.
    # A preference-pair dataset that KIMI also marks as useful for SFT should
    # still become DPO/RM data, not get downgraded to SFT.
    if format_type in {"preference_pair", "preference_pair_without_explicit_prompt"}:
        return "dpo"
    if format_type in {"raw_text"}:
        return "pt"
    if "dpo" in values:
        return "dpo"
    if "kto" in values:
        return "kto"
    if "continued_pretraining" in values or "cpt" in values or "pt" in values:
        return "pt"
    if "reward_model" in values:
        return "rm"
    if "sft" in values:
        return "sft"
    return "sft"


def dataset_info_for(formatting: str, stage: str, file_name: str) -> dict[str, Any]:
    if formatting == "sharegpt":
        entry: dict[str, Any] = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
        }
        if stage in {"dpo", "rm", "orpo", "simpo"}:
            entry["ranking"] = True
            entry["columns"] = {"messages": "conversations", "chosen": "chosen", "rejected": "rejected", "system": "system"}
        return entry

    if stage == "pt":
        return {"file_name": file_name, "columns": {"prompt": "text"}}
    if stage in {"dpo", "rm", "orpo", "simpo"}:
        return {
            "file_name": file_name,
            "ranking": True,
            "columns": {"prompt": "instruction", "query": "input", "chosen": "chosen", "rejected": "rejected", "system": "system"},
        }
    if stage == "kto":
        return {
            "file_name": file_name,
            "columns": {"prompt": "instruction", "query": "input", "response": "output", "kto_tag": "kto_tag", "system": "system"},
        }
    return {
        "file_name": file_name,
        "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"},
    }


class LlamaFactoryConverter:
    """Convert inspected dataset rows into LLaMA-Factory local JSONL files."""

    def __init__(self, dataset_dir: str | Path, name_prefix: str = "auto") -> None:
        self.dataset_dir = Path(dataset_dir)
        self.name_prefix = name_prefix
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def convert_rows(
        self,
        dataset_id: str,
        rows: Iterable[dict[str, Any]],
        format_type: str,
        training_types: list[str | TrainingType],
        requested_stage: str = "auto",
        max_rows: int | None = None,
    ) -> ConversionResult:
        stage = choose_stage(training_types, format_type, requested_stage=requested_stage)
        dataset_name = slugify_dataset_name(dataset_id, stage=stage, prefix=self.name_prefix)
        data_file_name = f"{dataset_name}.jsonl"
        data_path = self.dataset_dir / data_file_name
        rows_written = 0
        rows_skipped = 0
        rows_seen = 0
        sample_keys: list[str] = []
        formatting_counts: dict[str, int] = {}
        warnings: list[str] = []

        with data_path.open("w", encoding="utf-8") as f:
            for row in rows:
                rows_seen += 1
                if not sample_keys and isinstance(row, dict):
                    sample_keys = [str(k) for k in row.keys()][:50]
                converted: ConvertedRow | None
                if stage in {"dpo", "rm", "orpo", "simpo"}:
                    converted = _convert_preference(row)
                elif stage == "kto":
                    converted = _convert_kto(row)
                elif stage == "pt":
                    converted = _convert_pretrain(row)
                else:
                    converted = _convert_sft(row, format_type=format_type)

                if converted is None:
                    rows_skipped += 1
                    continue
                formatting_counts[converted.formatting] = formatting_counts.get(converted.formatting, 0) + 1
                f.write(json.dumps(converted.row, ensure_ascii=False) + "\n")
                rows_written += 1
                if max_rows is not None and rows_written >= max_rows:
                    break

        if rows_written == 0:
            warnings.append("No rows could be converted; inspect schema or add a custom converter.")
            try:
                data_path.unlink()
            except FileNotFoundError:
                pass
        if len(formatting_counts) > 1:
            warnings.append(f"Mixed formatting detected: {formatting_counts}; using the most common format in dataset_info.")
        formatting = max(formatting_counts.items(), key=lambda kv: kv[1])[0] if formatting_counts else "alpaca"
        entry = dataset_info_for(formatting=formatting, stage=stage, file_name=data_file_name)
        return ConversionResult(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            stage=stage,
            formatting=formatting,
            data_file=str(data_path),
            rows_written=rows_written,
            rows_skipped=rows_skipped,
            dataset_info_entry=entry,
            warnings=warnings,
            rows_seen=rows_seen,
            sample_keys=sample_keys,
        )


def write_dataset_info(dataset_dir: str | Path, conversions: list[ConversionResult], merge_existing: bool = False) -> Path:
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / "dataset_info.json"
    data: dict[str, Any] = {}
    if merge_existing and path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    for result in conversions:
        if result.rows_written > 0:
            data[result.dataset_name] = result.dataset_info_entry
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
