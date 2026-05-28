from __future__ import annotations

import re
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.llm.openai_compatible import OpenAICompatibleChatClient
from autopilot.models import ModelTrial, PromptExample

PROMPT_FIELDS = ["instruction", "question", "prompt", "query", "input", "problem", "task"]
AUX_INPUT_FIELDS = ["input", "context", "passage", "background"]
ANSWER_FIELDS = ["output", "answer", "response", "completion", "target", "chosen", "accepted", "reference", "solution"]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _stringify(value: Any, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    return str(value)[:limit]


def _get_by_norm(row: dict[str, Any], candidates: list[str]) -> tuple[str | None, Any]:
    norm_map = {_norm(k): k for k in row.keys()}
    for candidate in candidates:
        key = norm_map.get(_norm(candidate))
        if key is not None:
            return key, row.get(key)
    return None, None


def _extract_from_messages(messages: Any, row_index: int) -> PromptExample | None:
    if not isinstance(messages, list):
        return None
    user_turns: list[str] = []
    assistant_turns: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or msg.get("from") or msg.get("speaker") or "").lower()
        content = _stringify(msg.get("content") or msg.get("value") or msg.get("text"))
        if not content:
            continue
        if role in {"user", "human", "client", "prompter"}:
            user_turns.append(content)
        elif role in {"assistant", "gpt", "bot", "model"}:
            assistant_turns.append(content)
    if not user_turns:
        return None
    return PromptExample(
        row_index=row_index,
        prompt=user_turns[-1],
        reference_answer=assistant_turns[-1] if assistant_turns else None,
        source_fields={"prompt": "messages[-user]", "reference_answer": "messages[-assistant]"},
    )


def extract_prompt_example(row: dict[str, Any], row_index: int = 0) -> PromptExample | None:
    # Chat formats first.
    for key, value in row.items():
        if _norm(key) in {"messages", "conversations"}:
            found = _extract_from_messages(value, row_index=row_index)
            if found is not None:
                return found

    prompt_key, prompt_value = _get_by_norm(row, PROMPT_FIELDS)
    answer_key, answer_value = _get_by_norm(row, ANSWER_FIELDS)
    if not prompt_value:
        return None

    prompt = _stringify(prompt_value).strip()
    if prompt_key and _norm(prompt_key) == "instruction":
        aux_key, aux_value = _get_by_norm(row, AUX_INPUT_FIELDS)
        if aux_key and aux_key != prompt_key and aux_value:
            aux_text = _stringify(aux_value).strip()
            if aux_text and aux_text != prompt:
                prompt = f"{prompt}\n\n输入：{aux_text}"

    if not prompt or len(prompt) < 4:
        return None

    reference = _stringify(answer_value).strip() if answer_value is not None else None
    return PromptExample(
        row_index=row_index,
        prompt=prompt,
        reference_answer=reference or None,
        source_fields={"prompt": prompt_key or "", "reference_answer": answer_key or ""},
    )


def build_probe_examples(rows: list[dict[str, Any]], max_examples: int = 3) -> list[PromptExample]:
    examples: list[PromptExample] = []
    for i, row in enumerate(rows):
        example = extract_prompt_example(row, row_index=i)
        if example is not None:
            examples.append(example)
        if len(examples) >= max_examples:
            break
    return examples


def text_similarity(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    a = a.lower()
    b = b.lower()
    # Mixed Chinese/English tokenization: keep English words and individual CJK chars.
    def toks(s: str) -> set[str]:
        out = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", s))
        return {t for t in out if t.strip()}

    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return None
    precision = len(ta & tb) / len(ta)
    recall = len(ta & tb) / len(tb)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def probe_model_on_examples(
    client: "OpenAICompatibleChatClient",
    examples: list[PromptExample],
    system_prompt: str = "你是一个严谨、准确的助手。请直接回答用户问题。",
    temperature: float = 0.0,
    max_tokens: int = 768,
) -> list[ModelTrial]:
    trials: list[ModelTrial] = []
    for example in examples:
        started = time.perf_counter()
        try:
            response = client.chat(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": example.prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = time.perf_counter() - started
            sim = text_similarity(response, example.reference_answer)
            trials.append(
                ModelTrial(
                    row_index=example.row_index,
                    prompt=example.prompt,
                    reference_answer=example.reference_answer,
                    model_response=response,
                    latency_seconds=round(latency, 4),
                    similarity_to_reference=sim,
                )
            )
        except Exception as exc:
            latency = time.perf_counter() - started
            trials.append(
                ModelTrial(
                    row_index=example.row_index,
                    prompt=example.prompt,
                    reference_answer=example.reference_answer,
                    model_response=None,
                    latency_seconds=round(latency, 4),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return trials
