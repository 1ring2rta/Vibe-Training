from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from autopilot.models import to_jsonable


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_dumps(data: Any) -> str:
    return json.dumps(to_jsonable(data), ensure_ascii=False, sort_keys=True)


def stable_hash(data: Any) -> str:
    return hashlib.sha256(_json_dumps(data).encode("utf-8")).hexdigest()[:16]


def append_jsonl(path: str | Path, row: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def atomic_write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)


def _clip(value: Any, max_chars: int | None) -> Any:
    if max_chars is None or max_chars <= 0:
        return value
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 80] + f"\n...[truncated {len(value) - (max_chars - 80)} chars]"
    if isinstance(value, list):
        return [_clip(x, max_chars) for x in value]
    if isinstance(value, dict):
        return {k: _clip(v, max_chars) for k, v in value.items()}
    return value


def _sharegpt_messages(messages: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    role_map = {"system": "system", "user": "human", "assistant": "gpt", "tool": "observation"}
    out: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if content is None:
            continue
        text = str(content)
        if not text.strip():
            continue
        out.append({"from": role_map.get(role, role), "value": text})
    return out


def assistant_message_from_response(response: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.get("content") or "",
        "reasoning_content": response.get("reasoning_content"),
        "tool_calls": response.get("tool_calls") or [],
        "finish_reason": response.get("finish_reason"),
    }


@dataclass
class FrontierTrajectoryRecorder:
    """Append-only recorder for *every* frontier-model request and response.

    This recorder is intentionally lower level than the older conversation logger:
    it stores exact request payloads, tool schemas, parameters, raw response bodies,
    reasoning_content, tool calls, errors, and latency.  The trainable JSONL files
    are derived artifacts, not the source of truth.
    """

    root: Path
    enabled: bool = True
    max_chars_per_field: int = 0
    write_trainable: bool = True
    run_id: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")

    @classmethod
    def from_settings(cls, settings: Any, root: str | Path | None = None) -> "FrontierTrajectoryRecorder | None":
        raw_config = settings.raw_config if hasattr(settings, "raw_config") and isinstance(settings.raw_config, dict) else {}
        cfg = raw_config.get("frontier_trajectory") or raw_config.get("trajectory") or raw_config.get("model_trajectory") or {}
        if not isinstance(cfg, dict):
            cfg = {}
        enabled = cfg.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() not in {"0", "false", "no", "off"}
        if not bool(enabled):
            return None
        root_value = root or cfg.get("root") or os.getenv("AUTOPILOT_FRONTIER_TRAJECTORY_ROOT") or ".autopilot/frontier_trajectory"
        max_chars = cfg.get("max_chars_per_field", cfg.get("max_chars", 0))
        try:
            max_chars_i = int(max_chars or 0)
        except Exception:
            max_chars_i = 0
        run_id = str(cfg.get("run_id") or os.getenv("AUTOPILOT_RUN_ID") or f"run-{uuid.uuid4().hex[:12]}")
        return cls(root=Path(root_value), enabled=bool(enabled), max_chars_per_field=max_chars_i, write_trainable=bool(cfg.get("write_trainable", True)), run_id=run_id)

    @property
    def calls_path(self) -> Path:
        return self.root / "frontier_model_calls.jsonl"

    @property
    def requests_path(self) -> Path:
        return self.root / "frontier_model_requests.jsonl"

    @property
    def responses_path(self) -> Path:
        return self.root / "frontier_model_responses.jsonl"

    @property
    def errors_path(self) -> Path:
        return self.root / "frontier_model_errors.jsonl"

    @property
    def messages_path(self) -> Path:
        return self.root / "frontier_messages.jsonl"

    @property
    def sharegpt_path(self) -> Path:
        return self.root / "frontier_sharegpt.jsonl"

    @property
    def message_completions_path(self) -> Path:
        return self.root / "frontier_message_completions.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def paths(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "calls": str(self.calls_path),
            "requests": str(self.requests_path),
            "responses": str(self.responses_path),
            "errors": str(self.errors_path),
            "messages": str(self.messages_path),
            "sharegpt": str(self.sharegpt_path),
            "message_completions": str(self.message_completions_path),
            "manifest": str(self.manifest_path),
        }

    def _write_manifest(self) -> None:
        atomic_write_json(
            self.manifest_path,
            {
                "run_id": self.run_id,
                "updated_at": utc_now(),
                "description": "Append-only frontier model trajectory. Every request and every response is preserved.",
                "files": self.paths(),
                "trainable_formats": {
                    "frontier_messages": {"file_name": "frontier_messages.jsonl", "formatting": "sharegpt", "columns": {"messages": "messages"}},
                    "frontier_sharegpt": {"file_name": "frontier_sharegpt.jsonl", "formatting": "sharegpt", "columns": {"messages": "conversations"}, "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt", "system_tag": "system"}},
                    "frontier_message_completions": {"file_name": "frontier_message_completions.jsonl", "formatting": "message_completion", "columns": {"messages": "messages", "completion": "completion"}},
                },
            },
        )

    def begin_call(
        self,
        *,
        provider: str,
        client_name: str,
        model: str,
        purpose: str | None,
        request_payload: Mapping[str, Any],
        tools: list[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"call_id": uuid.uuid4().hex, "started_at": time.time()}
        call_id = uuid.uuid4().hex
        started_at = time.time()
        row = {
            "event": "request",
            "timestamp": utc_now(),
            "run_id": self.run_id,
            "call_id": call_id,
            "provider": provider,
            "client_name": client_name,
            "model": model,
            "purpose": purpose,
            "request_hash": stable_hash(request_payload),
            "request": _clip(dict(request_payload), self.max_chars_per_field),
            "tools": _clip(list(tools or []), self.max_chars_per_field),
            "metadata": dict(metadata or {}),
        }
        append_jsonl(self.requests_path, row)
        self._write_manifest()
        return {"call_id": call_id, "started_at": started_at, "request_row": row}

    def end_call(
        self,
        *,
        handle: Mapping[str, Any],
        provider: str,
        client_name: str,
        model: str,
        purpose: str | None,
        request_payload: Mapping[str, Any],
        response: Mapping[str, Any],
        raw_response: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        call_id = str(handle.get("call_id") or uuid.uuid4().hex)
        started = float(handle.get("started_at") or time.time())
        elapsed = round(time.time() - started, 4)
        response_row = {
            "event": "response",
            "timestamp": utc_now(),
            "run_id": self.run_id,
            "call_id": call_id,
            "provider": provider,
            "client_name": client_name,
            "model": model,
            "purpose": purpose,
            "elapsed_seconds": elapsed,
            "response_hash": stable_hash(response),
            "response": _clip(dict(response), self.max_chars_per_field),
            "raw_response": _clip(raw_response, self.max_chars_per_field),
            "metadata": dict(metadata or {}),
        }
        append_jsonl(self.responses_path, response_row)
        call_row = {
            "timestamp": utc_now(),
            "run_id": self.run_id,
            "call_id": call_id,
            "provider": provider,
            "client_name": client_name,
            "model": model,
            "purpose": purpose,
            "elapsed_seconds": elapsed,
            "request": _clip(dict(request_payload), self.max_chars_per_field),
            "response": _clip(dict(response), self.max_chars_per_field),
            "metadata": dict(metadata or {}),
        }
        append_jsonl(self.calls_path, call_row)
        if self.write_trainable:
            messages = list(request_payload.get("messages") or [])
            assistant_msg = assistant_message_from_response(response)
            train_messages: list[dict[str, Any]] = []
            for msg in messages + [assistant_msg]:
                content = msg.get("content") or ""
                if msg.get("role") == "assistant" and msg.get("reasoning_content"):
                    content = f"[reasoning_content]\n{msg.get('reasoning_content')}\n\n[content]\n{content}"
                train_messages.append({"role": str(msg.get("role") or "user"), "content": str(content)})
            meta = {"run_id": self.run_id, "call_id": call_id, "provider": provider, "client_name": client_name, "model": model, "purpose": purpose}
            append_jsonl(self.messages_path, {"messages": train_messages, "metadata": meta})
            append_jsonl(self.sharegpt_path, {"conversations": _sharegpt_messages(train_messages), "metadata": meta})
            append_jsonl(
                self.message_completions_path,
                {
                    "messages": [dict(x) for x in messages],
                    "completion": response.get("content") or "",
                    "reasoning_content": response.get("reasoning_content"),
                    "assistant": assistant_msg,
                    "metadata": meta,
                },
            )
        self._write_manifest()
        return {"call_id": call_id, "paths": self.paths()}


    def error_call(
        self,
        *,
        handle: Mapping[str, Any],
        provider: str,
        client_name: str,
        model: str,
        purpose: str | None,
        request_payload: Mapping[str, Any],
        error: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        call_id = str(handle.get("call_id") or uuid.uuid4().hex)
        started = float(handle.get("started_at") or time.time())
        row = {
            "event": "error",
            "timestamp": utc_now(),
            "run_id": self.run_id,
            "call_id": call_id,
            "provider": provider,
            "client_name": client_name,
            "model": model,
            "purpose": purpose,
            "elapsed_seconds": round(time.time() - started, 4),
            "request": _clip(dict(request_payload), self.max_chars_per_field),
            "error": error,
            "metadata": dict(metadata or {}),
        }
        append_jsonl(self.errors_path, row)
        append_jsonl(self.calls_path, row)
        self._write_manifest()
        return {"call_id": call_id, "paths": self.paths()}

    def audit(self) -> dict[str, Any]:
        """Return a completeness audit for message->completion collection.

        A request is complete if it has exactly one matching response or error.
        Successful responses should also have a trainable frontier_messages row.
        """
        def read_jsonl(path: Path) -> list[dict[str, Any]]:
            if not path.exists():
                return []
            rows: list[dict[str, Any]] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        rows.append(item)
                except Exception:
                    rows.append({"_parse_error": line[:500]})
            return rows

        requests = read_jsonl(self.requests_path)
        responses = read_jsonl(self.responses_path)
        errors = read_jsonl(self.errors_path)
        trainable = read_jsonl(self.messages_path)
        completions = read_jsonl(self.message_completions_path)

        request_ids = [str(r.get("call_id")) for r in requests if r.get("call_id")]
        response_ids = [str(r.get("call_id")) for r in responses if r.get("call_id")]
        error_ids = [str(r.get("call_id")) for r in errors if r.get("call_id")]
        completed = set(response_ids) | set(error_ids)
        missing = [x for x in request_ids if x not in completed]
        duplicate_responses = sorted({x for x in response_ids if response_ids.count(x) > 1})

        return {
            "root": str(self.root),
            "ok": not missing and not duplicate_responses,
            "request_count": len(requests),
            "response_count": len(responses),
            "error_count": len(errors),
            "trainable_messages_count": len(trainable),
            "message_completion_count": len(completions),
            "missing_completion_or_error_call_ids": missing,
            "duplicate_response_call_ids": duplicate_responses,
            "paths": self.paths(),
        }
