from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from autopilot.models import to_jsonable


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _clip_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 64] + f"\n...[truncated {len(text) - (max_chars - 64)} chars]"


def _safe_json_read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, item: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(item), ensure_ascii=False) + "\n")


def openai_to_sharegpt(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    role_map = {"system": "system", "user": "human", "assistant": "gpt", "tool": "observation"}
    out: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = str(msg.get("content") or "")
        if not content:
            continue
        out.append({"from": role_map.get(role, role), "value": content})
    return out


def write_llamafactory_dataset_info(output_dir: str | Path) -> Path:
    """Write dataset_info.json entries for the conversation logs.

    The JSONL files use two intentionally simple trainable formats:
    - kimi_messages.jsonl / kimi_multiturn_messages.jsonl: {"messages": [{role, content}, ...]}
    - kimi_sharegpt.jsonl / kimi_multiturn_sharegpt.jsonl: {"conversations": [{from, value}, ...]}
    """
    root = Path(output_dir)
    data: dict[str, Any] = {
        "kimi_single_turn_messages": {
            "file_name": "kimi_messages.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
        },
        "kimi_single_turn_sharegpt": {
            "file_name": "kimi_sharegpt.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt", "system_tag": "system"},
        },
        "kimi_multiturn_messages": {
            "file_name": "kimi_multiturn_messages.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
        },
        "kimi_multiturn_sharegpt": {
            "file_name": "kimi_multiturn_sharegpt.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt", "system_tag": "system"},
        },
    }
    path = root / "dataset_info.json"
    _atomic_write_json(path, data)
    return path


@dataclass
class ConversationRecorder:
    """Append KIMI interactions in trainable JSONL formats.

    This is intentionally lightweight: it does not decide whether a transcript is
    high-quality enough to train on.  It just keeps the raw model-controller
    dialogue in formats that can be later filtered and fed to LLaMA-Factory.
    """

    root: Path
    provider: str = "kimi"
    model: str = ""
    enabled: bool = True
    session_id: str = field(default_factory=lambda: f"session-{uuid.uuid4().hex[:12]}")
    max_chars_per_message: int = 20000
    write_sharegpt: bool = True
    write_multiturn: bool = True

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        root: str | Path | None = None,
        session_id: str | None = None,
        provider: str = "kimi",
        model: str | None = None,
    ) -> "ConversationRecorder | None":
        cfg = settings.raw_config if hasattr(settings, "raw_config") and isinstance(settings.raw_config, dict) else {}
        raw = cfg.get("conversation_logging") or cfg.get("conversation_log") or cfg.get("conversations") or {}
        if not isinstance(raw, dict):
            raw = {}
        enabled = raw.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() not in {"0", "false", "no", "off"}
        if not bool(enabled):
            return None
        root_value = root or raw.get("root") or raw.get("output_dir") or os.getenv("AUTOPILOT_CONVERSATION_ROOT") or ".autopilot/conversations"
        sid = session_id or raw.get("session_id") or os.getenv("AUTOPILOT_CONVERSATION_SESSION") or f"run-{uuid.uuid4().hex[:10]}"
        try:
            max_chars = int(raw.get("max_chars_per_message") or os.getenv("AUTOPILOT_CONVERSATION_MAX_CHARS") or 20000)
        except Exception:
            max_chars = 20000
        return cls(
            root=Path(root_value),
            provider=provider,
            model=model or getattr(settings, "kimi_model", ""),
            enabled=bool(enabled),
            session_id=str(sid),
            max_chars_per_message=max_chars,
            write_sharegpt=bool(raw.get("write_sharegpt", True)),
            write_multiturn=bool(raw.get("write_multiturn", True)),
        )

    @property
    def raw_calls_path(self) -> Path:
        return self.root / f"{self.provider}_raw_calls.jsonl"

    @property
    def messages_path(self) -> Path:
        return self.root / f"{self.provider}_messages.jsonl"

    @property
    def sharegpt_path(self) -> Path:
        return self.root / f"{self.provider}_sharegpt.jsonl"

    @property
    def state_path(self) -> Path:
        return self.root / f"{self.provider}_session_state.json"

    @property
    def multiturn_messages_path(self) -> Path:
        return self.root / f"{self.provider}_multiturn_messages.jsonl"

    @property
    def multiturn_sharegpt_path(self) -> Path:
        return self.root / f"{self.provider}_multiturn_sharegpt.jsonl"

    def paths(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "raw_calls": str(self.raw_calls_path),
            "messages": str(self.messages_path),
            "sharegpt": str(self.sharegpt_path),
            "multiturn_messages": str(self.multiturn_messages_path),
            "multiturn_sharegpt": str(self.multiturn_sharegpt_path),
            "dataset_info": str(self.root / "dataset_info.json"),
            "state": str(self.state_path),
        }

    def _clip_messages(self, messages: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for msg in messages:
            role = str(msg.get("role") or "user")
            content = _clip_text(msg.get("content") or "", self.max_chars_per_message)
            if content:
                out.append({"role": role, "content": content})
        return out

    def _load_state(self) -> dict[str, Any]:
        state = _safe_json_read(self.state_path, {"sessions": {}})
        if not isinstance(state, dict):
            state = {"sessions": {}}
        sessions = state.get("sessions")
        if not isinstance(sessions, dict):
            state["sessions"] = {}
        return state

    def _write_multiturn_latest(self, state: dict[str, Any]) -> None:
        if not self.write_multiturn:
            return
        sessions = state.get("sessions") or {}
        self.root.mkdir(parents=True, exist_ok=True)
        with self.multiturn_messages_path.open("w", encoding="utf-8") as fm, self.multiturn_sharegpt_path.open("w", encoding="utf-8") as fs:
            for sid, sess in sorted(sessions.items()):
                if not isinstance(sess, dict):
                    continue
                messages = sess.get("messages") or []
                if not isinstance(messages, list) or len(messages) < 3:
                    continue
                metadata = {
                    "provider": self.provider,
                    "model": sess.get("model") or self.model,
                    "session_id": sid,
                    "created_at": sess.get("created_at"),
                    "updated_at": sess.get("updated_at"),
                    "turn_count": sess.get("turn_count"),
                    "purposes": sess.get("purposes") or [],
                }
                fm.write(json.dumps({"messages": messages, "metadata": metadata}, ensure_ascii=False) + "\n")
                fs.write(json.dumps({"conversations": openai_to_sharegpt(messages), "metadata": metadata}, ensure_ascii=False) + "\n")

    def record_call(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        assistant: str | None = None,
        error: str | None = None,
        parsed_ok: bool | None = None,
        request: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        now = _now()
        call_id = uuid.uuid4().hex
        reasoning_content = None
        if metadata and metadata.get("reasoning_content"):
            reasoning_content = str(metadata.get("reasoning_content"))
        assistant_content = assistant if assistant is not None else (f"[ERROR] {error}" if error else "")
        assistant_for_training = (f"[reasoning_content]\n{reasoning_content}\n\n[content]\n{assistant_content}" if reasoning_content else assistant_content)
        messages = self._clip_messages([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant_for_training},
        ])
        meta = {
            "provider": self.provider,
            "model": self.model,
            "purpose": purpose,
            "session_id": self.session_id,
            "call_id": call_id,
            "timestamp": now,
            "parsed_ok": parsed_ok,
            "error": error,
        }
        if metadata:
            meta.update(dict(metadata))
        raw = {
            "timestamp": now,
            "session_id": self.session_id,
            "call_id": call_id,
            "provider": self.provider,
            "model": self.model,
            "purpose": purpose,
            "request": dict(request or {}),
            "messages": messages,
            "assistant": _clip_text(assistant_content, self.max_chars_per_message),
            "reasoning_content": _clip_text(reasoning_content, self.max_chars_per_message) if reasoning_content else None,
            "parsed_ok": parsed_ok,
            "error": error,
            "metadata": dict(metadata or {}),
        }
        _append_jsonl(self.raw_calls_path, raw)
        _append_jsonl(self.messages_path, {"messages": messages, "metadata": meta})
        if self.write_sharegpt:
            _append_jsonl(self.sharegpt_path, {"conversations": openai_to_sharegpt(messages), "metadata": meta})

        if self.write_multiturn:
            state = self._load_state()
            sessions = state.setdefault("sessions", {})
            sess = sessions.setdefault(
                self.session_id,
                {
                    "session_id": self.session_id,
                    "provider": self.provider,
                    "model": self.model,
                    "created_at": now,
                    "updated_at": now,
                    "turn_count": 0,
                    "purposes": [],
                    "messages": [
                        {
                            "role": "system",
                            "content": "Autopilot KIMI controller transcript. Each user turn includes the original system prompt, task purpose, and user payload; each assistant turn is KIMI's response.",
                        }
                    ],
                },
            )
            user_turn = (
                f"[purpose]\n{purpose}\n\n"
                f"[system]\n{_clip_text(system, self.max_chars_per_message)}\n\n"
                f"[user]\n{_clip_text(user, self.max_chars_per_message)}"
            )
            sess.setdefault("messages", []).extend([
                {"role": "user", "content": user_turn},
                {"role": "assistant", "content": _clip_text(assistant_for_training, self.max_chars_per_message)},
            ])
            sess["updated_at"] = now
            sess["turn_count"] = int(sess.get("turn_count") or 0) + 1
            purposes = sess.setdefault("purposes", [])
            if purpose not in purposes:
                purposes.append(purpose)
            _atomic_write_json(self.state_path, state)
            self._write_multiturn_latest(state)
        write_llamafactory_dataset_info(self.root)
        return {"call_id": call_id, "paths": self.paths()}


def export_conversation_logs(root: str | Path, output_dir: str | Path | None = None) -> dict[str, str]:
    """Copy/rewrite existing trainable logs into an output directory.

    This is useful after a long run when the user wants a clean data directory to
    feed into LLaMA-Factory or inspect manually.
    """
    src = Path(root)
    dst = Path(output_dir) if output_dir else src
    dst.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for name in [
        "kimi_messages.jsonl",
        "kimi_sharegpt.jsonl",
        "kimi_multiturn_messages.jsonl",
        "kimi_multiturn_sharegpt.jsonl",
        "kimi_raw_calls.jsonl",
        "kimi_session_state.json",
    ]:
        p = src / name
        if p.exists():
            q = dst / name
            if p.resolve() != q.resolve():
                q.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
            copied[name] = str(q)
    copied["dataset_info.json"] = str(write_llamafactory_dataset_info(dst))
    return copied
