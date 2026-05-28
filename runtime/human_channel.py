from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autopilot.models import to_jsonable
from autopilot.runtime.trajectory import append_jsonl, atomic_write_json, utc_now


USER_NOTES_TEMPLATE = """# AUTOPILOT_USER_NOTES

Write running advice for the autonomous agent here. The agent reads this file
at the start of each iteration. Prefer short bullets with optional tags.

Tags:
- @policy: hard rule or invariant.
- @preference: user preference.
- @advice: live guidance for the current/next run.
- @correction: correction to a behavior the agent just showed.
- @decision: a choice boundary where the agent should ask before proceeding.
- @memory: something the agent may draft into long-term memory.
- @skill: something the agent may draft into a skill update.

Examples:
- @policy AIME24 benchmark cases are eval_only and must never enter training data.
- @preference Repair bugs automatically; ask only for real choices.
- @advice If vLLM is still loading checkpoint shards, wait before killing it.
"""

RUN_INBOX_TEMPLATE = """# Run-local human inbox

Append notes for this run below. The agent reads this file at every iteration.
Use the same tags as AUTOPILOT_USER_NOTES.md.
"""


@dataclass
class HumanNote:
    id: str
    source: str
    kind: str
    text: str
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class HumanChannelSnapshot:
    user_notes_path: str
    inbox_path: str
    outbox_path: str
    decisions_path: str
    dialog_log_path: str
    notes: list[HumanNote] = field(default_factory=list)
    open_decisions: list[dict[str, Any]] = field(default_factory=list)
    instructions: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = to_jsonable(self)
        data["notes"] = [n.to_dict() for n in self.notes]
        return data


class HumanChannel:
    """Run-local, non-blocking human communication channel.

    The channel separates user notes from decision requests:
    - AUTOPILOT_USER_NOTES.md is long-lived project guidance.
    - .autopilot/human/INBOX.md is run-local guidance.
    - answer_human writes OUTBOX.md and dialog.jsonl.
    - only requires_response=true creates a WAITING_USER_DECISION state.
    """

    def __init__(self, run_root: str | Path, *, project_root: str | Path | None = None) -> None:
        self.run_root = Path(run_root)
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()
        self.human_dir = self.run_root / ".autopilot" / "human"
        self.user_notes_path = self.project_root / "AUTOPILOT_USER_NOTES.md"
        self.inbox_path = self.human_dir / "INBOX.md"
        self.outbox_path = self.human_dir / "OUTBOX.md"
        self.decisions_path = self.human_dir / "DECISIONS.md"
        self.dialog_path = self.human_dir / "dialog.jsonl"
        self.state_path = self.human_dir / "state.json"

    def materialize(self) -> None:
        self.human_dir.mkdir(parents=True, exist_ok=True)
        if not self.user_notes_path.exists():
            self.user_notes_path.write_text(USER_NOTES_TEMPLATE, encoding="utf-8")
        if not self.inbox_path.exists():
            self.inbox_path.write_text(RUN_INBOX_TEMPLATE, encoding="utf-8")
        if not self.outbox_path.exists():
            self.outbox_path.write_text("# Agent replies\n\n", encoding="utf-8")
        if not self.decisions_path.exists():
            self.decisions_path.write_text("# Open user decisions\n\n", encoding="utf-8")
        if not self.state_path.exists():
            atomic_write_json(self.state_path, {"created_at": utc_now(), "open_decisions": []})

    def snapshot(self, *, max_notes: int = 40, max_chars_per_note: int = 1200) -> HumanChannelSnapshot:
        self.materialize()
        notes = self._read_notes(max_notes=max_notes, max_chars=max_chars_per_note)
        open_decisions = self._open_decisions()
        return HumanChannelSnapshot(
            user_notes_path=str(self._rel(self.user_notes_path)),
            inbox_path=str(self._rel(self.inbox_path)),
            outbox_path=str(self._rel(self.outbox_path)),
            decisions_path=str(self._rel(self.decisions_path)),
            dialog_log_path=str(self._rel(self.dialog_path)),
            notes=notes,
            open_decisions=open_decisions,
            instructions=(
                "Read human_channel.notes as live user guidance. Acknowledge useful notes with the "
                "answer_human tool, but do not block unless you need a real choice. "
                "Use answer_human(kind='ack', requires_response=false) for normal replies. "
                "Use answer_human(kind='decision_request', requires_response=true, choices=[...]) only for real choices."
            ),
        )

    def answer(self, args: dict[str, Any]) -> dict[str, Any]:
        self.materialize()
        message = str(args.get("message") or args.get("answer") or args.get("text") or "").strip()
        if not message:
            return {"ok": False, "error": "answer_human requires message"}
        kind = str(args.get("kind") or ("decision_request" if args.get("requires_response") else "ack"))
        requires_response = bool(args.get("requires_response", False))
        choices = args.get("choices") or args.get("options") or []
        if choices is None:
            choices = []
        if not isinstance(choices, list):
            choices = [choices]
        related_note_ids = args.get("related_note_ids") or []
        if isinstance(related_note_ids, str):
            related_note_ids = [related_note_ids]
        now = utc_now()
        event_id = str(args.get("id") or args.get("decision_id") or self._event_id(kind, message, now))
        row = {
            "ts": now,
            "role": "agent",
            "kind": kind,
            "id": event_id,
            "message": message,
            "requires_response": requires_response,
            "choices": choices,
            "recommended_choice": args.get("recommended_choice"),
            "related_note_ids": related_note_ids,
            "related_action_id": args.get("related_action_id") or args.get("action_id"),
            "status": "open" if requires_response else "posted",
        }
        append_jsonl(self.dialog_path, row)
        self._append_outbox(row)
        if requires_response:
            self._record_decision(row)
        return {
            "ok": True,
            "kind": kind,
            "id": event_id,
            "requires_response": requires_response,
            "choices": choices,
            "outbox_path": str(self._rel(self.outbox_path)),
            "dialog_log_path": str(self._rel(self.dialog_path)),
            "decisions_path": str(self._rel(self.decisions_path)) if requires_response else None,
        }

    def decision_payload(self, answer_result: dict[str, Any]) -> dict[str, Any]:
        return {
            "question_id": answer_result.get("id"),
            "status": "open",
            "kind": "decision_request",
            "message": answer_result.get("message") or "Decision requested",
            "choices": answer_result.get("choices") or [],
            "outbox_path": str(self._rel(self.outbox_path)),
            "decisions_path": str(self._rel(self.decisions_path)),
            "dialog_log_path": str(self._rel(self.dialog_path)),
            "resume_after_answer": True,
            "since": utc_now(),
        }

    def _read_notes(self, *, max_notes: int, max_chars: int) -> list[HumanNote]:
        notes: list[HumanNote] = []
        for path in [self.user_notes_path, self.inbox_path]:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            notes.extend(self._parse_notes(text, source=str(self._rel(path)), max_chars=max_chars))
        # Keep the latest notes by source order/line order.
        return notes[-max_notes:]

    def _parse_notes(self, text: str, *, source: str, max_chars: int) -> list[HumanNote]:
        out: list[HumanNote] = []
        for idx, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("---"):
                continue
            if not (stripped.startswith("-") or "@" in stripped):
                continue
            kind = "note"
            m = re.search(r"@(policy|preference|advice|correction|decision|memory|skill)\b", stripped, flags=re.I)
            if m:
                kind = m.group(1).lower()
            cleaned = re.sub(r"^[-*]\s*", "", stripped).strip()
            if len(cleaned) > max_chars:
                cleaned = cleaned[:max_chars] + "…"
            nid = self._note_id(source, idx, cleaned)
            out.append(HumanNote(id=nid, source=source, kind=kind, text=cleaned, line=idx))
        return out

    def _open_decisions(self) -> list[dict[str, Any]]:
        state = self._read_json(self.state_path, {"open_decisions": []})
        rows = state.get("open_decisions") or []
        return [r for r in rows if isinstance(r, dict) and r.get("status") == "open"]

    def _record_decision(self, row: dict[str, Any]) -> None:
        state = self._read_json(self.state_path, {"open_decisions": []})
        rows = [r for r in state.get("open_decisions") or [] if isinstance(r, dict)]
        rows.append(row)
        state["open_decisions"] = rows
        state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, state)
        lines = ["# Open user decisions", ""]
        for item in rows:
            if item.get("status") != "open":
                continue
            lines.append(f"## {item.get('id')}")
            lines.append("")
            lines.append(str(item.get("message") or ""))
            choices = item.get("choices") or []
            if choices:
                lines.append("")
                lines.append("Choices:")
                for i, choice in enumerate(choices, start=1):
                    if isinstance(choice, dict):
                        label = choice.get("label") or choice.get("id") or choice
                    else:
                        label = choice
                    lines.append(f"{i}. {label}")
            rec = item.get("recommended_choice")
            if rec:
                lines.append("")
                lines.append(f"Recommended: {rec}")
            lines.append("")
            lines.append("Reply by appending a decision_answer line to INBOX.md or editing this section.")
            lines.append("")
        self.decisions_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _append_outbox(self, row: dict[str, Any]) -> None:
        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        with self.outbox_path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {row['ts']} — {row['kind']} — {row['id']}\n\n")
            f.write(str(row.get("message") or "").rstrip() + "\n")
            choices = row.get("choices") or []
            if choices:
                f.write("\nChoices:\n")
                for i, choice in enumerate(choices, start=1):
                    if isinstance(choice, dict):
                        label = choice.get("label") or choice.get("id") or choice
                    else:
                        label = choice
                    f.write(f"{i}. {label}\n")
            f.write("\n")

    def _read_json(self, path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _rel(self, path: Path) -> Path:
        try:
            return path.resolve().relative_to(self.run_root.resolve())
        except Exception:
            try:
                return path.resolve().relative_to(self.project_root.resolve())
            except Exception:
                return path

    @staticmethod
    def _note_id(source: str, line: int, text: str) -> str:
        h = hashlib.sha1(f"{source}:{line}:{text}".encode("utf-8", errors="replace")).hexdigest()[:10]
        return f"note-{h}"

    @staticmethod
    def _event_id(kind: str, message: str, now: str) -> str:
        h = hashlib.sha1(f"{kind}:{message}:{now}".encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"human-{h}"
