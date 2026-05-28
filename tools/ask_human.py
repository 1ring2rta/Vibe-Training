from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HumanQuestion:
    question_id: str
    question: str
    context: Any = ""
    priority: str = "normal"
    status: str = "pending"
    response: str | None = None
    choices: list[str] = field(default_factory=list)
    created_at: float = 0.0

    @property
    def answer(self) -> str | None:
        return self.response

    @property
    def urgency(self) -> str:
        return self.priority

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AskHumanTool:
    """File-backed ask_human tool for long-running agent loops.

    The tool never blocks by default. It queues questions as JSON/Markdown under
    ``.autopilot/human`` so a human can inspect or answer them while
    the training loop continues. Interactive blocking is available only when the
    caller explicitly requests it and stdin is a TTY.
    """

    def __init__(self, workspace: str | Path, mode: str = "queue") -> None:
        root = Path(workspace)
        if root.name == "human" or str(root).endswith(".autopilot/human"):
            self.workspace = root
        else:
            self.workspace = root / ".autopilot" / "human"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.queue_path = self.workspace / "ask_human_queue.jsonl"
        self.replies_path = self.workspace / "human_replies.jsonl"
        self.markdown_path = self.workspace / "ASK_HUMAN.md"
        if not self.markdown_path.exists():
            self.markdown_path.write_text(
                "# ask_human Queue\n\nQuestions queued by Autopilot for human discussion. "
                "Add replies with `autopilot-ask-human answer <question_id> <response>` or `autopilot-human reply <question_id> --answer ...`; you can also append JSONL to `human_replies.jsonl`.\n\n",
                encoding="utf-8",
            )

    def _responses(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.replies_path.exists():
            for line in self.replies_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                qid = data.get("question_id") or data.get("id")
                if qid:
                    out[str(qid)] = str(data.get("response") or data.get("answer") or "")
        return out

    @staticmethod
    def _context_to_markdown(context: Any) -> str:
        if context is None or context == "":
            return ""
        if isinstance(context, str):
            return context
        try:
            return "```json\n" + json.dumps(context, ensure_ascii=False, indent=2) + "\n```"
        except Exception:
            return str(context)

    def _write_question_files(self, q: HumanQuestion) -> tuple[Path, Path]:
        json_path = self.workspace / f"{q.question_id}.json"
        md_path = self.workspace / f"{q.question_id}.md"
        json_path.write_text(json.dumps(q.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        context_md = self._context_to_markdown(q.context)
        lines = [
            f"# ask_human: {q.question_id}",
            "",
            f"Priority: {q.priority}",
            f"Status: {q.status}",
            "",
            "## Question",
            q.question,
            "",
        ]
        if context_md:
            lines += ["## Context", context_md, ""]
        if q.choices:
            lines += ["## Suggested options", *[f"- {c}" for c in q.choices], ""]
        if q.response:
            lines += ["## Response", q.response, ""]
        md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return json_path, md_path

    def ask(
        self,
        question: str,
        *,
        context: Any = "",
        priority: str = "normal",
        urgency: str | None = None,
        choices: list[str] | None = None,
        suggested_options: list[str] | None = None,
        interactive: bool | None = None,
        blocking: bool | None = None,
    ) -> HumanQuestion:
        priority = str(urgency or priority or "normal")
        q = HumanQuestion(
            question_id=str(uuid.uuid4())[:12],
            question=str(question).strip(),
            context=context,
            priority=priority,
            choices=list(choices or suggested_options or []),
            created_at=time.time(),
        )
        should_prompt = bool(blocking) or bool(interactive) or self.mode in {"interactive", "blocking"}
        if should_prompt and sys.stdin.isatty():
            print("\n[ask_human] " + q.question, file=sys.stderr)
            context_md = self._context_to_markdown(q.context)
            if context_md:
                print("context:\n" + context_md, file=sys.stderr)
            if q.choices:
                print("choices: " + ", ".join(q.choices), file=sys.stderr)
            q.response = input("human> ").strip()
            q.status = "answered" if q.response else "pending"
        with self.queue_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
        json_path, md_path = self._write_question_files(q)
        with self.markdown_path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {q.question_id} [{q.priority}]\n\n{q.question}\n\n")
            context_md = self._context_to_markdown(q.context)
            if context_md:
                f.write("Context:\n" + context_md + "\n\n")
            if q.choices:
                f.write("Choices:\n" + "\n".join(f"- {c}" for c in q.choices) + "\n\n")
            f.write(f"Status: {q.status}\n")
            if q.response:
                f.write(f"\nAnswer: {q.response}\n")
            f.write(f"\nFiles: `{json_path}`, `{md_path}`\n")
        return q

    def list_questions(self, status: str | None = None) -> list[HumanQuestion]:
        replies = self._responses()
        questions: list[HumanQuestion] = []
        if not self.queue_path.exists():
            return questions
        for line in self.queue_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                allowed = {k: v for k, v in data.items() if k in HumanQuestion.__dataclass_fields__}
                q = HumanQuestion(**allowed)
            except Exception:
                continue
            if q.question_id in replies and replies[q.question_id]:
                q.status = "answered"
                q.response = replies[q.question_id]
            if status is None or q.status == status:
                questions.append(q)
        return questions

    def reply(self, question_id: str, response: str) -> Path:
        row = {"question_id": question_id, "response": response, "answered_at": time.time()}
        with self.replies_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Update aggregate markdown; per-question JSON is intentionally left as
        # the original queued question plus replies JSONL, preserving audit trail.
        with self.markdown_path.open("a", encoding="utf-8") as f:
            f.write(f"\n### Human reply to {question_id}\n\n{response}\n")
        return self.replies_path

    def answer(self, question_id: str, response: str) -> HumanQuestion:
        self.reply(question_id, response)
        return HumanQuestion(question_id=question_id, question="", status="answered", response=response, created_at=time.time())
