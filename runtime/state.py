from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopilot.models import to_jsonable
from autopilot.runtime.trajectory import append_jsonl, atomic_write_json, utc_now


TASK_STATES = {"PENDING", "RUNNING", "WAITING_HUMAN", "WAITING_USER_DECISION", "SUCCEEDED", "FAILED", "INTERRUPTED", "SKIPPED", "CANCELLED"}


@dataclass
class TaskRecord:
    task_id: str
    name: str
    objective: str = ""
    status: str = "PENDING"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    attempts: int = 0
    artifacts: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self.__dict__)


class RunStateStore:
    """Append-only state store for resumable Autopilot runs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.autopilot_dir = self.root / ".autopilot"
        self.autopilot_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "run_state.json"
        self.task_graph_path = self.root / "task_graph.json"
        self.event_log_path = self.root / "event_log.jsonl"
        self.artifacts_path = self.root / "artifacts.json"

    def _read_json(self, path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def append_event(self, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        row = {
            "event_id": uuid.uuid4().hex,
            "timestamp": utc_now(),
            "type": event_type,
            "payload": payload or {},
        }
        append_jsonl(self.event_log_path, row)
        return row

    def state(self) -> dict[str, Any]:
        return self._read_json(self.state_path, {"status": "NEW", "created_at": utc_now(), "updated_at": utc_now(), "current_task": None, "waiting_human": None, "waiting_user_decision": None})

    def write_state(self, **updates: Any) -> dict[str, Any]:
        data = self.state()
        data.update(updates)
        data["updated_at"] = utc_now()
        atomic_write_json(self.state_path, data)
        self.append_event("run_state_update", updates)
        return data

    def task_graph(self) -> dict[str, Any]:
        return self._read_json(self.task_graph_path, {"tasks": {}})

    def write_task_graph(self, graph: dict[str, Any]) -> None:
        graph["updated_at"] = utc_now()
        atomic_write_json(self.task_graph_path, graph)

    def upsert_task(self, name: str, objective: str = "", *, task_id: str | None = None, status: str = "PENDING", **updates: Any) -> TaskRecord:
        graph = self.task_graph()
        tasks = graph.setdefault("tasks", {})
        tid = task_id or updates.pop("task_id", None)
        if tid is None:
            for existing_id, row in tasks.items():
                if isinstance(row, dict) and row.get("name") == name:
                    tid = existing_id
                    break
        tid = tid or uuid.uuid4().hex[:12]
        row = tasks.get(tid) if isinstance(tasks.get(tid), dict) else {}
        rec = TaskRecord(
            task_id=tid,
            name=str(row.get("name") or name),
            objective=str(row.get("objective") or objective or ""),
            status=str(row.get("status") or status),
            created_at=str(row.get("created_at") or utc_now()),
            updated_at=utc_now(),
            attempts=int(row.get("attempts") or 0),
            artifacts=dict(row.get("artifacts") or {}),
            result=dict(row.get("result") or {}),
            error=row.get("error"),
        )
        if status:
            rec.status = status
        for k, v in updates.items():
            if hasattr(rec, k):
                setattr(rec, k, v)
        tasks[tid] = rec.to_dict()
        self.write_task_graph(graph)
        self.append_event("task_upsert", rec.to_dict())
        return rec

    def mark_task(self, task_id: str, status: str, **updates: Any) -> TaskRecord:
        if status not in TASK_STATES:
            raise ValueError(f"invalid task status: {status}")
        graph = self.task_graph()
        row = (graph.get("tasks") or {}).get(task_id)
        if not isinstance(row, dict):
            raise KeyError(task_id)
        return self.upsert_task(row.get("name") or task_id, row.get("objective") or "", task_id=task_id, status=status, **updates)

    def mark_running_interrupted(self) -> list[str]:
        graph = self.task_graph()
        interrupted: list[str] = []
        for tid, row in list((graph.get("tasks") or {}).items()):
            if isinstance(row, dict) and row.get("status") == "RUNNING":
                row["status"] = "INTERRUPTED"
                row["updated_at"] = utc_now()
                interrupted.append(tid)
        self.write_task_graph(graph)
        if interrupted:
            self.append_event("running_tasks_marked_interrupted", {"task_ids": interrupted})
        return interrupted

    def set_waiting_human(self, question_id: str, question: str, context: Any = None, blocking_task: str | None = None) -> dict[str, Any]:
        payload = {"question_id": question_id, "question": question, "context": context, "blocking_task": blocking_task, "since": utc_now(), "resume_after_answer": True}
        self.write_state(status="WAITING_HUMAN", waiting_human=payload, current_task=blocking_task)
        self.append_event("waiting_human", payload)
        if blocking_task:
            try:
                self.mark_task(blocking_task, "WAITING_HUMAN", result={"question_id": question_id})
            except Exception:
                pass
        return payload

    def add_artifact(self, name: str, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        artifacts = self._read_json(self.artifacts_path, {})
        artifacts[name] = {"path": str(path), "metadata": metadata or {}, "updated_at": utc_now()}
        atomic_write_json(self.artifacts_path, artifacts)
        self.append_event("artifact", {"name": name, "path": str(path), "metadata": metadata or {}})

    def resume_plan(self) -> dict[str, Any]:
        self.mark_running_interrupted()
        graph = self.task_graph()
        tasks = graph.get("tasks") or {}
        runnable = []
        failed = []
        waiting = []
        succeeded = []
        for tid, row in tasks.items():
            if not isinstance(row, dict):
                continue
            status = row.get("status")
            if status in {"PENDING", "INTERRUPTED"}:
                runnable.append(tid)
            elif status == "FAILED":
                failed.append(tid)
            elif status in {"WAITING_HUMAN", "WAITING_USER_DECISION"}:
                waiting.append(tid)
            elif status == "SUCCEEDED":
                succeeded.append(tid)
        plan = {"status": self.state().get("status"), "runnable": runnable, "failed": failed, "waiting_human": waiting, "succeeded": succeeded, "task_count": len(tasks)}
        self.append_event("resume_plan", plan)
        return plan
