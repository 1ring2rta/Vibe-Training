from __future__ import annotations

import json
import os
import re
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from autopilot.context import ContextManager
from autopilot.models import to_jsonable


def _slugify(name: str, *, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip()).strip("-._")
    return (text or "task")[:max_len]


def _is_quiet() -> bool:
    return str(os.getenv("AUTOPILOT_QUIET", "")).lower() in {"1", "true", "yes", "on"}


def _agent_log(message: str, *, depth: int = 0) -> None:
    if _is_quiet():
        return
    prefix = "  " * max(0, depth)
    print(prefix + message, flush=True)


@dataclass
class AgentArtifact:
    path: str
    kind: str
    description: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentTaskResult:
    task_id: str
    name: str
    objective: str
    status: str
    summary: str
    output: dict[str, Any] = field(default_factory=dict)
    artifacts: list[AgentArtifact] = field(default_factory=list)
    child_tasks: list[dict[str, Any]] = field(default_factory=list)
    task_dir: str = ""
    context_path: str = ""
    result_path: str = ""
    duration_seconds: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "success"


class AgentLoop:
    """A resumable, nestable agent loop.

    The loop is intentionally small and deterministic. It gives the workflow a
    structure that looks like coding-agent systems:

    - a parent loop creates a named task;
    - the task receives its own context state and workspace directory;
    - the task can record observations, write artifacts, and spawn subtasks;
    - on completion, result.json and summary.md are written;
    - the compact task result is returned to the parent loop.

    This class does not force one specific planner. A deterministic workflow,
    a frontier-model planner, or a future tool-use agent can all call
    ``run_task`` and recursively create subtasks.
    """

    def __init__(
        self,
        *,
        name: str,
        objective: str,
        context: ContextManager,
        workspace_dir: str | Path,
        task_id: str | None = None,
        parent_task_id: str | None = None,
        depth: int = 0,
        max_iterations: int = 64,
    ) -> None:
        self.name = name
        self.objective = objective
        self.context = context
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.task_id = task_id or str(uuid.uuid4())
        self.parent_task_id = parent_task_id
        self.depth = depth
        self.max_iterations = max_iterations
        self.iteration = 0
        self.artifacts: list[AgentArtifact] = []
        self.child_tasks: list[dict[str, Any]] = []
        self.result_summary = ""
        self.result_output: dict[str, Any] = {}

    @classmethod
    def root(
        cls,
        *,
        name: str,
        objective: str,
        context: ContextManager,
        workspace_dir: str | Path,
        max_iterations: int = 64,
    ) -> "AgentLoop":
        loop = cls(
            name=name,
            objective=objective,
            context=context,
            workspace_dir=workspace_dir,
            task_id="root",
            parent_task_id=None,
            depth=0,
            max_iterations=max_iterations,
        )
        loop.record(
            "loop_start",
            name,
            objective,
            {"workspace_dir": str(loop.workspace_dir), "task_id": loop.task_id},
            importance=2,
        )
        return loop

    def next_iteration(self) -> int:
        self.iteration += 1
        if self.iteration > self.max_iterations:
            raise RuntimeError(f"Agent loop exceeded max_iterations={self.max_iterations}: {self.name}")
        return self.iteration

    def record(
        self,
        kind: str,
        title: str,
        summary: str,
        payload: dict[str, Any] | None = None,
        *,
        importance: int = 1,
    ) -> None:
        self.next_iteration()
        payload = dict(payload or {})
        payload.setdefault("loop", self.name)
        payload.setdefault("task_id", self.task_id)
        payload.setdefault("depth", self.depth)
        self.context.add_event(kind, title, summary, to_jsonable(payload), importance=importance)

    def observe(self, title: str, summary: str, payload: dict[str, Any] | None = None, *, importance: int = 1) -> None:
        self.record("observation", title, summary, payload, importance=importance)

    def decide(self, title: str, summary: str, payload: dict[str, Any] | None = None, *, importance: int = 2) -> None:
        self.record("decision", title, summary, payload, importance=importance)

    def record_tool_call(
        self,
        tool_name: str,
        *,
        inputs: dict[str, Any] | None = None,
        output_summary: str = "",
        output: dict[str, Any] | None = None,
        importance: int = 1,
    ) -> None:
        self.record(
            "tool_call",
            tool_name,
            output_summary or f"Called {tool_name}",
            {"inputs": inputs or {}, "output": output or {}},
            importance=importance,
        )

    def add_artifact(self, path: str | Path, kind: str, description: str = "") -> AgentArtifact:
        artifact = AgentArtifact(path=str(path), kind=kind, description=description)
        self.artifacts.append(artifact)
        self.context.add_artifact(path, kind, description)
        self.record(
            "artifact",
            kind,
            description or str(path),
            {"path": str(path), "kind": kind},
            importance=2,
        )
        return artifact

    def write_json_artifact(self, relative_path: str, data: Any, *, kind: str, description: str = "") -> Path:
        path = self.workspace_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.add_artifact(path, kind, description)
        return path

    def write_text_artifact(self, relative_path: str, text: str, *, kind: str, description: str = "") -> Path:
        path = self.workspace_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        self.add_artifact(path, kind, description)
        return path

    def set_result(self, summary: str, output: dict[str, Any] | None = None) -> None:
        self.result_summary = summary
        self.result_output = to_jsonable(output or {})
        self.record("result_update", self.name, summary, self.result_output, importance=2)

    def compact_context(self, *, keep_recent_events: int | None = None) -> None:
        self.context.compact(keep_recent_events=keep_recent_events)
        self.record("context", "compact", "Compacted older loop events into rolling summary", importance=2)

    def _new_child_dir(self, name: str) -> tuple[str, Path]:
        task_id = uuid.uuid4().hex[:12]
        slug = _slugify(name)
        child_dir = self.workspace_dir / "tasks" / f"{task_id}-{slug}"
        child_dir.mkdir(parents=True, exist_ok=True)
        return task_id, child_dir

    @contextmanager
    def subtask(
        self,
        name: str,
        objective: str,
        *,
        inputs: dict[str, Any] | None = None,
        task_type: str = "task",
    ) -> Iterator["AgentLoop"]:
        task_id, child_dir = self._new_child_dir(name)
        context_path = child_dir / "context" / "session.json"
        child_context = ContextManager(
            context_path,
            project_root=self.context.project_root,
            max_chars=self.context.max_chars,
            keep_recent_events=self.context.keep_recent_events,
        )
        child = AgentLoop(
            name=name,
            objective=objective,
            context=child_context,
            workspace_dir=child_dir,
            task_id=task_id,
            parent_task_id=self.task_id,
            depth=self.depth + 1,
            max_iterations=self.max_iterations,
        )
        meta = {
            "task_id": task_id,
            "parent_task_id": self.task_id,
            "name": name,
            "type": task_type,
            "objective": objective,
            "inputs": to_jsonable(inputs or {}),
            "created_at": time.time(),
            "depth": child.depth,
            "context_path": str(context_path),
        }
        (child_dir / "task.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.record("task_created", name, objective, meta, importance=2)
        child.record("task_start", name, objective, {"inputs": inputs or {}, "parent_task_id": self.task_id}, importance=2)
        _agent_log(f"[task:start] {name} — {objective}", depth=child.depth)
        started = time.perf_counter()
        status = "success"
        error: str | None = None
        try:
            yield child
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            child.record(
                "task_error",
                name,
                error,
                {"traceback": traceback.format_exc(limit=20)},
                importance=3,
            )
            raise
        finally:
            duration = round(time.perf_counter() - started, 4)
            summary = child.result_summary or ("Task failed" if status == "failed" else "Task completed")
            result = AgentTaskResult(
                task_id=task_id,
                name=name,
                objective=objective,
                status=status,
                summary=summary,
                output=to_jsonable(child.result_output),
                artifacts=child.artifacts,
                child_tasks=child.child_tasks,
                task_dir=str(child_dir),
                context_path=str(context_path),
                result_path=str(child_dir / "result.json"),
                duration_seconds=duration,
                error=error,
            )
            result_json = json.dumps(to_jsonable(result), ensure_ascii=False, indent=2) + "\n"
            (child_dir / "result.json").write_text(result_json, encoding="utf-8")
            child.write_text_artifact(
                "summary.md",
                _render_task_summary(result),
                kind="task_summary",
                description=f"Summary for task {name}",
            )
            child.record("task_done", name, f"status={status}; {summary}", {"result_path": result.result_path}, importance=3)
            child.context.save()
            _agent_log(f"[task:{status}] {name} ({duration:.1f}s) — {summary}", depth=child.depth)
            compact_child = {
                "task_id": task_id,
                "name": name,
                "status": status,
                "summary": summary,
                "result_path": result.result_path,
                "context_path": result.context_path,
                "duration_seconds": duration,
                "error": error,
            }
            self.child_tasks.append(compact_child)
            self.record(
                "task_completed" if status == "success" else "task_failed",
                name,
                summary,
                compact_child,
                importance=3 if status != "success" else 2,
            )

    def run_task(
        self,
        name: str,
        objective: str,
        handler: Callable[["AgentLoop"], dict[str, Any] | None],
        *,
        inputs: dict[str, Any] | None = None,
        task_type: str = "task",
        raise_on_error: bool = True,
    ) -> AgentTaskResult:
        task_id, child_dir = self._new_child_dir(name)
        context_path = child_dir / "context" / "session.json"
        child_context = ContextManager(
            context_path,
            project_root=self.context.project_root,
            max_chars=self.context.max_chars,
            keep_recent_events=self.context.keep_recent_events,
        )
        child = AgentLoop(
            name=name,
            objective=objective,
            context=child_context,
            workspace_dir=child_dir,
            task_id=task_id,
            parent_task_id=self.task_id,
            depth=self.depth + 1,
            max_iterations=self.max_iterations,
        )
        meta = {
            "task_id": task_id,
            "parent_task_id": self.task_id,
            "name": name,
            "type": task_type,
            "objective": objective,
            "inputs": to_jsonable(inputs or {}),
            "created_at": time.time(),
            "depth": child.depth,
            "context_path": str(context_path),
        }
        (child_dir / "task.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.record("task_created", name, objective, meta, importance=2)
        child.record("task_start", name, objective, {"inputs": inputs or {}, "parent_task_id": self.task_id}, importance=2)
        _agent_log(f"[task:start] {name} — {objective}", depth=child.depth)
        started = time.perf_counter()
        status = "success"
        error: str | None = None
        try:
            output = handler(child)
            if output is not None:
                if not child.result_summary:
                    child.set_result("Task completed", output)
                else:
                    child.result_output.update(to_jsonable(output))
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            child.record("task_error", name, error, {"traceback": traceback.format_exc(limit=20)}, importance=3)
            if raise_on_error:
                # Write the failed task result before re-raising.
                result = self._finalize_child_result(child, child_dir, context_path, name, objective, status, started, error)
                self._attach_child_result(result)
                _agent_log(f"[task:{status}] {name} ({result.duration_seconds:.1f}s) — {result.summary}", depth=child.depth)
                raise
        result = self._finalize_child_result(child, child_dir, context_path, name, objective, status, started, error)
        self._attach_child_result(result)
        _agent_log(f"[task:{status}] {name} ({result.duration_seconds:.1f}s) — {result.summary}", depth=child.depth)
        return result

    def _finalize_child_result(
        self,
        child: "AgentLoop",
        child_dir: Path,
        context_path: Path,
        name: str,
        objective: str,
        status: str,
        started: float,
        error: str | None,
    ) -> AgentTaskResult:
        duration = round(time.perf_counter() - started, 4)
        summary = child.result_summary or ("Task failed" if status == "failed" else "Task completed")
        result_path = child_dir / "result.json"
        result = AgentTaskResult(
            task_id=child.task_id,
            name=name,
            objective=objective,
            status=status,
            summary=summary,
            output=to_jsonable(child.result_output),
            artifacts=child.artifacts,
            child_tasks=child.child_tasks,
            task_dir=str(child_dir),
            context_path=str(context_path),
            result_path=str(result_path),
            duration_seconds=duration,
            error=error,
        )
        result_path.write_text(json.dumps(to_jsonable(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary_path = child_dir / "summary.md"
        summary_path.write_text(_render_task_summary(result), encoding="utf-8")
        # Add summary artifact without rewriting summary.md through write_text_artifact.
        child.artifacts.append(AgentArtifact(path=str(summary_path), kind="task_summary", description=f"Summary for task {name}"))
        child.context.add_artifact(summary_path, "task_summary", f"Summary for task {name}")
        child.record("task_done", name, f"status={status}; {summary}", {"result_path": str(result_path)}, importance=3)
        child.context.save()
        return result

    def _attach_child_result(self, result: AgentTaskResult) -> None:
        compact_child = {
            "task_id": result.task_id,
            "name": result.name,
            "status": result.status,
            "summary": result.summary,
            "result_path": result.result_path,
            "context_path": result.context_path,
            "duration_seconds": result.duration_seconds,
            "error": result.error,
        }
        self.child_tasks.append(compact_child)
        self.record(
            "task_completed" if result.ok else "task_failed",
            result.name,
            result.summary,
            compact_child,
            importance=3 if not result.ok else 2,
        )

    def save_loop_index(self) -> Path:
        path = self.workspace_dir / "loop_index.json"
        data = {
            "task_id": self.task_id,
            "name": self.name,
            "objective": self.objective,
            "depth": self.depth,
            "workspace_dir": str(self.workspace_dir),
            "context_path": str(self.context.state_path),
            "child_tasks": self.child_tasks,
            "artifacts": [asdict(a) for a in self.artifacts],
        }
        path.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


def _render_task_summary(result: AgentTaskResult) -> str:
    lines = [
        f"# Task: {result.name}",
        "",
        f"- Status: `{result.status}`",
        f"- Objective: {result.objective}",
        f"- Duration: {result.duration_seconds:.4f}s",
        f"- Result file: `{result.result_path}`",
        f"- Context file: `{result.context_path}`",
        "",
        "## Summary",
        result.summary or "Task completed.",
        "",
    ]
    if result.error:
        lines.extend(["## Error", result.error, ""])
    if result.output:
        lines.extend(["## Output", "```json", json.dumps(to_jsonable(result.output), ensure_ascii=False, indent=2), "```", ""])
    if result.artifacts:
        lines.append("## Artifacts")
        for artifact in result.artifacts:
            lines.append(f"- `{artifact.kind}`: `{artifact.path}` — {artifact.description}")
        lines.append("")
    if result.child_tasks:
        lines.append("## Child Tasks")
        for task in result.child_tasks:
            lines.append(f"- `{task.get('status')}` {task.get('name')}: {task.get('summary')} (`{task.get('result_path')}`)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
