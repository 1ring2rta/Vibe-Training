from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from autopilot.config import Settings
from autopilot.models import to_jsonable
from autopilot.runtime.human_channel import HumanChannel
from autopilot.runtime.processes import ProcessRegistry
from autopilot.runtime.state import RunStateStore
from autopilot.tools.bash import BashRunner
from autopilot.tools.web_browser import WebBrowserTool
from autopilot.tools.web_search import WebSearchTool


class WaitingForHuman(RuntimeError):
    def __init__(self, question_id: str, question: str, payload: dict[str, Any]) -> None:
        super().__init__(f"waiting for human instruction: {question_id}: {question}")
        self.question_id = question_id
        self.question = question
        self.payload = payload


class WaitingForUserDecision(RuntimeError):
    def __init__(self, decision_id: str, message: str, payload: dict[str, Any]) -> None:
        super().__init__(f"waiting for user decision: {decision_id}: {message}")
        self.decision_id = decision_id
        self.message = message
        self.payload = payload


@dataclass
class ModelTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any] | None = None
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    def openai_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def prompt_schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters, "enabled": self.enabled, "tags": self.tags}


class ModelToolRegistry:
    def __init__(self, tools: list[ModelTool] | None = None) -> None:
        self.tools: dict[str, ModelTool] = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, tool: ModelTool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> ModelTool | None:
        return self.tools.get(name)

    def openai_tools(self) -> list[dict[str, Any]]:
        return [t.openai_schema() for t in self.tools.values() if t.enabled]

    def prompt_tools(self) -> list[dict[str, Any]]:
        return [t.prompt_schema() for t in self.tools.values() if t.enabled]

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self.get(name)
        if tool is None or not tool.enabled:
            raise KeyError(f"tool not available: {name}")
        if tool.handler is None:
            return {"ok": False, "error": f"tool {name} has no local handler", "arguments": arguments}
        return tool.handler(arguments)


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


def _safe_join(root: Path, rel: str | os.PathLike[str]) -> Path:
    root_resolved = root.resolve()
    p = (root_resolved / Path(rel)).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
    if not str(p).startswith(str(root_resolved)):
        raise ValueError(f"path outside workspace: {p}")
    return p


def build_default_model_tool_registry(
    *,
    workspace: str | Path = ".",
    run_state: RunStateStore | None = None,
    allow_bash: bool = False,
    interactive_human: bool | None = None,
    process_store: ProcessRegistry | None = None,
    settings: Settings | None = None,
) -> ModelToolRegistry:
    """Build the atomic autonomous tool surface.

    The autonomous kernel intentionally exposes only small, composable tools.
    There are no collect/prepare/train/deploy/eval macro tools here; the model
    must build those behaviors out of explicit bash/cat/grep/web_search/browser
    calls whose traces are visible in the event log.
    """

    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    proc_registry = process_store or ProcessRegistry(root)
    settings = settings or Settings()
    human_channel = HumanChannel(root, project_root=Path.cwd())
    human_channel.materialize()

    def cat_handler(args: dict[str, Any]) -> dict[str, Any]:
        try:
            path = _safe_join(root, str(args.get("path") or ""))
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        max_chars = int(args.get("max_chars") or args.get("limit") or 20000)
        offset = int(args.get("offset") or 0)
        if not path.exists() or not path.is_file():
            return {"ok": False, "error": "file not found", "path": str(path)}
        data = path.read_text(encoding="utf-8", errors="replace")
        chunk = data[offset: offset + max_chars]
        return {"ok": True, "path": str(path), "offset": offset, "content": chunk, "truncated": offset + max_chars < len(data), "size_chars": len(data)}

    def grep_handler(args: dict[str, Any]) -> dict[str, Any]:
        pattern = str(args.get("pattern") or args.get("query") or "")
        if not pattern:
            return {"ok": False, "error": "grep requires pattern"}
        try:
            base = _safe_join(root, str(args.get("path") or "."))
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        include = str(args.get("include") or "*")
        max_results = int(args.get("max_results") or args.get("limit") or 100)
        flags = re.I if bool(args.get("ignore_case", False)) else 0
        try:
            rx = re.compile(pattern, flags)
        except re.error:
            rx = re.compile(re.escape(pattern), flags)
        files = [base] if base.is_file() else [p for p in base.rglob(include) if p.is_file()]
        results: list[dict[str, Any]] = []
        for path in files:
            try:
                rel = str(path.resolve().relative_to(root.resolve()))
            except Exception:
                continue
            # Skip large binaries and caches by default.
            if any(part in {".git", "__pycache__", ".venv"} for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    results.append({"path": rel, "line": line_no, "text": line[:1000]})
                    if len(results) >= max_results:
                        return {"ok": True, "pattern": pattern, "root": str(base), "results": results, "truncated": True}
        return {"ok": True, "pattern": pattern, "root": str(base), "results": results, "truncated": False}

    def bash_handler(args: dict[str, Any]) -> dict[str, Any]:
        if not allow_bash:
            return {"ok": False, "error": "bash disabled by kernel settings"}
        command = str(args.get("command") or "")
        if not command.strip():
            return {"ok": False, "error": "bash requires command"}
        try:
            cwd = _safe_join(root, str(args.get("cwd") or "."))
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        env = {str(k): str(v) for k, v in dict(args.get("env") or {}).items()}
        if args.get("cuda_visible_devices") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.get("cuda_visible_devices"))
        setup = args.get("setup_command")
        env_name = args.get("environment")
        if env_name and not setup:
            setup = settings.environment_activation(str(env_name))
        if bool(args.get("detached", False)):
            return proc_registry.start_background(
                command,
                cwd=cwd,
                env=env,
                setup_command=setup,
                label=str(args.get("label") or args.get("name") or "bash_detached"),
                kind=str(args.get("kind") or "bash_service"),
                action_id=args.get("action_id"),
                environment=str(env_name) if env_name else None,
                log_file=args.get("log_file") or (root / ".autopilot/processes/bash_detached.log"),
                pid_file=args.get("pid_file"),
                metadata={"source": "atomic_tool:bash"},
            )
        timeout = int(args.get("timeout") or 120)
        runner = BashRunner(cwd=cwd, timeout=timeout)
        holder: dict[str, Any] = {}

        def on_started(info: dict[str, Any]) -> dict[str, Any]:
            rec = proc_registry.register(
                name=str(args.get("label") or "bash_tool"),
                kind=str(args.get("kind") or "bash_tool"),
                pid=int(info["pid"]),
                process_group_id=info.get("process_group_id"),
                command=str(info.get("command") or command),
                cwd=info.get("cwd") or cwd,
                action_id=args.get("action_id"),
                environment=str(env_name) if env_name else None,
                metadata={"source": "atomic_tool:bash"},
            )
            holder["process_id"] = rec.process_id
            return {"process_id": rec.process_id}

        res = runner.run(command, shell=True, timeout=timeout, stream_output=False, env=env, setup_command=setup, process_started_callback=on_started)
        if holder.get("process_id"):
            proc_registry.mark_finished(str(holder["process_id"]), status="SUCCEEDED" if res.ok else "FAILED", exit_code=res.returncode, reason="bash tool completed")
        return {
            "ok": res.ok,
            "returncode": res.returncode,
            "stdout": res.stdout[-20000:],
            "stderr": res.stderr[-20000:],
            "command": res.command,
            "cwd": res.cwd,
            "pid": res.pid,
            "process_group_id": res.process_group_id,
            "process_id": holder.get("process_id"),
            "process_registry": str(proc_registry.registry_path),
            "timed_out": res.timed_out,
            "duration_seconds": res.duration_seconds,
        }

    def answer_human_handler(args: dict[str, Any]) -> dict[str, Any]:
        result = human_channel.answer(args)
        if result.get("ok") and result.get("requires_response"):
            payload = human_channel.decision_payload({**result, "message": args.get("message") or args.get("answer") or args.get("text")})
            raise WaitingForUserDecision(str(result.get("id") or "decision"), str(args.get("message") or "Decision requested"), payload)
        return result

    def web_search_handler(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        if not query.strip():
            return {"ok": False, "error": "web_search requires query"}
        limit = int(args.get("limit") or 10)
        tool = WebSearchTool(settings=settings, timeout=float(args.get("timeout") or 20))
        try:
            results = [to_jsonable(r) for r in tool.search(query, limit=limit)]
            return {"ok": True, "query": query, "results": results}
        except Exception as exc:
            return {"ok": False, "query": query, "error": f"{type(exc).__name__}: {exc}"}

    def browser_handler(args: dict[str, Any]) -> dict[str, Any]:
        url = str(args.get("url") or args.get("href") or "")
        if not url:
            return {"ok": False, "error": "browser requires url"}
        max_chars = int(args.get("max_chars") or 12000)
        tool = WebBrowserTool(timeout=float(args.get("timeout") or 20))
        page = tool.fetch_text(url, max_chars=max_chars)
        return {"ok": page.error is None, **to_jsonable(page)}

    return ModelToolRegistry([
        ModelTool(
            "bash",
            "Execute one explicit shell command in the run workspace. Use for training, inference, conversion, process management, and small scripts. Supports detached=true for background processes with pid/log tracking.",
            _object_schema({"command": {"type": "string"}, "cwd": {"type": "string"}, "timeout": {"type": "integer"}, "detached": {"type": "boolean"}, "label": {"type": "string"}, "kind": {"type": "string"}, "environment": {"type": "string"}, "setup_command": {"type": "string"}, "log_file": {"type": "string"}, "pid_file": {"type": "string"}, "cuda_visible_devices": {"type": "string"}, "env": {"type": "object"}}, ["command"]),
            bash_handler,
            enabled=True,
            tags=["atomic", "shell", "stateful"],
        ),
        ModelTool(
            "cat",
            "Read a text file under the run workspace. Use before editing configs, reading skills, checking logs, or inspecting generated artifacts.",
            _object_schema({"path": {"type": "string"}, "offset": {"type": "integer"}, "max_chars": {"type": "integer"}}, ["path"]),
            cat_handler,
            tags=["atomic", "filesystem", "read_only"],
        ),
        ModelTool(
            "grep",
            "Search text files under the run workspace by regex or literal pattern. Use for locating configs, logs, contamination terms, and previous run evidence.",
            _object_schema({"pattern": {"type": "string"}, "path": {"type": "string"}, "include": {"type": "string"}, "ignore_case": {"type": "boolean"}, "max_results": {"type": "integer"}}, ["pattern"]),
            grep_handler,
            tags=["atomic", "filesystem", "read_only"],
        ),
        ModelTool(
            "answer_human",
            "Reply to the user through the run-local human channel. Use requires_response=false for acknowledgements. Use requires_response=true only for real decisions with choices.",
            _object_schema({
                "message": {"type": "string"},
                "kind": {"type": "string"},
                "requires_response": {"type": "boolean"},
                "choices": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "label": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["id", "label"],
                    },
                },
                "recommended_choice": {"type": "string"},
                "related_note_ids": {"type": "array", "items": {"type": "string"}},
                "related_action_id": {"type": "string"},
            }, ["message"]),
            answer_human_handler,
            tags=["atomic", "human", "communication"],
        ),
        ModelTool(
            "web_search",
            "Search the web. The agent must write the exact query. Use this instead of curl/wget/search wrappers for external discovery.",
            _object_schema({"query": {"type": "string"}, "limit": {"type": "integer"}, "timeout": {"type": "number"}}, ["query"]),
            web_search_handler,
            tags=["atomic", "network", "search"],
        ),
        ModelTool(
            "browser",
            "Fetch and extract readable text from one URL. Use after web_search to inspect pages, dataset cards, documentation, or papers.",
            _object_schema({"url": {"type": "string"}, "max_chars": {"type": "integer"}, "timeout": {"type": "number"}}, ["url"]),
            browser_handler,
            tags=["atomic", "network", "read"],
        ),
    ])


def tools_prompt_block(registry: ModelToolRegistry) -> str:
    return json.dumps(
        {
            "available_atomic_tools": registry.prompt_tools(),
            "tool_call_protocol": {
                "native": "Use OpenAI tool_calls if supported. The kernel will record exactly one tool call as the next run_tool action.",
                "json_fallback": {"action_type": "run_tool", "tool_name": "bash|cat|grep|web_search|browser|answer_human", "arguments": {}},
            },
        },
        ensure_ascii=False,
        indent=2,
    )
