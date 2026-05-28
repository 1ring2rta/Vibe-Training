from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from autopilot.llm.openai_compatible import parse_jsonish
from autopilot.models import to_jsonable

ActionType = Literal["run_tool", "stop"]

ATOMIC_ACTION_TYPES: tuple[str, ...] = ("run_tool", "stop")
ATOMIC_TOOL_NAMES: tuple[str, ...] = ("bash", "cat", "grep", "web_search", "browser", "answer_human")

# Backward-compatibility parser aliases. These are not exposed in prompts or tool
# inventories; they only stop old trajectories from forcing WAITING_HUMAN.
TOOL_ACTION_ALIASES: dict[str, str] = {
    "tool": "run_tool",
    "shell": "bash",
    "sh": "bash",
    "terminal": "bash",
    "command": "bash",
    "read": "cat",
    "read_file": "cat",
    "cat_file": "cat",
    "search": "grep",
    "grep_files": "grep",
    "web": "web_search",
    "web_fetch": "browser",
    "browse": "browser",
    "fetch": "browser",
    "ask_human": "answer_human",
    "human": "answer_human",
    "reply_human": "answer_human",
}

WORKFLOW_ACTION_TYPES: tuple[str, ...] = (
    "collect_data",
    "prepare_data",
    "train",
    "run_eval",
    "deploy_model",
    "prepare_eval_program",
    "refine_eval_program",
    "patch_repo",
    "update_memory",
    "start_process",
    "kill_process",
    "list_processes",
    "inspect_artifacts",
    "list_artifacts",
    "eval_program_write",
    "eval_program_refine",
)


@dataclass
class ResourceRequest:
    environment: str | None = None
    cuda_visible_devices: str | None = None
    exclusive_gpu: bool = False
    cpu_only: bool = False
    timeout_seconds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class ExpectedArtifact:
    path: str
    kind: str = "file_exists"
    required: bool = True
    description: str = ""
    min_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class AgentAction:
    action_id: str = field(default_factory=lambda: f"act-{uuid.uuid4().hex[:12]}")
    action_type: ActionType = "run_tool"
    objective: str = ""
    rationale: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    expected_artifacts: list[ExpectedArtifact] = field(default_factory=list)
    resource_request: ResourceRequest | None = None
    risk_level: Literal["low", "medium", "high"] = "low"
    requires_human: bool = False

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @property
    def is_valid_atomic(self) -> bool:
        if self.action_type not in ATOMIC_ACTION_TYPES:
            return False
        if self.action_type == "run_tool" and self.tool_name not in ATOMIC_TOOL_NAMES:
            return False
        return True

    @classmethod
    def invalid(cls, *, raw: Any, reason: str, objective: str = "invalid director action") -> "AgentAction":
        # Represent parser/protocol bugs as a failed atomic bash action. This keeps
        # the loop self-repairable and avoids asking the user to fix JSON.
        payload = json.dumps({"ok": False, "error": reason, "raw": raw}, ensure_ascii=False)
        return cls(
            action_type="run_tool",
            objective=objective,
            rationale=reason,
            tool_name="bash",
            arguments={"command": f"python - <<'PY'\nprint({payload!r})\nraise SystemExit(2)\nPY", "timeout": 30, "_autopilot_invalid_action": True, "_autopilot_error": reason},
            risk_level="low",
            requires_human=False,
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "AgentAction":
        raw_action_type = str(data.get("action_type") or data.get("type") or data.get("action") or "run_tool")
        action_type = TOOL_ACTION_ALIASES.get(raw_action_type, raw_action_type)

        args = data.get("arguments") or data.get("args") or {}
        if not isinstance(args, dict):
            args = {"raw": args}

        tool_name = data.get("tool_name") or data.get("tool") or data.get("name")
        if isinstance(action_type, str) and action_type in ATOMIC_TOOL_NAMES:
            tool_name = action_type
            action_type = "run_tool"
        elif isinstance(action_type, str) and action_type in TOOL_ACTION_ALIASES and TOOL_ACTION_ALIASES[action_type] in ATOMIC_TOOL_NAMES:
            tool_name = TOOL_ACTION_ALIASES[action_type]
            action_type = "run_tool"
        elif action_type == "run_tool" and tool_name in TOOL_ACTION_ALIASES:
            tool_name = TOOL_ACTION_ALIASES[str(tool_name)]

        if raw_action_type == "ask_human":
            action_type = "run_tool"
            tool_name = "answer_human"
            question = args.get("question") or data.get("objective") or "Need user decision"
            args = {
                "message": question,
                "kind": "decision_request" if args.get("choices") or args.get("options") else "ack",
                "requires_response": bool(args.get("choices") or args.get("options") or data.get("requires_human", False)),
                "choices": args.get("choices") or args.get("options") or [],
                "related_note_ids": args.get("related_note_ids") or [],
                "context": args.get("context"),
            }

        # Lossless, safe conversion for common old process/eval actions that already
        # contain an explicit command. This is not a workflow: the resulting action is
        # a single bash call with fully visible command text.
        if raw_action_type in {"start_process", "deploy_model", "run_eval", "train"} and args.get("command"):
            tool_name = "bash"
            action_type = "run_tool"
            if raw_action_type == "start_process" or args.get("background") or args.get("detached"):
                args = {**args, "detached": True}
        elif raw_action_type in {"kill_process", "list_processes", "inspect_artifacts", "list_artifacts"}:
            # Map old convenience actions to shell-level inspection/control so no
            # macro executor branch is needed.
            if raw_action_type in {"inspect_artifacts", "list_artifacts"}:
                path = str(args.get("path") or ".")
                limit = int(args.get("limit") or 200)
                args = {"command": f"find {path!r} -maxdepth 4 -print | sort | head -n {limit}", "timeout": 30}
            elif raw_action_type == "list_processes":
                args = {"command": "ps -eo pid,ppid,stat,etime,cmd --sort=pid | tail -n +1 | head -200", "timeout": 30}
            else:
                # Requiring a concrete pid/pattern keeps destructive process control explicit.
                pid = args.get("pid")
                pattern = args.get("pattern") or args.get("name")
                if pid:
                    args = {"command": f"kill -TERM {int(pid)}", "timeout": 30}
                elif pattern:
                    args = {"command": f"pkill -TERM -f {str(pattern)!r}", "timeout": 30}
                else:
                    return cls.invalid(raw=data, reason="kill_process requires pid or pattern; rewrite as an explicit bash command")
            action_type = "run_tool"
            tool_name = "bash"
        elif raw_action_type in WORKFLOW_ACTION_TYPES:
            return cls.invalid(
                raw=data,
                reason=(
                    f"workflow action '{raw_action_type}' is not available in atomic-agent mode; "
                    "use bash/cat/grep/web_search/browser one step at a time"
                ),
            )

        if action_type not in ATOMIC_ACTION_TYPES:
            return cls.invalid(raw=data, reason=f"unsupported action_type='{raw_action_type}'; allowed={list(ATOMIC_ACTION_TYPES)}")
        if action_type == "run_tool" and tool_name not in ATOMIC_TOOL_NAMES:
            return cls.invalid(raw=data, reason=f"unsupported tool_name='{tool_name}'; allowed={list(ATOMIC_TOOL_NAMES)}")

        artifacts: list[ExpectedArtifact] = []
        for row in data.get("expected_artifacts") or data.get("artifacts") or []:
            if isinstance(row, str):
                artifacts.append(ExpectedArtifact(path=row))
            elif isinstance(row, dict):
                artifacts.append(ExpectedArtifact(
                    path=str(row.get("path") or row.get("file") or ""),
                    kind=str(row.get("kind") or row.get("validator") or "file_exists"),
                    required=bool(row.get("required", True)),
                    description=str(row.get("description") or ""),
                    min_count=row.get("min_count"),
                ))

        rr = None
        raw_rr = data.get("resource_request") or data.get("resources")
        if isinstance(raw_rr, dict):
            gpu_alloc = raw_rr.get("gpu_allocation") if isinstance(raw_rr.get("gpu_allocation"), dict) else {}
            rr = ResourceRequest(
                environment=raw_rr.get("environment") or raw_rr.get("training_environment"),
                cuda_visible_devices=raw_rr.get("cuda_visible_devices") or gpu_alloc.get("cuda_visible_devices"),
                exclusive_gpu=bool(raw_rr.get("exclusive_gpu", False)),
                cpu_only=bool(raw_rr.get("cpu_only", False)),
                timeout_seconds=raw_rr.get("timeout_seconds") or raw_rr.get("timeout"),
            )

        return cls(
            action_id=str(data.get("action_id") or data.get("id") or f"act-{uuid.uuid4().hex[:12]}"),
            action_type=action_type,  # type: ignore[arg-type]
            objective=str(data.get("objective") or data.get("goal") or data.get("summary") or ""),
            rationale=str(data.get("rationale") or data.get("reason") or data.get("why") or ""),
            tool_name=str(tool_name) if tool_name else None,
            arguments=args,
            expected_artifacts=artifacts,
            resource_request=rr,
            risk_level=str(data.get("risk_level") or data.get("risk") or "low"),  # type: ignore[arg-type]
            requires_human=bool(data.get("requires_human", False)),
        )


def action_schema_for_prompt() -> dict[str, Any]:
    return {
        "action_id": "stable id, optional",
        "action_type": list(ATOMIC_ACTION_TYPES),
        "objective": "what this single atomic action accomplishes",
        "rationale": "brief reason grounded in current world state",
        "tool_name": list(ATOMIC_TOOL_NAMES) + ["required when action_type=run_tool"],
        "arguments": "JSON object for the selected atomic tool",
        "expected_artifacts": [
            {"path": "relative/path", "kind": "file_exists|json_exists|jsonl_nonempty|directory_exists", "required": True}
        ],
        "resource_request": {"environment": "optional env name", "cuda_visible_devices": "0,1", "timeout_seconds": 3600},
        "risk_level": "low|medium|high",
        "requires_human": False,
        "rules": [
            "Only use run_tool or stop. Use run_tool/tool_name=answer_human to talk to the user.",
            "Only use bash/cat/grep/web_search/browser/answer_human as tools.",
            "Skills are instructions, not tools. Read SKILL.md with cat before following one.",
            "Do not call hidden workflow CLIs or macro actions. Compose behavior from atomic tool calls.",
            "Use answer_human only for acknowledgements or real policy/resource/quality choices; never ask for JSON/schema/path/tool bugs.",
        ],
        "examples": [
            {"action_type": "run_tool", "tool_name": "cat", "objective": "Read a skill before using it", "arguments": {"path": ".autopilot/skills/trusted_eval/SKILL.md"}},
            {"action_type": "run_tool", "tool_name": "grep", "objective": "Find AIME leakage in training manifests", "arguments": {"pattern": "(?i)aime|1983-2024", "path": ".", "include": "*.json*"}},
            {"action_type": "run_tool", "tool_name": "web_search", "objective": "Find broad math reasoning datasets", "arguments": {"query": "olympiad style math reasoning dataset excluding AIME 2024", "limit": 10}},
            {"action_type": "run_tool", "tool_name": "bash", "objective": "Start vLLM with visible explicit command", "arguments": {"command": "CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/model --tensor-parallel-size 2 --port 8000", "timeout": 30, "detached": True, "label": "vllm"}},
            {"action_type": "run_tool", "tool_name": "answer_human", "objective": "Acknowledge a user note without blocking", "arguments": {"message": "Noted. I will avoid AIME-related training data and keep benchmark files eval_only.", "kind": "ack", "requires_response": False}},
            {"action_type": "run_tool", "tool_name": "answer_human", "objective": "Ask for a real choice", "arguments": {"message": "Choose data strategy: smaller clean data or larger risky data?", "kind": "decision_request", "requires_response": True, "choices": [{"id": "clean", "label": "smaller clean data"}, {"id": "risky", "label": "larger risky data"}], "recommended_choice": "clean"}},
        ],
    }


def parse_agent_action(content: str, *, default_objective: str = "") -> AgentAction:
    parsed = parse_jsonish(content or "")
    if isinstance(parsed, dict):
        if isinstance(parsed.get("action"), dict):
            return AgentAction.from_mapping(parsed["action"])
        if isinstance(parsed.get("next_action"), dict):
            return AgentAction.from_mapping(parsed["next_action"])
        if isinstance(parsed.get("tool_call"), dict):
            call = parsed["tool_call"]
            return AgentAction.from_mapping({"action_type": "run_tool", "tool_name": call.get("name"), "arguments": call.get("arguments") or {}})
        if any(k in parsed for k in ["action_type", "type", "tool_name", "arguments", "action"]):
            return AgentAction.from_mapping(parsed)
    return AgentAction.invalid(raw=content, reason="director did not return valid action JSON", objective=default_objective)
