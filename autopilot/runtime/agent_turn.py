from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from autopilot.llm.openai_compatible import parse_jsonish
from autopilot.runtime.clients import LLMClientRegistry
from autopilot.runtime.state import RunStateStore
from autopilot.runtime.tools import ModelToolRegistry, WaitingForHuman, WaitingForUserDecision, tools_prompt_block


@dataclass
class AgentTurnResult:
    status: str
    content: str = ""
    reasoning_content: str | None = None
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    waiting_human: dict[str, Any] | None = None
    raw: Any = None


class AgentTurnRunner:
    """One model turn with tools, event log, trajectory, and human-channel halt."""

    def __init__(self, clients: LLMClientRegistry, tools: ModelToolRegistry, state: RunStateStore | None = None) -> None:
        self.clients = clients
        self.tools = tools
        self.state = state

    @staticmethod
    def _parse_tool_calls_from_content(content: str) -> list[dict[str, Any]]:
        parsed = parse_jsonish(content or "")
        calls: list[dict[str, Any]] = []
        if isinstance(parsed, dict):
            if isinstance(parsed.get("tool_call"), dict):
                calls.append(parsed["tool_call"])
            if isinstance(parsed.get("tool_calls"), list):
                calls.extend([x for x in parsed["tool_calls"] if isinstance(x, dict)])
        return calls

    @staticmethod
    def _native_tool_call_to_simple(call: dict[str, Any]) -> dict[str, Any]:
        fn = call.get("function") or {}
        args = fn.get("arguments") or call.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"raw": args}
        return {"id": call.get("id"), "name": fn.get("name") or call.get("name"), "arguments": args}

    def run(
        self,
        *,
        role: str,
        objective: str,
        messages: list[dict[str, Any]],
        purpose: str | None = None,
        max_tool_iterations: int = 4,
        params: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        enable_tools: bool = True,
        return_first_tool_call_as_result: bool = False,
    ) -> AgentTurnResult:
        if self.state is not None:
            self.state.append_event("agent_turn_start", {"role": role, "purpose": purpose, "objective": objective})
        tool_schemas = self.tools.openai_tools() if enable_tools else []
        # JSON-fallback clients still see the same atomic tool inventory in prompt text.
        system_prefix = {"role": "system", "content": "You are an Autopilot atomic-tool agent turn. Tools are registered for this request. Choose one tool/action; do not execute hidden workflows.\n" + tools_prompt_block(self.tools)}
        work_messages = [system_prefix] + list(messages)
        tool_results: list[dict[str, Any]] = []
        for iteration in range(max_tool_iterations + 1):
            result = self.clients.call_role(
                role,
                work_messages,
                purpose=purpose or objective,
                tools=tool_schemas,
                tool_choice="auto",
                params=params,
                metadata={"objective": objective, "iteration": iteration, **dict(metadata or {})},
            )
            calls = [self._native_tool_call_to_simple(c) for c in (result.tool_calls or [])] if enable_tools else []
            if enable_tools and not calls:
                calls = self._parse_tool_calls_from_content(result.content)
            if calls and return_first_tool_call_as_result:
                call = calls[0]
                content = json.dumps({"action_type": "run_tool", "tool_name": call.get("name"), "arguments": call.get("arguments") or {}, "objective": objective, "rationale": "native tool_call selected by director"}, ensure_ascii=False)
                if self.state is not None:
                    self.state.append_event("agent_turn_complete", {"role": role, "purpose": purpose, "content_preview": content[:300], "returned_tool_call": call.get("name")})
                return AgentTurnResult(status="success", content=content, reasoning_content=result.reasoning_content, tool_results=tool_results, raw=result.raw)
            if not calls:
                if self.state is not None:
                    self.state.append_event("agent_turn_complete", {"role": role, "purpose": purpose, "content_preview": (result.content or "")[:300]})
                return AgentTurnResult(status="success", content=result.content, reasoning_content=result.reasoning_content, tool_results=tool_results, raw=result.raw)
            work_messages.append({"role": "assistant", "content": result.content or "", "tool_calls": result.tool_calls or []})
            for call in calls:
                name = str(call.get("name") or "")
                args = call.get("arguments") or {}
                if not isinstance(args, dict):
                    args = {"raw": args}
                try:
                    obs = self.tools.execute(name, args)
                    simple_obs = {"tool": name, "arguments": args, "result": obs}
                    tool_results.append(simple_obs)
                    if self.state is not None:
                        self.state.append_event("tool_call", simple_obs)
                    work_messages.append({"role": "tool", "tool_call_id": call.get("id") or name, "content": json.dumps(simple_obs, ensure_ascii=False)[:20000]})
                except WaitingForUserDecision as wait:
                    payload = wait.payload
                    if self.state is not None:
                        self.state.append_event("agent_turn_waiting_user_decision", payload)
                    return AgentTurnResult(status="waiting_human", waiting_human=payload, tool_results=tool_results, raw=result.raw)
                except WaitingForHuman as wait:
                    payload = wait.payload
                    if self.state is not None:
                        self.state.append_event("agent_turn_waiting_user_decision", payload)
                    return AgentTurnResult(status="waiting_human", waiting_human=payload, tool_results=tool_results, raw=result.raw)
                except Exception as exc:
                    obs = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "tool": name}
                    tool_results.append({"tool": name, "arguments": args, "result": obs})
                    work_messages.append({"role": "tool", "tool_call_id": call.get("id") or name, "content": json.dumps(obs, ensure_ascii=False)})
        return AgentTurnResult(status="max_tool_iterations", content="", tool_results=tool_results)
