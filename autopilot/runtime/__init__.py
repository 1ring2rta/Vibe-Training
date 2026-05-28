"""Runtime public API with lazy imports.

Keep this module import-light so low-level utilities such as BashRunner can import
`autopilot.runtime.processes` without triggering tool/agent imports and circulars.
"""
from __future__ import annotations

__all__ = [
    "AgentTurnResult",
    "AgentTurnRunner",
    "LLMClientRegistry",
    "LLMClientSpec",
    "RunStateStore",
    "TaskRecord",
    "ModelToolRegistry",
    "WaitingForHuman",
    "WaitingForUserDecision",
    "HumanChannel",
    "build_default_model_tool_registry",
    "FrontierTrajectoryRecorder",
    "ProcessRegistry",
    "ManagedProcessStore",
]


def __getattr__(name: str):
    if name in {"AgentTurnResult", "AgentTurnRunner"}:
        from autopilot.runtime.agent_turn import AgentTurnResult, AgentTurnRunner
        return {"AgentTurnResult": AgentTurnResult, "AgentTurnRunner": AgentTurnRunner}[name]
    if name in {"LLMClientRegistry", "LLMClientSpec"}:
        from autopilot.runtime.clients import LLMClientRegistry, LLMClientSpec
        return {"LLMClientRegistry": LLMClientRegistry, "LLMClientSpec": LLMClientSpec}[name]
    if name in {"RunStateStore", "TaskRecord"}:
        from autopilot.runtime.state import RunStateStore, TaskRecord
        return {"RunStateStore": RunStateStore, "TaskRecord": TaskRecord}[name]
    if name in {"ModelToolRegistry", "WaitingForHuman", "WaitingForUserDecision", "build_default_model_tool_registry"}:
        from autopilot.runtime.tools import ModelToolRegistry, WaitingForHuman, WaitingForUserDecision, build_default_model_tool_registry
        return {"ModelToolRegistry": ModelToolRegistry, "WaitingForHuman": WaitingForHuman, "WaitingForUserDecision": WaitingForUserDecision, "build_default_model_tool_registry": build_default_model_tool_registry}[name]
    if name == "HumanChannel":
        from autopilot.runtime.human_channel import HumanChannel
        return HumanChannel
    if name == "FrontierTrajectoryRecorder":
        from autopilot.runtime.trajectory import FrontierTrajectoryRecorder
        return FrontierTrajectoryRecorder
    if name in {"ProcessRegistry", "ManagedProcessStore"}:
        from autopilot.runtime.processes import ProcessRegistry, ManagedProcessStore
        return {"ProcessRegistry": ProcessRegistry, "ManagedProcessStore": ManagedProcessStore}[name]
    raise AttributeError(name)
