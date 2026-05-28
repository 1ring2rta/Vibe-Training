from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopilot.config import Settings
from autopilot.kernel.action_schema import AgentAction
from autopilot.kernel.artifact_policy import ArtifactPolicyReport, scan_benchmark_leakage
from autopilot.kernel.contracts import ContractReport, validate_contracts
from autopilot.kernel.permissions import PermissionPolicy
from autopilot.models import to_jsonable
from autopilot.runtime.state import RunStateStore
from autopilot.runtime.tools import ModelToolRegistry, WaitingForHuman, WaitingForUserDecision


@dataclass
class ActionOutcome:
    action_id: str
    action_type: str
    ok: bool
    status: str = "completed"
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    contracts: ContractReport | None = None
    artifact_policy: ArtifactPolicyReport | None = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


class ActionExecutor:
    """Execute exactly one atomic action.

    Autonomous behavior is intentionally not encoded here. There are no
    collect/prepare/train/deploy/eval branches. The model composes all of those
    from bash/cat/grep/web_search/browser, and this executor only enforces
    permissions, artifact contracts, and artifact-level safety policy.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        settings: Settings,
        tools: ModelToolRegistry,
        state: RunStateStore,
        execute: bool = True,
        permissions: PermissionPolicy | None = None,
        goal: str = "",
        target: str = "",
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.tools = tools
        self.state = state
        self.execute = execute
        self.permissions = permissions or PermissionPolicy()
        self.goal = goal
        self.target = target

    def execute_action(self, action: AgentAction) -> ActionOutcome:
        started = time.time()
        decision = self.permissions.check_decision(action.action_type, action.tool_name, action.risk_level, action.arguments)
        if not decision.allow:
            outcome = ActionOutcome(action.action_id, action.action_type, False, status="denied", error=decision.reason, elapsed_seconds=round(time.time() - started, 4))
            self.state.append_event("autonomous_action_denied", outcome.to_dict())
            return outcome

        task = self.state.upsert_task(action.action_id, action.objective or action.action_type, task_id=action.action_id, status="RUNNING")
        self.state.write_state(status="RUNNING", current_task=task.task_id)
        self.state.append_event("autonomous_action_start", action.to_dict())
        try:
            if action.action_type == "stop":
                return self._finish(action, task.task_id, {"ok": True, "stop_reason": action.rationale or action.objective}, started)
            if action.action_type != "run_tool":
                return self._finish(action, task.task_id, {"ok": False, "error": f"non-atomic action denied: {action.action_type}"}, started)
            if not action.tool_name:
                return self._finish(action, task.task_id, {"ok": False, "error": "run_tool missing tool_name"}, started)

            if not self.execute:
                result = {"ok": True, "planned_only": True, "tool_name": action.tool_name, "arguments": action.arguments}
            else:
                args = dict(action.arguments)
                args.setdefault("action_id", action.action_id)
                result = {"ok": True, "tool_name": action.tool_name, "tool_result": self.tools.execute(action.tool_name, args)}
            return self._finish(action, task.task_id, result, started)
        except WaitingForUserDecision as wait:
            self.state.mark_task(task.task_id, "WAITING_USER_DECISION", result=wait.payload)
            self.state.write_state(status="WAITING_USER_DECISION", waiting_user_decision=wait.payload, current_task=task.task_id)
            self.state.append_event("waiting_user_decision", wait.payload)
            return ActionOutcome(action.action_id, action.action_type, False, status="waiting_user_decision", result=wait.payload, elapsed_seconds=round(time.time() - started, 4))
        except WaitingForHuman as wait:
            self.state.mark_task(task.task_id, "WAITING_USER_DECISION", result=wait.payload)
            self.state.write_state(status="WAITING_USER_DECISION", waiting_user_decision=wait.payload, current_task=task.task_id)
            self.state.append_event("waiting_user_decision", wait.payload)
            return ActionOutcome(action.action_id, action.action_type, False, status="waiting_user_decision", result=wait.payload, elapsed_seconds=round(time.time() - started, 4))
        except Exception as exc:
            outcome = ActionOutcome(action.action_id, action.action_type, False, status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=round(time.time() - started, 4))
            self.state.mark_task(task.task_id, "FAILED", result=outcome.to_dict(), error=outcome.error)
            self.state.append_event("autonomous_action_failed", outcome.to_dict())
            return outcome

    @staticmethod
    def _is_real_choice(action: AgentAction) -> bool:
        args = action.arguments or {}
        options = args.get("options") or args.get("choices")
        if isinstance(options, list) and len(options) >= 2:
            return True
        question = str(args.get("question") or action.objective or "").lower()
        choice_words = ["choose", "select", "approve", "confirm", "which", "选择", "确认", "批准", "取舍"]
        bug_words = ["invalid", "schema", "json", "path", "file not found", "unsupported", "traceback", "exception", "bug", "error"]
        return any(w in question for w in choice_words) and not any(w in question for w in bug_words)

    def _finish(self, action: AgentAction, task_id: str, result: dict[str, Any], started: float, *, status: str = "completed") -> ActionOutcome:
        contracts = validate_contracts(self.root, action.expected_artifacts)
        artifact_policy = scan_benchmark_leakage(self.root, goal=self.goal, target=self.target)
        ok = bool(result.get("ok", True)) and contracts.ok and artifact_policy.ok
        outcome = ActionOutcome(
            action.action_id,
            action.action_type,
            ok,
            status=status,
            result=result,
            contracts=contracts,
            artifact_policy=artifact_policy,
            elapsed_seconds=round(time.time() - started, 4),
        )
        self.state.mark_task(task_id, "SUCCEEDED" if ok else "FAILED", result=outcome.to_dict(), error=None if ok else "artifact contract, policy, or action failed")
        self.state.append_event("autonomous_action_complete", outcome.to_dict())
        if not artifact_policy.ok:
            self.state.append_event("artifact_policy_violation", artifact_policy.to_dict())
        return outcome
