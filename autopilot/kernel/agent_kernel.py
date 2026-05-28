from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopilot.config import Settings
from autopilot.eval.benchmarks import BenchmarkRegistry
from autopilot.kernel.action_schema import AgentAction, action_schema_for_prompt, parse_agent_action
from autopilot.kernel.executor import ActionExecutor
from autopilot.kernel.permissions import PermissionPolicy
from autopilot.kernel.stop_policy import EvalPolicy
from autopilot.kernel.world_state import WorldStateBuilder
from autopilot.models import to_jsonable
from autopilot.runtime.agent_turn import AgentTurnRunner
from autopilot.runtime.clients import LLMClientRegistry
from autopilot.runtime.human_channel import HumanChannel
from autopilot.runtime.memory import AgentMemory
from autopilot.runtime.skills import SkillLibrary
from autopilot.runtime.state import RunStateStore
from autopilot.runtime.processes import ProcessRegistry
from autopilot.runtime.tools import build_default_model_tool_registry


@dataclass
class KernelRunReport:
    root: str
    status: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    trajectory_audit: dict[str, Any] | None = None
    maintenance_drafts: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


class AutonomousAgentKernel:
    """Atomic-tool autonomous post-training loop.

    The kernel is deliberately de-workflowed. It exposes a small tool surface
    (bash/cat/grep/web_search/browser) and instruction-only skills. It does not
    encode collect/prepare/train/deploy/eval macro actions; the director model
    must compose every run step from explicit atomic tool calls.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        root: str | Path,
        goal: str,
        target: str = "",
        execute: bool = True,
        role: str = "director",
        max_iterations: int = 64,
        max_seconds: float | None = None,
        allow_bash: bool = True,
        permission_policy: PermissionPolicy | None = None,
    ) -> None:
        self.settings = settings
        self.root = Path(root)
        self.goal = goal
        self.target = target
        self.execute = execute
        self.role = role
        self.max_iterations = max_iterations
        self.max_seconds = max_seconds
        self.state = RunStateStore(self.root)
        self.processes = ProcessRegistry(self.root)
        self.clients = LLMClientRegistry.from_settings(settings, trajectory_root=self.root / ".autopilot" / "frontier_trajectory")
        self.skills = SkillLibrary(self.root)
        self.memory = AgentMemory(self.root)
        self.human_channel = HumanChannel(self.root, project_root=Path.cwd())
        self.skills.materialize_builtin_skills()
        self.memory.materialize_builtin_memory()
        self.human_channel.materialize()
        self.tools = build_default_model_tool_registry(workspace=self.root, run_state=self.state, allow_bash=allow_bash, process_store=self.processes, settings=settings)
        self.world = WorldStateBuilder(self.root, settings=settings, goal=goal, target=target)
        self.permissions = permission_policy or PermissionPolicy()
        self.executor = ActionExecutor(root=self.root, settings=settings, tools=self.tools, state=self.state, execute=execute, permissions=self.permissions, goal=goal, target=target)
        self.turn_runner = AgentTurnRunner(self.clients, self.tools, self.state)
        self.eval_policy = EvalPolicy.from_config(settings.raw_config if isinstance(settings.raw_config, dict) else {})
        self.benchmark_registry = BenchmarkRegistry.default()

    def run(self) -> KernelRunReport:
        started = time.time()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state.write_state(status="RUNNING", goal=self.goal, target=self.target, kernel="atomic_autonomous", max_seconds=self.max_seconds)
        actions: list[dict[str, Any]] = []
        maintenance: dict[str, str] | None = None
        for i in range(self.max_iterations):
            elapsed = time.time() - started
            if self.max_seconds is not None and elapsed >= self.max_seconds:
                self.state.append_event("time_budget_exhausted", {"elapsed_seconds": round(elapsed, 4), "max_seconds": self.max_seconds, "iteration": i})
                self.state.write_state(status="TIME_BUDGET_EXHAUSTED", elapsed_seconds=round(elapsed, 4), max_seconds=self.max_seconds, stop_reason="max_seconds reached")
                break

            world = self._world(iteration=i, elapsed=elapsed)
            stop = self.eval_policy.decide(self._latest_evaluation(world))
            self.state.append_event("stop_policy_check", stop.to_dict())
            if stop.stop:
                self.state.write_state(status="SUCCEEDED", stop_reason=stop.reason)
                break

            action = self._ask_director(world, iteration=i, stop_policy=stop.to_dict())
            if action.action_type == "stop" and not stop.stop:
                # stop is a request; the policy is the hard gate.
                action = AgentAction(
                    action_type="run_tool",
                    tool_name="bash",
                    objective="Stop request denied by stop_policy; record denial and continue",
                    rationale=stop.reason,
                    arguments={"command": f"echo 'stop denied: {stop.reason}'", "timeout": 30},
                    risk_level="low",
                )

            outcome = self.executor.execute_action(action)
            actions.append({"action": action.to_dict(), "outcome": outcome.to_dict()})
            if outcome.status in {"waiting_human", "waiting_user_decision"}:
                self.state.write_state(status="WAITING_USER_DECISION", waiting_user_decision=outcome.result)
                break
            if action.action_type == "stop" and outcome.ok:
                self.state.write_state(status="STOPPED", stop_reason=action.rationale or action.objective)
                break
        else:
            self.state.write_state(status="MAX_ITERATIONS", max_iterations=self.max_iterations, elapsed_seconds=round(time.time() - started, 4), max_seconds=self.max_seconds)

        audit = self.clients.trajectory_recorder.audit() if self.clients.trajectory_recorder else None
        report = KernelRunReport(root=str(self.root), status=self.state.state().get("status", "UNKNOWN"), actions=actions, elapsed_seconds=round(time.time() - started, 4), trajectory_audit=audit)
        try:
            maintenance = self.memory.write_run_maintenance_drafts(report=report.to_dict(), event_log_path=self.root / "event_log.jsonl")
            report.maintenance_drafts = maintenance
            self.state.append_event("memory_skill_maintenance_drafts", maintenance)
        except Exception as exc:
            self.state.append_event("memory_skill_maintenance_failed", {"error": f"{type(exc).__name__}: {exc}"})
        out = self.root / "autonomous_kernel_report.json"
        out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report

    def _world(self, *, iteration: int, elapsed: float) -> dict[str, Any]:
        world = self.world.materialize()
        world["kernel_budget"] = {
            "elapsed_seconds": round(elapsed, 4),
            "max_seconds": self.max_seconds,
            "remaining_seconds": (round(max(self.max_seconds - elapsed, 4), 4) if self.max_seconds is not None else None),
            "iteration": iteration,
            "max_iterations": self.max_iterations,
        }
        world["atomic_tools"] = self.tools.prompt_tools()
        world["skill_index"] = self.skills.prompt_index()
        world["memory"] = self.memory.index().to_dict()
        world["human_channel"] = self.human_channel.snapshot().to_dict()
        world["mode"] = {
            "name": "atomic_autonomous",
            "actions": ["run_tool", "stop"],
            "tools": ["bash", "cat", "grep", "web_search", "browser", "answer_human"],
            "skills_are_tools": False,
            "workflow_actions_available": False,
        }
        return world

    def _latest_evaluation(self, world: dict[str, Any]) -> dict[str, Any] | None:
        for item in reversed(world.get("round_metrics_history") or []):
            if isinstance(item, dict):
                delta = item.get("metric_delta") or {}
                post = delta.get("post") or {}
                if post:
                    return post
        report = world.get("goal_loop_report") or {}
        if isinstance(report, dict):
            ev = report.get("evaluation") or report.get("current_evaluation")
            if isinstance(ev, dict):
                return ev
        candidates = [self.root / "evaluation_result.json"]
        candidates.extend(sorted((self.root / ".autopilot" / "eval_programs").glob("**/evaluation*_result.json")) if (self.root / ".autopilot" / "eval_programs").exists() else [])
        for path in reversed(candidates):
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict):
                data.setdefault("_path", str(path.relative_to(self.root) if path.is_relative_to(self.root) else path))
                return data
        return None

    def _ask_director(self, world: dict[str, Any], *, iteration: int, stop_policy: dict[str, Any]) -> AgentAction:
        system = {
            "role": "system",
            "content": (
                "You are an autonomous post-training agent running in atomic-tool mode. "
                "Choose exactly one next action as strict JSON or one native tool_call. "
                "Allowed actions: run_tool, stop. Allowed tools: bash, cat, grep, web_search, browser, answer_human. "
                "There are no collect_data/prepare_data/train/deploy/run_eval workflow actions. "
                "Compose all data search, conversion, inference, training, evaluation, process management, and memory updates from atomic tools. "
                "Skills are instruction files, not tools; read a relevant .autopilot/skills/*/SKILL.md with cat when needed. "
                "The model must write every web_search query itself. "
                "Never use target benchmark names as positive training-data search terms. "
                "Use answer_human to acknowledge user notes or request a real decision; repair bugs yourself. Do not use ask_human. "
                "Stop only when stop_policy says stop=true."
            ),
        }
        user = {
            "role": "user",
            "content": json.dumps(
                {
                    "iteration": iteration,
                    "goal": self.goal,
                    "target": self.target,
                    "world_state": world,
                    "stop_policy": stop_policy,
                    "recommended_real_benchmarks": [b.to_dict() for b in self.benchmark_registry.infer(self.goal, self.target)],
                    "required_action_schema": action_schema_for_prompt(),
                },
                ensure_ascii=False,
                indent=2,
            ),
        }
        try:
            result = self.turn_runner.run(
                role=self.role,
                objective="Choose the next atomic tool action.",
                purpose="atomic_autonomous_next_action",
                messages=[system, user],
                max_tool_iterations=0,
                metadata={"kernel_iteration": iteration, "mode": "atomic_autonomous"},
                enable_tools=True,
                return_first_tool_call_as_result=True,
            )
            if result.status == "waiting_human" and result.waiting_human:
                return AgentAction(action_type="run_tool", tool_name="answer_human", objective="Director requested user decision", arguments={**result.waiting_human, "requires_response": True}, requires_human=False)
            action = parse_agent_action(result.content, default_objective="director response")
            if action.arguments.get("_autopilot_invalid_action"):
                repaired = self._repair_action(world, raw_content=result.content, reason=str(action.arguments.get("_autopilot_error") or action.rationale), iteration=iteration)
                if repaired is not None:
                    action = repaired
            self.state.append_event("director_action_parsed", action.to_dict())
            return action
        except Exception as exc:
            self.state.append_event("director_action_failed", {"error": f"{type(exc).__name__}: {exc}"})
            return self._fallback_action(world, iteration=iteration)

    def _repair_action(self, world: dict[str, Any], *, raw_content: str, reason: str, iteration: int) -> AgentAction | None:
        repair_system = {
            "role": "system",
            "content": (
                "Repair the previous invalid autonomous action. Return one strict JSON action only. "
                "Allowed actions: run_tool, stop. Allowed tools: bash, cat, grep, web_search, browser, answer_human. "
                "Do not ask the human for schema/tool/action bugs. Use atomic tools."
            ),
        }
        repair_user = {
            "role": "user",
            "content": json.dumps({"invalid_reason": reason, "invalid_content": raw_content, "world_state": world, "required_action_schema": action_schema_for_prompt()}, ensure_ascii=False, indent=2),
        }
        try:
            result = self.turn_runner.run(
                role=self.role,
                objective="Repair invalid atomic action JSON.",
                purpose="atomic_action_repair",
                messages=[repair_system, repair_user],
                max_tool_iterations=0,
                metadata={"kernel_iteration": iteration, "repair": True},
                enable_tools=True,
                return_first_tool_call_as_result=True,
            )
            action = parse_agent_action(result.content, default_objective="repaired director response")
            if action.arguments.get("_autopilot_invalid_action"):
                return None
            self.state.append_event("director_action_repaired", action.to_dict())
            return action
        except Exception as exc:
            self.state.append_event("director_action_repair_failed", {"error": f"{type(exc).__name__}: {exc}", "reason": reason})
            return None

    def _fallback_action(self, world: dict[str, Any], *, iteration: int) -> AgentAction:
        # Offline deterministic fallback: inspect memory/skills, then workspace. No workflows.
        if iteration == 0:
            return AgentAction(action_type="run_tool", tool_name="cat", objective="Read always-on atomic-agent memory", arguments={"path": ".autopilot/memory/AGENTS.md", "max_chars": 12000})
        if iteration == 1:
            return AgentAction(action_type="run_tool", tool_name="bash", objective="List run workspace", arguments={"command": "find . -maxdepth 3 -print | sort | head -200", "timeout": 30})
        return AgentAction(action_type="run_tool", tool_name="grep", objective="Search for latest evaluation result", arguments={"pattern": "target_met|score|eval_source", "path": ".", "include": "*.json", "max_results": 50})
