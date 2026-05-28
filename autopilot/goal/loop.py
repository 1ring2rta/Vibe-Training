from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any, Sequence

from autopilot.agent import AgentLoop
from autopilot.config import Settings
from autopilot.context import ContextManager
from autopilot.eval.diagnose import FailureDiagnosis, diagnose_failures
from autopilot.eval.runner import EvaluationResult, evaluate_cases, write_eval_cases_jsonl
from autopilot.goal.spec import GoalSpec, default_eval_cases_for_goal, eval_case_from_mapping
from autopilot.llm.kimi import KimiClient
from autopilot.llm.vllm import VLLMClient
from autopilot.models import to_jsonable
from autopilot.rl.verifier import VerifierPlan, discover_verifiers
from autopilot.kernel.stop_policy import EvalPolicy
from autopilot.goal.round_trace import (
    find_round_conversation_roots,
    merge_kimi_trajectory_sources,
    write_round_metrics,
    write_round_metrics_history,
)
from autopilot.tools.ask_human import AskHumanTool
from autopilot.tools.bash import BashRunner
from autopilot.tools.discovery import discover_tools_for_goal
from autopilot.tools.registry import ToolRegistry
from autopilot.tools.memory import append_claude_memory, ensure_post_training_agent, read_claude_memory
from autopilot.tools.repo import RepoSnapshot, collect_repo_snapshot
from autopilot.tools.resource_allocation import ResourceAllocationPlan, fallback_resource_allocation, plan_from_mapping, repair_resource_allocation_plan, choose_environment_for_stage
from autopilot.tools.resources import ComputeResources, collect_compute_resources
from autopilot.tools.vllm_service import VLLMServiceManager, VLLMServicePlan
from autopilot.tools.web_search import WebSearchTool


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _render_markdown_report(
    spec: GoalSpec,
    *,
    eval_result: EvaluationResult | None,
    diagnosis: FailureDiagnosis | None,
    verifier_plan: VerifierPlan | None,
    commands: list[str],
    resources: ComputeResources | None = None,
    vllm_plan: VLLMServicePlan | None = None,
    model_plans: list[dict[str, Any]] | None = None,
    resource_plan: ResourceAllocationPlan | None = None,
    environments: list[dict[str, Any]] | None = None,
    human_queue: str | None = None,
    round_metrics: list[dict[str, Any]] | None = None,
    round_trajectories: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        f"# Goal Loop Report: {spec.name}",
        "",
        f"Goal: {spec.description}",
        f"Target: {spec.target.name} >= {spec.target.target}",
        f"Base model: {spec.base_model or 'not set'}",
        f"Eval cases: {len(spec.eval_cases)}",
        "",
    ]
    if environments:
        if isinstance(environments, dict):
            env_iter = [dict(v, name=k) if isinstance(v, dict) else {"name": k, "description": str(v)} for k, v in environments.items()]
        else:
            env_iter = environments
        lines += ["## Runtime Environments"]
        for env in env_iter:
            if not isinstance(env, dict):
                env = {"name": str(env)}
            activation = env.get("activation_command") or env.get("activation") or env.get("setup_command") or "unset"
            lines.append(f"- {env.get('name') or env.get('id')}: {env.get('description') or ''} | activation={activation}")
        lines.append("")
    if resources is not None:
        lines += [
            "## Compute Resources",
            f"host: {resources.hostname}",
            f"cpu_count: {resources.cpu_count}",
            f"memory_total_gb: {resources.memory_total_gb}",
            f"disk_free_gb: {resources.disk_free_gb}",
            f"nvidia_smi_ok: {resources.nvidia_smi_ok}",
            f"gpus: {len(resources.gpus)}",
            "",
        ]
    if vllm_plan is not None:
        lines += ["## vLLM Service Plan", f"enabled: {vllm_plan.enabled}", f"base_url: {vllm_plan.base_url}", f"pid_file: {vllm_plan.pid_file}", ""]
    if resource_plan is not None:
        lines += [
            "## Latest Resource Allocation",
            f"source: {resource_plan.source}",
            f"phase: {resource_plan.phase}",
            f"stage: {resource_plan.stage}",
            f"environment: {resource_plan.training_environment or 'none'}",
            f"vllm_action: {resource_plan.vllm_action}",
            f"notes: {resource_plan.notes}",
            "",
        ]
    if eval_result is not None:
        lines += ["## Current Evaluation", f"score: {eval_result.score}", f"target_met: {eval_result.target_met}", f"failures: {len(eval_result.failures)}", ""]
    if diagnosis is not None:
        lines.append("## Failure Diagnosis")
        for tag in diagnosis.weak_tags[:10]:
            lines.append(f"- weak tag: {tag['tag']} ({tag['count']})")
        for rec in diagnosis.recommendations:
            lines.append(f"- {rec}")
        lines.append("")
    if verifier_plan is not None:
        lines += ["## Verifier Plan", f"rl_ready: {verifier_plan.rl_ready}", f"backend: {verifier_plan.backend_suggestion}"]
        for candidate in verifier_plan.candidates[:10]:
            lines.append(f"- {candidate.name}: {candidate.kind}, confidence={candidate.confidence:.2f}; {candidate.notes}")
        lines.append("")
    if model_plans:
        lines.append("## KIMI Model-Director Plans")
        for plan in model_plans[-8:]:
            lines.append(f"- {plan.get('phase')}: {plan.get('summary') or plan.get('notes') or 'plan recorded'}")
        lines.append("")
    if human_queue:
        lines += ["## ask_human Queue", human_queue, ""]
    if round_metrics:
        lines.append("## Round Metric History")
        for item in round_metrics:
            delta = item.get("metric_delta") or {}
            pre = delta.get("pre") or {}
            post = delta.get("post") or {}
            lines.append(
                f"- round {item.get('round')}: {pre.get('score')} -> {post.get('score')} "
                f"(delta={delta.get('score_delta')}, stage={item.get('train_stage')}, training_ok={item.get('training_ok')})"
            )
        lines.append("")
    if round_trajectories:
        lines.append("## Round KIMI Trajectories")
        for item in round_trajectories:
            counts = item.get("counts") or {}
            lines.append(f"- round {item.get('round')}: {item.get('output_dir')} | calls={counts.get('kimi_raw_calls.jsonl', 0)} messages={counts.get('kimi_messages.jsonl', 0)}")
        lines.append("")
    if commands:
        lines.append("## Planned/Executed Commands")
        for cmd in commands:
            lines.append(f"```bash\n{cmd}\n```")
    return "\n".join(lines).rstrip() + "\n"


class GoalLoopRunner:
    """Target-driven nested-loop controller.

    v0.5.3 removes offline sample-only mode. Compute/resource/environment choices are
    model-directed resource-allocation tasks; deterministic fallback only records
    an observable conservative plan when KIMI is unavailable.
    """

    def __init__(
        self,
        *,
        spec: GoalSpec,
        settings: Settings,
        output_dir: str | Path,
        config_path: str | None = None,
        execute: bool = True,
        use_web_search: bool = True,
        use_kimi_samples: bool = True,
        evaluate: bool = True,
        max_generated_tests: int = 5,
        agent_max_iterations: int = 512,
        discover_resources: bool = True,
        self_improve_repo: bool = True,
        manage_vllm: bool = True,
        interactive_human: bool | None = None,
    ) -> None:
        self.spec = spec
        self.settings = settings
        self.output_dir = Path(output_dir)
        self.config_path = config_path or settings.config_path
        self.execute = execute
        self.use_web_search = use_web_search
        self.use_kimi_samples = use_kimi_samples
        self.evaluate = evaluate
        self.discover_resources = discover_resources
        self.self_improve_repo = self_improve_repo and settings.model_control_enabled
        self.manage_vllm = manage_vllm
        self.interactive_human = interactive_human
        self.max_generated_tests = max_generated_tests
        raw_runtime = settings.raw_config.get("runtime") if isinstance(settings.raw_config, dict) else None
        raw_repo = settings.raw_config.get("repo") if isinstance(settings.raw_config, dict) else None
        raw_model_control = settings.raw_config.get("model_control") if isinstance(settings.raw_config, dict) else None
        repo_path_explicit = (isinstance(raw_runtime, dict) and bool(raw_runtime.get("repo_path"))) or (isinstance(raw_repo, dict) and bool(raw_repo.get("path"))) or (isinstance(raw_model_control, dict) and bool(raw_model_control.get("repo_path")))
        memory_root = Path(settings.effective_repo_path).resolve() if repo_path_explicit else self.output_dir.resolve()
        self.context = ContextManager(self.output_dir / ".autopilot" / "context" / "session.json", project_root=memory_root)
        self.agent = AgentLoop.root(
            name="goal_loop",
            objective=f"Reach {spec.target.name}>={spec.target.target} for {spec.description}",
            context=self.context,
            workspace_dir=self.output_dir / ".autopilot" / "agent",
            max_iterations=agent_max_iterations,
        )
        self.kimi: KimiClient | None = None
        self.vllm: VLLMClient | None = None
        self.web_search: WebSearchTool | None = None
        self.registry: ToolRegistry | None = None
        self.verifier_plan: VerifierPlan | None = None
        self.eval_result: EvaluationResult | None = None
        self.diagnosis: FailureDiagnosis | None = None
        self.resources: ComputeResources | None = None
        self.repo_snapshot: RepoSnapshot | None = None
        self.vllm_plan: VLLMServicePlan | None = None
        self.resource_allocation_plan: ResourceAllocationPlan | None = None
        self.model_director_plans: list[dict[str, Any]] = []
        self.round_metric_history: list[dict[str, Any]] = []
        self.round_kimi_trajectories: list[dict[str, Any]] = []
        self.planned_commands: list[str] = []
        self.ask_human_tool = AskHumanTool(self.output_dir, mode=settings.ask_human_mode)
        self.memory_snapshot: dict[str, Any] = {}
        self.post_training_memory_path = ensure_post_training_agent(self.settings.effective_repo_path)

    @property
    def external_execute(self) -> bool:
        return bool(self.execute)

    def _config_arg(self) -> list[str]:
        return ["--config", self.config_path] if self.config_path else []

    def _shell_cmd(self, parts: Sequence[str] | str) -> str:
        if isinstance(parts, str):
            return parts
        return " ".join(shlex.quote(str(p)) for p in parts)

    def _record_command(
        self,
        loop: AgentLoop,
        parts: Sequence[str] | str,
        *,
        execute: bool | None = None,
        timeout: float = 600.0,
        cwd: str | Path | None = None,
        shell: bool | None = None,
        setup_command: str | None = None,
        fail_on_error: bool = False,
    ) -> dict[str, Any]:
        cmd = self._shell_cmd(parts)
        self.planned_commands.append(cmd)
        should_execute = self.external_execute if execute is None else bool(execute)
        if not should_execute:
            print(f"[cmd:plan] {cmd}", flush=True)
            loop.set_result("Planned command; not executed", {"command": cmd})
            return {"command": cmd, "executed": False}
        stream_output = str(os.getenv("AUTOPILOT_STREAM_OUTPUT", "1")).lower() not in {"0", "false", "no", "off"}
        print(f"[cmd:start] {cmd}", flush=True)
        result = BashRunner(cwd=cwd or Path.cwd(), timeout=timeout).run(
            parts,
            shell=shell,
            timeout=timeout,
            setup_command=setup_command,
            stream_output=stream_output,
            stream_prefix=f"[cmd:{loop.name}] " if stream_output else "",
        )
        print(f"[cmd:done] returncode={result.returncode} timed_out={result.timed_out} duration={result.duration_seconds}s :: {cmd}", flush=True)
        loop.record_tool_call(
            "bash.run",
            inputs={"command": result.command, "timeout": timeout, "cwd": result.cwd, "setup_command": result.setup_command},
            output_summary=f"returncode={result.returncode}, ok={result.ok}",
            output={"stdout_tail": result.stdout[-4000:], "stderr_tail": result.stderr[-4000:], "timed_out": result.timed_out},
        )
        loop.set_result("Executed command", {"command": result.command, "returncode": result.returncode, "ok": result.ok})
        if fail_on_error and not result.ok:
            raise RuntimeError(f"Command failed: returncode={result.returncode}, timed_out={result.timed_out}; command={result.command}")
        return {"command": result.command, "executed": True, "returncode": result.returncode, "ok": result.ok, "timed_out": result.timed_out}

    def _set_round_kimi_scope(self, round_idx: int) -> dict[str, Any]:
        """Route goal-controller KIMI calls into a round-local trainable trajectory."""
        if self.kimi is None or self.kimi.recorder is None:
            return {"enabled": False}
        root = self.output_dir / f"round_{round_idx}" / "kimi_trajectory" / "goal_controller"
        session_id = f"goal:{self.spec.name}:round:{round_idx}"
        self.kimi.recorder.root = root
        self.kimi.recorder.session_id = session_id
        return {"enabled": True, "root": str(root), "session_id": session_id}

    def _checkpoint_candidates(self, round_idx: int, stage: str | None = None) -> list[str]:
        """Best-effort list of checkpoint/adapter outputs associated with a round."""
        candidates: list[str] = []
        configs = self._available_training_configs(round_idx)
        yaml_path = Path(configs.get(stage or "", "")) if stage else None
        output_dirs: list[Path] = []
        if yaml_path and yaml_path.exists():
            try:
                import yaml  # type: ignore
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
                if isinstance(data, dict) and data.get("output_dir"):
                    output_dirs.append(Path(str(data["output_dir"])))
            except Exception:
                pass
        output_dirs.append(self.output_dir / f"round_{round_idx}")
        seen: set[str] = set()
        for root in output_dirs:
            if not root.exists():
                continue
            for pattern in ["checkpoint-*", "**/checkpoint-*", "adapter_model*", "**/adapter_model*"]:
                for path in root.glob(pattern):
                    value = str(path)
                    if value not in seen:
                        seen.add(value)
                        candidates.append(value)
                    if len(candidates) >= 20:
                        return candidates
        return candidates

    def _task_record_round_metrics(
        self,
        loop: AgentLoop,
        round_idx: int,
        *,
        pre_eval: EvaluationResult | None,
        post_eval: EvaluationResult | None,
        train_stage: str | None,
        training_ok: bool | None,
        training_result_path: str | None,
    ) -> dict[str, Any]:
        record = write_round_metrics(
            self.output_dir / f"round_{round_idx}",
            round_idx=round_idx,
            metric_name=self.spec.target.name,
            target_value=self.spec.target.target,
            pre_eval=pre_eval,
            post_eval=post_eval,
            train_stage=train_stage,
            training_ok=training_ok,
            training_result_path=training_result_path,
            model_under_test={
                "base_model": self.spec.base_model,
                "vllm_base_url": self.settings.vllm_base_url,
                "vllm_model": self.settings.vllm_model,
                "note": "The post metric reflects whatever model is currently served by the configured eval endpoint.",
            },
            checkpoint_candidates=self._checkpoint_candidates(round_idx, train_stage),
        )
        self.round_metric_history = [x for x in self.round_metric_history if x.get("round") != round_idx]
        self.round_metric_history.append(record)
        self.round_metric_history.sort(key=lambda x: int(x.get("round") or 0))
        history_path = write_round_metrics_history(self.output_dir, self.round_metric_history)
        paths = record.get("paths") or {}
        if paths.get("json"):
            loop.add_artifact(paths["json"], "round_metrics_json", "Before/after evaluation metric delta for this round")
        if paths.get("markdown"):
            loop.add_artifact(paths["markdown"], "round_metrics_markdown", "Human-readable before/after metric delta")
        loop.add_artifact(history_path, "round_metrics_history", "All round before/after metric deltas")
        delta = (record.get("metric_delta") or {}).get("score_delta")
        loop.set_result("Round metrics recorded", {"round": round_idx, "score_delta": delta, "paths": paths, "history": str(history_path)})
        return record

    def _task_finalize_round_kimi_trajectory(self, loop: AgentLoop, round_idx: int) -> dict[str, Any]:
        round_dir = self.output_dir / f"round_{round_idx}"
        sources = find_round_conversation_roots(round_dir, include_goal_controller=True)
        manifest = merge_kimi_trajectory_sources(
            round_idx=round_idx,
            sources=sources,
            output_dir=round_dir / "kimi_trajectory" / "combined",
        )
        self.round_kimi_trajectories = [x for x in self.round_kimi_trajectories if x.get("round") != round_idx]
        self.round_kimi_trajectories.append(manifest)
        self.round_kimi_trajectories.sort(key=lambda x: int(x.get("round") or 0))
        manifest_path = manifest.get("manifest") or str(round_dir / "kimi_trajectory" / "combined" / "manifest.json")
        loop.add_artifact(manifest_path, "round_kimi_trajectory_manifest", "Per-round merged KIMI trajectory manifest")
        for name, path in (manifest.get("merged_files") or {}).items():
            if Path(path).exists():
                loop.add_artifact(path, "round_kimi_trajectory_jsonl", f"Merged per-round KIMI trajectory: {name}")
        loop.add_artifact(manifest.get("dataset_info"), "round_kimi_dataset_info", "LLaMA-Factory dataset_info for round KIMI trajectories")
        loop.set_result("Round KIMI trajectory finalized", {"round": round_idx, "manifest": manifest_path, "counts": manifest.get("counts")})
        return manifest

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[goal:start] {self.spec.description}", flush=True)
        print(f"[goal:target] {self.spec.target.name}>={self.spec.target.target}; execute={self.execute}; output_dir={self.output_dir}", flush=True)
        self.agent.run_task("define_goal", "Persist and normalize target metric, eval set, and loop constraints", self._task_define_goal)
        if self.discover_resources:
            self.agent.run_task("inspect_compute_resources", "Inspect CPU/GPU/RAM/disk with nvidia-smi and local probes", self._task_inspect_compute_resources, raise_on_error=False)
        self.agent.run_task("inspect_runtime_environments", "Write the selectable runtime environment registry", self._task_inspect_runtime_environments, raise_on_error=False)
        self.agent.run_task("load_agent_memory", "Load Claude-compatible memory and PostTrainingAgent.md", self._task_load_agent_memory, raise_on_error=False)
        self.agent.run_task("initialize_clients", "Initialize KIMI, vLLM, and web search clients", self._task_initialize_clients)
        if self.self_improve_repo:
            self.agent.run_task("inspect_autopilot_repository", "Expose the Autopilot repository state to KIMI", self._task_inspect_repository, raise_on_error=False)
            self.agent.run_task("kimi_repo_director_initial_plan", "Let KIMI plan repository/tool improvements before the first training round", lambda loop: self._task_model_director_plan(loop, phase="initial"), raise_on_error=False)
        if self.manage_vllm:
            self.agent.run_task("plan_vllm_service", "Write vLLM start/stop commands for later resource-allocation decisions", self._task_plan_vllm_service, raise_on_error=False)
            self.agent.run_task("allocate_resources_for_eval", "Let KIMI decide whether to start/keep/stop vLLM for evaluation", lambda loop: self._task_allocate_resources(loop, phase="before_eval", stage="eval"), raise_on_error=False)
        self.agent.run_task("build_eval_set", "Load eval set or create KIMI/fallback tests", self._task_build_eval_set)
        self.agent.run_task("discover_tools", "Infer and discover tools required for the goal", self._task_discover_tools)
        self.agent.run_task("discover_verifiers", "Find verifier candidates for RL/RLVR", self._task_discover_verifiers)
        for round_idx in range(1, max(1, self.spec.max_rounds) + 1):
            result = self.agent.run_task(f"round:{round_idx}", f"Round {round_idx}: eval -> data -> prepare -> allocate -> train -> remediate", lambda loop, idx=round_idx: self._task_round(loop, idx), task_type="round", raise_on_error=False)
            if self.eval_result and self.eval_result.target_met:
                self.agent.decide("stop_condition_met", f"{self.spec.target.name}={self.eval_result.score} reached target {self.spec.target.target}")
                break
            if not result.ok:
                self.agent.observe("round_failed", result.error or "round failed", importance=3)
        self.agent.run_task("update_post_training_agent_memory", "Update PostTrainingAgent.md with durable lessons from this loop", self._task_update_post_training_memory, raise_on_error=False)

        report_json = _write_json(self.output_dir / "goal_loop_report.json", {
            "goal": self.spec.to_dict(),
            "compute_resources": self.resources.to_dict() if self.resources else None,
            "runtime_environments": self.settings.environment_summaries(),
            "repo_snapshot": self.repo_snapshot.to_dict() if self.repo_snapshot else None,
            "vllm_service_plan": self.vllm_plan.to_dict() if self.vllm_plan else None,
            "resource_allocation_plan": self.resource_allocation_plan.to_dict() if self.resource_allocation_plan else None,
            "model_director_plans": self.model_director_plans,
            "round_metrics": self.round_metric_history,
            "round_kimi_trajectories": self.round_kimi_trajectories,
            "evaluation": self.eval_result.to_dict() if self.eval_result else None,
            "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None,
            "verifier_plan": self.verifier_plan.to_dict() if self.verifier_plan else None,
            "planned_commands": self.planned_commands,
            "execute": self.execute,
            "kimi_conversations": self.kimi.recorder.paths() if self.kimi and self.kimi.recorder else None,
        })
        report_md = self.output_dir / "goal_loop_report.md"
        report_md.write_text(
            _render_markdown_report(
                self.spec,
                eval_result=self.eval_result,
                diagnosis=self.diagnosis,
                verifier_plan=self.verifier_plan,
                commands=self.planned_commands,
                resources=self.resources,
                vllm_plan=self.vllm_plan,
                resource_plan=self.resource_allocation_plan,
                model_plans=self.model_director_plans,
                environments=self.settings.environment_summaries(),
                human_queue=str(self.ask_human_tool.markdown_path),
                round_metrics=self.round_metric_history,
                round_trajectories=self.round_kimi_trajectories,
            ),
            encoding="utf-8",
        )
        self.agent.add_artifact(report_json, "goal_loop_report_json", "Goal-loop JSON report")
        self.agent.add_artifact(report_md, "goal_loop_report_markdown", "Goal-loop Markdown report")
        self.agent.set_result("Goal loop finished", {"report_json": str(report_json), "report_markdown": str(report_md)})
        self.agent.save_loop_index()
        self.context.save()
        print(f"[goal:done] report_json={report_json}", flush=True)
        return {"report_json": str(report_json), "report_markdown": str(report_md), "agent_root": str(self.output_dir / ".autopilot" / "agent")}

    def _task_define_goal(self, loop: AgentLoop) -> dict[str, Any]:
        path = loop.write_json_artifact("goal_spec.json", self.spec.to_dict(), kind="goal_spec", description="Normalized target-driven goal specification")
        loop.set_result("Goal specification saved", {"path": str(path), "target": self.spec.target.target})
        return {"goal_spec": self.spec.to_dict()}

    def _task_inspect_compute_resources(self, loop: AgentLoop) -> dict[str, Any]:
        self.resources = collect_compute_resources(cwd=Path.cwd())
        path = self.output_dir / "compute_resources.json"
        _write_json(path, self.resources.to_dict())
        loop.add_artifact(path, "compute_resources", "CPU/GPU/RAM/disk snapshot from nvidia-smi and local probes")
        loop.set_result("Compute resources inspected", {"path": str(path), "gpu_count": len(self.resources.gpus), "nvidia_smi_ok": self.resources.nvidia_smi_ok})
        return self.resources.to_dict()

    def _task_inspect_runtime_environments(self, loop: AgentLoop) -> dict[str, Any]:
        registry = self.settings.environment_registry()
        path = self.output_dir / "runtime_environments.json"
        _write_json(path, registry.to_dict())
        loop.add_artifact(path, "runtime_environments", "Selectable pre-installed environments; no environment is active by default")
        loop.set_result("Runtime environments inspected", {"path": str(path), "environment_count": len(registry.list()), "names": registry.names()})
        return {"path": str(path), "environments": registry.to_list()}

    def _task_load_agent_memory(self, loop: AgentLoop) -> dict[str, Any]:
        root = Path(self.settings.effective_repo_path)
        claude = read_claude_memory(root)
        post_path = ensure_post_training_agent(root)
        try:
            post = post_path.read_text(encoding="utf-8", errors="replace")[:25000]
        except Exception:
            post = ""
        self.memory_snapshot = {"claude_memory": claude, "post_training_agent_path": str(post_path), "post_training_agent": post}
        path = self.output_dir / "agent_memory_snapshot.json"
        _write_json(path, self.memory_snapshot)
        loop.add_artifact(path, "agent_memory_snapshot", "Claude-compatible memory plus PostTrainingAgent.md")
        loop.set_result("Agent memory loaded", {"path": str(path), "claude_chars": len(claude), "post_training_chars": len(post)})
        return {"path": str(path), "claude_chars": len(claude), "post_training_agent_path": str(post_path)}

    def _task_initialize_clients(self, loop: AgentLoop) -> dict[str, Any]:
        if self.settings.kimi_configured:
            try:
                self.kimi = KimiClient(self.settings, conversation_root=self.output_dir / ".autopilot" / "conversations", session_id=f"goal:{self.spec.name}")
                loop.record_tool_call("kimi_client", inputs={"base_url": self.settings.kimi_base_url, "model": self.settings.kimi_model}, output_summary="KIMI client initialized")
                if self.kimi.recorder:
                    loop.record_tool_call("kimi_conversation_recorder", inputs=self.kimi.recorder.paths(), output_summary="KIMI conversations will be recorded in trainable JSONL formats")
            except Exception as exc:
                loop.observe("kimi_init_failed", f"{type(exc).__name__}: {exc}")
        if self.settings.vllm_configured:
            try:
                self.vllm = VLLMClient.from_settings(self.settings)
                loop.record_tool_call("vllm_client", inputs={"base_url": self.settings.vllm_base_url, "model": self.settings.vllm_model}, output_summary="vLLM client initialized")
            except Exception as exc:
                loop.observe("vllm_init_failed", f"{type(exc).__name__}: {exc}")
        if self.use_web_search:
            self.web_search = WebSearchTool(self.settings)
        loop.set_result("Runtime clients initialized", {"kimi": self.kimi is not None, "vllm": self.vllm is not None, "web_search": self.web_search is not None})
        return {"kimi": self.kimi is not None, "vllm": self.vllm is not None, "web_search": self.web_search is not None}

    def _task_inspect_repository(self, loop: AgentLoop) -> dict[str, Any]:
        self.repo_snapshot = collect_repo_snapshot(self.settings.effective_repo_path)
        path = self.output_dir / "repo_snapshot.json"
        _write_json(path, self.repo_snapshot.to_dict())
        loop.add_artifact(path, "repo_snapshot", "Autopilot repository snapshot exposed to KIMI")
        loop.set_result("Repository inspected", {"path": str(path), "repo": self.repo_snapshot.path, "file_count": len(self.repo_snapshot.files), "branch": self.repo_snapshot.branch})
        return self.repo_snapshot.to_dict()

    def _apply_human_questions(self, loop: AgentLoop, questions: list[dict[str, Any]], *, phase: str) -> list[dict[str, Any]]:
        asked: list[dict[str, Any]] = []
        if not self.settings.ask_human_enabled:
            return asked
        for item in questions[:5]:
            question = str(item.get("question") or item.get("name") or "").strip()
            if not question:
                continue
            child = loop.run_task(
                f"ask_human:{item.get('name') or 'question'}",
                "Ask human for guidance instead of guessing",
                lambda sub, it=item, q=question: self._task_ask_human(sub, q, context=it.get("context") or it.get("reason") or {"phase": phase}, urgency=str(it.get("urgency") or it.get("priority") or "normal"), options=it.get("options") or it.get("suggested_options") or []),
                task_type="ask_human",
                raise_on_error=False,
            )
            asked.append({"question": question, "status": child.status, "result_path": str(child.result_path)})
        return asked

    def _task_model_director_plan(self, loop: AgentLoop, *, phase: str) -> dict[str, Any]:
        resources = self.resources.to_dict() if self.resources else {}
        repo = self.repo_snapshot.to_dict() if self.repo_snapshot else {}
        evaluation = self.eval_result.to_dict() if self.eval_result else {}
        diagnosis = self.diagnosis.to_dict() if self.diagnosis else {}
        if self.kimi is not None:
            try:
                plan = self.kimi.plan_autonomous_actions(
                    goal=self.spec.description,
                    phase=phase,
                    resources=resources,
                    environments=self.settings.environment_summaries(),
                    repo_snapshot=repo,
                    evaluation=evaluation,
                    diagnosis=diagnosis,
                    vllm_service_plan=self.vllm_plan.to_dict() if self.vllm_plan else {},
                    memory=self.memory_snapshot,
                    max_commands=3,
                )
            except Exception as exc:
                plan = {"phase": phase, "summary": "KIMI model-director planning failed", "error": f"{type(exc).__name__}: {exc}", "commands": [], "tasks": [], "ask_human": []}
        else:
            plan = {"phase": phase, "summary": "KIMI unavailable; deterministic loop continues", "commands": [], "tasks": [], "ask_human": []}
        plan["phase"] = phase
        self.model_director_plans.append(plan)
        path = loop.write_json_artifact("model_director_plan.json", plan, kind="model_director_plan", description="KIMI autonomous repo/training/tool plan")
        loop.add_artifact(path, "model_director_plan", f"Model-director plan for {phase}")

        asked = self._apply_human_questions(loop, [x for x in plan.get("ask_human", []) if isinstance(x, dict)], phase=phase) if isinstance(plan, dict) else []
        memory_paths: list[str] = []
        memory_notes = [note for note in (plan.get("memory_updates", []) if isinstance(plan, dict) else []) if isinstance(note, str) and note.strip()]
        for note in memory_notes:
            memory_paths.append(str(self.context.append_post_training_memory(note)))
        if memory_notes and isinstance(plan, dict) and bool(plan.get("write_to_claude_memory")):
            memory_paths.append(str(append_claude_memory(self.settings.effective_repo_path, memory_notes)))

        commands = [cmd for cmd in plan.get("commands", []) if isinstance(cmd, dict)] if isinstance(plan, dict) else []
        executed: list[dict[str, Any]] = []
        if commands and self.settings.model_control_execute_commands:
            for item in commands[:3]:
                cmd = str(item.get("command") or "").strip()
                if not cmd:
                    continue
                cwd_label = str(item.get("cwd") or "repo").lower()
                cwd = self.repo_snapshot.path if (cwd_label == "repo" and self.repo_snapshot) else Path.cwd()
                timeout = float(item.get("timeout") or 600)
                child = loop.run_task(
                    f"kimi_bash:{item.get('name') or 'command'}",
                    str(item.get("reason") or "KIMI requested bash command"),
                    lambda sub, command=cmd, c=cwd, t=timeout: self._record_command(sub, command, execute=self.external_execute, timeout=t, cwd=c, shell=True),
                    task_type="bash",
                    raise_on_error=False,
                )
                executed.append({"name": item.get("name"), "status": child.status, "result_path": str(child.result_path)})
        loop.set_result("Model-director plan recorded", {"path": str(path), "command_count": len(commands), "executed_subtasks": executed, "ask_human": asked, "memory_paths": memory_paths})
        return {"path": str(path), "plan": plan, "executed_subtasks": executed, "ask_human": asked, "memory_paths": memory_paths}

    def _task_ask_human(self, loop: AgentLoop, question: str, *, context: Any = None, urgency: str = "normal", options: list[str] | None = None) -> dict[str, Any]:
        if isinstance(context, (dict, list)):
            context_text = json.dumps(context, ensure_ascii=False, indent=2)
        else:
            context_text = str(context or "")
        q = self.ask_human_tool.ask(
            question,
            context=context_text,
            suggested_options=options or [],
            urgency=urgency,
            blocking=self.interactive_human,
        )
        loop.add_artifact(self.ask_human_tool.queue_path, "ask_human_queue", "File-backed human question queue")
        loop.add_artifact(self.ask_human_tool.markdown_path, "ask_human_markdown", "Markdown view of queued human questions")
        loop.set_result(
            "Human question queued",
            {
                "question_id": q.question_id,
                "status": q.status,
                "queue_path": str(self.ask_human_tool.queue_path),
                "markdown_path": str(self.ask_human_tool.markdown_path),
                "response": q.response,
            },
        )
        return q.to_dict()

    def _task_plan_vllm_service(self, loop: AgentLoop) -> dict[str, Any]:
        manager = VLLMServiceManager.from_settings(self.settings, self.output_dir / ".autopilot" / "vllm")
        self.vllm_plan = manager.plan()
        path = self.output_dir / "vllm_service_plan.json"
        _write_json(path, self.vllm_plan.to_dict())
        loop.add_artifact(path, "vllm_service_plan", "vLLM start/kill commands and endpoint metadata")
        loop.set_result("vLLM service commands planned", {"path": str(path), "base_url": self.vllm_plan.base_url, "enabled": self.vllm_plan.enabled})
        return self.vllm_plan.to_dict()

    def _vllm_status_snapshot(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "configured": bool(self.settings.vllm_configured),
            "base_url": self.settings.vllm_base_url,
            "model": self.settings.vllm_model,
            "reachable": False,
        }
        if not self.settings.vllm_configured:
            status["error"] = "vLLM base_url/model is not configured"
            return status
        client = self.vllm or VLLMClient.from_settings(self.settings)
        try:
            models = client.list_models()
            status.update({"reachable": True, "models": models})
        except Exception as exc:
            status.update({"reachable": False, "error": f"{type(exc).__name__}: {exc}"})
        return status

    def _task_allocate_resources(self, loop: AgentLoop, *, phase: str, stage: str) -> dict[str, Any]:
        vllm_status = self._vllm_status_snapshot() if (self.manage_vllm and (stage in {"eval", "probe", "serving"} or "eval" in phase or "probe" in phase)) else {}
        if vllm_status:
            loop.record_tool_call("vllm.status", inputs={"phase": phase, "stage": stage}, output_summary=f"reachable={vllm_status.get('reachable')}", output=vllm_status)
        if self.kimi is not None:
            try:
                data = self.kimi.plan_resource_allocation(
                    goal=self.spec.description,
                    phase=phase,
                    stage=stage,
                    resources=self.resources.to_dict() if self.resources else {},
                    environments=self.settings.environment_summaries(),
                    vllm_service_plan=self.vllm_plan.to_dict() if self.vllm_plan else {},
                    vllm_status=vllm_status,
                    evaluation=self.eval_result.to_dict() if self.eval_result else {},
                    diagnosis=self.diagnosis.to_dict() if self.diagnosis else {},
                    tools=self.registry.to_dict() if self.registry else {},
                )
                plan = plan_from_mapping(data, settings=self.settings, phase=phase, stage=stage)
                plan.source = "kimi"
            except Exception as exc:
                loop.observe("kimi_resource_allocation_failed", f"{type(exc).__name__}: {exc}")
                plan = fallback_resource_allocation(goal=self.spec.description, phase=phase, settings=self.settings, resources=self.resources, vllm_plan=self.vllm_plan, stage=stage, vllm_status=vllm_status)
        else:
            plan = fallback_resource_allocation(goal=self.spec.description, phase=phase, settings=self.settings, resources=self.resources, vllm_plan=self.vllm_plan, stage=stage, vllm_status=vllm_status)
        self.resource_allocation_plan = plan
        path = loop.write_json_artifact("resource_allocation_plan.json", plan.to_dict(), kind="resource_allocation_plan", description="KIMI/fallback resource and environment allocation")
        asked = self._apply_human_questions(loop, plan.ask_human, phase=phase)
        actions: list[dict[str, Any]] = []
        manager = VLLMServiceManager.from_settings(self.settings, self.output_dir / ".autopilot" / "vllm")
        action = (plan.vllm_action or "keep").lower()
        if self.manage_vllm and self.vllm_plan and action in {"stop", "kill"}:
            actions.append(self._record_command(loop, manager.build_kill_command(), execute=self.external_execute, timeout=30, shell=True))
        elif self.manage_vllm and self.vllm_plan and action == "start" and self.vllm_plan.start_command:
            actions.append(self._record_command(loop, self.vllm_plan.start_command, execute=self.external_execute, timeout=60, shell=True, setup_command=self.vllm_plan.setup_command, fail_on_error=True))
            if self.vllm_plan.wait_command:
                actions.append(self._record_command(loop, self.vllm_plan.wait_command, execute=self.external_execute, timeout=300, shell=True, setup_command=None, fail_on_error=True))
            if self.settings.vllm_configured:
                self.vllm = VLLMClient.from_settings(self.settings)
        elif self.manage_vllm and self.vllm_plan and action == "restart":
            actions.append(self._record_command(loop, manager.build_kill_command(), execute=self.external_execute, timeout=30, shell=True))
            if self.vllm_plan.start_command:
                actions.append(self._record_command(loop, self.vllm_plan.start_command, execute=self.external_execute, timeout=60, shell=True, setup_command=self.vllm_plan.setup_command, fail_on_error=True))
                if self.vllm_plan.wait_command:
                    actions.append(self._record_command(loop, self.vllm_plan.wait_command, execute=self.external_execute, timeout=300, shell=True, setup_command=None, fail_on_error=True))
                if self.settings.vllm_configured:
                    self.vllm = VLLMClient.from_settings(self.settings)
        for item in plan.pre_training_commands[:3]:
            cmd = str(item.get("command") or "").strip()
            if cmd:
                actions.append(self._record_command(loop, cmd, execute=self.external_execute, timeout=float(item.get("timeout") or 600), shell=True))
        loop.set_result("Resource allocation planned", {"path": str(path), "environment": plan.training_environment, "vllm_action": plan.vllm_action, "actions": actions, "ask_human": asked})
        return {"plan": plan.to_dict(), "actions": actions, "ask_human": asked}

    def _task_build_eval_set(self, loop: AgentLoop) -> dict[str, Any]:
        cases = list(self.spec.eval_cases)
        if not cases and self.use_kimi_samples and self.spec.use_kimi_generated_tests and self.kimi is not None:
            try:
                generated = self.kimi.generate_eval_samples(self.spec.description, n=self.max_generated_tests)
                for item in generated:
                    case = eval_case_from_mapping(item, source="kimi")
                    if case.prompt:
                        cases.append(case)
                loop.record_tool_call("kimi.generate_eval_samples", inputs={"goal": self.spec.description, "n": self.max_generated_tests}, output_summary=f"Generated {len(generated)} eval cases", output={"cases": generated[:3]})
            except Exception as exc:
                loop.observe("kimi_eval_generation_failed", f"{type(exc).__name__}: {exc}")
        if not cases:
            cases = default_eval_cases_for_goal(self.spec.description, n=self.max_generated_tests)
        self.spec.eval_cases = cases
        path = write_eval_cases_jsonl(cases, self.output_dir / "eval_cases.jsonl")
        loop.add_artifact(path, "eval_cases_jsonl", "Evaluation cases for the goal loop")
        loop.set_result(f"Prepared {len(cases)} eval cases", {"path": str(path), "case_count": len(cases)})
        return {"case_count": len(cases), "path": str(path)}

    def _task_discover_tools(self, loop: AgentLoop) -> dict[str, Any]:
        self.registry = discover_tools_for_goal(self.spec.description, registry=ToolRegistry.default(), web_search=self.web_search if self.use_web_search else None, max_web_results=3)
        path = self.registry.write(self.output_dir / "tool_registry.json")
        loop.add_artifact(path, "tool_registry", "Available and candidate tools discovered for this goal")
        loop.set_result("Tool registry updated", {"tool_count": len(self.registry.list()), "path": str(path)})
        return {"tool_count": len(self.registry.list()), "path": str(path)}

    def _task_discover_verifiers(self, loop: AgentLoop) -> dict[str, Any]:
        self.verifier_plan = discover_verifiers(self.spec.description, kimi_configured=self.kimi is not None, web_search=self.web_search if self.use_web_search else None, max_web_results=3)
        path = self.verifier_plan.write(self.output_dir / "verifier_plan.json")
        loop.add_artifact(path, "verifier_plan", "RL/RLVR verifier candidates and backend suggestion")
        loop.set_result("Verifier plan written", {"rl_ready": self.verifier_plan.rl_ready, "path": str(path), "recommended": self.verifier_plan.recommended})
        return self.verifier_plan.to_dict()

    def _round_search_queries(self, round_idx: int) -> list[str]:
        """Stage-diverse dataset search queries for a round."""
        base = self.spec.description
        diagnosis_queries = list(self.diagnosis.data_search_queries if self.diagnosis else [])
        preferred = [str(x).lower() for x in (self.diagnosis.suggested_training if self.diagnosis and self.diagnosis.suggested_training else self.spec.preferred_training or ["sft", "dpo", "rlvr", "pt"])]
        # Keep all major post-training routes in the search space. Rotate their
        # priority by round so later rounds do not keep rediscovering the same
        # high-score SFT instruction datasets.
        pools: dict[str, list[str]] = {
            "sft": [
                f"{base} code instruction", f"{base} python instruction output", "code sft", "python code instruction", "bug fix code instruction", "algorithm problem solution",
            ],
            "dpo": [
                f"{base} preference dpo", "code preference dpo", "programming chosen rejected preference", "code reward model preference", "code preference pair chosen rejected", "RLHF code preference dataset",
            ],
            "kto": [
                f"{base} kto feedback", "code KTO feedback", "programming binary feedback", "code positive negative feedback dataset",
            ],
            "pt": [
                f"{base} continued pretraining", "raw code corpus continued pretraining", "github code corpus", "python code corpus", "The Stack code dataset", "StarCoderData",
            ],
            "continued_pretraining": [
                f"{base} continued pretraining", "raw code corpus continued pretraining", "github code corpus", "python code corpus", "The Stack code dataset", "StarCoderData",
            ],
            "rlvr": [
                f"{base} rlvr unit tests", "code unit tests RLVR", "programming problems unit tests", "APPS programming problems tests", "HumanEval MBPP code tests", "CodeContest programming tests", "verifiable code generation dataset",
            ],
            "grpo": [
                f"{base} grpo code", "code unit tests RLVR", "verifiable code generation dataset", "programming problems tests reward",
            ],
        }
        stage_order = ["sft", "dpo", "rlvr", "pt", "kto"]
        if round_idx > 1:
            # Move non-SFT stages forward in later rounds to diversify.
            stage_order = ["dpo", "rlvr", "pt", "kto", "sft"]
        for stage in preferred:
            if stage not in stage_order:
                stage_order.insert(0, stage)
        queries: list[str] = []
        for stage in stage_order:
            for q in pools.get(stage, []):
                if q and q not in queries:
                    queries.append(q)
        for q in diagnosis_queries:
            if q and q not in queries:
                queries.append(q)
        return queries[:18]

    def _prepared_manifest(self, round_idx: int) -> dict[str, Any]:
        path = self.output_dir / f"round_{round_idx}" / "prepared" / "prepare_manifest.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _available_training_configs(self, round_idx: int) -> dict[str, str]:
        manifest = self._prepared_manifest(round_idx)
        configs = manifest.get("configs") if isinstance(manifest, dict) else None
        if isinstance(configs, dict):
            return {str(k): str(v) for k, v in configs.items()}
        config_dir = self.output_dir / f"round_{round_idx}" / "prepared" / "configs"
        out: dict[str, str] = {}
        for path in config_dir.glob("train_*.yaml"):
            stage = path.stem.replace("train_", "")
            out[stage] = str(path)
        return out

    def _select_training_stage(self, round_idx: int) -> str:
        configs = self._available_training_configs(round_idx)
        if not configs:
            return "sft"
        preferred = [str(x).lower() for x in (self.spec.preferred_training or [])]
        if round_idx == 1 and "sft" in configs:
            return "sft"
        for stage in preferred + ["sft", "dpo", "kto", "pt", "rm"]:
            normalized = "pt" if stage in {"continued_pretraining", "cpt"} else stage
            if normalized in configs:
                return normalized
        return sorted(configs)[0]

    def _pre_training_early_stop_decision(self, eval_result: EvaluationResult | None) -> dict[str, Any]:
        if eval_result is None or not eval_result.target_met:
            return {"stop": False, "reason": "target not met"}
        explicit_eval = bool(self.spec.target.eval_set)
        sources = {str(case.source or "") for case in self.spec.eval_cases}
        if explicit_eval:
            eval_source = "explicit"
        elif any(src and src not in {"fallback", "kimi", "kimi_generated", "generated", "config", "inline"} for src in sources):
            eval_source = "explicit"
        elif sources and sources <= {"config", "inline"}:
            eval_source = "explicit"
        else:
            eval_source = "smoke"
        data = eval_result.to_dict()
        data["eval_source"] = eval_source
        data["case_count"] = len(eval_result.case_results)
        decision = EvalPolicy.from_config(self.settings.raw_config if isinstance(self.settings.raw_config, dict) else {}).decide(data)
        return decision.to_dict()

    def _task_round(self, loop: AgentLoop, round_idx: int) -> dict[str, Any]:
        round_dir = self.output_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        kimi_scope = self._set_round_kimi_scope(round_idx)
        loop.write_json_artifact(
            "round_trace_start.json",
            {"round": round_idx, "kimi_scope": kimi_scope, "round_dir": str(round_dir)},
            kind="round_trace_start",
            description="Round-local KIMI trajectory and metric tracking scope",
        )

        if self.self_improve_repo:
            loop.run_task("kimi_round_director_plan", "Let KIMI decide repo/tool/training subtasks for this round", lambda sub: self._task_model_director_plan(sub, phase=f"round_{round_idx}_before_eval"), raise_on_error=False)
        eval_alloc = loop.run_task("allocate_resources_for_eval", "Let KIMI allocate resources for evaluation/probing", lambda sub: self._task_allocate_resources(sub, phase=f"round_{round_idx}_before_eval", stage="eval"), raise_on_error=False)
        if not eval_alloc.ok:
            loop.observe("eval_resource_allocation_failed", eval_alloc.error or eval_alloc.summary, importance=3)

        loop.run_task("evaluate_before_round", "Evaluate current model before this round's data/training actions", self._task_evaluate, raise_on_error=False)
        pre_eval_result = self.eval_result
        loop.run_task("diagnose_failures", "Cluster weak cases and plan targeted remediation from the before-round evaluation", self._task_diagnose, raise_on_error=False)
        early_stop_decision = self._pre_training_early_stop_decision(pre_eval_result)
        if pre_eval_result and pre_eval_result.target_met and early_stop_decision.get("stop"):
            loop.run_task(
                "record_round_metrics",
                "Record the before-round metric even though target is already met",
                lambda sub: self._task_record_round_metrics(
                    sub,
                    round_idx,
                    pre_eval=pre_eval_result,
                    post_eval=None,
                    train_stage=None,
                    training_ok=None,
                    training_result_path=None,
                ),
                raise_on_error=False,
            )
            loop.run_task("finalize_round_kimi_trajectory", "Merge KIMI controller/collector conversations for this round", lambda sub: self._task_finalize_round_kimi_trajectory(sub, round_idx), raise_on_error=False)
            loop.set_result("Round stopped because real target is already met", {"score": pre_eval_result.score, "round": round_idx, "early_stop_decision": early_stop_decision})
            return {"target_met": True, "round": round_idx, "pre_score": pre_eval_result.score, "early_stop_decision": early_stop_decision}
        if pre_eval_result and pre_eval_result.target_met and not early_stop_decision.get("stop"):
            loop.observe("early_stop_denied", early_stop_decision.get("reason", "stop policy denied pre-training early stop"), importance=3)

        collect_result = loop.run_task("collect_data", "Create or run data collection task", lambda sub: self._task_collect_data(sub, round_idx), raise_on_error=False)
        if not collect_result.ok:
            loop.run_task(
                "record_round_metrics",
                "Record metrics for a round that stopped after collect failure",
                lambda sub: self._task_record_round_metrics(
                    sub,
                    round_idx,
                    pre_eval=pre_eval_result,
                    post_eval=None,
                    train_stage=None,
                    training_ok=False,
                    training_result_path=None,
                ),
                raise_on_error=False,
            )
            loop.run_task("finalize_round_kimi_trajectory", "Merge KIMI controller/collector conversations for this round", lambda sub: self._task_finalize_round_kimi_trajectory(sub, round_idx), raise_on_error=False)
            loop.set_result("Round stopped after collect failure", {"round": round_idx, "error": collect_result.error, "collect_result": collect_result.result_path})
            return {"round": round_idx, "status": "collect_failed"}

        prepare_result = loop.run_task("prepare_training", "Create or run LLaMA-Factory preparation task", lambda sub: self._task_prepare_training(sub, round_idx), raise_on_error=False)
        if not prepare_result.ok:
            loop.run_task(
                "record_round_metrics",
                "Record metrics for a round that stopped after prepare failure",
                lambda sub: self._task_record_round_metrics(
                    sub,
                    round_idx,
                    pre_eval=pre_eval_result,
                    post_eval=None,
                    train_stage=None,
                    training_ok=False,
                    training_result_path=None,
                ),
                raise_on_error=False,
            )
            loop.run_task("finalize_round_kimi_trajectory", "Merge KIMI controller/collector conversations for this round", lambda sub: self._task_finalize_round_kimi_trajectory(sub, round_idx), raise_on_error=False)
            loop.set_result("Round stopped after prepare failure", {"round": round_idx, "error": prepare_result.error, "prepare_result": prepare_result.result_path})
            return {"round": round_idx, "status": "prepare_failed"}

        train_stage = self._select_training_stage(round_idx)
        loop.decide("selected_training_stage", f"Selected training stage {train_stage} from prepared configs", {"configs": self._available_training_configs(round_idx), "stage": train_stage})
        loop.run_task("allocate_resources_for_training", "Let KIMI choose GPUs, vLLM action, and environment before training", lambda sub: self._task_allocate_resources(sub, phase=f"round_{round_idx}_before_train", stage=train_stage), raise_on_error=False)
        train_result = loop.run_task("train_or_plan", "Create or run the training command", lambda sub: self._task_train(sub, round_idx, stage=train_stage), raise_on_error=False)
        if not train_result.ok:
            loop.observe("training_failed", train_result.error or train_result.summary, {"result_path": train_result.result_path}, importance=3)

        post_eval_result: EvaluationResult | None = None
        if train_result.ok and self.evaluate:
            loop.run_task(
                "allocate_resources_for_post_train_eval",
                "Let KIMI allocate resources for the after-training evaluation endpoint",
                lambda sub: self._task_allocate_resources(sub, phase=f"round_{round_idx}_after_train_before_eval", stage="eval"),
                raise_on_error=False,
            )
            post_task = loop.run_task("evaluate_after_training", "Evaluate current/deployed model after this round's training command", self._task_evaluate, raise_on_error=False)
            if post_task.ok:
                post_eval_result = self.eval_result
                loop.run_task("diagnose_post_training_failures", "Update failure diagnosis from the after-training evaluation", self._task_diagnose, raise_on_error=False)
        elif not train_result.ok:
            # Keep the before-round evaluation as the current state if training never completed.
            self.eval_result = pre_eval_result

        loop.run_task(
            "record_round_metrics",
            "Record before/after metric delta for this round",
            lambda sub: self._task_record_round_metrics(
                sub,
                round_idx,
                pre_eval=pre_eval_result,
                post_eval=post_eval_result,
                train_stage=train_stage,
                training_ok=train_result.ok,
                training_result_path=train_result.result_path,
            ),
            raise_on_error=False,
        )

        if self.self_improve_repo:
            loop.run_task("kimi_repo_improvement_after_training", "Let KIMI inspect outcomes and improve the repository loop", lambda sub: self._task_model_director_plan(sub, phase=f"round_{round_idx}_after_train"), raise_on_error=False)
        loop.run_task("plan_remediation", "Plan next turn from eval failures and verifier/tool state", lambda sub: self._task_remediation(sub, round_idx), raise_on_error=False)
        loop.run_task("finalize_round_kimi_trajectory", "Merge KIMI controller/collector conversations for this round", lambda sub: self._task_finalize_round_kimi_trajectory(sub, round_idx), raise_on_error=False)
        loop.set_result(
            "Round completed",
            {
                "round": round_idx,
                "target_met": bool(self.eval_result and self.eval_result.target_met),
                "train_stage": train_stage,
                "training_ok": train_result.ok,
                "pre_score": pre_eval_result.score if pre_eval_result else None,
                "post_score": post_eval_result.score if post_eval_result else None,
            },
        )
        return {"round": round_idx, "train_stage": train_stage, "training_ok": train_result.ok, "pre_score": pre_eval_result.score if pre_eval_result else None, "post_score": post_eval_result.score if post_eval_result else None}

    def _task_evaluate(self, loop: AgentLoop) -> dict[str, Any]:
        if not self.evaluate:
            loop.set_result("Evaluation disabled", {"score": None})
            return {"score": None}
        self.eval_result = evaluate_cases(
            self.spec.eval_cases,
            metric_name=self.spec.target.name,
            target=self.spec.target.target,
            vllm=self.vllm,
            kimi=self.kimi if self.spec.use_kimi_judge else None,
            goal=self.spec.description,
        )
        path = loop.write_json_artifact("evaluation_result.json", self.eval_result.to_dict(), kind="evaluation_result", description="Target metric evaluation result")
        loop.set_result("Evaluation completed", {"score": self.eval_result.score, "target_met": self.eval_result.target_met, "path": str(path)})
        return self.eval_result.to_dict()

    def _task_diagnose(self, loop: AgentLoop) -> dict[str, Any]:
        self.diagnosis = diagnose_failures(self.spec.description, self.eval_result)
        path = loop.write_json_artifact("failure_diagnosis.json", self.diagnosis.to_dict(), kind="failure_diagnosis", description="Failure clusters and remediation hints")
        loop.set_result("Failure diagnosis completed", {"failure_count": self.diagnosis.failure_count, "path": str(path)})
        return self.diagnosis.to_dict()

    def _task_collect_data(self, loop: AgentLoop, round_idx: int) -> dict[str, Any]:
        collect_dir = self.output_dir / f"round_{round_idx}" / "collection"
        extra_queries = self._round_search_queries(round_idx)
        parts = ["autopilot-collect", *self._config_arg(), "--goal", self.spec.description, "--output-dir", str(collect_dir)]
        for q in extra_queries[:18]:
            parts.extend(["--query", q])
        if self.use_web_search:
            parts.append("--use-web-search")
        if self.kimi is not None:
            parts += ["--use-llm-queries", "--use-llm-decision"]
        if self.vllm is not None:
            parts.append("--test-vllm")
        return self._record_command(loop, parts, execute=self.external_execute, timeout=3600.0, fail_on_error=True)

    def _task_prepare_training(self, loop: AgentLoop, round_idx: int) -> dict[str, Any]:
        report = self.output_dir / f"round_{round_idx}" / "collection" / "collection_report.json"
        prepared_dir = self.output_dir / f"round_{round_idx}" / "prepared"
        parts = [
            "autopilot-prepare", *self._config_arg(),
            "--report", str(report),
            "--output-dir", str(prepared_dir),
            "--actions", "accept,review",
            "--dataset-limit", "10",
            "--stage-quota", "sft:3,dpo:2,pt:2,kto:1,rlvr:2",
        ]
        return self._record_command(loop, parts, execute=self.external_execute, timeout=3600.0, fail_on_error=True)

    def _task_train(self, loop: AgentLoop, round_idx: int, *, stage: str | None = None) -> dict[str, Any]:
        configs = self._available_training_configs(round_idx)
        stage = stage or self._select_training_stage(round_idx)
        train_yaml = Path(configs.get(stage, str(self.output_dir / f"round_{round_idx}" / "prepared" / "configs" / f"train_{stage}.yaml")))
        if not train_yaml.exists():
            raise FileNotFoundError(f"No train yaml found for stage={stage}: {train_yaml}")

        # Last-mile guardrail: a KIMI resource plan may omit the environment, or
        # allocation may have failed and left an eval/vLLM plan as the most recent
        # plan.  Training must not run bare ``llamafactory-cli`` in that case.
        if self.resource_allocation_plan is None or self.resource_allocation_plan.stage != stage:
            self.resource_allocation_plan = fallback_resource_allocation(
                goal=self.spec.description,
                phase=f"round_{round_idx}_before_train",
                stage=stage,
                settings=self.settings,
                resources=self.resources,
                vllm_plan=self.vllm_plan,
                vllm_status={},
            )
            loop.observe("training_resource_plan_recreated", f"Created stage-aware resource plan for {stage} training before running train command.", self.resource_allocation_plan.to_dict(), importance=2)
        else:
            before_env = self.resource_allocation_plan.training_environment
            self.resource_allocation_plan = repair_resource_allocation_plan(
                self.resource_allocation_plan,
                settings=self.settings,
                phase=f"round_{round_idx}_before_train",
                stage=stage,
            )
            if self.resource_allocation_plan.training_environment != before_env:
                loop.observe("training_resource_plan_repaired", f"Repaired training environment for {stage}: {before_env!r} -> {self.resource_allocation_plan.training_environment!r}.", self.resource_allocation_plan.to_dict(), importance=2)

        monitor_dir = self.output_dir / f"round_{round_idx}" / "training_monitor"
        parts = ["autopilot-run", *self._config_arg(), "--train-yaml", str(train_yaml), "--timeout", "86400", "--monitor-dir", str(monitor_dir)]
        if self.resource_allocation_plan and self.resource_allocation_plan.training_environment:
            parts.extend(["--environment", self.resource_allocation_plan.training_environment])
        elif self.resource_allocation_plan and self.resource_allocation_plan.activation_command:
            parts.extend(["--env-setup", self.resource_allocation_plan.activation_command])
        else:
            env = choose_environment_for_stage(self.settings, stage)
            if env is not None:
                parts.extend(["--environment", env.name])
                loop.observe("training_environment_auto_selected", f"Auto-selected {env.name} for {stage} because no environment was present in the resource plan.", env.to_dict(), importance=2)

        if self.resource_allocation_plan:
            cuda_devices = (
                self.resource_allocation_plan.gpu_allocation.get("cuda_visible_devices")
                or self.resource_allocation_plan.gpu_allocation.get("CUDA_VISIBLE_DEVICES")
            )
            if cuda_devices:
                parts.extend(["--env", f"CUDA_VISIBLE_DEVICES={cuda_devices}"])
        return self._record_command(loop, parts, execute=self.external_execute, timeout=86400.0, fail_on_error=True)

    def _task_remediation(self, loop: AgentLoop, round_idx: int) -> dict[str, Any]:
        remediation: dict[str, Any] = {
            "round": round_idx,
            "target_met": bool(self.eval_result and self.eval_result.target_met),
            "next_training": self.diagnosis.suggested_training if self.diagnosis else ["sft"],
            "queries": self.diagnosis.data_search_queries if self.diagnosis else [self.spec.description],
            "verifiers": self.verifier_plan.recommended if self.verifier_plan else [],
            "tools_to_enable": [tool.name for tool in (self.registry.list(status="candidate") if self.registry else [])[:10]],
            "resource_summary": self.resources.to_dict() if self.resources else {},
            "runtime_environments": self.settings.environment_summaries(),
            "resource_allocation": self.resource_allocation_plan.to_dict() if self.resource_allocation_plan else {},
        }
        if self.kimi is not None and self.diagnosis and self.spec.use_kimi_generated_tests:
            try:
                remediation["kimi_generated_targeted_tests"] = self.kimi.generate_eval_samples(self.spec.description, failure_summary=self.diagnosis.to_dict(), n=min(5, self.max_generated_tests))
            except Exception as exc:
                remediation["kimi_generation_error"] = f"{type(exc).__name__}: {exc}"
        path = loop.write_json_artifact("remediation_plan.json", remediation, kind="remediation_plan", description="Next data/tool/verifier actions")
        loop.set_result("Remediation plan written", {"path": str(path), "next_training": remediation["next_training"]})
        return remediation

    def _task_update_post_training_memory(self, loop: AgentLoop, report_json: Path | None = None) -> dict[str, Any]:
        notes = [
            f"Goal '{self.spec.description}' target {self.spec.target.name}>={self.spec.target.target}; final score={self.eval_result.score if self.eval_result else None}.",
            "Use runtime.environments as selectable options; do not assume any env is active unless the resource allocation plan selected it.",
        ]
        if self.resource_allocation_plan:
            notes.append(f"Latest resource plan: environment={self.resource_allocation_plan.training_environment}, vllm_action={self.resource_allocation_plan.vllm_action}, source={self.resource_allocation_plan.source}.")
        if self.resources:
            notes.append(f"Resource snapshot: {len(self.resources.gpus)} GPUs, nvidia_smi_ok={self.resources.nvidia_smi_ok}, disk_free_gb={self.resources.disk_free_gb}.")
        if self.diagnosis and self.diagnosis.recommendations:
            notes.append("Latest remediation hint: " + self.diagnosis.recommendations[0])
        memory_path = "PostTrainingAgent.md" if self.external_execute else self.output_dir / "PostTrainingAgent.md"
        path = self.context.append_post_training_memory("\n".join(f"- {x}" for x in notes), path=memory_path)
        loop.add_artifact(path, "post_training_agent_memory", "Stable lessons appended for future interactions")
        loop.set_result("PostTrainingAgent memory updated", {"path": str(path), "note_count": len(notes)})
        return {"path": str(path), "notes": notes}
