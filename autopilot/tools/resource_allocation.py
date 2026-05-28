from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from autopilot.config import Settings
from autopilot.models import to_jsonable
from autopilot.tools.environments import RuntimeEnvironment
from autopilot.tools.resources import ComputeResources
from autopilot.tools.vllm_service import VLLMServicePlan


@dataclass
class ResourceAllocationPlan:
    """Model-directed resource plan for the next training/eval task."""

    phase: str
    stage: str = "sft"
    training_environment: str | None = None
    activation_command: str | None = None
    vllm_action: str = "keep"  # keep | start | stop | restart | none
    gpu_allocation: dict[str, Any] = field(default_factory=dict)
    pre_training_commands: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    ask_human: list[dict[str, Any]] = field(default_factory=list)
    source: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def _stage_from_goal(goal: str, default: str = "sft") -> str:
    lower = goal.lower()
    if any(x in lower for x in ["dpo", "preference", "偏好"]):
        return "dpo"
    if any(x in lower for x in ["rlvr", "grpo", "reinforcement", "强化"]):
        return "rlvr"
    if any(x in lower for x in ["pretrain", "continued pretraining", "继续预训练"]):
        return "pt"
    return default


def choose_environment_for_stage(settings: Settings, stage: str) -> RuntimeEnvironment | None:
    stage = (stage or "").lower()
    registry = settings.environment_registry()
    preferred_names = ["llamaf", "llama_factory", "llamafactory", "training", "legacy_training_env"]
    for name in preferred_names:
        env = registry.get(name)
        if env is not None:
            stages = [c.lower() for c in env.stages]
            if not stages or stage in stages or "training" in stages or "llamafactory" in stages:
                return env
    for env in registry.list():
        stages = [c.lower() for c in env.stages]
        if not stages or stage in stages or "training" in stages or "llamafactory" in stages:
            return env
    return None


TRAINING_STAGES = {"sft", "dpo", "kto", "pt", "rm", "ppo", "rlvr", "grpo", "orpo"}


def is_training_phase_or_stage(phase: str | None, stage: str | None) -> bool:
    phase_l = (phase or "").lower()
    stage_l = (stage or "").lower()
    return "train" in phase_l or stage_l in TRAINING_STAGES


def repair_resource_allocation_plan(
    plan: ResourceAllocationPlan,
    *,
    settings: Settings,
    phase: str | None = None,
    stage: str | None = None,
) -> ResourceAllocationPlan:
    """Make a KIMI/fallback plan executable without making env activation global.

    KIMI is allowed to choose resources, but LLaMA-Factory stages are not
    runnable unless an environment or activation command is selected.  If KIMI
    omits the environment, or names an alias such as ``llamafactory``, repair the
    plan using the explicit runtime environment catalog.  This still keeps envs
    as task-local decisions; it does not reintroduce a global ``env_setup``.
    """
    effective_phase = phase or plan.phase
    effective_stage = (stage or plan.stage or "sft").lower()
    notes = [plan.notes] if plan.notes else []
    repaired = False

    if plan.training_environment:
        env = settings.environment_registry().get(plan.training_environment)
        if env is not None:
            if env.name != plan.training_environment:
                notes.append(f"Resolved requested environment {plan.training_environment!r} to configured environment {env.name!r}.")
                plan.training_environment = env.name
                repaired = True
            if not plan.activation_command:
                plan.activation_command = env.activation_command()
        elif is_training_phase_or_stage(effective_phase, effective_stage):
            notes.append(f"Requested environment {plan.training_environment!r} is not configured for {effective_stage}; selecting a catalog environment instead.")
            plan.training_environment = None
            repaired = True

    if is_training_phase_or_stage(effective_phase, effective_stage) and not plan.training_environment and not plan.activation_command:
        env = choose_environment_for_stage(settings, effective_stage)
        if env is not None:
            plan.training_environment = env.name
            plan.activation_command = env.activation_command()
            notes.append(f"Auto-selected runtime environment {env.name!r} for {effective_stage} training because the planner did not choose an activatable environment.")
            repaired = True

    if repaired:
        suffix = "resource_repair"
        plan.source = f"{plan.source}+{suffix}" if suffix not in plan.source else plan.source
        plan.notes = " ".join(x for x in notes if x)
    return plan


def fallback_resource_allocation(
    *,
    goal: str,
    phase: str,
    settings: Settings,
    resources: ComputeResources | None,
    vllm_plan: VLLMServicePlan | None,
    stage: str | None = None,
    vllm_status: Mapping[str, Any] | None = None,
) -> ResourceAllocationPlan:
    """Fallback when KIMI resource planning is unavailable.

    This is not a global fixed workflow: it is a stage-aware emergency plan.
    It uses the explicit environment catalog and observed vLLM reachability so
    an otherwise healthy run can continue instead of asking the human for facts
    that are already encoded in the config.
    """
    stage = stage or _stage_from_goal(goal)
    registry = settings.environment_registry()
    candidate_env = choose_environment_for_stage(settings, stage)
    vllm_env = registry.get(settings.vllm_environment or "vllm")
    gpu_free = 0
    gpu_count = 0
    cuda_visible_devices: str | None = None
    if resources is not None:
        gpu_count = len(resources.gpus)
        gpu_free = int(sum(g.memory_free_mb or 0 for g in resources.gpus))
        visible = []
        for gpu in resources.gpus:
            try:
                visible.append(str(gpu.index))
            except Exception:
                continue
        if visible:
            cuda_visible_devices = ",".join(visible)
    notes: list[str] = ["KIMI resource planning unavailable; using stage-aware fallback from observed resources and environment catalog."]
    questions: list[dict[str, Any]] = []

    stage_l = (stage or "").lower()
    is_training = is_training_phase_or_stage(phase, stage_l)
    is_eval = stage_l in {"eval", "probe", "serving"} or "eval" in phase.lower() or "probe" in phase.lower()
    vllm_reachable = bool((vllm_status or {}).get("reachable"))

    training_environment = None
    activation_command = None
    vllm_action = "keep"
    if is_training:
        if candidate_env:
            training_environment = candidate_env.name
            activation_command = candidate_env.activation_command()
            notes.append(f"Selected {candidate_env.name} for {stage_l} because its catalog capabilities include this training stage.")
        else:
            questions.append({
                "question": f"Which runtime environment should Autopilot use for {stage} in phase {phase}?",
                "context": "No configured environment advertised this training capability.",
                "urgency": "high",
                "options": registry.names(),
            })
        if vllm_plan and vllm_plan.enabled and vllm_reachable and gpu_count and gpu_free < max(32000, gpu_count * 24000):
            vllm_action = "stop"
            notes.append("vLLM appears reachable and free GPU memory is relatively low; stop it before training.")
    elif is_eval:
        if vllm_plan and vllm_plan.enabled and not vllm_reachable:
            vllm_action = "start"
            if vllm_env:
                training_environment = vllm_env.name
                activation_command = vllm_env.activation_command()
                notes.append(f"Selected {vllm_env.name} to start/check vLLM because the endpoint is not reachable.")
            else:
                notes.append("vLLM endpoint is not reachable and deploy plan is enabled; start vLLM without a resolved environment.")
        elif vllm_reachable:
            vllm_action = "keep"
            notes.append("vLLM endpoint is reachable; keep it for eval/probe.")

    gpu_alloc = {"gpu_count": gpu_count, "gpu_memory_free_mb": gpu_free}
    if cuda_visible_devices:
        gpu_alloc["cuda_visible_devices"] = cuda_visible_devices
    return ResourceAllocationPlan(
        phase=phase,
        stage=stage,
        training_environment=training_environment,
        activation_command=activation_command,
        vllm_action=vllm_action,
        gpu_allocation=gpu_alloc,
        notes=" ".join(notes),
        ask_human=questions,
        source="deterministic_stage_aware_fallback",
    )

def _normalize_commands(raw: Any) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                commands.append({"name": "command", "command": item, "reason": "resource allocation command"})
            elif isinstance(item, Mapping):
                commands.append(dict(item))
    elif isinstance(raw, str) and raw.strip():
        commands.append({"name": "command", "command": raw.strip(), "reason": "resource allocation command"})
    return commands


def _normalize_questions(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, Mapping):
        return [dict(raw)]
    if isinstance(raw, list):
        return [dict(x) if isinstance(x, Mapping) else {"question": str(x)} for x in raw]
    if isinstance(raw, str) and raw.strip():
        return [{"question": raw.strip()}]
    return []


def plan_from_mapping(data: Mapping[str, Any], *, settings: Settings, phase: str, stage: str = "sft") -> ResourceAllocationPlan:
    training_block = data.get("training") if isinstance(data.get("training"), Mapping) else {}
    vllm_block = data.get("vllm") if isinstance(data.get("vllm"), Mapping) else {}
    gpu_block = data.get("gpu_allocation") or data.get("gpu") or training_block.get("gpu")

    env_name = (
        data.get("training_environment")
        or data.get("train_environment")
        or data.get("environment")
        or data.get("env")
        or training_block.get("environment")
        or training_block.get("training_environment")
    )
    env_name = str(env_name).strip() if env_name else None

    activation = data.get("activation_command") or data.get("env_setup") or training_block.get("activation_command")
    if not activation and env_name:
        activation = settings.environment_activation(env_name)

    cuda_devices = (
        data.get("cuda_visible_devices")
        or data.get("train_cuda_visible_devices")
        or training_block.get("cuda_visible_devices")
        or (gpu_block.get("cuda_visible_devices") if isinstance(gpu_block, Mapping) else None)
        or (gpu_block.get("CUDA_VISIBLE_DEVICES") if isinstance(gpu_block, Mapping) else None)
    )
    gpu_alloc = dict(gpu_block) if isinstance(gpu_block, Mapping) else {}
    if cuda_devices:
        gpu_alloc["cuda_visible_devices"] = str(cuda_devices)

    commands = _normalize_commands(data.get("pre_training_commands") or data.get("pre_commands") or training_block.get("commands") or data.get("commands") or [])
    raw_vllm_action = data.get("vllm_action") or vllm_block.get("action") or vllm_block.get("vllm_action") or "keep"
    vllm_action = str(raw_vllm_action).strip().lower()
    if vllm_action not in {"keep", "start", "stop", "restart", "kill", "none", "skip"}:
        vllm_action = "keep"

    plan = ResourceAllocationPlan(
        phase=phase,
        stage=str(data.get("training_stage") or data.get("stage") or training_block.get("stage") or stage),
        training_environment=env_name,
        activation_command=str(activation).strip() if activation else None,
        vllm_action=vllm_action,
        gpu_allocation=gpu_alloc,
        pre_training_commands=commands,
        notes=str(data.get("reason") or data.get("notes") or data.get("summary") or training_block.get("notes") or ""),
        ask_human=_normalize_questions(data.get("ask_human") or data.get("questions") or []),
        source=str(data.get("source") or "kimi"),
    )
    return repair_resource_allocation_plan(plan, settings=settings, phase=phase, stage=stage)
