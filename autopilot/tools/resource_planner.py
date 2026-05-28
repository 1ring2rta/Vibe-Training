from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from autopilot.models import to_jsonable
from autopilot.tools.environments import EnvironmentSpec, environment_by_name


@dataclass
class ResourceAllocationPlan:
    """Agent-selected runtime allocation for one loop phase/round."""

    phase: str
    summary: str
    training_stage: str = "sft"
    training_environment: str | None = None
    training_gpu_devices: str | None = None
    vllm_action: str = "keep"  # keep|start|stop|restart|skip
    vllm_gpu_devices: str | None = None
    reason: str = ""
    ask_human: list[dict[str, Any]] = field(default_factory=list)
    raw_plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return p


def _first_env_for_stage(catalog: list[EnvironmentSpec], stage: str) -> str | None:
    stage_l = (stage or "sft").lower()
    for env in catalog:
        hay = " ".join([env.name, env.description, " ".join(env.tags), " ".join(env.tools)]).lower()
        if stage_l in hay or "llamafactory" in hay or "llamaf" in hay:
            return env.name
    return catalog[0].name if catalog else None


def deterministic_resource_plan(
    *,
    phase: str,
    environments: list[EnvironmentSpec],
    resources: dict[str, Any] | None = None,
    preferred_stage: str = "sft",
    need_vllm: bool = True,
    vllm_deploy_enabled: bool = False,
) -> ResourceAllocationPlan:
    gpus = (resources or {}).get("gpus") or []
    gpu_devices = None
    if gpus:
        ids = [str(g.get("index")) for g in gpus if isinstance(g, dict) and g.get("index") is not None]
        gpu_devices = ",".join(ids) if ids else None
    env_name = _first_env_for_stage(environments, preferred_stage)
    vllm_action = "start" if need_vllm and vllm_deploy_enabled else "keep"
    return ResourceAllocationPlan(
        phase=phase,
        summary="Deterministic fallback resource allocation; KIMI did not provide a plan.",
        training_stage=preferred_stage,
        training_environment=env_name,
        training_gpu_devices=gpu_devices,
        vllm_action=vllm_action,
        vllm_gpu_devices=gpu_devices,
        reason="Use the known training environment for LLaMA-Factory stages and leave vLLM unchanged unless deploy is enabled.",
    )


def plan_from_mapping(data: dict[str, Any], *, phase: str, environments: list[EnvironmentSpec], fallback: ResourceAllocationPlan) -> ResourceAllocationPlan:
    if not isinstance(data, dict):
        return fallback
    training = data.get("training") if isinstance(data.get("training"), dict) else data
    vllm = data.get("vllm") if isinstance(data.get("vllm"), dict) else {}
    env_name = training.get("environment") or training.get("environment_name") or training.get("env") or data.get("training_environment")
    if env_name and environment_by_name(environments, str(env_name)) is None:
        # Keep the name in raw_plan but do not activate an unknown environment.
        env_name = None
    ask = data.get("ask_human") or data.get("questions") or []
    if not isinstance(ask, list):
        ask = []
    return ResourceAllocationPlan(
        phase=phase,
        summary=str(data.get("summary") or fallback.summary),
        training_stage=str(training.get("stage") or data.get("training_stage") or fallback.training_stage or "sft"),
        training_environment=str(env_name) if env_name else fallback.training_environment,
        training_gpu_devices=str(training.get("gpu_devices") or data.get("training_gpu_devices") or fallback.training_gpu_devices or "").strip() or None,
        vllm_action=str(vllm.get("action") or data.get("vllm_action") or fallback.vllm_action or "keep").lower(),
        vllm_gpu_devices=str(vllm.get("gpu_devices") or data.get("vllm_gpu_devices") or fallback.vllm_gpu_devices or "").strip() or None,
        reason=str(data.get("reason") or data.get("notes") or fallback.reason),
        ask_human=[x for x in ask if isinstance(x, dict)],
        raw_plan=data,
    )
