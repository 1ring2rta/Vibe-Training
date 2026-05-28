from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Mapping

from autopilot.config import normalize_openai_base_url
from autopilot.llm.openai_compatible import ChatCompletionResult, OpenAICompatibleChatClient
from autopilot.runtime.trajectory import FrontierTrajectoryRecorder


def _expand_env_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return os.path.expandvars(text)


def _looks_local_endpoint(base_url: str) -> bool:
    value = (base_url or "").lower()
    return any(x in value for x in ["localhost", "127.0.0.1", "0.0.0.0", "::1"])


def _bool_config(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


@dataclass
class LLMClientSpec:
    name: str
    type: str = "openai_compatible"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    timeout: float = 600.0
    params: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    record_trajectory: bool = True

    def build(self, *, trajectory_recorder: FrontierTrajectoryRecorder | None = None) -> OpenAICompatibleChatClient:
        if self.type != "openai_compatible":
            raise ValueError(f"unsupported client type: {self.type}")
        return OpenAICompatibleChatClient(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            timeout=self.timeout,
            provider_name=self.name,
            client_name=self.name,
            default_params=self.params,
            trajectory_recorder=trajectory_recorder if self.record_trajectory else None,
            auto_trajectory=self.record_trajectory,
        )


class LLMClientRegistry:
    def __init__(self, specs: Mapping[str, LLMClientSpec], roles: Mapping[str, str] | None = None, *, trajectory_recorder: FrontierTrajectoryRecorder | None = None, aliases: Mapping[str, str] | None = None) -> None:
        self.specs = dict(specs)
        self.roles = dict(roles or {})
        self.aliases = dict(aliases or {})
        self.trajectory_recorder = trajectory_recorder
        self._clients: dict[str, OpenAICompatibleChatClient] = {}

    @classmethod
    def from_settings(cls, settings: Any, *, trajectory_root: str | Path | None = None) -> "LLMClientRegistry":
        recorder = FrontierTrajectoryRecorder.from_settings(settings, root=trajectory_root)
        raw = settings.raw_config if hasattr(settings, "raw_config") and isinstance(settings.raw_config, dict) else {}
        raw_clients = raw.get("clients") if isinstance(raw.get("clients"), dict) else {}
        specs: dict[str, LLMClientSpec] = {}
        aliases: dict[str, str] = {}
        for name, cfg in raw_clients.items():
            if not isinstance(cfg, dict):
                continue
            ctype = str(cfg.get("type") or "openai_compatible")
            base_url = normalize_openai_base_url(_expand_env_value(cfg.get("base_url") or cfg.get("url")), provider=str(name)) or ""
            record_default = not _looks_local_endpoint(base_url)
            specs[str(name)] = LLMClientSpec(
                name=str(name),
                type=ctype,
                api_key=_expand_env_value(cfg.get("api_key") or ""),
                base_url=base_url,
                model=_expand_env_value(cfg.get("model") or ""),
                timeout=float(cfg.get("timeout") or 600.0),
                params=dict(cfg.get("params") or {}),
                extra_body=dict(cfg.get("extra_body") or {}),
                capabilities=dict(cfg.get("capabilities") or {}),
                record_trajectory=_bool_config(cfg.get("record_trajectory", cfg.get("trajectory")), record_default),
            )
        # Legacy provider-specific config becomes the generic remote teacher when
        # the new clients.teacher section is absent.  The alias keeps old YAML
        # such as roles.director: kimi working without making the control plane
        # provider-specific.
        if getattr(settings, "kimi_api_key", None) and "teacher" not in specs and "kimi" not in specs:
            kimi_raw = raw.get("kimi") if isinstance(raw.get("kimi"), dict) else {}
            params = dict(kimi_raw.get("params") or {})
            # K2.6/K2.5 current endpoint expects temperature=1 for many calls;
            # let explicit YAML override only if the service supports it.
            params.setdefault("temperature", kimi_raw.get("temperature", 1.0))
            if "top_p" in kimi_raw:
                params.setdefault("top_p", kimi_raw.get("top_p"))
            if "max_completion_tokens" in kimi_raw:
                params.setdefault("max_completion_tokens", kimi_raw.get("max_completion_tokens"))
            specs["teacher"] = LLMClientSpec(
                name="teacher",
                api_key=settings.kimi_api_key or "",
                base_url=settings.kimi_base_url,
                model=settings.kimi_model,
                timeout=float(kimi_raw.get("timeout") or 600.0),
                params={k: v for k, v in params.items() if v is not None},
                extra_body=dict(kimi_raw.get("extra_body") or ({"thinking": {"type": "enabled"}} if str(settings.kimi_model).lower().startswith(("kimi-k2.6", "kimi-k2.5")) else {})),
                capabilities={"reasoning_content": True, "native_tool_calls": True, "remote_teacher": True},
                record_trajectory=True,
            )
            aliases["kimi"] = "teacher"
        if getattr(settings, "vllm_base_url", None) and getattr(settings, "vllm_model", None) and "local_vllm" not in specs:
            specs["local_vllm"] = LLMClientSpec(
                name="local_vllm",
                api_key=settings.vllm_api_key or "",
                base_url=settings.vllm_base_url or "",
                model=settings.vllm_model or "",
                timeout=120.0,
                params={"temperature": 0.0, "max_tokens": 1024},
                capabilities={"reasoning_content": False, "native_tool_calls": False},
                record_trajectory=False,
            )
        raw_roles = raw.get("roles") if isinstance(raw.get("roles"), dict) else {}
        roles: dict[str, str] = {}
        for role, value in raw_roles.items():
            if isinstance(value, str):
                roles[str(role)] = value
            elif isinstance(value, dict):
                roles[str(role)] = str(value.get("client") or value.get("name") or "")
        roles.setdefault("director", "teacher" if "teacher" in specs else ("kimi" if "kimi" in specs else next(iter(specs), "")))
        roles.setdefault("judge", roles.get("director", ""))
        roles.setdefault("data_adapter_writer", roles.get("director", ""))
        roles.setdefault("eval_planner", roles.get("director", ""))
        if "local_vllm" in specs:
            roles.setdefault("local_probe", "local_vllm")
        return cls(specs, roles, trajectory_recorder=recorder, aliases=aliases)

    def names(self) -> list[str]:
        return sorted(self.specs)

    def resolve_name(self, name: str) -> str:
        return self.aliases.get(name, name)

    def role_client_name(self, role: str) -> str:
        requested = self.roles.get(role) or role
        name = self.resolve_name(requested)
        if name not in self.specs:
            if role in self.specs:
                return role
            raise KeyError(f"No LLM client configured for role={role!r}; available={self.names()}, roles={self.roles}, aliases={self.aliases}")
        return name

    def get(self, name_or_role: str = "director") -> OpenAICompatibleChatClient:
        name = self.resolve_name(name_or_role) if name_or_role in self.aliases else (self.role_client_name(name_or_role) if name_or_role not in self.specs else name_or_role)
        if name not in self._clients:
            self._clients[name] = self.specs[name].build(trajectory_recorder=self.trajectory_recorder)
        return self._clients[name]

    def spec_for_role(self, role: str) -> LLMClientSpec:
        return self.specs[self.role_client_name(role)]

    def call_role(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        purpose: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        params: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ChatCompletionResult:
        spec = self.spec_for_role(role)
        client = self.get(spec.name)
        merged_params = dict(spec.params)
        merged_params.update(dict(params or {}))
        merged_extra = dict(spec.extra_body)
        merged_extra.update(dict(extra_body or {}))
        return client.chat_result(
            messages=messages,
            temperature=merged_params.pop("temperature", None),
            top_p=merged_params.pop("top_p", None),
            max_tokens=merged_params.pop("max_tokens", None),
            max_completion_tokens=merged_params.pop("max_completion_tokens", None),
            tools=tools,
            tool_choice=tool_choice if tools else None,
            extra_body={**merged_params, **merged_extra} if (merged_params or merged_extra) else None,
            purpose=purpose,
            metadata={"role": role, **dict(metadata or {})},
        )

    def to_dict(self) -> dict[str, Any]:
        return {"clients": {name: {"type": spec.type, "base_url": spec.base_url, "model": spec.model, "timeout": spec.timeout, "params": spec.params, "capabilities": spec.capabilities, "record_trajectory": spec.record_trajectory} for name, spec in self.specs.items()}, "roles": self.roles, "aliases": self.aliases, "trajectory": self.trajectory_recorder.paths() if self.trajectory_recorder else None}
