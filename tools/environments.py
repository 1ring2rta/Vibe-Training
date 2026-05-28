from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass
class RuntimeEnvironment:
    """Description of a pre-installed runtime environment.

    The registry is descriptive: the model/resource planner decides when to use an
    environment.  Autopilot activates one only when a command explicitly names it.
    """

    id: str
    description: str = ""
    type: str = "shell"
    root: str | None = None
    conda_env: str | None = None
    activation: str | None = None
    stages: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    notes: str = ""
    install_commands: list[str] = field(default_factory=list)
    verify_commands: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.id

    @property
    def activate(self) -> str | None:
        return self.activation_command()

    @property
    def setup_command(self) -> str | None:
        return self.activation_command()

    @property
    def path(self) -> str | None:
        return self.root

    @property
    def tags(self) -> list[str]:
        return list(self.stages)

    def activation_command(self) -> str | None:
        if self.activation and str(self.activation).strip():
            return str(self.activation).strip()
        if self.type == "conda" and self.root and self.conda_env:
            return f"source {str(self.root).rstrip('/')}/bin/activate {self.conda_env}"
        return None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["name"] = self.id
        data["activation_command"] = self.activation_command()
        return data


EnvironmentSpec = RuntimeEnvironment


class EnvironmentRegistry:
    def __init__(self, environments: Iterable[RuntimeEnvironment] | None = None) -> None:
        self._envs: dict[str, RuntimeEnvironment] = {}
        for env in environments or []:
            self.add(env)

    def add(self, env: RuntimeEnvironment) -> None:
        if env.id.strip():
            self._envs[env.id] = env

    def get(self, name: str | None) -> RuntimeEnvironment | None:
        if not name:
            return None
        key = str(name).strip().lower()
        # Exact id/name match first.
        for env in self._envs.values():
            if env.id.lower() == key:
                return env
        # Then conda env / path-derived aliases. This lets a config say
        # vllm.deploy.environment: vllm even when the cluster environment is
        # named lapha-beta but has conda_env: vllm.
        for env in self._envs.values():
            if env.conda_env and env.conda_env.lower() == key:
                return env
        # Finally, match tool/stage capabilities. Keep this last so exact names
        # remain deterministic.
        for env in self._envs.values():
            caps = [str(x).lower() for x in [*env.stages, *env.tools]]
            if key in caps:
                return env
        return None

    def list(self) -> list[RuntimeEnvironment]:
        return list(self._envs.values())

    def names(self) -> list[str]:
        return sorted(self._envs.keys())

    def to_list(self) -> list[dict[str, Any]]:
        return [env.to_dict() for env in self.list()]

    def to_dict(self) -> dict[str, Any]:
        return {name: env.to_dict() for name, env in sorted(self._envs.items())}

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return p


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _str_list(value: Any) -> list[str]:
    return [str(x) for x in _as_list(value) if str(x).strip()]


def _command_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.rstrip() for line in value.splitlines() if line.strip()]
    return _str_list(value)


def _env_from_mapping(name: str | None, value: Mapping[str, Any]) -> RuntimeEnvironment:
    env_id = str(value.get("id") or value.get("name") or name or "environment").strip()
    root = value.get("root") or value.get("path") or value.get("conda_root")
    conda_env = value.get("conda_env") or value.get("env") or value.get("environment")
    activation = value.get("activation") or value.get("activate") or value.get("setup") or value.get("command")
    return RuntimeEnvironment(
        id=env_id,
        description=str(value.get("description") or value.get("desc") or ""),
        type=str(value.get("type") or value.get("kind") or ("conda" if conda_env or root else "shell")),
        root=str(root) if root not in (None, "") else None,
        conda_env=str(conda_env) if conda_env not in (None, "") else None,
        activation=str(activation) if activation not in (None, "") else None,
        stages=_str_list(value.get("stages") or value.get("training_stages") or value.get("suitable_for") or value.get("capabilities")),
        tools=_str_list(value.get("tools") or value.get("binaries") or value.get("capabilities")),
        notes=str(value.get("notes") or ""),
        install_commands=_command_list(value.get("install_commands") or value.get("install") or value.get("install_script")),
        verify_commands=_command_list(value.get("verify_commands") or value.get("verify") or value.get("check_commands")),
        metadata={k: v for k, v in value.items() if k not in {"id", "name", "description", "desc", "type", "kind", "root", "path", "conda_root", "conda_env", "env", "environment", "activation", "activate", "setup", "command", "stages", "training_stages", "suitable_for", "capabilities", "tools", "commands", "binaries", "notes", "install_commands", "install", "install_script", "verify_commands", "verify", "check_commands"}},
    )


def parse_environments(raw: Any) -> list[RuntimeEnvironment]:
    envs: list[RuntimeEnvironment] = []
    if isinstance(raw, Mapping):
        for name, value in raw.items():
            if isinstance(value, Mapping):
                envs.append(_env_from_mapping(str(name), value))
            elif isinstance(value, str):
                envs.append(RuntimeEnvironment(id=str(name), activation=value, description=f"Shell activation for {name}"))
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping):
                envs.append(_env_from_mapping(None, item))
            elif isinstance(item, str):
                envs.append(RuntimeEnvironment(id=item, activation=item))
    dedup: dict[str, RuntimeEnvironment] = {}
    for env in envs:
        if env.id and env.id not in dedup:
            dedup[env.id] = env
    return list(dedup.values())


def default_environment_catalog() -> list[RuntimeEnvironment]:
    return [
        RuntimeEnvironment(
            id="llamaf",
            description="Pre-installed LLaMA-Factory environment for SFT/DPO/KTO/PT/RM/PPO training.",
            type="conda",
            root="../condaEnvs/anaconda3",
            conda_env="llamaf",
            activation="source ../condaEnvs/anaconda3/bin/activate llamaf",
            stages=["sft", "dpo", "kto", "pt", "rm", "ppo", "llamafactory"],
            tools=["llamafactory-cli", "python", "pip"],
            notes="Default cluster environment described by the user. It is a candidate, not an automatic global env_setup.",
        ),
        RuntimeEnvironment(
            id="vllm",
            description="vLLM serving/probing environment. Use when starting an OpenAI-compatible vllm serve process or checking the local model endpoint.",
            type="conda",
            root="../condaEnvs/anaconda3",
            conda_env="vllm",
            activation="source ../condaEnvs/anaconda3/bin/activate vllm",
            stages=["eval", "probe", "serving", "vllm", "rlvr"],
            tools=["vllm", "python", "pip", "transformers"],
            install_commands=[
                "set -euo pipefail",
                "CONDA_ROOT=${CONDA_ROOT:-../condaEnvs/anaconda3}",
                "source ${CONDA_ROOT}/bin/activate base",
                "conda create -y -n vllm python=3.11",
                "source ${CONDA_ROOT}/bin/activate vllm",
                "python -m pip install -U pip setuptools wheel",
                "python -m pip install -U vllm",
                "python -m pip install -U transformers accelerate datasets openai requests trl",
                "python - <<'PY'\nimport vllm, transformers\nprint('vllm', getattr(vllm, '__version__', 'unknown'))\nprint('transformers', transformers.__version__)\nPY",
            ],
            verify_commands=[
                "source ../condaEnvs/anaconda3/bin/activate vllm && python -c 'import vllm, transformers; print(vllm.__version__); print(transformers.__version__)'",
                "source ../condaEnvs/anaconda3/bin/activate vllm && vllm --help | head -n 20",
            ],
            notes="Install only when the environment is absent; the agent should still decide when to activate it based on resource allocation.",
        ),
    ]


def build_environment_registry(raw_config: Mapping[str, Any] | None = None) -> EnvironmentRegistry:
    raw_config = raw_config or {}
    raw = raw_config.get("environments")
    if raw is None and isinstance(raw_config.get("runtime"), Mapping):
        raw = raw_config["runtime"].get("environments")
    defaults = {env.id: env for env in default_environment_catalog()}
    for env in parse_environments(raw):
        defaults[env.id] = env
    return EnvironmentRegistry(defaults.values())


def parse_environment_catalog(raw: Any) -> list[RuntimeEnvironment]:
    envs = parse_environments(raw)
    return envs or default_environment_catalog()


def environments_to_dict(envs: list[RuntimeEnvironment]) -> dict[str, Any]:
    return {env.id: env.to_dict() for env in envs}


def write_environment_registry(path: str | Path, envs: list[RuntimeEnvironment] | EnvironmentRegistry) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = envs.to_dict() if isinstance(envs, EnvironmentRegistry) else environments_to_dict(envs)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


def find_environment(envs: list[RuntimeEnvironment], name: str | None) -> RuntimeEnvironment | None:
    return EnvironmentRegistry(envs).get(name)


def environment_by_name(envs: list[RuntimeEnvironment], name: str | None) -> RuntimeEnvironment | None:
    return find_environment(envs, name)


def describe_environments(registry: EnvironmentRegistry) -> str:
    lines: list[str] = []
    for env in registry.list():
        parts = [env.name]
        if env.path:
            parts.append(f"path={env.path}")
        if env.description:
            parts.append(env.description)
        if env.stages:
            parts.append("stages=" + ",".join(env.stages))
        if env.tools:
            parts.append("tools=" + ",".join(env.tools))
        activation = env.activation_command()
        if activation:
            parts.append(f"activate={activation}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)
