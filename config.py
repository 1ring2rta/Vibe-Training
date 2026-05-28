from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from autopilot.tools.environments import EnvironmentRegistry, RuntimeEnvironment, build_environment_registry

try:
    from dotenv import dotenv_values, load_dotenv
except Exception:  # pragma: no cover
    dotenv_values = None  # type: ignore
    load_dotenv = None  # type: ignore

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


_CONFIG_CANDIDATES = (
    "autopilot.yaml",
    "autopilot.yml",
    "config/autopilot.yaml",
    "config/autopilot.yml",
    ".autopilot/config.yaml",
    ".autopilot/config.yml",
    "configs/local.yaml",
    "configs/local.yml",
)


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from YAML plus env/.env fallback."""

    hf_token: str | None = None
    hf_endpoint: str | None = None

    kimi_api_key: str | None = None
    kimi_base_url: str = "https://api.moonshot.ai/v1"
    kimi_model: str = "kimi-k2.6"

    vllm_api_key: str | None = None
    vllm_base_url: str | None = None
    vllm_model: str | None = None
    vllm_endpoint_type: str = "openai"

    # Legacy compatibility only. v0.5.3 uses the environment registry below and
    # never auto-applies a hard-coded setup to every training command.
    training_setup_command: str | None = None
    repo_path: str | None = None
    environments: Any = field(default_factory=list)
    model_control_enabled: bool = True
    model_control_execute_commands: bool = True
    ask_human_enabled: bool = True
    ask_human_mode: str = "queue"
    claude_memory_enabled: bool = True

    vllm_deploy_enabled: bool = False
    vllm_model_path: str | None = None
    vllm_served_model_name: str | None = None
    vllm_host: str = "0.0.0.0"
    vllm_port: int = 8000
    vllm_gpu_devices: str | None = None
    vllm_tensor_parallel_size: int | None = None
    vllm_max_model_len: int | None = None
    vllm_extra_args: list[str] = field(default_factory=list)
    vllm_environment: str | None = None
    web_search_provider: str | None = None
    serper_api_key: str | None = None
    brave_api_key: str | None = None
    tavily_api_key: str | None = None
    bocha_api_key: str | None = None
    bocha_endpoint: str = "https://api.bochaai.com/v1/web-search"

    config_path: str | None = None
    raw_config: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def config_file(self) -> str | None:
        return self.config_path

    @property
    def collect_defaults(self) -> dict[str, Any]:
        return self.defaults_for("collect")

    @property
    def prepare_defaults(self) -> dict[str, Any]:
        return self.defaults_for("prepare")

    @property
    def run_defaults(self) -> dict[str, Any]:
        return self.defaults_for("run")

    @property
    def kimi_configured(self) -> bool:
        return bool(self.kimi_api_key and self.kimi_base_url and self.kimi_model)

    @property
    def vllm_configured(self) -> bool:
        return bool(self.vllm_base_url and self.vllm_model)

    @property
    def effective_repo_path(self) -> str:
        return self.repo_path or str(Path.cwd())

    def defaults_for(self, section: str) -> dict[str, Any]:
        defaults = _as_dict(_deep_get(self.raw_config, ["defaults"], default={}))
        return _as_dict(defaults.get(section) or self.raw_config.get(section))

    def environment_registry(self) -> EnvironmentRegistry:
        raw = dict(self.raw_config)
        if self.environments:
            raw["environments"] = self.environments
        return build_environment_registry(raw)

    def environment_setup(self, name: str | None) -> str | None:
        env = self.environment_registry().get(name)
        return env.activation_command() if env else None

    def environment_activation(self, name: str | None) -> str | None:
        return self.environment_setup(name)

    def environment_by_name(self, name: str | None):
        return self.environment_registry().get(name)

    def environments_as_dict(self) -> dict[str, Any]:
        return self.environment_registry().to_dict()

    def environment_summaries(self) -> list[dict[str, Any]]:
        return self.environment_registry().to_list()

    def client_registry(self, trajectory_root: str | Path | None = None):
        from autopilot.runtime.clients import LLMClientRegistry
        return LLMClientRegistry.from_settings(self, trajectory_root=trajectory_root)

    def llm_role_client(self, role: str = "director"):
        return self.client_registry().get(role)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _empty_to_none(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _first_non_empty(*values: Any, default: Any = None) -> Any:
    for value in values:
        value = _empty_to_none(value)
        if value is not None:
            return value
    return default


def _deep_get(data: Mapping[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _deep_first(data: Mapping[str, Any], paths: Iterable[Iterable[str]], default: Any = None) -> Any:
    for path in paths:
        value = _deep_get(data, path, default=None)
        value = _empty_to_none(value)
        if value is not None:
            return value
    return default


def _clean_str(value: Any) -> str | None:
    value = _empty_to_none(value)
    if value is None:
        return None
    return str(value)


def _to_bool(value: Any, default: bool = False) -> bool:
    value = _empty_to_none(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _to_int(value: Any, default: int | None = None) -> int | None:
    value = _empty_to_none(value)
    if value is None:
        return default
    return int(value)


def _to_str_list(value: Any) -> list[str]:
    value = _empty_to_none(value)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    return [str(value)]


def normalize_openai_base_url(value: Any, *, provider: str | None = None) -> str | None:
    value = _clean_str(value)
    if value is None:
        return None
    value = value.strip().rstrip("/")
    if value.endswith("/chat/completions"):
        value = value[: -len("/chat/completions")].rstrip("/")
    if value.endswith("/responses"):
        value = value[: -len("/responses")].rstrip("/")
    if provider == "kimi" and value == "https://api.moonshot.ai":
        value = "https://api.moonshot.ai/v1"
    elif provider == "kimi" and value == "https://api.moonshot.cn":
        value = "https://api.moonshot.cn/v1"
    elif not value.endswith("/v1") and "/v1/" not in value and provider != "trl":
        value = value + "/v1"
    return value


def validate_settings(settings: Settings) -> list[str]:
    warnings: list[str] = []
    if settings.hf_endpoint:
        endpoint = settings.hf_endpoint.rstrip("/")
        if endpoint and not endpoint.startswith(("http://", "https://")):
            warnings.append(f"hf.endpoint should be a full URL; current value: {endpoint}")
    if settings.kimi_base_url:
        base = settings.kimi_base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            warnings.append("KIMI base_url should end at /v1, not /v1/chat/completions.")
        if not base.endswith("/v1"):
            warnings.append(f"KIMI base_url usually should end with /v1; current value: {base}")
    if settings.vllm_base_url:
        base = settings.vllm_base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            warnings.append("vLLM base_url should end at /v1, not /v1/chat/completions.")
    if settings.training_setup_command:
        warnings.append("runtime.training_setup_command is deprecated; define environments[].activation and let the model choose when to activate it.")
    if settings.vllm_deploy_enabled and not settings.vllm_model_path:
        warnings.append("vllm.deploy.enabled is true but vllm.deploy.model_path is empty; Autopilot cannot start vLLM without a model path.")
    provider = (settings.web_search_provider or "").lower().strip()
    if provider == "bocha" and not settings.bocha_api_key:
        warnings.append("web_search.provider is bocha but bocha_api_key is empty.")
    if provider == "serper" and not settings.serper_api_key:
        warnings.append("web_search.provider is serper but serper_api_key is empty.")
    if provider == "brave" and not settings.brave_api_key:
        warnings.append("web_search.provider is brave but brave_api_key is empty.")
    if provider == "tavily" and not settings.tavily_api_key:
        warnings.append("web_search.provider is tavily but tavily_api_key is empty.")
    return warnings


def _read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for YAML config. Install with: pip install PyYAML")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/object: {path}")
    return data


def resolve_config_path(config_file: str | Path | None = None, *, auto_discover: bool = True) -> Path | None:
    if config_file:
        path = Path(config_file).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return path.resolve()
    env_path = os.getenv("AUTOPILOT_CONFIG")
    if env_path:
        path = Path(env_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"AUTOPILOT_CONFIG points to a missing file: {path}")
        return path.resolve()
    if not auto_discover:
        return None
    for candidate in _CONFIG_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path.resolve()
    user_path = Path.home() / ".config" / "llm-training-autopilot" / "config.yaml"
    if user_path.exists():
        return user_path.resolve()
    return None


find_config_file = resolve_config_path


def load_config(config_file: str | Path | None = None, *, auto_discover: bool = True) -> tuple[dict[str, Any], Path | None]:
    path = resolve_config_path(config_file, auto_discover=auto_discover)
    if path is None:
        return {}, None
    return _read_yaml(path), path


def load_yaml_config(config_file: str | Path | None = None) -> tuple[dict[str, Any], Path | None]:
    return load_config(config_file)


def load_settings(
    env_file: str | Path | None = None,
    config_file: str | Path | None = None,
    *,
    auto_discover_config: bool | None = None,
) -> Settings:
    env_values: dict[str, str] = {}
    if env_file:
        env_path = Path(env_file).expanduser()
        if dotenv_values is not None:
            parsed = dotenv_values(str(env_path))
            env_values = {str(k): str(v) for k, v in parsed.items() if k and v is not None}
        elif env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                env_values[key.strip()] = value.strip().strip('"\'')
    elif load_dotenv is not None:
        load_dotenv(override=False)

    def _env(name: str, default: Any = None) -> Any:
        return _first_non_empty(env_values.get(name), os.getenv(name), default=default)

    if config_file is None:
        env_config = _empty_to_none(env_values.get("AUTOPILOT_CONFIG"))
        if env_config is not None:
            config_file = str(env_config)
    if auto_discover_config is None:
        auto_discover_config = env_file is None
    cfg, cfg_path = load_config(config_file, auto_discover=auto_discover_config)

    hf_token = _first_non_empty(_deep_first(cfg, [["api", "huggingface", "token"], ["providers", "huggingface", "token"], ["huggingface", "token"], ["hf", "token"], ["secrets", "hf_token"]]), cfg.get("HF_TOKEN"), _env("HF_TOKEN"))
    hf_endpoint = _first_non_empty(_deep_first(cfg, [["api", "huggingface", "endpoint"], ["providers", "huggingface", "endpoint"], ["huggingface", "endpoint"], ["hf", "endpoint"]]), cfg.get("HF_ENDPOINT"), _env("HF_ENDPOINT"))

    kimi_api_key = _first_non_empty(_deep_first(cfg, [["api", "kimi", "api_key"], ["providers", "kimi", "api_key"], ["kimi", "api_key"], ["secrets", "kimi_api_key"]]), cfg.get("KIMI_API_KEY"), _env("KIMI_API_KEY"))
    kimi_base_url = _first_non_empty(_deep_first(cfg, [["api", "kimi", "base_url"], ["providers", "kimi", "base_url"], ["kimi", "base_url"]]), cfg.get("KIMI_BASE_URL"), _env("KIMI_BASE_URL"), default="https://api.moonshot.ai/v1")
    kimi_model = _first_non_empty(_deep_first(cfg, [["api", "kimi", "model"], ["providers", "kimi", "model"], ["kimi", "model"]]), cfg.get("KIMI_MODEL"), _env("KIMI_MODEL"), default="kimi-k2.6")

    vllm_api_key = _first_non_empty(_deep_first(cfg, [["api", "vllm", "api_key"], ["providers", "vllm", "api_key"], ["vllm", "api_key"], ["secrets", "vllm_api_key"]]), cfg.get("VLLM_API_KEY"), _env("VLLM_API_KEY"), _env("LOCAL_MODEL_API_KEY"), default="EMPTY")
    vllm_base_url = _first_non_empty(_deep_first(cfg, [["api", "vllm", "base_url"], ["providers", "vllm", "base_url"], ["vllm", "base_url"], ["local_model", "base_url"]]), cfg.get("VLLM_BASE_URL"), _env("VLLM_BASE_URL"), _env("LOCAL_MODEL_BASE_URL"))
    vllm_model = _first_non_empty(_deep_first(cfg, [["api", "vllm", "model"], ["providers", "vllm", "model"], ["vllm", "model"], ["local_model", "model"]]), cfg.get("VLLM_MODEL"), _env("VLLM_MODEL"), _env("LOCAL_MODEL_NAME"))
    vllm_endpoint_type = _first_non_empty(_deep_first(cfg, [["api", "vllm", "endpoint_type"], ["providers", "vllm", "endpoint_type"], ["vllm", "endpoint_type"]]), _env("VLLM_ENDPOINT_TYPE"), default="openai")

    training_setup_command = _first_non_empty(_deep_first(cfg, [["runtime", "training_setup_command"], ["runtime", "conda_activate"], ["training", "setup_command"], ["defaults", "run", "setup_command"], ["defaults", "run", "env_setup"]]), _env("AUTOPILOT_TRAINING_SETUP_COMMAND"), default=None)
    raw_environments = _deep_first(cfg, [["environments"], ["runtime", "environments"]], default=[])
    if not raw_environments and training_setup_command:
        raw_environments = [{"id": "legacy_training_env", "description": "Legacy training setup command from runtime.training_setup_command/defaults.run.setup_command.", "activation": str(training_setup_command), "stages": ["sft", "dpo", "kto", "rm", "pt"]}]

    repo_path = _first_non_empty(_deep_first(cfg, [["runtime", "repo_path"], ["repo", "path"], ["model_control", "repo_path"]]), _env("AUTOPILOT_REPO_PATH"), default=str(Path.cwd()))
    model_control_enabled = _to_bool(_first_non_empty(_deep_first(cfg, [["model_control", "enabled"], ["agent", "model_control", "enabled"]]), _env("AUTOPILOT_MODEL_CONTROL_ENABLED")), default=True)
    model_control_execute_commands = _to_bool(_first_non_empty(_deep_first(cfg, [["model_control", "execute_commands"], ["agent", "model_control", "execute_commands"]]), _env("AUTOPILOT_MODEL_CONTROL_EXECUTE_COMMANDS")), default=True)
    ask_human_cfg = _as_dict(_deep_get(cfg, ["ask_human"], default={}) or _deep_get(cfg, ["tools", "ask_human"], default={}))
    ask_human_enabled = _to_bool(_first_non_empty(ask_human_cfg.get("enabled"), _env("AUTOPILOT_ASK_HUMAN_ENABLED")), default=True)
    ask_human_mode = str(_first_non_empty(ask_human_cfg.get("mode"), _env("AUTOPILOT_ASK_HUMAN_MODE"), default="queue"))
    memory_cfg = _as_dict(_deep_get(cfg, ["memory"], default={}))
    claude_cfg = _as_dict(memory_cfg.get("claude") or _deep_get(cfg, ["claude_memory"], default={}) or {})
    claude_memory_enabled = _to_bool(_first_non_empty(claude_cfg.get("enabled"), memory_cfg.get("claude_enabled"), _env("AUTOPILOT_CLAUDE_MEMORY_ENABLED")), default=True)

    vllm_deploy = _as_dict(_deep_get(cfg, ["vllm", "deploy"], default={}) or _deep_get(cfg, ["runtime", "vllm_deploy"], default={}))
    vllm_deploy_enabled = _to_bool(_first_non_empty(vllm_deploy.get("enabled"), _env("AUTOPILOT_VLLM_DEPLOY_ENABLED")), default=False)
    vllm_model_path = _first_non_empty(vllm_deploy.get("model_path"), vllm_deploy.get("model"), _env("AUTOPILOT_VLLM_MODEL_PATH"), default=vllm_model)
    vllm_served_model_name = _first_non_empty(vllm_deploy.get("served_model_name"), vllm_deploy.get("served_name"), _env("AUTOPILOT_VLLM_SERVED_MODEL_NAME"), default=vllm_model)
    vllm_host = _first_non_empty(vllm_deploy.get("host"), _env("AUTOPILOT_VLLM_HOST"), default="0.0.0.0")
    vllm_port = _to_int(_first_non_empty(vllm_deploy.get("port"), _env("AUTOPILOT_VLLM_PORT")), default=8000) or 8000
    vllm_gpu_devices = _first_non_empty(vllm_deploy.get("gpu_devices"), vllm_deploy.get("cuda_visible_devices"), _env("AUTOPILOT_VLLM_GPU_DEVICES"))
    vllm_tensor_parallel_size = _to_int(_first_non_empty(vllm_deploy.get("tensor_parallel_size"), _env("AUTOPILOT_VLLM_TENSOR_PARALLEL_SIZE")), default=None)
    vllm_max_model_len = _to_int(_first_non_empty(vllm_deploy.get("max_model_len"), _env("AUTOPILOT_VLLM_MAX_MODEL_LEN")), default=None)
    vllm_extra_args = _to_str_list(vllm_deploy.get("extra_args"))
    vllm_environment = _first_non_empty(vllm_deploy.get("environment"), vllm_deploy.get("env_name"), _env("AUTOPILOT_VLLM_ENVIRONMENT"))
    web_search_provider = _first_non_empty(_deep_first(cfg, [["api", "web_search", "provider"], ["providers", "web_search", "provider"], ["web_search", "provider"]]), cfg.get("WEB_SEARCH_PROVIDER"), _env("WEB_SEARCH_PROVIDER"), default="duckduckgo")
    serper_api_key = _first_non_empty(_deep_first(cfg, [["api", "web_search", "serper_api_key"], ["providers", "web_search", "serper_api_key"], ["web_search", "serper_api_key"], ["secrets", "serper_api_key"]]), cfg.get("SERPER_API_KEY"), _env("SERPER_API_KEY"))
    brave_api_key = _first_non_empty(_deep_first(cfg, [["api", "web_search", "brave_api_key"], ["providers", "web_search", "brave_api_key"], ["web_search", "brave_api_key"], ["secrets", "brave_api_key"]]), cfg.get("BRAVE_API_KEY"), _env("BRAVE_API_KEY"))
    tavily_api_key = _first_non_empty(_deep_first(cfg, [["api", "web_search", "tavily_api_key"], ["providers", "web_search", "tavily_api_key"], ["web_search", "tavily_api_key"], ["secrets", "tavily_api_key"]]), cfg.get("TAVILY_API_KEY"), _env("TAVILY_API_KEY"))
    bocha_api_key = _first_non_empty(_deep_first(cfg, [["api", "web_search", "bocha_api_key"], ["providers", "web_search", "bocha_api_key"], ["web_search", "bocha_api_key"], ["secrets", "bocha_api_key"]]), cfg.get("BOCHA_API_KEY"), _env("BOCHA_API_KEY"))
    bocha_endpoint = _first_non_empty(_deep_first(cfg, [["api", "web_search", "bocha_endpoint"], ["providers", "web_search", "bocha_endpoint"], ["web_search", "bocha_endpoint"]]), cfg.get("BOCHA_ENDPOINT"), _env("BOCHA_ENDPOINT"), default="https://api.bochaai.com/v1/web-search")

    return Settings(
        hf_token=_clean_str(hf_token),
        hf_endpoint=str(hf_endpoint).rstrip("/") if _clean_str(hf_endpoint) else None,
        kimi_api_key=_clean_str(kimi_api_key),
        kimi_base_url=str(normalize_openai_base_url(kimi_base_url, provider="kimi")),
        kimi_model=str(kimi_model),
        vllm_api_key=_clean_str(vllm_api_key),
        vllm_base_url=normalize_openai_base_url(vllm_base_url, provider=str(vllm_endpoint_type or "openai")),
        vllm_model=_clean_str(vllm_model),
        vllm_endpoint_type=str(vllm_endpoint_type or "openai"),
        training_setup_command=_clean_str(training_setup_command),
        repo_path=_clean_str(repo_path),
        environments=raw_environments,
        model_control_enabled=model_control_enabled,
        model_control_execute_commands=model_control_execute_commands,
        ask_human_enabled=ask_human_enabled,
        ask_human_mode=ask_human_mode,
        claude_memory_enabled=claude_memory_enabled,
        vllm_deploy_enabled=vllm_deploy_enabled,
        vllm_model_path=_clean_str(vllm_model_path),
        vllm_served_model_name=_clean_str(vllm_served_model_name),
        vllm_host=str(vllm_host or "0.0.0.0"),
        vllm_port=int(vllm_port),
        vllm_gpu_devices=_clean_str(vllm_gpu_devices),
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_max_model_len=vllm_max_model_len,
        vllm_extra_args=vllm_extra_args,
        vllm_environment=_clean_str(vllm_environment),
        web_search_provider=_clean_str(web_search_provider),
        serper_api_key=_clean_str(serper_api_key),
        brave_api_key=_clean_str(brave_api_key),
        tavily_api_key=_clean_str(tavily_api_key),
        bocha_api_key=_clean_str(bocha_api_key),
        bocha_endpoint=str(bocha_endpoint).rstrip("/"),
        config_path=str(cfg_path) if cfg_path else None,
        raw_config=cfg,
    )


def cli_option_present(argv: Iterable[str] | None, option: str) -> bool:
    raw = list(sys.argv[1:] if argv is None else argv)
    return any(token == option or token.startswith(option + "=") for token in raw)


def explicit_cli_dests(parser: Any, argv: Iterable[str] | None) -> set[str]:
    tokens = list(sys.argv[1:] if argv is None else argv)
    explicit: set[str] = set()
    for action in getattr(parser, "_actions", []):
        for opt in getattr(action, "option_strings", []):
            if not opt.startswith("-"):
                continue
            for token in tokens:
                if token == opt or token.startswith(opt + "="):
                    explicit.add(action.dest)
    return explicit


def _coerce_to_current_type(value: Any, current: Any) -> Any:
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]
    return value


def apply_config_defaults(args: Any, parser: Any, settings: Settings, section: str, argv: Iterable[str] | None = None, aliases: dict[str, str | list[str]] | None = None) -> Any:
    defaults = settings.defaults_for(section)
    if not defaults:
        return args
    explicit = explicit_cli_dests(parser, argv)
    aliases = aliases or {}
    for dest in vars(args):
        if dest in {"config", "env_file"} or dest in explicit:
            continue
        keys: list[str] = [dest]
        alias = aliases.get(dest)
        if isinstance(alias, str):
            keys.append(alias)
        elif isinstance(alias, list):
            keys.extend(alias)
        value = None
        found = False
        for key in keys:
            if key in defaults:
                value = defaults[key]
                found = True
                break
        if not found:
            continue
        current = getattr(args, dest)
        setattr(args, dest, _coerce_to_current_type(value, current))
    return args


def apply_cli_defaults(args: Any, argv: Iterable[str] | None, defaults: Mapping[str, Any], option_map: Mapping[str, str | tuple[str, ...]]) -> None:
    if not defaults:
        return
    for attr, keys in option_map.items():
        key_list = (keys,) if isinstance(keys, str) else keys
        found = False
        value = None
        for key in key_list:
            if key in defaults:
                value = defaults[key]
                found = True
                break
        if not found:
            continue
        cli_flag = "--" + attr.replace("_", "-")
        if cli_option_present(argv, cli_flag):
            continue
        setattr(args, attr, value)


def merge_config_queries(cli_queries: list[str] | None, defaults: Mapping[str, Any], argv: Iterable[str] | None = None) -> list[str]:
    config_queries = defaults.get("queries", defaults.get("query", [])) if defaults else []
    if isinstance(config_queries, str):
        config_queries = [config_queries]
    if not isinstance(config_queries, list):
        config_queries = []
    merged: list[str] = []
    for q in list(config_queries) + list(cli_queries or []):
        q = str(q).strip()
        if q and q not in merged:
            merged.append(q)
    return merged
