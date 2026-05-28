from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping

from autopilot.config import Settings
from autopilot.tools.bash import BashResult, BashRunner


@dataclass
class VLLMProcessResult:
    action: str
    command: str | None
    executed: bool
    ok: bool
    stdout: str = ""
    stderr: str = ""
    pid_file: str | None = None
    log_file: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _deep_get(data: Mapping[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def get_vllm_deploy_config(settings: Settings) -> dict[str, Any]:
    return _as_dict(_deep_get(settings.raw_config, ["vllm", "deploy"], default={}))


def _quote(value: Any) -> str:
    return shlex.quote(str(value))


def build_vllm_serve_command(settings: Settings, deploy: Mapping[str, Any] | None = None) -> str | None:
    deploy = deploy or get_vllm_deploy_config(settings)
    if deploy.get("command"):
        return str(deploy["command"])
    model_path = deploy.get("model_path") or _deep_get(settings.raw_config, ["vllm", "model_path"]) or settings.vllm_model
    if not model_path:
        return None
    host = str(deploy.get("host", "0.0.0.0"))
    port = int(deploy.get("port", 8000))
    tensor_parallel_size = deploy.get("tensor_parallel_size", deploy.get("tp", None))
    max_model_len = deploy.get("max_model_len", None)
    served_model_name = deploy.get("served_model_name") or settings.vllm_model or Path(str(model_path)).name
    api_key = deploy.get("api_key") or settings.vllm_api_key
    cuda_visible_devices = deploy.get("cuda_visible_devices")
    pieces: list[str] = []
    if cuda_visible_devices not in (None, ""):
        pieces.append(f"CUDA_VISIBLE_DEVICES={_quote(cuda_visible_devices)}")
    pieces += ["vllm", "serve", _quote(model_path), "--host", _quote(host), "--port", _quote(port)]
    if tensor_parallel_size not in (None, ""):
        pieces += ["--tensor-parallel-size", _quote(tensor_parallel_size)]
    if max_model_len not in (None, ""):
        pieces += ["--max-model-len", _quote(max_model_len)]
    if served_model_name:
        pieces += ["--served-model-name", _quote(served_model_name)]
    if api_key:
        pieces += ["--api-key", _quote(api_key)]
    return " ".join(pieces)


def launch_vllm_if_configured(settings: Settings, *, runner: BashRunner | None = None, workspace: str | Path = ".", execute: bool = True) -> VLLMProcessResult:
    deploy = get_vllm_deploy_config(settings)
    if not bool(deploy.get("enabled", False)):
        return VLLMProcessResult(action="deploy_vllm", command=None, executed=False, ok=True, notes="vllm.deploy.enabled is false or missing")
    cmd = build_vllm_serve_command(settings, deploy)
    if not cmd:
        return VLLMProcessResult(action="deploy_vllm", command=None, executed=False, ok=False, notes="No vLLM deploy command could be built; set vllm.deploy.command or vllm.deploy.model_path")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    log_file = workspace / str(deploy.get("log_file", "vllm_server.log"))
    pid_file = workspace / str(deploy.get("pid_file", "vllm_server.pid"))
    detached = f"nohup {cmd} > {_quote(log_file)} 2>&1 & echo $! > {_quote(pid_file)}"
    if not execute:
        return VLLMProcessResult(action="deploy_vllm", command=detached, executed=False, ok=True, pid_file=str(pid_file), log_file=str(log_file), notes="planned only")
    runner = runner or BashRunner(cwd=workspace, timeout=float(deploy.get("launch_timeout", 10)))
    result = runner.run(detached, shell=True, timeout=float(deploy.get("launch_timeout", 10)))
    wait_seconds = float(deploy.get("wait_seconds", 0))
    if wait_seconds > 0:
        time.sleep(min(wait_seconds, 60.0))
    return VLLMProcessResult(
        action="deploy_vllm",
        command=detached,
        executed=True,
        ok=result.ok,
        stdout=result.stdout[-2000:],
        stderr=result.stderr[-2000:],
        pid_file=str(pid_file),
        log_file=str(log_file),
    )


def kill_vllm_processes(*, runner: BashRunner | None = None, cwd: str | Path | None = None, execute: bool = True) -> VLLMProcessResult:
    # Match the common servers Autopilot uses. Keep this intentionally narrow.
    pattern = r"(vllm serve|python .*vllm.*api_server|trl vllm-serve)"
    inspect_cmd = f"pgrep -af {shlex.quote(pattern)} || true"
    kill_cmd = f"pkill -f {shlex.quote(pattern)} || true"
    command = f"{inspect_cmd}\n{kill_cmd}\n{inspect_cmd}"
    if not execute:
        return VLLMProcessResult(action="kill_vllm", command=command, executed=False, ok=True, notes="planned only")
    runner = runner or BashRunner(cwd=cwd or Path.cwd(), timeout=30)
    result: BashResult = runner.run(command, shell=True, timeout=30)
    return VLLMProcessResult(action="kill_vllm", command=command, executed=True, ok=result.ok, stdout=result.stdout[-4000:], stderr=result.stderr[-2000:])


def write_vllm_process_result(path: str | Path, result: VLLMProcessResult) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
