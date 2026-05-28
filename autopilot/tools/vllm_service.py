from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopilot.config import Settings
from autopilot.models import to_jsonable


@dataclass
class VLLMServicePlan:
    enabled: bool
    base_url: str | None
    start_command: str | None
    kill_command: str | None
    pid_file: str
    log_file: str
    setup_command: str | None = None
    status_command: str | None = None
    wait_command: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self.__dict__)


class VLLMServiceManager:
    """Build start/kill commands for a local OpenAI-compatible vLLM server."""

    def __init__(self, settings: Settings, runtime_dir: str | Path) -> None:
        self.settings = settings
        self.runtime_dir = Path(runtime_dir)
        deploy = settings.raw_config.get("vllm", {}).get("deploy", {}) if isinstance(settings.raw_config.get("vllm"), dict) else {}
        self.deploy_raw = deploy if isinstance(deploy, dict) else {}
        self.pid_file = self.runtime_dir / str(self.deploy_raw.get("pid_file") or 'vllm.pid')
        self.log_file = self.runtime_dir / str(self.deploy_raw.get("log_file") or 'vllm.log')

    @classmethod
    def from_settings(cls, settings: Settings, runtime_dir: str | Path) -> 'VLLMServiceManager':
        return cls(settings, runtime_dir)

    def base_url(self) -> str:
        host = 'localhost' if self.settings.vllm_host in {'0.0.0.0', '::'} else self.settings.vllm_host
        return f"http://{host}:{self.settings.vllm_port}/v1"

    def _wrap_detached(self, cmd: str) -> str:
        # Start vLLM in a new session/process group and write the launcher PID.
        # The process registry resolves the PGID from that PID, so the agent can
        # later kill the whole service tree rather than a single shell wrapper.
        runtime = shlex.quote(str(self.runtime_dir))
        log_file = shlex.quote(str(self.log_file))
        pid_file = shlex.quote(str(self.pid_file))
        base_file = shlex.quote(str(self.runtime_dir / 'vllm_base_url.txt'))
        return (
            f"mkdir -p {runtime} && "
            f"setsid bash -lc {shlex.quote(cmd)} > {log_file} 2>&1 & "
            f"echo $! > {pid_file} && "
            f"echo {shlex.quote(self.base_url())} > {base_file}"
        )

    def build_start_command(self) -> str | None:
        if not self.settings.vllm_deploy_enabled:
            return None
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        raw_command = str(self.deploy_raw.get("command") or "").strip()
        if raw_command:
            return self._wrap_detached(raw_command)
        if not self.settings.vllm_model_path:
            return None
        args = [
            'vllm', 'serve', self.settings.vllm_model_path,
            '--host', self.settings.vllm_host,
            '--port', str(self.settings.vllm_port),
        ]
        if self.settings.vllm_tensor_parallel_size:
            args += ['--tensor-parallel-size', str(self.settings.vllm_tensor_parallel_size)]
        if self.settings.vllm_max_model_len:
            args += ['--max-model-len', str(self.settings.vllm_max_model_len)]
        if self.settings.vllm_served_model_name:
            args += ['--served-model-name', self.settings.vllm_served_model_name]
        if self.settings.vllm_api_key:
            args += ['--api-key', self.settings.vllm_api_key]
        args += list(self.settings.vllm_extra_args or [])
        cmd = ' '.join(shlex.quote(str(x)) for x in args)
        if self.settings.vllm_gpu_devices:
            cmd = f"CUDA_VISIBLE_DEVICES={shlex.quote(str(self.settings.vllm_gpu_devices))} {cmd}"
        return self._wrap_detached(cmd)

    def build_status_command(self) -> str:
        url = f"{(self.base_url() if self.settings.vllm_deploy_enabled else (self.settings.vllm_base_url or '')).rstrip('/')}/models"
        key = self.settings.vllm_api_key or ""
        return (
            "python - <<'PY'\n"
            "import json, sys, urllib.request\n"
            f"url={url!r}\n"
            f"key={key!r}\n"
            "headers={}\n"
            "if key:\n"
            "    headers['Authorization']='Bearer '+key\n"
            "try:\n"
            "    req=urllib.request.Request(url, headers=headers)\n"
            "    with urllib.request.urlopen(req, timeout=5) as r:\n"
            "        body=r.read(500).decode('utf-8', errors='replace')\n"
            "    print(json.dumps({'reachable': True, 'url': url, 'sample': body[:200]}, ensure_ascii=False))\n"
            "except Exception as exc:\n"
            "    print(json.dumps({'reachable': False, 'url': url, 'error': type(exc).__name__ + ': ' + str(exc)}, ensure_ascii=False))\n"
            "    sys.exit(1)\n"
            "PY"
        )

    def build_wait_command(self, timeout_seconds: int = 240, interval_seconds: int = 5) -> str:
        url = f"{(self.base_url() if self.settings.vllm_deploy_enabled else (self.settings.vllm_base_url or '')).rstrip('/')}/models"
        key = self.settings.vllm_api_key or ""
        log_file = str(self.log_file)
        pid_file = str(self.pid_file)
        return (
            "python - <<'PY'\n"
            "import json, os, subprocess, sys, time, urllib.request\n"
            f"url={url!r}\n"
            f"key={key!r}\n"
            f"deadline=time.time()+{int(timeout_seconds)}\n"
            f"interval={int(interval_seconds)}\n"
            f"log_file={log_file!r}\n"
            f"pid_file={pid_file!r}\n"
            "headers={}\n"
            "if key:\n"
            "    headers['Authorization']='Bearer '+key\n"
            "last=None\n"
            "attempt=0\n"
            "started=time.time()\n"
            "def gpu_snapshot():\n"
            "    try:\n"
            "        p=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=5)\n"
            "        if p.returncode != 0:\n"
            "            return {'ok': False, 'error': (p.stderr or p.stdout or 'nvidia-smi failed').strip()}\n"
            "        return {'ok': True, 'lines': [x.strip() for x in p.stdout.splitlines() if x.strip()]}\n"
            "    except Exception as exc:\n"
            "        return {'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)}\n"
            "def tail_log(n=1200):\n"
            "    try:\n"
            "        if not os.path.exists(log_file):\n"
            "            return ''\n"
            "        with open(log_file, 'rb') as f:\n"
            "            f.seek(0, os.SEEK_END)\n"
            "            size=f.tell()\n"
            "            f.seek(max(0, size-n), os.SEEK_SET)\n"
            "            return f.read().decode('utf-8', errors='replace')\n"
            "    except Exception as exc:\n"
            "        return type(exc).__name__ + ': ' + str(exc)\n"
            "def pid_alive():\n"
            "    try:\n"
            "        pid=int(open(pid_file).read().strip())\n"
            "        os.kill(pid, 0)\n"
            "        return pid\n"
            "    except Exception:\n"
            "        return None\n"
            "while time.time() < deadline:\n"
            "    attempt += 1\n"
            "    try:\n"
            "        req=urllib.request.Request(url, headers=headers)\n"
            "        with urllib.request.urlopen(req, timeout=5) as r:\n"
            "            body=r.read(500).decode('utf-8', errors='replace')\n"
            "        print(json.dumps({'event': 'vllm_ready', 'reachable': True, 'url': url, 'elapsed_seconds': round(time.time()-started, 2), 'sample': body[:200]}, ensure_ascii=False), flush=True)\n"
            "        sys.exit(0)\n"
            "    except Exception as exc:\n"
            "        last=type(exc).__name__ + ': ' + str(exc)\n"
            "        print(json.dumps({'event': 'vllm_wait', 'attempt': attempt, 'elapsed_seconds': round(time.time()-started, 2), 'url': url, 'pid': pid_alive(), 'error': last, 'gpu': gpu_snapshot(), 'log_tail': tail_log(1000)[-800:]}, ensure_ascii=False), flush=True)\n"
            "        time.sleep(interval)\n"
            "print(json.dumps({'event': 'vllm_wait_timeout', 'reachable': False, 'url': url, 'elapsed_seconds': round(time.time()-started, 2), 'pid': pid_alive(), 'error': last, 'gpu': gpu_snapshot(), 'log_tail': tail_log(2000)[-1600:]}, ensure_ascii=False), flush=True)\n"
            "sys.exit(1)\n"
            "PY"
        )

    def build_kill_command(self) -> str:
        pid = shlex.quote(str(self.pid_file))
        model = self.settings.vllm_model_path or self.settings.vllm_model or 'vllm serve'
        pattern = shlex.quote(f"vllm serve {model}")
        return (
            f"if [ -f {pid} ]; then "
            f"PID=$(cat {pid} 2>/dev/null || true); "
            "PGID=$(ps -o pgid= -p $PID 2>/dev/null | tr -d ' ' || true); "
            "SELFPGID=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ' || true); "
            "if [ -n \"$PGID\" ] && [ \"$PGID\" != \"$SELFPGID\" ]; then kill -- -$PGID 2>/dev/null || true; else kill $PID 2>/dev/null || true; fi; "
            "sleep 2; "
            "if [ -n \"$PGID\" ] && ps -o pid= -g $PGID >/dev/null 2>&1; then kill -9 -- -$PGID 2>/dev/null || true; fi; "
            f"rm -f {pid}; fi; "
            f"pkill -f {pattern} 2>/dev/null || true"
        )

    def plan(self) -> VLLMServicePlan:
        notes: list[str] = []
        if not self.settings.vllm_deploy_enabled:
            notes.append('vllm.deploy.enabled is false; no start command generated.')
        if self.settings.vllm_deploy_enabled and not self.settings.vllm_model_path:
            notes.append('vllm.deploy.model_path is empty.')
        raw_base_url = self.deploy_raw.get("base_url") if isinstance(self.deploy_raw, dict) else None
        setup_command = None
        env_name = (self.deploy_raw.get("environment") or self.deploy_raw.get("env_name") or self.settings.vllm_environment or "vllm") if isinstance(self.deploy_raw, dict) else (self.settings.vllm_environment or "vllm")
        if env_name:
            env = self.settings.environment_registry().get(str(env_name))
            setup_command = env.activation_command() if env else None
            if env is None:
                notes.append(f'vllm.deploy.environment={env_name!r} is not configured.')
            elif env.name != str(env_name):
                notes.append(f'vllm.deploy.environment={env_name!r} resolved to configured environment {env.name!r}.')
        elif self.deploy_raw.get("setup_command") if isinstance(self.deploy_raw, dict) else None:
            setup_command = str(self.deploy_raw.get("setup_command"))
        return VLLMServicePlan(
            enabled=self.settings.vllm_deploy_enabled,
            base_url=str(raw_base_url) if raw_base_url else (self.base_url() if self.settings.vllm_deploy_enabled else self.settings.vllm_base_url),
            start_command=self.build_start_command(),
            kill_command=self.build_kill_command(),
            pid_file=str(self.pid_file),
            log_file=str(self.log_file),
            setup_command=setup_command,
            status_command=self.build_status_command(),
            wait_command=self.build_wait_command(),
            notes=notes,
        )
