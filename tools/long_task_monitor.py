from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from autopilot.models import to_jsonable


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _last_float(patterns: list[str], text: str) -> float | None:
    found: list[str] = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    if not found:
        return None
    value = found[-1]
    if isinstance(value, tuple):
        value = next((x for x in value if x), "")
    try:
        return float(value)
    except Exception:
        return None


def _last_int(patterns: list[str], text: str) -> int | None:
    value = _last_float(patterns, text)
    return int(value) if value is not None else None


def parse_training_metrics(text: str) -> dict[str, Any]:
    """Extract a small, best-effort metrics snapshot from LLaMA-Factory/Trainer logs.

    The parser intentionally accepts loose log formats: JSON-ish trainer dicts,
    tqdm text, and plain `loss=... learning_rate=... epoch=...` snippets.  It is
    not a source of truth; it gives the agent enough state to decide whether a
    long command is alive, improving, or failing.
    """
    tail = text[-20000:] if text else ""
    metrics: dict[str, Any] = {}
    loss = _last_float([
        rf"['\"]loss['\"]\s*[:=]\s*({_FLOAT})",
        rf"\bloss\s*[:=]\s*({_FLOAT})",
        rf"\btrain_loss\s*[:=]\s*({_FLOAT})",
    ], tail)
    eval_loss = _last_float([
        rf"['\"]eval_loss['\"]\s*[:=]\s*({_FLOAT})",
        rf"\beval_loss\s*[:=]\s*({_FLOAT})",
    ], tail)
    lr = _last_float([
        rf"['\"]learning_rate['\"]\s*[:=]\s*({_FLOAT})",
        rf"\blearning_rate\s*[:=]\s*({_FLOAT})",
        rf"\blr\s*[:=]\s*({_FLOAT})",
    ], tail)
    epoch = _last_float([
        rf"['\"]epoch['\"]\s*[:=]\s*({_FLOAT})",
        rf"\bepoch\s*[:=]\s*({_FLOAT})",
    ], tail)
    step = _last_int([
        rf"['\"]global_step['\"]\s*[:=]\s*(\d+)",
        rf"\bglobal_step\s*[:=]\s*(\d+)",
        rf"\bstep\s*[:=]\s*(\d+)",
    ], tail)
    percent = _last_float([r"(\d+(?:\.\d+)?)%\|"], tail)
    if loss is not None:
        metrics["loss"] = loss
    if eval_loss is not None:
        metrics["eval_loss"] = eval_loss
    if lr is not None:
        metrics["learning_rate"] = lr
    if epoch is not None:
        metrics["epoch"] = epoch
    if step is not None:
        metrics["global_step"] = step
    if percent is not None:
        metrics["progress_percent"] = percent
    if re.search(r"out of memory|cuda oom|CUDA out of memory", tail, flags=re.IGNORECASE):
        metrics["oom_detected"] = True
    if re.search(r"nan\b|inf\b", tail, flags=re.IGNORECASE):
        metrics["nan_or_inf_seen"] = True
    return metrics


def collect_gpu_snapshot() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=8)
    except FileNotFoundError:
        return {"ok": False, "error": "nvidia-smi not found", "gpus": []}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "gpus": []}
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout or "nvidia-smi failed").strip(), "gpus": []}
    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mb": int(float(parts[2])),
                "memory_used_mb": int(float(parts[3])),
                "memory_free_mb": int(float(parts[4])),
                "utilization_gpu_percent": int(float(parts[5])),
            })
        except Exception:
            gpus.append({"raw": line})
    return {"ok": True, "gpus": gpus}


def _safe_write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(data)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class LongTaskSupervisor:
    """Heartbeat writer for slow vLLM/training commands.

    This is deliberately lightweight and single-process.  It does not make the
    agent concurrent by itself; it turns long waits into observable state updates
    that the parent agent, `autopilot-agent status`, or a human can inspect.
    """

    root_dir: str | Path
    label: str
    interval_seconds: float = 30.0
    include_gpu: bool = True
    started_at: float = field(default_factory=time.time)
    heartbeat_count: int = 0

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.root_dir / "long_task_status.json"
        self.history_path = self.root_dir / "long_task_heartbeats.jsonl"

    def heartbeat(self, event: Mapping[str, Any]) -> dict[str, Any]:
        now = time.time()
        stdout_tail = str(event.get("stdout_tail") or "")[-12000:]
        stderr_tail = str(event.get("stderr_tail") or "")[-12000:]
        metrics = parse_training_metrics(stdout_tail + "\n" + stderr_tail)
        gpu = collect_gpu_snapshot() if self.include_gpu else {"ok": False, "gpus": []}
        snapshot: dict[str, Any] = {
            "label": self.label,
            "status": event.get("status") or "running",
            "pid": event.get("pid"),
            "command": event.get("command"),
            "elapsed_seconds": round(float(event.get("elapsed_seconds") or (now - self.started_at)), 3),
            "heartbeat_count": self.heartbeat_count + 1,
            "timestamp": now,
            "metrics": metrics,
            "gpu": gpu,
            "stdout_tail": stdout_tail[-4000:],
            "stderr_tail": stderr_tail[-4000:],
        }
        self.heartbeat_count += 1
        _safe_write_json(self.status_path, snapshot)
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(snapshot), ensure_ascii=False) + "\n")
        summary_bits = [f"elapsed={snapshot['elapsed_seconds']}s"]
        if "loss" in metrics:
            summary_bits.append(f"loss={metrics['loss']}")
        if "progress_percent" in metrics:
            summary_bits.append(f"progress={metrics['progress_percent']}%")
        if gpu.get("ok") and gpu.get("gpus"):
            gpu_bits = []
            for g in gpu["gpus"][:4]:
                if isinstance(g, dict) and "index" in g:
                    gpu_bits.append(f"gpu{g['index']} {g.get('memory_used_mb')}MB/{g.get('memory_total_mb')}MB util={g.get('utilization_gpu_percent')}%")
            if gpu_bits:
                summary_bits.append("; ".join(gpu_bits))
        print(f"[monitor:{self.label}] " + " | ".join(summary_bits), flush=True)
        return snapshot

    def finish(self, *, returncode: int | None, timed_out: bool, stdout_tail: str = "", stderr_tail: str = "") -> dict[str, Any]:
        status = "timeout" if timed_out else ("success" if returncode == 0 else "failed")
        return self.heartbeat({
            "status": status,
            "returncode": returncode,
            "timed_out": timed_out,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "elapsed_seconds": time.time() - self.started_at,
        })
