from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from autopilot.tools.bash import BashRunner


@dataclass
class GPUInfo:
    index: int | None = None
    name: str | None = None
    memory_total_mb: int | None = None
    memory_free_mb: int | None = None
    memory_used_mb: int | None = None
    utilization_gpu_percent: int | None = None


@dataclass
class ComputeResources:
    hostname: str
    platform: str
    python: str
    cwd: str
    cpu_count: int | None
    memory_total_mb: int | None
    disk_free_gb: float | None
    cuda_visible_devices: str | None
    gpus: list[GPUInfo] = field(default_factory=list)
    nvidia_smi_available: bool = False
    raw_nvidia_smi: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def nvidia_smi_ok(self) -> bool:  # compatibility with the earlier draft
        return self.nvidia_smi_available

    @property
    def memory_total_gb(self) -> float | None:  # compatibility with the earlier draft
        return round(self.memory_total_mb / 1024, 2) if self.memory_total_mb is not None else None

    @property
    def nvidia_smi_summary(self) -> str:  # compatibility with the earlier draft
        return self.raw_nvidia_smi

    @property
    def error(self) -> str | None:
        return self.errors[0] if self.errors else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gpu_count"] = len(self.gpus)
        data["gpu_memory_total_mb"] = sum(g.memory_total_mb or 0 for g in self.gpus) if self.gpus else 0
        data["gpu_memory_free_mb"] = sum(g.memory_free_mb or 0 for g in self.gpus) if self.gpus else 0
        data["nvidia_smi_ok"] = self.nvidia_smi_ok
        data["memory_total_gb"] = self.memory_total_gb
        data["nvidia_smi_summary"] = self.nvidia_smi_summary
        data["error"] = self.error
        return data

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


def _parse_int(value: str) -> int | None:
    value = str(value).strip()
    if not value or value.upper() in {"N/A", "NA", "NONE"}:
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def parse_nvidia_smi_csv(text: str) -> list[GPUInfo]:
    """Parse nvidia-smi CSV output.

    Supports both five-column output used in v0.5.1 and the older six-column
    output with memory.used.
    """
    gpus: list[GPUInfo] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        used: int | None = None
        util_index = 4
        if len(parts) >= 6:
            used = _parse_int(parts[4])
            util_index = 5
        gpus.append(
            GPUInfo(
                index=_parse_int(parts[0]),
                name=parts[1] or None,
                memory_total_mb=_parse_int(parts[2]),
                memory_free_mb=_parse_int(parts[3]),
                memory_used_mb=used,
                utilization_gpu_percent=_parse_int(parts[util_index]),
            )
        )
    return gpus


# Compatibility name from the earlier v0.5.1 draft.
def _parse_gpu_csv(text: str) -> list[dict[str, Any]]:
    return [g.__dict__ for g in parse_nvidia_smi_csv(text)]


def _read_total_memory_mb() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(int(parts[1]) / 1024)
    return None


def inspect_compute_resources(cwd: str | Path | None = None, runner: BashRunner | None = None, timeout: float = 10.0) -> ComputeResources:
    cwd_path = Path(cwd or os.getcwd()).resolve()
    disk_free_gb: float | None = None
    try:
        disk = shutil.disk_usage(cwd_path)
        disk_free_gb = round(disk.free / 1024 / 1024 / 1024, 2)
    except Exception:
        pass
    runner = runner or BashRunner(cwd=cwd_path, timeout=timeout)
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = runner.run(query, timeout=timeout)
    gpus: list[GPUInfo] = []
    errors: list[str] = []
    raw = ""
    if result.ok:
        raw = result.stdout
        gpus = parse_nvidia_smi_csv(result.stdout)
    else:
        errors.append((result.stderr or result.stdout or "nvidia-smi failed").strip()[:1000])
        fallback = runner.run(["nvidia-smi", "-L"], timeout=timeout)
        if fallback.ok:
            raw = fallback.stdout
            errors = []
            for line in fallback.stdout.splitlines():
                if line.strip().startswith("GPU "):
                    prefix, _, rest = line.partition(":")
                    gpus.append(GPUInfo(index=_parse_int(prefix.replace("GPU", "")), name=rest.split("(")[0].strip() or None))
        else:
            raw = (fallback.stdout or "") + (fallback.stderr or "")
            if fallback.stderr.strip():
                errors.append(fallback.stderr.strip()[:1000])
    return ComputeResources(
        hostname=socket.gethostname() or platform.node(),
        platform=platform.platform(),
        python=sys.version.split()[0],
        cwd=str(cwd_path),
        cpu_count=os.cpu_count(),
        memory_total_mb=_read_total_memory_mb(),
        disk_free_gb=disk_free_gb,
        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
        gpus=gpus,
        nvidia_smi_available=bool(result.ok or gpus),
        raw_nvidia_smi=raw[-4000:],
        errors=errors,
    )


# Compatibility name from earlier draft.
def collect_compute_resources(cwd: str | Path | None = None, timeout: float = 20.0) -> ComputeResources:
    return inspect_compute_resources(cwd=cwd, timeout=timeout)


def write_compute_resources(path: str | Path, resources: ComputeResources) -> Path:
    return resources.write(path)
