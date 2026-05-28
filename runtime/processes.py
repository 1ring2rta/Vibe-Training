from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from autopilot.models import to_jsonable
from autopilot.runtime.trajectory import append_jsonl, atomic_write_json, utc_now

ACTIVE_STATUSES = {"STARTING", "RUNNING", "ACTIVE", "UNKNOWN"}
TERMINAL_STATUSES = {"EXITED", "SUCCEEDED", "FAILED", "KILLED", "MISSING"}
ENV_ROOT = "AUTOPILOT_PROCESS_REGISTRY_ROOT"


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def pid_exists(pid: int | None) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _proc_state(pid: int) -> str | None:
    try:
        parts = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace").split()
        return parts[2] if len(parts) >= 3 else None
    except Exception:
        return None


def pid_status(pid: int | None) -> str:
    if pid is None or int(pid) <= 0:
        return "MISSING"
    if not pid_exists(int(pid)):
        return "MISSING"
    if _proc_state(int(pid)) == "Z":
        return "EXITED"
    return "RUNNING"


@dataclass
class ProcessRecord:
    process_id: str
    name: str
    kind: str
    pid: int
    process_group_id: int | None = None
    command: str = ""
    cwd: str = ""
    action_id: str | None = None
    task_id: str | None = None
    environment: str | None = None
    pid_file: str | None = None
    log_file: str | None = None
    status: str = "RUNNING"
    exit_code: int | None = None
    started_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    stopped_at: str | None = None
    signal: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self.__dict__)


class ProcessRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.dir = self.root / ".autopilot" / "processes"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.dir / "process_registry.json"
        self.events_path = self.dir / "process_events.jsonl"

    @property
    def snapshot_path(self) -> Path:
        return self.registry_path

    @property
    def active_path(self) -> Path:
        return self.registry_path

    def _load(self) -> dict[str, Any]:
        return _read_json(self.registry_path, {"processes": {}})

    def _save(self, data: dict[str, Any]) -> None:
        data["updated_at"] = utc_now()
        atomic_write_json(self.registry_path, data)

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        append_jsonl(self.events_path, {"event_id": uuid.uuid4().hex, "timestamp": utc_now(), "type": event_type, "payload": payload})

    def register(self, *, name: str = "process", kind: str = "process", pid: int | None, process_group_id: int | None = None, command: str = "", cwd: str | Path | None = None, action_id: str | None = None, task_id: str | None = None, environment: str | None = None, pid_file: str | Path | None = None, log_file: str | Path | None = None, metadata: dict[str, Any] | None = None, process_id: str | None = None, **extra: Any) -> ProcessRecord:
        if pid is None:
            raise ValueError("pid is required to register a process")
        data = self._load()
        rows = data.setdefault("processes", {})
        existing_id = process_id
        if existing_id is None:
            for key, row in rows.items():
                if isinstance(row, dict) and int(row.get("pid") or -1) == int(pid) and row.get("status") in ACTIVE_STATUSES:
                    existing_id = key
                    break
        process_id = existing_id or f"proc-{uuid.uuid4().hex[:12]}"
        old = rows.get(process_id) if isinstance(rows.get(process_id), dict) else {}
        pgid = process_group_id if process_group_id not in (None, "") else old.get("process_group_id")
        if pgid in (None, ""):
            try:
                pgid = os.getpgid(int(pid))
            except Exception:
                pgid = None
        meta = dict(old.get("metadata") or {}) | dict(metadata or {}) | dict(extra or {})
        rec = ProcessRecord(
            process_id=process_id,
            name=str(name or old.get("name") or process_id),
            kind=str(kind or old.get("kind") or "process"),
            pid=int(pid),
            process_group_id=int(pgid) if pgid not in (None, "") else None,
            command=str(command or old.get("command") or ""),
            cwd=str(cwd or old.get("cwd") or self.root),
            action_id=action_id or old.get("action_id"),
            task_id=task_id or old.get("task_id"),
            environment=environment or old.get("environment"),
            pid_file=str(pid_file) if pid_file else old.get("pid_file"),
            log_file=str(log_file) if log_file else old.get("log_file"),
            status="RUNNING" if pid_status(int(pid)) == "RUNNING" else "MISSING",
            exit_code=old.get("exit_code"),
            started_at=str(old.get("started_at") or utc_now()),
            updated_at=utc_now(),
            stopped_at=old.get("stopped_at"),
            signal=old.get("signal"),
            reason=old.get("reason"),
            metadata=meta,
        )
        rows[process_id] = rec.to_dict()
        self._save(data)
        self.append_event("process_registered", rec.to_dict())
        return rec

    def register_from_pid_file(self, *, pid_file: str | Path, name: str, kind: str, command: str = "", cwd: str | Path | None = None, action_id: str | None = None, environment: str | None = None, log_file: str | Path | None = None, metadata: dict[str, Any] | None = None, **extra: Any) -> ProcessRecord | None:
        path = Path(pid_file)
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
        except Exception:
            self.append_event("process_pid_file_missing", {"pid_file": str(path), "name": name, "kind": kind})
            return None
        return self.register(name=name, kind=kind, pid=pid, command=command, cwd=cwd, action_id=action_id, environment=environment, pid_file=path, log_file=log_file, metadata=metadata, **extra)

    def start_background(self, command: str, *, cwd: str | Path | None = None, env: Mapping[str, str] | None = None, setup_command: str | None = None, label: str = "background", kind: str = "service", action_id: str | None = None, environment: str | None = None, log_file: str | Path | None = None, pid_file: str | Path | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        cwd_path = Path(cwd or self.root).resolve()
        log_path = Path(log_file) if log_file else self.dir / f"{label}.log"
        pid_path = Path(pid_file) if pid_file else self.dir / f"{label}.pid"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        full_env = os.environ.copy()
        for k, v in dict(env or {}).items():
            full_env[str(k)] = str(v)
        run_command = f"set -e\n{setup_command}\n{command}" if setup_command else command
        log_f = log_path.open("ab")
        proc = subprocess.Popen(run_command, cwd=str(cwd_path), env=full_env, shell=True, executable="/bin/bash", stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
        pid_path.write_text(str(proc.pid) + "\n", encoding="utf-8")
        try:
            pgid = os.getpgid(proc.pid)
        except Exception:
            pgid = proc.pid
        rec = self.register(name=label, kind=kind, pid=proc.pid, process_group_id=pgid, command=command, cwd=cwd_path, action_id=action_id, environment=environment, pid_file=pid_path, log_file=log_path, metadata=dict(metadata or {}) | {"env": dict(env or {}), "setup_command": setup_command})
        return {"ok": True, "process": rec.to_dict(), "pid": proc.pid, "process_group_id": pgid, "pid_file": str(pid_path), "log_file": str(log_path), "command": command}

    def start_detached(self, command: str, *, name: str = "background", cwd: str | Path | None = None, env: Mapping[str, str] | None = None, environment_setup: str | None = None, setup_command: str | None = None, category: str | None = None, kind: str | None = None, action_id: str | None = None, environment: str | None = None, log_file: str | Path | None = None, pid_file: str | Path | None = None, metadata: dict[str, Any] | None = None, **extra: Any) -> ProcessRecord:
        result = self.start_background(
            command,
            cwd=cwd,
            env=env,
            setup_command=environment_setup if environment_setup is not None else setup_command,
            label=name,
            kind=kind or category or "service",
            action_id=action_id,
            environment=environment,
            log_file=log_file,
            pid_file=pid_file,
            metadata=dict(metadata or {}) | dict(extra or {}),
        )
        data = dict(result.get("process") or {})
        allowed = {f.name for f in fields(ProcessRecord)}
        return ProcessRecord(**{k: v for k, v in data.items() if k in allowed})

    def _update(self, process_id: str, **updates: Any) -> dict[str, Any] | None:
        data = self._load()
        row = (data.get("processes") or {}).get(process_id)
        if not isinstance(row, dict):
            return None
        row.update(updates)
        row["updated_at"] = utc_now()
        self._save(data)
        self.append_event("process_updated", {"process_id": process_id, "updates": updates})
        return row

    def mark_finished(self, process_id: str | None, *, status: str, exit_code: int | None = None, reason: str | None = None) -> None:
        if process_id:
            self._update(process_id, status=status, exit_code=exit_code, stopped_at=utc_now(), reason=reason)

    def refresh(self) -> list[dict[str, Any]]:
        data = self._load()
        rows = data.setdefault("processes", {})
        changed = False
        for _, row in list(rows.items()):
            if not isinstance(row, dict) or row.get("status") in TERMINAL_STATUSES:
                continue
            current = pid_status(int(row.get("pid") or -1))
            if current != "RUNNING":
                row["status"] = current
                row["stopped_at"] = row.get("stopped_at") or utc_now()
                row["updated_at"] = utc_now()
                changed = True
        if changed:
            self._save(data)
            self.append_event("process_registry_refreshed", {"changed": True})
        return list((data.get("processes") or {}).values())

    def list(self, *, active_only: bool = False, include_exited: bool = True, include_dead: bool | None = None, include_system_scan: bool = False, patterns: list[str] | None = None, **_: Any) -> list[dict[str, Any]]:
        rows = self.refresh()
        if include_dead is not None:
            include_exited = bool(include_dead)
        if active_only:
            rows = [r for r in rows if isinstance(r, dict) and r.get("status") in ACTIVE_STATUSES]
        elif not include_exited:
            rows = [r for r in rows if isinstance(r, dict) and r.get("status") not in TERMINAL_STATUSES]
        if include_system_scan:
            rows = list(rows) + self.scan_system(patterns=patterns)
        return sorted(rows, key=lambda r: str(r.get("updated_at") or r.get("source") or ""), reverse=True)

    def scan_system(self, *, patterns: list[str] | None = None, max_rows: int = 200) -> list[dict[str, Any]]:
        """Best-effort host process scan for visibility only.

        These rows are not automatically killable unless the agent explicitly passes
        allow_untracked=true/allow_unregistered=true to kill by pid.
        """
        pats = [str(x) for x in (patterns or ["vllm", "llamafactory", "autopilot-run", "python"])]
        try:
            out = subprocess.check_output(["ps", "-eo", "pid=,ppid=,pgid=,stat=,comm=,args="], text=True, stderr=subprocess.DEVNULL)
        except Exception:
            return []
        rows: list[dict[str, Any]] = []
        for line in out.splitlines():
            parts = line.strip().split(None, 5)
            if len(parts) < 6:
                continue
            pid, ppid, pgid, stat, comm, args = parts
            hay = f"{comm} {args}"
            if pats and not any(pat and pat in hay for pat in pats):
                continue
            try:
                ipid = int(pid)
            except Exception:
                continue
            if ipid in {os.getpid(), os.getppid()}:
                continue
            rows.append({
                "process_id": None,
                "name": comm,
                "kind": "system_scan",
                "pid": ipid,
                "parent_pid": int(ppid) if str(ppid).isdigit() else None,
                "process_group_id": int(pgid) if str(pgid).isdigit() else None,
                "status": "RUNNING" if "Z" not in stat else "EXITED",
                "command": args,
                "source": "system_scan",
                "updated_at": utc_now(),
                "kill_requires_allow_untracked": True,
            })
            if len(rows) >= max_rows:
                break
        return rows

    def select(self, *, process_id: str | None = None, pid: int | None = None, name: str | None = None, label: str | None = None, action_id: str | None = None, kind: str | None = None, pattern: str | None = None) -> list[dict[str, Any]]:
        needle = pattern or name or label
        out = []
        for row in self.list(active_only=False):
            if not isinstance(row, dict):
                continue
            if process_id and row.get("process_id") != process_id:
                continue
            if pid is not None and int(row.get("pid") or -1) != int(pid):
                continue
            if action_id and row.get("action_id") != action_id:
                continue
            if kind and row.get("kind") != kind:
                continue
            haystack = str(row.get("name") or "") + " " + str(row.get("command") or "") + " " + str(row.get("kind") or "")
            if needle and needle not in haystack:
                continue
            out.append(row)
        return out

    def kill(self, identifier: Any = None, *, process_id: str | None = None, pid: int | None = None, name: str | None = None, label: str | None = None, action_id: str | None = None, kind: str | None = None, pattern: str | None = None, sig: str | int | None = None, signal_name: str | None = None, process_group: bool | None = None, kill_process_group: bool | None = None, force_after_seconds: float | None = 5.0, wait_seconds: float | None = None, allow_unregistered: bool | None = None, allow_untracked: bool | None = None, reason: str | None = None, **_: Any) -> dict[str, Any]:
        if identifier is not None and process_id is None and pid is None and name is None:
            try:
                pid = int(identifier)
            except Exception:
                process_id = str(identifier)
        rows = self.select(process_id=process_id, pid=pid, name=name, label=label, action_id=action_id, kind=kind, pattern=pattern)
        allow_untracked_effective = bool(allow_untracked if allow_untracked is not None else allow_unregistered)
        if not rows and pid is not None and allow_untracked_effective:
            rec = self.register(name=f"untracked:{pid}", kind="untracked", pid=int(pid), metadata={"allow_untracked": True})
            rows = [rec.to_dict()]
        if not rows:
            return {"ok": False, "error": "no matching tracked process", "filters": {"process_id": process_id, "pid": pid, "name": name, "label": label, "action_id": action_id, "kind": kind, "pattern": pattern}}
        sig_value = sig or signal_name or "TERM"
        signum = sig_value if isinstance(sig_value, int) else getattr(signal, "SIG" + str(sig_value).upper().replace("SIG", ""), signal.SIGTERM)
        kill_group = bool(kill_process_group if kill_process_group is not None else (True if process_group is None else process_group))
        wait = float(wait_seconds if wait_seconds is not None else (force_after_seconds if force_after_seconds is not None else 0.0))
        killed: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for row in rows:
            target_pid = int(row.get("pid") or -1)
            pgid = row.get("process_group_id")
            if target_pid in {os.getpid(), os.getppid()}:
                errors.append({"pid": target_pid, "error": "refusing to kill current Python process or parent"})
                continue
            target = target_pid
            if kill_group and pgid:
                try:
                    if int(pgid) == os.getpgrp():
                        errors.append({"pid": target_pid, "pgid": pgid, "error": "refusing to kill current process group"})
                        continue
                    target = -int(pgid)
                except Exception:
                    target = target_pid
            try:
                os.kill(target, int(signum))
                deadline = time.time() + max(wait, 0.0)
                while time.time() < deadline and pid_status(target_pid) == "RUNNING":
                    time.sleep(0.1)
                if pid_status(target_pid) == "RUNNING" and int(signum) != int(signal.SIGKILL):
                    try:
                        os.kill(target, signal.SIGKILL)
                    except Exception:
                        os.kill(target_pid, signal.SIGKILL)
                self._update(str(row.get("process_id")), status="KILLED", stopped_at=utc_now(), signal=str(sig_value), reason=reason or "agent requested kill")
                killed.append({"process_id": row.get("process_id"), "pid": target_pid, "process_group_id": pgid, "signal": str(sig_value), "kill_process_group": kill_group})
            except Exception as exc:
                errors.append({"process_id": row.get("process_id"), "pid": target_pid, "error": f"{type(exc).__name__}: {exc}"})
        result = {"ok": bool(killed) and not errors, "killed": killed, "errors": errors, "requested": len(rows)}
        self.append_event("process_kill", result | {"reason": reason})
        return result

    def kill_matching(self, *, pattern: str, signal_name: str = "TERM", force_after_seconds: float = 5.0, max_matches: int = 20) -> dict[str, Any]:
        rows = self.select(pattern=pattern)[:max_matches]
        if not rows:
            return {"ok": False, "error": "no tracked process matches pattern", "pattern": pattern}
        results = [self.kill(process_id=row.get("process_id"), signal_name=signal_name, force_after_seconds=force_after_seconds, reason=f"pattern kill: {pattern}") for row in rows]
        return {"ok": all(r.get("ok") for r in results), "results": results}

    def to_dict(self) -> dict[str, Any]:
        return {"registry_path": str(self.registry_path), "events_path": str(self.events_path), "processes": self.list(active_only=False)}

# Backwards-compatible name used by early v0.7.1 refactor code.
ManagedProcessStore = ProcessRegistry
