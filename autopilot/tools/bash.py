from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from autopilot.runtime.processes import ProcessRegistry


@dataclass
class BashResult:
    command: str
    cwd: str
    returncode: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    env_overrides: dict[str, str] = field(default_factory=dict)
    setup_command: str | None = None
    pid: int | None = None
    pgid: int | None = None
    process_group_id: int | None = None
    process_id: str | None = None

    @property
    def env_setup(self) -> str | None:
        return self.setup_command

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def to_dict(self) -> dict[str, object]:
        return {
            "command": self.command,
            "cwd": self.cwd,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
            "env_overrides": self.env_overrides,
            "setup_command": self.setup_command,
            "pid": self.pid,
            "pgid": self.pgid,
            "process_group_id": self.process_group_id,
            "process_id": self.process_id,
        }


class BashRunner:
    def __init__(self, cwd: str | Path | None = None, timeout: float = 600.0, setup_command: str | None = None, env_setup: str | None = None) -> None:
        self.cwd = str(Path(cwd or os.getcwd()).resolve())
        self.timeout = timeout
        effective_setup = setup_command if setup_command is not None else env_setup
        self.setup_command = effective_setup.strip() if isinstance(effective_setup, str) and effective_setup.strip() else None

    @staticmethod
    def quote_command(command: Sequence[str] | str) -> str:
        return command if isinstance(command, str) else " ".join(shlex.quote(str(x)) for x in command)

    def run(
        self,
        command: Sequence[str] | str,
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        shell: bool | None = None,
        setup_command: str | None = None,
        stream_output: bool = False,
        stream_prefix: str = "",
        heartbeat_interval: float | None = None,
        heartbeat_callback: Callable[[dict[str, Any]], Any] | None = None,
        process_registry: ProcessRegistry | None = None,
        process_label: str | None = None,
        process_kind: str = "command",
        action_id: str | None = None,
        environment_name: str | None = None,
        process_metadata: dict[str, Any] | None = None,
        # Backwards-compatible callback used by older tests/callers.
        process_started_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> BashResult:
        actual_cwd = str(Path(cwd or self.cwd).resolve())
        command_display = self.quote_command(command)
        setup = setup_command if setup_command is not None else self.setup_command
        setup = setup.strip() if isinstance(setup, str) and setup.strip() else None
        started = time.perf_counter()
        full_env = os.environ.copy()
        env_overrides: dict[str, str] = {}
        for key, value in dict(env or {}).items():
            full_env[str(key)] = str(value)
            env_overrides[str(key)] = str(value)

        if setup:
            use_shell = True
            run_command: Sequence[str] | str = f"set -e\n{setup}\n{command_display}"
            command_for_result = f"{setup} && {command_display}"
        else:
            if not isinstance(command, str):
                first = str(command[0]) if command else ""
                has_path = "/" in first
                if first and ((has_path and not Path(first).exists()) or (not has_path and shutil.which(first) is None)):
                    duration = time.perf_counter() - started
                    return BashResult(command=command_display, cwd=actual_cwd, returncode=127, stdout="", stderr=f"FileNotFoundError: [Errno 2] No such file or directory: {first!r}", duration_seconds=round(duration, 4), env_overrides=env_overrides)
                use_shell = True if shell is None else shell
                run_command = command_display if use_shell else command
            else:
                use_shell = True if shell is None else shell
                run_command = command
            command_for_result = command_display

        kwargs: dict[str, object] = {"executable": "/bin/bash"} if use_shell else {}
        # For Autopilot-managed processes, start a new session so kill_process can
        # terminate the whole subtree without touching the controller.  Keep legacy
        # unregistered foreground calls in the current session; some CI/pytest
        # sandboxes mishandle foreground PIPE subprocesses that call setsid().
        managed_session = process_registry is not None or process_started_callback is not None
        try:
            proc = subprocess.Popen(run_command, cwd=actual_cwd, env=full_env, shell=use_shell, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=managed_session, **kwargs)
        except FileNotFoundError as exc:
            duration = time.perf_counter() - started
            return BashResult(command=command_for_result, cwd=actual_cwd, returncode=127, stdout="", stderr=f"FileNotFoundError: {exc}", duration_seconds=round(duration, 4), env_overrides=env_overrides, setup_command=setup)
        except OSError as exc:
            duration = time.perf_counter() - started
            return BashResult(command=command_for_result, cwd=actual_cwd, returncode=127, stdout="", stderr=f"OSError: {exc}", duration_seconds=round(duration, 4), env_overrides=env_overrides, setup_command=setup)

        try:
            pgid = os.getpgid(proc.pid)
        except Exception:
            pgid = None
        process_record = None
        if process_registry is not None:
            try:
                process_record = process_registry.register(pid=proc.pid, name=process_label or process_kind, kind=process_kind, command=command_for_result, cwd=actual_cwd, environment=environment_name, action_id=action_id, metadata={"setup_command": setup, "stream_output": stream_output, "env_overrides": env_overrides, **(process_metadata or {})}, process_group_id=pgid)
            except Exception:
                process_record = None
        if process_started_callback is not None:
            try:
                process_started_callback({"pid": proc.pid, "process_group_id": pgid, "pgid": pgid, "process_id": process_record.process_id if process_record else None, "command": command_for_result, "cwd": actual_cwd, "setup_command": setup, "env_overrides": dict(env_overrides), "started_at_perf": started})
            except Exception:
                pass

        def finish_status(returncode: int | None, timed_out: bool) -> None:
            if process_registry is not None and process_record is not None:
                try:
                    status = "FAILED" if timed_out else ("SUCCEEDED" if returncode == 0 else "FAILED")
                    process_registry.mark_finished(process_record.process_id, status=status, exit_code=returncode, reason="timed_out" if timed_out else None)
                except Exception:
                    pass

        if stream_output:
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []

            def _reader(pipe, chunks: list[str], target) -> None:
                try:
                    for line in iter(pipe.readline, ""):
                        chunks.append(line)
                        if stream_prefix:
                            target.write(stream_prefix)
                        target.write(line)
                        target.flush()
                except Exception:
                    return

            out_thread = threading.Thread(target=_reader, args=(proc.stdout, stdout_chunks, sys.stdout), daemon=True)
            err_thread = threading.Thread(target=_reader, args=(proc.stderr, stderr_chunks, sys.stderr), daemon=True)
            out_thread.start()
            err_thread.start()
            timed_out = False
            effective_timeout = max(float(timeout or self.timeout), 30.0)
            next_heartbeat = time.perf_counter() + float(heartbeat_interval or 0) if heartbeat_callback and heartbeat_interval and heartbeat_interval > 0 else None
            while True:
                returncode = proc.poll()
                now = time.perf_counter()
                if next_heartbeat is not None and now >= next_heartbeat:
                    try:
                        heartbeat_callback({"status": "running", "pid": proc.pid, "pgid": pgid, "process_group_id": pgid, "process_id": process_record.process_id if process_record else None, "command": command_for_result, "cwd": actual_cwd, "elapsed_seconds": now - started, "stdout_tail": "".join(stdout_chunks)[-12000:], "stderr_tail": "".join(stderr_chunks)[-12000:]})
                    except Exception:
                        pass
                    next_heartbeat = now + float(heartbeat_interval or 0)
                if returncode is not None:
                    break
                if effective_timeout and now - started >= effective_timeout:
                    timed_out = True
                    try:
                        if managed_session and pgid and int(pgid) != os.getpgrp():
                            os.killpg(pgid, signal.SIGTERM)
                        else:
                            proc.kill()
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    try:
                        proc.wait(timeout=2.0)
                    except Exception:
                        try:
                            if managed_session and pgid and int(pgid) != os.getpgrp():
                                os.killpg(pgid, signal.SIGKILL)
                        except Exception:
                            pass
                        try:
                            proc.wait(timeout=2.0)
                        except Exception:
                            pass
                    break
                time.sleep(0.25)
            out_thread.join(timeout=2.0)
            err_thread.join(timeout=2.0)
            duration = time.perf_counter() - started
            stderr_text = "".join(stderr_chunks)
            if timed_out and not stderr_text:
                stderr_text = f"Timed out after {effective_timeout} seconds"
            final_returncode = None if timed_out else proc.returncode
            finish_status(final_returncode, timed_out)
            return BashResult(command=command_for_result, cwd=actual_cwd, returncode=final_returncode, stdout="".join(stdout_chunks), stderr=stderr_text, duration_seconds=round(duration, 4), timed_out=timed_out, env_overrides=env_overrides, setup_command=setup, pid=proc.pid, pgid=pgid, process_group_id=pgid, process_id=process_record.process_id if process_record else None)

        try:
            effective_timeout = max(float(timeout or self.timeout), 30.0)
            stdout, stderr = proc.communicate(timeout=effective_timeout)
            duration = time.perf_counter() - started
            finish_status(proc.returncode, False)
            return BashResult(command=command_for_result, cwd=actual_cwd, returncode=proc.returncode, stdout=stdout or "", stderr=stderr or "", duration_seconds=round(duration, 4), env_overrides=env_overrides, setup_command=setup, pid=proc.pid, pgid=pgid, process_group_id=pgid, process_id=process_record.process_id if process_record else None)
        except subprocess.TimeoutExpired:
            try:
                if managed_session and pgid and int(pgid) != os.getpgrp():
                    os.killpg(pgid, signal.SIGTERM)
                else:
                    proc.kill()
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                stdout, stderr = proc.communicate(timeout=2.0)
            except Exception:
                try:
                    if managed_session and pgid and int(pgid) != os.getpgrp():
                        os.killpg(pgid, signal.SIGKILL)
                    else:
                        proc.kill()
                except Exception:
                    pass
                stdout, stderr = proc.communicate()
            duration = time.perf_counter() - started
            finish_status(None, True)
            return BashResult(command=command_for_result, cwd=actual_cwd, returncode=None, stdout=stdout or "", stderr=(stderr or f"Timed out after {timeout or self.timeout} seconds"), duration_seconds=round(duration, 4), timed_out=True, env_overrides=env_overrides, setup_command=setup, pid=proc.pid, pgid=pgid, process_group_id=pgid, process_id=process_record.process_id if process_record else None)
