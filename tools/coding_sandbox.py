from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class PythonSandboxTool:
    """Small subprocess-based coding sandbox for future verifier/literature tools.

    It is intentionally simple: execute Python code in a temporary directory with
    a timeout, capture stdout/stderr, and return the result.
    """

    def __init__(self, timeout_seconds: int = 10) -> None:
        self.timeout_seconds = timeout_seconds

    def run_python(self, code: str) -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="autopilot-sandbox-") as tmp:
            script = Path(tmp) / "snippet.py"
            script.write_text(code, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, str(script)],
                    cwd=tmp,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                )
                return SandboxResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
            except subprocess.TimeoutExpired as exc:
                return SandboxResult(
                    returncode=-1,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or f"Timed out after {self.timeout_seconds}s",
                    timed_out=True,
                )
