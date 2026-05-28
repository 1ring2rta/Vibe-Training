from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PermissionDecision:
    allow: bool
    reason: str = "allowed"
    requires_human: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PermissionPolicy:
    """Capability-level policy for the atomic tool runtime.

    The policy intentionally checks capabilities and artifacts, not workflow names.
    Inspired by OpenClaw/Codex/Claude Code style layering: small tool surface,
    sandbox-ish workspace boundary, pre-tool hooks, and deny-first precedence.
    """

    allow_bash: bool = True
    allow_network: bool = False
    allow_nonlocal_filesystem: bool = False
    allow_core_repo_edits: bool = False
    allow_process_kill: bool = True
    allow_skill_memory_autopromote: bool = True
    max_high_risk_without_human: int = 0
    denied_tools: set[str] = field(default_factory=set)

    def check(self, action_type: str, tool_name: str | None = None, risk_level: str = "low", arguments: dict[str, Any] | None = None) -> tuple[bool, str]:
        decision = self.check_decision(action_type, tool_name, risk_level, arguments or {})
        return decision.allow, decision.reason

    def check_decision(self, action_type: str, tool_name: str | None = None, risk_level: str = "low", arguments: dict[str, Any] | None = None) -> PermissionDecision:
        arguments = arguments or {}
        name = tool_name or action_type
        if name in self.denied_tools:
            return PermissionDecision(False, f"tool/action {name} denied by policy")
        if action_type not in {"run_tool", "stop"}:
            return PermissionDecision(False, f"non-atomic action '{action_type}' denied")
        if action_type == "run_tool" and name not in {"bash", "cat", "grep", "web_search", "browser", "answer_human"}:
            return PermissionDecision(False, f"tool '{name}' is not in the atomic tool allowlist")
        if action_type == "run_tool" and name == "bash":
            return self.check_bash(str(arguments.get("command") or ""), risk_level=risk_level)
        if action_type == "run_tool" and name in {"web_search", "browser"}:
            if not self.allow_network:
                # Allow these tools even when bash network is denied; network access is meant
                # to go through explicit/search/browser tools rather than hidden shell curl.
                return PermissionDecision(True, "explicit web tool allowed")
        if risk_level == "high" and self.max_high_risk_without_human <= 0:
            return PermissionDecision(False, "high-risk atomic action requires answer_human decision or explicit permission")
        return PermissionDecision(True, "allowed")

    def check_bash(self, command: str, *, risk_level: str = "low") -> PermissionDecision:
        if not self.allow_bash:
            return PermissionDecision(False, "bash disabled")
        lowered = command.lower()
        if not self.allow_network and self._looks_like_nonlocal_network(lowered):
            return PermissionDecision(False, "bash network access denied; use web_search/browser for external network access")
        if not self.allow_nonlocal_filesystem and self._looks_like_external_sensitive_fs(lowered):
            return PermissionDecision(False, "bash attempts broad/sensitive filesystem access outside the run workspace")
        if not self.allow_core_repo_edits and self._looks_like_core_repo_edit(command):
            return PermissionDecision(False, "editing core autopilot source is denied in post-training runtime")
        if not self.allow_process_kill and re.search(r"\b(kill|pkill|killall)\b", lowered):
            return PermissionDecision(False, "process killing disabled")
        if self._looks_like_policy_relaxation(lowered):
            return PermissionDecision(False, "policy/decontamination relaxation requires human choice", requires_human=True)
        if risk_level == "high" and self.max_high_risk_without_human <= 0:
            return PermissionDecision(False, "high-risk bash command requires answer_human decision or explicit permission")
        return PermissionDecision(True, "allowed")

    @staticmethod
    def _looks_like_nonlocal_network(command: str) -> bool:
        if re.search(r"\b(curl|wget|aria2c|scp|rsync|ssh|git\s+clone|pip\s+install|npm\s+install)\b", command):
            return True
        urls = re.findall(r"https?://[^\s'\"]+", command)
        for url in urls:
            if not re.search(r"https?://(127\.0\.0\.1|localhost|0\.0\.0\.0)([:/]|$)", url):
                return True
        return False

    @staticmethod
    def _looks_like_external_sensitive_fs(command: str) -> bool:
        # Do not block normal absolute model paths; block obviously broad scans/writes
        # of system/home/secret locations.
        sensitive = ["/etc/", "/root/", "/home/", "~/.ssh", ".ssh/", ".aws/", ".config/", "/var/run/"]
        if any(x in command for x in sensitive):
            return True
        if re.search(r"\bfind\s+/(\s|$)", command):
            return True
        return False

    @staticmethod
    def _looks_like_core_repo_edit(command: str) -> bool:
        mutators = r"(cat\s+>\s*|tee\s+|sed\s+-i|python\s+-.*open\(|apply_patch|git\s+apply|patch\s+)"
        if not re.search(mutators, command, re.S):
            return False
        return bool(re.search(r"\bautopilot/(kernel|runtime|tools|cli|data|eval|llm|llamafactory)/", command))

    @staticmethod
    def _looks_like_policy_relaxation(command: str) -> bool:
        patterns = [
            "--allow-benchmark-contamination",
            "--disable-decontam",
            "--allow-contaminated",
            "--max-zero-row-fraction 1.0",
            "--max-zero-row-fraction=1.0",
            "--actions accept,review",
            "--actions=accept,review",
        ]
        return any(p in command for p in patterns)
