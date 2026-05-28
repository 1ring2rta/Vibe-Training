from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopilot.models import to_jsonable
from autopilot.tools.web_search import WebSearchTool


@dataclass
class VerifierCandidate:
    name: str
    kind: str  # exact_match | unit_test | judge | external_repo | trajectory | dataset_answer
    source: str = "heuristic"
    confidence: float = 0.5
    reward_signal: str = ""
    tool_requirements: list[str] = field(default_factory=list)
    url: str | None = None
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierPlan:
    goal: str
    candidates: list[VerifierCandidate] = field(default_factory=list)
    recommended: list[str] = field(default_factory=list)
    rl_ready: bool = False
    backend_suggestion: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return p


def heuristic_verifiers(goal: str, *, kimi_configured: bool = False) -> list[VerifierCandidate]:
    lower = goal.lower()
    out: list[VerifierCandidate] = []
    if any(x in lower for x in ["code", "coding", "python", "代码", "bug", "算法"]):
        out.append(
            VerifierCandidate(
                name="python_unit_tests",
                kind="unit_test",
                confidence=0.9,
                reward_signal="pass_rate over executable tests",
                tool_requirements=["coding_sandbox", "pytest"],
                notes="Best default verifier for coding RLVR/GRPO when tasks include tests or generated tests can be validated.",
            )
        )
    if any(x in lower for x in ["math", "数学", "gsm", "proof", "推理"]):
        out.append(
            VerifierCandidate(
                name="exact_or_symbolic_answer",
                kind="exact_match",
                confidence=0.82,
                reward_signal="exact/final-answer match",
                tool_requirements=["exact_answer_checker"],
                notes="Works for math tasks with normalized boxed/final answers.",
            )
        )
    if any(x in lower for x in ["agent", "tool", "工具", "browser", "浏览器"]):
        out.append(
            VerifierCandidate(
                name="trajectory_and_final_answer_judge",
                kind="trajectory",
                confidence=0.72,
                reward_signal="tool-call correctness plus final-answer quality",
                tool_requirements=["tool_use_eval", "trajectory_judge", "kimi_judge"],
                notes="Agentic RL needs trajectory-level verification, not just final-answer scoring.",
            )
        )
    if kimi_configured:
        out.append(
            VerifierCandidate(
                name="kimi_llm_judge",
                kind="judge",
                source="kimi",
                confidence=0.62,
                reward_signal="rubric score from KIMI judge",
                tool_requirements=["kimi_judge"],
                notes="Use as fallback or tie-breaker; prefer executable/exact verifiers when available.",
            )
        )
    if not out:
        out.append(
            VerifierCandidate(
                name="reference_answer_or_kimi_judge",
                kind="dataset_answer",
                confidence=0.45,
                reward_signal="reference answer similarity or judge rubric score",
                tool_requirements=["kimi_judge"] if kimi_configured else [],
                notes="No obvious verifier class inferred; start with SFT/DPO and design task-specific evals.",
            )
        )
    return out


def discover_verifiers(
    goal: str,
    *,
    kimi_configured: bool = False,
    web_search: WebSearchTool | None = None,
    max_web_results: int = 5,
) -> VerifierPlan:
    candidates = heuristic_verifiers(goal, kimi_configured=kimi_configured)
    notes = []
    if web_search is not None and max_web_results > 0:
        queries = [
            f"{goal} verifier unit tests benchmark",
            f"{goal} reward function github",
            f"{goal} RLVR verifier dataset",
        ]
        for query in queries:
            try:
                hits = web_search.search(query, limit=max_web_results)
            except Exception as exc:
                notes.append(f"web verifier search failed for {query}: {type(exc).__name__}: {exc}")
                hits = []
            for hit in hits[:max_web_results]:
                candidates.append(
                    VerifierCandidate(
                        name="web_verifier_" + str(abs(hash(hit.url)) % 1_000_000),
                        kind="external_repo",
                        source=hit.source or "web_search",
                        confidence=0.45,
                        reward_signal="external verifier candidate; inspect before enabling",
                        url=hit.url,
                        notes=((hit.title or hit.url) + " — " + (hit.snippet or ""))[:500],
                        metadata={"query": query},
                    )
                )
    recommended = [c.name for c in sorted(candidates, key=lambda c: c.confidence, reverse=True)[:3]]
    rl_ready = any(c.kind in {"unit_test", "exact_match", "trajectory"} and c.confidence >= 0.7 for c in candidates)
    backend = "TRL/OpenRLHF GRPO/RLVR" if rl_ready else "SFT/DPO first; RL only after a verifier is validated"
    if kimi_configured:
        notes.append("KIMI judge is available, but executable/exact verifiers should be preferred for RL rewards.")
    return VerifierPlan(goal=goal, candidates=candidates, recommended=recommended, rl_ready=rl_ready, backend_suggestion=backend, notes=notes)


# Compatibility wrappers used by the v0.5 goal loop.
VerifierSpec = VerifierCandidate


def discover_verifier_candidates(goal: str, *, web_tool=None, kimi=None, limit: int = 12) -> list[VerifierCandidate]:
    plan = discover_verifiers(goal, kimi_configured=bool(kimi), web_search=web_tool, max_web_results=3)
    return plan.candidates[:limit]


def write_verifiers(verifiers: list[VerifierCandidate], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"verifiers": to_jsonable(verifiers)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p
