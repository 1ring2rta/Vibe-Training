from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from autopilot.models import to_jsonable


@dataclass
class ToolSpec:
    name: str
    kind: str
    purpose: str
    status: str = "available"  # available | candidate | enabled | missing
    command: str | None = None
    package: str | None = None
    source_url: str | None = None
    confidence: float = 0.5
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    def __init__(self, tools: Iterable[ToolSpec] | None = None) -> None:
        self.tools: dict[str, ToolSpec] = {}
        for tool in tools or []:
            self.add(tool)

    @classmethod
    def default(cls) -> "ToolRegistry":
        return cls(
            [
                ToolSpec("hf_search", "data_search", "Search Hugging Face datasets via Hub API."),
                ToolSpec("web_search", "web_search", "Search the public web for datasets, papers, repos, and verifier code."),
                ToolSpec("web_browser", "browser", "Read web pages, dataset cards, papers, and docs."),
                ToolSpec("coding_sandbox", "code_execution", "Run small local Python snippets for conversion, verifier tests, and diagnostics."),
                ToolSpec("bash_runner", "command", "Execute local commands and capture logs as agent-loop tasks."),
                ToolSpec("ask_human", "human_in_loop", "Ask a human for uncertain resource, data, training, or repository decisions and record responses in a file-backed queue."),
                ToolSpec("runtime_environment_registry", "environment", "Describe preinstalled environments so KIMI can decide when to activate them."),
                ToolSpec("repo_agent", "code_agent", "Let KIMI inspect and optionally modify the Autopilot repository during the training loop."),
                ToolSpec("claude_memory", "memory", "Read Claude-compatible project memory such as CLAUDE.md and .claude/memory.md."),
                ToolSpec("post_training_agent_memory", "memory", "Write stable post-training lessons to PostTrainingAgent.md."),
                ToolSpec("nvidia_smi", "resource_monitor", "Inspect GPU count, memory, and utilization so planning is resource-aware.", command="nvidia-smi"),
                ToolSpec("vllm_manager", "process_manager", "Deploy or kill vLLM servers to probe models and free GPU memory for training."),
                ToolSpec("vllm_probe", "model_inference", "Probe the local model through an OpenAI-compatible vLLM endpoint."),
                ToolSpec("kimi_judge", "llm_judge", "Use KIMI as planner, judge, eval sample generator, and fallback verifier."),
                ToolSpec("llamafactory", "training_backend", "Run SFT/DPO/KTO/PT through LLaMA-Factory YAML configs."),
            ]
        )

    def add(self, tool: ToolSpec) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec | None:
        return self.tools.get(name)

    def list(self, *, status: str | None = None) -> list[ToolSpec]:
        values = list(self.tools.values())
        if status is not None:
            values = [tool for tool in values if tool.status == status]
        return values

    def to_dict(self) -> dict[str, Any]:
        return {name: to_jsonable(tool) for name, tool in sorted(self.tools.items())}

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return p


def infer_tool_needs(goal: str) -> list[ToolSpec]:
    lower = goal.lower()
    needs: list[ToolSpec] = []
    if any(x in lower for x in ["agent", "tool", "工具", "浏览器", "browser", "web"]):
        needs.extend(
            [
                ToolSpec("tool_use_eval", "eval_tool", "Evaluate whether the model can decide when and how to call tools.", status="candidate", confidence=0.7),
                ToolSpec("browser_env", "agent_env", "A browser or web navigation environment for agentic training/evaluation.", status="candidate", confidence=0.65),
                ToolSpec("trajectory_judge", "trajectory_verifier", "Judge multi-step tool-use trajectories, including intermediate calls and final answers.", status="candidate", confidence=0.65),
            ]
        )
    if any(x in lower for x in ["code", "coding", "python", "代码", "bug", "算法"]):
        needs.extend(
            [
                ToolSpec("pytest", "verifier_tool", "Run Python unit tests for coding tasks.", command="python -m pytest", package="pytest", status="candidate", confidence=0.85),
                ToolSpec("sandboxed_python", "verifier_tool", "Execute generated code against small unit tests.", command="python", status="candidate", confidence=0.8),
            ]
        )
    if any(x in lower for x in ["math", "数学", "推理", "gsm", "olympiad"]):
        needs.append(ToolSpec("exact_answer_checker", "verifier_tool", "Check boxed/final numeric or symbolic answers.", status="candidate", confidence=0.75))
    if any(x in lower for x in ["paper", "literature", "文献", "论文"]):
        needs.extend(
            [
                ToolSpec("paper_search", "literature_search", "Find papers and benchmark descriptions on the web.", status="candidate", confidence=0.75),
                ToolSpec("pdf_reader", "document_reader", "Read and summarize PDF papers for data/tool ideas.", status="candidate", confidence=0.7),
            ]
        )
    # Deduplicate while keeping stronger confidence.
    dedup: dict[str, ToolSpec] = {}
    for tool in needs:
        old = dedup.get(tool.name)
        if old is None or tool.confidence > old.confidence:
            dedup[tool.name] = tool
    return list(dedup.values())


def deterministic_tool_queries(goal: str) -> list[str]:
    lower = goal.lower()
    queries = [
        f"{goal} benchmark evaluation tool",
        f"{goal} dataset verifier reward function",
        f"{goal} Hugging Face dataset tests",
    ]
    if any(x in lower for x in ["code", "coding", "python", "代码", "算法"]):
        queries.extend(["code generation benchmark unit tests", "HumanEval MBPP APPS verifier", "python programming dataset unit tests"])
    if any(x in lower for x in ["math", "数学", "推理"]):
        queries.extend(["math reasoning benchmark exact match verifier", "GSM8K MATH verifier reward function"])
    if any(x in lower for x in ["agent", "tool", "工具", "browser"]):
        queries.extend(["agent tool use benchmark", "LLM tool calling evaluation framework"])
    dedup: list[str] = []
    for q in queries:
        if q not in dedup:
            dedup.append(q)
    return dedup


def discover_tool_candidates(goal: str, *, web_tool=None, kimi=None, limit: int = 12) -> list[ToolSpec]:
    """Return candidate tools/benchmarks to add to the registry.

    The first candidates are deterministic inferred needs; optional web search
    adds external benchmark/repo candidates. These are recorded as candidates,
    not automatically installed.
    """
    candidates = infer_tool_needs(goal)
    if web_tool is not None:
        for query in deterministic_tool_queries(goal)[:6]:
            try:
                hits = web_tool.search(query, limit=3)
            except Exception:
                hits = []
            for hit in hits:
                title = hit.title or hit.url or "external_tool"
                name = "external_" + "".join(ch if ch.isalnum() else "_" for ch in title.lower())[:80]
                candidates.append(
                    ToolSpec(
                        name=name,
                        kind="external_tool",
                        purpose=(hit.snippet or title)[:500],
                        status="candidate",
                        source_url=hit.url,
                        confidence=0.45,
                        notes=f"Found by {getattr(hit, 'source', '') or 'web_search'} for query: {query}",
                        metadata={"title": hit.title, "snippet": hit.snippet},
                    )
                )
                if len(candidates) >= limit:
                    return candidates[:limit]
    return candidates[:limit]
