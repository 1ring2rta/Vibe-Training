from __future__ import annotations

from typing import Iterable

from autopilot.tools.registry import ToolRegistry, ToolSpec, infer_tool_needs
from autopilot.tools.web_search import WebSearchResult, WebSearchTool


def _candidate_from_hit(hit: WebSearchResult, *, query: str) -> ToolSpec:
    title = (hit.title or hit.url).strip()[:80]
    name = "web_tool_candidate_" + str(abs(hash(hit.url)) % 1_000_000)
    return ToolSpec(
        name=name,
        kind="external_tool_candidate",
        purpose=f"Potential tool/resource found for query: {query}",
        status="candidate",
        source_url=hit.url,
        confidence=0.45,
        notes=(title + " — " + (hit.snippet or ""))[:500],
        metadata={"source": hit.source, "query": query},
    )


def discover_tools_for_goal(
    goal: str,
    *,
    registry: ToolRegistry | None = None,
    web_search: WebSearchTool | None = None,
    max_web_results: int = 5,
) -> ToolRegistry:
    """Discover missing tools as a loop step.

    The deterministic part infers likely tool needs from the goal. If web search
    is available, candidates from docs/repos/papers are added as untrusted
    candidates. A later subtask can inspect and enable them.
    """
    registry = registry or ToolRegistry.default()
    for tool in infer_tool_needs(goal):
        if registry.get(tool.name) is None:
            registry.add(tool)

    if web_search is not None and max_web_results > 0:
        queries = [
            f"{goal} tool use benchmark verifier",
            f"{goal} agent environment open source",
            f"{goal} evaluation tool github",
        ]
        for query in queries:
            try:
                hits = web_search.search(query, limit=max_web_results)
            except Exception:
                hits = []
            for hit in hits[:max_web_results]:
                registry.add(_candidate_from_hit(hit, query=query))
    return registry
