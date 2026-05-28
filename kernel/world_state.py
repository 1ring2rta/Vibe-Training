from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autopilot.runtime.state import RunStateStore
from autopilot.runtime.trajectory import FrontierTrajectoryRecorder
from autopilot.runtime.processes import ProcessRegistry
from autopilot.eval.programs import EvalProgramWorkspace


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _tail_jsonl(path: Path, limit: int = 20) -> list[Any]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            rows.append({"raw": line[:1000]})
    return rows


class WorldStateBuilder:
    def __init__(self, root: str | Path, *, settings: Any = None, goal: str = "", target: str = "") -> None:
        self.root = Path(root)
        self.settings = settings
        self.goal = goal
        self.target = target
        self.store = RunStateStore(self.root)

    def materialize(self) -> dict[str, Any]:
        traj_root = self.root / ".autopilot" / "frontier_trajectory"
        rec = FrontierTrajectoryRecorder(root=traj_root) if traj_root.exists() else None
        state = {
            "root": str(self.root),
            "canonical_paths": {
                "run_root": str(self.root.resolve()),
                "path_rule": "Action paths are run-root-relative by default. Do not prefix paths with runs/<run_name> when root already is that run.",
            },
            "goal": self.goal,
            "target": self.target,
            "self_repair_policy": {
                "bugs_do_not_block_human": True,
                "answer_human_only_for_choices": True,
                "examples_to_repair_without_human": [
                    "invalid action_type",
                    "tool/action alias mismatch",
                    "path/file not found",
                    "empty think/no-op response",
                    "failed command with obvious next diagnostic",
                ],
            },
            "run_state": self.store.state(),
            "task_graph": self.store.task_graph(),
            "artifacts": _read_json(self.store.artifacts_path, {}),
            "recent_events": _tail_jsonl(self.store.event_log_path, 30),
            "round_metrics_history": _read_json(self.root / "round_metrics_history.json", []),
            "goal_loop_report": _read_json(self.root / "goal_loop_report.json", {}),
            "known_files": self._known_files(),
            "latest_evaluation_result": _read_json(self.root / "evaluation_result.json", {}),
            "latest_decontamination_report": self._latest_decontamination_report(),
            "latest_prepare_manifest": self._latest_prepare_manifest(),
        }
        try:
            state["processes"] = ProcessRegistry(self.root).list(active_only=False)
            state["process_registry"] = {"registry_path": str(ProcessRegistry(self.root).registry_path), "processes": state["processes"]}
        except Exception as exc:
            state["process_registry"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            state["processes"] = []

        try:
            state["eval_programs"] = EvalProgramWorkspace(self.root).list()
        except Exception as exc:
            state["eval_programs"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if rec is not None:
            state["frontier_trajectory_audit"] = rec.audit()
        if self.settings is not None:
            state["runtime_environments"] = getattr(self.settings, "environment_summaries", lambda: [])()
        return state

    def _latest_prepare_manifest(self) -> dict[str, Any]:
        manifests = sorted(self.root.glob("round_*/prepared/prepare_manifest.json"))
        if not manifests:
            return {}
        return _read_json(manifests[-1], {})

    def _latest_decontamination_report(self) -> dict[str, Any]:
        reports = sorted(self.root.glob("round_*/prepared/decontamination_report.json"))
        if not reports:
            report = self.root / "decontamination_report.json"
            return _read_json(report, {}) if report.exists() else {}
        return _read_json(reports[-1], {})

    def _known_files(self) -> list[str]:
        out: list[str] = []
        for pattern in ["eval_cases.jsonl", "aime24.jsonl", "evaluation_result.json", "round_*/collection/collection_report.json", "round_*/prepared/prepare_manifest.json", "round_*/prepared/decontamination_report.json", "round_*/prepared/configs/train_*.yaml", "round_*/metrics/round_metrics.json", ".autopilot/eval_programs/*/eval_program_spec.json", ".autopilot/processes/process_registry.json"]:
            out.extend(str(p.relative_to(self.root)) for p in self.root.glob(pattern))
        return sorted(out)[:200]

