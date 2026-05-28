from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopilot.eval.benchmarks import BenchmarkRegistry, BenchmarkSpec
from autopilot.models import to_jsonable
from autopilot.runtime.trajectory import append_jsonl, atomic_write_json, utc_now


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return value[:80] or f"eval_{uuid.uuid4().hex[:8]}"


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


@dataclass
class EvalProgramSpec:
    name: str
    benchmark: str
    kind: str
    status: str = "planned"
    metric: str | None = None
    runner: str | None = None
    verifier: str | None = None
    repo_url: str | None = None
    commit: str | None = None
    install_commands: list[str] = field(default_factory=list)
    run_command: str | None = None
    parser: str | None = None
    program_dir: str | None = None
    created_by: str = "autonomous_agent"
    version: int = 1
    notes: str = ""
    files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self.__dict__)


class EvalProgramWorkspace:
    """Run-local workspace for evaluator programs.

    The agent can plan external benchmark harnesses, write evaluator code, and
    refine scripts here without mutating the core repository.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.dir = self.root / ".autopilot" / "eval_programs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.dir / "eval_programs.json"
        self.events_path = self.dir / "eval_program_events.jsonl"

    def _load(self) -> dict[str, Any]:
        return _read_json(self.registry_path, {"programs": {}})

    def _save(self, data: dict[str, Any]) -> None:
        data["updated_at"] = utc_now()
        atomic_write_json(self.registry_path, data)

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        append_jsonl(self.events_path, {"event_id": uuid.uuid4().hex, "timestamp": utc_now(), "type": event_type, "payload": payload})

    def register(self, spec: EvalProgramSpec) -> EvalProgramSpec:
        data = self._load()
        data.setdefault("programs", {})[spec.name] = spec.to_dict()
        self._save(data)
        self.append_event("eval_program_registered", spec.to_dict())
        return spec

    def list(self) -> list[dict[str, Any]]:
        return sorted(list((self._load().get("programs") or {}).values()), key=lambda x: str(x.get("name") or ""))

    def get(self, name: str) -> dict[str, Any] | None:
        row = (self._load().get("programs") or {}).get(name)
        return row if isinstance(row, dict) else None

    def plan_from_benchmark(self, benchmark: str, *, goal: str = "", target: str = "", metric: str | None = None) -> EvalProgramSpec:
        registry = BenchmarkRegistry.default()
        bench = registry.benchmarks.get(benchmark)
        if bench is None:
            inferred = registry.infer(goal, target)
            bench = inferred[0] if inferred else BenchmarkSpec(name=benchmark or "custom_eval", task_type="custom", metric=metric or "score", min_cases=1, early_stop_allowed=False, notes="Custom benchmark; evaluator must be provided or written by the agent.")
        program_name = _slug(bench.name)
        program_dir = self.dir / program_name
        program_dir.mkdir(parents=True, exist_ok=True)
        kind = "agentic" if bench.task_type.startswith("agentic") or "swe" in bench.name else ("external_repo" if bench.install.get("repo") else "builtin")
        repo_url = bench.install.get("repo") if isinstance(bench.install, dict) else None
        commit = bench.install.get("pin") if isinstance(bench.install, dict) else None
        install_commands: list[str] = []
        if repo_url:
            tool_dir = program_dir / "repo"
            install_commands = [f"git clone {repo_url} {tool_dir}"]
            if commit and commit != "pinned_commit_required":
                install_commands.append(f"cd {tool_dir} && git checkout {commit}")
            else:
                install_commands.append(f"cd {tool_dir} && git rev-parse HEAD > {program_dir / 'PINNED_COMMIT.txt'}")
            install_commands.append(f"cd {tool_dir} && python -m pip install -e . || python -m pip install -r requirements.txt || true")
        run_command = None
        parser = None
        if bench.name == "aime24_all":
            run_command = f"python {program_dir / 'eval_aime24.py'} --cases <aime24.jsonl> --predictions <predictions.jsonl> --output <evaluation_result.json>"
            parser = "aime_integer_exact_json"
            self._write_aime_template(program_dir)
        elif "spider" in bench.name:
            run_command = f"python {program_dir / 'repo' / 'evaluation.py'} --gold <gold> --pred <pred> --db <db_dir> --table <tables_json>"
            parser = "parse_spider_test_suite_stdout"
        elif "swe" in bench.name:
            run_command = f"python {program_dir / 'repo' / 'swebench' / 'harness' / 'run_evaluation.py'} --dataset_name <dataset> --predictions_path <predictions.jsonl> --run_id <run_id>"
            parser = "parse_swebench_report"
        spec = EvalProgramSpec(name=program_name, benchmark=bench.name, kind=kind, status="planned", metric=bench.metric, runner=bench.runner, verifier=bench.verifier, repo_url=repo_url, commit=commit, install_commands=install_commands, run_command=run_command, parser=parser, program_dir=str(program_dir), notes=bench.notes, metadata={"goal": goal, "target": target, "benchmark_spec": bench.to_dict()})
        (program_dir / "eval_program_spec.json").write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return self.register(spec)

    def write_generated_program(self, *, name: str, benchmark: str, files: dict[str, str], notes: str = "", metric: str | None = None, parser: str | None = None) -> EvalProgramSpec:
        program_name = _slug(name)
        program_dir = self.dir / program_name
        program_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}
        for rel, content in files.items():
            path = (program_dir / rel).resolve()
            if not str(path).startswith(str(program_dir.resolve())):
                raise ValueError(f"refusing to write outside eval program dir: {rel}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8")
            written[str(rel)] = str(path)
        spec = EvalProgramSpec(name=program_name, benchmark=benchmark, kind="generated", status="draft", metric=metric, parser=parser, program_dir=str(program_dir), notes=notes, files=written)
        (program_dir / "eval_program_spec.json").write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return self.register(spec)

    def record_refinement(self, *, name: str, instructions: str, patch: str | None = None, files: dict[str, str] | None = None) -> dict[str, Any]:
        current = self.get(_slug(name)) or self.get(name)
        if current is None:
            return {"ok": False, "error": f"eval program not found: {name}"}
        program_dir = Path(str(current.get("program_dir") or self.dir / _slug(name)))
        version = int(current.get("version") or 1) + 1
        ref_dir = program_dir / "refinements" / f"v{version:03d}"
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / "instructions.md").write_text(instructions, encoding="utf-8")
        if patch:
            (ref_dir / "patch.diff").write_text(patch, encoding="utf-8")
        for rel, content in (files or {}).items():
            path = (program_dir / rel).resolve()
            if not str(path).startswith(str(program_dir.resolve())):
                raise ValueError(f"refusing to write outside eval program dir: {rel}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8")
        current["version"] = version
        current["status"] = "refined"
        current["updated_at"] = utc_now()
        data = self._load()
        data.setdefault("programs", {})[str(current.get("name"))] = current
        self._save(data)
        result = {"ok": True, "name": current.get("name"), "version": version, "refinement_dir": str(ref_dir), "program_dir": str(program_dir)}
        self.append_event("eval_program_refined", result | {"instructions": instructions[:2000]})
        return result

    def _write_aime_template(self, program_dir: Path) -> None:
        path = program_dir / "eval_aime24.py"
        if path.exists():
            return
        path.write_text('''#!/usr/bin/env python3\nfrom __future__ import annotations\nimport argparse, json, re\nfrom pathlib import Path\n\ndef norm(text):\n    text = str(text or '')\n    m = re.findall(r"\\\\boxed\\{([0-9]{1,3})\\}|(?:answer is|final answer:?|答案[:：])\\s*([0-9]{1,3})|\\b([0-9]{1,3})\\b", text, flags=re.I)\n    vals = [x for tup in m for x in tup if x]\n    return str(int(vals[-1])) if vals else ''\n\ndef load_jsonl(path):\n    rows=[]\n    for line in Path(path).read_text(encoding='utf-8').splitlines():\n        if line.strip(): rows.append(json.loads(line))\n    return rows\n\ndef main():\n    ap=argparse.ArgumentParser()\n    ap.add_argument('--cases', required=True)\n    ap.add_argument('--predictions', required=True)\n    ap.add_argument('--output', required=True)\n    ns=ap.parse_args()\n    cases=load_jsonl(ns.cases)\n    preds=load_jsonl(ns.predictions)\n    by_id={str(r.get('id') or r.get('case_id') or i): r for i,r in enumerate(preds)}\n    results=[]; correct=0\n    for i,c in enumerate(cases):\n        cid=str(c.get('id') or c.get('case_id') or i)\n        pred=by_id.get(cid) or (preds[i] if i < len(preds) else {})\n        got=norm(pred.get('response') or pred.get('prediction') or pred.get('answer'))\n        gold=str(int(str(c.get('answer') or c.get('expected')).strip()))\n        ok=got==gold\n        correct+=int(ok)\n        results.append({'id': cid, 'gold': gold, 'prediction': got, 'passed': ok})\n    score=correct/len(cases) if cases else 0.0\n    out={'ok': True, 'eval_source': 'benchmark', 'benchmark': 'aime24_all', 'metric_name': 'exact_match_accuracy', 'case_count': len(cases), 'correct': correct, 'score': score, 'target_met': score>=0.8 and len(cases)>=30, 'case_results': results}\n    Path(ns.output).write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\\n', encoding='utf-8')\n    print(json.dumps(out, ensure_ascii=False))\nif __name__ == '__main__': main()\n''', encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {"registry_path": str(self.registry_path), "events_path": str(self.events_path), "programs": self.list()}
