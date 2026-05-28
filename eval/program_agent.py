from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopilot.eval.benchmarks import BenchmarkRegistry, BenchmarkSpec
from autopilot.models import to_jsonable
from autopilot.runtime.trajectory import utc_now


@dataclass
class EvalProgramPlan:
    benchmark: str
    program_dir: str
    status: str = "planned"
    eval_source: str = "benchmark"
    metric: str | None = None
    install_commands: list[str] = field(default_factory=list)
    run_command: str | None = None
    parser_command: str | None = None
    expected_artifacts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    can_early_stop: bool = True
    min_cases: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


class EvalProgramAgent:
    """Run-local evaluator workspace generator.

    The teacher agent can download external benchmark harnesses, write wrappers,
    and refine parsers under eval_programs/<benchmark>/ without hard-coding every
    benchmark into Autopilot.
    """

    def __init__(self, root: str | Path, registry: BenchmarkRegistry | None = None) -> None:
        self.root = Path(root)
        self.registry = registry or BenchmarkRegistry.default()
        self.eval_root = self.root / "eval_programs"

    def infer_spec(self, benchmark: str | None, goal: str = "", target: str = "") -> BenchmarkSpec:
        if benchmark and benchmark in self.registry.benchmarks:
            return self.registry.benchmarks[benchmark]
        inferred = self.registry.infer(goal, target)
        if inferred:
            return inferred[0]
        name = benchmark or "custom_real_eval"
        return BenchmarkSpec(name=name, task_type="custom", metric="score", min_cases=1, early_stop_allowed=False, notes="Custom evaluator requires teacher/human supplied command and parser before early stop is trusted.")

    def prepare(self, *, benchmark: str | None = None, goal: str = "", target: str = "", repo_url: str | None = None, commit: str | None = None, run_command: str | None = None, execute_clone: bool = False) -> EvalProgramPlan:
        spec = self.infer_spec(benchmark, goal, target)
        program_dir = self.eval_root / spec.name
        program_dir.mkdir(parents=True, exist_ok=True)
        (program_dir / "artifacts").mkdir(exist_ok=True)
        install_cfg = spec.install if isinstance(spec.install, dict) else {}
        repo = repo_url or install_cfg.get("repo")
        pin = commit or install_cfg.get("pin") or install_cfg.get("commit")
        tool_dir = program_dir / "tool_repo"
        install_commands: list[str] = []
        notes: list[str] = []
        if repo:
            install_commands.append(f"git clone {shlex.quote(str(repo))} {shlex.quote(str(tool_dir))}")
            if pin and pin != "pinned_commit_required":
                install_commands.append(f"cd {shlex.quote(str(tool_dir))} && git checkout {shlex.quote(str(pin))}")
            else:
                notes.append("External evaluator repo must be pinned to a commit before its score can stop training.")
            install_commands.append(f"if [ -f {shlex.quote(str(tool_dir / 'requirements.txt'))} ]; then python -m pip install -r {shlex.quote(str(tool_dir / 'requirements.txt'))}; fi")
        if run_command is None:
            run_command = self._default_run_command(spec, program_dir)
        parser_command = f"python {shlex.quote(str(program_dir / 'parse_metrics.py'))} --input {shlex.quote(str(program_dir / 'artifacts'))} --output {shlex.quote(str(program_dir / 'evaluation_result.json'))}"
        plan = EvalProgramPlan(
            benchmark=spec.name,
            program_dir=str(program_dir),
            eval_source="benchmark" if spec.early_stop_allowed else "custom_or_untrusted",
            metric=spec.metric,
            install_commands=install_commands,
            run_command=run_command,
            parser_command=parser_command,
            expected_artifacts=[str(program_dir / "evaluation_result.json"), str(program_dir / "artifacts")],
            notes=(notes + ([spec.notes] if spec.notes else [])),
            can_early_stop=bool(spec.early_stop_allowed and spec.min_cases > 0),
            min_cases=spec.min_cases,
        )
        self._write_scaffold(program_dir, spec, plan)
        return plan

    def refine(self, *, benchmark: str, instruction: str, patch: str | None = None) -> dict[str, Any]:
        program_dir = self.eval_root / benchmark
        program_dir.mkdir(parents=True, exist_ok=True)
        refine_path = program_dir / "REFINEMENTS.md"
        with refine_path.open("a", encoding="utf-8") as f:
            f.write(f"\n\n## {utc_now()}\n{instruction.strip()}\n")
            if patch:
                f.write("\n```diff\n" + patch.strip() + "\n```\n")
        return {"ok": True, "benchmark": benchmark, "program_dir": str(program_dir), "refinement_log": str(refine_path), "note": "Evaluator refinement staged in run-local workspace."}

    def _default_run_command(self, spec: BenchmarkSpec, program_dir: Path) -> str:
        artifacts = program_dir / "artifacts"
        if spec.name == "aime24_all":
            return f"python {shlex.quote(str(program_dir / 'run_aime24_eval.py'))} --output {shlex.quote(str(artifacts))}"
        if spec.name == "swe_bench_lite":
            return f"bash {shlex.quote(str(program_dir / 'run_swebench_lite.sh'))}"
        if spec.name == "spider_test_suite":
            return f"bash {shlex.quote(str(program_dir / 'run_spider_eval.sh'))}"
        if spec.name == "humaneval_plus":
            return f"bash {shlex.quote(str(program_dir / 'run_humaneval_plus.sh'))}"
        return f"bash {shlex.quote(str(program_dir / 'run_eval.sh'))}"

    def _write_scaffold(self, program_dir: Path, spec: BenchmarkSpec, plan: EvalProgramPlan) -> None:
        (program_dir / "eval_program_plan.json").write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (program_dir / "benchmark_spec.json").write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        install_sh = "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(plan.install_commands or ["echo '[info] no external install commands required yet'"]) + "\n"
        (program_dir / "install_eval_program.sh").write_text(install_sh, encoding="utf-8")
        parser = r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
root = Path(args.input)
metrics = {
    'ok': False,
    'eval_source': 'benchmark',
    'benchmark': root.parent.name,
    'metric': None,
    'score': None,
    'target_met': False,
    'early_stop_allowed': False,
    'note': 'Scaffold parser: refine this to parse real evaluator output before trusting the score.'
}
Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(metrics, ensure_ascii=False))
'''
        (program_dir / "parse_metrics.py").write_text(parser, encoding="utf-8")
        readme = f"""# Eval program: {spec.name}

Task type: `{spec.task_type}`  
Metric: `{spec.metric}`  
Minimum cases for trusted early stop: `{spec.min_cases}`

This directory is run-local and agent-owned. The teacher agent may clone/pin an
external evaluator, write wrappers, run smoke checks, refine parser code, and
then produce `evaluation_result.json`. A score is trusted for early stopping only
when `eval_source=benchmark`, `early_stop_allowed=true`, and `case_count >= min_cases`.

Run command:

```bash
{plan.run_command}
```

Parser command:

```bash
{plan.parser_command}
```
"""
        (program_dir / "README.md").write_text(readme, encoding="utf-8")
        for name in ["run_eval.sh", "run_swebench_lite.sh", "run_spider_eval.sh", "run_humaneval_plus.sh"]:
            path = program_dir / name
            if not path.exists():
                path.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho '[todo] refine this evaluator runner'\n", encoding="utf-8")
        aime = program_dir / "run_aime24_eval.py"
        if spec.name == "aime24_all" and not aime.exists():
            aime.write_text(r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
(out / 'raw_results.json').write_text(json.dumps({'ok': False, 'note': 'AIME24 cases/model endpoint not configured yet; use deterministic integer exact-match, not LLM judge.'}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('[todo] configure AIME24 cases and local model endpoint')
''', encoding="utf-8")
