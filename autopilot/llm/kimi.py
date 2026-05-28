from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autopilot.config import Settings
from autopilot.data.schema import compact_sample_rows
from autopilot.llm.conversation_recorder import ConversationRecorder
from autopilot.llm.openai_compatible import OpenAICompatibleChatClient, parse_jsonish
from autopilot.runtime.tools import WaitingForHuman, build_default_model_tool_registry
from autopilot.runtime.trajectory import FrontierTrajectoryRecorder
from autopilot.models import DatasetClassification, DatasetInspection, DatasetWebOverview, ModelTrial, TrainingType


class KimiClient:
    """KIMI API client using the OpenAI-compatible SDK interface."""

    def __init__(self, settings: Settings, conversation_root: str | Path | None = None, session_id: str | None = None) -> None:
        if not settings.kimi_configured:
            raise ValueError("KIMI_API_KEY, KIMI_BASE_URL, and KIMI_MODEL are required.")
        self.settings = settings
        self.conversation_root = Path(conversation_root) if conversation_root is not None else None
        # Keep legacy trainable KIMI logs under `.autopilot/conversations`, but
        # keep the source-of-truth frontier trajectory beside it at
        # `.autopilot/frontier_trajectory` so all model clients in one run can
        # append to the same request/response ledger.
        if self.conversation_root is not None and self.conversation_root.name == "conversations":
            trajectory_root = self.conversation_root.parent / "frontier_trajectory"
        else:
            trajectory_root = (self.conversation_root / "frontier_trajectory") if self.conversation_root is not None else None
        self.trajectory_recorder = FrontierTrajectoryRecorder.from_settings(settings, root=trajectory_root)
        client_cfg = (settings.raw_config.get("clients", {}) or {}).get("kimi", {}) if isinstance(settings.raw_config.get("clients"), dict) else {}
        kimi_cfg = settings.raw_config.get("kimi", {}) if isinstance(settings.raw_config.get("kimi"), dict) else {}
        params = dict(client_cfg.get("params") or kimi_cfg.get("params") or {})
        for key in ["temperature", "top_p", "max_completion_tokens", "max_tokens"]:
            if key in kimi_cfg and key not in params:
                params[key] = kimi_cfg[key]
        timeout = float(client_cfg.get("timeout") or kimi_cfg.get("timeout") or 600.0)
        self.chat_client = OpenAICompatibleChatClient(
            api_key=settings.kimi_api_key or "",
            base_url=settings.kimi_base_url,
            model=settings.kimi_model,
            timeout=timeout,
            provider_name="kimi",
            client_name="kimi",
            default_params=params,
            trajectory_recorder=self.trajectory_recorder,
        )
        workspace = self.conversation_root.parent if self.conversation_root is not None else Path(settings.effective_repo_path)
        self.model_tools = build_default_model_tool_registry(workspace=workspace, allow_bash=False)
        self.recorder = ConversationRecorder.from_settings(
            settings,
            root=conversation_root,
            session_id=session_id,
            provider="kimi",
            model=settings.kimi_model,
        )

    @staticmethod
    def _temperature_retry_needed(exc: Exception, temperature: float) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return temperature != 1.0 and "invalid temperature" in text and "only 1" in text

    def _temperature_attempts(self, temperature: float) -> list[float]:
        model = (self.settings.kimi_model or "").lower()
        if model.startswith(("kimi-k2.6", "kimi-k2.5")):
            return [1.0]
        first = float(temperature)
        return [first] if first == 1.0 else [first, 1.0]

    def _kimi_extra_body(self) -> dict[str, Any]:
        cfg = self.settings.raw_config.get("kimi", {}) if isinstance(self.settings.raw_config.get("kimi"), dict) else {}
        client_cfg = (self.settings.raw_config.get("clients", {}) or {}).get("kimi", {}) if isinstance(self.settings.raw_config.get("clients"), dict) else {}
        extra = dict(client_cfg.get("extra_body") or cfg.get("extra_body") or {})
        model = (self.settings.kimi_model or "").lower()
        if model.startswith(("kimi-k2.6", "kimi-k2.5")):
            extra.setdefault("thinking", {"type": "enabled"})
        return extra

    def _max_completion_tokens(self, max_tokens: int) -> int | None:
        cfg = self.settings.raw_config.get("kimi", {}) if isinstance(self.settings.raw_config.get("kimi"), dict) else {}
        client_cfg = (self.settings.raw_config.get("clients", {}) or {}).get("kimi", {}) if isinstance(self.settings.raw_config.get("clients"), dict) else {}
        params = dict(client_cfg.get("params") or cfg.get("params") or {})
        return int(params.get("max_completion_tokens") or cfg.get("max_completion_tokens") or max_tokens)

    def _handle_tool_calls(self, tool_calls: list[dict[str, Any]], *, purpose: str) -> None:
        for call in tool_calls or []:
            fn = call.get("function") or {}
            name = fn.get("name") or call.get("name")
            args = fn.get("arguments") or call.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"raw": args}
            if name == "ask_human":
                self.model_tools.execute("ask_human", args if isinstance(args, dict) else {"raw": args})
            # Other tools are advertised but intentionally not executed by the
            # legacy KimiClient JSON methods. New AgentTurnRunner executes them.

    def _record_legacy_conversation(self, *, purpose: str, system: str, user: str, content: str | None, reasoning_content: str | None = None, parsed_ok: bool | None = None, request: dict[str, Any] | None = None, error: str | None = None) -> None:
        if not self.recorder:
            return
        assistant = content
        metadata = {"reasoning_content": reasoning_content} if reasoning_content else {}
        self.recorder.record_call(
            purpose=purpose,
            system=system,
            user=user,
            assistant=assistant,
            error=error,
            parsed_ok=parsed_ok,
            request=request,
            metadata=metadata,
        )

    def _chat_json(self, system: str, user: str, *, purpose: str, max_tokens: int = 1200, temperature: float = 0.0) -> Any:
        last_exc: Exception | None = None
        for attempt_idx, actual_temperature in enumerate(self._temperature_attempts(temperature), start=1):
            request = {"temperature": actual_temperature, "max_completion_tokens": self._max_completion_tokens(max_tokens), "response_format": "jsonish", "attempt": attempt_idx, "tools_registered": True}
            try:
                result = self.chat_client.chat_result(
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=actual_temperature,
                    max_completion_tokens=self._max_completion_tokens(max_tokens),
                    tools=self.model_tools.openai_tools(),
                    tool_choice="auto",
                    extra_body=self._kimi_extra_body(),
                    purpose=purpose,
                    metadata={"legacy_kimi_client": True, "response_format": "jsonish"},
                )
                if result.tool_calls:
                    self._handle_tool_calls(result.tool_calls, purpose=purpose)
                data = parse_jsonish(result.content or "")
                self._record_legacy_conversation(purpose=purpose, system=system, user=user, content=result.content, reasoning_content=result.reasoning_content, parsed_ok=isinstance(data, (dict, list)) and bool(result.content), request=request)
                return data
            except Exception as exc:
                last_exc = exc
                self._record_legacy_conversation(purpose=purpose, system=system, user=user, content=None, error=f"{type(exc).__name__}: {exc}", parsed_ok=False, request=request)
                if self._temperature_retry_needed(exc, actual_temperature):
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def _chat_text(self, system: str, user: str, *, purpose: str, max_tokens: int = 1200, temperature: float = 0.0) -> str:
        last_exc: Exception | None = None
        for attempt_idx, actual_temperature in enumerate(self._temperature_attempts(temperature), start=1):
            request = {"temperature": actual_temperature, "max_completion_tokens": self._max_completion_tokens(max_tokens), "response_format": "text", "attempt": attempt_idx, "tools_registered": True}
            try:
                result = self.chat_client.chat_result(
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=actual_temperature,
                    max_completion_tokens=self._max_completion_tokens(max_tokens),
                    tools=self.model_tools.openai_tools(),
                    tool_choice="auto",
                    extra_body=self._kimi_extra_body(),
                    purpose=purpose,
                    metadata={"legacy_kimi_client": True, "response_format": "text"},
                )
                if result.tool_calls:
                    self._handle_tool_calls(result.tool_calls, purpose=purpose)
                self._record_legacy_conversation(purpose=purpose, system=system, user=user, content=result.content, reasoning_content=result.reasoning_content, parsed_ok=None, request=request)
                return result.content or ""
            except Exception as exc:
                last_exc = exc
                self._record_legacy_conversation(purpose=purpose, system=system, user=user, content=None, error=f"{type(exc).__name__}: {exc}", parsed_ok=False, request=request)
                if self._temperature_retry_needed(exc, actual_temperature):
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def generate_search_queries(self, goal: str, max_queries: int = 8) -> list[str]:
        system = "你是数据集检索专家。只输出 JSON 数组，不要解释。"
        user = f"""
用户想训练/改进一个 LLM。请为 Hugging Face datasets 搜索生成 {max_queries} 个短查询词。
要求：中英文都可以，覆盖任务领域、数据格式、训练类型，如 instruction、sft、preference、dpo、qa。
目标：{goal}
只输出 JSON 数组，例如 ["chinese legal qa", "法律 问答"]。
""".strip()
        data = self._chat_json(system, user, purpose="generate_search_queries", max_tokens=600, temperature=1.0)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()][:max_queries]
        if isinstance(data, dict) and isinstance(data.get("queries"), list):
            return [str(x).strip() for x in data["queries"] if str(x).strip()][:max_queries]
        return []

    def review_dataset_samples(
        self,
        goal: str,
        inspection: DatasetInspection,
        classification: DatasetClassification,
    ) -> dict[str, Any]:
        payload = {
            "goal": goal,
            "dataset_id": inspection.dataset_id,
            "columns": inspection.columns,
            "format_type": classification.format_type,
            "recommended_training": [t.value for t in classification.recommended_training],
            "sample_rows": compact_sample_rows(inspection.sample_rows, max_rows=6, max_chars_per_value=600),
        }
        system = "你是严谨的数据审阅员，负责判断 Hugging Face 数据集是否适合 LLM 后训练。只输出 JSON。"
        user = f"""
请审阅下面的数据集样本，输出严格 JSON：
{{
  "domain_match_score": 0到1之间的数字,
  "training_value_score": 0到1之间的数字,
  "quality_score": 0到1之间的数字,
  "recommended_action": "accept|review|reject",
  "best_training_types": ["sft"|"dpo"|"reward_model"|"kto"|"rlvr"|"continued_pretraining"|"unknown"],
  "notes": "一句话说明"
}}
数据：
{json.dumps(payload, ensure_ascii=False)}
""".strip()
        data = self._chat_json(system, user, purpose="review_dataset_samples", max_tokens=1200, temperature=1.0)
        return data if isinstance(data, dict) else {"raw": str(data)[:1000]}

    def decide_dataset_adoption(
        self,
        goal: str,
        overview: DatasetWebOverview | None,
        inspection: DatasetInspection,
        classification: DatasetClassification,
        trials: list[ModelTrial],
    ) -> dict[str, Any]:
        payload = {
            "goal": goal,
            "dataset_id": inspection.dataset_id,
            "hub_url": overview.hub_url if overview else None,
            "card_excerpt": (overview.card_excerpt[:2500] if overview and overview.card_excerpt else ""),
            "files": [f.path for f in (overview.files[:20] if overview else [])],
            "selected_config": overview.selected_config if overview else inspection.config_name,
            "selected_split": overview.selected_split if overview else inspection.split,
            "columns": inspection.columns,
            "format_type": classification.format_type,
            "rule_recommended_training": [t.value for t in classification.recommended_training],
            "examples": compact_sample_rows(inspection.sample_rows, max_rows=5, max_chars_per_value=500),
            "local_model_trials": [
                {
                    "row_index": t.row_index,
                    "prompt": t.prompt[:900],
                    "reference_answer": (t.reference_answer or "")[:900],
                    "model_response": (t.model_response or "")[:900],
                    "similarity_to_reference": t.similarity_to_reference,
                    "error": t.error,
                }
                for t in trials[:5]
            ],
        }
        system = "你是 LLM 后训练数据采集负责人。根据数据集页面、样本、本地模型回答，决定是否采纳数据。只输出 JSON。"
        user = f"""
请根据下面信息，决定这个 Hugging Face 数据集是否应该进入后续训练数据池，并判断训练类型。
重点：
1. 数据是否和目标任务匹配；
2. 数据格式是否可以转换成 SFT/DPO/RLVR/continued pretraining；
3. 本地 vLLM 模型在样本问题上的回答是否显示出训练需求；
4. 不要做安全审查，不需要讨论隐私/毒性等问题。

输出严格 JSON：
{{
  "action": "accept|review|reject",
  "final_score": 0到1之间的数字,
  "data_value_score": 0到1之间的数字,
  "model_gap_score": 0到1之间的数字,
  "training_types": ["sft"|"dpo"|"reward_model"|"kto"|"rlvr"|"continued_pretraining"|"unknown"],
  "reason": "一句话结论",
  "conversion_notes": "如何转换成训练数据"
}}

信息：
{json.dumps(payload, ensure_ascii=False)}
""".strip()
        data = self._chat_json(system, user, purpose="decide_dataset_adoption", max_tokens=1500, temperature=1.0)
        return data if isinstance(data, dict) else {"raw": str(data)[:1000]}


    def generate_eval_cases(self, goal: str, weakness_areas: list[str] | None = None, max_cases: int = 8) -> list[dict[str, Any]]:
        """Ask KIMI to create held-out sample tests for the current goal.

        The returned schema is intentionally the same as autopilot.eval.EvalCase
        so the eval layer can load it without depending on this client.
        """
        system = "你是严格的 LLM 评测集设计员。只输出 JSON，不要解释。"
        user = f"""
请为下面的模型训练目标生成 {max_cases} 条小型 held-out 测试样例，用来发现模型哪里不行。
目标：{goal}
已知薄弱点：{json.dumps(weakness_areas or [], ensure_ascii=False)}

输出严格 JSON：
{{
  "cases": [
    {{
      "id": "短 id",
      "prompt": "给模型的测试问题",
      "metric": "exact_match|contains|llm_judge|python_unit_tests",
      "expected": "可选。exact_match/contains 的期望值",
      "reference_answer": "可选。llm_judge 的参考答案",
      "tests": "可选。python_unit_tests 的 pytest/assert 片段",
      "tags": ["领域或能力标签"],
      "weakness_area": "这条测试覆盖的薄弱点"
    }}
  ]
}}
要求：
1. 对 coding 目标，优先给 python_unit_tests；
2. 对数学/可验证任务，优先 exact_match；
3. 对开放问答，使用 llm_judge 并给 reference_answer；
4. 不要把训练数据样本原文复用为测试题。
""".strip()
        data = self._chat_json(system, user, purpose="generate_eval_cases", max_tokens=2500, temperature=0.7)
        if isinstance(data, dict) and isinstance(data.get("cases"), list):
            return [x for x in data["cases"] if isinstance(x, dict)][:max_cases]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)][:max_cases]
        return []

    def judge_eval_answer(self, case: dict[str, Any], response: str) -> dict[str, Any]:
        """Score one eval answer using KIMI as a judge."""
        system = "你是严格但公平的模型评测 judge。只输出 JSON。"
        user = f"""
请根据测试用例评价模型回答。输出严格 JSON：
{{
  "passed": true/false,
  "score": 0到1之间的数字,
  "feedback": "简短说明哪里对/错",
  "weakness_area": "如果失败，指出薄弱点"
}}

测试用例：
{json.dumps(case, ensure_ascii=False)}

模型回答：
{response[:6000]}
""".strip()
        data = self._chat_json(system, user, purpose="judge_eval_answer", max_tokens=1200, temperature=0.0)
        return data if isinstance(data, dict) else {"passed": False, "score": 0.0, "feedback": str(data)[:1000]}

    def diagnose_eval_failures(
        self,
        goal: str,
        eval_report: dict[str, Any],
        target_metric: str,
        target_value: float,
    ) -> dict[str, Any]:
        """Diagnose failed eval cases and propose data/tool/RL remedies."""
        system = "你是 LLM post-training 诊断 agent。只输出 JSON。"
        user = f"""
训练目标：{goal}
目标指标：{target_metric}>={target_value}
当前评测结果：
{json.dumps(eval_report, ensure_ascii=False)[:12000]}

请诊断模型主要弱点，并给出下一轮补救方向。输出严格 JSON：
{{
  "weaknesses": [{{"area": "薄弱点", "evidence": "失败样例证据", "priority": 1到5}}],
  "data_queries": ["用于 Hugging Face/web search 的查询词"],
  "synthetic_test_requests": ["应该让 KIMI 继续生成哪些测试"],
  "tool_queries": ["需要搜索或添加哪些工具/benchmark/verifier"],
  "rl_verifier_queries": ["如果适合 RLVR，需要找哪些 verifier/test/reward"],
  "recommended_training": ["sft"|"dpo"|"kto"|"rlvr"|"continued_pretraining"],
  "notes": "一句话策略"
}}
""".strip()
        data = self._chat_json(system, user, purpose="diagnose_eval_failures", max_tokens=2500, temperature=0.4)
        return data if isinstance(data, dict) else {"raw": str(data)[:2000]}
    def generate_eval_samples(self, goal: str, failure_summary: dict[str, Any] | None = None, n: int = 5) -> list[dict[str, Any]]:
        payload = {"goal": goal, "failure_summary": failure_summary or {}, "n": n}
        system = "你是评测集设计专家。只输出 JSON 数组，不要解释。"
        user = f"""
请为下面的训练目标生成 {n} 条小而尖锐的评测样本，用于发现模型短板。
每条必须是 JSON object，字段包括：prompt, expected, tags, verifier, rubric。
verifier 可选 contains/exact_match/kimi_judge/python_unit_test。
如果是 coding 任务，优先生成可检查的 expected 或 unit test；如果是主观任务，verifier 用 kimi_judge。

信息：
{json.dumps(payload, ensure_ascii=False)}
""".strip()
        data = self._chat_json(system, user, purpose="generate_eval_samples", max_tokens=1800, temperature=1.0)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)][:n]
        if isinstance(data, dict) and isinstance(data.get("cases"), list):
            return [x for x in data["cases"] if isinstance(x, dict)][:n]
        return []

    def judge_eval_case(
        self,
        *,
        goal: str,
        prompt: str,
        expected: str | None,
        response: str,
        rubric: str | None = None,
    ) -> dict[str, Any]:
        payload = {"goal": goal, "prompt": prompt, "expected": expected, "response": response, "rubric": rubric}
        system = "你是严格的 LLM 评测裁判。只输出 JSON。"
        user = f"""
请按目标和 rubric 评估模型回答。输出严格 JSON：
{{"passed": true/false, "score": 0到1之间的数字, "reason": "一句话原因", "weakness_tags": ["..."]}}

信息：
{json.dumps(payload, ensure_ascii=False)}
""".strip()
        data = self._chat_json(system, user, purpose="judge_eval_case", max_tokens=900, temperature=0.0)
        return data if isinstance(data, dict) else {"raw": str(data)[:1000]}

    def plan_autonomous_actions(
        self,
        *,
        goal: str,
        phase: str,
        resources: dict[str, Any] | None = None,
        repo_snapshot: dict[str, Any] | None = None,
        evaluation: dict[str, Any] | None = None,
        diagnosis: dict[str, Any] | None = None,
        max_commands: int = 3,
        environments: list[dict[str, Any]] | dict[str, Any] | None = None,
        runtime_environments: list[dict[str, Any]] | dict[str, Any] | None = None,
        vllm_service_plan: dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ask KIMI to act as the high-level controller for the current loop."""
        envs = environments if environments is not None else runtime_environments
        payload = {
            "goal": goal,
            "phase": phase,
            "resources": resources or {},
            "environments": envs or [],
            "vllm_service_plan": vllm_service_plan or {},
            "repo_snapshot": repo_snapshot or {},
            "evaluation": evaluation or {},
            "diagnosis": diagnosis or {},
            "memory": memory or {},
            "max_commands": max_commands,
        }
        system = "你是这个 Autopilot 代码仓库的 autonomous post-training engineering agent。只输出严格 JSON。"
        user = f"""
你负责把训练目标转成持续的 agent loop。请根据计算资源、预装环境、vLLM 服务计划、仓库状态、Claude/PostTrainingAgent memory 和评测失败信号，决定下一步。
原则：
1. 不要依赖固定工作流；每一步都可以创建 subtask、运行 bash、向 human 提问、或更新 memory。
2. 资源由你分配：根据 nvidia-smi 和显存决定是否启动/停止/重启/保留 vLLM、用哪些 GPU、训练阶段用哪个环境。
3. 预装环境只是一组候选。不要默认激活；需要时在 command 中写 environment，例如训练用 llamaf、服务/评测用 vllm。
4. 如果拿不准，使用 ask_human，而不是瞎猜。
5. 可以改进 Autopilot 仓库本身，但命令要短小、可观测，优先查看/测试，再修改。
6. vLLM 由资源分配 plan 决定 action: keep/start/stop/restart；不要使用固定的“训练前一律停止 vLLM”策略。

输出严格 JSON：
{{
  "summary": "一句话策略",
  "resource_allocation": {{
    "notes": "资源分配理由",
    "gpu_plan": "如何使用 GPU/显存",
    "training_environment": "可选，环境名",
    "vllm_action": "keep|start|stop|restart",
    "environments": {{"sft": "可选环境名", "dpo": "可选环境名", "rlvr": "可选环境名"}}
  }},
  "tasks": [{{"name": "task 名", "objective": "目标", "priority": 1}}],
  "ask_human": [{{"name": "问题名", "question": "需要问 human 的问题", "context": "为什么不确定", "priority": "low|normal|high", "options": []}}],
  "commands": [
    {{
      "name": "命令名",
      "command": "bash 命令，不要包含危险的无限循环",
      "cwd": "repo|run|cwd",
      "timeout": 600,
      "environment": "可选，environments 中的环境名",
      "reason": "为什么运行"
    }}
  ],
  "memory_updates": ["应该写入 PostTrainingAgent.md 的稳定经验"],
  "write_to_claude_memory": false,
  "notes": "给父 loop 的压缩说明"
}}
最多给 {max_commands} 条 commands。没有必要运行命令时 commands 为空数组。

上下文：
{json.dumps(payload, ensure_ascii=False)[:40000]}
""".strip()
        data = self._chat_json(system, user, purpose="plan_autonomous_actions", max_tokens=4500, temperature=0.3)
        if not isinstance(data, dict):
            return {"summary": str(data)[:1000], "tasks": [], "commands": [], "ask_human": []}
        commands = data.get("commands")
        if not isinstance(commands, list):
            data["commands"] = []
        else:
            data["commands"] = [x for x in commands if isinstance(x, dict)][:max_commands]
        for key in ["tasks", "ask_human", "memory_updates"]:
            if not isinstance(data.get(key), list):
                data[key] = []
        if not isinstance(data.get("resource_allocation"), dict):
            data["resource_allocation"] = {}
        return data

    def plan_resource_allocation(
        self,
        *,
        goal: str,
        phase: str,
        stage: str = "sft",
        resources: dict[str, Any] | None = None,
        environments: list[dict[str, Any]] | dict[str, Any] | None = None,
        vllm_service_plan: dict[str, Any] | None = None,
        vllm_plan: dict[str, Any] | None = None,
        vllm_status: dict[str, Any] | None = None,
        evaluation: dict[str, Any] | None = None,
        diagnosis: dict[str, Any] | None = None,
        tools: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "goal": goal,
            "phase": phase,
            "stage": stage,
            "resources": resources or {},
            "environments": environments or [],
            "vllm_service_plan": vllm_service_plan or vllm_plan or {},
            "vllm_status": vllm_status or {},
            "evaluation": evaluation or {},
            "diagnosis": diagnosis or {},
            "tools": tools or {},
        }
        system = "你是训练资源调度 agent。只输出严格 JSON。"
        user = f"""
请根据计算资源、已安装环境、vLLM 服务计划、工具状态和当前训练/评测阶段，决定下一步资源分配。不要使用固定规则；如果信息不足，用 ask_human。

输出严格 JSON：
{{
  "summary": "一句话资源策略",
  "stage": "sft|dpo|kto|pt|rlvr|eval|unknown",
  "training_environment": "可选环境名，例如 llamaf 或 vllm；没有必要则为 null",
  "activation_command": null,
  "vllm_action": "keep|start|stop|restart",
  "gpu_allocation": {{"CUDA_VISIBLE_DEVICES": "可选，例如 0,1", "reason": "原因"}},
  "pre_training_commands": [
    {{"name": "命令名", "command": "可选 bash 命令", "timeout": 600, "reason": "为什么执行"}}
  ],
  "ask_human": [
    {{"question": "需要用户判断的问题", "context": "为什么需要问", "urgency": "low|normal|high", "options": []}}
  ],
  "notes": "为什么这样分配"
}}

注意：
- 环境只是候选；只有 training_environment 或 activation_command 被明确选中时才激活。
- vLLM 不按固定开关处理；按当前显存、阶段、vLLM endpoint 可达性和服务计划选择 keep/start/stop/restart。
- 如果 eval/probe 需要本地模型、vLLM endpoint 不可达、且 vllm_service_plan.enabled=true，通常应选择 vllm_action=start，并选择合适的 vllm/serving 环境。
- SFT/DPO/KTO/PT 通常需要训练环境 llamaf；eval/probe/serving 通常需要 vllm 环境和 vLLM 服务，但是否启动由你判断。
- 不要因为信息已经在 environments / compute_resources 中明示而 ask_human；只有缺少关键事实或有多个高风险选择时才问人。

上下文：
{json.dumps(payload, ensure_ascii=False)[:30000]}
""".strip()
        data = self._chat_json(system, user, purpose="plan_resource_allocation", max_tokens=2800, temperature=0.2)
        if not isinstance(data, dict):
            return {"summary": str(data)[:1000], "stage": stage, "vllm_action": "keep", "pre_training_commands": [], "ask_human": []}
        if not isinstance(data.get("pre_training_commands"), list):
            data["pre_training_commands"] = []
        if not isinstance(data.get("ask_human"), list):
            data["ask_human"] = []
        if str(data.get("vllm_action") or "").lower() not in {"keep", "start", "stop", "restart", "kill"}:
            data["vllm_action"] = "keep"
        if not data.get("stage"):
            data["stage"] = stage
        return data

    def summarize_post_training_experience(
        self,
        *,
        goal: str,
        phase: str,
        resources: dict[str, Any] | None = None,
        resource_plan: dict[str, Any] | None = None,
        evaluation: dict[str, Any] | None = None,
        diagnosis: dict[str, Any] | None = None,
        commands: list[str] | None = None,
    ) -> str:
        payload = {
            "goal": goal,
            "phase": phase,
            "resources": resources or {},
            "resource_plan": resource_plan or {},
            "evaluation": evaluation or {},
            "diagnosis": diagnosis or {},
            "commands": (commands or [])[-10:],
        }
        system = "你是 PostTrainingAgent 的长期记忆整理员。只输出 Markdown 片段，不要寒暄。"
        user = f"""
请把这轮交互/训练循环中值得长期保留的经验总结成 3-8 条 Markdown bullet。
重点记录：资源分配经验、环境选择经验、vLLM/训练互斥经验、失败原因、下一轮应避免的坑。
不要记录 API key；路径可只保留必要前缀。

上下文：
{json.dumps(payload, ensure_ascii=False)[:30000]}
""".strip()
        return self._chat_text(system, user, purpose="summarize_post_training_experience", max_tokens=1500, temperature=0.2).strip()
