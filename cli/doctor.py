from __future__ import annotations

import argparse
import sys
from typing import Any

from autopilot.config import load_settings, validate_settings
from autopilot.llm.kimi import KimiClient
from autopilot.llm.vllm import VLLMClient
from autopilot.runtime.clients import LLMClientRegistry
from autopilot.tools.web_search import WebSearchTool, extract_hf_dataset_ids
from autopilot.tools.resources import inspect_compute_resources


def _mask(value: str | None) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "***"
    return value[:4] + "..." + value[-4:]


def _model_ids(data: dict[str, Any], limit: int = 12) -> list[str]:
    items = data.get("data")
    ids: list[str] = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
            elif isinstance(item, str):
                ids.append(item)
            if len(ids) >= limit:
                break
    return ids


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check YAML/env configuration, generic LLM clients, and runtime resources.")
    parser.add_argument("--config", default=None, help="YAML config path.")
    parser.add_argument("--env-file", default=None, help="Optional .env file path.")
    parser.add_argument("--check-client", default=None, help="Call /models for a configured generic client or role, e.g. teacher/director/local_vllm.")
    parser.add_argument("--check-role", default=None, help="Alias for --check-client with a role name, e.g. director/judge/local_probe.")
    parser.add_argument("--check-kimi", action="store_true", help="Legacy: call KIMI /models and a tiny chat completion.")
    parser.add_argument("--check-vllm", action="store_true", help="Legacy: call vLLM /models and a tiny chat completion.")
    parser.add_argument("--check-web-search", action="store_true", help="Call the configured web search provider once and extract HF dataset URLs.")
    parser.add_argument("--check-resources", action="store_true", help="Run nvidia-smi/resource inspection and print GPU/CPU summary.")
    parser.add_argument("--web-query", default="site:huggingface.co/datasets code instruction", help="Query used by --check-web-search.")
    parser.add_argument("--chat", action="store_true", help="Also run a tiny chat request. By default only /models is checked.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds for checks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    settings = load_settings(config_file=args.config, env_file=args.env_file)

    print("Configuration")
    print(f"  config: {settings.config_path or '<none>'}")
    print(f"  hf_token: {_mask(settings.hf_token)}")
    print(f"  hf_endpoint: {settings.hf_endpoint or '<official>'}")
    registry = LLMClientRegistry.from_settings(settings)
    print("  clients:")
    for name, spec in registry.specs.items():
        print(f"    - {name}: type={spec.type} model={spec.model or '<empty>'} base_url={spec.base_url or '<empty>'} record_trajectory={spec.record_trajectory}")
    print(f"  roles: {registry.roles}")
    print(f"  aliases: {registry.aliases}")
    print("  legacy compatibility:")
    print(f"    kimi.base_url: {settings.kimi_base_url}")
    print(f"    kimi.model: {settings.kimi_model}")
    print(f"    kimi.api_key: {_mask(settings.kimi_api_key)}")
    print(f"    vllm.base_url: {settings.vllm_base_url or '<empty>'}")
    print(f"    vllm.model: {settings.vllm_model or '<empty>'}")
    print(f"    vllm.api_key: {_mask(settings.vllm_api_key)}")
    print(f"  web_search.provider: {settings.web_search_provider or '<empty>'}")
    print(f"  web_search.serper_api_key: {_mask(settings.serper_api_key)}")
    print(f"  web_search.brave_api_key: {_mask(settings.brave_api_key)}")
    print(f"  web_search.tavily_api_key: {_mask(settings.tavily_api_key)}")
    print(f"  web_search.bocha_api_key: {_mask(settings.bocha_api_key)}")
    print(f"  web_search.bocha_endpoint: {settings.bocha_endpoint}")

    warnings = validate_settings(settings)
    if warnings:
        print("\nWarnings")
        for warning in warnings:
            print(f"  - {warning}")

    rc = 0

    check_client_name = args.check_client or args.check_role
    if check_client_name:
        print(f"\nGeneric client check: {check_client_name}")
        try:
            client = registry.get(str(check_client_name))
            client.timeout = args.timeout
            data = client.list_models()
            ids = _model_ids(data)
            print(f"  [ok] GET {client.models_url}")
            if ids:
                print(f"  models: {', '.join(ids)}")
            if args.chat:
                text = client.chat(messages=[{"role": "user", "content": "Return only: pong"}], temperature=0, max_tokens=20)
                print(f"  [ok] chat response: {text[:120]!r}")
        except Exception as exc:
            print(f"  [fail] {exc}")
            rc = 2

    if args.check_kimi:
        print("\nKIMI check")
        if not settings.kimi_configured:
            print("  [fail] KIMI is not fully configured.")
            rc = 2
        else:
            try:
                kimi = KimiClient(settings)
                kimi.chat_client.timeout = args.timeout
                data = kimi.chat_client.list_models()
                ids = _model_ids(data)
                print(f"  [ok] GET {kimi.chat_client.models_url}")
                if ids:
                    print(f"  models: {', '.join(ids)}")
                if settings.kimi_model and ids and settings.kimi_model not in ids:
                    print(f"  [warn] configured model {settings.kimi_model!r} was not in first {len(ids)} listed models")
                if args.chat:
                    text = kimi.chat_client.chat(
                        messages=[{"role": "user", "content": "只回答 pong"}],
                        temperature=0,
                        max_tokens=20,
                    )
                    print(f"  [ok] chat response: {text[:120]!r}")
            except Exception as exc:
                print(f"  [fail] {exc}")
                rc = 2

    if args.check_vllm:
        print("\nvLLM check")
        if not settings.vllm_configured:
            print("  [fail] vLLM is not fully configured.")
            rc = 2
        else:
            try:
                vllm = VLLMClient.from_settings(settings)
                vllm.timeout = args.timeout
                data = vllm.list_models()
                ids = _model_ids(data)
                print(f"  [ok] GET {vllm.models_url}")
                if ids:
                    print(f"  models: {', '.join(ids)}")
                if args.chat:
                    text = vllm.chat(
                        messages=[{"role": "user", "content": "Return only: pong"}],
                        temperature=0,
                        max_tokens=20,
                    )
                    print(f"  [ok] chat response: {text[:120]!r}")
            except Exception as exc:
                print(f"  [fail] {exc}")
                rc = 2


    if args.check_resources:
        print("\nCompute resource check")
        resources = inspect_compute_resources(timeout=args.timeout)
        data = resources.to_dict()
        print(f"  hostname: {data['hostname']}")
        print(f"  cpu_count: {data['cpu_count']}")
        print(f"  memory_total_mb: {data['memory_total_mb']}")
        print(f"  gpu_count: {data['gpu_count']}")
        for gpu in data.get("gpus", []):
            print(f"  - GPU {gpu.get('index')}: {gpu.get('name')} total={gpu.get('memory_total_mb')}MB free={gpu.get('memory_free_mb')}MB util={gpu.get('utilization_gpu_percent')}%")
        if data.get("error"):
            print(f"  [warn] {data['error']}")

    if args.check_web_search:
        print("\nWeb search check")
        try:
            tool = WebSearchTool(settings, timeout=args.timeout)
            hits = tool.search(args.web_query, limit=5)
            ids = extract_hf_dataset_ids(hits)
            print(f"  [ok] provider={settings.web_search_provider}; hits={len(hits)}; hf_dataset_ids={len(ids)}")
            for hit in hits[:5]:
                print(f"  - [{hit.source}] {hit.title[:80]} -> {hit.url}")
            if ids:
                print(f"  extracted: {', '.join(ids[:10])}")
        except Exception as exc:
            print(f"  [fail] {exc}")
            rc = 2

    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
