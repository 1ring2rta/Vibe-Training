from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import requests

try:  # The SDK is convenient when installed, but the project should also work with plain HTTP.
    from openai import OpenAI as _OpenAI
except Exception:  # pragma: no cover - depends on local environment
    _OpenAI = None  # type: ignore[assignment]


@dataclass
class ChatCompletionResult:
    content: str = ""
    reasoning_content: str | None = None
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Any | None = None
    request_payload: dict[str, Any] = field(default_factory=dict)
    response_payload: dict[str, Any] = field(default_factory=dict)


def _error_hint(base_url: str, endpoint: str | None = None) -> str:
    hints: list[str] = []
    if base_url.endswith("/chat/completions"):
        hints.append("base_url 应该只到 /v1，不要包含 /chat/completions。")
    if not base_url.rstrip("/").endswith("/v1") and not base_url.rstrip("/").endswith("/api"):
        hints.append("OpenAI-compatible base_url 通常应该以 /v1 结尾。")
    if endpoint and endpoint.endswith("/responses"):
        hints.append("当前使用 Chat Completions；不要调用 /v1/responses。")
    return " ".join(hints)


def _format_http_error(provider: str, base_url: str, endpoint: str, exc: Exception, response_text: str | None = None) -> RuntimeError:
    msg = f"{provider} request failed for {endpoint}: {exc}"
    hint = _error_hint(base_url, endpoint)
    if response_text:
        trimmed = response_text.strip().replace("\n", " ")[:500]
        if trimmed:
            msg += f"; response={trimmed}"
    if hint:
        msg += f" Hint: {hint}"
    return RuntimeError(msg)


def _message_to_dict(message: Any) -> dict[str, Any]:
    if message is None:
        return {}
    if isinstance(message, dict):
        return dict(message)
    if hasattr(message, "model_dump"):
        try:
            return dict(message.model_dump())
        except Exception:
            pass
    out: dict[str, Any] = {}
    for key in ["role", "content", "reasoning_content", "tool_calls", "refusal", "audio"]:
        if hasattr(message, key):
            value = getattr(message, key)
            if value is not None:
                out[key] = value
    return out


def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        return dict(tool_call)
    if hasattr(tool_call, "model_dump"):
        try:
            return dict(tool_call.model_dump())
        except Exception:
            pass
    data: dict[str, Any] = {}
    for key in ["id", "type", "function"]:
        if hasattr(tool_call, key):
            data[key] = getattr(tool_call, key)
    if "function" in data and not isinstance(data["function"], dict):
        fn = data["function"]
        data["function"] = {"name": getattr(fn, "name", None), "arguments": getattr(fn, "arguments", None)}
    return data


def _response_to_jsonable(response: Any) -> Any:
    if response is None:
        return None
    if isinstance(response, (dict, list, str, int, float, bool)):
        return response
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump()
        except Exception:
            pass
    if hasattr(response, "to_dict"):
        try:
            return response.to_dict()
        except Exception:
            pass
    return repr(response)




def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _default_frontier_recorder() -> Any | None:
    """Create a recorder for direct OpenAICompatibleChatClient use.

    This is the last-resort guardrail that makes `message -> completion`
    collection default-on even when a caller forgot to inject a recorder.
    Set AUTOPILOT_DISABLE_FRONTIER_TRAJECTORY=1 only for explicit opt-out.
    """
    if _env_bool("AUTOPILOT_DISABLE_FRONTIER_TRAJECTORY", default=False):
        return None
    try:
        from pathlib import Path
        from autopilot.runtime.trajectory import FrontierTrajectoryRecorder
        root = os.getenv("AUTOPILOT_FRONTIER_TRAJECTORY_ROOT") or ".autopilot/frontier_trajectory"
        return FrontierTrajectoryRecorder(root=Path(root))
    except Exception:
        return None




_SDK_CHAT_COMPLETIONS_KWARGS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "tools",
    "tool_choice",
    "response_format",
    "stream",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "n",
    "user",
    "metadata",
    "extra_headers",
    "extra_query",
    "extra_body",
    "timeout",
}


def _sdk_kwargs_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate wire JSON payload into kwargs accepted by OpenAI SDK.

    OpenAI-compatible providers often support vendor-specific request fields
    such as Kimi's `thinking`. These fields belong in the HTTP JSON body,
    but the Python SDK does not accept them as direct keyword arguments.
    Put unknown body fields under `extra_body` for SDK calls while keeping
    the original `payload` unchanged for trajectory logging and requests fallback.
    """
    kwargs: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}

    for key, value in payload.items():
        if value is None:
            continue

        if key == "extra_body" and isinstance(value, dict):
            extra_body.update(value)
        elif key in _SDK_CHAT_COMPLETIONS_KWARGS:
            kwargs[key] = value
        else:
            extra_body[key] = value

    if extra_body:
        existing = kwargs.get("extra_body")
        if isinstance(existing, dict):
            merged = dict(existing)
            merged.update(extra_body)
            kwargs["extra_body"] = merged
        else:
            kwargs["extra_body"] = extra_body

    return kwargs


_RETRYABLE_UNSUPPORTED_PARAMS = {
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "response_format",
    "tool_choice",
}


def _unsupported_request_param(exc: Exception, payload: Mapping[str, Any]) -> str | None:
    """Return a retryable request parameter rejected by an OpenAI-compatible gateway.

    LiteLLM/Azure style errors often look like:
    ``Unsupported parameter: 'top_p' is not supported with this model``.
    Some gateways instead expose ``param: top_p``.  We only retry by removing
    optional generation/request-shaping parameters, never by removing the model,
    messages, or actual tool definitions.
    """
    text = str(exc)
    lowered = text.lower()
    if "unsupported parameter" not in lowered and "unsupported_param" not in lowered:
        return None

    patterns = [
        r"Unsupported parameter:\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?",
        r"param['\"]?\s*[:=]\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            candidate = match.group(1)
            if candidate in payload and candidate in _RETRYABLE_UNSUPPORTED_PARAMS:
                return candidate

    for candidate in _RETRYABLE_UNSUPPORTED_PARAMS:
        if candidate in payload and re.search(rf"\b{re.escape(candidate)}\b", text):
            return candidate
    return None


class OpenAICompatibleChatClient:
    """Minimal wrapper for OpenAI-compatible chat endpoints.

    The wrapper now has a first-class ``chat_result`` method that preserves
    content, reasoning_content, tool calls, finish_reason, raw response, request
    payload, and response payload.  ``chat`` remains backward compatible and
    returns only the final content string.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        provider_name: str = "OpenAI-compatible",
        *,
        trajectory_recorder: Any | None = None,
        client_name: str | None = None,
        default_params: Mapping[str, Any] | None = None,
        auto_trajectory: bool = True,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url[: -len("/chat/completions")].rstrip("/")
        self.model = model
        self.timeout = timeout
        self.provider_name = provider_name
        self.client_name = client_name or provider_name.lower().replace(" ", "_")
        self.default_params = dict(default_params or {})
        self.trajectory_recorder = trajectory_recorder if trajectory_recorder is not None else (_default_frontier_recorder() if auto_trajectory else None)
        self._sdk_client = _OpenAI(api_key=api_key, base_url=self.base_url, timeout=timeout) if _OpenAI is not None else None
        self._session = requests.Session()
        self.last_result: ChatCompletionResult | None = None

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/models"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_models(self) -> dict[str, Any]:
        response = self._session.get(self.models_url, headers=self._headers(), timeout=self.timeout)
        if response.status_code >= 400:
            raise _format_http_error(self.provider_name, self.base_url, self.models_url, requests.HTTPError(f"{response.status_code} {response.reason}"), response.text)
        try:
            data = response.json()
        except Exception as exc:
            raise ValueError(f"Invalid /models response from {self.models_url}: {response.text[:500]!r}") from exc
        return data if isinstance(data, dict) else {"raw": data}

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params = dict(self.default_params)
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if max_completion_tokens is not None:
            params["max_completion_tokens"] = max_completion_tokens
        elif max_tokens is not None:
            # KIMI accepts both in current tests, but model-specific configs can
            # set max_completion_tokens.  Keep legacy max_tokens for vLLM.
            params.setdefault("max_tokens", max_tokens)
        if response_format is not None:
            params["response_format"] = response_format
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        payload.update({k: v for k, v in params.items() if v is not None})
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if extra_body:
            payload.update(extra_body)
        return payload

    def _send_payload(self, payload: dict[str, Any]) -> ChatCompletionResult:
        if self._sdk_client is not None:
            kwargs = _sdk_kwargs_from_payload(payload)
            response = self._sdk_client.chat.completions.create(**kwargs)
            raw = _response_to_jsonable(response)
            choice = response.choices[0]
            msg_dict = _message_to_dict(choice.message)
            tool_calls = [_tool_call_to_dict(x) for x in (getattr(choice.message, "tool_calls", None) or [])]
            return ChatCompletionResult(
                content=msg_dict.get("content") or "",
                reasoning_content=msg_dict.get("reasoning_content"),
                finish_reason=getattr(choice, "finish_reason", None),
                tool_calls=tool_calls,
                raw=raw,
                request_payload=payload,
                response_payload={"message": msg_dict, "finish_reason": getattr(choice, "finish_reason", None), "tool_calls": tool_calls},
            )

        response = self._session.post(self.chat_completions_url, headers=self._headers(), json=payload, timeout=self.timeout)
        if response.status_code >= 400:
            raise _format_http_error(self.provider_name, self.base_url, self.chat_completions_url, requests.HTTPError(f"{response.status_code} {response.reason}"), response.text)
        data = response.json()
        try:
            choice = data["choices"][0]
            msg = dict(choice.get("message") or {})
        except Exception as exc:
            raise ValueError(f"Invalid OpenAI-compatible chat response: {data!r}") from exc
        tool_calls = [dict(x) for x in (msg.get("tool_calls") or []) if isinstance(x, dict)]
        return ChatCompletionResult(
            content=msg.get("content") or "",
            reasoning_content=msg.get("reasoning_content"),
            finish_reason=choice.get("finish_reason"),
            tool_calls=tool_calls,
            raw=data,
            request_payload=payload,
            response_payload={"message": msg, "finish_reason": choice.get("finish_reason"), "tool_calls": tool_calls},
        )

    def chat_result(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = 0.0,
        max_tokens: int | None = 1024,
        *,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        extra_body: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        purpose: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ChatCompletionResult:
        payload = self._build_payload(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            extra_body=extra_body,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )
        retry_payload = payload
        attempted_removed_params: set[str] = set()

        while True:
            handle = None
            if self.trajectory_recorder is not None:
                handle = self.trajectory_recorder.begin_call(
                    provider=self.provider_name,
                    client_name=self.client_name,
                    model=self.model,
                    purpose=purpose,
                    request_payload=retry_payload,
                    tools=tools,
                    metadata=dict(metadata or {}),
                )
            try:
                result = self._send_payload(retry_payload)
                self.last_result = result
                if self.trajectory_recorder is not None and handle is not None:
                    self.trajectory_recorder.end_call(
                        handle=handle,
                        provider=self.provider_name,
                        client_name=self.client_name,
                        model=self.model,
                        purpose=purpose,
                        request_payload=retry_payload,
                        response={
                            "content": result.content,
                            "reasoning_content": result.reasoning_content,
                            "finish_reason": result.finish_reason,
                            "tool_calls": result.tool_calls,
                        },
                        raw_response=result.raw,
                        metadata=dict(metadata or {}),
                    )
                return result
            except Exception as exc:
                if self.trajectory_recorder is not None and handle is not None:
                    self.trajectory_recorder.error_call(
                        handle=handle,
                        provider=self.provider_name,
                        client_name=self.client_name,
                        model=self.model,
                        purpose=purpose,
                        request_payload=retry_payload,
                        error=f"{type(exc).__name__}: {exc}",
                        metadata=dict(metadata or {}),
                    )
                unsupported = _unsupported_request_param(exc, retry_payload)
                if unsupported and unsupported not in attempted_removed_params:
                    attempted_removed_params.add(unsupported)
                    retry_payload = dict(retry_payload)
                    retry_payload.pop(unsupported, None)
                    continue
                raise

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        result = self.chat_result(messages=messages, temperature=temperature, max_tokens=max_tokens, extra_body=extra_body, **kwargs)
        return result.content or ""

    def chat_json(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 1200) -> Any:
        content = self.chat(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=temperature, max_tokens=max_tokens)
        return parse_jsonish(content)


def parse_jsonish(content: str) -> Any:
    content = (content or "").strip()
    try:
        return json.loads(content)
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.S | re.I)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = content.find(start_char)
        end = content.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except Exception:
                continue
    return {"raw": content}
