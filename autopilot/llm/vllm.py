from __future__ import annotations

from autopilot.config import Settings
from autopilot.llm.openai_compatible import OpenAICompatibleChatClient
class VLLMClient(OpenAICompatibleChatClient):
    @classmethod
    def from_settings(cls, settings: Settings) -> "VLLMClient":
        if not settings.vllm_base_url or not settings.vllm_model:
            raise ValueError("VLLM_BASE_URL and VLLM_MODEL must be configured.")
        return cls(
            api_key=settings.vllm_api_key or "EMPTY",
            base_url=settings.vllm_base_url,
            model=settings.vllm_model,
            timeout=120.0,
            provider_name="local_vllm",
            client_name="local_vllm",
            trajectory_recorder=None,
            auto_trajectory=False,
        )
