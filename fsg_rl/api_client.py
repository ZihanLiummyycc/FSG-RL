"""Small, provider-neutral clients for teacher and optional external services."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class APIConfigurationError(ValueError):
    """Raised when a required remote model setting is missing."""


@dataclass(frozen=True)
class ChatAPIConfig:
    model: str
    api_key_env: str
    base_url: Optional[str] = None
    timeout_seconds: float = 120.0
    max_retries: int = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any], role: str) -> "ChatAPIConfig":
        model = str(data.get("model", "")).strip()
        api_key_env = str(data.get("api_key_env", "")).strip()
        base_url = str(data.get("base_url", "")).strip().rstrip("/")
        if not model:
            raise APIConfigurationError(f"{role}.model is required")
        if not api_key_env:
            raise APIConfigurationError(f"{role}.api_key_env is required")
        if not base_url:
            raise APIConfigurationError(
                f"{role}.base_url is required for the OpenAI-compatible endpoint"
            )
        return cls(
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            timeout_seconds=float(data.get("timeout_seconds", 120.0)),
            max_retries=int(data.get("max_retries", 3)),
        )


class OpenAICompatibleChatClient:
    """Calls an OpenAI-compatible chat-completions endpoint.

    The API key is resolved only at call time and is never stored in configs,
    summaries, or exception messages.
    """

    def __init__(self, config: ChatAPIConfig):
        self.config = config
        self._client: Any = None

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_response: bool = False,
    ) -> str:
        client = self._get_client()
        request = build_chat_completion_request(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_response:
            request["response_format"] = {"type": "json_object"}

        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:
            if json_response and _response_format_may_be_unsupported(exc):
                request.pop("response_format", None)
                response = client.chat.completions.create(**request)
            else:
                raise RuntimeError(
                    f"Chat API request failed for model {self.config.model!r}: "
                    f"{type(exc).__name__}"
                ) from exc

        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Chat API model {self.config.model!r} returned empty content")
        return content.strip()

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        content = self.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_response=True,
        )
        return extract_json_object(content)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise APIConfigurationError(
                f"Missing environment variable {self.config.api_key_env!r} for "
                f"model {self.config.model!r}"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to use remote model APIs") from exc

        kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.config.timeout_seconds,
            "max_retries": self.config.max_retries,
        }
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = OpenAI(**kwargs)
        return self._client


def extract_json_object(content: str) -> Dict[str, Any]:
    """Parse JSON returned directly or wrapped in a Markdown code fence."""

    stripped = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response did not contain a JSON object")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model response JSON must be an object")
    return value


def _response_format_may_be_unsupported(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format" in message or "json_object" in message


def build_chat_completion_request(
    *,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Use parameter names accepted by GPT-5-compatible gateways."""

    request: Dict[str, Any] = {"model": model, "messages": messages}
    if model.lower().startswith("gpt-5"):
        request["max_completion_tokens"] = max_tokens
    else:
        request["temperature"] = temperature
        request["max_tokens"] = max_tokens
    return request
