"""SolarLLM - LangChain LLM wrapper for Upstage Solar API"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any, Literal

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

API_URL = "https://solar-chat.upstage.ai/api/chat"


class SolarLLM(LLM):
    """LangChain LLM wrapper for Upstage Solar Chat API.

    Example:
        >>> from solar_langchain import SolarLLM
        >>> llm = SolarLLM(reasoning_effort="medium")
        >>> llm.invoke("Hello, how are you?")
    """

    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    """Reasoning effort level for the model."""

    timeout: float = 120.0
    """Request timeout in seconds."""

    streaming: bool = False
    """Whether to enable streaming by default."""

    @property
    def _llm_type(self) -> str:
        return "solar"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "reasoning_effort": self.reasoning_effort,
            "timeout": self.timeout,
        }

    def _get_timestamp_id(self) -> str:
        """Generate timestamp-based message ID."""
        return str(int(time.time() * 1000))

    def _parse_sse_line(self, line: str) -> tuple[str | None, str | None]:
        """Parse SSE line to extract content and reasoning_content."""
        line = line.strip()
        if not line.startswith("data: "):
            return None, None

        data = line[6:]
        if data == "[DONE]":
            return None, None

        try:
            obj = json.loads(data)
            delta = obj.get("choices", [{}])[0].get("delta", {})
            return delta.get("content"), delta.get("reasoning_content")
        except json.JSONDecodeError:
            return None, None

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        """Build API request payload."""
        return {
            "messages": [
                {
                    "id": self._get_timestamp_id(),
                    "role": "user",
                    "parts": [{"type": "text", "text": prompt}],
                }
            ],
            "reasoning_effort": self.reasoning_effort,
        }

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute synchronous call to Solar API.

        Args:
            prompt: The prompt to send to the model.
            stop: Stop sequences (not supported by Solar API).
            run_manager: Callback manager for the run.
            **kwargs: Additional keyword arguments.

        Returns:
            The generated text response.
        """
        payload = self._build_payload(prompt)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(API_URL, json=payload)
            response.raise_for_status()

            content_parts: list[str] = []
            for line in response.text.split("\n"):
                content, _ = self._parse_sse_line(line)
                if content:
                    content_parts.append(content)

            return "".join(content_parts)

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Execute streaming call to Solar API.

        Args:
            prompt: The prompt to send to the model.
            stop: Stop sequences (not supported by Solar API).
            run_manager: Callback manager for the run.
            **kwargs: Additional keyword arguments.

        Yields:
            GenerationChunk objects containing streamed text.
        """
        payload = self._build_payload(prompt)

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", API_URL, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    content, _ = self._parse_sse_line(line)
                    if content:
                        chunk = GenerationChunk(text=content)
                        if run_manager:
                            run_manager.on_llm_new_token(content, chunk=chunk)
                        yield chunk
