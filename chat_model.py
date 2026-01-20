"""SolarChatModel - LangChain ChatModel wrapper for Upstage Solar API"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any, Literal

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

API_URL = "https://solar-chat.upstage.ai/api/chat"


class SolarChatModel(BaseChatModel):
    """LangChain ChatModel wrapper for Upstage Solar Chat API.

    Example:
        >>> from solar_langchain import SolarChatModel
        >>> from langchain_core.messages import HumanMessage
        >>> chat = SolarChatModel(reasoning_effort="medium")
        >>> chat.invoke([HumanMessage(content="Hello!")])
    """

    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    """Reasoning effort level for the model."""

    timeout: float = 120.0
    """Request timeout in seconds."""

    streaming: bool = False
    """Whether to enable streaming by default."""

    @property
    def _llm_type(self) -> str:
        return "solar-chat"

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

    def _convert_message_to_solar(self, message: BaseMessage) -> dict[str, Any]:
        """Convert LangChain message to Solar API format."""
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = "user"

        content = message.content
        if isinstance(content, list):
            # Handle multi-part content
            text_parts = [
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            ]
            content = "".join(text_parts)

        return {
            "id": self._get_timestamp_id(),
            "role": role,
            "parts": [{"type": "text", "text": content}],
        }

    def _build_payload(self, messages: list[BaseMessage]) -> dict[str, Any]:
        """Build API request payload from LangChain messages."""
        solar_messages = [self._convert_message_to_solar(msg) for msg in messages]
        return {
            "messages": solar_messages,
            "reasoning_effort": self.reasoning_effort,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion from Solar API.

        Args:
            messages: List of messages to send.
            stop: Stop sequences (not supported by Solar API).
            run_manager: Callback manager for the run.
            **kwargs: Additional keyword arguments.

        Returns:
            ChatResult containing the generated response.
        """
        payload = self._build_payload(messages)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(API_URL, json=payload)
            response.raise_for_status()

            content_parts: list[str] = []
            for line in response.text.split("\n"):
                content, _ = self._parse_sse_line(line)
                if content:
                    content_parts.append(content)

            response_text = "".join(content_parts)
            message = AIMessage(content=response_text)

            return ChatResult(
                generations=[ChatGeneration(message=message)],
                llm_output={"reasoning_effort": self.reasoning_effort},
            )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completion from Solar API.

        Args:
            messages: List of messages to send.
            stop: Stop sequences (not supported by Solar API).
            run_manager: Callback manager for the run.
            **kwargs: Additional keyword arguments.

        Yields:
            ChatGenerationChunk objects containing streamed responses.
        """
        payload = self._build_payload(messages)

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", API_URL, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    content, _ = self._parse_sse_line(line)
                    if content:
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=content)
                        )
                        if run_manager:
                            run_manager.on_llm_new_token(content, chunk=chunk)
                        yield chunk
