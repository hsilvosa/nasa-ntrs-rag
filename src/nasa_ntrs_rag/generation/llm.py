"""Pluggable LLM clients supporting local GGUFs (llama-server/Ollama), OpenAI-compatible APIs, and Gemini."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from openai import OpenAI

from nasa_ntrs_rag.config import Settings, get_settings


class BaseLLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]: ...


class OpenAICompatibleLLM:
    """Client for llama-server (D:\\models GGUFs), Ollama, OpenAI, or Groq."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


class GeminiLLM:
    """Client for Google Gemini API via google-genai."""

    def __init__(self, settings: Settings) -> None:
        from google import genai  # type: ignore[import-not-found]
        from google.genai import types  # type: ignore[import-not-found]

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be set when LLM_PROVIDER=gemini")
        self.settings = settings
        self._types = types
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.models.generate_content(
            model=self.settings.llm_model,
            contents=user_prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.settings.llm_temperature,
                max_output_tokens=self.settings.llm_max_tokens,
            ),
        )
        return (resp.text or "").strip()

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        for chunk in self.client.models.generate_content_stream(
            model=self.settings.llm_model,
            contents=user_prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.settings.llm_temperature,
                max_output_tokens=self.settings.llm_max_tokens,
            ),
        ):
            if chunk.text:
                yield chunk.text


def create_llm_client(settings: Settings | None = None) -> BaseLLMClient:
    """Instantiate the configured LLM backend."""
    settings = settings or get_settings()
    if settings.llm_provider == "gemini":
        return GeminiLLM(settings)
    return OpenAICompatibleLLM(settings)
