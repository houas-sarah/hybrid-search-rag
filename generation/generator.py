from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from config import LLMProvider, Settings
from generation.prompts import SYSTEM_PROMPT, build_user_prompt
from models import RetrievedChunk

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _messages(self, query: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query, chunks)},
        ]

    @abstractmethod
    async def generate(self, query: str, chunks: list[RetrievedChunk]) -> str: ...


class OllamaGenerator(BaseGenerator):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        from ollama import AsyncClient

        self._client = AsyncClient(host=settings.ollama_host)
        self._model = settings.ollama_model

    async def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        response = await self._client.chat(
            model=self._model,
            messages=self._messages(query, chunks),
            options={
                "temperature": self._settings.llm_temperature,
                "num_predict": self._settings.llm_max_tokens,
            },
        )
        # newer ollama returns a typed object; older ones a plain dict
        try:
            return response["message"]["content"].strip()
        except (TypeError, KeyError):
            return response.message.content.strip()


class GroqGenerator(BaseGenerator):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        from groq import AsyncGroq

        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self._model = settings.groq_model

    async def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._messages(query, chunks),
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
        )
        return (response.choices[0].message.content or "").strip()


def build_generator(settings: Settings) -> BaseGenerator:
    if settings.llm_provider is LLMProvider.OLLAMA:
        return OllamaGenerator(settings)
    if settings.llm_provider is LLMProvider.GROQ:
        return GroqGenerator(settings)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
