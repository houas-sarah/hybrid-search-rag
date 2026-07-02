"""Quick check that the configured LLM backend actually answers.

Run it after editing .env, before starting the server:  python check_backend.py
"""

from __future__ import annotations

import asyncio

from config import LLMProvider, get_settings
from generation import build_generator
from models import RetrievedChunk


async def main() -> None:
    settings = get_settings()
    provider = settings.llm_provider
    print(f"Backend: {provider.value}")

    if provider is LLMProvider.GROQ:
        print(f"Model:   {settings.groq_model}")
        if not settings.groq_api_key:
            print("\nGROQ_API_KEY is empty in .env — paste your key and re-run.")
            return
        print(f"API key: ...{settings.groq_api_key[-4:]}")
    else:
        print(f"Model:   {settings.ollama_model}")
        print(f"Host:    {settings.ollama_host}")

    print("\nSending a test prompt ...")
    sample = RetrievedChunk(
        id="readme.md::0",
        text="The setup requires Python 3.11 and PostgreSQL 15.",
        source="readme.md",
    )
    try:
        answer = await build_generator(settings).generate(
            "What Python version and database are required?", [sample]
        )
    except Exception as exc:
        print(f"\nBackend call failed: {type(exc).__name__}: {exc}")
        if provider is LLMProvider.OLLAMA:
            print("Is Ollama installed and running? Try: ollama serve")
        else:
            print("Check the API key is valid and the model name exists.")
        return

    print("\nOK. Sample answer:\n")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
