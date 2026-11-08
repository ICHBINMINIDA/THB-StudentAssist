"""
Chat model factory.

Two backends, selected with ``LLM_PROVIDER``:

* ``ollama``  talks to an Ollama server directly (default, no extra services).
* ``openai``  talks to any OpenAI-compatible endpoint: a LiteLLM gateway, vLLM,
              or a hosted API. This is what the Docker Compose stack uses, so
              the app does not need to know which engine serves the model.
"""

from langchain_core.language_models.chat_models import BaseChatModel

import config


def make_llm() -> BaseChatModel:
    if config.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
            num_ctx=config.LLM_NUM_CTX,
        )

    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        # The context window is configured on the serving side here
        # (OLLAMA_CONTEXT_LENGTH behind LiteLLM, --max-model-len for vLLM).
        return ChatOpenAI(
            model=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY or "not-needed",
            temperature=config.LLM_TEMPERATURE,
            max_retries=1,
        )

    raise ValueError(f"Unknown LLM_PROVIDER {config.LLM_PROVIDER!r}; use 'ollama' or 'openai'.")


def describe() -> str:
    """Human-readable backend description for logs and error messages."""
    if config.LLM_PROVIDER == "openai":
        return f"{config.LLM_MODEL} via OpenAI-compatible API at {config.LLM_BASE_URL}"
    return f"{config.OLLAMA_MODEL} via Ollama at {config.OLLAMA_BASE_URL}"


def health_request() -> tuple[str, dict[str, str]]:
    """URL and headers of a cheap request that proves the backend is reachable."""
    if config.LLM_PROVIDER == "openai":
        headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"} if config.LLM_API_KEY else {}
        return f"{config.LLM_BASE_URL.rstrip('/')}/models", headers
    return f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/tags", {}
