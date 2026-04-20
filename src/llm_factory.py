"""
LLM Factory — Multi-Provider Abstraction
=========================================
Reads LLM_PROVIDER (and related env vars) from the environment and returns
the correct LangChain chat model + embeddings pair.

Usage:
    from src.llm_factory import build_llm, build_embeddings

    llm        = build_llm()
    embeddings = build_embeddings()

Supported providers (set via LLM_PROVIDER env var):
    openai | anthropic | google | groq | mistral | ollama |
    cohere | together | bedrock | azure-openai
"""

from __future__ import annotations

import os
from typing import Any

# ── Cost table: (input $/1M tok, output $/1M tok) ─────────────
# Approximate April 2026 prices — update as needed
COST_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    # openai
    "gpt-5.4":            (10.00, 30.00),
    "gpt-5.4-mini":       (0.15,  0.60),
    "gpt-5.4-nano":       (0.075, 0.30),
    "gpt-4o":             (5.00,  15.00),
    "gpt-4o-mini":        (0.15,  0.60),
    # anthropic
    "claude-opus-4-6":    (15.00, 75.00),
    "claude-sonnet-4-6":  (3.00,  15.00),
    "claude-haiku-4-5":   (0.25,  1.25),
    # google
    "gemini-2.5-pro":     (7.00,  21.00),
    "gemini-2.5-flash":   (0.30,  2.50),
    # groq (pricing is inference cost, very cheap)
    "llama-3.3-70b-versatile": (0.59, 0.79),
    # mistral
    "mistral-large-latest": (3.00, 9.00),
    "mistral-small-latest": (0.20, 0.60),
    # cohere
    "command-r-plus": (3.00, 15.00),
    # ollama / local — zero cost
    "llama3.2": (0.0, 0.0),
    "qwen2.5":  (0.0, 0.0),
    "phi-4":    (0.0, 0.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for a single LLM call."""
    prices = COST_PER_1M_TOKENS.get(model, (0.0, 0.0))
    return round(
        (input_tokens / 1_000_000) * prices[0]
        + (output_tokens / 1_000_000) * prices[1],
        6,
    )


def build_llm() -> Any:
    """
    Return a LangChain BaseChatModel for the configured provider.
    Reads: LLM_PROVIDER, CHAT_MODEL, TEMPERATURE, MAX_TOKENS and
           the provider-specific API key env var.
    """
    provider   = os.getenv("LLM_PROVIDER", os.getenv("SANDBOX_LLM_PROVIDER", "openai")).lower()
    model      = os.getenv("CHAT_MODEL", _default_chat_model(provider))
    temperature = float(os.getenv("TEMPERATURE", "0.0"))
    max_tokens  = int(os.getenv("MAX_TOKENS", "2048"))

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature, max_tokens=max_tokens)  # type: ignore[call-arg]

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_output_tokens=max_tokens)

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=model, temperature=temperature, max_tokens=max_tokens)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, temperature=temperature, base_url=base_url)

    if provider == "cohere":
        from langchain_cohere import ChatCohere
        return ChatCohere(model=model, temperature=temperature)

    if provider == "together":
        from langchain_together import ChatTogether
        return ChatTogether(model=model, temperature=temperature, max_tokens=max_tokens)

    if provider == "bedrock":
        from langchain_aws import ChatBedrock
        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        return ChatBedrock(model_id=model, region_name=region,
                           model_kwargs={"temperature": temperature, "max_tokens": max_tokens})

    if provider == "azure-openai":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", model),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: '{provider}'. "
        "Valid: openai | anthropic | google | groq | mistral | ollama | "
        "cohere | together | bedrock | azure-openai"
    )


def build_embeddings() -> Any:
    """
    Return a LangChain Embeddings instance for the configured provider.
    Falls back to OpenAI embeddings for providers that don't offer their own
    (e.g. Groq, Together, Bedrock with Claude).
    """
    provider    = os.getenv("LLM_PROVIDER", os.getenv("SANDBOX_LLM_PROVIDER", "openai")).lower()
    embed_model = os.getenv("EMBEDDING_MODEL", _default_embedding_model(provider))

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=embed_model)

    if provider == "anthropic":
        # Anthropic has no embedding API — use OpenAI if key exists, else local
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model="text-embedding-3-small")
        return _local_embeddings()

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=embed_model)

    if provider == "groq":
        # Groq has no embedding API — use OpenAI fallback
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model="text-embedding-3-small")
        return _local_embeddings()

    if provider == "mistral":
        from langchain_mistralai import MistralAIEmbeddings
        return MistralAIEmbeddings(model=embed_model)

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddings(model=embed_model, base_url=base_url)

    if provider == "cohere":
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings(model=embed_model)

    if provider == "together":
        from langchain_together import TogetherEmbeddings
        return TogetherEmbeddings(model=embed_model)

    if provider == "bedrock":
        from langchain_aws import BedrockEmbeddings
        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        return BedrockEmbeddings(model_id=embed_model, region_name=region)

    if provider == "azure-openai":
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", embed_model),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        )

    raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'")


def _local_embeddings() -> Any:
    """Fallback: HuggingFace local embeddings — no API key needed."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _default_chat_model(provider: str) -> str:
    defaults = {
        "openai":       "gpt-4o-mini",
        "anthropic":    "claude-sonnet-4-6",
        "google":       "gemini-2.5-flash",
        "groq":         "llama-3.3-70b-versatile",
        "mistral":      "mistral-small-latest",
        "ollama":       "llama3.2",
        "cohere":       "command-r-plus",
        "together":     "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "bedrock":      "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "azure-openai": "gpt-4o-mini",
    }
    return defaults.get(provider, "gpt-4o-mini")


def _default_embedding_model(provider: str) -> str:
    defaults = {
        "openai":       "text-embedding-3-small",
        "google":       "models/text-embedding-004",
        "mistral":      "mistral-embed",
        "ollama":       "nomic-embed-text",
        "cohere":       "embed-english-v3.0",
        "together":     "togethercomputer/m2-bert-80M-8k-retrieval",
        "bedrock":      "amazon.titan-embed-text-v2:0",
        "azure-openai": "text-embedding-3-small",
    }
    return defaults.get(provider, "text-embedding-3-small")
