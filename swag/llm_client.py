"""Modular LLM client — OpenAI by default, extensible to other providers."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from swag.config import LLMConfig, SWAGConfig

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a text completion."""

    @abstractmethod
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Generate a JSON-structured completion."""

    def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        """Generate a final RAG answer from retrieved context."""
        context = "\n\n---\n\n".join(context_chunks)
        system = (
            "You are a helpful assistant. Answer the user's question using ONLY "
            "the provided context. If the context does not contain enough "
            "information, say so. Do not invent facts."
        )
        user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        return self.complete(system, user)


class OpenAILLMClient(BaseLLMClient):
    """OpenAI chat completions client."""

    def __init__(self, config: LLMConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError as exc:
                raise ImportError("Install openai: pip install openai") from exc
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)


class MockLLMClient(BaseLLMClient):
    """
    Fallback LLM for local demo without API keys.
    Uses heuristic extraction instead of real LLM calls.
    """

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if "Question:" in user_prompt:
            query = user_prompt.split("Question:")[-1].split("Answer:")[0].strip()
            return (
                f"Based on the retrieved context, here is a response to: {query}\n"
                "(Mock LLM — set OPENAI_API_KEY for real answers.)"
            )
        return "Mock LLM response."

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        # Extract chunk text from user prompt for heuristic profile generation
        text = user_prompt
        if "CHUNK TEXT:" in text:
            text = text.split("CHUNK TEXT:")[-1].split("Return JSON")[0].strip()
        elif "QUERY:" in text:
            text = text.split("QUERY:")[-1].split("Return JSON")[0].strip()

        return _heuristic_swag_json(text)


def _heuristic_swag_json(text: str) -> dict[str, Any]:
    """Build a minimal SWAG profile from text without an LLM."""
    import re

    words = re.findall(r"\b[A-Za-z]{4,}\b", text)
    # Simple frequency-based term extraction
    freq: dict[str, int] = {}
    for w in words:
        lw = w.lower()
        if lw not in {"that", "this", "with", "from", "have", "been", "were", "their", "which"}:
            freq[lw] = freq.get(lw, 0) + 1

    top_terms = sorted(freq, key=freq.get, reverse=True)[:8]
    domain = "technology" if any(t in text.lower() for t in ["software", "api", "data", "system"]) else "general"

    weighted = []
    for i, term in enumerate(top_terms[:5]):
        weighted.append(
            {
                "term": term,
                "category": "primary_concept" if i < 2 else "domain_term",
                "weight": round(0.9 - i * 0.08, 2),
                "reason": f"Extracted from chunk context (heuristic)",
            }
        )

    return {
        "primary_concepts": top_terms[:3],
        "domain_terms": top_terms[3:6],
        "direct_synonyms": [],
        "adjacent_concepts": top_terms[6:8],
        "business_goals": [],
        "user_intents": [],
        "negative_contexts": ["unrelated domain context"],
        "confidence": 0.5,
        "detected_domain": domain,
        "weighted_terms": weighted,
    }


def create_llm_client(swag_config: Optional[SWAGConfig] = None) -> BaseLLMClient:
    """Factory: create LLM client based on config."""
    cfg = swag_config or SWAGConfig.from_env()
    provider = cfg.llm.provider.lower()

    if provider == "openai":
        if cfg.openai_api_key:
            return OpenAILLMClient(cfg.llm, cfg.openai_api_key)
        logger.warning("No OPENAI_API_KEY — using MockLLMClient for SWAG profiles and answers.")
        return MockLLMClient()

    # Placeholder for future providers (anthropic, ollama, etc.)
    raise ValueError(
        f"Unsupported LLM provider: {provider}. "
        "Supported: openai. Extend llm_client.py to add more."
    )
