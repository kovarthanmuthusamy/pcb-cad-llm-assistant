"""Orchestrator back-ends. All providers share the same tools and agent; only the model and a
few prompt/loop settings differ, which is what the profiles below capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pcb_assistant.config import Settings, get_settings

# Prompt text is tuned empirically with the models below; edit it deliberately.
BASE_INSTRUCTION = (
    "You are chat assistant specializing in PCB design. You have access to expert tools to "
    "retrieve accurate technical information, perform simulations, and analyze design "
    "principles.provide accurate and broad answers.You may only take ONE action at a time "
    "and wait for observation"
)


@dataclass(frozen=True)
class ProviderProfile:
    """Per-provider differences of the ReAct loop."""

    name: str
    max_iterations: int
    instruction_suffix: str = ""
    include_history_in_input: bool = False

    @property
    def system_instruction(self) -> str:
        return BASE_INSTRUCTION + self.instruction_suffix


PROFILES: dict[str, ProviderProfile] = {
    "ollama": ProviderProfile("ollama", max_iterations=4),
    "openai": ProviderProfile(
        "openai",
        max_iterations=4,
        instruction_suffix=",dont miss captions - Action: , Think etc..",
        include_history_in_input=True,
    ),
    "google": ProviderProfile("google", max_iterations=6),
}
PROVIDERS = tuple(PROFILES)


def build_llm(provider: str, settings: Settings | None = None) -> Any:
    """Create the chat model for ``provider`` (``ollama``, ``openai`` or ``google``).

    OpenAI and Google read ``OPENAI_API_KEY`` / ``GOOGLE_API_KEY`` from the environment.
    """

    settings = settings or get_settings()
    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        # The ``ollama`` client used by the RAG modules reads OLLAMA_HOST itself; this older
        # LangChain wrapper needs the URL passed explicitly.
        host = {"base_url": settings.ollama_host} if settings.ollama_host else {}
        return ChatOllama(
            model=settings.ollama_model,
            response_format="json",
            temperature=settings.temperature,
            **host,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model, temperature=settings.temperature, verbose=True
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.google_model, temperature=settings.temperature, verbose=True
        )
    raise ValueError(f"Unknown provider {provider!r}; expected one of {list(PROFILES)}")
