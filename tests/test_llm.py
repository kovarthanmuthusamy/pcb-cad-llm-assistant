"""``build_llm`` with the LangChain chat classes replaced by a recorder."""

import sys
from types import ModuleType
from typing import Any

import pytest

from pcb_assistant.config import Settings
from pcb_assistant.llm import build_llm


class Recorder:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def fake_chat_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, attribute in [
        ("langchain_community.chat_models", "ChatOllama"),
        ("langchain_openai", "ChatOpenAI"),
        ("langchain_google_genai", "ChatGoogleGenerativeAI"),
    ]:
        module = ModuleType(name)
        setattr(module, attribute, Recorder)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "langchain_community", ModuleType("langchain_community"))


def test_ollama_uses_localhost_unless_a_host_is_configured() -> None:
    local = build_llm("ollama", Settings())
    assert "base_url" not in local.kwargs
    assert local.kwargs["model"] == "deepseek-r1:70b"

    remote = build_llm("ollama", Settings(ollama_host="http://ollama:11434"))
    assert remote.kwargs["base_url"] == "http://ollama:11434"


@pytest.mark.parametrize(
    ("provider", "model"), [("openai", "gpt-4o"), ("google", "gemini-1.5-pro")]
)
def test_hosted_providers_get_their_model_and_temperature(provider: str, model: str) -> None:
    llm = build_llm(provider, Settings(temperature=0.1))
    assert llm.kwargs["model"] == model
    assert llm.kwargs["temperature"] == 0.1


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="anthropic"):
        build_llm("anthropic", Settings())
