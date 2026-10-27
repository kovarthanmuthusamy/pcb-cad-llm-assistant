"""Wiring tests for the agent: LangChain is replaced by small stand-ins, so no model is needed."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from pcb_assistant import agent
from pcb_assistant.llm import PROFILES


class FakeMemory:
    def __init__(self, memory_key: str, return_messages: bool) -> None:
        self.memory_key = memory_key
        self.chat_memory = SimpleNamespace(
            messages=[SimpleNamespace(content="old"), SimpleNamespace(content="latest")]
        )


class FakeExecutor:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        prompt = SimpleNamespace(messages=["default-1", "default-2"])
        self.agent = SimpleNamespace(llm_chain=SimpleNamespace(prompt=prompt))

    def invoke(self, payload: dict[str, str]) -> dict[str, str]:
        self.payload = payload
        trailing_action = '{"action": "x", "action_input": "y"}'
        return {"output": f"<think>planning</think>The line is 50 ohm.\n{trailing_action}"}


@pytest.fixture
def fake_langchain(monkeypatch: pytest.MonkeyPatch) -> None:
    def module(name: str, **attributes: Any) -> None:
        stand_in = ModuleType(name)
        for key, value in attributes.items():
            setattr(stand_in, key, value)
        monkeypatch.setitem(sys.modules, name, stand_in)

    module("langchain")
    module(
        "langchain.agents",
        AgentType=SimpleNamespace(STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION="structured-chat"),
        initialize_agent=lambda **kwargs: FakeExecutor(**kwargs),
    )
    module("langchain.memory", ConversationBufferMemory=FakeMemory)
    module("langchain.prompts")
    module(
        "langchain.prompts.chat",
        ChatPromptTemplate=SimpleNamespace(from_messages=lambda messages: ("template", messages)),
        SystemMessagePromptTemplate=SimpleNamespace(from_template=lambda text: ("system", text)),
    )
    monkeypatch.setattr(agent, "build_llm", lambda provider, settings: f"llm:{provider}")
    monkeypatch.setattr(agent, "build_tools", lambda: ["tool-a", "tool-b"])


@pytest.mark.parametrize("provider", ["ollama", "openai", "google"])
def test_build_assistant_uses_the_provider_profile(fake_langchain: None, provider: str) -> None:
    assistant = agent.build_assistant(provider)
    profile = PROFILES[provider]

    options = assistant.executor.kwargs
    assert options["llm"] == f"llm:{provider}"
    assert options["tools"] == ["tool-a", "tool-b"]
    assert options["agent"] == "structured-chat"
    assert options["max_iterations"] == profile.max_iterations
    assert options["handle_parsing_errors"] is True
    assert options["memory"] is assistant.memory

    # The PCB-assistant instruction goes in front of the agent's default prompt messages.
    assert assistant.executor.agent.llm_chain.prompt == (
        "template",
        [("system", profile.system_instruction), "default-1", "default-2"],
    )


def test_assistant_answers_with_cleaned_text_and_exposes_history(fake_langchain: None) -> None:
    assistant = agent.build_assistant("ollama")
    assert assistant.ask("hello") == "The line is 50 ohm."
    assert assistant.executor.payload == {"input": "hello"}
    assert assistant.last_message() == "latest"
