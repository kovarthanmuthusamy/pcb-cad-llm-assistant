"""The ReAct agent that routes requests to the Simulation, PCB Expert and RAG tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pcb_assistant.config import Settings, get_settings
from pcb_assistant.llm import PROFILES, ProviderProfile, build_llm
from pcb_assistant.text import final_answer_text
from pcb_assistant.tools import build_tools


@dataclass
class Assistant:
    """A configured agent executor with its conversation memory."""

    executor: Any
    memory: Any
    profile: ProviderProfile

    def last_message(self) -> str:
        """The most recent message in the conversation memory (empty if none)."""

        return "\n".join([message.content for message in self.memory.chat_memory.messages][-1:])

    def ask(self, text: str) -> str:
        """Run one agent turn and return the cleaned answer."""

        response = self.executor.invoke({"input": text})
        return final_answer_text(response["output"])


def build_assistant(provider: str, settings: Settings | None = None) -> Assistant:
    """Create the agent for ``provider``: model, tools, memory and system instruction."""

    from langchain.agents import AgentType, initialize_agent
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate

    settings = settings or get_settings()
    profile = PROFILES[provider]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # LangChain recommends LangGraph for new agents; the classic agent is kept for now.
    executor = initialize_agent(
        llm=build_llm(provider, settings),
        tools=build_tools(),
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=profile.max_iterations,
    )

    # Put the PCB-assistant instruction in front of the agent's default prompt messages.
    messages = executor.agent.llm_chain.prompt.messages.copy()
    messages.insert(0, SystemMessagePromptTemplate.from_template(profile.system_instruction))
    executor.agent.llm_chain.prompt = ChatPromptTemplate.from_messages(messages)

    return Assistant(executor=executor, memory=memory, profile=profile)
