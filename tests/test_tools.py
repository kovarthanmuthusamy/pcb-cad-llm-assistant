from pathlib import Path
from typing import Any

import pytest

from pcb_assistant import tools
from pcb_assistant.config import Settings
from pcb_assistant.contracts import SimulationRequest
from pcb_assistant.simulation.ecadstar import SimulationError


@pytest.mark.parametrize(
    ("args", "kwargs", "keys", "expected"),
    [
        (("SI with 100 ohm",), {}, ("query",), "SI with 100 ohm"),
        ((), {"query": "what is TDR?"}, ("query",), "what is TDR?"),
        ((), {"type": "FD with 28 ohm"}, ("type", "query"), "FD with 28 ohm"),
        ((), {"action_input": "plain text"}, ("query",), "plain text"),
        ((), {"action_input": {"query": "nested"}}, ("query",), "nested"),
        (({"query": "positional dict"},), {}, ("query",), "positional dict"),
        (({"query": ["a", "b"]},), {}, ("query",), "a, b"),
        ((), {"query": {"query": "doubly wrapped"}}, ("query",), "doubly wrapped"),
        ((), {"query": {"topic": "TDR"}}, ("query",), "topic: TDR"),
        ((), {"unexpected": "shape"}, ("query",), "unexpected: shape"),
    ],
)
def test_unwrap_tool_input_accepts_every_shape_the_orchestrator_sends(
    args: tuple[Any, ...], kwargs: dict[str, Any], keys: tuple[str, ...], expected: str
) -> None:
    assert tools.unwrap_tool_input(args, kwargs, keys) == expected


@pytest.mark.parametrize(("args", "kwargs"), [((), {}), ((None,), {}), (("   ",), {})])
def test_unwrap_tool_input_returns_none_for_empty_input(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    assert tools.unwrap_tool_input(args, kwargs) is None


def test_simulate_passes_the_request_through_the_real_parser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: a request with the ohm sign used to lose its resistor value."""

    seen: list[SimulationRequest] = []

    def fake_run(request: SimulationRequest, settings: Settings) -> Path:
        seen.append(request)
        return tmp_path / "result.png"

    monkeypatch.setattr("pcb_assistant.simulation.ecadstar.run_simulation", fake_run)
    monkeypatch.setattr(
        "pcb_assistant.vision.granite.describe_image", lambda path: " description = ok"
    )
    monkeypatch.setattr(tools, "get_settings", Settings)
    monkeypatch.setattr("pcb_assistant.simulation.service.get_settings", Settings)

    result = tools.simulate(type="Run signal integrity with 100 Ω and 103 mm")

    assert seen == [SimulationRequest("SI", 100.0, 103.0)]
    assert result == (
        "Simulation completed for Run signal integrity with 100 Ω and 103 mm, "
        "Result Image description = ok"
    )


def test_simulate_reports_an_unclear_request_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(text: str) -> None:
        raise ValueError("Request must identify SI, FD, or TDR analysis")

    monkeypatch.setattr("pcb_assistant.simulation.service.simulate_from_text", refuse)
    assert tools.simulate("do something").startswith("Simulation was not started: Request must")


def test_simulate_reports_a_missing_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(text: str) -> None:
        raise SimulationError("The eCADSTAR adapter only runs on Windows")

    monkeypatch.setattr("pcb_assistant.simulation.service.simulate_from_text", unavailable)
    assert tools.simulate("SI") == "Simulation failed: The eCADSTAR adapter only runs on Windows"


def test_simulate_without_input_is_an_error_message() -> None:
    assert tools.simulate().startswith("Error:")


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"query": "q", "result": "  The answer.  "}, "The answer."),
        ("Response: text\nSources: ['a:1:0']", "Response: text\nSources: ['a:1:0']"),
        (None, "RAG returned no result."),
        ({"unexpected": 1}, '{\n  "unexpected": 1\n}'),
    ],
)
def test_format_rag_result(result: Any, expected: str) -> None:
    assert tools.format_rag_result(result) == expected


def test_rag_tool_strips_reasoning_and_uses_configured_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(rag_cleanup_delay_s=0, rag_num_results=7, rag_search_engine="bing")
    captured: dict[str, Any] = {}

    def fake_answer(question: str, **options: Any) -> dict[str, str]:
        captured.update(question=question, **options)
        return {"query": question, "result": "<think>checking</think>Use 50 ohm."}

    monkeypatch.setattr(tools, "get_settings", lambda: settings)
    monkeypatch.setattr("pcb_assistant.rag.pipeline.answer", fake_answer)

    assert tools.rag_tool(query="line impedance?") == "Use 50 ohm."
    assert captured["question"] == "line impedance?"
    assert captured["num_results"] == 7
    assert captured["searcher"] == "bing"
    assert captured["is_link"] is False


def test_rag_tool_returns_document_answers_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "get_settings", lambda: Settings(rag_cleanup_delay_s=0))
    monkeypatch.setattr(
        "pcb_assistant.rag.pipeline.answer", lambda question, **options: 'Response: "quoted"'
    )
    assert tools.rag_tool("q") == 'Response: "quoted"'  # not JSON-escaped


def test_rag_tool_without_query() -> None:
    assert tools.rag_tool() == "Error: No query provided to RAG Tool."


def test_pcb_expert_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pcb_assistant.experts.pcb_expert.get_response", lambda q: f"expert: {q}")
    assert tools.pcb_expert_tool(query={"query": "crosstalk"}) == "expert: crosstalk"


def test_pcb_expert_tool_reports_an_empty_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pcb_assistant.experts.pcb_expert.get_response", lambda q: "  ")
    assert tools.pcb_expert_tool("anything") == "Failed to get a response from PCB Expert."


def test_build_tools_registers_the_three_tools() -> None:
    pytest.importorskip("langchain")
    assert [tool.name for tool in tools.build_tools()] == ["Simulation", "PCB Expert", "RAG"]
