import pytest

from pcb_assistant import extract_parameters, flatten_input, parse_simulation_request


def test_flatten_input_handles_nested_tool_payload() -> None:
    payload = {"action_input": {"query": "run SI", "values": [50, "ohm"]}}
    assert flatten_input(payload) == "action_input: query: run SI, values: 50, ohm"


@pytest.mark.parametrize(
    ("text", "analysis_type"),
    [
        ("Run signal integrity with 50 ohm termination", "SI"),
        ("Perform frequency-domain analysis", "FD"),
        ("Execute time-domain reflectometry", "TDR"),
    ],
)
def test_parse_simulation_analysis_aliases(text: str, analysis_type: str) -> None:
    assert parse_simulation_request(text).analysis_type == analysis_type


def test_parse_simulation_extracts_resistance_and_length() -> None:
    request = parse_simulation_request("Run SI with 28.5 ohms and a 103 mm line")
    assert request.resistance_ohm == 28.5
    assert request.line_length_mm == 103.0


@pytest.mark.parametrize("symbol", ["Ω", "Ω"], ids=["greek-omega", "ohm-sign"])
def test_parse_simulation_supports_omega_symbols(symbol: str) -> None:
    assert parse_simulation_request(f"TDR with 100 {symbol}").resistance_ohm == 100.0
    assert parse_simulation_request(f"TDR with 100{symbol}").resistance_ohm == 100.0


@pytest.mark.parametrize("text", ["", "run a simulation"])
def test_parse_simulation_rejects_incomplete_requests(text: str) -> None:
    with pytest.raises(ValueError):
        parse_simulation_request(text)


def test_extract_parameters_without_analysis_type() -> None:
    assert extract_parameters("a 28 ohm resistor on a 103 millimeters line") == (28.0, 103.0)
    assert extract_parameters("nothing numeric here") == (None, None)
