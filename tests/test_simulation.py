import sys
from pathlib import Path

import pytest

from pcb_assistant.config import Settings
from pcb_assistant.contracts import SimulationRequest
from pcb_assistant.simulation import ecadstar
from pcb_assistant.simulation.ecadstar import SimulationError, fill_template, run_simulation
from pcb_assistant.simulation.request import resolve_simulation_request
from pcb_assistant.simulation.service import simulate_from_text


class FakeClassifier:
    def __init__(self, analysis_type: str, similarity: float) -> None:
        self.result = (analysis_type, similarity)
        self.calls: list[str] = []

    def __call__(self, text: str) -> tuple[str, float]:
        self.calls.append(text)
        return self.result


def test_explicit_request_never_consults_the_classifier() -> None:
    classifier = FakeClassifier("FD", 0.99)
    request = resolve_simulation_request("Run SI with 100 Ω and 100 mm", classifier)
    assert request == SimulationRequest("SI", 100.0, 100.0)
    assert classifier.calls == []


def test_classifier_names_the_analysis_when_the_text_does_not() -> None:
    classifier = FakeClassifier("TDR", 0.62)
    request = resolve_simulation_request("check the reflections on a 30 mm line", classifier)
    assert request == SimulationRequest("TDR", None, 30.0)


def test_weak_semantic_match_is_rejected_instead_of_guessed() -> None:
    classifier = FakeClassifier("FD", 0.12)
    with pytest.raises(ValueError, match="closest match FD scored 0.12"):
        resolve_simulation_request("what is the weather", classifier, min_similarity=0.3)


def test_missing_analysis_without_classifier_is_an_error() -> None:
    with pytest.raises(ValueError, match="SI, FD, or TDR"):
        resolve_simulation_request("run a simulation")


def test_empty_text_is_rejected_even_with_a_classifier() -> None:
    classifier = FakeClassifier("SI", 1.0)
    with pytest.raises(ValueError):
        resolve_simulation_request("   ", classifier)
    assert classifier.calls == []


def test_fill_template_scales_values_for_the_sidf_file() -> None:
    template = "R=RESISTOR_VALUE L=LENGTH_VALUE"
    assert fill_template(template, 100, 100) == "R=10000 L=10000000"


def test_fill_template_rounds_instead_of_truncating() -> None:
    assert fill_template("RESISTOR_VALUE", 0.29, None) == "29"  # int(0.29 * 100) would be 28


def test_fill_template_leaves_unset_placeholders_alone() -> None:
    assert fill_template("R=RESISTOR_VALUE L=LENGTH_VALUE", None, 1) == "R=RESISTOR_VALUE L=100000"


def test_all_supported_analyses_have_hotkeys() -> None:
    assert set(ecadstar.HOTKEYS) == {"SI", "TDR", "FD"}


def test_run_simulation_rejects_unknown_analysis() -> None:
    with pytest.raises(ValueError, match="Unsupported analysis type"):
        run_simulation(SimulationRequest("XYZ"), settings=Settings())


def test_run_simulation_rejects_unknown_topology() -> None:
    with pytest.raises(ValueError, match="Unknown topology"):
        run_simulation(SimulationRequest("SI"), topology="Ring", settings=Settings())


def test_run_simulation_requires_configured_paths() -> None:
    with pytest.raises(SimulationError, match="not configured"):
        run_simulation(SimulationRequest("SI"), settings=Settings())


def test_run_simulation_writes_the_sidf_then_needs_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = tmp_path / "template.txt"
    template.write_text("R=RESISTOR_VALUE L=LENGTH_VALUE")
    sidf = tmp_path / "circuit.sidf"
    settings = Settings(
        ecadstar_p2p_sidf=str(sidf),
        ecadstar_p2p_template=str(template),
        ecadstar_output_dir=tmp_path / "images",
    )
    monkeypatch.setattr(sys, "platform", "linux")

    with pytest.raises(SimulationError, match="only runs on Windows"):
        run_simulation(SimulationRequest("SI", 28.5, 103.0), settings=settings)

    assert sidf.read_text() == "R=2850 L=10300000"


def test_simulate_from_text_uses_the_injected_runner(tmp_path: Path) -> None:
    seen: list[SimulationRequest] = []

    def fake_runner(request: SimulationRequest, settings: Settings) -> Path:
        seen.append(request)
        return tmp_path / "result.png"

    image = simulate_from_text(
        "TDR with 50 ohm",
        settings=Settings(),
        classifier=FakeClassifier("SI", 1.0),
        runner=fake_runner,
    )
    assert image == tmp_path / "result.png"
    assert seen == [SimulationRequest("TDR", 50.0, None)]
