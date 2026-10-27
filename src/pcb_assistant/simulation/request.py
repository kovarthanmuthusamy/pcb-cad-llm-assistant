"""Turn free text into a validated :class:`SimulationRequest`."""

from __future__ import annotations

from collections.abc import Callable

from pcb_assistant.contracts import SimulationRequest, extract_parameters, parse_simulation_request

# text -> (analysis type, similarity in 0..1)
Classifier = Callable[[str], tuple[str, float]]


def resolve_simulation_request(
    text: str,
    classifier: Classifier | None = None,
    min_similarity: float = 0.3,
) -> SimulationRequest:
    """Parse ``text`` deterministically; fall back to ``classifier`` for the analysis type.

    The fallback only accepts a match with ``similarity >= min_similarity``. Below that the
    request is rejected with a ``ValueError`` so the agent can ask the user to name SI, FD or
    TDR, instead of silently running the least-bad analysis.
    """

    try:
        return parse_simulation_request(text)
    except ValueError as explicit_error:
        if classifier is None or not text.strip():
            raise
        analysis_type, similarity = classifier(text)
        if similarity < min_similarity:
            raise ValueError(
                f"Could not tell which analysis (SI, FD or TDR) is meant by {text!r}; "
                f"closest match {analysis_type} scored {similarity:.2f}"
            ) from explicit_error
        resistance, line_length = extract_parameters(text)
        return SimulationRequest(analysis_type, resistance, line_length)
