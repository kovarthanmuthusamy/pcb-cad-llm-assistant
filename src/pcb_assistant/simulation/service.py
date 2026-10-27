"""Text in, simulation result image out."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from pcb_assistant.config import Settings, get_settings
from pcb_assistant.contracts import SimulationRequest

from .request import Classifier, resolve_simulation_request

logger = logging.getLogger(__name__)

Runner = Callable[..., Path]


def simulate_from_text(
    text: str,
    settings: Settings | None = None,
    classifier: Classifier | None = None,
    runner: Runner | None = None,
) -> Path:
    """Understand ``text``, run the simulation and return the exported result image.

    ``classifier`` and ``runner`` default to the Sentence-BERT fallback and the eCADSTAR GUI
    adapter; both are loaded lazily and can be replaced (for example in tests).

    Raises:
        ValueError: the request does not identify a supported analysis.
        SimulationError: the simulator is not configured or not available.
    """

    settings = settings or get_settings()
    if classifier is None:
        from .semantic import classify_analysis_type

        classifier = classify_analysis_type
    request: SimulationRequest = resolve_simulation_request(
        text, classifier, settings.sim_min_similarity
    )
    logger.info(
        "Simulation request: type=%s resistance_ohm=%s line_length_mm=%s",
        request.analysis_type,
        request.resistance_ohm,
        request.line_length_mm,
    )
    if runner is None:
        from .ecadstar import run_simulation

        runner = run_simulation
    return runner(request, settings=settings)
