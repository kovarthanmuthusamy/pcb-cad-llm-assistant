"""Deterministic input contracts that can be tested without models or eCADSTAR."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SimulationRequest:
    """Validated parameters extracted from a natural-language simulation request."""

    analysis_type: str
    resistance_ohm: float | None = None
    line_length_mm: float | None = None


def flatten_input(value: Any) -> str:
    """Convert nested tool input into a stable, human-readable string."""

    if isinstance(value, Mapping):
        return ", ".join(f"{key}: {flatten_input(item)}" for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ", ".join(flatten_input(item) for item in value)
    return str(value)


def _first_number(text: str, units: str) -> float | None:
    match = re.search(rf"(?P<value>\d+(?:\.\d+)?)\s*(?:{units})\b", text, re.IGNORECASE)
    return float(match.group("value")) if match else None


def parse_simulation_request(text: str) -> SimulationRequest:
    """Parse explicit analysis type, resistance, and line length from a request.

    Semantic fallback remains in the model-backed adapter. This contract intentionally handles
    explicit engineering terms so it can provide deterministic validation in CI.
    """

    if not text or not text.strip():
        raise ValueError("Simulation request must not be empty")

    patterns = (
        ("TDR", r"\b(?:TDR|time[- ]domain reflectometry)\b"),
        ("FD", r"\b(?:FD|frequency[- ]domain|frequency analysis)\b"),
        ("SI", r"\b(?:SI|signal integrity)\b"),
    )
    analysis_type = next(
        (name for name, pattern in patterns if re.search(pattern, text, re.IGNORECASE)),
        None,
    )
    if analysis_type is None:
        raise ValueError("Request must identify SI, FD, or TDR analysis")

    resistance = _first_number(text.replace("Ω", " ohm"), r"ohms?")
    line_length = _first_number(text, r"mm|millimeters?")
    return SimulationRequest(analysis_type, resistance, line_length)
