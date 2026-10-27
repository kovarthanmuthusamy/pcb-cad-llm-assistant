"""Multi-agent LLM assistant for a Signal-Integrity EDA workflow.

Importing the package is cheap: model, GPU and GUI libraries are only loaded when the
corresponding tool is actually used.
"""

from .config import Settings, get_settings
from .contracts import (
    SimulationRequest,
    extract_parameters,
    flatten_input,
    parse_simulation_request,
)

__all__ = [
    "Settings",
    "SimulationRequest",
    "extract_parameters",
    "flatten_input",
    "get_settings",
    "parse_simulation_request",
]
