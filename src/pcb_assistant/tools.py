"""LangChain tools exposed to the orchestrator: Simulation, PCB Expert and RAG.

The orchestrator model emits tool input in several shapes (a bare string, ``{"query": ...}``,
``{"action_input": {...}}``). :func:`unwrap_tool_input` turns all of them into one string, and
the tools only deal with that string. Heavy dependencies are imported inside the tools, so this
module is cheap to import.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from pcb_assistant.config import get_settings
from pcb_assistant.contracts import flatten_input
from pcb_assistant.text import remove_think_blocks

logger = logging.getLogger(__name__)

# The wording of these descriptions is part of the agent prompt; change it deliberately.
SIMULATION_DESCRIPTION = """
    This action is used to invoke simulations within the eCADstar Electrical Editor simulation tool using the default P2P topology circuit file. The tool supports the following analysis Signal Integrity (SI) Analysis,Frequency Domain Analysis,Time-Domain Reflectometry (TDR) Analysis.
    You can dynamically adjust the following pvalues before simulations or leave as null,because there is default values for this parameters in the simulator - Resistor Value (in ohms),Transmission Line Length (in mm).
    The tool processes run simulation and provide relevant analysis results as image description based on the input configuration.Form questions like below to call the tool
    example tool input -  SI with resistor value 100 ohm and transmission value of 100 mm .
    """

PCB_EXPERT_DESCRIPTION = """Primary tool for PCB-related queries, including design, manufacturing,
        noise issues, and electronics-specific topics. If a query partially overlaps with PCB topics,
        prioritize this tool before considering others.Ensure responses are precise, detailed,
        and relevant using retrieved information."""

RAG_DESCRIPTION = """Retrieves accurate and relevant information from local databases,
        user-provided documents, or web sources. Use this tool when other actions do not provide sufficient or reliable results."""


class SimulationInput(BaseModel):
    type: str = Field(..., description="query for eCADstar simulations ")


class RagInput(BaseModel):
    query: str = Field(
        ..., description="Query to retrieve factual information from databases or the web."
    )


class PCBExpertInput(BaseModel):
    # ``Any``: the orchestrator sometimes sends a dict or list instead of a string.
    query: Any = Field(..., description="ask question")


def _as_text(value: Any, keys: tuple[str, ...]) -> str | None:
    if isinstance(value, Mapping):
        if "action_input" in value:
            return _as_text(value["action_input"], keys)
        for key in keys:
            if key in value:
                return _as_text(value[key], keys)
    if value is None:
        return None
    return flatten_input(value)


def unwrap_tool_input(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    keys: tuple[str, ...] = ("query",),
) -> str | None:
    """Extract the request text from whatever shape the orchestrator passed.

    Looks at the first positional argument, else at ``kwargs`` (or ``kwargs["action_input"]``),
    and reads the first of ``keys`` found in a mapping. Returns ``None`` if nothing usable is left.
    """

    value = args[0] if args else kwargs.get("action_input", kwargs)
    text = _as_text(value, keys)
    return text if text and text.strip() else None


def simulate(*args: Any, **kwargs: Any) -> str:
    """Run an eCADSTAR simulation described in plain text and describe the result image."""

    from pcb_assistant.simulation.ecadstar import SimulationError
    from pcb_assistant.simulation.service import simulate_from_text
    from pcb_assistant.vision.granite import describe_image

    request_text = unwrap_tool_input(args, kwargs, keys=("type", "query"))
    if request_text is None:
        return "Error: no simulation request was provided."

    logger.info("Simulation in progress for: %s", request_text)
    try:
        image_path = simulate_from_text(request_text)
    except ValueError as exc:
        return f"Simulation was not started: {exc}"
    except SimulationError as exc:
        return f"Simulation failed: {exc}"

    description = describe_image(image_path)
    return f"Simulation completed for {request_text}, Result Image{description}"


def format_rag_result(result: Any) -> str:
    """Reduce a RAG result (answer text, chain output dict or ``None``) to plain text."""

    if isinstance(result, Mapping) and "result" in result:
        return str(result["result"]).strip()
    if isinstance(result, str):
        return result.strip()
    if result is None:
        return "RAG returned no result."
    return json.dumps(result, indent=2, ensure_ascii=False, default=str)


def rag_tool(*args: Any, **kwargs: Any) -> str:
    """Answer a factual question from the local corpora or the web."""

    from pcb_assistant.rag.pipeline import answer

    query = unwrap_tool_input(args, kwargs, keys=("query",))
    if query is None:
        return "Error: No query provided to RAG Tool."

    settings = get_settings()
    result = answer(
        query,
        num_results=settings.rag_num_results,
        searcher=settings.rag_search_engine,
        is_link=False,
        model=settings.rag_model,
        settings=settings,
    )
    cleaned = remove_think_blocks(format_rag_result(result))

    # Kept from the original implementation ("ensure cleanup"): gives the vector store time to
    # release its files before the next call. Set PCB_RAG_CLEANUP_DELAY_S=0 to disable.
    time.sleep(settings.rag_cleanup_delay_s)
    return cleaned


def pcb_expert_tool(*args: Any, **kwargs: Any) -> str:
    """Ask the fine-tuned PCB expert model."""

    from pcb_assistant.experts.pcb_expert import get_response

    query = unwrap_tool_input(args, kwargs, keys=("query",))
    if query is None:
        return "Error: No query provided to PCB Expert."

    response = get_response(query)
    if response and response.strip():
        return response

    logger.warning("No response received from the PCB expert")
    return "Failed to get a response from PCB Expert."


def build_tools() -> list[Any]:
    """The three LangChain ``StructuredTool`` objects registered with the orchestrator."""

    from langchain.tools import StructuredTool

    return [
        StructuredTool.from_function(
            simulate,
            name="Simulation",
            description=SIMULATION_DESCRIPTION,
            args_schema=SimulationInput,
        ),
        StructuredTool.from_function(
            pcb_expert_tool,
            name="PCB Expert",
            description=PCB_EXPERT_DESCRIPTION,
            args_schema=PCBExpertInput,
        ),
        StructuredTool.from_function(
            rag_tool,
            name="RAG",
            description=RAG_DESCRIPTION,
            args_schema=RagInput,
        ),
    ]
