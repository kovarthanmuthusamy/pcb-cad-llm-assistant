"""Interactive command line for the assistant: ``pcb-assistant --provider ollama|openai|google``."""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Callable, Sequence
from typing import Any

from pcb_assistant.llm import PROVIDERS

logger = logging.getLogger(__name__)

MAX_IMAGES = 3


def build_agent_input(
    query: str,
    image_descriptions: Sequence[str],
    history: str = "",
    include_history: bool = False,
) -> str:
    """Compose the text sent to the agent: optional history, image descriptions, the query."""

    image_info = "\n".join(
        f"Image {index}: {description}" for index, description in enumerate(image_descriptions, 1)
    )
    image_input = f"Attached images:\n{image_info}\n\n" if image_descriptions else ""
    if include_history:
        return f"previous history = {history}\n{image_input}\nUser: {query}"
    return f"{image_input}\nUser: {query}"


def collect_image_paths(
    read: Callable[[str], str],
    write: Callable[[str], None],
    limit: int = MAX_IMAGES,
) -> list[str]:
    """Ask for up to ``limit`` image paths; an empty answer stops, a bad path uses up one try."""

    paths: list[str] = []
    for number in range(1, limit + 1):
        path = read(f"Enter image path {number} (or press Enter to skip): ").strip()
        if not path:
            break
        if not os.path.exists(path):
            write(f"Image not found: {path}. Try again.")
            continue
        paths.append(path)
    return paths


def describe_images(
    paths: Sequence[str],
    describe: Callable[[str], str],
    write: Callable[[str], None],
) -> list[str]:
    """Turn images into text; a failing image becomes a placeholder instead of an error."""

    descriptions: list[str] = []
    for path in paths:
        try:
            descriptions.append(describe(path))
        except Exception as exc:  # the vision model is optional; the chat must go on
            write(f"Error describing image {path}: {exc}")
            descriptions.append("Image description unavailable")
    return descriptions


def chat_loop(
    assistant: Any,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    describe: Callable[[str], str] | None = None,
) -> None:
    """Chat until the user types ``exit``; ``read``, ``write`` and ``describe`` are injectable."""

    write("PCB AI Chatbot started. Type 'exit' to quit.\n")
    while True:
        try:
            query = read("Enter your query: ")
        except (EOFError, KeyboardInterrupt):
            write("\nExiting chatbot.")
            return
        if query.strip().lower() == "exit":
            write("Exiting chatbot.")
            return
        if not query.strip():
            continue

        started = time.perf_counter()
        image_descriptions: list[str] = []
        image_paths = collect_image_paths(read, write)
        if image_paths:
            write("Processing images...")
            if describe is None:
                from pcb_assistant.vision.granite import describe_image

                describe = describe_image
            image_descriptions = describe_images(image_paths, describe, write)

        agent_input = build_agent_input(
            query,
            image_descriptions,
            history=assistant.last_message(),
            include_history=assistant.profile.include_history_in_input,
        )
        logger.debug("Agent input: %s", agent_input)

        try:
            write(assistant.ask(agent_input))
        except Exception as exc:  # keep the session alive after a failed turn
            logger.debug("Agent call failed", exc_info=True)
            write(f"Error: {exc}")
        write(f"Time taken: {time.perf_counter() - started:.2f} seconds\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pcb-assistant",
        description="Multi-agent LLM assistant for a Signal-Integrity EDA workflow.",
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default="ollama",
        help="orchestrator back-end (default: ollama, a local DeepSeek-R1 model)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="logging verbosity (default: WARNING)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        from pcb_assistant.agent import build_assistant

        assistant = build_assistant(args.provider)
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Install the runtime packages with:")
        print("    pip install -r requirements.txt")
        return 1
    chat_loop(assistant)
    return 0
