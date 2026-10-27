"""Drives the eCADSTAR Electrical Editor through its Windows GUI.

This adapter needs Windows, a licensed eCADSTAR install and the SIDF circuit files configured
through the environment (see ``.env.example``). It works by sending keystrokes, so it relies on
fixed waits (``ECADSTAR_OPEN_WAIT_S``, ``ECADSTAR_RUN_WAIT_S``); a headless interface would be
the robust replacement. ``pyautogui`` is imported only when a simulation runs, so importing this
module is safe on any platform.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pcb_assistant.config import Settings, get_settings
from pcb_assistant.contracts import SimulationRequest

logger = logging.getLogger(__name__)

# Unit scale factors expected by the SIDF circuit file (value written = user value * scale).
SIDF_RESISTANCE_SCALE = 100
SIDF_LENGTH_SCALE = 100_000

# Menu key sequences that start each analysis in the Electrical Editor.
HOTKEYS: dict[str, list[str]] = {
    "SI": ["alt", "t"],
    "TDR": ["alt", "t", "down"],
    "FD": ["alt", "t", "down", "down", "down"],
}


class SimulationError(RuntimeError):
    """The simulator is not configured, not available, or failed to run."""


def fill_template(
    template_text: str, resistance_ohm: float | None, line_length_mm: float | None
) -> str:
    """Substitute the resistor and line-length placeholders of a SIDF template."""

    text = template_text
    if resistance_ohm is not None:
        text = text.replace("RESISTOR_VALUE", str(round(resistance_ohm * SIDF_RESISTANCE_SCALE)))
    if line_length_mm is not None:
        text = text.replace("LENGTH_VALUE", str(round(line_length_mm * SIDF_LENGTH_SCALE)))
    return text


def _topology_paths(topology: str, settings: Settings) -> tuple[str, str]:
    paths = {
        "P2P": (settings.ecadstar_p2p_sidf, settings.ecadstar_p2p_template),
        "Star": (settings.ecadstar_star_sidf, settings.ecadstar_star_template),
    }
    if topology not in paths:
        raise ValueError(f"Unknown topology {topology!r}; expected one of {sorted(paths)}")
    sidf_path, template_path = paths[topology]
    if not sidf_path or not template_path:
        raise SimulationError(f"eCADSTAR paths for {topology} are not configured; see .env.example")
    return sidf_path, template_path


def _export_image(gui: Any, image_path: Path) -> None:
    """Export the result window as an image through the File menu."""

    gui.hotkey("alt", "f")
    gui.hotkey("down")
    gui.hotkey("enter")
    time.sleep(0.5)
    gui.hotkey("down")
    time.sleep(0.5)
    gui.hotkey("tab")
    gui.hotkey("backspace")
    time.sleep(1)
    gui.typewrite(str(image_path))
    time.sleep(1)
    gui.hotkey("enter")
    time.sleep(1)
    gui.hotkey("enter")
    time.sleep(1)
    gui.hotkey("enter")
    time.sleep(0.5)
    gui.hotkey("enter")
    time.sleep(1)
    gui.hotkey("winleft", "down")


def _drive_gui(sidf_path: str, analysis_type: str, image_path: Path, settings: Settings) -> None:
    if sys.platform != "win32":
        raise SimulationError("The eCADSTAR adapter only runs on Windows")
    try:
        import pyautogui
    except ImportError as exc:
        raise SimulationError("pyautogui is required for the eCADSTAR adapter") from exc

    os.startfile(sidf_path)  # opens the manipulated simulation file with eCADSTAR
    time.sleep(settings.ecadstar_open_wait_s)

    pyautogui.press("enter")  # pop-ups: locked simulation file or "accept changes"
    time.sleep(1)

    for key in HOTKEYS[analysis_type]:
        pyautogui.press(key)
    pyautogui.press("enter")

    time.sleep(settings.ecadstar_run_wait_s)  # wait for the simulation to finish
    subprocess.call(["taskkill", "/F", "/IM", "eCS_eleceditor.exe"])
    _export_image(pyautogui, image_path)


def run_simulation(
    request: SimulationRequest,
    topology: str = "P2P",
    settings: Settings | None = None,
) -> Path:
    """Run one eCADSTAR simulation and return the path of the exported result image.

    Raises:
        ValueError: unknown analysis type or topology.
        SimulationError: paths not configured, not on Windows, or ``pyautogui`` missing.
    """

    settings = settings or get_settings()
    if request.analysis_type not in HOTKEYS:
        raise ValueError(
            f"Unsupported analysis type {request.analysis_type!r}; "
            f"expected one of {sorted(HOTKEYS)}"
        )
    sidf_path, template_path = _topology_paths(topology, settings)

    started = time.monotonic()
    if request.resistance_ohm is not None or request.line_length_mm is not None:
        template_text = Path(template_path).read_text()
        Path(sidf_path).write_text(
            fill_template(template_text, request.resistance_ohm, request.line_length_mm)
        )

    output_dir = settings.ecadstar_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"image_{datetime.now():%Y%m%d_%H%M%S}.png"

    _drive_gui(sidf_path, request.analysis_type, image_path, settings)

    logger.info(
        "Simulation completed in %.1f s, image at %s", time.monotonic() - started, image_path
    )
    return image_path
