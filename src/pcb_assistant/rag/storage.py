"""Small filesystem helpers for the vector stores and document folders."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def delete_directory(path: Path | str) -> bool:
    """Remove ``path`` and everything below it. Returns ``False`` if it did not exist."""

    target = Path(path)
    if not target.exists():
        logger.info("Directory %s does not exist", target)
        return False
    shutil.rmtree(target)
    logger.info("Directory %s has been deleted", target)
    return True


def clear_files(folder: Path | str) -> int:
    """Delete the files directly inside ``folder`` (subfolders and hidden files stay)."""

    removed = 0
    for entry in Path(folder).glob("*"):
        if entry.is_file() and not entry.name.startswith("."):
            entry.unlink()
            logger.info("Deleted %s", entry)
            removed += 1
    return removed
